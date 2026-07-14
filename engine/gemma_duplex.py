"""Gemma-4-E2B-Duplex — the frame-synchronous text-out duplex engine.

Gemma-4-E2B + the ``dockhardman/gemma-4-E2B-duplex`` LoRA runs one 80ms frame at a time
(2 audio tokens + 1 text token), emitting TEXT ONLY. Speech is synthesized by Cartesia
sonic-3.5 over a single websocket (one context per utterance, ``cancel()`` on barge-in),
and the resulting PCM is drained into each ``OutputFrame`` — so from the driver's view Cartesia
is just this model's internal vocoder.

Core mechanics:
  - encoder: per-frame sliding window (16 history blocks + 60ms lookahead), so we run one frame
    behind the live edge (each frame needs 60ms of future audio);
  - LM step: inputs_embeds + manual get_per_layer_inputs (the Gemma "PLE" pairing) + DynamicCache;
  - tool loop: collect a <|tool_call> span -> mock execute -> inject <|tool_response> after a short
    delay; a span containing <unused1> is aborted and never executed;
  - text input: burst-inject <unused2>{text}<unused3> (the trained typed-text channel).

Engine guards (env-tunable, tuned defaults — keep them on for a smooth live experience): adaptive
noise floor, voiced-run arming, open-nudge, and input AGC. The bare adapter is already strong; the
guards mainly make it robust to laptop-mic noise floors.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

import numpy as np

from .duplex import InputFrame, OutputFrame

logger = logging.getLogger(__name__)

BASE = os.environ.get("GEMMA_BASE", "google/gemma-4-E2B-it")
ADAPTER = os.environ.get("HF_MODEL", "dockhardman/gemma-4-E2B-duplex")

# Cartesia sonic-3 speech speed (0.6–1.5). Default 1.5x for a snappier demo feel.
TTS_SPEED = min(1.5, max(0.6, float(os.environ.get("CARTESIA_SPEED", "1.5"))))

FRAME_S = 0.08
SR = 16000
FRAME_SAMPLES = int(SR * FRAME_S)          # 1280
TOK_SAMPLES = 640                          # 40ms per audio token
BLOCK_TOK = 12
P_BLOCKS = 16                              # encoder receptive field, in blocks
LOOKAHEAD = 960                            # 60ms of future audio each frame needs
LAT_FRAMES = 8                             # frames to wait before injecting a tool result
AGG_MAX_TOK = 14                           # aggregator: max tokens buffered before a TTS flush
PUNCT = set("。！？!?；;，,")

# Silence freeze (see step()): training sessions have short inter-turn gaps, so >~0.8s of silence is
# out-of-distribution. Once silence runs long enough we stop advancing the model's timeline (don't
# append audio, don't step) so it only ever experiences in-distribution gaps.
SILENCE_RMS = float(os.environ.get("GEMMA_SILENCE_RMS", "0.006"))
SILENCE_HOLD_FRAMES = int(float(os.environ.get("GEMMA_SILENCE_HOLD_S", "0.72")) / 0.08)
# Open nudge: if the user has stopped and the model still hasn't answered, penalize the wait token
# for a few frames before the freeze to push it into taking a turn (instead of waiting forever).
OPEN_NUDGE = float(os.environ.get("GEMMA_OPEN_NUDGE", "3.0"))
OPEN_NUDGE_FRAMES = 8
FLOOR_K = float(os.environ.get("GEMMA_FLOOR_K", "3.0"))       # effective silence threshold = floor x K
VOICED_MIN = int(os.environ.get("GEMMA_VOICED_MIN", "6"))     # consecutive voiced frames to arm a reply
# Input loudness normalization (AGC): pull the user's rolling RMS up to the training corpus level.
# Applied only at the model's input (does not affect the driver's barge/freeze gates). 0 = disabled.
AGC_TARGET = float(os.environ.get("GEMMA_AGC_TARGET", "0.12"))

# Long-conversation KV management: the global layers keep a preallocated buffer with anchor + sliding
# window trimming (sink prefix / recent window / trigger length).
KV_SINK, KV_RECENT, KV_TRIGGER = 512, 16000, 24000

# System prompt. The adapter was trained with a Chinese instruction; this is an English rendering for the
# English demo. Verified equivalent in practice: conversation is clean, and tool-calling — which is
# VOICE-triggered — still emits the correct <|tool_call> special tokens. INSTR language does not affect
# the tool path (typed text never emits tool tokens regardless of language; speech always does).
INSTR = ("You are a real-time voice assistant. Below is the user's live speech stream. Rules: "
         "while the user is still speaking, output <unused0> on every frame; "
         "once the user finishes, directly speak a short, conversational reply token by token; "
         "when you need a tool, output <|tool_call>call:...<tool_call|>; the result arrives as "
         "<|tool_response>, and after receiving it summarize out loud; "
         "the user may also send a text message (wrapped in <unused2>...<unused3>) — answer it directly; "
         "if the user speaks again while you are replying, stop immediately, output <unused0>, and listen; "
         "if a tool call is interrupted mid-emit, output <unused1> then close <tool_call|> (the call is "
         "voided), then output <unused0> and listen.")

# Tool declarations. Descriptions are bilingual as trained (zh / en) — kept verbatim; load-bearing.
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "查詢城市目前天氣 / get current weather of a city",
        "parameters": {"type": "object", "required": ["city"],
                       "properties": {"city": {"type": "string", "description": "城市名 / city"}}}}},
    {"type": "function", "function": {
        "name": "set_timer", "description": "設定倒數計時器 / set a countdown timer",
        "parameters": {"type": "object", "required": ["minutes"],
                       "properties": {"minutes": {"type": "string", "description": "分鐘數 / minutes"}}}}},
    {"type": "function", "function": {
        "name": "play_music", "description": "播放音樂 / play music by song or artist",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string", "description": "歌名或歌手 / song or artist"}}}}},
    {"type": "function", "function": {
        "name": "turn_on_light", "description": "開燈 / turn on the light in a room",
        "parameters": {"type": "object", "required": ["room"],
                       "properties": {"room": {"type": "string", "description": "房間 / room"}}}}},
    {"type": "function", "function": {
        "name": "send_message", "description": "傳訊息給某人 / send a message to someone",
        "parameters": {"type": "object", "required": ["person", "text"],
                       "properties": {"person": {"type": "string"}, "text": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "上網搜尋 / search the web",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string"}}}}},
]

CALL_RE = re.compile(r"call:(\w+)\{(.*)\}$")
ARG_RE = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>')


def mock_execute(name: str, args: dict) -> dict:
    """Demo mock executor (results shaped like the trained tool-response format)."""
    if name == "get_weather":
        return {"temp": "28", "desc": "sunny"}
    if name == "set_timer":
        return {"status": "ok"}
    if name == "play_music":
        return {"status": "playing"}
    if name == "turn_on_light":
        return {"status": "on"}
    if name == "send_message":
        return {"status": "sent"}
    if name == "web_search":
        return {"top": f"summary of search results for '{args.get('query', '')}'"}
    return {"status": "unknown_tool"}


def render_response(name: str, result: dict) -> str:
    body = ",".join(f'{k}:<|"|>{v}<|"|>' for k, v in sorted(result.items()))
    return f"<|tool_response>response:{name}{{{body}}}<tool_response|>"


class GemmaState:
    """Per-session state: KV cache, rolling audio/mel buffers, turn/tool state machines."""

    def __init__(self):
        self.cache = None
        self.abs_pos = 0        # logical position (monotonic, feeds cache_position / RoPE)
        self.raw = np.zeros(0, dtype=np.float32)   # rolling 16k float mic buffer (see raw_off)
        self.raw_off = 0                            # absolute sample index that raw[0] maps to
        self.total_samples = 0
        self.mel = None                             # rolling mel buffer [1, T, 128] (cuda bf16)
        self.mel_off = 0                            # absolute mel-frame index that mel[0] maps to
        self.k = 0                                  # frames processed so far
        self.t_stats = {"mel": [], "enc": [], "lm": []}  # per-stage ms (diagnostics)
        self.prev_text: int | None = None
        # tool state machine
        self.in_call = False
        self.call_buf: list[int] = []
        self.call_aborted = False
        self.inject_at: int | None = None
        self.pending_resp: str | None = None
        # turn / aggregator
        self.speaking = False
        self.turn_ids: list[int] = []               # speakable tokens emitted this turn
        self.emitted_chars = 0
        self.unsent_chars = ""
        self.turn_lang = "zh"
        self.quiet_ticks = 0
        self.silent_run = 0                         # consecutive silent frames (silence freeze)
        self.open_nudge = 0                         # remaining wait-logit penalty frames
        self.spoke_since_user = False               # whether the model already answered since last speech
        self.agc_rms = None                         # EMA of the user's voiced loudness (AGC)
        self.noise_floor = None                     # adaptive noise floor (min-tracking, slow-up/fast-down)
        self.voiced_run = 0                         # consecutive voiced frames (reply arming)
        self.speak_hold = 0
        self.tts: _CartesiaTTS | None = None
        self.text_inbox: list[str] = []             # queued user text messages to inject
        self.lock = threading.Lock()


class GemmaDuplexModel:
    """Gemma-4-E2B + duplex LoRA under the frame contract: frame-sync, text-out, Cartesia speech."""

    frame_rate_hz = 12.5

    def __init__(self, base: str = BASE, adapter: str = ADAPTER, tts: bool = True):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        self.torch = torch
        logger.info("loading %s + %s ...", base, adapter)
        self.processor = AutoProcessor.from_pretrained(base)
        self.tok = self.processor.tokenizer
        model = Gemma4ForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
        # adapter may be an HF repo id or a local path — PeftModel.from_pretrained accepts either.
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        cfg = model.config
        t = self.tok.convert_tokens_to_ids
        self.audio_id = cfg.audio_token_id
        self.pad_id = cfg.text_config.pad_token_id
        self.wait_id = t("<unused0>")
        self.abort_id = t("<unused1>")
        self.txt_open, self.txt_close = t("<unused2>"), t("<unused3>")
        self.call_open, self.call_close = t("<|tool_call>"), t("<tool_call|>")
        self.resp_ids = (t("<|tool_response>"), t("<tool_response|>"))
        self._boa_id = cfg.boa_token_id
        self.prefix_ids = self._build_prefix(None)          # default prefix = all tools declared
        self._special = {self.wait_id, self.abort_id, self.txt_open, self.txt_close,
                         self.call_open, self.call_close, *self.resp_ids}
        self._use_tts = tts and bool(os.environ.get("CARTESIA_API_KEY"))
        self._warmup()
        logger.info("gemma-duplex ready (prefix=%d tok, tts=%s)", len(self.prefix_ids), self._use_tts)

    def begin_session(self, *, system_prompt: str = "", voice: str | None = None,
                      tools: list[str] | None = None, tts: bool = True) -> GemmaState:
        from transformers.cache_utils import DynamicCache, DynamicLayer
        torch = self.torch
        st = GemmaState()
        st.cache = DynamicCache(config=self.model.config.text_config)
        st.cache.layers = [_TrimLayer(None) if type(layer) is DynamicLayer else layer
                           for layer in st.cache.layers]
        prefix = self.prefix_ids if tools is None else self._build_prefix(tools)
        logger.info("session prefix=%d tok, tools=%s", len(prefix), "ALL" if tools is None else tools)
        with torch.inference_mode():
            n = len(prefix)
            self.model.model(input_ids=torch.tensor([prefix], device="cuda"),
                             past_key_values=st.cache, use_cache=True,
                             cache_position=torch.arange(0, n, device="cuda"))
            st.abs_pos = n
        if self._use_tts and tts:            # tts: per-session toggle; _use_tts: globally available (key set)
            st.tts = _CartesiaTTS(voice_id=voice)
        return st

    def step(self, state: GemmaState, frame: InputFrame) -> OutputFrame:
        pcm = np.frombuffer(frame.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # Silence freeze: once silence runs past SILENCE_HOLD we stop advancing the model's timeline,
        # so it only experiences in-distribution short gaps. Voice energy / busy model / incoming text
        # unfreeze immediately.
        frozen = False
        if len(pcm):
            rms = float(np.sqrt(np.mean(pcm * pcm)))
            # Adaptive noise floor (min-tracking, slow-up/fast-down): effective silence threshold =
            # max(fixed, floor x FLOOR_K). Handles echo residue / ambient noise on built-in mics.
            state.noise_floor = rms if state.noise_floor is None else min(
                rms, state.noise_floor * 1.005 + 1e-6)
            thr = max(SILENCE_RMS, state.noise_floor * FLOOR_K)
            # AGC: track the RMS EMA of voiced frames and pull input up to the training corpus level.
            if AGC_TARGET > 0:
                if rms >= thr:
                    state.agc_rms = rms if state.agc_rms is None else 0.9 * state.agc_rms + 0.1 * rms
                if state.agc_rms:
                    pcm = np.clip(pcm * min(max(AGC_TARGET / (state.agc_rms + 1e-9), 0.5), 10.0), -1.0, 1.0)
            busy_now = state.speaking or state.in_call or state.inject_at is not None or state.text_inbox
            state.voiced_run = state.voiced_run + 1 if rms >= thr else 0
            if rms >= thr:
                state.open_nudge = 0                # real voice cancels an in-progress nudge
            if state.voiced_run >= VOICED_MIN:
                state.spoke_since_user = False       # enough voiced frames -> arm a new reply obligation
            if state.speaking:
                state.spoke_since_user = True
            state.silent_run = 0 if (rms >= thr or busy_now) else state.silent_run + 1
            # About to freeze (2 frames out) and the model hasn't answered -> start the open nudge.
            if (OPEN_NUDGE > 0 and state.silent_run == SILENCE_HOLD_FRAMES - 2
                    and not busy_now and not state.spoke_since_user):
                state.open_nudge = OPEN_NUDGE_FRAMES
            frozen = state.silent_run > SILENCE_HOLD_FRAMES and state.open_nudge == 0
        if not frozen:
            state.raw = np.concatenate([state.raw, pcm])
            state.total_samples += len(pcm)
        self._extend_mel(state)

        text_out = []
        # User text messages: burst inject (frames continue as normal).
        with state.lock:
            inbox, state.text_inbox = state.text_inbox, []
        for msg in inbox:
            self._inject(state, [self.txt_open] + self.tok.encode(msg, add_special_tokens=False)
                         + [self.txt_close])

        # Advance every frame that has enough material this tick (normally exactly 1; catch up if behind).
        while state.total_samples >= (state.k + 1) * FRAME_SAMPLES + LOOKAHEAD:
            text_out.append(self._advance_frame(state))
        if state.k and state.k % 300 == 0 and state.t_stats["lm"]:   # per-stage timing every ~24s
            msg = "  ".join(f"{k} p50={np.percentile(v, 50):.0f} p95={np.percentile(v, 95):.0f}ms"
                            for k, v in state.t_stats.items() if v)
            logger.info("step timing (last %d frames): %s", len(state.t_stats["lm"]), msg)
            for v in state.t_stats.values():
                v.clear()

        pcm_out = state.tts.drain() if state.tts else b""
        # tts_active stays True until Cartesia has fully sent this turn's audio, so a short reply whose
        # speech is still in flight does not end the turn early (which would drop the trailing audio).
        tts_active = state.tts.is_active() if state.tts else False
        busy = state.speaking or state.in_call or state.inject_at is not None
        # is_speaking gets a short hold: audio arrives in chunks, and the gaps between chunks should
        # not be read as a falling edge (that would flicker the turn indicator).
        if busy or pcm_out or tts_active:
            state.speak_hold = 4                       # ~320ms
        else:
            state.speak_hold = max(0, state.speak_hold - 1)
        return OutputFrame(pcm=pcm_out, text="".join(text_out),
                           is_speaking=busy or bool(pcm_out) or tts_active or state.speak_hold > 0)

    def reset_listen(self, state: GemmaState) -> None:
        if state.tts:
            state.tts.cancel()
        state.unsent_chars = ""
        state.inject_at, state.pending_resp = None, None   # barge-in drops any pending tool result
        state.speaking = False

    def post_text(self, state: GemmaState, text: str) -> None:
        with state.lock:
            state.text_inbox.append(text)

    def end_session(self, state: GemmaState) -> None:
        if state.tts:
            state.tts.cancel()
            state.tts.close()
            state.tts = None

    def _build_prefix(self, tool_names: list[str] | None) -> list[int]:
        """Build the session prefix. None -> all tools; [] -> no tools; subset -> only the named tools."""
        tools = TOOLS if tool_names is None else [t for t in TOOLS
                                                  if t["function"]["name"] in set(tool_names)]
        sys_txt = self.tok.apply_chat_template(
            [{"role": "system", "content": INSTR}], tools=tools or None, tokenize=False)
        return (self.tok.encode(sys_txt, add_special_tokens=False)
                + self.tok.encode("<|turn>user\n", add_special_tokens=False) + [self._boa_id])

    def _warmup(self):
        use_tts, self._use_tts = self._use_tts, False      # no Cartesia connection during warmup
        try:
            st = self.begin_session(system_prompt="", voice=None)
            for _ in range(4):
                self.step(st, InputFrame(pcm=b"\x00" * (FRAME_SAMPLES * 2)))
        finally:
            self._use_tts = use_tts

    def _extend_mel(self, state: GemmaState):
        """Incrementally turn new raw samples into mel and append to the rolling buffer, then trim raw
        and mel down to the minimum history the sliding window needs (kills O(session) growth)."""
        torch = self.torch
        MEL_HOP, MEL_WIN = 160, 320
        n_have = state.mel_off + (0 if state.mel is None else state.mel.shape[1])
        n_can = max(0, (state.total_samples - MEL_WIN) // MEL_HOP + 1)
        if n_can > n_have:
            s0 = n_have * MEL_HOP
            s1 = (n_can - 1) * MEL_HOP + MEL_WIN
            chunk = state.raw[s0 - state.raw_off: s1 - state.raw_off]
            t0 = time.time()
            feats = self.processor.feature_extractor(
                [chunk], sampling_rate=SR, return_tensors="pt", padding=False)
            new = feats["input_features"].to("cuda", dtype=torch.bfloat16)
            state.t_stats["mel"].append((time.time() - t0) * 1000)
            state.mel = new if state.mel is None else torch.cat([state.mel, new], dim=1)
        # Trim: keep only the mel history the sliding window needs, and the raw tail not yet mel-ized.
        keep_mel_from = max(0, ((2 * state.k + 1) // BLOCK_TOK - P_BLOCKS - 2) * BLOCK_TOK * 4)
        if state.mel is not None and keep_mel_from > state.mel_off:
            state.mel = state.mel[:, keep_mel_from - state.mel_off:]
            state.mel_off = keep_mel_from
        keep_raw_from = n_can * MEL_HOP                     # start of the next mel batch
        if keep_raw_from > state.raw_off:
            state.raw = state.raw[keep_raw_from - state.raw_off:]
            state.raw_off = keep_raw_from

    def _encode_frame(self, state: GemmaState):
        """The 2 audio embeddings for frame k (sliding window; mel from the rolling buffer)."""
        torch = self.torch
        j = 2 * state.k + 1
        start_block = max(0, j // BLOCK_TOK - P_BLOCKS)
        m0 = start_block * BLOCK_TOK * 4                    # absolute mel index at the window start
        m1 = ((j + 1) * TOK_SAMPLES + LOOKAHEAD - 320) // 160 + 1
        mel = state.mel[:, m0 - state.mel_off: m1 - state.mel_off]
        t0 = time.time()
        with torch.inference_mode():
            o = self.model.model.get_audio_features(
                input_features=mel,
                input_features_mask=torch.ones(1, mel.shape[1], dtype=torch.bool, device="cuda"),
                return_dict=True)
        emb = o.pooler_output[o.attention_mask.to(o.pooler_output.device)]
        state.t_stats["enc"].append((time.time() - t0) * 1000)
        rel = j - start_block * BLOCK_TOK
        return emb[rel - 1: rel + 1]

    def _lm_step(self, state: GemmaState, audio_emb):
        torch = self.torch
        lm = self.model.model
        embed = lm.get_input_embeddings()
        step = ([self.pad_id, self.pad_id] if state.prev_text is None
                else [state.prev_text, self.pad_id, self.pad_id])
        ids = torch.tensor([step], device="cuda")
        t0 = time.time()
        with torch.inference_mode():
            emb = embed(ids)
            emb[0, -2] = audio_emb[0]
            emb[0, -1] = audio_emb[1]
            pli = lm.language_model.get_per_layer_inputs(ids, embed(ids))
            cp = torch.arange(state.abs_pos, state.abs_pos + ids.shape[1], device="cuda")
            out = lm(inputs_embeds=emb, per_layer_inputs=pli,
                     past_key_values=state.cache, use_cache=True, cache_position=cp)
            state.abs_pos += ids.shape[1]
            logits = self.model.lm_head(out.last_hidden_state[:, -1])
            if state.open_nudge > 0:
                # Open nudge: penalty ramps per frame; the last frame masks wait entirely (forced turn).
                if state.open_nudge == 1:
                    logits[0, self.wait_id] = float("-inf")
                else:
                    step_i = OPEN_NUDGE_FRAMES - state.open_nudge
                    logits[0, self.wait_id] -= OPEN_NUDGE * (1 + step_i)
                # Nudge forces "take a turn", not "act": ban tool-calling on the nudged first step.
                logits[0, self.call_open] = float("-inf")
                state.open_nudge -= 1
            tok_id = int(logits.argmax(-1))
        state.t_stats["lm"].append((time.time() - t0) * 1000)
        return tok_id

    def _inject(self, state: GemmaState, block_ids: list[int]):
        """Burst-inject input tokens (tool response / user text).

        Feed all but the LAST token into the cache, and hand that last token to the next _lm_step as
        prev_text. The old code fed everything and then predicted the next token into prev_text — but
        that predicted token is the model's first real reply token, and stashing it in prev_text without
        emitting it made the frame loop resume from the SECOND token, dropping the first word of every
        injected reply (visible on tool-response summaries: "It's 28°" → "'s 28°"). Letting the normal
        frame loop produce it from prev_text=<last injected token> emits it through the aggregator."""
        torch = self.torch
        ids = ([state.prev_text] if state.prev_text is not None else []) + block_ids
        feed = ids[:-1]
        with torch.inference_mode():
            if feed:
                cp = torch.arange(state.abs_pos, state.abs_pos + len(feed), device="cuda")
                self.model.model(input_ids=torch.tensor([feed], device="cuda"),
                                 past_key_values=state.cache, use_cache=True, cache_position=cp)
                state.abs_pos += len(feed)
        state.prev_text = ids[-1]

    def _advance_frame(self, state: GemmaState) -> str:
        tok_id = self._lm_step(state, self._encode_frame(state))
        state.k += 1
        if state.k % 25 == 0:                       # check KV trimming every ~2s (frame boundary)
            for layer in state.cache.layers:
                if hasattr(layer, "maybe_trim"):
                    layer.maybe_trim()
        state.prev_text = tok_id
        shown = ""

        # Tool schedule: inject the result once due (barge-in / user-speech drop is handled by the driver).
        if state.inject_at is not None and state.k >= state.inject_at and state.pending_resp:
            self._inject(state, self.tok.encode(state.pending_resp, add_special_tokens=False))
            shown += "[tool result injected]\n"
            state.inject_at, state.pending_resp = None, None

        if tok_id == self.call_open and not state.in_call:
            state.in_call, state.call_buf, state.call_aborted = True, [tok_id], False
        elif state.in_call:
            state.call_buf.append(tok_id)
            if tok_id == self.abort_id:
                state.call_aborted = True
            if tok_id == self.call_close:
                state.in_call = False
                span = self.tok.decode(state.call_buf, skip_special_tokens=False)
                if state.call_aborted:
                    shown += "[tool call voided]\n"
                else:
                    m = CALL_RE.search(span.replace("<|tool_call>", "").replace("<tool_call|>", ""))
                    if m:
                        name, args = m.group(1), dict(ARG_RE.findall(m.group(2)))
                        result = mock_execute(name, args)
                        state.pending_resp = render_response(name, result)
                        state.inject_at = state.k + LAT_FRAMES
                        import json as _json
                        shown += ("[tool call " + _json.dumps({"name": name, "args": args, "result": result},
                                                              ensure_ascii=False) + "]\n")
        elif tok_id not in self._special and tok_id != self.pad_id:
            # A speakable reply token -> aggregator.
            if not state.speaking:
                state.speaking = True
                state.turn_ids, state.emitted_chars, state.unsent_chars = [], 0, ""
                if state.tts:
                    state.tts.begin_turn()
            state.turn_ids.append(tok_id)
            full = self.tok.decode(state.turn_ids, skip_special_tokens=True)
            delta = full[state.emitted_chars:]
            if delta:
                state.emitted_chars = len(full)
                state.unsent_chars += delta
                shown += delta
                if (any(c in PUNCT for c in delta)
                        or len(self.tok.encode(state.unsent_chars, add_special_tokens=False)) >= AGG_MAX_TOK):
                    self._flush_tts(state)
            state.quiet_ticks = 0
        elif state.speaking and tok_id == self.wait_id:
            state.quiet_ticks += 1
            if state.quiet_ticks >= 3:                       # end of the reply
                self._flush_tts(state)
                if state.tts:
                    state.tts.end_turn()
                state.speaking = False
        return shown

    def _flush_tts(self, state: GemmaState):
        seg = state.unsent_chars.strip()
        state.unsent_chars = ""
        if not seg or not state.tts:
            return
        if state.turn_ids and state.emitted_chars <= len(seg) + 2:  # first segment -> pick the language
            kana = sum(1 for c in seg if "぀" <= c <= "ヿ")   # hiragana / katakana
            han = sum(1 for c in seg if "一" <= c <= "鿿")
            if kana / max(len(seg), 1) > 0.05:
                state.turn_lang = "ja"
            else:
                state.turn_lang = "zh" if han / max(len(seg), 1) > 0.25 else "en"
        state.tts.send_segment(seg, state.turn_lang)


class _CartesiaTTS:
    """One websocket for the whole session; one context per utterance; barge-in = cancel."""

    def __init__(self, voice_id: str | None = None):
        from cartesia import Cartesia
        self._client = Cartesia(api_key=os.environ["CARTESIA_API_KEY"])
        self._voice = {"mode": "id", "id": voice_id or os.environ["CARTESIA_VOICE_ID"]}
        self._fmt = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000}
        self._mgr = None
        self._ws = None
        self._ctx = None
        self._rx: threading.Thread | None = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._first_sent = False
        self._active = False        # True while Cartesia is still synthesizing/sending this turn's audio
        self._connect()
        self._warmup()

    def begin_turn(self):
        # Liveness check: Cartesia closes idle connections after ~5 min, and sends on a dead one fail
        # silently. Reconnect if the socket is closed.
        try:
            conn = getattr(self._ws, "_connection", None) or getattr(self._ws, "websocket", None)
            if conn is not None and getattr(conn, "close_code", None) is not None:
                raise ConnectionError("cartesia ws closed")
        except ConnectionError:
            logger.info("cartesia ws dead — reconnecting for new turn")
            self._connect()
        self._ctx = self._ws.context()
        self._first_sent = False
        self._active = True

    def send_segment(self, text: str, lang: str):
        if self._ctx is None:
            self.begin_turn()
        gen_cfg = {"speed": TTS_SPEED, "volume": 1.0}
        try:
            self._ctx.send(model_id="sonic-3.5", transcript=text, voice=self._voice,
                           language=lang, output_format=self._fmt, continue_=True,
                           flush=not self._first_sent, generation_config=gen_cfg)
        except Exception:
            logger.warning("cartesia send failed — reconnecting", exc_info=True)
            self._connect()
            self.begin_turn()
            self._ctx.send(model_id="sonic-3.5", transcript=text, voice=self._voice,
                           language=lang, output_format=self._fmt, continue_=True, flush=True,
                           generation_config=gen_cfg)
        if not self._first_sent:
            self._first_sent = True
            ctx = self._ctx

            def rx():
                try:
                    for chunk in ctx.receive():
                        a = getattr(chunk, "audio", None)
                        if a:
                            with self._lock:
                                self._buf += a
                except Exception:
                    pass                                    # cancel/close both land here; swallow quietly
                finally:
                    self._active = False                    # this turn's audio is fully received

            self._rx = threading.Thread(target=rx, daemon=True)
            self._rx.start()

    def end_turn(self):
        if self._ctx is not None:
            try:
                self._ctx.no_more_inputs()
            except Exception:
                pass
            self._ctx = None

    def cancel(self):
        self._active = False
        if self._ctx is not None:
            try:
                self._ctx.cancel()
            except Exception:
                pass
            self._ctx = None
        with self._lock:
            self._buf.clear()

    def is_active(self) -> bool:
        """True from begin_turn until this turn's audio is fully received (or cancelled)."""
        return self._active

    def drain(self) -> bytes:
        with self._lock:
            out = bytes(self._buf)
            self._buf.clear()
        return out

    def close(self):
        try:
            if self._mgr is not None:
                self._mgr.__exit__(None, None, None)
        except Exception:
            pass

    def _warmup(self):
        """Prime the connection + synth pipeline with a throwaway request so the first real turn's
        time-to-first-audio is short. Runs at session start (off the event loop — see app.py)."""
        try:
            self.begin_turn()
            self.send_segment(".", "en")
            time.sleep(0.3)
            self.cancel()
        except Exception:
            logger.debug("cartesia warmup skipped", exc_info=True)

    def _connect(self):
        self._mgr = self._client.tts.websocket_connect()
        self._ws = self._mgr.__enter__()


class _TrimLayer:
    """Global-layer KV: index_copy_ into a preallocated buffer (no O(N) cat), trimmed to sink+recent
    at frame boundaries once it exceeds the trigger length. The engine keeps cache_position monotonic
    so RoPE positions stay correct across a trim."""

    def __new__(cls, base_cls):
        from transformers.cache_utils import DynamicLayer

        class Impl(DynamicLayer):
            CAP = KV_TRIGGER + 4096

            def __init__(self):
                super().__init__()
                self._len = 0
                self._kbuf = self._vbuf = None

            def update(self, key_states, value_states, *a, **k):
                if not self.is_initialized:
                    self.lazy_initialization(key_states, value_states)
                if self._kbuf is None:
                    import torch
                    B, H, _, D = key_states.shape
                    self._kbuf = torch.empty(B, H, self.CAP, D,
                                             dtype=key_states.dtype, device=key_states.device)
                    self._vbuf = torch.empty_like(self._kbuf)
                import torch
                n = key_states.shape[-2]
                idx = torch.arange(self._len, self._len + n, device=key_states.device)
                self._kbuf.index_copy_(2, idx, key_states)
                self._vbuf.index_copy_(2, idx, value_states)
                self._len += n
                self.keys = self._kbuf[:, :, :self._len]
                self.values = self._vbuf[:, :, :self._len]
                return self.keys, self.values

            def maybe_trim(self):
                if self._len > KV_TRIGGER:
                    kt = self._kbuf[:, :, self._len - KV_RECENT: self._len].clone()
                    vt = self._vbuf[:, :, self._len - KV_RECENT: self._len].clone()
                    self._kbuf[:, :, KV_SINK: KV_SINK + KV_RECENT] = kt
                    self._vbuf[:, :, KV_SINK: KV_SINK + KV_RECENT] = vt
                    self._len = KV_SINK + KV_RECENT
                    self.keys = self._kbuf[:, :, :self._len]
                    self.values = self._vbuf[:, :, :self._len]

            def get_seq_length(self):
                return self._len

        return Impl()
