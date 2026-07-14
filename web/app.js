// Gemma-4-E2B-Duplex demo client.
// Mic -> downsample to 16k PCM16 -> ws(audio_input). ws(audio_delta, 24k PCM16) -> gapless playback.
// Barge-in is automatic (the server's energy gate cuts the reply when you speak); Stop ends the session.

const IN_RATE = 16000, OUT_RATE = 24000;
const PREBUFFER_S = 1.5;   // buffer this much before starting playback -> absorbs sub-realtime synthesis
const TOOL_ICON = { get_weather: "🌤", set_timer: "⏱", play_music: "🎵",
                    turn_on_light: "💡", send_message: "✉️", web_search: "🔎" };

let ws = null;
let micCtx = null, micNode = null, micStream = null, micAnalyser = null;
let outCtx = null, masterGain = null, outAnalyser = null, playHead = 0, sources = [];
let pending = [], pendingDur = 0, buffering = true;   // per-turn pre-buffer
let curAssist = null, curAssistRaw = "";              // the assistant bubble being streamed
let levelRAF = 0;

const $ = (id) => document.getElementById(id);
const connected = () => ws && ws.readyState === 1;

// ============================ base64 <-> bytes ============================
function b64encode(u8) { let s = ""; for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]); return btoa(s); }
function b64decode(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }

// ============================ status ============================
function setState(s) { document.body.dataset.state = s; }
function setStatus(text) { $("status").textContent = text; }

// ============================ transcript ============================
function addMsg(who, cls) {
  $("empty")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerHTML = `<div class="who"></div><div class="body"></div>`;
  d.querySelector(".who").textContent = who;
  $("transcript").appendChild(d);
  $("transcript").scrollTop = 1e9;
  return d.querySelector(".body");
}
function ensureAssist() {
  if (!curAssist) { curAssistRaw = ""; curAssist = addMsg("Assistant", "assistant"); }
  return curAssist;
}
function finalizeTurn() { curAssist = null; curAssistRaw = ""; }

// Render the assistant text stream, turning the server's inline tool markers into cards.
// Markers: [tool call {json}] , [tool result injected] , [tool call voided].
const MARKER = /\[tool call (\{.*?\})\]\n?|\[tool result injected\]\n?|\[tool call voided\]\n?/g;
function renderAssist(el, raw) {
  el.textContent = "";
  const put = (txt) => { if (txt) el.appendChild(document.createTextNode(txt)); };
  let last = 0, m;
  MARKER.lastIndex = 0;
  while ((m = MARKER.exec(raw)) !== null) {
    put(raw.slice(last, m.index));
    last = MARKER.lastIndex;
    if (m[1]) {
      const card = document.createElement("div");
      card.className = "toolcard";
      try {
        const d = JSON.parse(m[1]);
        card.innerHTML =
          `<div class="tc-head"><span>${TOOL_ICON[d.name] || "⚙️"}</span>` +
          `<span class="tc-name">${d.name}</span><span class="tc-tag">tool call</span></div>` +
          `<div class="tc-row"><span class="tc-k">args</span><code>${JSON.stringify(d.args)}</code></div>` +
          `<div class="tc-row"><span class="tc-k">result</span><code>${JSON.stringify(d.result)}</code></div>`;
      } catch (e) { card.textContent = "⚙️ tool call"; }
      el.appendChild(card);
    } else {
      const chip = document.createElement("span");
      const voided = m[0].includes("voided");
      chip.className = "toolchip" + (voided ? " abort" : "");
      chip.textContent = voided ? "✂ tool call interrupted — voided" : "⤷ result injected, summarizing";
      el.appendChild(chip);
    }
  }
  put(raw.slice(last));
  $("transcript").scrollTop = 1e9;
}

// ============================ playback (gapless, pre-buffered) ============================
function ensureOut() {
  if (!outCtx) {
    outCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: OUT_RATE });
    masterGain = outCtx.createGain(); masterGain.connect(outCtx.destination);
    outAnalyser = outCtx.createAnalyser(); outAnalyser.fftSize = 1024; masterGain.connect(outAnalyser);
  }
  if (outCtx.state === "suspended") outCtx.resume();
}
function playPCM(int16, sr) {
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
  const buf = outCtx.createBuffer(1, f32.length, sr || OUT_RATE); buf.copyToChannel(f32, 0);
  const src = outCtx.createBufferSource(); src.buffer = buf; src.connect(masterGain);
  const t = Math.max(playHead, outCtx.currentTime);
  src.start(t); playHead = t + buf.duration;
  sources.push(src); src.onended = () => { sources = sources.filter((s) => s !== src); };
}
function enqueueAudio(int16, sr) {
  if (!buffering) { playPCM(int16, sr); return; }
  pending.push([int16, sr]); pendingDur += int16.length / (sr || OUT_RATE);
  if (pendingDur >= PREBUFFER_S) { buffering = false; for (const [a, s] of pending) playPCM(a, s); pending = []; pendingDur = 0; }
}
function endTurnAudio() { for (const [a, s] of pending) playPCM(a, s); pending = []; pendingDur = 0; buffering = true; }
function flushPlayback() {
  for (const s of sources) { try { s.stop(); } catch (e) {} }
  sources = []; playHead = outCtx ? outCtx.currentTime : 0;
  pending = []; pendingDur = 0; buffering = true;
}

// ============================ mic capture ============================
function downsample(f32, inRate) {
  if (inRate === IN_RATE) return f32;
  const ratio = inRate / IN_RATE, outLen = Math.floor(f32.length / ratio), out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio), end = Math.floor((i + 1) * ratio);
    let sum = 0, n = 0; for (let j = start; j < end && j < f32.length; j++) { sum += f32[j]; n++; }
    out[i] = n ? sum / n : 0;
  }
  return out;
}
function floatToPCM16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); out[i] = s * 32767; }
  return out;
}
async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  micCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = micCtx.createMediaStreamSource(micStream);
  micAnalyser = micCtx.createAnalyser(); micAnalyser.fftSize = 1024; src.connect(micAnalyser);
  micNode = micCtx.createScriptProcessor(2048, 1, 1);
  src.connect(micNode); micNode.connect(micCtx.destination);
  micNode.onaudioprocess = (e) => {
    if (!connected()) return;
    const pcm = floatToPCM16(downsample(e.inputBuffer.getChannelData(0), micCtx.sampleRate));
    wsSend({ type: "audio_input", pcm: b64encode(new Uint8Array(pcm.buffer)) });
  };
}
function stopMic() {
  if (micNode) micNode.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close();
  micCtx = micNode = micStream = micAnalyser = null;
}

// ============================ orb level meter ============================
function analyserLevel(a) {
  if (!a) return 0;
  const buf = new Uint8Array(a.fftSize); a.getByteTimeDomainData(buf);
  let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
  return Math.sqrt(sum / buf.length);
}
function startLevel() {
  cancelAnimationFrame(levelRAF);
  const tick = () => {
    const lvl = Math.min(1, Math.max(analyserLevel(micAnalyser), analyserLevel(outAnalyser)) * 3.5);
    $("orb").style.setProperty("--level", lvl.toFixed(3));
    levelRAF = requestAnimationFrame(tick);
  };
  tick();
}
function stopLevel() { cancelAnimationFrame(levelRAF); $("orb").style.setProperty("--level", "0"); }

// ============================ ws send + server events ============================
function wsSend(obj) { if (connected()) ws.send(JSON.stringify(obj)); }

function handleEvent(ev) {
  switch (ev.type) {
    case "ready": onReady(); break;
    case "audio_delta":
      enqueueAudio(new Int16Array(b64decode(ev.pcm).buffer), ev.sample_rate);
      ensureAssist();
      if (document.body.dataset.state !== "speaking") { setState("speaking"); setStatus("Speaking"); }
      break;
    case "text_delta":
      curAssistRaw += ev.text; renderAssist(ensureAssist(), curAssistRaw);
      break;
    case "interrupted": flushPlayback(); finalizeTurn(); listeningState(); break;
    case "turn_complete": endTurnAudio(); finalizeTurn(); listeningState(); break;
    case "error": setState("idle"); setStatus("Error: " + (ev.message || "unknown")); break;
  }
}
function listeningState() { setState("listening"); setStatus("Listening — speak any time to interrupt"); }

async function onReady() {
  ensureOut();
  await startMic();
  startLevel();
  listeningState();
  $("textin").disabled = false; $("send").disabled = false;
}

// ============================ text input ============================
$("textform").addEventListener("submit", (e) => {
  e.preventDefault();
  const txt = $("textin").value.trim();
  if (!txt || !connected()) return;
  wsSend({ type: "text_input", text: txt });
  addMsg("You", "user").textContent = txt;
  $("textin").value = "";
  finalizeTurn();
});

// ============================ connect / stop ============================
function connect() {
  ensureOut();   // must be created inside the click handler to satisfy autoplay policy
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  setState("connecting"); setStatus("Connecting…");
  $("start").hidden = true; $("stop").hidden = false;
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => reset();
  ws.onerror = () => { setStatus("Connection error"); };
}
function stop() { if (ws) ws.close(); reset(); }
function reset() {
  stopMic(); stopLevel(); flushPlayback(); finalizeTurn();
  ws = null;
  setState("idle"); setStatus("Not connected");
  $("start").hidden = false; $("stop").hidden = true;
  $("textin").disabled = true; $("send").disabled = true;
}
$("start").addEventListener("click", connect);
$("stop").addEventListener("click", stop);

// ============================ boot ============================
reset();
