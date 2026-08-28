const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
const download = document.getElementById("download");
const streamToggle = document.getElementById("stream");
const bufferInput = document.getElementById("buffer");
const bufferOut = document.getElementById("bufval");
const stopBtn = document.getElementById("stop");
let wsPort = 0;

function syncBuffer() {
  bufferOut.textContent = Number(bufferInput.value).toFixed(2) + "s";
  bufferInput.disabled = !streamToggle.checked;
  bufferInput.parentElement.classList.toggle("off", !streamToggle.checked);
}

bufferInput.addEventListener("input", syncBuffer);
streamToggle.addEventListener("change", syncBuffer);
syncBuffer();

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    panels.forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
    // conversion hands back the whole clip at once, so the streaming controls do nothing there
    document.querySelector(".output .row").classList.toggle("hide", tab.dataset.tab === "changer");
  });
});

function fields(panel) {
  const out = {};
  panel.querySelectorAll("[data-f]").forEach(el => { out[el.dataset.f] = el; });
  return out;
}

// picking a saved voice replaces the upload, so the clip fields get out of the way
function syncVoice(panel) {
  const f = fields(panel);
  if (!f.voice_id) return;
  const block = panel.querySelector(".refblock");
  if (block) block.classList.toggle("hide", !!f.voice_id.value);
}

async function loadVoices() {
  let list = [];
  try {
    const res = await fetch("/v1/voices");
    if (res.ok) list = await res.json();
  } catch (e) { return; }
  document.querySelectorAll("select[data-f=voice_id]").forEach(sel => {
    const keep = sel.value;
    sel.innerHTML = "<option value=\"\">upload a clip instead</option>";
    for (const v of list) {
      const o = document.createElement("option");
      o.value = v.id;
      o.textContent = v.id + "  (" + v.seconds.toFixed(1) + "s)" + (v.saved ? "" : "  temporary");
      sel.appendChild(o);
    }
    sel.value = list.some(v => v.id === keep) ? keep : "";
    syncVoice(sel.closest(".panel"));
  });
}

async function saveVoice(panel) {
  const f = fields(panel);
  const name = (f.save_name.value || "").trim();
  if (!name) { statusEl.textContent = "NAME THE VOICE FIRST"; return; }
  if (!/^[A-Za-z0-9_-]+$/.test(name)) {
    statusEl.textContent = "NAME CAN ONLY USE LETTERS, DIGITS, DASH AND UNDERSCORE";
    return;
  }
  if (!f.ref_audio.files.length || !(f.ref_text.value || "").trim()) {
    statusEl.textContent = "A CLIP AND ITS TRANSCRIPT ARE REQUIRED TO SAVE A VOICE";
    return;
  }
  const btn = panel.querySelector(".save-voice");
  btn.disabled = true;
  statusEl.textContent = "SAVING VOICE ...";
  try {
    const form = new FormData();
    form.append("name", name);
    form.append("ref_audio", f.ref_audio.files[0]);
    form.append("ref_text", f.ref_text.value);
    const res = await fetch("/v1/voices", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) {
      statusEl.textContent = res.status === 409 ? "BUSY - ONE REQUEST AT A TIME"
                                               : "ERROR: " + (body.error || res.status);
    } else {
      await loadVoices();
      document.querySelectorAll("select[data-f=voice_id]").forEach(sel => { sel.value = body.id; });
      document.querySelectorAll(".panel").forEach(syncVoice);
      statusEl.textContent = "SAVED " + body.id + " - " + body.seconds.toFixed(2) + "s";
    }
  } catch (e) {
    statusEl.textContent = "ERROR: " + e.message;
  }
  btn.disabled = false;
}

function makeWav(pcm, sampleRate) {
  const dataLen = pcm.byteLength;
  const buf = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); view.setUint32(4, 36 + dataLen, true); str(8, "WAVE");
  str(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, dataLen, true);
  new Uint8Array(buf, 44).set(new Uint8Array(pcm));
  return new Blob([buf], { type: "audio/wav" });
}

function join(parts) {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

let audioCtx = null;

function openContext(sampleRate) {
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (audioCtx) audioCtx.close().catch(() => {});
  try {
    audioCtx = new Ctor({ sampleRate });
  } catch (e) {
    audioCtx = new Ctor();
  }
  audioCtx.resume().catch(() => {});
  return audioCtx;
}

// holds chunks back until there is a real cushion of audio, then plays them gapless
function makePlayer(sr, lead) {
  const ac = openContext(sr);
  const parts = [];
  let leftover = new Uint8Array(0);
  let playhead = 0;
  let samples = 0;
  let underruns = 0;
  let queued = [];
  let queuedSecs = 0;
  let started = false;

  const play = buf => {
    const src = ac.createBufferSource();
    src.buffer = buf;
    src.connect(ac.destination);
    let at = playhead;
    if (playhead <= ac.currentTime) {
      at = ac.currentTime + 0.02;
      // ran dry, so rebuild a deeper cushion before letting it happen again
      if (started) { underruns++; lead = Math.min(lead * 1.5 + 0.25, 5); }
    }
    src.start(at);
    started = true;
    playhead = at + buf.duration;
  };

  return {
    push(value) {
      parts.push(value);
      let bytes = value;
      if (leftover.length) {
        bytes = new Uint8Array(leftover.length + value.length);
        bytes.set(leftover, 0);
        bytes.set(value, leftover.length);
      }
      const n = bytes.length >> 1;
      // a chunk boundary can split a sample, so carry the odd byte over
      leftover = bytes.slice(n * 2);
      if (!n) return;

      const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
      const buf = ac.createBuffer(1, n, sr);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < n; i++) ch[i] = view.getInt16(i * 2, true) / 32768;

      if (queued) {
        queued.push(buf);
        queuedSecs += buf.duration;
        // generation outruns playback, so once the cushion exists it only ever grows
        if (queuedSecs >= lead) { queued.forEach(play); queued = null; }
      } else {
        play(buf);
      }
      samples += n;
    },
    finish() {
      if (queued) queued.forEach(play);
      queued = null;
      return join(parts);
    },
    stop() { try { ac.close(); } catch (e) {} return join(parts); },
    get samples() { return samples; },
    get underruns() { return underruns; }
  };
}

async function readStreaming(res, sr, lead, onProgress) {
  const p = makePlayer(sr, lead);
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    p.push(value);
    onProgress(p.samples / sr, p.underruns);
  }
  return p.finish();
}

async function readBuffered(res, onProgress) {
  const reader = res.body.getReader();
  const parts = [];
  let bytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    bytes += value.length;
    onProgress(bytes);
  }
  return join(parts);
}

// the socket protocol takes a voice id, so an uploaded clip is registered first. that caches it
// too, so picking the same file again skips the encode
async function resolveVoice(f) {
  if (f.voice_id && f.voice_id.value) return f.voice_id.value;
  if (!f.ref_audio || !f.ref_audio.files.length) return "";
  statusEl.textContent = "ENCODING REFERENCE ...";
  const form = new FormData();
  form.append("ref_audio", f.ref_audio.files[0]);
  form.append("ref_text", f.ref_text ? f.ref_text.value || "" : "");
  const r = await fetch("/v1/voices", { method: "POST", body: form });
  if (!r.ok) throw new Error("could not encode the reference");
  return (await r.json()).id;
}

let liveSocket = null;

function wsGenerate(panel, tabName) {
  const f = fields(panel);
  const text = (f.text.value || "").trim();
  if (!text) { statusEl.textContent = "TEXT IS REQUIRED"; return Promise.resolve(); }

  const btn = panel.querySelector(".go");
  const lead = Number(bufferInput.value);
  const streaming = streamToggle.checked;

  return resolveVoice(f).then(voice => new Promise(resolve => {
    const url = (location.protocol === "https:" ? "wss://" : "ws://") +
                location.hostname + ":" + wsPort;
    const sock = new WebSocket(url);
    sock.binaryType = "arraybuffer";
    liveSocket = sock;
    btn.disabled = true;
    stopBtn.classList.add("live");
    download.classList.remove("ready");
    player.pause();
    player.removeAttribute("src");
    statusEl.textContent = "CONNECTING ...";

    let sr = 24000;
    let p = null;
    const done = () => {
      const pcm = p ? p.finish() : new Uint8Array(0);
      if (pcm.length) {
        const url2 = URL.createObjectURL(makeWav(pcm.buffer, sr));
        player.src = url2;
        download.href = url2;
        download.classList.add("ready");
      }
      btn.disabled = false;
      stopBtn.classList.remove("live");
      liveSocket = null;
      try { sock.close(); } catch (e) {}
      resolve(pcm);
    };

    sock.onmessage = ev => {
      if (ev.data instanceof ArrayBuffer) {
        if (!p) p = makePlayer(sr, streaming ? lead : 1e9);
        p.push(new Uint8Array(ev.data));
        statusEl.textContent = (streaming ? "STREAMING - " : "GENERATING - ") +
                               (p.samples / sr).toFixed(2) + "s" +
                               (p.underruns ? "  (" + p.underruns + " REBUFFER)" : "");
        return;
      }
      const m = JSON.parse(ev.data);
      if (m.type === "ready") {
        sr = m.sample_rate || 24000;
        sock.send(JSON.stringify({
          type: "start",
          voice_id: voice,
          instruction: f.instruction ? (f.instruction.value || "Speak clearly and naturally.")
                                     : "Speak clearly and naturally.",
          ref_text: f.ref_text ? f.ref_text.value || "" : "",
          cfg_scale: Number(f.cfg ? f.cfg.value : 1),
          seed: Number(f.seed ? f.seed.value : 42)
        }));
        sock.send(JSON.stringify({ type: "end", text }));
        statusEl.textContent = "GENERATING ...";
      } else if (m.type === "queued") {
        statusEl.textContent = "QUEUED, WAITING FOR THE GPU ...";
      } else if (m.type === "cancelled") {
        statusEl.textContent = "STOPPED" + (p ? " - " + (p.samples / sr).toFixed(2) + "s" : "");
        done();
      } else if (m.type === "done") {
        const secs = p ? p.samples / sr : 0;
        done();
        statusEl.textContent = "DONE - " + secs.toFixed(2) + "s @ " + sr + "Hz";
        if (!streaming) player.play().catch(() => {});
      } else if (m.type === "error") {
        statusEl.textContent = "ERROR: " + (m.message || "unknown");
        done();
      }
    };
    sock.onerror = () => { statusEl.textContent = "WEBSOCKET ERROR"; done(); };
    sock.onclose = () => { if (liveSocket === sock) done(); };
  })).catch(e => {
    statusEl.textContent = "ERROR: " + e.message;
    btn.disabled = false;
    stopBtn.classList.remove("live");
  });
}

async function generate(panel, tabName) {
  const f = fields(panel);
  const text = (f.text.value || "").trim();
  if (!text) { statusEl.textContent = "TEXT IS REQUIRED"; return; }

  const form = new FormData();
  form.append("text", text);
  form.append("seed", f.seed ? f.seed.value : "42");
  form.append("cfg_scale", f.cfg ? f.cfg.value : "1.0");
  form.append("instruction", f.instruction ? (f.instruction.value || "Speak clearly and naturally.")
                                           : "Speak clearly and naturally.");
  if (f.ref_text) form.append("ref_text", f.ref_text.value || "");
  if (f.voice_id && f.voice_id.value) form.append("voice_id", f.voice_id.value);
  else if (f.ref_audio && f.ref_audio.files.length) form.append("ref_audio", f.ref_audio.files[0]);

  const streaming = streamToggle.checked;
  const btn = panel.querySelector(".go");
  btn.disabled = true;
  statusEl.textContent = streaming ? "STREAMING ..." : "GENERATING ...";
  download.classList.remove("ready");
  player.pause();
  player.removeAttribute("src");

  try {
    const res = await fetch("/v1/audio/speech", { method: "POST", body: form });
    if (!res.ok) {
      statusEl.textContent = res.status === 409 ? "BUSY - ONE REQUEST AT A TIME" : "ERROR " + res.status;
      btn.disabled = false;
      return;
    }
    const sr = parseInt(res.headers.get("X-Sample-Rate") || "24000", 10);

    const pcm = streaming
      ? await readStreaming(res, sr, Number(bufferInput.value),
                            (s, under) => {
                              statusEl.textContent = "STREAMING - " + s.toFixed(2) + "s"
                                + (under ? "  (" + under + " REBUFFER" + (under > 1 ? "S" : "") + ")" : "");
                            })
      : await readBuffered(res, b => { statusEl.textContent = "GENERATING - " + (b / 2 / sr).toFixed(2) + "s"; });

    const url = URL.createObjectURL(makeWav(pcm.buffer, sr));
    player.src = url;
    download.href = url;
    download.classList.add("ready");
    statusEl.textContent = "DONE - " + (pcm.length / 2 / sr).toFixed(2) + "s @ " + sr + "Hz";
    if (!streaming) player.play().catch(() => {});
  } catch (e) {
    statusEl.textContent = "ERROR: " + e.message;
  }
  btn.disabled = false;
}

async function convert(panel) {
  const f = fields(panel);
  const voice = f.voice_id ? f.voice_id.value : "";
  if (!f.source.files.length) {
    statusEl.textContent = "A SOURCE WAV IS REQUIRED";
    return;
  }
  if (!voice && !f.ref_audio.files.length) {
    statusEl.textContent = "PICK A SAVED VOICE OR UPLOAD A TARGET VOICE WAV";
    return;
  }
  if (!voice && !(f.ref_text.value || "").trim()) {
    statusEl.textContent = "TARGET VOICE TRANSCRIPT IS REQUIRED";
    return;
  }
  const form = new FormData();
  form.append("source", f.source.files[0]);
  if (voice) form.append("voice_id", voice);
  else {
    form.append("ref_audio", f.ref_audio.files[0]);
    form.append("ref_text", f.ref_text.value);
  }
  form.append("text", (f.text.value || "").trim());
  form.append("keep_acoustic", f.keep_acoustic.value);
  form.append("seed", f.seed.value);

  const btn = panel.querySelector(".go");
  btn.disabled = true;
  statusEl.textContent = "CONVERTING ...";
  download.classList.remove("ready");
  player.pause();
  player.removeAttribute("src");

  try {
    const res = await fetch("/v1/audio/convert", { method: "POST", body: form });
    if (!res.ok) {
      statusEl.textContent = res.status === 409 ? "BUSY - ONE REQUEST AT A TIME" : "ERROR " + res.status;
      btn.disabled = false;
      return;
    }
    const sr = parseInt(res.headers.get("X-Sample-Rate") || "24000", 10);
    const pcm = await readBuffered(res, b => {
      statusEl.textContent = "RECEIVING - " + (b / 2 / sr).toFixed(2) + "s";
    });
    const url = URL.createObjectURL(makeWav(pcm.buffer, sr));
    player.src = url;
    download.href = url;
    download.classList.add("ready");
    statusEl.textContent = "DONE - " + (pcm.length / 2 / sr).toFixed(2) + "s @ " + sr + "Hz";
    player.play().catch(() => {});
  } catch (e) {
    statusEl.textContent = "ERROR: " + e.message;
  }
  btn.disabled = false;
}

document.querySelectorAll(".panel").forEach(panel => {
  const btn = panel.querySelector(".go");
  btn.addEventListener("click", () => {
    if (panel.id === "changer") return convert(panel);
    // conversion has no socket path, everything else prefers it when the server opened one
    return wsPort ? wsGenerate(panel, panel.id) : generate(panel, panel.id);
  });
  const save = panel.querySelector(".save-voice");
  if (save) save.addEventListener("click", () => saveVoice(panel));
  const sel = panel.querySelector("select[data-f=voice_id]");
  if (sel) sel.addEventListener("change", () => syncVoice(panel));
});

stopBtn.addEventListener("click", () => {
  if (liveSocket) liveSocket.send(JSON.stringify({ type: "cancel" }));
});

async function loadHealth() {
  try {
    const r = await fetch("/health");
    if (r.ok) wsPort = (await r.json()).ws_port || 0;
  } catch (e) { wsPort = 0; }
}

loadHealth();
loadVoices();
