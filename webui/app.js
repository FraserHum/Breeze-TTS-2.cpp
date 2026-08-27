const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
const download = document.getElementById("download");
const streamToggle = document.getElementById("stream");
const bufferInput = document.getElementById("buffer");
const bufferOut = document.getElementById("bufval");

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
  });
});

function fields(panel) {
  const out = {};
  panel.querySelectorAll("[data-f]").forEach(el => { out[el.dataset.f] = el; });
  return out;
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

// plays each chunk as it lands, scheduled after whatever is already queued
async function readStreaming(res, sr, lead, onProgress) {
  const ac = openContext(sr);
  const reader = res.body.getReader();
  const parts = [];
  let leftover = new Uint8Array(0);
  let playhead = 0;
  let samples = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
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
    if (!n) continue;

    const view = new DataView(bytes.buffer, bytes.byteOffset, n * 2);
    const buf = ac.createBuffer(1, n, sr);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = view.getInt16(i * 2, true) / 32768;

    const src = ac.createBufferSource();
    src.buffer = buf;
    src.connect(ac.destination);
    // falling back to the lead time also re-buffers after an underrun
    const at = Math.max(ac.currentTime + lead, playhead);
    src.start(at);
    playhead = at + buf.duration;

    samples += n;
    onProgress(samples / sr);
  }
  return join(parts);
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
  if (f.ref_audio && f.ref_audio.files.length) form.append("ref_audio", f.ref_audio.files[0]);

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
                            s => { statusEl.textContent = "STREAMING - " + s.toFixed(2) + "s"; })
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

document.querySelectorAll(".panel").forEach(panel => {
  const btn = panel.querySelector(".go");
  btn.addEventListener("click", () => generate(panel, panel.id));
});
