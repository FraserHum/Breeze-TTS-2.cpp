const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusEl = document.getElementById("status");
const player = document.getElementById("player");
const download = document.getElementById("download");

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

async function generate(panel, tabName) {
  const f = fields(panel);
  const text = (f.text.value || "").trim();
  if (!text) { statusEl.textContent = "TEXT IS REQUIRED"; return; }

  const form = new FormData();
  form.append("text", text);
  form.append("seed", f.seed ? f.seed.value : "42");
  form.append("cfg_scale", f.cfg ? f.cfg.value : (tabName === "clone" ? "1.0" : "4.0"));
  if (f.instruction) form.append("instruction", f.instruction.value || "Speak clearly and naturally.");
  else form.append("instruction", "Speak clearly and naturally.");
  if (f.ref_text) form.append("ref_text", f.ref_text.value || "");
  if (f.ref_audio && f.ref_audio.files.length) form.append("ref_audio", f.ref_audio.files[0]);

  const btn = panel.querySelector(".go");
  btn.disabled = true;
  statusEl.textContent = "GENERATING ...";
  download.classList.remove("ready");

  try {
    const res = await fetch("/v1/audio/speech", { method: "POST", body: form });
    if (!res.ok) { statusEl.textContent = "ERROR " + res.status; btn.disabled = false; return; }
    const sr = parseInt(res.headers.get("X-Sample-Rate") || "24000", 10);
    const pcm = await res.arrayBuffer();
    const blob = makeWav(pcm, sr);
    const url = URL.createObjectURL(blob);
    player.src = url;
    download.href = url;
    download.classList.add("ready");
    const seconds = (pcm.byteLength / 2 / sr).toFixed(2);
    statusEl.textContent = "DONE - " + seconds + "s @ " + sr + "Hz";
    player.play().catch(() => {});
  } catch (e) {
    statusEl.textContent = "ERROR: " + e.message;
  }
  btn.disabled = false;
}

document.querySelectorAll(".panel").forEach(panel => {
  const btn = panel.querySelector(".go");
  btn.addEventListener("click", () => generate(panel, panel.id));
});
