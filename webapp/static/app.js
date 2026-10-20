// Chart Analyzer front end. No frameworks: take photo -> shrink -> POST /api/analyze -> show result.
// The server holds the API key; this file never sees it. A signal is only ever rendered from a
// successful server response; every failure path shows an error instead.

const MAX_SIDE = 1568;          // longest side sent to the vision model (larger adds cost, not detail)
const JPEG_QUALITY = 0.85;
const TIMEOUT_MS = 150000;

const $ = (id) => document.getElementById(id);
const views = ["home", "busy", "result", "error"];

function show(name) {
  for (const v of views) $(`view-${v}`).hidden = v !== name;
  window.scrollTo({ top: 0 });
}

function showError(message) {
  $("error-text").textContent = message;
  show("error");
}

// Decode the photo (EXIF rotation applied), scale it down, and re-encode as JPEG.
async function prepareImage(file) {
  let source;
  try {
    source = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    source = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("decode"));
      img.src = URL.createObjectURL(file);
    });
  }
  const w = source.width, h = source.height;
  if (!w || !h) throw new Error("decode");
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  return { dataUrl, base64: dataUrl.split(",")[1] };
}

function accessToken() {
  try { return localStorage.getItem("accessToken") || ""; } catch { return ""; }
}

async function postAnalysis(base64, retriedToken = false) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch("api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Access-Token": accessToken() },
      body: JSON.stringify({ image: base64, media_type: "image/jpeg" }),
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try { body = await res.json(); } catch { /* handled below */ }
  if (res.status === 401 && body && body.code === "token_required" && !retriedToken) {
    const code = window.prompt("This app needs an access code:");
    if (code) {
      try { localStorage.setItem("accessToken", code.trim()); } catch { /* private mode */ }
      return postAnalysis(base64, true);
    }
  }
  if (!res.ok || !body || body.ok !== true || !body.result) {
    throw new Error((body && body.error) || `The server returned an error (${res.status}). No signal is shown.`);
  }
  return body.result;
}

function renderResult(r) {
  const sig = $("signal");
  const kind = r.signal.toLowerCase();
  sig.className = `signal ${kind}`;
  $("signal-icon").textContent = { buy: "▲", sell: "▼", hold: "■" }[kind] || "";
  $("signal-text").textContent = r.signal;
  $("conf-value").textContent = `${r.confidence}%`;
  $("conf-bar").style.width = `${r.confidence}%`;
  $("explanation").textContent = r.explanation;
  $("note").hidden = !r.note;
  $("note").textContent = r.note || "";
  $("f-instrument").textContent = r.instrument;
  $("f-timeframe").textContent = r.timeframe;
  $("f-trend").textContent = r.trend;
  for (const [id, items] of [["f-observations", r.observations], ["f-levels", r.key_levels]]) {
    const ul = $(id);
    ul.replaceChildren(...(items.length ? items : ["none reported"]).map((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      return li;
    }));
  }
  const cost = typeof r.cost_usd === "number" ? ` · API cost about $${r.cost_usd.toFixed(3)}` : "";
  $("f-meta").textContent = `Analyzed by ${r.model}${cost}`;
  show("result");
}

async function handleFile(file) {
  if (!file) return;
  if (!navigator.onLine) {
    showError("You're offline. Connect to the internet to analyze a chart.");
    return;
  }
  let prepared;
  try {
    prepared = await prepareImage(file);
  } catch {
    showError("This photo couldn't be opened. Please take it again.");
    return;
  }
  $("preview").src = prepared.dataUrl;
  $("preview-result").src = prepared.dataUrl;
  $("status-text").textContent = "Analyzing chart…";
  show("busy");
  try {
    renderResult(await postAnalysis(prepared.base64));
  } catch (err) {
    showError(err.name === "AbortError"
      ? "The analysis took too long and was stopped. No signal is shown. Please try again."
      : err.message || "Analysis failed. No signal is shown.");
  }
}

function openCamera() {
  const input = $("photo");
  input.value = "";       // allow picking the same file twice
  input.click();
}

// The big home button is a <label for="photo">, so a tap opens the camera natively (iOS needs a
// direct user gesture). Clear the old value first so the same photo can be chosen twice.
$("analyze-btn").addEventListener("click", () => { $("photo").value = ""; });
$("photo").addEventListener("change", (e) => handleFile(e.target.files && e.target.files[0]));
$("analyze-btn").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCamera(); }
});
$("again-btn").addEventListener("click", openCamera);
$("retry-btn").addEventListener("click", openCamera);

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => { /* PWA install needs HTTPS; app still works */ });
  });
}
