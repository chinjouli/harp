"""Generate a self-contained HTML human-evaluation interface.

Embeds the episode FLAC and all referenced WAV clips as base64.
Presents MCQ queries with inline audio-example play buttons, a
global note-taking timeline, per-question confidence + difficulty
ratings, and downloads all answers as JSON on submit.

Paths are resolved from a HARP conf YAML:
  dataset.data_dir          → flac at data_dir/podcasts_flac/
                            → WAV clips at data_dir/../audio/
  inference.queries         → queries JSONL (all episodes combined)
  inference.audio_dir       → override for flac dir (optional)

Usage (from the repo root):
    python human_eval/make_eval.py \\
        --conf conf/emotion_hybrid_all.yaml \\
        --episode MSP-PODCAST_0046 \\
        --output human_eval/eval_MSP-PODCAST_0046.html
"""
import argparse
import base64
import json
import re
import subprocess
from pathlib import Path

import yaml

WAV_PAT = re.compile(r"\[([^\]]+\.(wav|mp3))\]")

_MIME = {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg"}


def _b64_uri(path: Path, mime: str | None = None) -> str:
    mime = mime or _MIME.get(path.suffix.lower(), "audio/wav")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _b64_uri_resampled(path: Path, sr: int = 16000, fmt: str = "wav") -> str:
    """Resample to mono at sr via ffmpeg; fmt='wav' or 'mp3'."""
    if fmt == "mp3":
        codec = ["-codec:a", "libmp3lame", "-b:a", "32k"]
        mime = "audio/mpeg"
    else:
        codec = ["-acodec", "pcm_s16le"]
        mime = "audio/wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ar", str(sr), "-ac", "1",
         *codec, "-f", fmt, "pipe:1"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.decode()[-300:]}")
    b64 = base64.b64encode(r.stdout).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _js(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _inject_choices(queries: list, task: str, options: dict) -> None:
    """Fill choices/input_type from query_options; fall back to 'default'."""
    task_opts = options.get(task, {})
    for q in queries:
        if "choices" in q:
            continue
        opt = (task_opts.get(q.get("query_type", ""))
               or task_opts.get("default"))
        if opt is None:
            continue
        q["input_type"] = opt["type"]
        q["choices"] = opt.get("choices", [])
        for k in ("placeholder", "note", "multi"):
            if k in opt:
                q[k] = opt[k]


# Placeholder text → field name (emotion task)
_AUDIO_PLACEHOLDERS = [
    ("[A's audio example]",  "audio_spk"),
    ("[B's audio example]",  "audio_spk2"),
    ("[audio example]",      "audio_spk"),
    ("[audio example]",      "audio_emo"),  # fallback for locate_hard
    ("[speaker example]",    "audio_spk"),
    ("[emotion example]",    "audio_emo"),
]


def _patch_query_audio(
    queries: list, task: str, out_dir: Path, audio_dir: Path
) -> dict:
    """Inject clip play-buttons into query text; return {name: Path} map."""
    path_map: dict[str, Path] = {}
    for q in queries:
        # Health: derive episode_id from audio_path stem
        if task == "health" and "audio_path" in q and "episode_id" not in q:
            q["episode_id"] = Path(q["audio_path"]).stem

        text = q.get("query", "")

        # Emotion: replace [placeholder] with [basename.wav]
        for placeholder, field in _AUDIO_PLACEHOLDERS:
            if placeholder not in text:
                continue
            rel = q.get(field)
            if not rel:
                continue
            p = out_dir / rel
            name = p.name
            text = text.replace(placeholder, f"[{name}]")
            if p.exists():
                path_map[name] = p

        # Music "audio" type: strip inline prefix, prepend clip buttons
        if task == "music" and q.get("query_type") == "audio":
            # Remove the "Song A: [...]. Song B: [...]. ... " prefix
            text = re.sub(
                r"^Song [AB]: \[.*?\]\. Song [AB]: \[.*?\]\. [^?]*\. ",
                "", text,
            )
            clips_prefix = ""
            for key, label in [("song_a", "Song A"), ("song_b", "Song B")]:
                rel = (q.get(key) or {}).get("file_name")
                if not rel:
                    continue
                p = audio_dir.parent / rel
                name = p.name
                clips_prefix += f"{label}: [{name}]\n"
                if p.exists():
                    path_map[name] = p
            text = clips_prefix + text

        q["query"] = text
    return path_map


def _collect_episode_audios(
    queries: list, task: str, flac_dir: Path, audio_dir: Path
) -> dict:
    """Return {episode_id: Path-to-main-audio}, preserving query order."""
    result: dict[str, Path] = {}
    for q in queries:
        ep = q.get("episode_id")
        if not ep or ep in result:
            continue
        if task == "health":
            p = flac_dir / Path(q["audio_path"]).name
        elif task == "music":
            p = flac_dir / f"{ep}.mp3"
        else:
            p = flac_dir / f"{ep}.flac"
        try:
            accessible = p.exists()
        except OSError:
            accessible = p.open("rb").close() is None  # stat fails but open works
        if accessible:
            result[ep] = p
        else:
            print(f"  WARNING: main audio for {ep} not found at {p}")
    return result


# CSS
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:system-ui,'Segoe UI',sans-serif;
  background:#f0f2f5;color:#222;
  padding:24px;max-width:860px;margin:0 auto;
}
/* page header */
.page-header{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:16px;
}
h1{font-size:1.05rem;font-weight:600;color:#444;margin:0}
.header-controls{display:flex;align-items:center;gap:12px;flex-shrink:0}
#faq-btn{
  padding:5px 12px;border:1.5px solid #ccc;border-radius:6px;
  background:#fff;font-size:.78rem;cursor:pointer;color:#555;white-space:nowrap;
}
#faq-btn:hover{background:#f0f0f0;border-color:#bbb}
/* instructions modal */
#faq-overlay{
  display:none;position:fixed;inset:0;
  background:rgba(0,0,0,.45);z-index:900;
  align-items:center;justify-content:center;
}
#faq-overlay.open{display:flex}
#faq-box{
  background:#fff;border-radius:10px;padding:24px 28px 20px;
  max-width:560px;width:90%;max-height:80vh;overflow-y:auto;
  box-shadow:0 8px 40px rgba(0,0,0,.25);position:relative;
}
#faq-box h2{font-size:.95rem;font-weight:600;margin-bottom:12px;color:#333}
#faq-body{font-size:.83rem;color:#444;line-height:1.75;white-space:pre-wrap}
#faq-close{
  position:absolute;top:10px;right:14px;border:none;
  background:none;font-size:1.1rem;cursor:pointer;color:#999;
}
#faq-close:hover{color:#333}
.section{
  background:#fff;border-radius:10px;
  padding:18px 20px;margin-bottom:14px;
  box-shadow:0 1px 3px rgba(0,0,0,.07);
}

/* audio controls */
.time-row{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:6px;
}
#time-display{
  font-variant-numeric:tabular-nums;
  font-size:.88rem;color:#555;font-weight:500;
}
.track-layout{
  display:grid;
  grid-template-columns:36px 1fr;
  grid-template-rows:auto auto auto;
  column-gap:10px;row-gap:0;
  align-items:center;
  margin-bottom:8px;
}
#play-btn{
  grid-column:1;grid-row:1;
  width:26px;height:26px;border-radius:50%;
  border:none;background:#222;color:#fff;
  font-size:11px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  justify-self:center;
}
#play-btn:hover{background:#444}
.notes-side-label{
  grid-column:1;grid-row:2;
  font-size:.65rem;color:#aaa;
  align-self:center;display:flex;
  align-items:center;justify-content:center;gap:2px;
  position:relative;cursor:default;
  margin-top:6px;
}
#speed-ctl{
  display:flex;align-items:center;gap:5px;
  font-size:.73rem;color:#888;
}
#speed-sel{
  border:1px solid #ddd;border-radius:4px;
  font-size:.73rem;padding:1px 4px;cursor:pointer;
}
/* hover time tooltip */
#hover-tip{
  position:fixed;z-index:300;
  background:#333;color:#fff;
  font-size:.68rem;padding:2px 7px;border-radius:3px;
  pointer-events:none;display:none;
  font-variant-numeric:tabular-nums;
  white-space:nowrap;
}

/* seek bar */
#seek-wrap{
  grid-column:2;grid-row:1;
  position:relative;height:18px;cursor:pointer;
  display:flex;align-items:center;
}
#seek-bg{
  width:100%;height:5px;background:#e0e0e0;
  border-radius:3px;position:relative;
}
#seek-buf,#seek-prog{
  position:absolute;left:0;top:0;height:100%;border-radius:3px;width:0%;
}
#seek-buf{background:#c8c8c8}
#seek-prog{background:#222}
#seek-head{
  position:absolute;top:50%;width:14px;height:14px;
  background:#222;border-radius:50%;
  transform:translate(-50%,-50%);pointer-events:none;left:0%;
}
#seek-wrap:hover #seek-head{transform:translate(-50%,-50%) scale(1.25)}

/* ticks */
#seek-ticks{
  grid-column:2;grid-row:3;
  position:relative;height:16px;margin-bottom:2px;
  font-size:.62rem;color:#aaa;
}
.s-tick{position:absolute;transform:translateX(-50%);white-space:nowrap}
.s-tick::before{
  content:"";display:block;
  width:1px;height:4px;background:#ccc;margin:0 auto 1px;
}

/* notes track */
#notes-track{
  grid-column:2;grid-row:2;
  position:relative;height:18px;background:#e8e8e8;
  border-radius:4px;cursor:crosshair;overflow:visible;
  margin-top:6px;
}
#notes-track:hover{background:#e0e0e0}
#notes-track:active{background:#d8d8d8}
#notes-ph{
  position:absolute;top:0;width:2px;height:100%;
  background:#aaa;pointer-events:none;opacity:.5;
  border-radius:1px;
}
.note-marker{
  position:absolute;top:0;width:0;height:0;
  border-left:6px solid transparent;
  border-right:6px solid transparent;
  border-bottom:11px solid #f9a825;
  transform:translateX(-50%);cursor:pointer;z-index:10;
}
.note-marker:hover{filter:brightness(.8)}

/* note popover */
#note-pop{
  position:fixed;z-index:200;
  background:#fff;border:1px solid #d0d0d0;border-radius:8px;
  padding:10px 12px;min-width:230px;
  box-shadow:0 4px 18px rgba(0,0,0,.14);display:none;
}
#note-pop-time{
  font-size:.7rem;color:#888;margin-bottom:5px;
  font-variant-numeric:tabular-nums;
}
#note-ta{
  width:100%;height:64px;resize:vertical;
  border:1px solid #ddd;border-radius:4px;
  font-size:.8rem;padding:6px;font-family:inherit;margin-bottom:7px;
}
.pop-btns{display:flex;gap:5px;justify-content:flex-end}
.pop-btns button{
  padding:3px 10px;border:1px solid #ccc;border-radius:4px;
  font-size:.75rem;cursor:pointer;background:#fff;color:#444;
}
.pop-btns button.primary{background:#222;color:#fff;border-color:#222}
.pop-btns button.danger{
  background:#c62828;color:#fff;border-color:#c62828;
}

/* query panel */
#q-nav{
  display:flex;align-items:center;gap:10px;margin-bottom:14px;
}
#q-counter{font-size:.78rem;color:#888;white-space:nowrap}
.q-prog-bar{flex:1;height:4px;background:#eee;border-radius:2px}
.q-prog-fill{
  height:100%;background:#43a047;border-radius:2px;transition:width .2s;
}
#q-text{font-size:.9rem;line-height:1.65;margin-bottom:14px}
.clip-btn{
  display:inline-flex;align-items:center;gap:4px;
  padding:2px 9px;border:1px solid #bbb;border-radius:12px;
  background:#fafafa;font-size:.75rem;cursor:pointer;
  color:#444;vertical-align:middle;margin:0 2px;
}
.clip-btn:hover:not(:disabled){background:#ebebeb}
.clip-btn.playing{background:#fff8e1;border-color:#f9a825}
.clip-btn:disabled{opacity:.45;cursor:not-allowed}
.clip-icon{font-size:.8rem}
.clip-name{font-size:.7rem;color:#666}

/* choices */
#choices{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.choice{
  padding:7px 18px;border-radius:20px;border:1.5px solid #e0e0e0;
  cursor:pointer;font-size:.86rem;
  transition:border-color .08s,background .08s;
}
.choice:hover{border-color:#bbb;background:#fafafa}
.choice.selected{border-color:#222;background:#222;color:#fff;font-weight:500}
.no-choices{font-size:.8rem;color:#aaa;font-style:italic;margin:8px 0}

/* timestamp input */
.ts-wrap{margin-bottom:18px}
.ts-note{font-size:.78rem;color:#666;margin-bottom:8px}
.ts-input{
  width:120px;padding:7px 10px;border:1.5px solid #ccc;
  border-radius:6px;font-size:.92rem;font-family:inherit;
  font-variant-numeric:tabular-nums;
}
.ts-input:focus{outline:none;border-color:#555}

/* sub-questions */
.q-sub{display:flex;flex-direction:column;gap:12px;margin-bottom:18px}
.sub-group{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.sub-label{font-size:.78rem;color:#555;font-weight:500;white-space:nowrap}
.hint-icon{
  display:inline-flex;align-items:center;justify-content:center;
  width:15px;height:15px;border-radius:50%;
  background:#e0e0e0;color:#666;font-size:.62rem;font-weight:700;
  cursor:help;margin-left:4px;vertical-align:middle;
  position:relative;
}
.hint-icon[title]:hover::after{
  content:attr(title);
  position:absolute;bottom:120%;left:50%;transform:translateX(-50%);
  background:#333;color:#fff;font-size:.68rem;padding:4px 10px;
  border-radius:4px;pointer-events:none;z-index:400;
  font-weight:400;width:max-content;max-width:600px;
  white-space:pre-wrap;text-align:left;line-height:1.5;
}
.btn-group{display:flex;gap:5px;flex-wrap:wrap}
.btn-group button{
  padding:4px 13px;border:1.5px solid #ddd;border-radius:16px;
  background:#fff;font-size:.78rem;cursor:pointer;color:#555;
}
.btn-group button:hover{background:#f0f0f0;border-color:#bbb}
.btn-group button.sel{background:#333;color:#fff;border-color:#333}

/* nav footer */
#q-footer{
  display:flex;justify-content:space-between;align-items:center;
}
.nav-btn{
  padding:7px 18px;border:1.5px solid #c0c0c0;border-radius:6px;
  background:#fff;font-size:.82rem;cursor:pointer;color:#444;
}
.nav-btn:hover:not(:disabled){background:#f0f0f0}
.nav-btn:disabled{opacity:.3;cursor:default}
#answered-count{font-size:.74rem;color:#aaa}

/* submit */
#submit-section{text-align:center}
#submit-btn{
  padding:10px 30px;background:#2e7d32;color:#fff;
  border:none;border-radius:6px;font-size:.88rem;
  cursor:pointer;font-weight:600;
}
#submit-btn:hover{background:#1b5e20}
#submit-msg{font-size:.76rem;color:#888;margin-top:8px}
"""

# JavaScript
JS_TMPL = """
const TASK_LABEL      = __TASK_LABEL__;
const TASK_KEY        = __TASK_KEY__;
const QUESTION_SET_ID = __QUESTION_SET_ID__;
const INSTRUCTIONS    = __INSTRUCTIONS__;
const QUERIES         = __QUERIES__;
const CLIPS           = __CLIPS__;
const DURATION        = __DURATION__;
const EPISODE_IDS     = __EPISODE_IDS__;
const AUDIO_SRCS      = __AUDIO_SRCS__;

const audio = document.getElementById("audio");
let currentEpId = EPISODE_IDS[0];
audio.src = AUDIO_SRCS[currentEpId];
function switchEpisode(epId) {
  if (epId === currentEpId) return;
  audio.pause();
  audio.src = AUDIO_SRCS[epId];
  currentEpId = epId;
  renderMarkers();
}

/* state */
const answers = QUERIES.map(q => ({
  choice: q.multi ? [] : null, sure: null, difficulty: null,
}));
/* per-episode timeline notes */
const episodeNotes = {};
const noteSeqs = {};
EPISODE_IDS.forEach(id => { episodeNotes[id] = []; noteSeqs[id] = 0; });
let notes      = [];
let noteSeq    = 0;
let qIdx       = 0;
let activeClip = null;

/* helpers */
function fmt(s) {
  if (!isFinite(s)) return "\\u2013:\\u2013\\u2013";
  const m = Math.floor(s / 60), sc = Math.floor(s % 60);
  return m + ":" + String(sc).padStart(2, "0");
}
function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

/* seek bar */
const seekBg   = document.getElementById("seek-bg");
const seekProg = document.getElementById("seek-prog");
const seekHead = document.getElementById("seek-head");
const seekBuf  = document.getElementById("seek-buf");
const notesPh  = document.getElementById("notes-ph");
const timeLbl  = document.getElementById("time-display");

function seekFrac(cx) {
  const r = seekBg.getBoundingClientRect();
  return Math.max(0, Math.min(1, (cx - r.left) / r.width));
}

let seeking = false;
seekBg.addEventListener("mousedown", e => {
  seeking = true;
  audio.currentTime =
    seekFrac(e.clientX) * (audio.duration || DURATION);
  e.preventDefault();
});
document.addEventListener("mousemove", e => {
  if (!seeking) return;
  audio.currentTime =
    seekFrac(e.clientX) * (audio.duration || DURATION);
});
document.addEventListener("mouseup", () => { seeking = false; });

function updateSeek() {
  const dur = audio.duration || DURATION;
  const f   = audio.currentTime / dur;
  seekProg.style.width = (f * 100) + "%";
  seekHead.style.left  = (f * 100) + "%";
  notesPh.style.left   = (f * 100) + "%";
  timeLbl.textContent  =
    fmt(audio.currentTime) + " / " + fmt(audio.duration);
}
audio.addEventListener("timeupdate", updateSeek);
audio.addEventListener("progress", () => {
  if (!audio.buffered.length) return;
  const b = audio.buffered.end(audio.buffered.length - 1)
            / (audio.duration || DURATION);
  seekBuf.style.width = (b * 100) + "%";
});
audio.addEventListener("loadedmetadata", () => {
  buildTicks(); updateSeek(); renderMarkers();
});

/* play / pause */
const playBtn = document.getElementById("play-btn");
playBtn.addEventListener("click", togglePlay);
audio.addEventListener("play",  () => { playBtn.textContent = "\\u23f8"; });
audio.addEventListener("pause", () => { playBtn.textContent = "\\u25b6"; });
function togglePlay() {
  if (audio.paused) audio.play(); else audio.pause();
}

/* hover time tooltip */
const hoverTip = document.getElementById("hover-tip");
function showTip(cx, cy, t) {
  hoverTip.textContent = fmt(t);
  hoverTip.style.display = "block";
  hoverTip.style.left = (cx + 8) + "px";
  hoverTip.style.top  = (cy - 26) + "px";
}
function hideTip() { hoverTip.style.display = "none"; }
seekBg.addEventListener("mousemove", e => {
  showTip(e.clientX, e.clientY,
    seekFrac(e.clientX) * (audio.duration || DURATION));
});
seekBg.addEventListener("mouseleave", hideTip);
document.getElementById("speed-sel").addEventListener("change", e => {
  audio.playbackRate = parseFloat(e.target.value);
});

/* keyboard shortcuts */
document.addEventListener("keydown", e => {
  const tag = e.target.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); togglePlay(); }
  if (e.key === "ArrowLeft")
    audio.currentTime = Math.max(0, audio.currentTime - 5);
  if (e.key === "ArrowRight")
    audio.currentTime =
      Math.min(audio.duration || DURATION, audio.currentTime + 5);
});

/* tick marks */
function buildTicks() {
  const div = document.getElementById("seek-ticks");
  div.innerHTML = "";
  const d = audio.duration || DURATION;
  const steps = [10,20,30,60,120,300,600,1800];
  const step  = steps.find(s => d / s <= 14) || 1800;
  for (let t = 0; t <= d; t += step) {
    const el = document.createElement("div");
    el.className  = "s-tick";
    el.style.left = (t / d * 100) + "%";
    el.textContent = fmt(t);
    div.appendChild(el);
  }
}
buildTicks();

/* notes track */
const notesTrack = document.getElementById("notes-track");
const notePop    = document.getElementById("note-pop");
const noteTA     = document.getElementById("note-ta");
const noteDelBtn = document.getElementById("note-del-btn");
let editNoteId  = null;
let editNoteEpId = null;
let popTime     = 0;

notesTrack.addEventListener("mousemove", e => {
  const r = notesTrack.getBoundingClientRect();
  const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  showTip(e.clientX, e.clientY, f * (audio.duration || DURATION));
});
notesTrack.addEventListener("mouseleave", hideTip);

notesTrack.addEventListener("click", e => {
  if (e.target.classList.contains("note-marker")) return;
  const r = notesTrack.getBoundingClientRect();
  popTime     = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
                * (audio.duration || DURATION);
  editNoteId  = null;
  editNoteEpId = currentEpId;
  noteTA.value = "";
  noteDelBtn.style.display = "none";
  document.getElementById("note-pop-time").textContent = "@" + fmt(popTime);
  openPop(e.clientX, e.clientY);
});

function openPop(cx, cy) {
  notePop.style.display = "block";
  const h = notePop.offsetHeight, w = notePop.offsetWidth;
  let x = cx + 10, y = cy - h - 10;
  if (y < 8)                         y = cy + 10;
  if (x + w > window.innerWidth - 8) x = cx - w - 10;
  notePop.style.left = x + "px";
  notePop.style.top  = y + "px";
  noteTA.focus();
}

document.getElementById("note-save-btn").addEventListener("click", () => {
  const text = noteTA.value.trim();
  if (!text) { closePop(); return; }
  const ns = episodeNotes[editNoteEpId];
  if (editNoteId !== null) {
    const n = ns.find(n => n.id === editNoteId);
    if (n) n.text = text;
  } else {
    ns.push({ id: noteSeqs[editNoteEpId]++, time: popTime, text });
  }
  closePop();
  renderMarkers();
});

noteDelBtn.addEventListener("click", () => {
  if (editNoteId !== null)
    episodeNotes[editNoteEpId] =
      episodeNotes[editNoteEpId].filter(n => n.id !== editNoteId);
  closePop();
  renderMarkers();
});

document.getElementById("note-cancel-btn").addEventListener("click", closePop);

function closePop() {
  notePop.style.display = "none";
  editNoteId = null;
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && notePop.style.display !== "none") closePop();
});

function renderMarkers() {
  notesTrack.querySelectorAll(".note-marker").forEach(el => el.remove());
  const ns = episodeNotes[currentEpId] || [];
  const d  = audio.duration || DURATION;
  [...ns].sort((a, b) => a.time - b.time).forEach(n => {
    const el = document.createElement("div");
    el.className  = "note-marker";
    el.style.left = (n.time / d * 100) + "%";
    el.title      = fmt(n.time) + ": " + n.text;
    el.addEventListener("click", e => {
      e.stopPropagation();
      editNoteId   = n.id;
      editNoteEpId = currentEpId;
      popTime      = n.time;
      noteTA.value = n.text;
      noteDelBtn.style.display = "";
      document.getElementById("note-pop-time").textContent = "@" + fmt(n.time);
      openPop(e.clientX, e.clientY);
    });
    notesTrack.appendChild(el);
  });
}

/* FAQ modal */
const faqOverlay = document.getElementById("faq-overlay");
document.getElementById("faq-body").textContent = INSTRUCTIONS;
document.getElementById("faq-btn").addEventListener("click", () => {
  faqOverlay.classList.add("open");
});
document.getElementById("faq-close").addEventListener("click", () => {
  faqOverlay.classList.remove("open");
});
faqOverlay.addEventListener("click", e => {
  if (e.target === faqOverlay) faqOverlay.classList.remove("open");
});

/* set page title */
const _h1 = document.querySelector("h1");
if (_h1) _h1.textContent = "Human Evaluation — " + TASK_LABEL;

/* convert raw seconds in query text to mm:ss */
function convertSec(text) {
  return text
    .replace(/From (\\d+) to (\\d+) seconds/g,
      (_, a, b) => "From " + fmt(+a) + " to " + fmt(+b))
    .replace(/Between (\\d+) and (\\d+) seconds/g,
      (_, a, b) => "Between " + fmt(+a) + " and " + fmt(+b));
}

/* clip play buttons (event delegation) */
function stopClip() {
  if (activeClip) { activeClip.pause(); activeClip = null; }
  document.querySelectorAll(".clip-btn.playing").forEach(b => {
    b.classList.remove("playing");
    b.querySelector(".clip-icon").textContent = "\\u25b6";
  });
}

document.addEventListener("click", e => {
  const btn = e.target.closest(".clip-btn[data-clip]");
  if (!btn || btn.disabled) return;
  const fname = btn.dataset.clip;
  if (!(fname in CLIPS)) return;
  if (btn.classList.contains("playing")) { stopClip(); return; }
  stopClip();
  const a = new Audio(CLIPS[fname]);
  activeClip = a;
  btn.classList.add("playing");
  btn.querySelector(".clip-icon").textContent = "\\u23f8";
  a.play();
  a.addEventListener("ended", () => {
    btn.classList.remove("playing");
    btn.querySelector(".clip-icon").textContent = "\\u25b6";
    activeClip = null;
  });
});

/* query rendering */
const WAV_RE = /\\[([^\\]]+\\.(wav|mp3))\\]/g;

function renderQueryText(text) {
  const parts = [];
  let last = 0, m;
  WAV_RE.lastIndex = 0;
  while ((m = WAV_RE.exec(text)) !== null) {
    if (m.index > last)
      parts.push(esc(text.slice(last, m.index)).replace(/\\n/g, "<br>"));
    const fname = m[1], has = fname in CLIPS;
    parts.push(
      `<button class="clip-btn" data-clip="${esc(fname)}"` +
      (has ? "" : ' disabled title="clip not found"') + ">" +
      `<span class="clip-icon">\\u25b6</span> ` +
      `<span class="clip-name">${esc(fname)}</span></button>`
    );
    last = m.index + m[0].length;
  }
  if (last < text.length)
    parts.push(esc(text.slice(last)).replace(/\\n/g, "<br>"));
  return parts.join("");
}

function isAnswered(a) {
  const hasChoice = Array.isArray(a.choice)
    ? a.choice.length > 0 : a.choice !== null;
  return hasChoice && a.sure !== null && a.difficulty !== null;
}

function updateCount() {
  const done = answers.filter(isAnswered).length;
  document.getElementById("answered-count").textContent =
    done + " / " + QUERIES.length + " complete";
}

function renderQuery() {
  const q   = QUERIES[qIdx];
  const ans = answers[qIdx];

  switchEpisode(q.episode_id || EPISODE_IDS[0]);

  document.getElementById("q-counter").textContent =
    "Question " + (qIdx + 1) + " of " + QUERIES.length;
  document.querySelector(".q-prog-fill").style.width =
    ((qIdx + 1) / QUERIES.length * 100) + "%";

  document.getElementById("q-text").innerHTML =
    renderQueryText(convertSec(q.query || ""));

  /* choices / timestamp input */
  const cd = document.getElementById("choices");
  cd.innerHTML = "";
  if (q.input_type === "timestamp") {
    const note = q.note || "Enter the timestamp";
    const ph   = q.placeholder || "mm:ss";
    cd.innerHTML =
      `<div class="ts-wrap">` +
      `<div class="ts-note">${esc(note)}</div>` +
      `<input id="ts-input" class="ts-input" type="text" ` +
      `placeholder="${esc(ph)}" value="${esc(ans.choice || '')}">` +
      `</div>`;
    document.getElementById("ts-input").addEventListener("input", e => {
      ans.choice = e.target.value.trim() || null;
      updateCount();
    });
  } else {
    const choices = q.choices || [];
    if (!choices.length) {
      cd.innerHTML =
        '<p class="no-choices">(Choices not defined for this query)</p>';
    }
    const multi = !!q.multi;
    choices.forEach(c => {
      const div = document.createElement("div");
      const sel = multi
        ? (ans.choice || []).includes(c.label)
        : ans.choice === c.label;
      div.className = "choice" + (sel ? " selected" : "");
      div.textContent = c.text;
      div.addEventListener("click", () => {
        if (multi) {
          const idx = ans.choice.indexOf(c.label);
          if (idx >= 0) ans.choice.splice(idx, 1);
          else ans.choice.push(c.label);
        } else {
          ans.choice = c.label;
        }
        renderQuery();
      });
      cd.appendChild(div);
    });
  }

  /* sure */
  ["yes","no"].forEach(v => {
    const btn = document.getElementById("sure-" + v);
    btn.classList.toggle("sel", ans.sure === v);
    btn.onclick = () => { ans.sure = v; renderQuery(); };
  });

  /* difficulty */
  for (let i = 0; i <= 4; i++) {
    const btn = document.getElementById("diff-" + i);
    btn.classList.toggle("sel", ans.difficulty === i);
    btn.onclick = () => { ans.difficulty = i; renderQuery(); };
  }

  /* nav */
  document.getElementById("prev-btn").disabled = qIdx === 0;
  document.getElementById("next-btn").disabled =
    qIdx === QUERIES.length - 1;

  updateCount();
}

document.getElementById("prev-btn").addEventListener("click", () => {
  if (qIdx > 0) { qIdx--; renderQuery(); }
});
document.getElementById("next-btn").addEventListener("click", () => {
  if (qIdx < QUERIES.length - 1) { qIdx++; renderQuery(); }
});

/* submit */
document.getElementById("submit-btn").addEventListener("click", () => {
  const done = answers.filter(isAnswered).length;
  if (done < QUERIES.length &&
      !confirm((QUERIES.length - done) +
               " question(s) incomplete. Submit anyway?"))
    return;
  const payload = {
    episode_ids:  EPISODE_IDS,
    submitted_at: new Date().toISOString(),
    notes: Object.fromEntries(
      Object.entries(episodeNotes).map(([ep, ns]) =>
        [ep, ns.map(({time, text}) => ({time, text}))]
      )
    ),
    answers: QUERIES.map((q, i) => ({
      query_id:   q.query_id ?? i,
      episode_id: q.episode_id || EPISODE_IDS[0],
      query_type: q.query_type,
      choice:     answers[i].choice,
      sure:       answers[i].sure,
      difficulty: answers[i].difficulty,
    })),
  };
  const blob = new Blob(
    [JSON.stringify(payload, null, 2)],
    {type: "application/json"}
  );
  const a = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = TASK_KEY + "_" + QUESTION_SET_ID + ".json";
  a.click();
  document.getElementById("submit-msg").textContent =
    "Saved \\u2014 you can close this page.";
});

/* init */
renderQuery();
"""

# HTML template
HTML_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Human Evaluation</title>
<style>__CSS__</style>
</head>
<body>
<div class="page-header">
  <h1>Human Evaluation</h1>
  <div class="header-controls">
    <button id="faq-btn">ℹ Instructions</button>
  </div>
</div>

<div id="faq-overlay">
  <div id="faq-box">
    <button id="faq-close">&times;</button>
    <h2>Task Instructions</h2>
    <div id="faq-body"></div>
  </div>
</div>

<div class="section">
  <audio id="audio" preload="metadata"></audio>
  <div class="time-row">
    <span id="time-display">–:–– / –:––</span>
    <div id="speed-ctl">
      <span>Speed</span>
      <select id="speed-sel">
        <option value="0.75">0.75×</option>
        <option value="1" selected>1×</option>
        <option value="1.25">1.25×</option>
        <option value="1.5">1.5×</option>
        <option value="2">2×</option>
      </select>
    </div>
  </div>
  <div class="track-layout">
    <button id="play-btn" title="Space: play/pause">▶</button>
    <div id="seek-wrap">
      <div id="seek-bg">
        <div id="seek-buf"></div>
        <div id="seek-prog"></div>
      </div>
      <div id="seek-head"></div>
    </div>
    <span class="notes-side-label">Notes
      <span class="hint-icon" title="Click bar to add a note">?</span>
    </span>
    <div id="notes-track">
      <div id="notes-ph"></div>
    </div>
    <div id="seek-ticks"></div>
  </div>
  <div id="hover-tip"></div>
</div>

<div class="section">
  <div id="q-nav">
    <span id="q-counter"></span>
    <div class="q-prog-bar"><div class="q-prog-fill"></div></div>
  </div>
  <div id="q-text"></div>
  <div id="choices"></div>
  <div class="q-sub">
    <div class="sub-group">
      <div class="sub-label">Are you sure about your answer?
        <span class="hint-icon" title="Select Yes only when you're certain about your answer">?</span>
      </div>
      <div class="btn-group">
        <button id="sure-yes">Yes</button>
        <button id="sure-no">No</button>
      </div>
    </div>
    <div class="sub-group">
      <div class="sub-label">How difficult is it to answer this question?
        <span class="hint-icon" title="0 = very simple, effortless&#10;4 = too difficult, impossible">?</span>
      </div>
      <div class="btn-group">__DIFF_BTNS__</div>
    </div>
  </div>
  <div id="q-footer">
    <button class="nav-btn" id="prev-btn">← Previous</button>
    <span id="answered-count"></span>
    <button class="nav-btn" id="next-btn">Next →</button>
  </div>
</div>

<div class="section" id="submit-section">
  <button id="submit-btn">Submit &amp; Download Answers</button>
  <div id="submit-msg"></div>
</div>

<div id="note-pop">
  <div id="note-pop-time"></div>
  <textarea id="note-ta" placeholder="Note…"></textarea>
  <div class="pop-btns">
    <button class="danger" id="note-del-btn">Delete</button>
    <button id="note-cancel-btn">Cancel</button>
    <button class="primary" id="note-save-btn">Save</button>
  </div>
</div>

<script>__JS__</script>
</body>
</html>"""


def _load_conf(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_paths(conf: dict) -> tuple[Path, Path, Path]:
    """Return (flac_dir, out_dir, queries_path) from conf."""
    data_dir = Path(conf["dataset"]["data_dir"])
    inf = conf.get("inference", {})
    if "audio_dir" in inf:
        flac_dir = Path(inf["audio_dir"])
    else:
        flac_dir = data_dir / "podcasts_flac"
    out_dir = Path(inf.get("out_dir", "data"))
    queries_path = Path(inf["queries"])
    return flac_dir, out_dir, queries_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--conf", required=True)
    ap.add_argument("--episode", default=None,
                    help="Episode ID, or comma-separated list for multi-episode HTML")
    ap.add_argument("--output",    default=None)
    ap.add_argument("--flac-dir",  default=None)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--queries",       default=None)
    ap.add_argument("--max-per-type",  type=int, default=None,
                    help="Cap number of queries per query_type")
    args = ap.parse_args()

    _TASK_LABELS = {
        "emotion": "Emotion Understanding",
        "music": "Song Aesthetic Evaluation",
        "health": "Clinical Symptom Analysis",
    }

    conf = _load_conf(args.conf)
    flac_dir, out_dir, qpath = _resolve_paths(conf)
    if args.flac_dir:
        flac_dir = Path(args.flac_dir)
    if args.audio_dir:
        out_dir = Path(args.audio_dir)
    if args.queries:
        qpath = Path(args.queries)

    task = conf.get("inference", {}).get("task", "emotion")
    task_label = _TASK_LABELS.get(task, task.replace("_", " ").title())

    # load all queries, derive episode_id early for health
    if qpath.is_dir() and args.episode:
        qpath = qpath / f"{args.episode}.jsonl"
    all_queries: list[dict] = []
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            # health: episode_id from audio_path stem
            if task == "health" and "audio_path" in q and "episode_id" not in q:
                q["episode_id"] = Path(q["audio_path"]).stem
            all_queries.append(q)

    # select episodes
    seen: dict[str, list] = {}
    for q in all_queries:
        ep = q.get("episode_id", "")
        seen.setdefault(ep, []).append(q)

    # Health: --episode may be a set ID from health_episodes.json
    health_set_id: str | None = None
    if task == "health" and args.episode:
        hep_path = Path(__file__).parent / "health_episodes.json"
        if hep_path.exists():
            with open(hep_path, encoding="utf-8") as f:
                health_sets = json.load(f)
            if args.episode.strip() in health_sets:
                health_set_id = args.episode.strip()
                ep_list = health_sets[health_set_id]
            else:
                ep_list = [e.strip() for e in args.episode.split(",")]
        else:
            ep_list = [e.strip() for e in args.episode.split(",")]
    elif args.episode:
        ep_list = [e.strip() for e in args.episode.split(",")]
    else:
        ep_list = list(seen.keys())

    queries = [q for ep in ep_list for q in seen.get(ep, [])]
    if not queries:
        raise ValueError(f"No queries found for {ep_list} in {qpath}")

    if args.max_per_type:
        type_counts: dict[str, int] = {}
        filtered = []
        for q in queries:
            qt = q.get("query_type", "")
            if type_counts.get(qt, 0) < args.max_per_type:
                filtered.append(q)
                type_counts[qt] = type_counts.get(qt, 0) + 1
        queries = filtered

    print(f"Loaded {len(queries)} queries across {len(ep_list)} episode(s)")

    # inject choices
    opts_path = Path(__file__).parent / "query_options.json"
    with open(opts_path, encoding="utf-8") as f:
        options = json.load(f)
    _inject_choices(queries, task, options)
    injected = sum(1 for q in queries if "input_type" in q)
    print(f"  Choices injected for {injected}/{len(queries)} queries")

    # patch query text: inject clip play-buttons
    wav_path_map = _patch_query_audio(queries, task, out_dir, flac_dir)

    # embed clip audio (WAV / MP3 referenced in query text)
    all_clips: set[str] = set()
    for q in queries:
        all_clips.update(m[0] for m in WAV_PAT.findall(q.get("query", "")))

    clips_b64: dict[str, str] = {}
    for name in sorted(all_clips):
        p = wav_path_map.get(name)
        if not (p and p.exists()):
            print(f"  WARNING: clip {name} not found")
            continue
        if task in ("health", "music"):
            clips_b64[name] = _b64_uri_resampled(p, sr=16000, fmt="mp3")
            print(f"  Embedded clip {name} → 16 kHz MP3")
        else:
            clips_b64[name] = _b64_uri(p)
            try:
                kb = p.stat().st_size // 1024
            except OSError:
                kb = "?"
            print(f"  Embedded clip {name} ({kb} KB)")

    # embed main audio per episode
    ep_audio_paths = _collect_episode_audios(queries, task, flac_dir, out_dir)
    audio_srcs_b64: dict[str, str] = {}
    for ep, p in ep_audio_paths.items():
        try:
            mb = p.stat().st_size / 1e6
        except OSError:
            mb = float("nan")
        if task == "health":
            print(f"Resampling {ep}: {p.name} ({mb:.1f} MB → 16 kHz mono WAV) ...")
            audio_srcs_b64[ep] = _b64_uri_resampled(p, sr=16000, fmt="wav")
        elif task == "music":
            print(f"Resampling {ep}: {p.name} ({mb:.1f} MB → 16 kHz mono MP3) ...")
            audio_srcs_b64[ep] = _b64_uri_resampled(p, sr=16000, fmt="mp3")
        else:
            print(f"Encoding {ep}: {p.name} ({mb:.1f} MB) ...")
            audio_srcs_b64[ep] = _b64_uri(p)

    # question set ID
    if health_set_id:
        question_set_id = health_set_id
    elif task == "emotion" and len(ep_list) == 1:
        question_set_id = ep_list[0].removeprefix("MSP-PODCAST_")
    elif task == "music" and len(ep_list) == 1:
        question_set_id = ep_list[0].removeprefix("track_")
    elif len(ep_list) == 1:
        question_set_id = ep_list[0]
    else:
        question_set_id = f"{task}_{len(ep_list)}eps"

    # output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"human_eval/eval_{task}_{question_set_id}.html")

    # load instructions
    inst_path = Path(__file__).parent / "instructions.json"
    instructions_text = ""
    if inst_path.exists():
        with open(inst_path, encoding="utf-8") as f:
            inst_all = json.load(f)
        instructions_text = inst_all.get(task, "")

    diff_btns = "".join(
        f'<button id="diff-{i}">{i}</button>' for i in range(5)
    )
    js = (JS_TMPL
          .replace("__TASK_LABEL__",     _js(task_label))
          .replace("__TASK_KEY__",       _js(task))
          .replace("__QUESTION_SET_ID__", _js(question_set_id))
          .replace("__INSTRUCTIONS__",   _js(instructions_text))
          .replace("__QUERIES__",        _js(queries))
          .replace("__CLIPS__",          _js(clips_b64))
          .replace("__DURATION__",       "7200")
          .replace("__EPISODE_IDS__",    _js(ep_list))
          .replace("__AUDIO_SRCS__",     _js(audio_srcs_b64)))

    html = (HTML_TMPL
            .replace("__CSS__",       CSS)
            .replace("__DIFF_BTNS__", diff_btns)
            .replace("__JS__",        js))

    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Written {out_path}  ({size_mb:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
