"""
Local video generation jobs for Maya (LTX-2 on Apple Silicon via MLX).

Standard library only. Spawns the `ltx-2-mlx` CLI from tools/ltx/ltx-2-mlx/.venv,
streams its progress, and stores finished clips in data/videos/.

Jobs are serialised through one worker lock because a render uses most of the
machine's memory. Status is persisted to data/videos/<id>.json so the UI can
recover after a page reload or server restart.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOL_DIR = ROOT / "tools" / "ltx" / "ltx-2-mlx"
VENV_BIN = TOOL_DIR / ".venv" / "bin"
CLI = VENV_BIN / "ltx-2-mlx"
HF_CLI = VENV_BIN / "hf"
VIDEOS = ROOT / "data" / "videos"

DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q4"
# Files needed for the distilled text-to-video pipeline (skips dev transformer, LoRAs, upscalers).
TEXT_ENCODER = "mlx-community/gemma-3-12b-it-4bit"   # LTX-2 encodes prompts with Gemma 3 (~8 GB)
DISTILLED_FILES = [
    "transformer-distilled.safetensors", "connector.safetensors",
    "vae_decoder.safetensors", "vae_encoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "audio_vae.safetensors", "vocoder.safetensors",
    "*.json", "LICENSE*", "README*",
]
REQUIRED = ["transformer-distilled.safetensors", "connector.safetensors", "vae_decoder.safetensors",
            "vae_encoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors", "audio_vae.safetensors", "vocoder.safetensors"]

_jobs = {}            # id -> dict
_lock = threading.Lock()
_worker_lock = threading.Lock()   # one render at a time
_procs = {}           # id -> Popen


def hf_cache_dir():
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def model_cache_path(model):
    return hf_cache_dir() / ("models--" + model.replace("/", "--"))


def _snapshot_has(model, names):
    base = model_cache_path(model) / "snapshots"
    if not base.exists():
        return False
    for snap in base.iterdir():
        try:
            if all((snap / n).exists() and (snap / n).stat().st_size > 1_000_000 for n in names):
                return True
        except OSError:
            pass
    return False


def text_encoder_downloaded():
    return _snapshot_has(TEXT_ENCODER, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])


def model_downloaded(model=DEFAULT_MODEL):
    """True when every file the distilled pipeline loads is in the HF cache (LTX pack + Gemma text encoder)."""
    return _snapshot_has(model, REQUIRED) and text_encoder_downloaded()


def download_progress(model=DEFAULT_MODEL):
    """Rough bytes-on-disk for the LTX pack + text encoder (incomplete blobs included)."""
    total = 0
    for m in (model, TEXT_ENCODER):
        p = model_cache_path(m) / "blobs"
        if not p.exists():
            continue
        for f in p.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def status():
    return {
        "installed": CLI.exists(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "model": DEFAULT_MODEL,
        "model_downloaded": model_downloaded(),
        "text_encoder_downloaded": text_encoder_downloaded(),
        "model_bytes_on_disk": download_progress(),
        "model_bytes_expected": 28_500_000_000,
        "busy": _worker_lock.locked(),
        "jobs_pending": sum(1 for j in _jobs.values() if j["status"] in ("queued", "running")),
    }


def _save(job):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    tmp = VIDEOS / (job["id"] + ".json.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1))
    os.replace(tmp, VIDEOS / (job["id"] + ".json"))


def load_jobs():
    VIDEOS.mkdir(parents=True, exist_ok=True)
    for f in VIDEOS.glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:  # noqa
            continue
        if j.get("status") in ("queued", "running"):
            j["status"] = "error"
            j["error"] = "Server restarted while this clip was rendering."
            _save(j)
        _jobs[j["id"]] = j


def get(job_id):
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def cancel(job_id):
    with _lock:
        j = _jobs.get(job_id)
        p = _procs.get(job_id)
    if not j:
        return False
    if p and p.poll() is None:
        try:
            p.terminate()
        except Exception:  # noqa
            pass
    with _lock:
        if j["status"] in ("queued", "running"):
            j["status"] = "cancelled"
            j["error"] = "Stopped."
            j["finished"] = time.time()
            _save(j)
    return True


def _snap_frames(seconds, fps):
    """LTX wants 8k+1 frames."""
    n = max(9, int(round(seconds * fps)))
    k = max(1, round((n - 1) / 8))
    return 8 * k + 1


def submit(prompt, settings, unload_ollama=None):
    """Queue a text-to-video render. Returns the job dict."""
    width = int(settings.get("video_width") or 768)
    height = int(settings.get("video_height") or 512)
    fps = int(settings.get("video_fps") or 24)
    seconds = float(settings.get("video_seconds") or 4)
    frames = _snap_frames(seconds, fps)
    model = settings.get("video_model") or DEFAULT_MODEL
    job = {
        "id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "prompt": prompt, "model": model,
        "width": width - width % 64 or 64, "height": height - height % 64 or 64,
        "frames": frames, "fps": fps, "seconds": round((frames - 1) / fps, 2),
        "status": "queued", "progress": 0.0, "stage": "queued",
        "created": time.time(), "started": None, "finished": None,
        "url": None, "error": None, "log": [],
    }
    with _lock:
        _jobs[job["id"]] = job
    _save(job)
    t = threading.Thread(target=_run, args=(job["id"], settings, unload_ollama), daemon=True)
    t.start()
    return dict(job)


PCT = re.compile(r"(\d{1,3})%")
STEP = re.compile(r"(\d+)\s*/\s*(\d+)")


def _run(job_id, settings, unload_ollama):
    with _worker_lock:
        with _lock:
            job = _jobs[job_id]
            if job["status"] == "cancelled":
                return
            job["status"] = "running"
            job["started"] = time.time()
            job["stage"] = "freeing memory"
        _save(job)

        if unload_ollama:
            try:
                unload_ollama()
            except Exception:  # noqa
                pass

        if not CLI.exists():
            _fail(job, "Video engine is not installed (tools/ltx/ltx-2-mlx/.venv missing). See README.")
            return
        if not model_downloaded(job["model"]):
            _fail(job, f"Video model {job['model']} is not downloaded yet. Run ./tools/ltx/download.sh and try again.")
            return

        out = VIDEOS / (job_id + ".mp4")
        cmd = [str(CLI), "generate", "--prompt", job["prompt"], "--distilled",
               "--model", job["model"],
               "-H", str(job["height"]), "-W", str(job["width"]),
               "-f", str(job["frames"]), "--frame-rate", str(job["fps"]),
               "-o", str(out)]
        seed = settings.get("video_seed")
        if seed not in (None, "", 0):
            cmd += ["--seed", str(int(seed))]
        if settings.get("video_low_ram"):
            cmd += ["--low-ram"]
        extra = settings.get("video_extra_args")
        if extra:
            cmd += str(extra).split()

        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")           # never reach for the network mid-apocalypse
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["PATH"] = str(VENV_BIN) + os.pathsep + env.get("PATH", "")

        with _lock:
            job["stage"] = "loading model"
            job["cmd"] = " ".join(cmd)
        _save(job)
        try:
            p = subprocess.Popen(cmd, cwd=str(TOOL_DIR), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        except Exception as e:  # noqa
            _fail(job, f"Could not start the video engine: {e}")
            return
        with _lock:
            _procs[job_id] = p

        buf = ""
        try:
            while True:
                ch = p.stdout.read(1)
                if ch == "" and p.poll() is not None:
                    break
                if ch in ("\r", "\n"):
                    line = buf.strip()
                    buf = ""
                    if line:
                        _ingest(job, line)
                else:
                    buf += ch
        except Exception as e:  # noqa
            _ingest(job, f"[reader error] {e}")
        rc = p.wait()
        with _lock:
            _procs.pop(job_id, None)
            if job["status"] == "cancelled":
                _save(job)
                return
        if rc == 0 and out.exists() and out.stat().st_size > 0:
            with _lock:
                job["status"] = "done"
                job["progress"] = 1.0
                job["stage"] = "done"
                job["url"] = f"/videos/{out.name}"
                job["finished"] = time.time()
                job["size"] = out.stat().st_size
            _save(job)
        else:
            tail = "\n".join(job["log"][-6:])
            _fail(job, f"Render failed (exit {rc}).\n{tail}")


# Progress timeline (fractions of the bar). The distilled two-stage pipeline denoises
# twice: a quick half-res pass, then a slower full-res refine pass, then decodes.
_PREP_END, _PASS1_END, _PASS2_END, _DECODE_END = 0.12, 0.48, 0.92, 0.99


def _ingest(job, line):
    low = line.lower()
    with _lock:
        job["log"].append(line[:300])
        if len(job["log"]) > 80:
            del job["log"][:-80]
        passes = job.setdefault("_passes", 1)
        cur = job.setdefault("_pass", 0)
        if "two-stage" in low:
            job["_passes"] = passes = 2
        if "[estimate] denoising:" in low and "forwards" in low:
            job["_pass"] = cur = cur + 1
            job["stage"] = "rendering frames" if cur == 1 else "refining frames"
        s = STEP.search(line)
        if "denois" in low and s and int(s.group(2)) > 0:
            frac = min(1.0, int(s.group(1)) / int(s.group(2)))
            if passes == 1 or cur <= 1:
                lo, hi = _PREP_END, (_PASS2_END if passes == 1 else _PASS1_END)
            else:
                lo, hi = _PASS1_END, _PASS2_END
            job["progress"] = max(job["progress"], round(lo + (hi - lo) * frac, 3))
        if "load" in low and "text encoder" in low:
            job["stage"] = "loading language model"
            job["progress"] = max(job["progress"], 0.02)
        elif "encoding prompt" in low:
            job["stage"] = "reading prompt"
            job["progress"] = max(job["progress"], 0.05)
        elif "loading transformer" in low:
            job["stage"] = "loading video model"
            job["progress"] = max(job["progress"], 0.09)
        elif "loading decoders" in low:
            job["stage"] = "decoding video"
            job["progress"] = max(job["progress"], _PASS2_END)
        elif "decoding video" in low or "vae-decode" in low:
            job["stage"] = "decoding video + sound"
            job["progress"] = max(job["progress"], 0.95)
        elif low.startswith("saved to"):
            job["stage"] = "encoding mp4"
            job["progress"] = max(job["progress"], _DECODE_END)
    if time.time() - job.get("_saved", 0) > 2:
        job["_saved"] = time.time()
        _save({k: v for k, v in job.items() if not k.startswith("_")})


def _fail(job, msg):
    with _lock:
        job["status"] = "error"
        job["error"] = msg
        job["finished"] = time.time()
    _save({k: v for k, v in job.items() if not k.startswith("_")})
