"""
Local image generation for Maya: FLUX.2 Klein 4B (4-bit) through mflux on Apple Silicon.

Standard library only. Runs the `mflux-generate-flux2` CLI as a subprocess, parses its
step progress, and saves PNGs into data/images/. Shares the render lock with videogen so
an image and a video never fight for memory at the same time.
"""
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import videogen

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "images"

DEFAULT_MODEL = "flux2-klein-4b"
DEFAULT_WEIGHTS = "ar9av/FLUX.2-klein-4B-mflux-4bit"   # ~4.6 GB, pre-quantized for mflux
WEIGHT_FILES_MIN_BYTES = 3_500_000_000

CANDIDATE_BINS = [
    Path.home() / ".local" / "bin" / "mflux-generate-flux2",
    Path("/opt/homebrew/bin/mflux-generate-flux2"),
]


def cli():
    for p in CANDIDATE_BINS:
        if p.exists():
            return p
    w = shutil.which("mflux-generate-flux2")
    return Path(w) if w else None


def weights_downloaded(repo=DEFAULT_WEIGHTS):
    base = videogen.model_cache_path(repo) / "snapshots"
    if not base.exists():
        return False
    for snap in base.iterdir():
        total = 0
        for f in snap.rglob("*.safetensors"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
        if total > WEIGHT_FILES_MIN_BYTES:
            return True
    return False


def weights_bytes(repo=DEFAULT_WEIGHTS):
    p = videogen.model_cache_path(repo) / "blobs"
    if not p.exists():
        return 0
    return sum((f.stat().st_size for f in p.iterdir() if f.is_file()), 0)


def status():
    c = cli()
    return {
        "installed": c is not None,
        "cli": str(c) if c else None,
        "model": DEFAULT_MODEL,
        "weights": DEFAULT_WEIGHTS,
        "weights_downloaded": weights_downloaded(),
        "weights_bytes_on_disk": weights_bytes(),
        "weights_bytes_expected": 4_600_000_000,
        "ready": c is not None and weights_downloaded(),
        "busy": videogen._worker_lock.locked(),
    }


PCT = re.compile(r"(\d{1,3})%")
STEP = re.compile(r"(\d+)\s*/\s*(\d+)")


def generate(prompt, settings, on_progress=None, cancel_check=None):
    """Blocking. Yields nothing; calls on_progress(completed, total, stage); returns result dict."""
    c = cli()
    if not c:
        return {"error": "Image engine not installed. Run: uv tool install mflux"}
    if not weights_downloaded():
        gb = weights_bytes() / 1e9
        return {"error": f"Image model still downloading ({gb:.1f} GB of ~4.6 GB). It will work automatically once it finishes."}
    width = int(settings.get("image_width") or 1024)
    height = int(settings.get("image_height") or 1024)
    width -= width % 16
    height -= height % 16
    steps = int(settings.get("image_steps") or 0) or 4
    seed = settings.get("image_seed")
    IMAGES.mkdir(parents=True, exist_ok=True)
    iid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = IMAGES / f"{iid}.png"
    # mflux: --model takes the HF repo of the pre-quantized weights, --base-model names the architecture
    cmd = [str(c), "--model", settings.get("image_weights") or DEFAULT_WEIGHTS,
           "--base-model", (settings.get("image_model") if str(settings.get("image_model","")).startswith("flux2-") else DEFAULT_MODEL),
           "--prompt", prompt, "--steps", str(steps),
           "--width", str(width), "--height", str(height),
           "--output", str(out)]
    if seed not in (None, "", 0):
        cmd += ["--seed", str(int(seed))]
    extra = settings.get("image_extra_args")
    if extra:
        cmd += str(extra).split()
    if settings.get("image_low_ram"):
        cmd += ["--low-ram"]
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    t0 = time.time()
    with videogen._worker_lock:
        try:
            p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        except Exception as e:  # noqa
            return {"error": f"Could not start the image engine: {e}"}
        log = []
        buf = ""
        completed, total = 0, steps
        if on_progress:
            on_progress(0, total, "loading model")
        try:
            while True:
                if cancel_check and cancel_check():
                    p.terminate()
                    return {"error": "Stopped."}
                ch = p.stdout.read(1)
                if ch == "" and p.poll() is not None:
                    break
                if ch in ("\r", "\n"):
                    line = buf.strip()
                    buf = ""
                    if not line:
                        continue
                    log.append(line[:240])
                    if len(log) > 60:
                        del log[:-60]
                    s = STEP.search(line)
                    m = PCT.search(line)
                    changed = False
                    if s and 0 < int(s.group(2)) <= 200:
                        completed, total = int(s.group(1)), int(s.group(2))
                        changed = True
                    elif m:
                        completed, total = int(m.group(1)), 100
                        changed = True
                    if changed and on_progress:
                        on_progress(completed, total, "rendering")
                else:
                    buf += ch
        finally:
            rc = p.wait()
    if rc == 0 and out.exists() and out.stat().st_size > 0:
        return {"image_url": f"/images/{out.name}", "model": DEFAULT_MODEL, "prompt": prompt,
                "width": width, "height": height, "seconds": round(time.time() - t0, 1)}
    tail = "\n".join(log[-5:])
    return {"error": f"Image render failed (exit {rc}).\n{tail}"}
