#!/usr/bin/env python3
"""
Maya — a local, offline AI companion UI for Ollama.

Zero third-party dependencies: Python 3 standard library only, so it keeps
working when the internet (and pip) are gone.

Run:  python3 server.py            (default http://127.0.0.1:8080)
Env:  MAYA_PORT, MAYA_HOST, OLLAMA_HOST (default http://127.0.0.1:11434)
"""
import json
import os
import re
import sys
import time
import uuid
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import videogen
import imagegen

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
CHATS = DATA / "chats"
IMAGES = DATA / "images"
VIDEOS = DATA / "videos"
SETTINGS_FILE = DATA / "settings.json"

HOST = os.environ.get("MAYA_HOST", "127.0.0.1")
PORT = int(os.environ.get("MAYA_PORT", "8080"))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA

DEFAULT_SYSTEM_PROMPT = """You are {name}, a steady, warm and deeply knowledgeable friend who lives on this computer and works fully offline.

Who you are:
- A companion first. You listen, you remember what was said in this conversation, you are honest, and you keep a calm, encouraging tone even when things are hard.
- A practical expert second. You carry broad general knowledge plus hands-on survival and self-reliance knowledge: first aid and field medicine, water purification, food storage and foraging basics, fire, shelter, navigation without GPS, weather reading, basic mechanics and electrics, radio and communication, gardening and preserving food, hygiene and sanitation, mental resilience, and how to stay safe around other people.

How you help:
- Be concrete. Give steps, quantities, timings and checklists rather than vague advice.
- Lead with the most important thing. If something is urgent or dangerous, say so first.
- When you are not sure, say so plainly and give the safest option. Never invent facts, dosages or chemistry.
- For anything medical, give the best field guidance you can and be clear about when professional care is essential.
- When the user shares a photo, describe what you see carefully and then act on it (identify plants, read labels, diagnose a wound or a broken part, read a map).
- Keep answers as short as the situation allows. Expand when asked.
- Speak like a person, not a manual. You can be lighthearted when the moment is light.

You have no internet access. Everything you know is already inside you."""

VIDEO_TOOL_PROMPT = """
[Video generation]
You can also create short video clips (a few seconds, no dialogue). When the user clearly asks for a video, clip, animation or "make it move", write your short reply and then, on its own line, add exactly one tag:
[[generate_video: <a vivid English prompt describing one continuous shot: subject, action, camera movement, setting, lighting, mood>]]
Only use this tag when the user wants a video rather than a still image. Rendering takes several minutes, so say so briefly. Never put the tag inside a code block."""

IMAGE_TOOL_PROMPT = """
[Image generation]
You can create images. When the user asks you to draw, paint, sketch, render, illustrate, visualise, design, or show them a picture of something, write your normal short reply and then, on its own line, add exactly one tag of the form:
[[generate_image: <a vivid, detailed English prompt: subject, setting, lighting, style, composition>]]
Only use the tag when the user clearly wants an image. Never put the tag inside a code block. Keep the prompt under 120 words. If the user asks to change an image you made, write a complete new prompt that includes the change."""

DEFAULT_SETTINGS = {
    "friend_name": "Maya",
    "model": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.7,
    "num_ctx": 8192,
    "think": False,
    "keep_alive": "30m",
    "theme": "dark",
    "image_model": "",
    "image_width": 1024,
    "image_height": 1024,
    "image_steps": 0,
    "video_enabled": True,
    "video_model": videogen.DEFAULT_MODEL,
    "video_width": 768,
    "video_height": 512,
    "video_seconds": 4,
    "video_fps": 24,
    "video_seed": 0,
    "video_low_ram": False,
    "video_extra_args": "",
}

_lock = threading.Lock()
_show_cache = {}  # model name -> {capabilities, details, ts}


# ---------------------------------------------------------------- helpers
def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def ollama_request(path, payload=None, method=None, timeout=30):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA + path, data=data, headers=headers,
                                 method=method or ("POST" if data else "GET"))
    return urllib.request.urlopen(req, timeout=timeout)


def ollama_json(path, payload=None, timeout=30):
    with ollama_request(path, payload, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def model_info(name):
    """Capabilities for a model, cached (vision / thinking / tools)."""
    now = time.time()
    with _lock:
        c = _show_cache.get(name)
        if c and now - c["ts"] < 600:
            return c
    try:
        info = ollama_json("/api/show", {"model": name}, timeout=20)
        entry = {
            "ts": now,
            "capabilities": info.get("capabilities", []),
            "parameter_size": (info.get("details") or {}).get("parameter_size", ""),
            "quantization": (info.get("details") or {}).get("quantization_level", ""),
            "family": (info.get("details") or {}).get("family", ""),
            "context_length": next((v for k, v in (info.get("model_info") or {}).items()
                                    if k.endswith(".context_length")), None),
        }
    except Exception as e:  # noqa
        entry = {"ts": now, "capabilities": [], "parameter_size": "", "quantization": "",
                 "family": "", "context_length": None, "error": str(e)}
    with _lock:
        _show_cache[name] = entry
    return entry


def settings_get():
    s = dict(DEFAULT_SETTINGS)
    s.update(read_json(SETTINGS_FILE, {}))
    return s


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Maya/1.0"

    def log_message(self, fmt, *args):
        if "/api/chat" in (args[0] if args else ""):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ----- responses
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return self.send_json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def send_media(self, path, ctype):
        """Static file with HTTP Range support (Safari needs it for <video>)."""
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return self.send_json({"error": "not found"}, 404)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        partial = False
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            try:
                if a:
                    start = int(a)
                    end = int(b) if b else size - 1
                elif b:
                    start = max(0, size - int(b))
                partial = True
            except ValueError:
                start, end, partial = 0, size - 1, False
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ----- routing
    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
        if p == "/api/health":
            return self.api_health()
        if p == "/api/models":
            return self.api_models()
        if p == "/api/settings":
            return self.send_json(settings_get())
        if p == "/api/defaults":
            return self.send_json({"system_prompt": DEFAULT_SYSTEM_PROMPT})
        if p == "/api/video/status":
            return self.send_json(videogen.status())
        m = re.match(r"^/api/video/([A-Za-z0-9_-]+)$", p)
        if m:
            j = videogen.get(m.group(1))
            if not j:
                return self.send_json({"error": "no such job"}, 404)
            j.pop("cmd", None)
            j["log"] = j.get("log", [])[-3:]
            return self.send_json(j)
        m = re.match(r"^/videos/([A-Za-z0-9_-]+\.mp4)$", p)
        if m:
            return self.send_media(VIDEOS / m.group(1), "video/mp4")
        m = re.match(r"^/images/([A-Za-z0-9_-]+\.(png|jpg|jpeg|webp))$", p)
        if m:
            ext = m.group(2).lower()
            return self.send_file(IMAGES / m.group(1), {"png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg"))
        if p == "/api/chats":
            return self.api_chats_list()
        m = re.match(r"^/api/chats/([A-Za-z0-9_-]+)$", p)
        if m:
            cid = m.group(1)
            chat = read_json(CHATS / f"{cid}.json", None)
            return self.send_json(chat) if chat else self.send_json({"error": "not found"}, 404)
        if p.startswith("/static/"):
            rel = p[len("/static/"):]
            if ".." in rel:
                return self.send_json({"error": "bad path"}, 400)
            ext = rel.rsplit(".", 1)[-1].lower()
            ctype = {"css": "text/css", "js": "application/javascript", "svg": "image/svg+xml",
                     "png": "image/png", "ico": "image/x-icon", "woff2": "font/woff2"}.get(ext, "application/octet-stream")
            return self.send_file(STATIC / rel, ctype)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        if p == "/api/chat":
            return self.api_chat()
        if p == "/api/image":
            return self.api_image()
        if p == "/api/video":
            return self.api_video()
        m = re.match(r"^/api/video/([A-Za-z0-9_-]+)/cancel$", p)
        if m:
            return self.send_json({"ok": videogen.cancel(m.group(1))})
        if p == "/api/chats":
            return self.api_chat_save(None)
        if p == "/api/settings":
            return self.api_settings_save()
        if p == "/api/unload":
            body = self.read_body()
            try:
                ollama_json("/api/generate", {"model": body.get("model", ""), "keep_alive": 0}, timeout=20)
            except Exception as e:  # noqa
                return self.send_json({"ok": False, "error": str(e)})
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)

    def do_PUT(self):
        p = self.path.split("?", 1)[0]
        m = re.match(r"^/api/chats/([A-Za-z0-9_-]+)$", p)
        if m:
            return self.api_chat_save(m.group(1))
        self.send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        p = self.path.split("?", 1)[0]
        m = re.match(r"^/api/chats/([A-Za-z0-9_-]+)$", p)
        if m:
            try:
                (CHATS / f"{m.group(1)}.json").unlink()
            except FileNotFoundError:
                pass
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)

    # ----- api
    def api_health(self):
        try:
            v = ollama_json("/api/version", timeout=5)
            try:
                ps = ollama_json("/api/ps", timeout=5).get("models", [])
            except Exception:  # noqa
                ps = []
            return self.send_json({"ok": True, "ollama": v.get("version"),
                                   "loaded": [{"name": m.get("name"), "size_vram": m.get("size_vram"),
                                               "size": m.get("size")} for m in ps],
                                   "video": videogen.status(), "image": imagegen.status()})
        except Exception as e:  # noqa
            return self.send_json({"ok": False, "error": str(e)}, 200)

    def api_models(self):
        """Every model installed in Ollama, regardless of which project pulled it."""
        try:
            tags = ollama_json("/api/tags", timeout=20).get("models", [])
        except Exception as e:  # noqa
            return self.send_json({"error": f"Ollama unreachable: {e}", "models": []}, 200)
        out = []
        threads = []
        results = {}

        def fetch(n):
            results[n] = model_info(n)

        for m in tags:
            t = threading.Thread(target=fetch, args=(m["name"],), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=30)
        for m in tags:
            info = results.get(m["name"], {})
            caps = info.get("capabilities", [])
            det = m.get("details") or {}
            out.append({
                "name": m["name"],
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at"),
                "family": det.get("family") or info.get("family", ""),
                "parameter_size": det.get("parameter_size") or info.get("parameter_size", ""),
                "quantization": det.get("quantization_level") or info.get("quantization", ""),
                "context_length": info.get("context_length"),
                "vision": "vision" in caps,
                "thinking": "thinking" in caps,
                "tools": "tools" in caps,
                "embedding": "embedding" in caps,
                "image": "image" in caps,
                "capabilities": caps,
            })
        # chat models first, then image generators, then embeddings; big to small
        out.sort(key=lambda x: (x["embedding"], x["image"], -x["size"]))
        return self.send_json({"models": out})

    def api_chats_list(self):
        items = []
        for f in CHATS.glob("*.json"):
            c = read_json(f, None)
            if not c:
                continue
            items.append({"id": c.get("id", f.stem), "title": c.get("title", "Untitled"),
                          "updated": c.get("updated", 0), "model": c.get("model", ""),
                          "count": len(c.get("messages", []))})
        items.sort(key=lambda x: -x["updated"])
        return self.send_json({"chats": items})

    def api_chat_save(self, cid):
        chat = self.read_body()
        if not isinstance(chat, dict):
            return self.send_json({"error": "bad body"}, 400)
        cid = cid or chat.get("id") or uuid.uuid4().hex[:12]
        if not SAFE_ID.match(cid):
            return self.send_json({"error": "bad id"}, 400)
        chat["id"] = cid
        chat["updated"] = time.time()
        chat.setdefault("created", chat["updated"])
        chat.setdefault("title", "New chat")
        chat.setdefault("messages", [])
        write_json(CHATS / f"{cid}.json", chat)
        return self.send_json({"ok": True, "id": cid, "updated": chat["updated"]})

    def api_settings_save(self):
        body = self.read_body()
        s = settings_get()
        for k in DEFAULT_SETTINGS:
            if k in body:
                s[k] = body[k]
        write_json(SETTINGS_FILE, s)
        return self.send_json(s)

    def api_chat(self):
        """Stream a chat completion from Ollama to the browser as NDJSON."""
        body = self.read_body()
        s = settings_get()
        model = body.get("model") or s.get("model")
        if not model:
            return self.send_json({"error": "no model selected"}, 400)
        messages = body.get("messages") or []
        system = body.get("system")
        if system is None:
            system = s.get("system_prompt", "")
        system = (system or "").replace("{name}", s.get("friend_name", "Maya"))
        if body.get("image_tool", True) and imagegen.status()["ready"]:
            system = system.rstrip() + "\n" + IMAGE_TOOL_PROMPT
        if s.get("video_enabled", True) and body.get("video_tool", True) and videogen.CLI.exists() and videogen.model_downloaded(s.get("video_model") or videogen.DEFAULT_MODEL):
            system = system.rstrip() + "\n" + VIDEO_TOOL_PROMPT
        if system.strip():
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": s.get("keep_alive", "30m"),
            "options": {
                "temperature": float(body.get("temperature", s.get("temperature", 0.7))),
                "num_ctx": int(body.get("num_ctx", s.get("num_ctx", 8192))),
            },
        }
        caps = model_info(model).get("capabilities", [])
        if "thinking" in caps:
            payload["think"] = bool(body.get("think", s.get("think", False)))

        try:
            upstream = ollama_request("/api/chat", payload, timeout=600)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
            except Exception:  # noqa
                msg = str(e)
            return self.send_json({"error": msg}, 502)
        except Exception as e:  # noqa
            return self.send_json({"error": f"Cannot reach Ollama at {OLLAMA}: {e}"}, 502)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            with upstream:
                for line in upstream:
                    if not line.strip():
                        continue
                    self.wfile.write(line if line.endswith(b"\n") else line + b"\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Browser hit "stop" — closing `upstream` cancels generation in Ollama.
            pass
        except Exception as e:  # noqa
            try:
                self.wfile.write(json.dumps({"error": str(e), "done": True}).encode() + b"\n")
            except Exception:  # noqa
                pass
        finally:
            self.close_connection = True


    def api_image(self):
        """Generate an image locally with mflux (FLUX.2 Klein 4B, 4-bit); stream step progress as NDJSON."""
        body = self.read_body()
        s = settings_get()
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return self.send_json({"error": "empty prompt"}, 400)
        st = imagegen.status()
        if not st["installed"]:
            return self.send_json({"error": "Image engine not installed. Run: uv tool install mflux"}, 400)
        if not st["weights_downloaded"]:
            return self.send_json({"error": f"Image model still downloading ({st['weights_bytes_on_disk']/1e9:.1f} GB of ~4.6 GB)."}, 400)
        for k in ("width", "height", "steps", "seed"):
            if body.get(k) not in (None, ""):
                s["image_" + k] = body[k]

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        dead = {"v": False}

        def emit(obj):
            try:
                self.wfile.write(json.dumps(obj).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                dead["v"] = True

        emit({"completed": 0, "total": 0, "stage": "queued"})
        res = imagegen.generate(prompt, s,
                                on_progress=lambda c, t, stage: emit({"completed": c, "total": t, "stage": stage}),
                                cancel_check=lambda: dead["v"])
        res["done"] = True
        emit(res)
        self.close_connection = True

    def api_video(self):
        body = self.read_body()
        s = settings_get()
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return self.send_json({"error": "empty prompt"}, 400)
        st = videogen.status()
        if not st["installed"]:
            return self.send_json({"error": "Video engine not installed. See README → Video."}, 400)
        if not st["model_downloaded"]:
            gb = st["model_bytes_on_disk"] / 1e9
            return self.send_json({"error": f"Video model still downloading ({gb:.1f} GB so far). Run ./tools/ltx/download.sh to resume."}, 400)
        for k in ("width", "height", "seconds", "seed"):
            if body.get(k) not in (None, ""):
                s["video_" + k] = body[k]

        def unload():
            try:
                for m in ollama_json("/api/ps", timeout=5).get("models", []):
                    ollama_json("/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=30)
            except Exception:  # noqa
                pass
        job = videogen.submit(prompt, s, unload_ollama=unload)
        job.pop("cmd", None)
        return self.send_json(job)


def main():
    CHATS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    videogen.load_jobs()
    if not SETTINGS_FILE.exists():
        write_json(SETTINGS_FILE, DEFAULT_SETTINGS)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"Maya is listening on http://{HOST}:{PORT}  (Ollama: {OLLAMA})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
