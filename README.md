# Maya — your offline friend

A private, fully local AI companion for when the internet is gone. Chat, show it photos, ask it to draw, and lean on it for survival knowledge. Everything runs on this Mac through [Ollama](https://ollama.com); nothing leaves the machine.

## Start

```bash
./start.sh        # starts Ollama if needed, starts the UI, opens http://127.0.0.1:8080
./stop.sh         # stops the UI server (Ollama keeps running as a background service)
```

Requirements already on this machine: Ollama (Homebrew service, auto-starts on login) and Python 3. The UI server uses only the Python standard library, so it needs no `pip install` and no network, ever.

## Models

Every model installed in Ollama shows up in the model picker, whatever project pulled it. Badges tell you what each one can do: **vision** (sees photos), **thinking** (reasons before answering), **image gen** (draws).

| Model | Size | What it is for |
|---|---|---|
| `gemma4:12b` | 7.7 GB | Default. Vision + text, 256K context. Fast on 24 GB. |
| `qwen3.8:27b` | 18 GB | Smarter, slower. Vision + thinking. Tight fit on 24 GB; expect swapping with long contexts. |
| `qwen2.5:14b`, `qwen2.5:7b` | 9 / 4.7 GB | Older text-only models you already had. Still useful and quick. |

Add more any time:

```bash
ollama pull gemma4:e4b          # tiny, fast fallback (vision + audio)
ollama pull x/z-image-turbo     # higher-quality image model, 13 GB
ollama list                     # see what you have
```

Then hit **↻ refresh** in the model menu.

## What the UI does

- **Chat with streaming**, stop button, regenerate, edit-and-resend, copy.
- **Photos**: attach, drag-drop or paste an image. Vision models describe, identify, read labels and maps.
- **Image generation** from any model, rendered locally by FLUX.2 Klein 4B (4-bit) through [mflux](https://github.com/filipstrand/mflux) on MLX. About 4 steps, well under a minute per image. Three ways:
  - Ask naturally: "draw me a shelter design". The chat model writes a detailed prompt and hands it to the image model.
  - Type `/imagine <prompt>` to go straight to the image model.
  - Toggle the ✦ button and type a prompt.
  Generated images are saved to `data/images/`.
- **Video generation** from any model (see below).
- **Thinking mode** for models that support it (💡 button, or default in Settings).
- **Persona**: rename your friend and edit the system prompt in Settings. The default is a warm companion with practical survival knowledge.
- **Conversations** are saved as JSON in `data/chats/`, searchable, exportable to Markdown.
- Light and dark theme. Keyboard: `⌘K` new chat, `⌘,` settings, `Esc` stop.

## Image generation setup

Ollama's own image generation was experimental and current releases (0.40) reject image models, so Maya renders images with mflux instead. Done once:

```bash
uv tool install mflux
tools/ltx/ltx-2-mlx/.venv/bin/hf download ar9av/FLUX.2-klein-4B-mflux-4bit   # ~4.6 GB
```

Settings → Image generation shows readiness. Steps default to 4 (the model is distilled; more steps don't help much).

## Ollama engine note

The Homebrew `ollama` formula is not used any more. The official Ollama.app build was copied to `~/.ollama/engine/` and runs from a login LaunchAgent (`~/Library/LaunchAgents/com.maya.ollama.plist`), because the binary inside the app bundle hangs at launch on this macOS version due to a protected provenance attribute. `ollama` on the command line points at that copy. To upgrade later: update Ollama.app, re-copy `Contents/Resources` to `~/.ollama/engine`, run `xattr -rc ~/.ollama/engine`, then `launchctl kickstart -k gui/$(id -u)/com.maya.ollama`.

## Where things live

```
server.py          zero-dependency HTTP server + Ollama proxy (port 8080)
imagegen.py        image renders (FLUX.2 Klein via mflux), progress streaming
videogen.py        video render jobs (LTX-2 via MLX), progress, cancel
static/index.html  the whole UI, one file, no CDN
tools/ltx/         video engine checkout + download script
data/settings.json persona, model choices, image size
data/chats/        your conversations
data/images/       generated images
data/videos/       generated clips + job records
```

Environment: `MAYA_PORT`, `MAYA_HOST`, `OLLAMA_HOST`.

## Memory notes (24 GB Mac)

Ollama keeps a model in memory for the time set in Settings and swaps models as needed. gemma4:12b plus flux2-klein fit together. qwen3.8:27b alone uses most of the machine; set the context window to 8K or lower with it, and use "Free memory now" in Settings if things get sluggish.

## Video generation

Short clips (2 to 8 seconds, with sound) rendered locally with **LTX-2.3** through [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), a pure MLX port for Apple Silicon. Three ways to use it:

- Ask Maya naturally: "make a short video of waves at night". The chat model writes a shot description and hands it to the video engine.
- Type `/video <prompt>` to go straight to the engine.
- "Again" on any finished clip re-renders it.

Clips land in `data/videos/`. Rendering takes minutes, not seconds, and Maya unloads chat models from memory first because the renderer needs most of the 24 GB. You can keep chatting while it renders; the card updates itself.

Measured on the M4 / 24 GB machine: a 2 s clip at 512×320 with sound takes about 2 minutes end to end (half-res pass, full-res refine pass, decode). Longer or larger clips scale roughly with pixels × frames, so the default 4 s at 768×512 is in the 8–12 minute range. The progress bar reflects the real pipeline stages.

**Setup** (done once; everything after that is offline):

```bash
brew install ffmpeg
cd tools/ltx && git clone --depth 1 https://github.com/dgrauet/ltx-2-mlx.git
cd ltx-2-mlx && uv sync --all-packages --all-extras --python 3.11
./tools/ltx/download.sh      # ~28 GB: LTX-2.3 int4 distilled weights + Gemma 3 text encoder
```

Settings → Video lets you change resolution, length, and turn on **Low-RAM mode** (streams the model from disk; slower but safer). The sidebar shows whether the engine and weights are ready.

Why this stack: on a 24 GB Mac, LTX-2.x distilled via MLX is the only route that is both feasible and reasonably fast (minutes of compute per second of video). Wan 2.2 has no stable Metal path; one Mac test took 82 minutes for a 2 second clip ([benchmark](https://lilting.ch/en/articles/ltx2-wan22-mac-local-video-gen)). Alternatives considered: [rapid-mlx](https://rapidmlx.com/docs/models/families/video) (OpenAI-style server, but its recommended LTX-2.5 pack is a 68 GB download) and [ltx-video-mac](https://github.com/james-see/ltx-video-mac) (GUI app, states 32 GB minimum).

## Gotchas on this Mac

- **iCloud Desktop & Documents sync is on**, and this folder lives in `~/Documents`. Two effects: the video engine checkout (`tools/ltx`, several GB) and everything in `data/` get uploaded to iCloud, and iCloud's file provider marks dot-files as macOS-hidden. Python 3.11 ignores hidden `.pth` files, which silently broke the video venv once. The venv is now installed with `uv sync --all-packages --all-extras --no-editable` so it does not rely on `.pth` files. If you want to stop the uploads, move the project outside `~/Documents` (for example `~/maya`) and run `./start.sh` from there.
- Chrome keeps idle connections to port 8080, so never kill "whatever has port 8080 open"; `stop.sh` only targets the listener.

## Later: offline knowledge base

Models remember a lot but not everything. Pairing Maya with an offline copy of Wikipedia via [Kiwix](https://kiwix.org) (the full English ZIM is about 100 GB; the medical-only one is a few GB) covers the gaps. There is 660 GB free on this disk.
