#!/bin/zsh
# Download everything the distilled LTX-2.3 text-to-video pipeline needs (~28 GB total):
#   - dgrauet/ltx-2.3-mlx-q4  (distilled transformer, connector, VAEs, upscaler, audio) ~20 GB
#   - mlx-community/gemma-3-12b-it-4bit (text encoder) ~8 GB
# Safe to re-run: it resumes.
set -e
cd "$(dirname "$0")/ltx-2-mlx"
MODEL="${1:-dgrauet/ltx-2.3-mlx-q4}"
.venv/bin/hf download mlx-community/gemma-3-12b-it-4bit
exec .venv/bin/hf download "$MODEL" \
  --include "transformer-distilled.safetensors" "connector.safetensors" \
            "vae_decoder.safetensors" "vae_encoder.safetensors" \
            "spatial_upscaler_x2_v1_1.safetensors" \
            "audio_vae.safetensors" "vocoder.safetensors" \
            "*.json" "LICENSE*" "README*"
