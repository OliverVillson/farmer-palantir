#!/usr/bin/env bash
# One-time setup of the robot pipeline on the evroc B200 VM (Ubuntu 24.04).
#   NVME=/mnt/nvme HF_TOKEN=hf_... bash scripts/setup_robot_vm.sh
# Standalone: mounts the local NVMe if needed and installs uv. Safe to re-run.
# Local NVMe is wiped if the VM is stopped.
#
# What it does, following the Isaac-GR00T README (dGPU x86_64 install):
#   0. Mounts the blank local NVMe at $NVME (only a disk with no filesystem or
#      partitions is formatted).
#   1. apt: ffmpeg (torchcodec, the only GR00T video backend; FFmpeg 4-7 only),
#      libegl1/libgl1/libglvnd0 (headless MuJoCo on EGL), libosmesa6 (CPU
#      fallback), git-lfs.
#   2. Isaac-GR00T at $FARMPAL_GR00T_REPO: `uv sync --python 3.12` builds its
#      .venv (torch 2.9 cu128, flash-attn 2.8.3, transformers 4.57.3,
#      torchcodec 0.8). Submodules (LIBERO, SimplerEnv, robocasa) are only for
#      NVIDIA's sim benchmarks and are not needed.
#   3. The farm sim's packages into that same venv, constrained to the versions
#      uv sync installed (so numpy stays 1.26.4 and torch is never touched),
#      and this repo installed editable (--no-deps). compress and simeval run
#      sim and policy in one process with this python (FARMPAL_GR00T_PY);
#      robot/finetune.py launches launch_finetune.py with it too (not `uv run`,
#      which would re-sync the venv).
#   3b. A venv of its own, $NVME/venv-farmpal, with this repo installed
#      editable with [sim,api]: the API server and the site, demos and report
#      stages run there.
#   4. Checks: GPU, torchcodec decode, flash-attn kernel on this GPU, MuJoCo EGL.
#   5. Weights: nvidia/GR00T-N1.7-3B to $FARMPAL_MODELS and the backbone into
#      $HF_HOME.
#
# HF_TOKEN: the GR00T backbone nvidia/Cosmos-Reason2-2B is GATED on Hugging
# Face, and every GR00T checkpoint (base or fine-tuned) loads it by hub id at
# load time (launch_finetune.py hard-codes model_name="nvidia/Cosmos-Reason2-2B").
# Request access on https://huggingface.co/nvidia/Cosmos-Reason2-2B with the
# account whose token you use, then `export HF_TOKEN=hf_...` before running this
# script (or log in when asked). The pipeline later needs the same HF_HOME
# ($NVME/hf-cache), which robot/finetune.py sets when it is unset; source
# $NVME/robot.env (written at the end) in every shell that runs the pipeline.
set -euo pipefail

NVME=${NVME:-/mnt/nvme}
# FARMPAL_* wins; LOBBOT_* (the repo this started in) is the fallback.
GR00T_REPO=${FARMPAL_GR00T_REPO:-${LOBBOT_GR00T_REPO:-$NVME/Isaac-GR00T}}
GR00T_MODEL=${FARMPAL_GR00T_MODEL:-${LOBBOT_GR00T_MODEL:-nvidia/GR00T-N1.7-3B}}
BACKBONE=nvidia/Cosmos-Reason2-2B
MODELS=${FARMPAL_MODELS:-${LOBBOT_MODELS:-$NVME/models}}
APP_VENV=$NVME/venv-farmpal
REPO=$(cd "$(dirname "$0")/.." && pwd)

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

# evroc's GPU image installs the NVIDIA driver on first boot and then reboots.
command -v cloud-init >/dev/null && sudo cloud-init status --wait >/dev/null || true

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

say "Local NVMe at $NVME"
if ! mountpoint -q "$NVME"; then
  # The local disk arrives blank. Only a disk with no filesystem or partitions is formatted.
  DISK=$(lsblk -dbno NAME,SIZE,TYPE | awk '$3=="disk" && $2>1e12 {print "/dev/"$1}' | while read -r d; do
    [ -z "$(sudo blkid -o value -s TYPE "$d")" ] && [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ] && echo "$d"; done | head -1)
  [ -n "$DISK" ] || { echo "No blank disk over 1 TB to mount at $NVME"; exit 1; }
  sudo mkfs.ext4 -q -L nvme -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 "$DISK"
  sudo mkdir -p "$NVME"
  grep -q "LABEL=nvme" /etc/fstab || echo "LABEL=nvme $NVME ext4 defaults,noatime,nofail 0 2" | sudo tee -a /etc/fstab
  sudo mount "$NVME"
fi
sudo chown "$(id -u):$(id -g)" "$NVME"
df -h "$NVME"

export HF_HOME="$NVME/hf-cache"
export UV_CACHE_DIR="$NVME/uv-cache"   # the root disk is too small for the torch wheels
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$MODELS" "$NVME/sites" "$NVME/jobs" "$HF_HOME"

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs curl python3-venv ffmpeg libegl1 libgl1 libglvnd0 libosmesa6 libaio-dev
git lfs install --skip-repo >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
FFMPEG_MAJOR=$(ffmpeg -version | head -1 | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')
if [ "${FFMPEG_MAJOR:-0}" -ge 8 ] 2>/dev/null; then
  warn "ffmpeg $FFMPEG_MAJOR found; torchcodec 0.8 needs FFmpeg 4-7 (Ubuntu 24.04 ships 6)"
fi
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  warn "no NVIDIA EGL vendor file; MuJoCo EGL needs the driver's GL libs (apt install libnvidia-gl-<driver>-server)"
fi

say "Isaac-GR00T at $GR00T_REPO"
if [ -d "$GR00T_REPO/.git" ]; then
  git -C "$GR00T_REPO" pull --ff-only || echo "(pull skipped)"
else
  git clone https://github.com/NVIDIA/Isaac-GR00T "$GR00T_REPO"
fi
(cd "$GR00T_REPO" && uv sync --python 3.12)
VENV_PY="$GR00T_REPO/.venv/bin/python"
"$VENV_PY" -c 'import gr00t, torch; print("gr00t ok, torch", torch.__version__)'

say "Farm sim packages in the GR00T venv (pinned to what uv sync installed)"
CONSTRAINTS=$(mktemp)
uv pip freeze --python "$VENV_PY" | grep -v -e '^-e' -e ' @ ' -e '^gr00t' > "$CONSTRAINTS" || true
uv pip install --python "$VENV_PY" -q -c "$CONSTRAINTS" mujoco imageio imageio-ffmpeg pillow pyarrow pandas \
  pyproj tifffile imagecodecs matplotlib pytest
rm -f "$CONSTRAINTS"
uv pip install --python "$VENV_PY" -q --no-deps -e "$REPO"
"$VENV_PY" -c 'import numpy, torch, farmsim.lerobot, robot.gr00t_policy; print("numpy", numpy.__version__, "torch", torch.__version__)'

say "App venv $APP_VENV (API, site, demos, report)"
uv venv -q --python 3.12 --allow-existing "$APP_VENV"
uv pip install --python "$APP_VENV/bin/python" -q -e "$REPO[sim,api]" pytest
"$APP_VENV/bin/python" -c 'import fastapi, mujoco, farmsim.site, api.server; print("app venv ok")'

say "torchcodec decode (GR00T's video loader)"
TMPV=$(mktemp -d)
ffmpeg -loglevel error -f lavfi -i testsrc=size=64x64:rate=10 -frames:v 5 -c:v libx264 -pix_fmt yuv420p "$TMPV/t.mp4"
"$VENV_PY" - "$TMPV/t.mp4" <<'EOF'
import sys, numpy as np
from gr00t.utils.video_utils import get_frames_by_indices
f = get_frames_by_indices(sys.argv[1], np.arange(5))
assert f.shape[0] == 5 and f.shape[-1] == 3, f.shape
print("torchcodec ok", f.shape)
EOF
rm -rf "$TMPV"

say "flash-attn kernel on this GPU"
if ! "$VENV_PY" - <<'EOF'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.bfloat16)
print("flash-attn ok", tuple(flash_attn_func(q, q, q).shape), torch.cuda.get_device_name(0))
EOF
then
  warn "flash-attn failed on this GPU. GR00T falls back to sdpa only when flash_attn cannot be imported:"
  warn "  uv pip uninstall --python $VENV_PY flash-attn"
fi

say "Headless MuJoCo render on EGL"
MUJOCO_GL=egl "$VENV_PY" - <<'EOF' || warn "EGL render failed; MUJOCO_GL=osmesa works (CPU, slower)"
import mujoco
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><light pos='0 0 3'/><geom type='box' size='1 1 .1' rgba='.2 .6 .2 1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
assert img.shape == (64, 64, 3) and img.max() > 0, "EGL render returned an empty image"
print("mujoco", mujoco.__version__, "EGL render ok, mean pixel", round(float(img.mean()), 1))
EOF

say "Hugging Face login (gated backbone $BACKBONE)"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "using HF_TOKEN from the environment"
elif ! (cd "$GR00T_REPO" && "$GR00T_REPO/.venv/bin/hf" auth whoami >/dev/null 2>&1); then
  "$GR00T_REPO/.venv/bin/hf" auth login || warn "not logged in; the backbone download will fail"
fi

say "Weights: $GR00T_MODEL and $BACKBONE"
"$GR00T_REPO/.venv/bin/hf" download "$GR00T_MODEL" --local-dir "$MODELS/${GR00T_MODEL##*/}"
# Into the HF_HOME cache, not a local dir: GR00T loads it by hub id.
"$GR00T_REPO/.venv/bin/hf" download "$BACKBONE" \
  || warn "$BACKBONE download failed: request access on its HF page, export HF_TOKEN, re-run"

cat > "$NVME/robot.env" <<EOF
export FARMPAL_GR00T_REPO=$GR00T_REPO
export FARMPAL_GR00T_PY=$VENV_PY
export FARMPAL_GR00T_MODEL=$GR00T_MODEL
export FARMPAL_MODELS=$MODELS
export FARMPAL_SITES=$NVME/sites
export FARMPAL_JOBS=$NVME/jobs
export HF_HOME=$HF_HOME
export UV_CACHE_DIR=$UV_CACHE_DIR
export MUJOCO_GL=egl
export PATH=$APP_VENV/bin:$GR00T_REPO/.venv/bin:\$PATH
# Secrets (HF_TOKEN, FARMPAL_TOKEN, LANTMATERIET_USER/PASSWORD) live outside the repo, mode 600.
if [ -f \$HOME/.farmpal-env ]; then . \$HOME/.farmpal-env; fi
EOF

say "GR00T compatibility tests (CPU, tiny random model)"
(cd "$REPO" && FARMPAL_GR00T_REPO="$GR00T_REPO" MUJOCO_GL=egl "$VENV_PY" -m pytest -q tests/test_gr00t_compat.py) \
  || warn "tests/test_gr00t_compat.py failed: fix before the real run"

say "Done"
cat <<EOF
source $NVME/robot.env
python robot_pipeline.py --job <dir>     # python = $APP_VENV/bin/python; GPU stages use $VENV_PY
FARMPAL_TOKEN=... bash scripts/serve_api.sh   # then ssh -L 8700:127.0.0.1:8700 and open http://localhost:8700/map
EOF
