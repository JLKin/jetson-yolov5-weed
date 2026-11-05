#!/usr/bin/env bash
# Explicit bootstrap for the user's L4T 35.5.0 / system Python 3.8 environment.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(uname -m)" != "aarch64" ]] || ! grep -q 'R35.*REVISION: 5.0' /etc/nv_tegra_release; then
  echo "This recipe targets L4T 35.5.0 only. See docs/jetson.md for other versions." >&2
  exit 1
fi
if [[ "$(/usr/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.8" ]]; then
  echo "The NVIDIA wheel requires system Python 3.8." >&2
  exit 1
fi
if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  echo "CUDA development tools are missing. Install the matching JetPack compute packages first." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  /usr/bin/python3 -m venv --system-site-packages .venv
fi
if .venv/bin/python -c 'import torch, torchvision; assert torch.cuda.is_available(); torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device="cuda"),torch.tensor([.9],device="cuda"),.5)' >/dev/null 2>&1; then
  echo "GPU PyTorch and torchvision NMS already work; keeping them."
  exit 0
fi
TORCH_WHEEL='https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl'
.venv/bin/python -m pip install 'numpy==1.23.5' 'setuptools>=70' wheel \
  typing_extensions sympy filelock networkx jinja2 'pillow>=10.3' 'requests>=2.32.2'
.venv/bin/python -m pip install --no-deps "$TORCH_WHEEL"
mkdir -p work/build
VISION_SOURCE="work/build/torchvision-0.15.1"
if [[ ! -d "$VISION_SOURCE" ]]; then
  git clone --branch v0.15.1 --depth 1 https://github.com/pytorch/vision.git "$VISION_SOURCE"
fi
if [[ "$(git -C "$VISION_SOURCE" describe --tags --exact-match)" != "v0.15.1" ]]; then
  echo "Existing torchvision source is not v0.15.1; inspect $VISION_SOURCE." >&2
  exit 1
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-2}"
# Orin (8.7); torchvision detects the actual GPU architecture through torch.
.venv/bin/python -m pip install --no-deps --no-build-isolation "$VISION_SOURCE"
.venv/bin/python -c 'import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__); print(torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device="cuda"),torch.tensor([.9],device="cuda"),.5))'
