# Installation

This page covers hardware and software requirements, container-based setup, editable installs for development, third-party stack components, and a short verification snippet.

## Requirements

| Component | Version / hardware |
| --- | --- |
| Python | >= 3.10 |
| PyTorch | >= 2.4 (ROCm build recommended) |
| ROCm | >= 6.2 |
| GPU | AMD Instinct MI250 or MI300 series |

```{note}
Match your PyTorch wheel to the ROCm version on the host or inside the container. Mismatched userspace stacks are the most common source of runtime failures on AMD clusters.
```

## Docker install (recommended)

The upstream README recommends the ATOM developer image, which bundles many ROCm inference dependencies.

```bash
docker pull rocm/atom-dev:latest

docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=16G \
  -v $HOME:/home/$USER \
  rocm/atom-dev:latest
```

Inside the container, clone LumenRL and install the package:

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git && cd Lumen-RL
pip install -e ".[all]"
```

The `[all]` extra pulls training and inference integrations (including Lumen and ATOM) needed for end-to-end recipes.

## Developer install

On a ROCm host with compatible PyTorch already available:

```bash
git clone https://github.com/ZhangDanyang-AMD/Lumen-RL.git && cd Lumen-RL

# Recommended for docs, examples, and FP8/MoE paths
pip install -e ".[all]"

# Minimal package only
pip install -e "."
```

If you iterate on Ray workers or configs, keep the editable install so local changes are picked up without reinstalling.

## Third-party libraries

| Library | PyPI / import | Role in LumenRL |
| --- | --- | --- |
| [Lumen](https://github.com/ZhangDanyang-AMD/Lumen) | `lumen` | Quantized training: hybrid FP8/MXFP8, AITER-backed kernels, MORI communication hooks |
| [ATOM](https://github.com/ROCm/ATOM) | `atom` | High-throughput inference: FP8 rollout, MoE expert parallel, speculative decoding |
| [AITER](https://github.com/ROCm/aiter) | `amd-aiter` | Low-level GPU kernels (attention, GEMM, MoE, norm) |
| [MORI](https://github.com/ROCm/mori) | `mori` | RDMA-aware GPU collectives and MoE expert dispatch |
| [Ray](https://github.com/ray-project/ray) | `ray` | Controller/worker orchestration and resource placement |

## Verify installation

Run a lightweight import check and print key versions:

```bash
python - <<'PY'
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda/hip available:", torch.cuda.is_available())

import lumenrl  # noqa: F401
from lumenrl.core.config import LumenRLConfig

cfg = LumenRLConfig()
print("LumenRLConfig OK:", type(cfg).__name__)
PY
```

If `torch.cuda.is_available()` is false on a GPU node, reinstall a ROCm-enabled PyTorch build before debugging LumenRL itself.

Next: {doc}`/quickstart/quick_start` for a minimal GRPO launch and links into the deeper guides.
