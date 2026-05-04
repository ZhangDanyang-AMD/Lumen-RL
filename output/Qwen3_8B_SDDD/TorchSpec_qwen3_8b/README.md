# Qwen3-8B Eagle3 SDDD on MI350 (TorchSpec + vLLM)

Speculative Decoding Draft Distillation (SDDD) of Qwen3-8B using TorchSpec with vLLM inference backend on AMD MI350 GPUs (ROCm, gfx950).

## Architecture

```
Dataset Prompts
    │
    ▼
vLLM Engine (GPUs 0-1, TP=2)
    │  extract_hidden_states mode
    │  MooncakeHiddenStatesConnector
    ▼
Mooncake TCP Transfer
    │
    ▼
Eagle3 Draft Model Training (GPUs 2-3, FSDP2, BF16)
    │  890M trainable params + 622M frozen embedding
    │  Forward KL loss with position decay
    ▼
Checkpoint
```

## Software Stack

| Component | Version / Commit |
|-----------|-----------------|
| Base image | `vllm/vllm-openai-rocm:v0.19.1` |
| vLLM | 0.19.1 + TorchSpec patches |
| Mooncake | `f2853a80` (source-built, `-DUSE_HIP=ON -DUSE_ETCD=ON`) |
| ROCm arch | gfx950 (MI350) |
| PyTorch | ROCm build from base image |
| TorchSpec | editable install from `/home/danyzhan/TorchSpec` |

## Key Mooncake Build Notes

Mooncake must be built from source with HIP support for ROCm. The pip package ships CUDA-linked `.so` files that segfault on AMD GPUs.

Build strategy in the Dockerfile:
1. Install pip package first (provides Python scaffolding + `mooncake_master` binary)
2. Build from source with `cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON`
3. Replace pip's `.so` files with source-built HIP versions
4. Critical: cmake output is in `mooncake-integration/` (not `mooncake-store/` or `mooncake-transfer-engine/`)
5. Critical: `export PATH="/usr/local/go/bin:$PATH"` before cmake — `dependencies.sh` installs Go 1.23.8 but apt's Go 1.18 would be found first otherwise, breaking ETCD build

## Quick Start

### 1. Build Docker Image

```bash
bash build_docker_mi350.sh
# or
docker build -f /home/danyzhan/TorchSpec/docker/vllm/rocm_mi350/Dockerfile \
    -t torchspec-vllm-mi350:latest /home/danyzhan/TorchSpec
```

### 2. Run Smoke Test (5 steps)

```bash
bash run_vllm_mi350.sh
```

### 3. Run Full Training (500 steps)

```bash
bash run_vllm_mi350.sh --steps 500
```

### 4. Direct Docker Run

```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --shm-size=64g \
    --security-opt seccomp=unconfined \
    -v /dev/shm/Qwen3-8B:/dev/shm/Qwen3-8B:ro \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e TORCHSPEC_LOG_LEVEL=INFO \
    -e WANDB_MODE=disabled \
    torchspec-vllm-mi350:latest \
    python3 -m torchspec.train_entry \
        --config configs/vllm_qwen3_8b.yaml \
        model.target_model_path=/dev/shm/Qwen3-8B \
        training.num_train_steps=5 \
        training.training_num_gpus_per_node=2 \
        inference.inference_num_gpus=2 \
        inference.inference_num_gpus_per_engine=2 \
        inference.inference_num_gpus_per_node=4 \
        inference.vllm.tp_size=2 \
        mooncake.protocol=tcp
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | GPU allocation (do NOT set `HIP_VISIBLE_DEVICES`) |
| `PYTORCH_ROCM_ARCH` | `gfx950` | MI350 GPU architecture |
| `HIP_FORCE_DEV_KERNARG` | `1` | Required for MI350 kernel args |
| `WANDB_MODE` | `disabled` | Disable W&B logging for smoke tests |

## Smoke Test Results (2026-05-01)

5-step smoke test on 4x MI350 GPUs:

| Step | Loss | Accuracy | Acc Len |
|------|------|----------|---------|
| 1 | 12.179 | 0.000 | 0.00 |
| 2 | 10.539 | 0.042 | 0.08 |
| 3 | 9.753 | 0.041 | 0.08 |
| 4 | 8.548 | 0.052 | 0.15 |
| 5 | 8.768 | 0.060 | 0.09 |

Eval at step 5: loss=9.5926, acc=0.0513, sim_acc_len=0.10

Total time: ~28 seconds for 5 training steps (excluding initialization).

## Files

| File | Description |
|------|-------------|
| `run_vllm_mi350.sh` | Run training in Docker (configurable steps) |
| `build_docker_mi350.sh` | Build the Docker image |
| `run_rdma.log` | MI300 reference run log (SGLang, 500 steps, RDMA) |
| `run_rdma_MI350.log` | MI350 vLLM smoke test log (5 steps, TCP) |
| `README.md` | This file |

## Troubleshooting

- **`libcudart.so.12: cannot open shared object file`**: Mooncake `.so` replacement failed. Check that the Dockerfile's `find` uses `*/mooncake-integration/*` path filter and `store*.so` glob pattern.
- **`batch_remove() not found`**: Mooncake commit too old. Must use `f2853a80` or later.
- **`HIP error: invalid device ordinal`**: Do not set `HIP_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES`. Use only `CUDA_VISIBLE_DEVICES`.
- **ETCD build fails (`invalid go version '1.23.0'`)**: Need `export PATH="/usr/local/go/bin:$PATH"` before cmake so Go 1.23.8 from `dependencies.sh` takes precedence over apt's Go 1.18.
