# Qwen3-8B Eagle3 Speculative Decoding Draft Distillation

Disaggregated architecture matching TorchSpec's `qwen3-8b-single-node` example:

- **2 inference GPUs**: SGLang Engine + ATOM plugin (Aiter-optimized kernels)
- **2 training GPUs**: Eagle3 draft model (FSDP2 + Aiter CK kernels)
- **Mooncake RDMA**: Hidden state transfer between inference and training

## Data Flow

```
Dataset sequences
  -> SGLang+ATOM Forward (GPU 2-3, no grad, spec_training mode)
  -> hidden_states stored to Mooncake RDMA
  -> Training GPUs (0-1) fetch from Mooncake
  -> Eagle3 Draft Model (with grad)
  -> Position-weighted CE loss (0.8^i)
  -> Backward -> Update Draft
```

## Requirements

- 4x AMD MI300X (2 inference + 2 training)
- ROCm 6.4+, PyTorch 2.6+
- SGLang with TorchSpec spec_training patches (`patches/sglang/`)
- ATOM (`/home/danyzhan/ATOM_repo` or set `ATOM_REPO`)
- `mooncake-transfer-engine >= 0.3.10.post1` (build from source for RDMA)
- RDMA NIC (e.g., `mlx5_8` for RoCE)

## Quick Start

Smoke test (2 steps, HF backend, no SGLang/ATOM/Mooncake):

```bash
bash examples/Qwen3_8B_SDDD/run_opd.sh --smoke-test
```

Full training (500 steps, SGLang + ATOM + Mooncake RDMA):

```bash
bash examples/Qwen3_8B_SDDD/run_opd.sh
```

## Architecture Comparison

| Component | TorchSpec | Lumen-RL (this example) |
|-----------|-----------|------------------------|
| Inference engine | SGLang (Ray actor) | SGLang + ATOM plugin (subprocess) |
| Model kernels | SGLang native | ATOM Aiter kernels |
| Transfer layer | Mooncake (TCP/RDMA) | Mooncake (RDMA) |
| Training framework | FSDP | FSDP2 + Lumen/Aiter |
| Draft model | Eagle3 (custom) | Eagle3 (Lumen-RL) |
| Attention kernels | FlexAttention | Aiter CK kernels |
| GPU split | 2 inference + 2 training | 2 inference + 2 training |
| Orchestration | Ray + AsyncController | torchrun + SpecDistillTrainer |
