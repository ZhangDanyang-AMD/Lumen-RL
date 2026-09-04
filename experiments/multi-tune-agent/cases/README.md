# Case catalogs

This committed directory stores stable batch manifests and default optimization
objectives. Kernel source, generated harnesses, build output, and private task
data belong under `../examples/tasks/`, which is intentionally ignored by Git.

Each catalog has this shape:

```yaml
tasks:
  - id: unique-kernel-id
    type: gemm
    kernel_path: ../examples/tasks/unique-kernel-id
    direction: >
      Permanent optimization objective and constraints.
```

Supported case types include `gemm`, `scaled_quant_gemm`, `quant_fp4_mxfp`,
`fused_attention`, `grouped_gemm`, and `aiter_generated`.
`kernel_path` is resolved relative to the catalog file.

## Batch generation request example

`generation_requests.yaml` describes kernels that should be generated and
validated before they are added to `examples_cases.yaml`:

```yaml
version: 1
requests:
  - id: mi308-gemm-m1-n128-k128-fp8
    request: >
      在 MI308 上生成 FP8 A8W8 GEMM kernel，MNK 分别为 1/128/128，
      activation 使用 per-token FP32 scale，weight 使用
      per-output-channel FP32 scale，FP16 输出，使用 Triton。

  - id: mi350-gemm-m1-n128-k128-mxfp8
    request: >
      在 MI350 上生成原生 MXFP8 GEMM kernel，MNK 分别为 1/128/128，
      activation 和 weight 使用 per-block scale，block size 为 32，
      BF16 输出，使用 FlyDSL。
```

Each request should state the target GPU, operator, complete dimensions,
format/dtypes, quantization scale contract, output dtype, and implementation
language. Model generation and static validation may run concurrently, while
GPU compile/correctness/performance remains serialized per GPU. Only templates
that pass the complete trust gate may be registered in `examples_cases.yaml`.

The manifest format is present for batch preparation. The planned
`multi-tune generate --manifest ...` command is not implemented yet; use
interactive menu option `[2]` to generate one request at a time.

Run every entry sequentially:

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml run --stream
```

The interactive CLI can generate an FP16 dense-GEMM task from a request that
contains `M`, `N`, and `K`. It writes the local task under `examples/tasks/` and
registers the task and permanent request in this catalog. Other operator
families still require an existing trustworthy GEAK harness.
