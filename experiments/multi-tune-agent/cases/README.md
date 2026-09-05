# Case catalogs

This committed directory stores stable batch manifests and default optimization
objectives. Kernel source, generated harnesses, build output, and private task
data belong under `../examples/tasks/`, which is intentionally ignored by Git.

## Phase 1 lifecycle

Phase 1 uses four separate artifact tiers. Do not point `cases_path` at the
first three tiers:

```text
phase1-case-candidates.yaml
  candidates: extracted cross-language contracts; not runnable
        ↓ source/hash/lineage review
phase1-aiter-seeds.yaml
  seeds: registered extraction sources; not runnable
        ↓ frozen generation request
phase1-generation-requests.yaml
  requests: batch generation input; not runnable
        ↓ static + GPU trust gate
phase1-cases.yaml
  tasks: validated GEAK harnesses; runnable by load_tasks()
```

Cross-language reuse is allowed for shapes, dtypes, layouts, scale contracts,
and independent oracles. `source_language` and `target_language` must remain
separate. Mechanical translations share one `implementation_family_id`, and
all language variants from the same source/implementation family must remain
in one split group.

Candidate status values are:

```text
extracted_candidate -> registered_seed -> validated_case
```

Only `validated_case` entries belong in a `tasks:` catalog.

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

Batch generation is available:

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml generate \
  --manifest cases/phase1-generation-requests.yaml \
  --output-catalog cases/phase1-cases.yaml \
  --stream
```

The command registers only templates that pass the complete static and GPU
trust gate. Preserve candidate/seed provenance in each request's
`seed_provenance` mapping.

Run every entry sequentially:

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml run --stream
```

The interactive CLI can generate an FP16 dense-GEMM task from a request that
contains `M`, `N`, and `K`. It writes the local task under `examples/tasks/` and
registers the task and permanent request in this catalog. Other operator
families still require an existing trustworthy GEAK harness.
