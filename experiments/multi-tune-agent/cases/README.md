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

Supported case types are `gemm`, `fused_attention`, and `grouped_gemm`.
`kernel_path` is resolved relative to the catalog file.

Run every entry sequentially:

```bash
PYTHONPATH=src:. python3 -m multi_tune_agent.cli \
  --config configs/mi300x.yaml run --stream
```

The interactive CLI can generate an FP16 dense-GEMM task from a request that
contains `M`, `N`, and `K`. It writes the local task under `examples/tasks/` and
registers the task and permanent request in this catalog. Other operator
families still require an existing trustworthy GEAK harness.
