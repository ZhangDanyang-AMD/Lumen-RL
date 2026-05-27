# LumenRL Registry + Strong Config Assembly Design

## Context

LumenRL currently has clear worker roles and backend choices, but object construction is still mostly imperative and spread across workers/trainers. The project goal is to keep the current runtime behavior and backend choices unchanged while introducing two framework-level capabilities:

1. Strong configuration-driven assembly with dataclass-based config and automatic object wiring.
2. A standard extension path based on abstract base classes (ABCs), registry bindings, and config keys.

Out of scope for this phase:

- Changing algorithm behavior.
- Replacing training backends (still only FSDP or Megatron).
- Replacing inference backend (still only ATOM).
- Large runtime behavior changes in workers, trainer loops, or scheduling.

## Goals and Non-Goals

### Goals

- Add typed assembly inputs and backend policy validation at startup.
- Route worker and backend construction through registries and a single assembler.
- Introduce role-specific worker contracts so new worker implementations are pluggable.
- Include trainer construction in the new assembly path.
- Keep existing functional behavior stable during migration.

### Non-Goals

- Introduce new backends.
- Remove old pathways in one shot.
- Rework data protocol format (`DataProto`) or training math.

## Target Architecture

The runtime will be split into four layers.

### 1) Config Layer

Typed dataclass config is the source of truth for runtime composition. Proposed logical groups:

- `RoleConfig` family: actor, critic, rollout, reward, teacher, ref.
- `BackendConfig` family:
  - training backend (`fsdp` or `megatron`)
  - inference backend (`atom`)
- `RuntimeAssemblyConfig`:
  - topology (`hybrid` or `decoupled`)
  - role enablement and implementation keys
  - feature flags for migration stages

### 2) Contract Layer (ABCs)

Role-specific contracts define stable behavior boundaries:

- `ActorWorkerABC`
- `CriticWorkerABC`
- `RolloutWorkerABC`
- `RewardWorkerABC`
- `TeacherWorkerABC`

Backend contracts define runtime capabilities:

- `TrainingBackendABC`
- `InferenceBackendABC`

### 3) Registry Layer

Three primary registries:

- worker role registry (`role_key -> worker class`)
- training backend registry (`backend_key -> impl`)
- inference backend registry (`backend_key -> impl`)

Optional:

- trainer registry (`trainer_key -> trainer class`)

### 4) Assembly Layer

A single `RuntimeAssembler` constructs runtime objects from config + registries, validates policy constraints, and returns a runtime graph consumed by trainers and controller components.

## Policy Constraints (Hard Requirements)

The assembler must enforce:

- `training_backend in {"fsdp", "megatron"}`
- `inference_backend == "atom"`

Violations fail at startup with clear error messages.

## Proposed Module Layout

Proposed new package subtree:

- `lumenrl/architecture/config/`
  - `worker_config.py`
  - `backend_config.py`
  - `assembly_config.py`
- `lumenrl/architecture/abc/`
  - `worker_roles.py`
  - `backends.py`
- `lumenrl/architecture/registry/`
  - `worker_registry.py`
  - `training_backend_registry.py`
  - `inference_backend_registry.py`
  - `trainer_registry.py` (optional in phase 1)
- `lumenrl/architecture/assembly/`
  - `runtime_assembler.py`
  - `default_bindings.py`
  - `policy_validator.py`

## Integration Plan (Strangler Pattern / Option B)

### Phase B1: Add assembly shell with behavior parity

- Implement config adapters and assembler scaffolding.
- Register existing concrete implementations as defaults.
- Keep existing direct construction path as fallback.

### Phase B2: Switch core trainers to assembler

- Route `RLTrainer` and `AsyncRLTrainer` through `RuntimeAssembler`.
- Keep old code path behind feature flag for rollback.

### Phase B3: Switch advanced trainers

- Route `OPDTrainer` and `SpecDistillTrainer` through assembler.
- Keep interfaces behavior-compatible.

### Phase B4: Consolidate and remove legacy constructor paths

- Remove direct ad-hoc object construction once parity checks pass.
- Keep backward-compatible config parsing only where needed.

## Worker and Backend Mapping (Phase 1 Defaults)

Initial bindings keep behavior unchanged:

- Actor role -> `LumenActorWorker`
- Rollout role -> `AtomRolloutWorker`
- Critic role -> `CriticWorker`
- Reward role -> `RewardWorker`
- Teacher role -> `TeacherWorker`
- Ref role -> `RefPolicyWorker`
- Hybrid topology -> `HybridWorker`
- Training backend `fsdp` -> existing `FSDP2Backend` wrapper path
- Training backend `megatron` -> existing `MegatronBackend` wrapper path
- Inference backend `atom` -> existing `AtomEngine` path

## Error Handling and Observability

### Error Handling

- Config errors: fail at parse/validation.
- Registry resolution errors: fail during assembly.
- Runtime init errors: include role, rank, backend, and config key in exception context.

### Observability

- Log a startup assembly manifest:
  - selected topology
  - role -> implementation key
  - backend -> implementation key
- Add timing/health metrics for:
  - worker init
  - rollout generation latency
  - weight sync latency and frequency

## Test and Acceptance Plan

### Unit

- policy validator constraints
- registry registration/lookup conflict checks
- assembler mapping and failure cases

### Integration

- trainer smoke runs through assembler path
- role composition in `hybrid` and `decoupled` topologies
- weight sync round-trip remains stable

### Behavior Parity

For fixed seed and same config:

- loss trend remains within expected tolerance
- rollout output schema unchanged
- reward/ref/teacher tensors and keys unchanged
- no added distributed hangs

## Risks and Mitigations

- Risk: hidden behavior drift from constructor reordering.
  - Mitigation: phase flags + parity tests before each cutover.
- Risk: registry import-order issues.
  - Mitigation: explicit `default_bindings` bootstrap in assembler init.
- Risk: config migration churn.
  - Mitigation: single typed adapter layer and deterministic validation errors.

## Implementation Readiness

This design is ready for implementation planning under a staged migration model (B1 -> B4), with no required backend changes and strict policy enforcement aligned with current product constraints.
