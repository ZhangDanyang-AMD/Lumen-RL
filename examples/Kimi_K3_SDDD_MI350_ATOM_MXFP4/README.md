# Kimi-K3 DSpark MXFP4 Draft Training on MI350

Trains the [Kimi-K3 DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)
speculative-decoding draft from scratch against a frozen Kimi-K3 teacher, with
the draft's MLP and MLA projections running in MXFP4. The result is an MXFP4
draft that ATOM serves directly.

- **Topology:** five MI350 nodes. Four each run one ATOM TP=8 teacher; the
  fifth runs the 8-rank draft trainer.
- **Data flow:** the training data already holds Kimi-K3's answers, so the
  teachers only prefill and extract hidden states. Hidden states reach the
  draft over Mooncake TCP; Ray carries only tokens, masks and keys.
- **Precision:** draft master weights stay BF16/FP32. Lumen quantizes the
  selected Linear operands to MXFP4 for the forward and backward GEMMs;
  gradients are all-reduced in BF16 over RCCL.
- **Formal run:** 3 epochs, 57,849 steps, global batch 128, 32,768-token
  window.

```
teacher0 TP=8 ─┐
teacher1 TP=8 ─┼─ Mooncake TCP ──> draft node, ranks 0..7
teacher2 TP=8 ─┤                   (each rank fetches 16 of 128 sequences)
teacher3 TP=8 ─┘
```

## Requirements

- Five MI350 nodes with eight gfx950 GPUs each, held for the whole run, with
  no other GPU processes and NUMA auto-balancing off
  (`/proc/sys/kernel/numa_balancing` = 0).
- On every node: Docker, `python3`, `rocm-smi`, `/dev/infiniband`, the Ionic
  RDMA userspace provider, and unlimited memlock for containers.
- Node-local storage (`DATA_ROOT`) for Kimi-K3 (1.56 TB) and the token cache.
  `selfcheck/host_preflight.sh` rejects a model outside `DATA_ROOT`.
- A shared filesystem mounted at the same path on every node for both
  repositories, the training data, checkpoints and logs. Containers mount only
  `DATA_ROOT` and `SHARED_ROOT`, so every input must live under one of them.
- Access to the gated
  [training data](https://huggingface.co/datasets/ZhuoBohang-AMD/Kimi-K3-Draft-mxfp4-training-data):
  request it on the dataset page, then run `hf auth login`
  (`pip install -U huggingface_hub`).

## 1. Get the code

Clone both repositories side by side on the shared filesystem:

```bash
git clone -b k3-dspark-from-scratch-mxfp4 https://github.com/ZhuoBohang-AMD/Lumen-RL.git
git clone -b k3-dspark-from-scratch-mxfp4 https://github.com/ZhuoBohang-AMD/Lumen.git
git -C Lumen submodule update --init --recursive
cd Lumen-RL
```

Run every later command from the Lumen-RL root, in the same shell.

## 2. Configure

```bash
export K3_NODES='node1 node2 node3 node4 node5'     # onnode IDs; first is the Ray head, last is the draft
export K3_NODELIST='node1,node2,node3,node4,node5'  # the same nodes, for the srun commands below
export ONNODE_ENV=/path/to/onnode.sh
export DATA_ROOT=/path/to/local_nvme                # node-local
export SHARED_ROOT=/path/to/shared/lumenrl          # shared
export MODEL="$DATA_ROOT/models/Kimi-K3"
export DATASET_PATH="$SHARED_ROOT/datasets/Kimi-K3-Draft-mxfp4-training-data/data/train.jsonl"
export TOKEN_CACHE_DIR="$DATA_ROOT/cache/lumenrl"
export EVAL_FIXED_PATH="$SHARED_ROOT/eval/kimi_k3_dspark_mxfp4_fixed_128.pt"
export CKPT_DIR="$SHARED_ROOT/checkpoints/kimi_k3_dspark_mxfp4"
export TRAIN_IMAGE=kimi_k3_dspark_mxfp4_atom:latest
```

`ONNODE_ENV` is a shell file that defines `onnode <node> <command>`: run the
command string with bash on that node and return its exit status. The launcher
starts each node's process with `setsid nohup ... &`, so that process must keep
running after `onnode` returns. `ssh` does this; a SLURM job step does not,
because SLURM kills a step's processes when the step ends. For example:

```bash
onnode() { ssh -o BatchMode=yes "$1" "$2"; }
```

Set `K3_DRAFT_NODE` to put the draft on a node other than the last one. The
`srun` commands below are helpers for one-off jobs; adapt them to your
scheduler.

## 3. Build and distribute the image

```bash
bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/docker/build_mxfp4_atom.sh
mkdir -p "$SHARED_ROOT/images"
docker save "$TRAIN_IMAGE" -o "$SHARED_ROOT/images/kimi_k3_dspark_mxfp4_atom.tar"
srun --partition=default --nodes=5 --ntasks=5 --nodelist="$K3_NODELIST" \
  docker load -i "$SHARED_ROOT/images/kimi_k3_dspark_mxfp4_atom.tar"
docker run --rm "$TRAIN_IMAGE" \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/selfcheck/preflight.py
```

The build needs no GPU. It looks for Lumen at `../Lumen` (`LUMEN_ROOT`
overrides) and refuses any Lumen or AITER commit other than the ones pinned in
the script. Docker images are node-local: build once and load the same archive
everywhere, so all five nodes run an identical image; the launcher checks this.
`preflight.py` takes seconds and catches ATOM API drift and config mistakes
before a 20-minute teacher load.

## 4. Download the model and data

Kimi-K3 goes onto every node at the revision `host_preflight.sh` checks. The
image provides `hf`:

```bash
srun --partition=default --nodes=5 --ntasks=5 --nodelist="$K3_NODELIST" \
  docker run --rm -v "$DATA_ROOT:$DATA_ROOT" "$TRAIN_IMAGE" \
  hf download moonshotai/Kimi-K3 \
    --revision a590ce090cb049c93a33dfe8c208ec652aa20503 --local-dir "$MODEL"
```

Copying a finished download between nodes also works if its `.cache/` directory
comes along; the revision check reads it.

Download the training data once to shared storage:

```bash
hf download ZhuoBohang-AMD/Kimi-K3-Draft-mxfp4-training-data --repo-type dataset \
  --local-dir "$SHARED_ROOT/datasets/Kimi-K3-Draft-mxfp4-training-data"
```

Other data works in the same format: JSONL with one `conversations` (or
`messages`) list of `role`/`content` turns per row. Only the final assistant
turn is supervised, so it should be Kimi-K3's own answer; rows without the list
are skipped.

## 5. Build the token cache and eval slice

Only the draft reads the token cache, so build it on the draft node. Neither
step needs a GPU.

```bash
DRAFT_HOST=node5   # the draft node
srun --partition=default --nodes=1 --ntasks=1 --nodelist="$DRAFT_HOST" \
  docker run --rm -v "$DATA_ROOT:$DATA_ROOT" -v "$SHARED_ROOT:$SHARED_ROOT" \
  -e HF_DATASETS_CACHE="$DATA_ROOT/cache/hf_datasets" "$TRAIN_IMAGE" \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/selfcheck/preprocess_dataset.py \
    --dataset "$DATASET_PATH" --tokenizer "$MODEL" \
    --max-length 32768 --max-prompt-tokens 0 --cache-dir "$TOKEN_CACHE_DIR"

srun --partition=default --nodes=1 --ntasks=1 --nodelist="$DRAFT_HOST" \
  docker run --rm -v "$DATA_ROOT:$DATA_ROOT" -v "$SHARED_ROOT:$SHARED_ROOT" \
  -e HF_DATASETS_CACHE="$DATA_ROOT/cache/hf_datasets" "$TRAIN_IMAGE" \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/selfcheck/build_eval_slice.py \
    --dataset "$DATASET_PATH" --tokenizer "$MODEL" \
    --max-length 32768 --max-prompt-tokens 0 --num-samples 128 \
    --cache-dir "$TOKEN_CACHE_DIR" --out "$EVAL_FIXED_PATH"
```

`preprocess_dataset.py` must report 2,468,457 rows and 57,849 steps, the
`num_training_steps` in the config. With other data, pass the printed step
count at launch as `EXTRA_OVERRIDES="num_training_steps=<N>"`; it keeps
training off the 128 rows held out for eval. `build_eval_slice.py` writes those
128 rows to `EVAL_FIXED_PATH`.

## 6. Train

```bash
bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/scripts/launch_multinode.sh
```

The launcher first runs `selfcheck/host_preflight.sh` on every node (GPUs,
model revision, data, image, NUMA). It then starts each node's container,
prints the run ID and log directory, and returns once all five Ray nodes have
joined; training keeps running on the nodes. Loading the teachers takes about
20 minutes.

**Monitor.** The draft log `node-00-<draft-host>.log` has the step, loss and
eval lines; `rank-<r>-<node>.out` has each node's startup. Both are in
`$SHARED_ROOT/logs/run/kimi_k3_dspark_mxfp4_atom_<RUN_ID>/`. Teacher worker
logs stay on their hosts in `/dev/shm/lumenrl_teacher_hidden/`. A checkpoint
lands in `CKPT_DIR` every 500 steps (the last 10 are kept), and eval runs every
1,500 steps.

**Stop.** Remove the run's container on every node. Rank 0 then records the
exit and the other nodes follow:

```bash
source "$ONNODE_ENV"
RUN_ID=<run ID printed by the launcher>
for n in $K3_NODES; do onnode "$n" "docker rm -f kimi-k3-dspark-mxfp4-$RUN_ID"; done
```

**Resume.** Relaunch with the same `CKPT_DIR` and any overrides of the original
launch, plus `checkpointing.resume=true`. The trainer loads the highest-step
`checkpoint_*.pt` and restores the model, optimizer, FP32 master weights,
scheduler and RNG state; keep `num_training_steps` unchanged.

```bash
EXTRA_OVERRIDES="checkpointing.resume=true" \
  bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/scripts/launch_multinode.sh
```

## 7. Export the draft

Convert the final checkpoint to ATOM's MXFP4 1x32 format. The exporter
quantizes with Lumen on a GPU and reads `$MODEL`, so run it on one of the five
nodes once training has finished. `MXFP4_EXPORT_TMPDIR` keeps the intermediate
BF16 export on local disk:

```bash
EXPORT_HOST=node5
srun --partition=default --nodes=1 --ntasks=1 --nodelist="$EXPORT_HOST" \
  docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  -v "$DATA_ROOT:$DATA_ROOT" -v "$SHARED_ROOT:$SHARED_ROOT" \
  -e MXFP4_EXPORT_TMPDIR="$DATA_ROOT/tmp/mxfp4-export" "$TRAIN_IMAGE" \
  python3 examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/scripts/export_mxfp4.py \
    --ckpt "$CKPT_DIR/checkpoint_57849.pt" --base-model "$MODEL" \
    --train-config examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/configs/train_from_scratch.yaml \
    --output "$SHARED_ROOT/drafts/dspark-mxfp4"
```

The output directory is the draft ATOM loads; `mxfp4_manifest.json` in it
lists every quantized weight.

## 8. Benchmark acceptance length

Bring your own question sets. Each set is `$BENCH_DIR_ROOT/data/<name>.jsonl`,
one object per line with an `id` and a `turns` list; only the first turn is
sent, as a single user message:

```json
{"id": "gsm8k-0", "turns": ["Janet's ducks lay 16 eggs per day. ..."]}
```

Run on one of the five nodes once its GPUs are free; it needs `$MODEL` and all
eight GPUs. It serves
Kimi-K3 with the draft in the training image, sends every prompt with greedy
decoding, and restarts the server and resumes if it stalls:

```bash
BENCH_HOST=node5
srun --partition=default --nodes=1 --ntasks=1 --nodelist="$BENCH_HOST" \
  env DRAFT="$SHARED_ROOT/drafts/dspark-mxfp4" LABEL=mxfp4-epoch3 SETS="gsm8k math500" \
    BENCH_DIR_ROOT="$SHARED_ROOT/bench" TARGET_MODEL="$MODEL" \
  bash examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/scripts/run_benchmark.sh

python3 examples/Kimi_K3_SDDD_MI350_ATOM_MXFP4/benchmark/summarize_runs.py \
  --results "$SHARED_ROOT/bench/results" --label mxfp4-epoch3
```

Results land in `$BENCH_DIR_ROOT/results/bench_<LABEL>-<set>.json`, with the
server log beside them. The summary prints tok/fwd (1 + accepted draft tokens
per target forward step) for each set and their mean. The defaults are a
16,384-token window, a 10,240-token batched-token budget and a 12,288-token
output cap. Chunked prefill is off, so every prompt must fit in
`MAX_NUM_BATCHED_TOKENS`; for longer prompts raise it together with
`MAX_MODEL_LEN`, for example both to 32768 with `MAX_TOKENS=8192`.

## Settings to keep

- **The YAML is baked into the image.** Override one launch with
  `EXTRA_OVERRIDES`, space-separated `key=value` pairs; editing
  `configs/train_from_scratch.yaml` takes effect only after rebuilding and
  redistributing the image.
- **`kv_cache_dtype` and `index_cache_dtype` are `fp8`**, as in ATOM serving, so
  the draft trains on the hidden states it will see when served.
- **Batch 128, lr `5e-5` and max grad norm 1.0** are the TorchSpec reference
  optimizer settings; change them together.
- **`max_num_batched_tokens >= max_model_len`, with chunked prefill and prefix
  caching off.** ATOM rewrites the same Mooncake key on every scheduler step,
  so a split prefill silently keeps only its last chunk. Preflight enforces
  this.
- **`separate_last_hidden: false`.** ATOM's `hidden_states` already holds all
  five aux layers (5 × 7168); `true` would feed `fc` 6 × 7168 and apply RMSNorm
  twice.
- **Mooncake runs over TCP, with a 256 GB segment per teacher**, set by
  `run_multinode_rank.sh`. Ionic allows one RDMA client per HCA per process,
  which cannot hold a 32,768-token batch.
- **The MXFP4 weight selection** is listed in full in the config. The exporter
  quantizes its include list, and the exporter's `ATOM_EXCLUDE` (the layers
  ATOM loads as BF16) must be updated with any change to the selection.

## Layout

```
scripts/launch_multinode.sh   # training entry: preflight, run ID, one rank per node
run_multinode_rank.sh         # per-node container and Ray startup
ray_multinode_launcher.py     # Ray driver on rank 0: teacher actors + draft torchrun
configs/train_from_scratch.yaml
scripts/export_mxfp4.py       # training checkpoint -> ATOM MXFP4 draft
scripts/run_benchmark.sh      # benchmark driver with restart and resume
benchmark/run_client.py       # acceptance-length client for one ATOM server
benchmark/summarize_runs.py   # per-set table and mean tok/fwd for one label
docker/                       # AITER -> Lumen -> Lumen-RL image build; torchspec/
                              # maps ATOM's torchspec imports to lumenrl.transfer
selfcheck/preflight.py        # ATOM API and config checks, no GPU
selfcheck/host_preflight.sh   # per-node GPU, model, data and image checks
selfcheck/preprocess_dataset.py, build_eval_slice.py   # token cache and eval slice
selfcheck/run_gpu_correctness_checks.sh                # RoPE/YaRN and Flex Attention on GPU
```

## Reference

- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — draft model config
- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — target model
- [DSpark paper (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
- [ZhuoBohang-AMD/Kimi-K3-Draft-mxfp4-training-data](https://huggingface.co/datasets/ZhuoBohang-AMD/Kimi-K3-Draft-mxfp4-training-data) — training data
