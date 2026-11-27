"""Replace decode-step logprobs with one full-sequence prefill.

Past 2048 keys, q_len=1 varlen and the packed prefill no longer match.
The scores that enter the loss are recomputed by the same prefill the
training forward uses. Enable with LUMENRL_PREFILL_RESCORE=1. The engine
child writes one float32 tensor per prefill; the server reads it back.
"""

import hashlib
import importlib.machinery
import os
import sys

import torch


_INPUT_IDS = None


def enabled():
    return os.environ.get("LUMENRL_PREFILL_RESCORE", "0") == "1" and bool(
        os.environ.get("LUMENRL_RESCORE_DIR", "")
    )


def sequence_key(ids):
    raw = ",".join(str(int(token)) for token in ids).encode()
    return hashlib.sha256(raw).hexdigest()


def rescore_path(directory, ids):
    return os.path.join(directory, sequence_key(ids) + ".pt")


def load_rescore(directory, ids):
    path = rescore_path(directory, ids)
    if not os.path.isfile(path):
        return None
    saved = torch.load(path, map_location="cpu")
    saved_ids = saved["ids"].to(torch.int64).view(-1).tolist()
    if saved_ids != [int(token) for token in ids]:
        return None
    return saved["logprobs"].to(torch.float32).view(-1)


def patch_runner(module):
    runner = module.ModelRunner
    orig = runner.run_model

    def run_model(self, input_ids, batch, *args, **kwargs):
        global _INPUT_IDS
        _INPUT_IDS = input_ids
        try:
            return orig(self, input_ids, batch, *args, **kwargs)
        finally:
            _INPUT_IDS = None

    runner.run_model = run_model


def patch_lm_head(module):
    head = module.ParallelLMHead
    orig = head.forward

    def forward(self, x):
        _record(self, x)
        return orig(self, x)

    head.forward = forward
    print("prefill rescore records full-sequence logprobs", flush=True)


def _record(self, x):
    if not enabled() or x.shape[0] <= 1 or _INPUT_IDS is None:
        return
    if getattr(self, "tp_rank", 0) not in (0, None):
        return
    from atom.utils.forward_context import get_forward_context

    fwd = get_forward_context()
    context = getattr(fwd, "context", None)
    meta = getattr(fwd, "attn_metadata", None)
    if context is None or not bool(getattr(context, "is_prefill", False)) or meta is None:
        return
    ids = _INPUT_IDS.detach().view(-1)
    if ids.shape[0] != x.shape[0]:
        return
    from aiter.dist.communication_op import tensor_model_parallel_all_gather
    from aiter.tuned_gemm import tgemm
    from atom.utils import envs

    logits = tgemm.mm(x, self.weight, self.bias)
    if getattr(self, "tp_size", 1) > 1:
        logits = tensor_model_parallel_all_gather(
            logits, use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER
        )
    cu = meta.cu_seqlens_q.detach().to("cpu").tolist()
    ids_cpu = ids.to("cpu", dtype=torch.int64)
    directory = os.environ["LUMENRL_RESCORE_DIR"]
    os.makedirs(directory, exist_ok=True)
    for start, end in zip(cu, cu[1:]):
        start = int(start)
        end = int(end)
        if end - start < 2:
            continue
        rows = logits[start : end - 1].float()
        targets = ids[start + 1 : end].to(device=rows.device, dtype=torch.long)
        chosen = torch.log_softmax(rows, dim=-1).gather(1, targets[:, None]).squeeze(1)
        seq_ids = ids_cpu[start:end].contiguous()
        path = rescore_path(directory, seq_ids.tolist())
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(
            {"ids": seq_ids, "logprobs": chosen.detach().to("cpu", dtype=torch.float32)},
            tmp,
        )
        os.replace(tmp, path)
        print(f"prefill rescore wrote {end - start} tokens {path}", flush=True)


def install():
    if os.environ.get("LUMENRL_PREFILL_RESCORE", "0") != "1":
        return
    patches = {
        "atom.model_engine.model_runner": patch_runner,
        "atom.model_ops.embed_head": patch_lm_head,
    }

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname not in patches:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module
            name = fullname

            def exec_module(module, _orig=orig_exec, _name=name):
                _orig(module)
                patches[_name](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
