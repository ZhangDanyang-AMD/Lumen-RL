"""Print what each torchrun rank thinks the device count is, before anything else.

Used to isolate why probe_rs_coalesced.py sees torch.cuda.device_count() == 0
under torchrun on primus while a single process in the same shell sees 8.
"""

import os

import torch

print(
    f"rank {os.environ.get('RANK')} local {os.environ.get('LOCAL_RANK')} "
    f"count {torch.cuda.device_count()} avail {torch.cuda.is_available()} "
    f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"HIP={os.environ.get('HIP_VISIBLE_DEVICES')} "
    f"ROCR={os.environ.get('ROCR_VISIBLE_DEVICES')}",
    flush=True,
)
