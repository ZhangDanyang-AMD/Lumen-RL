"""Stub for miles' tilelang FP8 block activation-quantization kernel.

Only reachable from ``ops.qat.fp8_simulate``, i.e. FP8 QAT. With FP8 off (the
starting configuration) it is never called, and it carries no parameters.
"""


def act_quant(*args, **kwargs):
    raise NotImplementedError(
        'act_quant requires tilelang, which is intentionally absent; run with FP8 QAT off'
    )
