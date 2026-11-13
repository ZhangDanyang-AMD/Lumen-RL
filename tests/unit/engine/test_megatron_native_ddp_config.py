"""The DistributedDataParallelConfig the native Megatron engine builds."""

from __future__ import annotations

import pytest

from lumenrl.engine.training.megatron_native_engine import ddp_config_kwargs


@pytest.mark.parametrize(("ec", "fp32"), [({}, True), ({"grad_reduce_in_fp32": True}, True),
                                          ({"grad_reduce_in_fp32": False}, False)])
def test_grad_reduce_precision_follows_the_config(ec, fp32):
    assert ddp_config_kwargs(ec)["grad_reduce_in_fp32"] is fp32


def test_param_gather_overlap_turns_on_grad_reduce_overlap():
    kwargs = ddp_config_kwargs({"overlap_param_gather": True, "ddp_bucket_size": "40000000"})

    assert kwargs["overlap_grad_reduce"] and kwargs["overlap_param_gather"]
    assert kwargs["bucket_size"] == 40000000
