import pytest

from examples.GRPO.dsv4.validate_weight_sync import compare_snapshots


def test_compare_snapshots_reports_token_and_logprob_differences() -> None:
    before = {
        "token_ids": [[1, 2, 3], [4, 5]],
        "logprobs": [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
    }
    after = {
        "token_ids": [[1, 2, 3], [4, 6]],
        "logprobs": [[-0.1, -0.25, -0.3], [-0.4, -0.7]],
    }

    summary = compare_snapshots(before, after)

    assert summary["exact_token_matches"] == 1
    assert summary["total"] == 2
    assert summary["first_token_mismatch"] == 1
    assert summary["matching_logprob_mae"] == pytest.approx(0.05 / 3)
