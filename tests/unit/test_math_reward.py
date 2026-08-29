from lumenrl.rewards.math_reward import compute_score


def test_compute_score_accepts_last_boxed_answer_without_answer_prefix() -> None:
    result = compute_score(
        r"The absolute value is \(\frac{25}{8}\), so \(m+n=33\). \boxed{33}",
        "33",
    )

    assert result == {"score": 1.0, "acc": True, "pred": "33"}
