from tests.integration import run_dsv4_weight_sync_integrity as integrity


def test_summarize_replica_reports_locates_first_bad_worker() -> None:
    reports = [
        [
            {"local_rank": 0, "all_finite": True, "first_bad": None},
            {
                "local_rank": 1,
                "all_finite": False,
                "first_bad": {"name": "layers.2.weight_scale_inv"},
            },
        ]
    ]

    summary = integrity.summarize_replica_reports(reports)

    assert summary["all_finite"] is False
    assert summary["first_bad"]["replica_rank"] == 0
    assert summary["first_bad"]["worker_rank"] == 1
    assert (
        summary["first_bad"]["report"]["first_bad"]["name"]
        == "layers.2.weight_scale_inv"
    )
