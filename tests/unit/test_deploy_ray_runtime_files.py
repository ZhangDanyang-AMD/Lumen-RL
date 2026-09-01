from tests.integration.deploy_ray_runtime_files import FILES


def test_rdma_protocol_is_deployed_with_runtime_files() -> None:
    assert "lumenrl/engine/inference/rdma_protocol.py" in FILES


def test_ray_worker_group_is_deployed_with_runtime_files() -> None:
    assert "lumenrl/controller/ray_worker_group.py" in FILES


def test_isolated_weight_integrity_runtime_is_deployed() -> None:
    assert "lumenrl/engine/inference/weight_integrity.py" in FILES
    assert "tests/integration/run_dsv4_weight_sync_integrity.py" in FILES
