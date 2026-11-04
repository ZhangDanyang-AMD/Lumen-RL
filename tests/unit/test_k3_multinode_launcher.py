from __future__ import annotations

import pytest

from examples.Kimi_K3_SDDD_MI350_ATOM_MXFP4.ray_multinode_launcher import (
    _partition_nodes,
)


def test_partition_nodes_pins_explicit_draft() -> None:
    nodes = [
        {"NodeManagerAddress": "10.0.0.40"},
        {"NodeManagerAddress": "10.0.0.39"},
        {"NodeManagerAddress": "10.0.0.41"},
    ]

    teachers, draft = _partition_nodes(nodes, "10.0.0.39")

    assert draft["NodeManagerAddress"] == "10.0.0.39"
    assert [node["NodeManagerAddress"] for node in teachers] == [
        "10.0.0.40",
        "10.0.0.41",
    ]


def test_partition_nodes_rejects_unknown_draft() -> None:
    with pytest.raises(RuntimeError, match="available"):
        _partition_nodes(
            [{"NodeManagerAddress": "10.0.0.40"}],
            "10.0.0.39",
        )
