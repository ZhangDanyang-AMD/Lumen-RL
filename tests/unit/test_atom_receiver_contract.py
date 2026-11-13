"""The RDMA sender against ATOM's receiver of its stream.

ATOM (ROCm/ATOM, atom/rollout/rdma_weight_receiver.py) decodes what
lumenrl/engine/inference/rdma_weight_transfer.py sends. Both are read as text:
ATOM is a separate repo, and importing it pulls in GPU-only code. Set ATOM_ROOT
to an ATOM checkout to run; .github/workflows/atom-receiver-contract.yaml does,
on a change to the sender and daily, since ATOM can move without LumenRL
changing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SENDER = _REPO / "lumenrl/engine/inference/rdma_weight_transfer.py"
_RECEIVER = Path("atom/rollout/rdma_weight_receiver.py")
_DECODER = Path("atom/rollout/weight_updater.py")
_HEADER = ("command", "metadata_bytes", "payload_bytes", "version")


def _atom(relative: Path) -> str:
    root = os.environ.get("ATOM_ROOT")
    if not root:
        pytest.skip("set ATOM_ROOT to an ATOM checkout to check the receiver")
    path = Path(root, relative)
    # Named but missing fails rather than skips.
    assert path.is_file(), f"ATOM_ROOT={root} has no {relative}"
    return path.read_text()


def _constant(src: str, name: str) -> int | None:
    m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.MULTILINE)
    return int(m[1]) if m else None


def _function(src: str, name: str) -> str:
    # Up to the next top-level statement; a signature's closing ")" is not one.
    m = re.search(
        rf"^def {name}\(.*?(?=^(?:def |class |@|[A-Za-z_])|\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"no top-level {name}()"
    return m[0]


def test_the_commands_and_header_width_agree():
    sender, receiver = _SENDER.read_text(), _atom(_RECEIVER)
    for const in ("_CMD_END", "_CMD_BUCKET", "_HEADER_WORDS"):
        sent, read = _constant(sender, const), _constant(receiver, const)
        assert sent is not None and sent == read, f"{const}: sends {sent}, reads {read}"


def test_the_header_is_read_in_the_order_it_is_sent():
    """Swapped sizes make the two sides post broadcasts of different lengths,
    which RCCL does not check."""
    packs = _function(_SENDER.read_text(), "_broadcast_header")
    unpacks = _function(_atom(_RECEIVER), "_recv_header")
    sent = re.search(r"\[\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*,?\s*\]", packs)
    read = re.search(r"(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*\(", unpacks)
    assert sent and sent.groups() == _HEADER
    assert read and read.groups() == _HEADER


def test_each_tensor_is_described_by_what_the_decoder_reads():
    entry = re.search(
        r"entries\.append\(\s*\{(.*?)\}\s*\)", _SENDER.read_text(), re.DOTALL
    )
    assert entry, "the sender builds no entries"
    decoder = _function(_atom(_DECODER), "decode_bucket")
    sent = set(re.findall(r'"(\w+)":', entry[1]))
    read = set(re.findall(r'fields\.get\("(\w+)"\)', decoder))
    assert sent == read == {"dtype", "name", "nbytes", "offset", "shape"}
