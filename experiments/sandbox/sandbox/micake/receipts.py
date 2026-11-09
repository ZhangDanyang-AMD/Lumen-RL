"""Map MICake worker results to backend-neutral ReceiptRef handles."""

from __future__ import annotations

from typing import Any, Mapping

from lumenrl.agentic.core.protocols import ReceiptRef

PROVIDER = "micake"


def receipts_for(result: Mapping[str, Any]) -> list[ReceiptRef]:
    """Receipt handles for the evidence a worker result carries."""
    stage = result.get("stage")
    refs: list[ReceiptRef] = []
    digest = result.get("candidate_digest")
    if digest:
        refs.append(ReceiptRef(PROVIDER, "c144_candidate", digest, digest, stage))
    source = result.get("source_sha256")
    if source:
        refs.append(ReceiptRef(PROVIDER, "c144_generated_source", source, source, stage))
    if "correctness" in result:
        refs.append(ReceiptRef(PROVIDER, "c144_gpu_correctness", digest or "", None, result["correctness"].get("status")))
    if "performance" in result:
        refs.append(ReceiptRef(PROVIDER, "c144_paired_measurement", digest or "", None, result["performance"].get("validity")))
    return refs
