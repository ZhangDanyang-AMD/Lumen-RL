"""CLI for composing selective FP8/MXFP4 Kimi-K3 DSpark checkpoints."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from lumenrl.quantization.atom_dspark_ptq import (
    MIXED_PROJECTION_GROUPS,
    compose_mixed_dspark_checkpoint,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the selective DSpark mixed-quantization parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compose a selective FP8/MXFP4 DSpark checkpoint from existing "
            "Phase 1 and Phase 2 quantized tensors without re-quantization."
        )
    )
    parser.add_argument("--fp8-source", required=True)
    parser.add_argument("--mxfp4-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mxfp4-projection",
        action="append",
        choices=MIXED_PROJECTION_GROUPS,
        required=True,
        dest="mxfp4_projections",
        help="Projection group to source from MXFP4; repeat for multiple groups",
    )
    parser.add_argument(
        "--mxfp4-layer",
        action="append",
        type=int,
        dest="mxfp4_layers",
        help="Layer to source from MXFP4; omit to select all draft layers",
    )
    parser.add_argument(
        "--selection-name",
        default="selective",
        help="Name recorded in config.json and the composition manifest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compose one selective mixed-quantization checkpoint."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = compose_mixed_dspark_checkpoint(
        args.fp8_source,
        args.mxfp4_source,
        args.output,
        mxfp4_projections=args.mxfp4_projections,
        mxfp4_layers=args.mxfp4_layers,
        selection_name=args.selection_name,
        overwrite=args.overwrite,
    )
    logger.info(
        "Composed %s with %d MXFP4 and %d FP8 weights; output=%s",
        manifest["profile"],
        manifest["mxfp4_weight_count"],
        manifest["fp8_weight_count"],
        manifest["output"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
