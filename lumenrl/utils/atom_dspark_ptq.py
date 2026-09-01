"""CLI for producing ATOM-compatible Kimi-K3 DSpark quantized checkpoints."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from lumenrl.quantization.atom_dspark_ptq import (
    PTQ_FORMATS,
    PTQ_PROFILES,
    convert_dspark_checkpoint,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the offline DSpark PTQ command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Quantize selected Kimi-K3 DSpark draft weights offline. ATOM "
            "dynamically quantizes activations to the matching format at inference."
        )
    )
    parser.add_argument("--source", required=True, help="BF16 draft checkpoint directory")
    parser.add_argument("--output", required=True, help="Quantized checkpoint directory")
    parser.add_argument(
        "--profile",
        choices=sorted(PTQ_PROFILES),
        default="phase1",
        help="Layer selection profile (phase2 is available for validation only)",
    )
    parser.add_argument(
        "--quant-format",
        choices=PTQ_FORMATS,
        default="ptpc_fp8",
        help="Weight/runtime activation format: FP8 PTPC or MXFP4 A4W4",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Quantization device; auto selects cuda and CPU is unsupported",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report selected tensors without writing output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Kimi-K3 DSpark offline weight PTQ CLI."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = convert_dspark_checkpoint(
        args.source,
        args.output,
        profile_name=args.profile,
        quant_format=args.quant_format,
        device=args.device,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    logger.info(
        "%s profile %s as %s: selected %d weights%s",
        "Validated" if args.dry_run else "Converted",
        manifest["profile"],
        manifest["quant_format"],
        manifest["selected_weight_count"],
        " (dry run)" if args.dry_run else f"; output={manifest['output']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

