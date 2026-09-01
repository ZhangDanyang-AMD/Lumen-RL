"""CLI for exporting the ATOM-native Kimi-K3 DSpark FP8 checkpoint."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from lumenrl.quantization.atom_dspark_native import (
    export_atom_native_dspark_checkpoint,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ATOM-native DSpark export CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge the validated Phase 1 FP8 PTPC checkpoint into ATOM runtime "
            "projection names while preserving gfx950-native E4M3FN weights."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Portable Phase 1 FP8 PTPC checkpoint directory",
    )
    parser.add_argument("--output", required=True, help="Native output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the native contract without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ATOM-native DSpark export."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = export_atom_native_dspark_checkpoint(
        args.source,
        args.output,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    logger.info(
        "%s ATOM-native %s checkpoint%s",
        "Validated" if args.dry_run else "Exported",
        manifest["source_profile"],
        " (dry run)" if args.dry_run else f"; output={manifest['output']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
