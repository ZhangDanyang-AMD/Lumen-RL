"""HuggingFace <-> Megatron checkpoint converter.

Usage:
    python examples/converters/convert_checkpoint.py \\
        --source hf --target megatron \\
        --input-path /path/to/hf/model \\
        --output-path /path/to/megatron/ckpt \\
        --tp-size 4 --ep-size 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lumenrl.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def convert_hf_to_megatron(
    input_path: str,
    output_path: str,
    tp_size: int = 1,
    ep_size: int = 1,
) -> None:
    """Convert HuggingFace checkpoint to Megatron format."""
    logger.info(
        "Converting HF -> Megatron: %s -> %s (TP=%d, EP=%d)",
        input_path, output_path, tp_size, ep_size,
    )
    # TODO: implement full HF -> Megatron conversion with TP/EP resharding
    logger.info("Conversion complete (placeholder)")


def convert_megatron_to_hf(
    input_path: str,
    output_path: str,
) -> None:
    """Convert Megatron checkpoint to HuggingFace format."""
    logger.info("Converting Megatron -> HF: %s -> %s", input_path, output_path)
    # TODO: implement full Megatron -> HF conversion
    logger.info("Conversion complete (placeholder)")


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Checkpoint converter")
    parser.add_argument("--source", choices=["hf", "megatron"], required=True)
    parser.add_argument("--target", choices=["hf", "megatron"], required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    args = parser.parse_args()

    if args.source == "hf" and args.target == "megatron":
        convert_hf_to_megatron(args.input_path, args.output_path, args.tp_size, args.ep_size)
    elif args.source == "megatron" and args.target == "hf":
        convert_megatron_to_hf(args.input_path, args.output_path)
    else:
        logger.error("Conversion from %s to %s not supported", args.source, args.target)


if __name__ == "__main__":
    main()
