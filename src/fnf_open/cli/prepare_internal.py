"""Command line helpers for preprocessing the internal dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ..preprocessing import build_internal_pickles

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _parse_size(values: Sequence[int], argument: str) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"{argument} expects two integers (width height)")
    return int(values[0]), int(values[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("excel", type=Path, help="Path to the labeling Excel file")
    parser.add_argument("raw", type=Path, help="Directory containing per-serial DICOM files")
    parser.add_argument("ap_xml", type=Path, help="Directory containing AP XML annotations")
    parser.add_argument("lat_xml", type=Path, help="Directory containing LAT XML annotations")
    parser.add_argument("output", type=Path, help="Directory where pickle files will be written")
    parser.add_argument(
        "--detection-size",
        type=int,
        nargs=2,
        default=(800, 800),
        metavar=("WIDTH", "HEIGHT"),
        help="Detection image size (default: 800 800)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("WIDTH", "HEIGHT"),
        help="Classification crop size (default: 256 256)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    detection_size = _parse_size(args.detection_size, "--detection-size")
    crop_size = _parse_size(args.crop_size, "--crop-size")

    saved_paths = build_internal_pickles(
        excel_file_path=args.excel,
        raw_directory=args.raw,
        ap_xml_directory=args.ap_xml,
        lat_xml_directory=args.lat_xml,
        output_directory=args.output,
        desired_detection_size=detection_size,
        crop_size=crop_size,
    )

    for process, path in saved_paths.items():
        LOGGER.info("Saved %s pickle to %s", process, path)

    return 0
