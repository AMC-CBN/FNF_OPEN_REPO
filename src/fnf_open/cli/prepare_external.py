"""Command line helpers for preprocessing the external dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ..preprocessing import (
    build_combined_pickle,
    build_single_view_pickle,
    save_pickle,
    split_patient_files,
)

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _add_common_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("excel", type=Path, help="Path to the Excel metadata file")
    parser.add_argument("split", type=Path, help="Directory containing per-patient files")
    parser.add_argument("output", type=Path, help="Destination pickle file")


def _command_split(args: argparse.Namespace) -> None:
    copied = split_patient_files(args.source, args.target)
    LOGGER.info("Copied %s files", copied)


def _command_combined(args: argparse.Namespace) -> None:
    dataset = build_combined_pickle(
        excel_file_path=args.excel,
        split_directory=args.split,
        ap_xml_directory=args.ap_xml,
        lat_xml_directory=args.lat_xml,
    )
    save_pickle(dataset, args.output)


def _command_single(args: argparse.Namespace) -> None:
    dataset = build_single_view_pickle(
        excel_file_path=args.excel,
        split_directory=args.split,
        xml_directory=args.xml,
        view=args.view,
    )
    save_pickle(dataset, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Group raw files into patient folders")
    split_parser.add_argument("source", type=Path, help="Directory containing the raw files")
    split_parser.add_argument("target", type=Path, help="Destination directory for grouped files")
    split_parser.set_defaults(func=_command_split)

    combined_parser = subparsers.add_parser(
        "combined",
        help="Create a pickle file containing both AP and LAT image paths",
    )
    _add_common_dataset_arguments(combined_parser)
    combined_parser.add_argument("ap_xml", type=Path, help="Directory containing AP XML labels")
    combined_parser.add_argument("lat_xml", type=Path, help="Directory containing LAT XML labels")
    combined_parser.set_defaults(func=_command_combined)

    single_parser = subparsers.add_parser(
        "single",
        help="Create a pickle file for a single projection (AP or LAT)",
    )
    _add_common_dataset_arguments(single_parser)
    single_parser.add_argument("xml", type=Path, help="Directory containing XML labels")
    single_parser.add_argument("view", choices=("AP", "LAT"), help="Projection to preprocess")
    single_parser.set_defaults(func=_command_single)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    args.func(args)
    return 0
