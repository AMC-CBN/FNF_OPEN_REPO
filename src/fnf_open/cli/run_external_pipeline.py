"""Utility CLI to run the full external dataset preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from ..preprocessing import (
    build_combined_pickle,
    build_single_view_pickle,
    save_pickle,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_COMBINED_NAME = "external_combined.pkl"
DEFAULT_AP_NAME = "external_ap.pkl"
DEFAULT_LAT_NAME = "external_lat.pkl"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _resolve_path(root: Path, value: Path) -> Path:
    """Return ``value`` if absolute or ``root / value`` when relative."""

    if value.is_absolute():
        return value
    return root / value


def _validate_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    return path


def _validate_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {path}")
    return path


def _run_pipeline(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    LOGGER.debug("Using dataset root: %s", root)

    excel_path = _validate_file(
        _resolve_path(root, args.excel), "Excel metadata file"
    )
    split_directory = _validate_directory(
        _resolve_path(root, args.split_dir), "Split patient directory"
    )
    ap_xml_directory = _validate_directory(
        _resolve_path(root, args.ap_xml_dir), "AP XML directory"
    )
    lat_xml_directory = _validate_directory(
        _resolve_path(root, args.lat_xml_dir), "LAT XML directory"
    )

    output_directory = _resolve_path(root, args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Saving outputs to: %s", output_directory)

    combined_output = output_directory / args.combined_name
    combined_dataset = build_combined_pickle(
        excel_file_path=excel_path,
        split_directory=split_directory,
        ap_xml_directory=ap_xml_directory,
        lat_xml_directory=lat_xml_directory,
    )
    save_pickle(combined_dataset, combined_output)

    if args.skip_single_view:
        LOGGER.info("Skipped single-view preprocessing as requested")
        return

    ap_output = output_directory / args.ap_output_name
    lat_output = output_directory / args.lat_output_name

    ap_dataset = build_single_view_pickle(
        excel_file_path=excel_path,
        split_directory=split_directory,
        xml_directory=ap_xml_directory,
        view="AP",
    )
    save_pickle(ap_dataset, ap_output)

    lat_dataset = build_single_view_pickle(
        excel_file_path=excel_path,
        split_directory=split_directory,
        xml_directory=lat_xml_directory,
        view="LAT",
    )
    save_pickle(lat_dataset, lat_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset workspace directory (usually the repository's dataset folder)",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("metadata.xlsx"),
        help="Relative or absolute path to the Excel metadata file",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("split"),
        help="Relative or absolute path to the split patient directory",
    )
    parser.add_argument(
        "--ap-xml-dir",
        type=Path,
        default=Path("xml_ap"),
        help="Relative or absolute path to the AP XML directory",
    )
    parser.add_argument(
        "--lat-xml-dir",
        type=Path,
        default=Path("xml_lat"),
        help="Relative or absolute path to the LAT XML directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory where the generated pickle files will be stored",
    )
    parser.add_argument(
        "--combined-name",
        default=DEFAULT_COMBINED_NAME,
        help="Filename for the combined AP/LAT pickle",
    )
    parser.add_argument(
        "--ap-output-name",
        default=DEFAULT_AP_NAME,
        help="Filename for the AP-only pickle",
    )
    parser.add_argument(
        "--lat-output-name",
        default=DEFAULT_LAT_NAME,
        help="Filename for the LAT-only pickle",
    )
    parser.add_argument(
        "--skip-single-view",
        action="store_true",
        help="Only generate the combined pickle and skip AP/LAT specific ones",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    _run_pipeline(args)
    return 0


__all__ = ["build_parser", "main"]

