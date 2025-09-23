"""File system helpers for the CBNU external dataset."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

# Regular expression used to extract the numeric part of the filename.
PATIENT_ID_PATTERN = re.compile(r"P(\d+)")


def _iter_source_files(source_directory: Path) -> Iterable[Path]:
    """Yield relevant files from ``source_directory``.

    Only files that start with ``"P"`` are considered.  Any directories or
    non-matching files are ignored.
    """

    for path in sorted(source_directory.iterdir()):
        if path.is_file() and path.name.startswith("P"):
            yield path


def _destination_folder_name(identifier: str) -> str | None:
    """Return the folder name derived from ``identifier``.

    The identifier is expected to contain between five and seven digits.
    Depending on its length, a different portion of the identifier is used to
    group the files:

    * five digits  -> first digit
    * six digits   -> first two digits
    * seven digits -> first three digits
    """

    if len(identifier) == 5:
        return identifier[0]
    if len(identifier) == 6:
        return identifier[:2]
    if len(identifier) == 7:
        return identifier[:3]
    return None


def split_patient_files(source_directory: str | Path, target_directory: str | Path) -> int:
    """Copy patient files into grouped folders.

    Parameters
    ----------
    source_directory:
        Folder that contains the original files.  Files must start with the
        letter ``"P"`` followed by the patient identifier.
    target_directory:
        Destination directory where grouped folders will be created.  The
        directory will be created if it does not exist.

    Returns
    -------
    int
        The number of files that were copied.

    Notes
    -----
    Only identifiers that match the expected format of five to seven digits are
    considered.
    """

    source_path = Path(source_directory).expanduser().resolve()
    target_path = Path(target_directory).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    copied_files = 0

    for original in _iter_source_files(source_path):
        match = PATIENT_ID_PATTERN.search(original.name)
        if not match:
            LOGGER.debug("Skipping %s: unable to extract patient identifier", original)
            continue

        identifier = match.group(1)
        folder_name = _destination_folder_name(identifier)
        if folder_name is None:
            LOGGER.debug("Skipping %s: unsupported identifier length", original)
            continue

        destination_dir = target_path / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / original.name
        shutil.copy2(original, destination)
        copied_files += 1

    LOGGER.info("Copied %s files into %s", copied_files, target_path)
    return copied_files


__all__ = ["split_patient_files"]
