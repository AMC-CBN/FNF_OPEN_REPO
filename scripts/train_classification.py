#!/usr/bin/env python3
"""Legacy wrapper for the :mod:`fnf_open.cli.train_classifier` entry point."""

from __future__ import annotations

from typing import List

from fnf_open.cli.train_classifier import main as _main


def main(argv: List[str] | None = None) -> None:
    _main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
