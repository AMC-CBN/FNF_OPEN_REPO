"""Legacy wrapper around :mod:`fnf_open.cli.prepare_external`."""

from __future__ import annotations

from typing import Sequence

from fnf_open.cli.prepare_external import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
