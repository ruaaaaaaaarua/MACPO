#!/usr/bin/env python3
"""Summarize MAPPO/STAS HPO artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpo_mappo_stas_microgrid import DEFAULT_ROOT, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    summarize(args.root.resolve())


if __name__ == "__main__":
    main()
