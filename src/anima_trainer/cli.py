"""Command-line entry point: train from a TOML config without the web UI."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import TrainConfig
from .device import environment_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anima-train", description=__doc__)
    parser.add_argument("config", nargs="?", help="Path to a TOML training config.")
    parser.add_argument("--write-default", metavar="PATH",
                        help="Write a default config to PATH and exit.")
    parser.add_argument("--env", action="store_true",
                        help="Print the backend/device report and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.env:
        print(environment_report())
        return 0

    if args.write_default:
        TrainConfig().save(args.write_default)
        print(f"Wrote default config to {args.write_default}")
        return 0

    if not args.config:
        parser.error("a config path is required (or use --write-default / --env)")

    cfg = TrainConfig.load(args.config)
    # Import lazily so --env / --write-default work without torch/diffusers present.
    from .train import train

    state = train(cfg)
    if state.error:
        print(f"Training failed: {state.error}", file=sys.stderr)
        return 1
    print(f"Done. Final LoRA: {state.last_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
