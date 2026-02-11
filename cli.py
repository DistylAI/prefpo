"""Minimal CLI for running PrefPO on BBH datasets."""

import argparse
import asyncio
import logging
from pathlib import Path

from prefpo.config import PrefPOConfig
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader
from prefpo.optimize import optimize_async


async def main() -> None:
    parser = argparse.ArgumentParser(description="PrefPO — Preference-based Prompt Optimization")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--dataset", choices=["bbh"], required=True, help="Dataset to use")
    parser.add_argument("--subset", type=str, required=True, help="Dataset subset name")
    parser.add_argument("--train-size", type=int, required=True, help="Number of training samples")
    parser.add_argument("--val-size", type=int, required=True, help="Number of validation samples")
    parser.add_argument("--test-size", type=int, default=None, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for data split")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = PrefPOConfig.from_yaml(args.config)

    if args.dataset == "bbh":
        train, val, test = load_bbh(
            args.subset, args.train_size, args.val_size, args.test_size, args.seed
        )
        grader = get_bbh_grader(args.subset)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    result = await optimize_async(config, grader=grader, train=train, val=val, test=test)

    print(f"\nRun ID: {result.run_id}")
    print(f"Best score (val): {result.best_score:.4f}")
    if result.best_test_score is not None:
        print(f"Best score (test): {result.best_test_score:.4f}")
    print(f"Best prompt:\n{result.best_prompt.value}")


if __name__ == "__main__":
    asyncio.run(main())
