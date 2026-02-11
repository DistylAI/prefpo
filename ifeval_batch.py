"""Batch runner for IFEval — runs PrefPO on multiple samples."""

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from prefpo.config import PrefPOConfig
from prefpo.data.ifeval import build_ifeval_config, load_ifeval_sample
from prefpo.data.ifeval_hard import build_ifeval_hard_config, load_ifeval_hard_sample
from prefpo.optimize import OptimizationResult, optimize_async

logger = logging.getLogger(__name__)


async def run_ifeval_sample(
    sample_idx: int,
    base_config: PrefPOConfig,
    n_eval_trials: int,
    dataset: str = "ifeval",
) -> tuple[int, OptimizationResult | None]:
    """Run PrefPO on a single IFEval sample with error handling.

    Returns:
        (sample_idx, result) — result is None on error
    """
    try:
        if dataset == "ifeval_hard":
            sample = load_ifeval_hard_sample(sample_idx)
            config, grader = build_ifeval_hard_config(
                sample, base_config=base_config, n_eval_trials=n_eval_trials
            )
        else:
            sample = load_ifeval_sample(sample_idx)
            config, grader = build_ifeval_config(
                sample, base_config=base_config, n_eval_trials=n_eval_trials
            )
        # Override output_dir to per-sample subdir
        config.run.output_dir = str(
            Path(base_config.run.output_dir) / f"sample_{sample_idx}"
        )
        result = await optimize_async(config, grader=grader)
        return sample_idx, result
    except Exception as e:
        logger.error("Error processing sample %d: %s", sample_idx, e)
        return sample_idx, None


def compute_aggregate_metrics(
    results: dict[int, OptimizationResult],
) -> dict[str, Any]:
    """Compute aggregate metrics across completed samples.

    Args:
        results: Mapping of sample_idx -> OptimizationResult

    Returns:
        Dict with aggregate statistics
    """
    scores = [r.best_score for r in results.values()]
    n = len(scores)
    if n == 0:
        return {"total_samples": 0}

    success_count = sum(1 for s in scores if s >= 1.0)
    avg_score = statistics.mean(scores)

    return {
        "total_samples": n,
        "success_count": success_count,
        "success_rate": success_count / n,
        "avg_final_score": avg_score,
        "std_final_score": statistics.stdev(scores) if n > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
        "per_sample": {
            idx: {
                "best_score": r.best_score,
                "best_prompt": r.best_prompt.value[:200],
                "iterations": len(r.history),
            }
            for idx, r in results.items()
        },
    }


async def run_ifeval_batch(
    sample_indices: list[int],
    base_config: PrefPOConfig | None = None,
    n_eval_trials: int = 20,
    batch_size: int = 20,
    dataset: str = "ifeval",
) -> dict[str, Any]:
    """Run PrefPO optimization across multiple IFEval samples.

    Args:
        sample_indices: List of IFEval sample indices to process
        base_config: Base PrefPOConfig (will be adapted per-sample)
        n_eval_trials: Number of evaluation trials per grading call
        batch_size: Max samples to run concurrently
        dataset: Which dataset to use — "ifeval" (541 samples) or "ifeval_hard" (148 samples)

    Returns:
        Dict with aggregate_metrics, per-sample results, and output dir
    """
    if base_config is None:
        base_config = PrefPOConfig(
            mode="standalone",
            task_model={"name": "openai/gpt-4o"},
            discriminator={"show_expected": True},
            pool={"initial_prompts": ["placeholder"], "prompt_role": "user"},
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(base_config.run.output_dir) / f"batch_{timestamp}"
    base_config.run.output_dir = str(batch_dir)

    batch_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(batch_size)
    start_time = time.time()

    logger.info(
        "Starting IFEval batch: %d samples, batch_size=%d",
        len(sample_indices), batch_size,
    )

    async def run_with_semaphore(idx: int):
        async with semaphore:
            logger.info("Starting sample %d", idx)
            t0 = time.time()
            result = await run_ifeval_sample(idx, base_config, n_eval_trials, dataset=dataset)
            elapsed = time.time() - t0
            _, res = result
            if res:
                logger.info(
                    "Sample %d: score=%.3f [%.1fs]", idx, res.best_score, elapsed
                )
            else:
                logger.warning("Sample %d: FAILED [%.1fs]", idx, elapsed)
            return result

    tasks = [run_with_semaphore(idx) for idx in sample_indices]
    raw_results = await asyncio.gather(*tasks)

    completed = {idx: res for idx, res in raw_results if res is not None}
    elapsed = time.time() - start_time

    metrics = compute_aggregate_metrics(completed)
    metrics["total_time_seconds"] = elapsed

    # Save batch summary
    summary_path = batch_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(
        "Batch complete: %d/%d succeeded, avg_score=%.3f, success_rate=%.1f%% [%.0fs]",
        len(completed), len(sample_indices),
        metrics.get("avg_final_score", 0),
        metrics.get("success_rate", 0) * 100,
        elapsed,
    )

    return {
        "batch_dir": str(batch_dir),
        "aggregate_metrics": metrics,
        "completed_samples": len(completed),
        "total_samples": len(sample_indices),
    }
