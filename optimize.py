"""Core PrefPO optimization loop — single iteration, full run, and multi-trial."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from prefpo.config import (
    DiscriminatorConfig,
    ModelConfig,
    OptimizerConfig,
    PrefPOConfig,
)
from prefpo.generate import generate_outputs, generate_standalone
from prefpo.grading.base import GradeResult, Grader
from prefpo.llm.client import (
    _extract_usage,
    call_discriminator_with_messages,
    call_optimizer_with_messages,
)
from prefpo.pool import PromptPool
from prefpo.progress import MultiTrialDisplay, ProgressDisplay
from prefpo.prompts.discriminator import (
    DISCRIMINATOR_SCHEMA,
    build_discriminator_prompt,
    build_instruction_trajectory,
    build_standalone_trajectory,
)
from prefpo.prompts.optimizer import OPTIMIZER_SCHEMA, build_optimizer_prompt
from prefpo.prompts.variant import generate_prompt_variant
from prefpo.results import RunLogger
from prefpo.types import (
    DiscriminatorResult,
    IterationRecord,
    Prompt,
    PromptRole,
    Sample,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleIterationResult:
    preferred: int
    feedback: str
    improved_prompt: Prompt
    discriminator_usage: dict[str, int]
    optimizer_usage: dict[str, int]


async def run_iteration(
    prompt_a: Prompt,
    prompt_b: Prompt,
    mode: Literal["instruction", "standalone"],
    train_samples: list[Sample] | None,
    grader: Grader,
    task_model: ModelConfig,
    disc_config: DiscriminatorConfig,
    opt_config: OptimizerConfig,
    iteration_index: int,
    semaphore: asyncio.Semaphore,
) -> SingleIterationResult:
    """One PRPO iteration: generate trajectories, discriminate, improve."""

    # Step 1 — Generate trajectories (mode-dependent, parallel)
    if mode == "instruction":
        outputs_a, outputs_b = await asyncio.gather(
            generate_outputs(prompt_a, train_samples, task_model, semaphore),
            generate_outputs(prompt_b, train_samples, task_model, semaphore),
        )
        traj_a = build_instruction_trajectory(
            outputs_a, train_samples, disc_config.show_expected
        )
        traj_b = build_instruction_trajectory(
            outputs_b, train_samples, disc_config.show_expected
        )
    else:
        outputs_a, outputs_b = await asyncio.gather(
            generate_standalone(prompt_a, task_model, semaphore),
            generate_standalone(prompt_b, task_model, semaphore),
        )
        traj_a = build_standalone_trajectory(
            outputs_a, grader, disc_config.show_expected, prompt_a.value
        )
        traj_b = build_standalone_trajectory(
            outputs_b, grader, disc_config.show_expected, prompt_b.value
        )

    # Step 2 — Discriminate (message-passing)
    sys_prompt, user_prompt = build_discriminator_prompt(traj_a, traj_b, disc_config)
    disc_output, disc_messages, disc_response = await call_discriminator_with_messages(
        model=disc_config.model.name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        is_reasoning=disc_config.model.is_reasoning,
        reasoning_effort=disc_config.model.reasoning_effort,
        json_schema=DISCRIMINATOR_SCHEMA,
    )
    disc_result = DiscriminatorResult(
        preferred=disc_output["preferred"],
        feedback=disc_output["feedback"],
    )
    disc_usage = _extract_usage(disc_response)

    # Step 3 — Optimize (pass disc_messages for context)
    non_pref_prompt = prompt_b if disc_result.preferred == 1 else prompt_a
    opt_user_prompt = build_optimizer_prompt(
        preferred=disc_result.preferred,
        non_preferred_prompt=non_pref_prompt,
        feedback=disc_result.feedback,
        config=opt_config,
    )
    opt_output, opt_response = await call_optimizer_with_messages(
        model=opt_config.model.name,
        messages=disc_messages,
        optimizer_prompt=opt_user_prompt,
        is_reasoning=opt_config.model.is_reasoning,
        reasoning_effort=opt_config.model.reasoning_effort,
        json_schema=OPTIMIZER_SCHEMA,
    )
    opt_usage = _extract_usage(opt_response)

    # Step 4 — Create improved prompt
    new_text = opt_output["prompt"].strip()
    if not new_text:
        logger.warning(
            "Optimizer returned empty prompt at iteration %d, keeping non-preferred",
            iteration_index,
        )
        new_text = non_pref_prompt.value

    improved = Prompt(
        value=new_text,
        role=prompt_a.role,
        name=f"improved_{iteration_index}",
    )

    return SingleIterationResult(
        preferred=disc_result.preferred,
        feedback=disc_result.feedback,
        improved_prompt=improved,
        discriminator_usage=disc_usage,
        optimizer_usage=opt_usage,
    )


# ---------------------------------------------------------------------------
# Full optimization run
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    run_id: str
    best_prompt: Prompt
    best_score: float
    best_test_score: float | None
    final_pool: dict
    history: list[IterationRecord]
    total_tokens: dict[str, dict[str, int]]


async def _score_prompt(
    grader: Grader,
    prompt: Prompt,
    samples: list[Sample] | None,
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
    mode: str,
) -> GradeResult:
    """Score a prompt using the grader (mode-dependent sample passing)."""
    if mode == "standalone":
        return await grader.grade(prompt, None, model_config, semaphore)
    return await grader.grade(prompt, samples, model_config, semaphore)


async def optimize_async(
    config: PrefPOConfig,
    grader: Grader,
    train: list[Sample] | None = None,
    val: list[Sample] | None = None,
    test: list[Sample] | None = None,
    _trial_index: int | None = None,
    _on_iteration: Callable[[int, float], None] | None = None,
) -> OptimizationResult:
    """Run the full PrefPO optimization (async version).

    If config.run.n_trials > 1, dispatches to optimize_multi_trial()
    and returns the best trial's OptimizationResult.
    """
    # Dispatch to multi-trial if needed
    if config.run.n_trials > 1:
        multi = await optimize_multi_trial(config, grader, train, val, test)
        best_trial = max(multi.trials, key=lambda t: t.best_score)

        # Log aggregate stats so they're accessible from disk
        import uuid
        from datetime import datetime, timezone
        from pathlib import Path
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        summary = {
            "n_trials": config.run.n_trials,
            "mean_val": multi.mean_val,
            "std_val": multi.std_val,
            "mean_test": multi.mean_test,
            "std_test": multi.std_test,
            "best_trial_run_id": best_trial.run_id,
            "best_trial_score": best_trial.best_score,
            "trials": [
                {"run_id": t.run_id, "val_score": t.best_score, "test_score": t.best_test_score}
                for t in multi.trials
            ],
        }
        summary_path = Path(config.run.output_dir) / f"multi_trial_summary_{timestamp}_{suffix}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Multi-trial complete: mean_val=%.3f (std=%.3f), best=%.3f",
            multi.mean_val, multi.std_val, best_trial.best_score,
        )
        return best_trial

    # --- Input validation ---
    if config.mode == "instruction":
        if not train:
            raise ValueError("Instruction mode requires train samples")
    elif config.mode == "standalone":
        if train or val or test:
            raise ValueError("Standalone mode does not accept train/val/test")
        if config.pool.prompt_role != "user":
            raise ValueError("Standalone mode requires prompt_role='user'")
        for p_text in config.pool.initial_prompts:
            if not p_text.strip():
                raise ValueError("Standalone mode requires non-empty initial prompts")
        # Validate grader.check_output() if show_expected=True
        if config.discriminator.show_expected:
            test_result = grader.check_output("test output", config.pool.initial_prompts[0])
            if test_result is None:
                raise ValueError(
                    "show_expected=True requires grader.check_output() to return a dict. "
                    "Override check_output() in your Grader subclass."
                )

    # --- Setup ---
    if config.run.verbose:
        n_prompts = len(config.pool.initial_prompts)
        print(
            f"Starting {config.mode} optimization "
            f"({config.run.iterations} iterations, {n_prompts} seed prompt{'s' if n_prompts != 1 else ''})"
        )

    semaphore = asyncio.Semaphore(config.run.max_concurrent)
    run_logger = RunLogger(config, trial_index=_trial_index)

    # Build prompt pool
    role = PromptRole(config.pool.prompt_role)
    prompts = [Prompt(value=text, role=role) for text in config.pool.initial_prompts]
    pool = PromptPool(prompts, seed=config.pool.sampling_seed)

    # Handle single-prompt pools — generate a variant so the first iteration
    # has two meaningfully different prompts to compare.
    if len(pool.entries) == 1:
        variant_text = await generate_prompt_variant(
            pool.entries[0].value,
            config.discriminator.criteria,
            config.optimizer.model,
            semaphore,
        )
        pool.add(Prompt(value=variant_text, role=role, name="variant_0"))

    # Determine scoring samples
    scoring_samples = val if val else train

    history: list[IterationRecord] = []
    total_disc_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_opt_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Progress display (no-op when _on_iteration is set — multi-trial manages its own)
    show_progress = _on_iteration is None
    progress = ProgressDisplay(
        config.run.iterations, verbose=config.run.verbose and show_progress,
    )
    progress.start()

    # --- Main loop ---
    for i in range(config.run.iterations):
        logger.info("Iteration %d/%d", i + 1, config.run.iterations)

        # Sample pair
        prompt_a, prompt_b = pool.sample_pair()

        # Score both in parallel (skip if already cached)
        score_a = pool.get_score(prompt_a)
        score_b = pool.get_score(prompt_b)

        scores_to_get = []
        if score_a is None:
            scores_to_get.append(("a", prompt_a))
        if score_b is None:
            scores_to_get.append(("b", prompt_b))

        if scores_to_get:
            progress.set_status("scoring prompts...")
            score_results = await asyncio.gather(
                *[
                    _score_prompt(
                        grader, p, scoring_samples,
                        config.task_model, semaphore, config.mode,
                    )
                    for _, p in scores_to_get
                ]
            )
            for (label, prompt), result in zip(scores_to_get, score_results):
                pool.set_score(prompt, result.score)
                if config.run.save_outputs:
                    run_logger.log_grading_outputs(
                        i, prompt.name, f"prompt_{label}",
                        result.score, result.outputs, result.per_sample,
                    )
                if label == "a":
                    score_a = result.score
                else:
                    score_b = result.score

        # Run iteration
        progress.set_status("generating trajectories + judging...")
        iter_result = await run_iteration(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            mode=config.mode,
            train_samples=train,
            grader=grader,
            task_model=config.task_model,
            disc_config=config.discriminator,
            opt_config=config.optimizer,
            iteration_index=i,
            semaphore=semaphore,
        )

        # Update pool
        if config.pool.update_strategy == "add":
            pool.add(iter_result.improved_prompt)
        else:
            pool.replace_non_preferred(
                iter_result.improved_prompt,
                prompt_a.name,
                prompt_b.name,
                iter_result.preferred,
            )

        # Score improved prompt
        progress.set_status("scoring improved prompt...")
        improved_grade = await _score_prompt(
            grader, iter_result.improved_prompt, scoring_samples,
            config.task_model, semaphore, config.mode,
        )
        pool.set_score(iter_result.improved_prompt, improved_grade.score)
        if config.run.save_outputs:
            run_logger.log_grading_outputs(
                i, iter_result.improved_prompt.name, "improved",
                improved_grade.score, improved_grade.outputs, improved_grade.per_sample,
            )

        # Accumulate token usage
        for k in total_disc_tokens:
            total_disc_tokens[k] += iter_result.discriminator_usage.get(k, 0)
            total_opt_tokens[k] += iter_result.optimizer_usage.get(k, 0)

        # Log iteration
        record = IterationRecord(
            iteration=i,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            prompt_a_score=score_a,
            prompt_b_score=score_b,
            preferred=iter_result.preferred,
            feedback=iter_result.feedback,
            improved_prompt=iter_result.improved_prompt,
            improved_score=improved_grade.score,
            discriminator_usage=iter_result.discriminator_usage,
            optimizer_usage=iter_result.optimizer_usage,
        )
        history.append(record)
        run_logger.log_iteration(record)

        best_so_far = pool.get_score(pool.best()) or 0.0
        progress.complete_iteration(
            i, improved_grade.score, best_so_far, iter_result.preferred,
        )
        if _on_iteration is not None:
            _on_iteration(i, best_so_far)

        logger.info(
            "  Iter %d: preferred=%d, improved_score=%.3f",
            i, iter_result.preferred, improved_grade.score,
        )

    # --- Finalize ---
    best = pool.best()
    best_score = pool.get_score(best) or 0.0

    # Test score for best prompt only
    best_test_score: float | None = None
    if test and config.mode == "instruction":
        test_grade = await _score_prompt(
            grader, best, test,
            config.task_model, semaphore, config.mode,
        )
        best_test_score = test_grade.score

    # Log final state — add test score for best prompt only
    pool_state = pool.to_dict()
    if best_test_score is not None:
        for entry in pool_state["prompts"]:
            if entry["name"] == best.name:
                entry["test_score"] = best_test_score
            else:
                entry["test_score"] = None
    run_logger.log_final_pool(pool_state)

    total_tokens = {
        "discriminator": total_disc_tokens,
        "optimizer": total_opt_tokens,
    }
    summary = {
        "run_id": run_logger.run_id,
        "best_prompt_name": best.name,
        "best_prompt_value": best.value,
        "best_val_score": best_score,
        "best_test_score": best_test_score,
        "iterations": config.run.iterations,
        "total_tokens": total_tokens,
    }
    run_logger.log_summary(summary)

    progress.finish(
        best_score, best_test_score, best.name, run_logger.run_id,
        results_dir=str(run_logger.run_dir),
    )

    logger.info(
        "Optimization complete. Best score: %.3f, test: %s",
        best_score,
        f"{best_test_score:.3f}" if best_test_score is not None else "N/A",
    )

    return OptimizationResult(
        run_id=run_logger.run_id,
        best_prompt=best,
        best_score=best_score,
        best_test_score=best_test_score,
        final_pool=pool_state,
        history=history,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def optimize(
    config: PrefPOConfig,
    grader: Grader,
    train: list[Sample] | None = None,
    val: list[Sample] | None = None,
    test: list[Sample] | None = None,
) -> OptimizationResult:
    """Run the full PrefPO optimization.

    Sync entry point — calls optimize_async() internally.
    For async contexts, use optimize_async() directly.
    """
    return asyncio.run(optimize_async(
        config, grader, train=train, val=val, test=test,
    ))


# ---------------------------------------------------------------------------
# Multi-trial runner
# ---------------------------------------------------------------------------


@dataclass
class MultiTrialResult:
    trials: list[OptimizationResult]
    mean_val: float
    std_val: float
    mean_test: float | None
    std_test: float | None


async def optimize_multi_trial(
    config: PrefPOConfig,
    grader: Grader,
    train: list[Sample] | None = None,
    val: list[Sample] | None = None,
    test: list[Sample] | None = None,
) -> MultiTrialResult:
    """Run config.run.n_trials independent trials and aggregate.

    By default all trials use the same sampling_seed from config.
    If config.run.vary_seed=True, each trial gets sampling_seed + trial_index.
    """
    n_trials = config.run.n_trials
    if config.run.verbose:
        print(
            f"Starting {config.mode} optimization "
            f"({n_trials} trials, {config.run.iterations} iterations each)"
        )
    display = MultiTrialDisplay(n_trials, config.run.iterations, verbose=config.run.verbose)
    display.start()

    async def _run_trial(trial_idx: int) -> OptimizationResult:
        logger.info("=== Trial %d/%d ===", trial_idx + 1, n_trials)
        trial_config = config.model_copy(deep=True)
        trial_config.run.n_trials = 1  # Prevent recursive dispatch
        trial_config.run.verbose = False  # Multi-trial display handles progress
        if config.run.vary_seed:
            trial_config.pool.sampling_seed = config.pool.sampling_seed + trial_idx
        result = await optimize_async(
            trial_config, grader, train, val, test,
            _trial_index=trial_idx,
            _on_iteration=display.make_callback(trial_idx),
        )
        display.complete_trial(trial_idx, result.best_score)
        return result

    trials = list(await asyncio.gather(*[_run_trial(i) for i in range(n_trials)]))

    # Aggregate stats
    val_scores = [t.best_score for t in trials]
    mean_val = sum(val_scores) / len(val_scores)
    std_val = (
        (sum((s - mean_val) ** 2 for s in val_scores) / len(val_scores)) ** 0.5
    )

    test_scores = [t.best_test_score for t in trials if t.best_test_score is not None]
    if test_scores:
        mean_test = sum(test_scores) / len(test_scores)
        std_test = (
            (sum((s - mean_test) ** 2 for s in test_scores) / len(test_scores)) ** 0.5
        )
    else:
        mean_test = None
        std_test = None

    best_trial_score = max(t.best_score for t in trials)
    display.finish(
        mean_val, std_val, best_trial_score, mean_test, std_test,
        results_dir=config.run.output_dir,
    )

    return MultiTrialResult(
        trials=trials,
        mean_val=mean_val,
        std_val=std_val,
        mean_test=mean_test,
        std_test=std_test,
    )
