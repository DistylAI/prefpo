"""Extensive integration test suite — real API calls, real datasets, all parameter combinations.

Tests are ordered: short tests first, long tests last. Each test writes output to
tests/output/{test_name}/ for post-run analysis. Tests are independent — failures
don't stop subsequent tests.

These tests require API keys and cost money. They are excluded from the default
test run. To run them:

    pytest -m live tests/test_integration.py -v --tb=long -s

Run short tests only:
    pytest -m live tests/test_integration.py -v --tb=long -s -k "short or multi_trial"

Run long tests only:
    pytest -m live tests/test_integration.py -v --tb=long -s -k "long"
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prefpo.config import PrefPOConfig
from prefpo.data.bbh import load_bbh
from prefpo.data.ifeval import build_ifeval_config, load_ifeval_sample
from prefpo.grading import get_bbh_grader
from prefpo.optimize import optimize_async, optimize_multi_trial

pytestmark = pytest.mark.live

OUTPUT_DIR = Path(__file__).parent / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
# ---------------------------------------------------------------------------
# Task-specific prompts and optimizer constraints
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "disambiguation_qa": [
        (
            "Answer the following multiple choice question. The last line of your "
            "response should be of the following format: 'ANSWER: $LETTER' (without "
            "quotes) where LETTER is one of A,B,C."
        ),
        (
            "Answer the following multiple choice question. The last line of your "
            "response should be of the following format: 'ANSWER: $LETTER' (without "
            "quotes) where LETTER is one of A,B,C. Think step by step before answering."
        ),
    ],
    "sports_understanding": [
        (
            "Answer the following question. The last line of your response should "
            "be of the following format: 'ANSWER: $WORD' (without quotes) where "
            "WORD is yes or no."
        ),
        (
            "Answer the following question. The last line of your response should "
            "be of the following format: 'ANSWER: $WORD' (without quotes) where "
            "WORD is yes or no. Think step by step before answering."
        ),
    ],
    "object_counting": [
        (
            "Answer the following question. The last line of your response should "
            "be of the following format: 'ANSWER: $VALUE' (without quotes)."
        ),
        (
            "Answer the following question. The last line of your response should "
            "be of the following format: 'ANSWER: $VALUE' (without quotes). Think "
            "step by step before answering."
        ),
    ],
}

TASK_CONSTRAINTS = {
    "disambiguation_qa": (
        "Do not remove or modify the answer formatting rules "
        "(e.g. 'ANSWER: $LETTER'). These are required for automated grading."
    ),
    "sports_understanding": (
        "Do not remove or modify the answer formatting rules "
        "(e.g. 'ANSWER: $WORD'). These are required for automated grading."
    ),
    "object_counting": (
        "Do not remove or modify the answer formatting rules "
        "(e.g. 'ANSWER: $VALUE'). These are required for automated grading."
    ),
}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_bbh_config(
    task: str,
    role: str,
    show_expected: bool,
    strategy: str,
    iterations: int,
    output_dir: str,
    n_trials: int = 1,
    vary_seed: bool = False,
) -> PrefPOConfig:
    """Build a PrefPOConfig for a BBH integration test."""
    return PrefPOConfig(
        mode="instruction",
        task_model={"name": "gpt-4o", "temperature": 0.0},
        discriminator={
            "model": {
                "name": "gpt-5",
                "is_reasoning": True,
                "reasoning_effort": "medium",
            },
            "show_expected": show_expected,
            "criteria": "correctness and quality of reasoning",
        },
        optimizer={
            "model": {
                "name": "gpt-5",
                "is_reasoning": True,
                "reasoning_effort": "medium",
            },
            "constraints": TASK_CONSTRAINTS[task],
        },
        pool={
            "initial_prompts": TASK_PROMPTS[task],
            "prompt_role": role,
            "update_strategy": strategy,
            "sampling_seed": 42,
        },
        run={
            "iterations": iterations,
            "n_trials": n_trials,
            "vary_seed": vary_seed,
            "max_concurrent": 50,
            "output_dir": output_dir,
        },
    )
def _write_result_summary(
    output_dir: Path,
    test_id: str,
    *,
    result=None,
    multi_result=None,
    error: str | None = None,
    duration: float = 0.0,
) -> None:
    """Write result_summary.json for post-run analysis."""
    summary = {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 1),
        "status": "pass" if error is None else "fail",
        "error": error,
    }
    if result is not None:
        summary.update({
            "run_id": result.run_id,
            "best_score": result.best_score,
            "best_test_score": result.best_test_score,
            "best_prompt_preview": result.best_prompt.value[:500],
            "pool_size": len(result.final_pool["prompts"]),
            "iterations_logged": len(result.history),
            "total_tokens": result.total_tokens,
        })
    if multi_result is not None:
        summary.update({
            "n_trials": len(multi_result.trials),
            "mean_val": multi_result.mean_val,
            "std_val": multi_result.std_val,
            "mean_test": multi_result.mean_test,
            "std_test": multi_result.std_test,
            "per_trial_scores": [t.best_score for t in multi_result.trials],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_output_dir(request):
    """Create a per-test output directory under tests/output/."""
    name = request.node.name
    for char in "[]":
        name = name.replace(char, "_")
    d = OUTPUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
# ===========================================================================
# Group 1: Short BBH Tests (2 iterations, train=5, val=10)
# ===========================================================================

BBH_SHORT_PARAMS = [
    # (task, role, show_expected, strategy)
    ("disambiguation_qa", "user", True, "add"),
    ("disambiguation_qa", "user", False, "add"),
    ("disambiguation_qa", "system", True, "add"),
    ("disambiguation_qa", "system", False, "add"),
    ("sports_understanding", "user", True, "add"),
    ("sports_understanding", "user", False, "add"),
    ("sports_understanding", "system", True, "add"),
    ("object_counting", "user", True, "add"),
    ("object_counting", "user", False, "add"),
    ("object_counting", "system", True, "add"),
    ("disambiguation_qa", "user", True, "replace"),
]

BBH_SHORT_IDS = [
    f"{task}-{role}-target_{show}-{strat}"
    for task, role, show, strat in BBH_SHORT_PARAMS
]
@pytest.mark.parametrize(
    "task,role,show_expected,strategy",
    BBH_SHORT_PARAMS,
    ids=BBH_SHORT_IDS,
)
def test_bbh_short(task, role, show_expected, strategy, test_output_dir):
    """Short BBH: 2 iterations across task types, roles, show_expected, strategies."""
    start = time.perf_counter()
    test_id = f"bbh_short-{task}-{role}-target_{show_expected}-{strategy}"

    try:
        train, val, _ = load_bbh(task, train_size=5, val_size=10, seed=42)
        grader = get_bbh_grader(task)
        config = _make_bbh_config(
            task, role, show_expected, strategy,
            iterations=2,
            output_dir=str(test_output_dir),
        )
        result = asyncio.run(optimize(config, grader=grader, train=train, val=val))
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(test_output_dir, test_id, result=result, duration=duration)

    assert 0.0 <= result.best_score <= 1.0, (
        f"best_score={result.best_score} out of [0, 1]"
    )
    assert result.best_prompt.value, "best_prompt.value is empty"
    assert len(result.history) == 2, (
        f"Expected 2 iterations, got {len(result.history)}"
    )
    assert result.run_id.startswith("run_"), (
        f"Unexpected run_id format: {result.run_id}"
    )
    assert "discriminator" in result.total_tokens, "Missing discriminator token usage"
    assert "optimizer" in result.total_tokens, "Missing optimizer token usage"

    if strategy == "replace":
        assert len(result.final_pool["prompts"]) == 2, (
            f"Replace strategy should keep pool at 2, "
            f"got {len(result.final_pool['prompts'])}"
        )
    else:
        assert len(result.final_pool["prompts"]) >= 2, (
            f"Pool should have >= 2 prompts, "
            f"got {len(result.final_pool['prompts'])}"
        )
def test_bbh_multi_trial(test_output_dir):
    """Multi-trial BBH: n_trials=2, same seed, 2 iterations."""
    start = time.perf_counter()
    test_id = "bbh_multi_trial"

    try:
        train, val, _ = load_bbh(
            "disambiguation_qa", train_size=5, val_size=10, seed=42,
        )
        grader = get_bbh_grader("disambiguation_qa")
        config = _make_bbh_config(
            "disambiguation_qa", "user", True, "add",
            iterations=2,
            output_dir=str(test_output_dir),
            n_trials=2,
        )
        multi_result = asyncio.run(
            optimize_multi_trial(config, grader=grader, train=train, val=val)
        )
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(
        test_output_dir, test_id,
        multi_result=multi_result, duration=duration,
    )

    assert len(multi_result.trials) == 2, (
        f"Expected 2 trials, got {len(multi_result.trials)}"
    )
    assert 0.0 <= multi_result.mean_val <= 1.0, (
        f"mean_val={multi_result.mean_val} out of [0, 1]"
    )
    assert multi_result.std_val >= 0.0, (
        f"std_val={multi_result.std_val} is negative"
    )
    for i, trial in enumerate(multi_result.trials):
        assert 0.0 <= trial.best_score <= 1.0, (
            f"Trial {i} best_score={trial.best_score} out of [0, 1]"
        )
        assert trial.best_prompt.value, f"Trial {i} best_prompt.value is empty"
        assert len(trial.history) == 2, (
            f"Trial {i}: expected 2 iterations, got {len(trial.history)}"
        )
def test_bbh_multi_trial_vary_seed(test_output_dir):
    """Multi-trial BBH: n_trials=2, vary_seed=True, 2 iterations."""
    start = time.perf_counter()
    test_id = "bbh_multi_trial_vary_seed"

    try:
        train, val, _ = load_bbh(
            "disambiguation_qa", train_size=5, val_size=10, seed=42,
        )
        grader = get_bbh_grader("disambiguation_qa")
        config = _make_bbh_config(
            "disambiguation_qa", "user", True, "add",
            iterations=2,
            output_dir=str(test_output_dir),
            n_trials=2,
            vary_seed=True,
        )
        multi_result = asyncio.run(
            optimize_multi_trial(config, grader=grader, train=train, val=val)
        )
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(
        test_output_dir, test_id,
        multi_result=multi_result, duration=duration,
    )

    assert len(multi_result.trials) == 2, (
        f"Expected 2 trials, got {len(multi_result.trials)}"
    )
    assert 0.0 <= multi_result.mean_val <= 1.0, (
        f"mean_val={multi_result.mean_val} out of [0, 1]"
    )
    assert multi_result.std_val >= 0.0, (
        f"std_val={multi_result.std_val} is negative"
    )
# ===========================================================================
# Group 2: Short IFEval Tests (2 iterations, n_eval_trials=1)
# ===========================================================================
@pytest.mark.parametrize(
    "show_expected",
    [True, False],
    ids=["target_True", "target_False"],
)
def test_ifeval_short(show_expected, test_output_dir):
    """Short IFEval: sample 5 (challenging subset), 2 iterations."""
    start = time.perf_counter()
    test_id = f"ifeval_short-target_{show_expected}"

    try:
        sample = load_ifeval_sample(5)
        config, grader = build_ifeval_config(sample, n_eval_trials=1)
        config.run.iterations = 2
        config.run.max_concurrent = 50
        config.run.output_dir = str(test_output_dir)
        config.discriminator.show_expected = show_expected

        result = asyncio.run(optimize(config, grader=grader))
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(test_output_dir, test_id, result=result, duration=duration)

    assert 0.0 <= result.best_score <= 1.0, (
        f"best_score={result.best_score} out of [0, 1]"
    )
    assert result.best_prompt.value, "best_prompt.value is empty"
    assert len(result.history) == 2, (
        f"Expected 2 iterations, got {len(result.history)}"
    )
    assert len(result.final_pool["prompts"]) >= 2, (
        f"Pool should have >= 2 prompts, "
        f"got {len(result.final_pool['prompts'])}"
    )
# ===========================================================================
# Group 3: Long BBH Tests (10 iterations, train=5, val=10, test=50)
# ===========================================================================

BBH_LONG_PARAMS = [
    ("disambiguation_qa", "user", True, "add"),
    ("disambiguation_qa", "system", False, "add"),
    ("sports_understanding", "user", True, "add"),
]

BBH_LONG_IDS = [
    f"{task}-{role}-target_{show}-{strat}"
    for task, role, show, strat in BBH_LONG_PARAMS
]
@pytest.mark.parametrize(
    "task,role,show_expected,strategy",
    BBH_LONG_PARAMS,
    ids=BBH_LONG_IDS,
)
def test_bbh_long(task, role, show_expected, strategy, test_output_dir):
    """Long BBH: 10 iterations with test-set scoring for generalization."""
    start = time.perf_counter()
    test_id = f"bbh_long-{task}-{role}-target_{show_expected}-{strategy}"

    try:
        train, val, test_set = load_bbh(
            task, train_size=5, val_size=10, test_size=50, seed=42,
        )
        grader = get_bbh_grader(task)
        config = _make_bbh_config(
            task, role, show_expected, strategy,
            iterations=10,
            output_dir=str(test_output_dir),
        )
        result = asyncio.run(
            optimize(config, grader=grader, train=train, val=val, test=test_set)
        )
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(test_output_dir, test_id, result=result, duration=duration)

    assert 0.0 <= result.best_score <= 1.0, (
        f"best_score={result.best_score} out of [0, 1]"
    )
    assert result.best_prompt.value, "best_prompt.value is empty"
    assert len(result.history) == 10, (
        f"Expected 10 iterations, got {len(result.history)}"
    )
    assert result.best_test_score is not None, (
        "Long test with test set should produce best_test_score"
    )
    assert 0.0 <= result.best_test_score <= 1.0, (
        f"best_test_score={result.best_test_score} out of [0, 1]"
    )
    assert result.run_id.startswith("run_"), (
        f"Unexpected run_id format: {result.run_id}"
    )
    assert len(result.final_pool["prompts"]) >= 2, (
        f"Pool should have >= 2 prompts, "
        f"got {len(result.final_pool['prompts'])}"
    )
# ===========================================================================
# Group 4: Long IFEval Tests (3 iterations, 10 challenging samples)
# ===========================================================================

IFEVAL_CHALLENGING = [5, 7, 9, 11, 12, 14, 19, 22, 23, 33]
@pytest.mark.parametrize(
    "sample_idx",
    IFEVAL_CHALLENGING,
    ids=[f"sample_{i}" for i in IFEVAL_CHALLENGING],
)
def test_ifeval_long(sample_idx, test_output_dir):
    """Long IFEval: 3 iterations per challenging sample, n_eval_trials=1."""
    start = time.perf_counter()
    test_id = f"ifeval_long-sample_{sample_idx}"

    try:
        sample = load_ifeval_sample(sample_idx)
        config, grader = build_ifeval_config(sample, n_eval_trials=1)
        config.run.iterations = 3
        config.run.max_concurrent = 50
        config.run.output_dir = str(test_output_dir)

        result = asyncio.run(optimize(config, grader=grader))
    except Exception as e:
        _write_result_summary(
            test_output_dir, test_id,
            error=repr(e), duration=time.perf_counter() - start,
        )
        raise

    duration = time.perf_counter() - start
    _write_result_summary(test_output_dir, test_id, result=result, duration=duration)

    assert 0.0 <= result.best_score <= 1.0, (
        f"best_score={result.best_score} out of [0, 1]"
    )
    assert result.best_prompt.value, "best_prompt.value is empty"
    assert len(result.history) == 3, (
        f"Expected 3 iterations, got {len(result.history)}"
    )
    assert len(result.final_pool["prompts"]) >= 2, (
        f"Pool should have >= 2 prompts, "
        f"got {len(result.final_pool['prompts'])}"
    )
