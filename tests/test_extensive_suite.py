"""Extensive real-API PrefPO test suite (sequential, JSONL logging)."""

import json
import random
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import uuid

import pytest
import yaml

from prefpo.config import PrefPOConfig
from prefpo.data.bbh import (
    BINARY_TASKS,
    EXACT_MATCH_TASKS,
    MULTIPLE_CHOICE_TASKS,
    load_bbh,
)
from prefpo.data.ifeval import build_ifeval_config, load_ifeval_sample
from prefpo.grading import get_bbh_grader
from prefpo.optimize import optimize_async
pytestmark = pytest.mark.live

@dataclass
class Case:
    name: str
    mode: Literal["instruction", "standalone"]
    show_target: bool
    prompt_role: Literal["user", "system"] | None = None
    update_strategy: Literal["add", "replace"] | None = None
    bbh_subset: str | None = None
    ifeval_idx: int | None = None
    iterations: int = 1
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    n_eval_trials: int | None = None
MC_PROMPTS = [
    (
        "Answer the following multiple choice question. The last line of your response "
        "must be in the format 'ANSWER: <LETTER>' where LETTER is the option label from "
        "the question."
    ),
    (
        "Answer the following multiple choice question. Think step by step, then provide "
        "the final answer. The last line of your response must be in the format "
        "'ANSWER: <LETTER>' where LETTER is the option label from the question."
    ),
]

BINARY_PROMPTS = [
    (
        "Answer the following question with yes or no. The last line of your response "
        "must be either 'ANSWER: yes' or 'ANSWER: no'."
    ),
    (
        "Answer the following question with a single yes/no conclusion. The last line of "
        "your response must be either 'ANSWER: yes' or 'ANSWER: no'."
    ),
]

EXACT_MATCH_PROMPTS = [
    (
        "Solve the task. The last line of your response must be in the format "
        "'ANSWER: <final answer>'."
    ),
    (
        "Solve the task carefully. The last line of your response must be "
        "'ANSWER: <final answer>'."
    ),
]
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
def _load_ifeval_indices() -> list[int]:
    config_path = _repo_root() / "inspect_evals/ifeval/configs/batch_config.yaml"
    data = yaml.safe_load(config_path.read_text())
    return list(data["sample_indices"])
def _select_ifeval_indices() -> list[int]:
    indices = _load_ifeval_indices()
    rng = random.Random(42)
    return rng.sample(indices, 10)
def _prompt_pair_for_subset(subset: str) -> list[str]:
    if subset in MULTIPLE_CHOICE_TASKS:
        return MC_PROMPTS
    if subset in BINARY_TASKS:
        return BINARY_PROMPTS
    if subset in EXACT_MATCH_TASKS:
        return EXACT_MATCH_PROMPTS
    raise ValueError(f"Unknown BBH subset: {subset}")
def _optimizer_constraints_for_subset(subset: str) -> str:
    if subset in MULTIPLE_CHOICE_TASKS:
        return "Do not remove or change the required 'ANSWER: <LETTER>' format."
    if subset in BINARY_TASKS:
        return "Do not remove or change the required 'ANSWER: yes/no' format."
    return "Do not remove or change the required 'ANSWER: <final answer>' format."
def _build_instruction_config(case: Case, output_dir: Path) -> tuple[PrefPOConfig, list]:
    train, val, test = load_bbh(
        case.bbh_subset,
        train_size=case.train_size or 2,
        val_size=case.val_size or 2,
        test_size=case.test_size or 2,
        seed=42,
    )
    prompts = _prompt_pair_for_subset(case.bbh_subset)
    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "gpt-4o", "temperature": 0.0},
        discriminator={
            "model": {"name": "gpt-5", "is_reasoning": True, "reasoning_effort": "medium"},
            "show_target": case.show_target,
            "criteria": "correctness and adherence to output format",
        },
        optimizer={
            "model": {"name": "gpt-5", "is_reasoning": True, "reasoning_effort": "medium"},
            "constraints": _optimizer_constraints_for_subset(case.bbh_subset),
        },
        pool={
            "initial_prompts": prompts,
            "prompt_role": case.prompt_role or "user",
            "update_strategy": case.update_strategy or "add",
            "sampling_seed": 42,
        },
        run={
            "iterations": case.iterations,
            "max_concurrent": 50,
            "output_dir": str(output_dir),
        },
    )
    grader = get_bbh_grader(case.bbh_subset)
    return config, grader, train, val, test
def _build_ifeval_config(case: Case, output_dir: Path) -> tuple[PrefPOConfig, object]:
    sample = load_ifeval_sample(case.ifeval_idx)
    base_config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "gpt-4o", "temperature": 0.0},
        discriminator={"model": {"name": "gpt-5", "is_reasoning": True, "reasoning_effort": "medium"}},
        optimizer={"model": {"name": "gpt-5", "is_reasoning": True, "reasoning_effort": "medium"}},
        pool={"initial_prompts": ["placeholder"], "prompt_role": "user"},
        run={"iterations": 1, "max_concurrent": 50, "output_dir": str(output_dir)},
    )
    config, grader = build_ifeval_config(
        sample,
        base_config=base_config,
        n_eval_trials=case.n_eval_trials or 3,
    )
    config.discriminator.show_target = case.show_target
    config.run.iterations = case.iterations
    config.run.max_concurrent = 50
    config.run.output_dir = str(output_dir)
    if case.update_strategy:
        config.pool.update_strategy = case.update_strategy
    return config, grader
def _write_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
@pytest.mark.asyncio
async def test_extensive_suite():
    output_root = Path(__file__).resolve().parent / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"extensive_suite_{timestamp}_{uuid.uuid4().hex[:6]}.jsonl"

    ifeval_indices = _select_ifeval_indices()

    cases: list[Case] = [
        Case(
            name="instr_mc_user_no_gt_add",
            mode="instruction",
            show_target=False,
            prompt_role="user",
            update_strategy="add",
            bbh_subset="disambiguation_qa",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
        Case(
            name="instr_mc_user_gt_add",
            mode="instruction",
            show_target=True,
            prompt_role="user",
            update_strategy="add",
            bbh_subset="disambiguation_qa",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
        Case(
            name="instr_mc_system_gt_add",
            mode="instruction",
            show_target=True,
            prompt_role="system",
            update_strategy="add",
            bbh_subset="disambiguation_qa",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
        Case(
            name="instr_mc_user_gt_replace",
            mode="instruction",
            show_target=True,
            prompt_role="user",
            update_strategy="replace",
            bbh_subset="disambiguation_qa",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
        Case(
            name="instr_binary_user_gt_add",
            mode="instruction",
            show_target=True,
            prompt_role="user",
            update_strategy="add",
            bbh_subset="navigate",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
        Case(
            name="instr_exact_system_gt_add",
            mode="instruction",
            show_target=True,
            prompt_role="system",
            update_strategy="add",
            bbh_subset="object_counting",
            iterations=1,
            train_size=2,
            val_size=2,
            test_size=2,
        ),
    ]

    for idx in ifeval_indices:
        cases.append(
            Case(
                name=f"ifeval_{idx}_no_gt",
                mode="standalone",
                show_target=False,
                ifeval_idx=idx,
                iterations=1,
                n_eval_trials=3,
            )
        )

    for idx in ifeval_indices:
        cases.append(
            Case(
                name=f"ifeval_{idx}_gt",
                mode="standalone",
                show_target=True,
                ifeval_idx=idx,
                iterations=1,
                n_eval_trials=3,
            )
        )

    cases.append(
        Case(
            name="instr_mc_user_gt_add_iter10",
            mode="instruction",
            show_target=True,
            prompt_role="user",
            update_strategy="add",
            bbh_subset="disambiguation_qa",
            iterations=10,
            train_size=2,
            val_size=2,
            test_size=2,
        )
    )
    cases.append(
        Case(
            name=f"ifeval_{ifeval_indices[0]}_gt_iter10",
            mode="standalone",
            show_target=True,
            ifeval_idx=ifeval_indices[0],
            iterations=10,
            n_eval_trials=5,
        )
    )

    failures: list[dict] = []
    print(f"Extensive suite log: {log_path}")

    for case in cases:
        case_output_dir = output_root / "runs" / case.name
        case_output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        status = "passed"
        result = None
        error = None
        tb = None

        try:
            if case.mode == "instruction":
                config, grader, train, val, test = _build_instruction_config(
                    case, case_output_dir
                )
                result = await optimize_async(
                    config, grader=grader, train=train, val=val, test=test
                )
            else:
                config, grader = _build_ifeval_config(case, case_output_dir)
                result = await optimize_async(config, grader=grader)
        except Exception as e:
            status = "failed"
            error = repr(e)
            tb = traceback.format_exc()

        duration = time.perf_counter() - start
        record = {
            "case": asdict(case),
            "status": status,
            "error": error,
            "traceback": tb,
            "duration_s": duration,
            "output_dir": str(case_output_dir),
        }
        if result is not None:
            record.update(
                {
                    "run_id": result.run_id,
                    "best_score": result.best_score,
                    "best_test_score": result.best_test_score,
                    "total_tokens": result.total_tokens,
                }
            )

        _write_jsonl(log_path, record)
        print(f"[{status}] {case.name} ({duration:.1f}s)")

        if status != "passed":
            failures.append(
                {
                    "case": case.name,
                    "error": error,
                }
            )

    if failures:
        failure_lines = "\n".join(
            f"- {f['case']}: {f['error']}" for f in failures
        )
        pytest.fail(f"Extensive suite failures:\n{failure_lines}")
