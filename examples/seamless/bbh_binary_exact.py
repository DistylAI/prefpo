"""Short BBH coverage for binary and exact-match graders."""

from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


def run_binary() -> None:
    task = "sports_understanding"
    train, val, test = load_bbh(task, train_size=1, val_size=1, test_size=1, seed=42)
    grader = get_bbh_grader(task)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "show_expected": True,
            "criteria": "correctness of yes/no decision",
        },
        optimizer={"model": {"name": "openai/gpt-4.1", "is_reasoning": False}},
        pool={
            "initial_prompts": [
                "Answer and end with exactly 'ANSWER: yes' or 'ANSWER: no'.",
                "Reason briefly, then end with exactly 'ANSWER: yes' or 'ANSWER: no'.",
            ],
            "prompt_role": "user",
            "update_strategy": "replace",
        },
        run={
            "iterations": 1,
            "max_concurrent": 1,
            "output_dir": "results/seamless/bbh_binary_exact/binary",
        },
    )

    result = optimize(config, grader=grader, train=train, val=val, test=test)
    print(f"[binary] run_id={result.run_id} val={result.best_score:.3f}")


def run_exact() -> None:
    task = "object_counting"
    train, val, test = load_bbh(task, train_size=1, val_size=1, test_size=1, seed=42)
    grader = get_bbh_grader(task)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "show_expected": True,
            "criteria": "correctness of the final answer",
        },
        optimizer={"model": {"name": "openai/gpt-4.1", "is_reasoning": False}},
        pool={
            "initial_prompts": [
                "Solve the question and end with only the final answer.",
                "Think briefly and end with only the final answer.",
            ],
            "prompt_role": "user",
            "update_strategy": "replace",
        },
        run={
            "iterations": 1,
            "max_concurrent": 1,
            "output_dir": "results/seamless/bbh_binary_exact/exact",
        },
    )

    result = optimize(config, grader=grader, train=train, val=val, test=test)
    print(f"[exact] run_id={result.run_id} val={result.best_score:.3f}")


if __name__ == "__main__":
    run_binary()
    run_exact()
