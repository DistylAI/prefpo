"""Tiny instruction-mode smoke test."""

from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


if __name__ == "__main__":
    task = "disambiguation_qa"
    train, val, test = load_bbh(task, train_size=2, val_size=2, test_size=2, seed=42)
    grader = get_bbh_grader(task)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "show_expected": True,
            "criteria": "correctness of the final answer",
        },
        optimizer={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "constraints": "Do not remove the ANSWER format.",
        },
        pool={
            "initial_prompts": [
                "Answer the multiple choice question and end with 'ANSWER: $LETTER'.",
                "Think briefly, then answer and end with 'ANSWER: $LETTER'.",
            ],
            "prompt_role": "user",
            "update_strategy": "replace",
            "sampling_seed": 42,
        },
        run={
            "iterations": 1,
            "max_concurrent": 1,
            "output_dir": "results/seamless/minimal_instruction_smoke",
        },
    )

    result = optimize(config, grader=grader, train=train, val=val, test=test)
    print(f"Run ID: {result.run_id}")
    print(f"Best val score: {result.best_score:.3f}")
    if result.best_test_score is not None:
        print(f"Best test score: {result.best_test_score:.3f}")
