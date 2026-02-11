"""Direct API usage with optimize_multi_trial()."""

import asyncio

from prefpo import PrefPOConfig, optimize_multi_trial
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


async def main() -> None:
    task = "disambiguation_qa"
    train, val, test = load_bbh(task, train_size=1, val_size=1, test_size=0, seed=42)
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
            "n_trials": 2,
            "vary_seed": True,
            "max_concurrent": 1,
            "output_dir": "results/seamless/optimize_multi_trial_direct",
        },
    )

    result = await optimize_multi_trial(config, grader=grader, train=train, val=val, test=test)
    print(f"Mean val score: {result.mean_val:.3f}")
    print(f"Std val score: {result.std_val:.3f}")
    for idx, trial in enumerate(result.trials):
        print(f"Trial {idx + 1} run_id: {trial.run_id} val={trial.best_score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
