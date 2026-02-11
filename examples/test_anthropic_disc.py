"""Anthropic discriminator test with system role and replace strategy.

Exercises: Claude Sonnet 4.5 as discriminator (non-reasoning),
prompt_role="system", update_strategy="replace", instruction mode with BBH.

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY.
"""

from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


if __name__ == "__main__":
    task = "disambiguation_qa"
    train, val, _ = load_bbh(task, train_size=3, val_size=5, seed=42)
    grader = get_bbh_grader(task)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {
                "name": "anthropic/claude-sonnet-4-5-20250929",
                "is_reasoning": False,
            },
            "show_expected": True,
            "criteria": "correctness of the final answer",
        },
        optimizer={
            "constraints": "Do not remove or modify the answer formatting rules (e.g. 'ANSWER: $LETTER'). These are required for automated grading.",
        },
        pool={
            "initial_prompts": [
                "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C.",
                "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C. Think step by step before answering.",
            ],
            "prompt_role": "system",
            "update_strategy": "replace",
        },
        run={"iterations": 2, "output_dir": f"results/test_anthropic_disc/{task}"},
    )

    result = optimize(config, grader=grader, train=train, val=val)
    print(f"\nBest score: {result.best_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value[:200]}")
