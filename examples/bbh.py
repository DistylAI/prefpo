"""BBH (BIG-Bench-Hard) example — uses built-in data loader and grader.

This is the simplest way to run PrefPO on a BBH task. The library provides
load_bbh() for data loading and get_bbh_grader() which auto-selects the right
grader (multiple choice, binary, or exact match) based on the task name.

BBH has 27 tasks across three types:
  - Multiple choice (e.g. disambiguation_qa, date_understanding)
  - Binary (e.g. sports_understanding, navigate, boolean_expressions)
  - Exact match (e.g. object_counting, word_sorting)

See prefpo/data/bbh.py for the full list of tasks and their types.
"""

from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


if __name__ == "__main__":
    # Pick any BBH task — the grader is auto-selected based on task type
    task = "disambiguation_qa"

    train, val, test = load_bbh(task, train_size=30, val_size=30, test_size=30, seed=42)
    grader = get_bbh_grader(task)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        discriminator={
            "show_target": True,
            "criteria": "correctness of the final answer",
        },
        optimizer={
            "constraints": "Do not remove or modify the answer formatting rules (e.g. 'ANSWER: $LETTER'). These are required for automated grading.",
        },
        pool={"initial_prompts": [
            "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C.",
            "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C. Think step by step before answering.",
        ]},
        run={"iterations": 3, "output_dir": f"results/bbh/{task}"},
    )

    result = optimize(config, grader=grader, train=train, val=val, test=test)

    print(f"\nRun ID: {result.run_id}")
    print(f"Best score (val): {result.best_score:.3f}")
    if result.best_test_score is not None:
        print(f"Best score (test): {result.best_test_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value}")
