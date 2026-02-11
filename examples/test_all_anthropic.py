"""All-Anthropic pipeline test — no OpenAI dependency.

Exercises: Anthropic for all three roles (task, discriminator, optimizer),
standalone mode, IFEval-Hard single sample. Proves the library works
without any OpenAI API key.

Requires only ANTHROPIC_API_KEY.
Requires: pip install prefpo[ifeval]
"""

from prefpo import PrefPOConfig, optimize
from prefpo.data.ifeval_hard import load_ifeval_hard_sample, build_ifeval_hard_config


if __name__ == "__main__":
    sample = load_ifeval_hard_sample(5)

    print(f"Sample key: {sample['key']}")
    print(f"Criteria: {sample['criteria']}")

    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "anthropic/claude-haiku-4-5-20251001"},
        discriminator={
            "model": {
                "name": "anthropic/claude-sonnet-4-5-20250929",
                "is_reasoning": False,
            },
            "show_expected": True,
            "criteria": sample["criteria"],
        },
        optimizer={
            "model": {
                "name": "anthropic/claude-sonnet-4-5-20250929",
                "is_reasoning": False,
            },
        },
        pool={"initial_prompts": [sample["prompt"]], "prompt_role": "user"},
        run={
            "iterations": 2,
            "output_dir": "results/test_all_anthropic",
        },
    )

    result = optimize(config, grader=build_ifeval_hard_config(sample, n_eval_trials=5)[1])
    print(f"\nBest score: {result.best_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value[:200]}")
