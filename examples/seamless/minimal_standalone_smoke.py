"""Tiny standalone-mode smoke test using one IFEval-Hard sample."""

from prefpo import optimize
from prefpo.data.ifeval_hard import build_ifeval_hard_config, load_ifeval_hard_sample


if __name__ == "__main__":
    sample = load_ifeval_hard_sample(0)
    config, grader = build_ifeval_hard_config(sample, n_eval_trials=1)

    config.task_model.name = "openai/gpt-4o-mini"
    config.discriminator.model.name = "openai/gpt-4.1"
    config.discriminator.model.is_reasoning = False
    config.optimizer.model.name = "openai/gpt-4.1"
    config.optimizer.model.is_reasoning = False
    config.pool.initial_prompts = [
        sample["prompt"],
        sample["prompt"] + "\nFollow every requirement exactly.",
    ]
    config.run.iterations = 1
    config.run.max_concurrent = 1
    config.run.output_dir = "results/seamless/minimal_standalone_smoke"

    result = optimize(config, grader=grader)
    print(f"Run ID: {result.run_id}")
    print(f"Best score: {result.best_score:.3f}")
