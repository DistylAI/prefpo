"""Load config from YAML and run a tiny BBH optimization."""

from pathlib import Path

from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    config = PrefPOConfig.from_yaml(root / "examples" / "bbh_config.yaml")

    config.task_model.name = "openai/gpt-4o-mini"
    config.discriminator.model.name = "openai/gpt-4.1"
    config.discriminator.model.is_reasoning = False
    config.optimizer.model.name = "openai/gpt-4.1"
    config.optimizer.model.is_reasoning = False
    config.run.iterations = 1
    config.run.max_concurrent = 1
    config.run.output_dir = "results/seamless/from_yaml_bbh"

    task = "disambiguation_qa"
    train, val, test = load_bbh(task, train_size=2, val_size=2, test_size=2, seed=42)
    grader = get_bbh_grader(task)

    result = optimize(config, grader=grader, train=train, val=val, test=test)
    print(f"Run ID: {result.run_id}")
    print(f"Best val score: {result.best_score:.3f}")
    if result.best_test_score is not None:
        print(f"Best test score: {result.best_test_score:.3f}")
