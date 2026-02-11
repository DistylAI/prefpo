"""IFEval example — uses built-in data loader, config builder, and grader.

IFEval is a standalone-mode task where each prompt has its own set of formatting
constraints (e.g. "respond in all caps", "include exactly 3 bullet points").
PrefPO optimizes each prompt individually to maximize constraint compliance.

The library provides:
  - load_ifeval_sample() to load individual samples with human-readable criteria
  - build_ifeval_config() to create a per-sample config + grader in one call
  - IFEvalGrader which uses the official Google IFEval checker

The grader generates N responses per evaluation and scores:
  score = (responses passing ALL constraints) / N
"""

from prefpo import PrefPOConfig, optimize
from prefpo.data.ifeval import load_ifeval_sample, build_ifeval_config


if __name__ == "__main__":
    # IFEval has 541 samples total. Each is optimized independently.
    from prefpo.data.ifeval import load_ifeval_dataset
    all_samples = load_ifeval_dataset()
    sample_indices = list(range(len(all_samples)))
    results = []

    for i, idx in enumerate(sample_indices):
        sample = load_ifeval_sample(idx)
        print(f"\n{'='*60}")
        print(f"Sample {i + 1}/{len(sample_indices)} (key={sample['key']})")
        print(f"Constraints: {sample['instruction_id_list']}")
        for c in sample["criteria"]:
            print(f"  - {c}")

        # build_ifeval_config handles all the wiring — creates a standalone config
        # with the right criteria, initial prompt, and an IFEvalGrader with the
        # official checker. You can pass a base_config to override defaults.
        config, grader = build_ifeval_config(sample, n_eval_trials=5)
        config.run.iterations = 3
        config.run.output_dir = f"results/ifeval/sample_{idx}"
        config.run.save_outputs = True

        result = optimize(config, grader=grader)
        results.append((idx, sample, result))

        print(f"\nResult: score={result.best_score:.3f}")
        print(f"Best prompt:\n{result.best_prompt.value[:200]}...")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for idx, sample, result in results:
        print(f"  Sample {idx}: {result.best_score:.3f}  constraints={sample['instruction_id_list']}")
