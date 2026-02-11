"""IFEval-Hard example — batch optimization using run_ifeval_batch().

IFEval-Hard filters the original 541 IFEval samples down to 148 that GPT-4o
scores below 100% on, making it a harder and more discriminative benchmark
for prompt optimization.

Dataset: https://huggingface.co/datasets/rahul-singhal/IFEval-Hard

This example uses run_ifeval_batch() to optimize multiple samples concurrently
rather than looping sequentially. The batch runner handles concurrency, error
recovery, and writes a batch_summary.json with aggregate metrics.
"""

import asyncio

from prefpo import PrefPOConfig
from prefpo.ifeval_batch import run_ifeval_batch


async def main():
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        discriminator={"show_expected": True},
        pool={"initial_prompts": ["placeholder"], "prompt_role": "user"},
        run={
            "iterations": 3,
            "output_dir": "results/ifeval_hard",
            "save_outputs": True,
        },
    )

    # IFEval-Hard has 148 samples. Here we optimize 3 as a demo.
    results = await run_ifeval_batch(
        sample_indices=[0, 5, 10],
        base_config=config,
        n_eval_trials=5,
        batch_size=3,  # how many samples to optimize concurrently
        dataset="ifeval_hard",
    )

    print(f"\n{'='*60}")
    print("Batch Summary")
    print(f"{'='*60}")
    metrics = results["aggregate_metrics"]
    print(f"  Completed: {results['completed_samples']}/{results['total_samples']}")
    print(f"  Avg score: {metrics.get('avg_final_score', 0):.3f}")
    print(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
    print(f"  Results saved to: {results['batch_dir']}")


if __name__ == "__main__":
    asyncio.run(main())
