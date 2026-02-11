"""IFEval-Hard with an LLM judge grader — no programmatic checks needed.

This example shows how to replace the official IFEval checker with an LLM
judge. Instead of running code to verify constraints (e.g. counting words,
checking for banned characters), we ask an LLM to read the criteria and
decide whether the response satisfies all of them.

This is useful when:
  - You don't have programmatic checkers for your constraints
  - Your criteria are nuanced and hard to check with code
  - You want a quick prototype before writing custom grading logic

The LLM judge grader works like this:
  1. Generate N responses from the task model (here N=3)
  2. Randomly sample one response (to save compute — you could grade all N
     if you want more reliable scores, at the cost of more LLM calls)
  3. Ask the judge LLM: "Does this response satisfy ALL of these criteria?"
  4. The judge returns a structured JSON with score (0 or 1) and reasoning
  5. That score becomes the grader's output

The rest of the optimization loop is unchanged — PrefPO's discriminator
and optimizer still compare prompt pairs and generate improvements.

Compare with ifeval_hard.py which uses the official IFEval checker instead.
"""

import asyncio
import random

from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_standalone
from prefpo.llm.client import call_llm_json
from prefpo.data.ifeval_hard import load_ifeval_hard_sample
from prefpo.types import Prompt, Sample
from prefpo.config import ModelConfig

# JSON schema for structured judge output
LLM_GRADE_SCHEMA = {
    "type": "json_schema",
    "name": "llm_grade",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1]},
        },
        "required": ["reasoning", "score"],
        "additionalProperties": False,
    },
}


class LLMJudgeGrader(Grader):
    """Grader that uses an LLM to check if a response meets criteria.

    Instead of programmatic checks, this sends the response and criteria
    to a judge LLM and asks for a binary pass/fail decision.

    Args:
        criteria: List of human-readable criteria strings.
        n_eval_trials: Number of responses to generate from the task model.
            One is randomly sampled for grading to save compute.
        judge_model: Model to use for the judge LLM call.
    """

    def __init__(
        self,
        criteria: list[str],
        n_eval_trials: int = 3,
        judge_model: str = "openai/gpt-4.1",
    ) -> None:
        self.criteria = criteria
        self.n_eval_trials = n_eval_trials
        self.judge_model = judge_model

    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample],
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        """Generate responses, sample one, and grade it with an LLM judge."""
        # Step 1: Generate N responses from the task model
        outputs = await generate_standalone(
            prompt, model_config, semaphore, n=self.n_eval_trials
        )

        # Step 2: Randomly sample one response to grade.
        # This saves compute — grading all N would give more reliable scores
        # but costs N judge calls per evaluation instead of 1.
        sampled = random.choice(outputs)

        # Step 3: Ask the LLM judge if the response passes all criteria
        criteria_text = "\n".join(f"- {c}" for c in self.criteria)
        parsed, _response = await call_llm_json(
            model=self.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grader evaluating whether a response satisfies "
                        "a set of criteria. You must check ALL criteria — the "
                        "response only passes if every single criterion is met."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Criteria:\n{criteria_text}\n\n"
                        f"Response:\n{sampled.response}\n\n"
                        "Does this response satisfy ALL of the criteria above?\n"
                        "Provide your reasoning first, then score 1 if ALL criteria "
                        "are satisfied, 0 if any criterion is not met."
                    ),
                },
            ],
            temperature=0.0,
            json_schema=LLM_GRADE_SCHEMA,
        )

        score = parsed["score"]
        raw_outputs = [{"sample_index": -1, "response": o.response} for o in outputs]
        per_sample = [{
            "sampled_response": sampled.response,
            "judge_reasoning": parsed["reasoning"],
            "judge_score": score,
        }]
        return GradeResult(score=float(score), n=1, per_sample=per_sample, outputs=raw_outputs)

    def check_output(self, output: str, prompt_text: str | None = None) -> dict | None:
        """Synchronous annotation for show_target — not available with LLM judge.

        check_output() must be synchronous, so we can't call the LLM here.
        Return None to skip trajectory annotation. The optimization loop
        still works, it just won't show pass/fail in the discriminator context.
        """
        return None


if __name__ == "__main__":
    sample = load_ifeval_hard_sample(0)

    print(f"Sample key: {sample['key']}")
    print(f"Original prompt: {sample['prompt'][:100]}...")
    print(f"Criteria ({len(sample['criteria'])}):")
    for c in sample["criteria"]:
        print(f"  - {c}")

    # Build a standalone config. Note: show_target is False here because
    # LLMJudgeGrader.check_output() returns None (can't call LLM synchronously).
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        discriminator={
            "show_target": False,
            "criteria": sample["criteria"],
        },
        pool={
            "initial_prompts": [sample["prompt"]],
            "prompt_role": "user",
        },
        run={
            "iterations": 3,
            "output_dir": f"results/ifeval_hard_llm_judge/sample_0",
            "save_outputs": True,
        },
    )

    grader = LLMJudgeGrader(
        criteria=sample["criteria"],
        n_eval_trials=3,
        judge_model="openai/gpt-4.1",
    )

    result = optimize(config, grader=grader)

    print(f"\nRun ID: {result.run_id}")
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value}")
