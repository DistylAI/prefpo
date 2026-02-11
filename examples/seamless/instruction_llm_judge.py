"""Instruction-mode example with a custom LLM judge grader."""

from prefpo import GradeResult, Grader, PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.generate import generate_outputs
from prefpo.llm.client import call_llm_json


JUDGE_SCHEMA = {
    "type": "json_schema",
    "name": "instruction_judge",
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
    def __init__(self, judge_model: str = "openai/gpt-4.1") -> None:
        self.judge_model = judge_model

    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        sample_by_index = {sample.index: sample for sample in samples}

        total = 0
        per_sample = []
        for output in outputs:
            sample = sample_by_index[output.sample_index]
            parsed, _ = await call_llm_json(
                model=self.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{sample.question}\n\n"
                            f"Expected answer:\n{sample.target}\n\n"
                            f"Model response:\n{output.response}\n\n"
                            "Return score=1 if the response is correct, else score=0."
                        ),
                    }
                ],
                temperature=0.0,
                json_schema=JUDGE_SCHEMA,
                parse_retries=0,
                max_retries=2,
            )
            score = int(parsed["score"])
            total += score
            per_sample.append(
                {
                    "index": output.sample_index,
                    "score": score,
                    "reasoning": parsed["reasoning"],
                }
            )

        n = len(outputs)
        final_score = total / n if n else 0.0
        return GradeResult(score=final_score, n=n, per_sample=per_sample)


if __name__ == "__main__":
    task = "disambiguation_qa"
    train, val, test = load_bbh(task, train_size=1, val_size=1, test_size=0, seed=42)
    grader = LLMJudgeGrader()

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "show_expected": False,
            "criteria": "maximize judged correctness",
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
            "output_dir": "results/seamless/instruction_llm_judge",
        },
    )

    result = optimize(config, grader=grader, train=train, val=val, test=test)
    print(f"Run ID: {result.run_id}")
    print(f"Best val score: {result.best_score:.3f}")
