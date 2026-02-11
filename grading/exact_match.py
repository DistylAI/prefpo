"""Exact match grader for BBH tasks."""

import asyncio
import math
import string

from prefpo.config import ModelConfig
from prefpo.generate import generate_outputs
from prefpo.grading.base import GradeResult, Grader
from prefpo.types import Prompt, Sample


def _strip_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def _match_end(output: str, target: str) -> bool:
    """Check if output ends with target (case-insensitive, punctuation-stripped)."""
    output = _strip_punctuation(output.strip().casefold())
    target = _strip_punctuation(target.strip().casefold())
    return output.endswith(target)


class ExactMatchGrader(Grader):
    """Grader for BBH exact match tasks.

    Checks if model output ends with sample.target
    (case-insensitive, punctuation-stripped).
    """

    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample],
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        # Validate all samples have targets
        for s in samples:
            if not s.target:
                raise ValueError(f"Sample {s.index} has empty target — required for exact match grading")

        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        sample_map = {s.index: s for s in samples}

        per_sample: list[dict] = []
        correct = 0
        for output in outputs:
            sample = sample_map[output.sample_index]
            is_correct = _match_end(output.response, sample.target)
            if is_correct:
                correct += 1
            per_sample.append({
                "index": output.sample_index,
                "response_end": output.response.strip()[-50:] if output.response else "",
                "target": sample.target,
                "correct": is_correct,
            })

        n = len(outputs)
        accuracy = correct / n if n else 0.0
        stderr = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0
        raw_outputs = [{"sample_index": o.sample_index, "response": o.response} for o in outputs]
        return GradeResult(score=accuracy, n=n, stderr=stderr, per_sample=per_sample, outputs=raw_outputs)

    def check_output(self, output: str, prompt_text: str | None = None) -> dict | None:
        return None
