"""Binary choice grader for BBH tasks."""

import asyncio
import math
import re

from prefpo.config import ModelConfig
from prefpo.generate import generate_outputs
from prefpo.grading.base import GradeResult, Grader
from prefpo.types import Prompt, Sample

_STRICT_RE = re.compile(
    r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
    flags=re.MULTILINE,
)
_LOOSE_RE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def _parse_answer_word(response_text: str) -> str | None:
    """Extract a word answer from model response after 'ANSWER:'."""
    match = _STRICT_RE.search(response_text)
    if match is None:
        match = _LOOSE_RE.search(response_text)
    if match is None:
        return None
    return match.group(1).strip().rstrip(".").lower()


class BinaryGrader(Grader):
    """Grader for BBH binary choice tasks.

    Extracts word after 'ANSWER:' via regex, compares to sample.target
    (case-insensitive).
    """

    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample] | None,
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        if samples is None:
            raise ValueError("BinaryGrader requires samples (instruction mode only)")
        # Validate all samples have targets
        for s in samples:
            if not s.target:
                raise ValueError(f"Sample {s.index} has empty target — required for binary grading")

        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        sample_map = {s.index: s for s in samples}

        per_sample: list[dict] = []
        correct = 0
        for output in outputs:
            sample = sample_map[output.sample_index]
            pred = _parse_answer_word(output.response)
            target_lower = sample.target.lower()
            is_correct = pred == target_lower if pred is not None else False
            if is_correct:
                correct += 1
            per_sample.append({
                "index": output.sample_index,
                "pred": pred,
                "target": target_lower,
                "correct": is_correct,
            })

        n = len(outputs)
        accuracy = correct / n if n else 0.0
        stderr = math.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0.0
        raw_outputs = [{"sample_index": o.sample_index, "response": o.response} for o in outputs]
        return GradeResult(score=accuracy, n=n, stderr=stderr, per_sample=per_sample, outputs=raw_outputs)

    def check_output(self, output: str, prompt_text: str | None = None) -> dict | None:
        pred = _parse_answer_word(output)
        return {"pred": pred}
