"""IFEval grader — uses the official IFEval checker for standalone mode."""

import asyncio
from typing import Any

from instruction_following_eval.evaluation import (
    InputExample,
    ensure_nltk_resource,
    test_instruction_following,
)

from prefpo.config import ModelConfig
from prefpo.generate import generate_standalone
from prefpo.grading.base import GradeResult, Grader
from prefpo.types import Prompt, Sample

# Ensure langdetect is initialized (thread-safe, preloads profiles)
try:
    from langdetect import detect as _detect
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
except ImportError:
    pass


def get_human_readable_criteria(instruction_id: str, kwargs: dict) -> str:
    """Convert instruction ID and kwargs to human-readable criterion."""
    criteria_map = {
        "change_case:capital_word_frequency": lambda k: f"Response must have {k.get('capital_relation', '')} {k.get('capital_frequency', '')} words in ALL CAPS",
        "change_case:english_capital": lambda k: "Entire response must be in ALL CAPITAL LETTERS",
        "change_case:english_lowercase": lambda k: "Entire response must be in all lowercase letters",
        "combination:repeat_prompt": lambda k: f"Response must start by repeating this prompt: '{k.get('prompt_to_repeat', '')}'",
        "combination:two_responses": lambda k: "Must give two different responses separated by 6 asterisks (******)",
        "detectable_content:number_placeholders": lambda k: f"Response must contain at least {k.get('num_placeholders', '')} placeholders (square brackets like [address])",
        "detectable_content:postscript": lambda k: f"Response must include a postscript starting with '{k.get('postscript_marker', '')}'",
        "detectable_format:constrained_response": lambda k: "Response must contain one of: 'My answer is yes.', 'My answer is no.', or 'My answer is maybe.'",
        "detectable_format:json_format": lambda k: "Entire response must be in valid JSON format",
        "detectable_format:multiple_sections": lambda k: f"Response must have at least {k.get('num_sections', '')} sections separated by '{k.get('section_spliter', '')} X' where X is a number",
        "detectable_format:number_bullet_lists": lambda k: f"Response must have exactly {k.get('num_bullets', '')} bullet points",
        "detectable_format:number_highlighted_sections": lambda k: f"Response must have at least {k.get('num_highlights', '')} sections highlighted with markdown (e.g., *highlighted section*)",
        "detectable_format:title": lambda k: "Response must include a title wrapped in double angular brackets (e.g., <<title>>)",
        "keywords:existence": lambda k: f"Response must contain all of these keywords (case-insensitive): {', '.join(k.get('keywords', []))}",
        "keywords:forbidden_words": lambda k: f"Response must NOT contain any of these words (case-insensitive): {', '.join(k.get('forbidden_words', []))}",
        "keywords:frequency": lambda k: f"The keyword '{k.get('keyword', '')}' must appear {k.get('relation', '')} {k.get('frequency', '')} times (case-insensitive)",
        "keywords:letter_frequency": lambda k: f"The letter '{k.get('letter', '')}' must appear {k.get('let_relation', '')} {k.get('let_frequency', '')} times (case-insensitive)",
        "language:response_language": lambda k: f"Entire response must be in language code: {k.get('language', '')}",
        "length_constraints:nth_paragraph_first_word": lambda k: f"Response must have exactly {k.get('num_paragraphs', '')} paragraphs (paragraphs separated by \\n\\n), and the {k.get('nth_paragraph', '')}-th paragraph must start with the word '{k.get('first_word', '')}' (case-insensitive)",
        "length_constraints:number_paragraphs": lambda k: f"Response must have exactly {k.get('num_paragraphs', '')} paragraphs separated by the markdown divider (***)",
        "length_constraints:number_sentences": lambda k: f"Response must have {k.get('relation', '')} {k.get('num_sentences', '')} sentences",
        "length_constraints:number_words": lambda k: f"Response must have {k.get('relation', '')} {k.get('num_words', '')} words",
        "punctuation:no_comma": lambda k: "Response must NOT contain any commas",
        "startend:end_checker": lambda k: f"Response must end with the exact phrase: '{k.get('end_phrase', '')}'",
        "startend:quotation": lambda k: "Entire response must be wrapped in double quotation marks",
    }

    if instruction_id in criteria_map:
        return criteria_map[instruction_id](kwargs)
    return f"Unknown instruction: {instruction_id}"


def grade_ifeval_response(
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    prompt: str,
    response: str,
    key: str = "sample",
) -> dict[str, Any]:
    """Grade a single response using the official IFEval checker.

    Returns dict with prompt_level_strict, inst_level_strict, etc.
    """
    ensure_nltk_resource()

    new_kwargs = {}
    for index in range(len(instruction_id_list)):
        filtered = {k: v for k, v in kwargs_list[index].items() if v}
        new_kwargs[index] = filtered

    eval_input = InputExample(
        key=key,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=new_kwargs,
    )

    out_strict = test_instruction_following(eval_input, response, strict=True)
    out_loose = test_instruction_following(eval_input, response, strict=False)

    return {
        "prompt_level_strict": out_strict.follow_all_instructions,
        "inst_level_strict": sum(out_strict.follow_instruction_list),
        "prompt_level_loose": out_loose.follow_all_instructions,
        "inst_level_loose": sum(out_loose.follow_instruction_list),
        "num_instructions": len(out_loose.follow_instruction_list),
        "follow_instruction_list_strict": out_strict.follow_instruction_list,
        "follow_instruction_list_loose": out_loose.follow_instruction_list,
    }


class IFEvalGrader(Grader):
    """Grader for IFEval standalone mode.

    Each instance is constructed per-sample with the instruction IDs and kwargs
    needed by the official checker. Generates N responses and checks each.
    """

    def __init__(
        self,
        instruction_id_list: list[str],
        kwargs: list[dict[str, Any]],
        n_eval_trials: int = 20,
    ) -> None:
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.n_eval_trials = n_eval_trials

    async def grade(
        self,
        prompt: Prompt,
        samples: list[Sample] | None,
        model_config: ModelConfig,
        semaphore: asyncio.Semaphore,
    ) -> GradeResult:
        """Generate N responses, grade each with official checker."""
        outputs = await generate_standalone(
            prompt, model_config, semaphore, n=self.n_eval_trials
        )

        per_trial: list[dict] = []
        passed_count = 0
        for output in outputs:
            grade = grade_ifeval_response(
                self.instruction_id_list,
                self.kwargs,
                prompt.value,
                output.response,
            )
            per_trial.append(grade)
            if grade["prompt_level_strict"]:
                passed_count += 1

        n = len(outputs)
        pass_rate = passed_count / n if n else 0.0
        raw_outputs = [{"sample_index": -1, "response": o.response} for o in outputs]
        return GradeResult(score=pass_rate, n=n, per_sample=per_trial, outputs=raw_outputs)

    def check_output(self, output: str, prompt_text: str | None = None) -> dict | None:
        """Grade a single output for trajectory annotation."""
        grade = grade_ifeval_response(
            self.instruction_id_list,
            self.kwargs,
            prompt_text or "",
            output,
        )
        return {
            "passed_strict": grade["prompt_level_strict"],
            "passed_loose": grade["prompt_level_loose"],
            "inst_strict": f"{grade['inst_level_strict']}/{grade['num_instructions']}",
        }
