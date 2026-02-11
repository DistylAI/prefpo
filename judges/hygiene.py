"""LLM judge to evaluate prompt hygiene (readability, spec quality, maintainability)."""

import json
from pathlib import Path
from typing import Any

import yaml

from prefpo.llm.client import call_llm_json

HYGIENE_RESULT_SCHEMA = {
    "type": "json_schema",
    "name": "hygiene_result",
    "schema": {
        "type": "object",
        "properties": {
            "overall_reasoning": {"type": "string"},
            "readability_reasoning": {"type": "string"},
            "readability_score": {"type": "integer"},
            "spec_quality_reasoning": {"type": "string"},
            "spec_quality_score": {"type": "integer"},
            "maintainability_reasoning": {"type": "string"},
            "maintainability_score": {"type": "integer"},
        },
        "required": [
            "overall_reasoning",
            "readability_reasoning",
            "readability_score",
            "spec_quality_reasoning",
            "spec_quality_score",
            "maintainability_reasoning",
            "maintainability_score",
        ],
        "additionalProperties": False,
    },
}


def _load_examples() -> list[dict]:
    """Load few-shot examples from YAML file."""
    examples_path = Path(__file__).parent / "hygiene_examples.yaml"
    with open(examples_path) as f:
        data = yaml.safe_load(f)
    return data["examples"]


def _format_examples(examples: list[dict]) -> str:
    """Format few-shot examples for the prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        context_str = "\n".join(f'   "{c}"' for c in ex["criteria"])
        formatted.append(f"""Example {i}:
Prompt: {json.dumps(ex["prompt"])}
Context:[
{context_str}
]
Overall Reasoning: {ex["overall_reasoning"]}
Readability Reasoning: {ex["readability_reasoning"]}
Specification Quality Reasoning: {ex["spec_quality_reasoning"]}
Maintainability Reasoning: {ex["maintainability_reasoning"]}
Scores:
- Readability: {ex["readability_score"]}/2
- Specification Quality: {ex["spec_quality_score"]}/2
- Maintainability: {ex["maintainability_score"]}/2""")
    return "\n\n".join(formatted)


def _build_prompt(
    prompt: str,
    context: str | None,
    few_shot_examples: list[dict],
) -> tuple[str, str]:
    """Build the system and user prompts for the judge."""
    system_prompt = """You are an expert evaluator assessing the quality of prompts based on hygiene metrics.

## Your Task
Evaluate the prompt below for readability, specification quality, and maintainability. Your evaluation should be thorough, fair, and consistent."""

    if context is not None:
        system_prompt += """

Important: Do not penalize the prompt for instructions that are required to satisfy the given context. The context represents constraints that must be in the prompt - only evaluate how well those constraints are expressed, not whether they should exist."""

    user_prompt = """## Evaluation Criteria

### Readability (0-2 points)
what we're measuring: Does the prompt read like clear, natural language? Is it easy to understand on first read?
- 0: Dense sentences with nested clauses, parentheticals, or semicolon chains. Ideas jump around without logical connection. Reader would have to re-read multiple times to understand.
- 1: Mix of clear and confusing sections. Some awkward phrasing but overall understandable. Has logical flow but it's not smooth. Occasional dense or hard-to-parse sentences.
- 2: Reads like natural human writing. Ideas flow logically. Easy to understand on first read. Related instructions are grouped together.

### Specification Quality (0-2 points)
what we're measuring: Does the prompt give requirements at the right level of detail? Does it tell the model what the output should be, or does it try to control every detail of how to produce it?
Signs of poor specification: Defensive clauses (many "do not" instructions), verification instructions (telling the model to check its own work), excessive "if X then Y, unless Z" logic that adds unnecessary complexity.
- 0: Extensive over-prescription of structure and format. Many defensive "do not" clauses. Includes verification/self-check instructions.
- 1: Some over-specification but clear high-level goals exist. Some defensive clauses but not excessive. Mix of necessary and unnecessary detail.
- 2: States what the output should accomplish. Constraints feel necessary rather than overly prescriptive. Uses positive framing (what to do) more than negative (what not to do).

### Maintainability (0-2 points)
what we're measuring: If this prompt produces incorrect output, how easy would it be to find and fix the problem?
Signs of poor maintainability: Repetition (same requirement stated multiple times), intertwined instructions (multiple requirements packed into single sentences), monolithic structure (wall of text with no clear organization).
- 0: Requirements repeated multiple times in different words. If something goes wrong, unclear which instruction caused it. Instructions are packed together and hard to separate. Changing one thing might accidentally affect other requirements.
- 1: Some structure is visible. Minor repetition but not excessive. Could probably identify problem areas with some effort.
- 2: Each requirement stated once, clearly. Clear logical structure. If output is wrong, you could point to which instruction is responsible. Could modify one part without unintended side effects.

## Output Format
Provide:
1. Overall reasoning (detailed summary)
2. Readability reasoning (brief) and score (0-2)
3. Specification Quality reasoning (brief) and score (0-2)
4. Maintainability reasoning (brief) and score (0-2)

"""

    few_shot_str = _format_examples(few_shot_examples)

    if context is not None:
        user_prompt += f"""{few_shot_str}

Remember: Do not penalize for instructions required by the context provided. Evaluate how the requirements are expressed, not whether they should exist.

Provide your evaluation with overall reasoning first, then reasoning and score for each axis.

Now evaluate this prompt:
Prompt: {json.dumps(prompt)}
Context:
{context}
"""
    else:
        user_prompt += f"""{few_shot_str}

Provide your evaluation with overall reasoning first, then reasoning and score for each axis.

Now evaluate this prompt:
Prompt: {json.dumps(prompt)}
"""

    return system_prompt, user_prompt


async def judge_prompt_hygiene(
    prompt: str,
    *,
    context: str | None = None,
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Judge the hygiene of a prompt.

    Args:
        prompt: The prompt text to evaluate.
        context: Optional context describing constraints the prompt must satisfy.
            When provided, the judge won't penalize for instructions required by
            the context.
        model: Model to use for judging.
        temperature: Sampling temperature.

    Returns:
        Dict with overall_reasoning, per-axis reasoning, and scores (0-2)
        for readability, spec_quality, and maintainability.
    """
    few_shot_examples = _load_examples()
    system_prompt, user_prompt = _build_prompt(prompt, context, few_shot_examples)

    parsed, _response = await call_llm_json(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        json_schema=HYGIENE_RESULT_SCHEMA,
    )

    return parsed
