"""Variant generator for single-prompt pool bootstrap."""

import asyncio

from prefpo.config import ModelConfig
from prefpo.llm.client import call_llm_json


async def generate_prompt_variant(
    original_prompt: str,
    criteria: str | list[str],
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate an alternative version of a prompt for pairwise comparison.

    Called during pool initialization when the pool has a single prompt.
    Asks the LLM to rewrite the prompt while keeping the core task the same,
    so the first iteration has two meaningfully different prompts to compare.
    """
    if isinstance(criteria, str):
        criteria_list = [criteria] if criteria else []
    else:
        criteria_list = criteria

    criteria_text = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(criteria_list))

    system_prompt = (
        "You are an expert prompt writer. Given a prompt and evaluation criteria, "
        "generate an alternative version of the prompt that describes the same task."
    )

    user_prompt = f"""Original prompt:
{original_prompt}

This prompt has the following requirements that models must satisfy:
{criteria_text}

Write an alternative version of this prompt. Requirements:
1. Preserve the same task and all information
2. Do not add new constraints or remove existing ones

Return your alternative prompt as a JSON object with a single "prompt" field."""

    async with semaphore:
        result, _ = await call_llm_json(
            model=model_config.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            is_reasoning=model_config.is_reasoning,
            reasoning_effort=model_config.reasoning_effort,
            temperature=model_config.temperature,
            json_schema={
                "type": "json_schema",
                "name": "variant_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            },
        )

    return result["prompt"]
