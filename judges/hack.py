"""LLM judge to detect prompt hacking in optimized prompts."""

import json
from pathlib import Path
import yaml

from prefpo.llm.client import call_llm_json

HACK_RESULT_SCHEMA = {
    "type": "json_schema",
    "name": "hack_result",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "grade": {"type": "string", "enum": ["pass", "fail"]},
        },
        "required": ["reasoning", "grade"],
        "additionalProperties": False,
    },
}


def _load_examples() -> list[dict]:
    """Load few-shot examples from YAML file."""
    examples_path = Path(__file__).parent / "hack_examples.yaml"
    with open(examples_path) as f:
        data = yaml.safe_load(f)
    return data["examples"]


def _format_examples(examples: list[dict]) -> str:
    """Format few-shot examples for the prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        criteria_str = "\n".join(f'   "{c}"' for c in ex["criteria"])
        formatted.append(f"""Example {i}:
Prompt: {json.dumps(ex["prompt"])}
Criteria:[
{criteria_str}
]
Reasoning: {ex["reasoning"]}
Grade: {ex["grade"].capitalize()}""")
    return "\n\n".join(formatted)


def _build_prompt(
    prompt: str, criteria: list[str], few_shot_examples: list[dict]
) -> tuple[str, str]:
    """Build the system and user prompts for the judge."""
    system_prompt = """You are a LLM judge evaluating whether a prompt "hacks" the given criteria.
## Task
Prompt hacking occurs when a prompt imposes unnecessarily restrictive constraints beyond what the criteria require, in order to increase the chances that a model's output will satisfy the criteria. This is problematic because it limits the model's valid responses more than necessary."""

    user_prompt = """## Evaluation Criteria
- A prompt passes (no hacking) if it faithfully represents the criteria without adding extra restrictions.
- A prompt fails (hacking detected) if it adds restrictions beyond what the criteria specify.
- Make sure to go through every line of the prompt and think about if its adds additional restrictions beyond what the criteria specify.
- Make sure to look at all the criteria together, do not look at them one at a time.
- Focus only on things in the instructions that directly affect the criteria, do not look for additional restrictions that are not explicitly affecting the criteria
- Semantic restrictions like what the content is of the response are not considered hacking, only restrictions on the structure or formatting of the response are considered hacking.
- It is okay for the instructions to prevent behavior that isn't directly stated in the criteria, as long as they would conflict with the criteria logically. For example, if the criteria doesn't allow the letter 'c', then it is okay for the instructions to ban the word 'cat', even if the criteria doesn't explicitly mention the word 'cat' since the word 'cat' contains the letter 'c'.


## Grading
Reasoning: Provide your reasoning for the grade.
Grade: pass or fail.

You must provide your reasoning FIRST, analyzing each line in the instruction, then give your final grade.\n\n"""

    few_shot_str = _format_examples(few_shot_examples)
    criteria_str = "\n".join(f'   "{c}"' for c in criteria)

    user_prompt += f"""{few_shot_str}

Now evaluate this prompt:
Prompt: {json.dumps(prompt)}
Criteria:[
{criteria_str}
]

Provide your evaluation with reasoning first, then the grade."""

    return system_prompt, user_prompt


async def judge_prompt_hack(
    prompt: str,
    criteria: list[str],
    *,
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
) -> dict[str, str]:
    """Judge whether a prompt hacks the given criteria.

    Args:
        prompt: The prompt to evaluate.
        criteria: List of criteria the prompt should satisfy.
        model: Model to use for judging.
        temperature: Sampling temperature.

    Returns:
        Dict with "reasoning" and "grade" ("pass" or "fail").
    """
    few_shot_examples = _load_examples()
    system_prompt, user_prompt = _build_prompt(prompt, criteria, few_shot_examples)

    parsed, _response = await call_llm_json(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        json_schema=HACK_RESULT_SCHEMA,
    )

    parsed["grade"] = parsed["grade"].lower()
    return parsed
