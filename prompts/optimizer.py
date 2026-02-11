"""Optimizer prompt builder — asks the optimizer to improve the non-preferred prompt."""

from prefpo.config import OptimizerConfig
from prefpo.types import Prompt


def _format_constraints_block(constraints: str | list[str]) -> str:
    """Format constraints as a bulleted block."""
    if not constraints:
        return ""
    if isinstance(constraints, str):
        items = [constraints]
    else:
        items = constraints
    bullets = "\n".join(f"- {item}" for item in items)
    return f"\nCONSTRAINTS FOR YOUR OUTPUT:\n{bullets}\n"


def build_optimizer_prompt(
    preferred: int,
    non_preferred_prompt: Prompt,
    feedback: str,
    config: OptimizerConfig,
) -> str:
    """Returns the user prompt for the optimizer.

    No system prompt — the optimizer chains via previous_response_id
    from the discriminator call.
    """
    version_label = "Version 1" if preferred == 2 else "Version 2"

    parts: list[str] = []
    parts.append(f"{version_label} Instruction: {non_preferred_prompt.value}")
    parts.append(f"\nFeedback: {feedback}")
    parts.append(
        "\n<Task>Produce an improved instruction for the non-preferred "
        "instruction based on the feedback.</Task>"
    )

    constraints_block = _format_constraints_block(config.constraints)
    if constraints_block:
        parts.append(constraints_block)

    parts.append(
        '<Output>The output should be a JSON object with the following fields: '
        '"prompt": string.</Output>'
    )

    return "\n".join(parts)


OPTIMIZER_SCHEMA = {
    "type": "json_schema",
    "name": "optimizer_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}
