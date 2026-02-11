"""Prompt quality judges — evaluate prompts for hacking and hygiene.

PrefPO includes two standalone LLM judges for assessing prompt quality:

  1. Hack detection — does a prompt add restrictions beyond what criteria
     require? For example, tightening "under 200 words" to "under 150 words"
     to game the evaluation. Returns pass/fail with reasoning.

  2. Hygiene scoring — how readable, well-specified, and maintainable is the
     prompt? Scores each axis 0-2 with per-axis reasoning.

These are useful as quality gates during optimization (e.g. reject prompts
that hack the criteria) or for post-hoc analysis of optimized prompts.

Both judges use openai/gpt-4.1 by default and accept a `model` parameter
for any litellm-supported provider.
"""

import asyncio

from prefpo.judges import judge_prompt_hack, judge_prompt_hygiene


async def main():
    # ── Hack Detection ──────────────────────────────────────────────────

    criteria = [
        "Response must have less than 200 words",
        "The keyword 'unfortunately' must appear at least 3 times (case-insensitive)",
        "The keyword 'unfortunately' must appear less than 6 times (case-insensitive)",
    ]

    # A clean prompt that faithfully represents the criteria (should pass)
    clean_prompt = (
        "Write a template for a newspaper ad for a dog cage with less than "
        "200 words. Make sure the word unfortunately appears 3 to 5 times in the ad."
    )

    # An over-restricted prompt that tightens the criteria (should fail)
    hacked_prompt = (
        "Write a template for a newspaper ad for a dog cage. Keep it under "
        "150 words. Use the word unfortunately exactly 3 times."
    )

    print("=" * 60)
    print("HACK DETECTION")
    print("=" * 60)

    for label, prompt in [("Clean prompt", clean_prompt), ("Hacked prompt", hacked_prompt)]:
        result = await judge_prompt_hack(prompt=prompt, criteria=criteria)
        print(f"\n{label}:")
        print(f"  Grade: {result['grade']}")
        print(f"  Reasoning: {result['reasoning']}")

    # ── Hygiene Scoring ─────────────────────────────────────────────────

    clean_prompt = (
        "Write a short blog post about a trip to Japan. Keep it under 300 words."
    )

    messy_prompt = (
        "Write a blog post about Japan. Must be under 300 words. Do NOT exceed "
        "300 words. Make sure to count your words. Target 250 words. Hard cap at "
        "280. Do not use markdown. Do not use bullet points. Do not use headers. "
        "Before submitting, verify your word count is under 300. Re-check that "
        "no markdown is present. Output only plain text."
    )

    print("\n" + "=" * 60)
    print("HYGIENE SCORING")
    print("=" * 60)

    # Without context — evaluates the prompt in isolation
    print("\nClean prompt (no context):")
    result = await judge_prompt_hygiene(prompt=clean_prompt)
    print(f"  Readability:      {result['readability_score']}/2")
    print(f"  Spec Quality:     {result['spec_quality_score']}/2")
    print(f"  Maintainability:  {result['maintainability_score']}/2")

    print("\nMessy prompt (no context):")
    result = await judge_prompt_hygiene(prompt=messy_prompt)
    print(f"  Readability:      {result['readability_score']}/2")
    print(f"  Spec Quality:     {result['spec_quality_score']}/2")
    print(f"  Maintainability:  {result['maintainability_score']}/2")

    # With context — the judge won't penalize for instructions required
    # by the context, only evaluates how well they're expressed
    print("\nClean prompt (with context):")
    result = await judge_prompt_hygiene(
        prompt=clean_prompt,
        context="Response must have less than 300 words",
    )
    print(f"  Readability:      {result['readability_score']}/2")
    print(f"  Spec Quality:     {result['spec_quality_score']}/2")
    print(f"  Maintainability:  {result['maintainability_score']}/2")
    print(f"  Overall: {result['overall_reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())
