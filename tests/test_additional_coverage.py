"""Additional unit tests to cover uncovered PrefPO behaviors."""

import asyncio
import json
from pathlib import Path

import pytest
from prefpo.config import ModelConfig, PrefPOConfig
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader
from prefpo.grading.base import GradeResult, Grader
from prefpo.llm import client as llm_client
from prefpo.optimize import optimize_async
from prefpo.pool import PromptPool
from prefpo.prompts.discriminator import build_standalone_trajectory
from prefpo.types import ModelOutput, Prompt, PromptRole, Sample
class _NoOpGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        return GradeResult(score=0.0, n=1)

    def check_output(self, output, prompt_text=None):
        return None
@pytest.mark.asyncio
async def test_call_llm_accepts_system_for_reasoning():
    response = await llm_client.call_llm(
        model="openai/gpt-5",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        is_reasoning=True,
    )
    assert isinstance(response.output_text, str)
@pytest.mark.asyncio
async def test_call_llm_accepts_system_for_non_reasoning():
    response = await llm_client.call_llm(
        model="openai/gpt-4o",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        is_reasoning=False,
    )
    assert isinstance(response.output_text, str)
@pytest.mark.asyncio
async def test_call_llm_json_parses_schema():
    parsed, resp = await llm_client.call_llm_json(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Return a JSON object with {\"ok\": true}."}],
        is_reasoning=False,
        temperature=0.0,
        json_schema={
            "type": "json_schema",
            "name": "ok_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    )

    assert parsed["ok"] is True
    assert isinstance(resp.output_text, str)
@pytest.mark.asyncio
async def test_generate_outputs_prompt_sent_contains_all_messages():
    from prefpo.generate import generate_outputs

    p = Prompt(value="You are helpful.", role=PromptRole.SYSTEM)
    s = Sample(index=0, question="What is 2+2?")

    out = await generate_outputs(
        prompt=p,
        samples=[s],
        model_config=ModelConfig(name="openai/gpt-4o", is_reasoning=False, reasoning_effort="medium", temperature=0.0),
        semaphore=asyncio.Semaphore(1),
    )

    assert "[system] You are helpful." in out[0].prompt_sent
    assert "[user] What is 2+2?" in out[0].prompt_sent
def test_build_standalone_trajectory_passes_prompt_text():
    class _Grader:
        def check_output(self, output, prompt_text=None):
            assert prompt_text == "PROMPT"
            return {"ok": True}

    outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="hi")]
    traj = build_standalone_trajectory(outputs, _Grader(), show_target=True, prompt_text="PROMPT")
    assert "Grade:" in traj
def test_build_standalone_trajectory_requires_check_output():
    class _Grader:
        def check_output(self, output, prompt_text=None):
            return None

    outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="hi")]
    with pytest.raises(ValueError):
        build_standalone_trajectory(outputs, _Grader(), show_target=True, prompt_text="PROMPT")
def test_pool_score_cache_respects_role():
    pool = PromptPool([
        Prompt(value="same", role=PromptRole.USER),
        Prompt(value="other", role=PromptRole.USER),
    ])
    pool.set_score(pool.entries[0], 0.9)
    system_prompt = Prompt(value="same", role=PromptRole.SYSTEM, name="sys")
    assert pool.get_score(system_prompt) is None
def test_pool_replace_non_preferred_raises_when_missing():
    pool = PromptPool([
        Prompt(value="a", role=PromptRole.USER),
        Prompt(value="b", role=PromptRole.USER),
    ])
    with pytest.raises(ValueError):
        pool.replace_non_preferred(
            Prompt(value="c", role=PromptRole.USER, name="improved"),
            "seed_0",
            "missing",
            preferred=1,
        )
@pytest.mark.asyncio
async def test_optimize_standalone_requires_check_output(tmp_path):
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        discriminator={"show_target": True},
        pool={"initial_prompts": ["test prompt"]},
        run={"iterations": 1, "output_dir": str(tmp_path)},
    )

    with pytest.raises(ValueError, match="check_output"):
        await optimize_async(config, grader=_NoOpGrader())
@pytest.mark.asyncio
async def test_optimize_writes_multi_trial_summary(tmp_path):
    train, val, test = load_bbh(
        "disambiguation_qa",
        train_size=1,
        val_size=1,
        test_size=1,
        seed=42,
    )
    grader = get_bbh_grader("disambiguation_qa")
    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["p1", "p2"]},
        run={"iterations": 1, "n_trials": 2, "output_dir": str(tmp_path), "max_concurrent": 10},
    )

    await optimize_async(config, grader=grader, train=train, val=val, test=test)

    summaries = list(Path(tmp_path).glob("multi_trial_summary_*.json"))
    assert len(summaries) == 1
    data = json.loads(summaries[0].read_text())
    assert data["n_trials"] == 2
