"""Gap coverage suite for CLI, loaders, and error paths."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from prefpo import cli
from prefpo.config import ModelConfig, PrefPOConfig
from prefpo.data import bbh as bbh_data
from prefpo.data import ifeval as ifeval_data
from prefpo.llm import client as llm_client
from prefpo.prompts.variant import generate_prompt_variant
from prefpo.types import Prompt, PromptRole, Sample
@pytest.mark.asyncio
async def test_cli_bbh_invokes_optimize(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "instruction",
                "task_model": {"name": "openai/gpt-4o"},
                "pool": {"initial_prompts": ["p1", "p2"]},
                "run": {"iterations": 1, "output_dir": str(tmp_path)},
            }
        )
    )

    train = [Sample(index=0, question="Q", target="A")]
    val = [Sample(index=0, question="Q", target="A")]
    test = [Sample(index=0, question="Q", target="A")]
    monkeypatch.setattr(cli, "load_bbh", lambda *args, **kwargs: (train, val, test))
    monkeypatch.setattr(cli, "get_bbh_grader", lambda *args, **kwargs: MagicMock())

    result = MagicMock()
    result.run_id = "run_test"
    result.best_score = 0.5
    result.best_test_score = None
    result.best_prompt = Prompt(value="x", role=PromptRole.USER, name="seed_0")
    optimize_mock = AsyncMock(return_value=result)
    monkeypatch.setattr(cli, "optimize_async", optimize_mock)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--config",
            str(config_path),
            "--dataset",
            "bbh",
            "--subset",
            "disambiguation_qa",
            "--train-size",
            "1",
            "--val-size",
            "1",
            "--test-size",
            "1",
        ],
    )

    await cli.main()
    optimize_mock.assert_awaited_once()
def test_load_bbh_raises_when_size_exceeds_total(monkeypatch):
    dataset = [{"question": "Q", "target": "A"}] * 2
    monkeypatch.setattr(bbh_data.datasets, "load_dataset", lambda *args, **kwargs: dataset)

    with pytest.raises(ValueError, match="only has"):
        bbh_data.load_bbh("disambiguation_qa", train_size=2, val_size=1, test_size=0)
def test_load_ifeval_sample_out_of_range(monkeypatch):
    monkeypatch.setattr(ifeval_data, "_get_dataset", lambda: [{"key": "k"}])

    with pytest.raises(IndexError):
        ifeval_data.load_ifeval_sample(5)
def test_load_ifeval_sample_kwargs_length_mismatch(monkeypatch):
    sample = {
        "key": "k",
        "prompt": "p",
        "instruction_id_list": ["a", "b"],
        "kwargs": [{"x": 1}],
    }
    monkeypatch.setattr(ifeval_data, "_get_dataset", lambda: [sample])

    with pytest.raises(IndexError):
        ifeval_data.load_ifeval_sample(0)
@pytest.mark.asyncio
async def test_call_llm_json_parse_error(monkeypatch):
    from prefpo.llm.client import LLMResponse

    bad = LLMResponse(output_text="not json", id="test", usage=None)
    call_mock = AsyncMock(side_effect=[bad, bad])
    monkeypatch.setattr(llm_client, "call_llm", call_mock)

    with pytest.raises(json.JSONDecodeError):
        await llm_client.call_llm_json(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            parse_retries=1,
        )
@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_prompt_variant_real_api():
    text = await generate_prompt_variant(
        original_prompt="Write a haiku about clouds.",
        criteria=["3 lines", "no rhyming"],
        model_config=ModelConfig(name="openai/gpt-4o", is_reasoning=False, temperature=0.0),
        semaphore=asyncio.Semaphore(1),
    )
    assert text.strip()
