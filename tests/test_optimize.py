"""Integration tests for optimize() with fully mocked LLM calls."""

import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prefpo.config import PrefPOConfig
from prefpo.grading.base import GradeResult, Grader
from prefpo.optimize import optimize_async
from prefpo.types import ModelOutput, Prompt, PromptRole, Sample
class MockGrader(Grader):
    """Grader that returns a fixed score without making API calls."""

    def __init__(self, score: float = 0.7):
        self._score = score
        self._call_count = 0

    async def grade(self, prompt, samples, model_config, semaphore):
        self._call_count += 1
        return GradeResult(score=self._score, n=len(samples) if samples else 1)

    def check_output(self, output, prompt_text=None):
        return {"passed": True}
def _make_mock_response(output_text: str, response_id: str = "resp_test"):
    mock = MagicMock()
    mock.output_text = output_text
    mock.id = response_id
    mock.usage = MagicMock()
    mock.usage.model_dump.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return mock
@pytest.mark.asyncio
async def test_optimize_instruction_mode():
    """Full optimize() with instruction mode and mocked LLM calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": ["prompt A", "prompt B"]},
            run={"iterations": 1, "max_concurrent": 10, "output_dir": tmpdir},
        )

        train = [Sample(index=0, question="Q1", target="A")]
        val = [Sample(index=0, question="Q1", target="A")]
        grader = MockGrader(score=0.8)

        # Mock generate_outputs to return canned results
        mock_outputs = [ModelOutput(sample_index=0, prompt_sent="p", response="ANSWER: A")]

        disc_json = json.dumps({"preferred": 1, "feedback": "good"})
        opt_json = json.dumps({"prompt": "improved prompt"})

        disc_resp = _make_mock_response(disc_json, "resp_disc")
        opt_resp = _make_mock_response(opt_json, "resp_opt")

        with patch("prefpo.optimize.generate_outputs", new_callable=AsyncMock, return_value=mock_outputs), \
             patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
                 json.loads(disc_json),
                 [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": disc_json}],
                 disc_resp,
             )), \
             patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
                 json.loads(opt_json), opt_resp,
             )):

            result = await optimize_async(config, grader=grader, train=train, val=val)

        assert result.run_id.startswith("run_")
        assert result.best_score == 0.8
        assert len(result.history) == 1
        assert len(result.final_pool["prompts"]) >= 2
@pytest.mark.asyncio
async def test_optimize_single_prompt_instruction():
    """Single prompt in instruction mode should generate a variant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": ["only prompt"]},
            run={"iterations": 1, "max_concurrent": 10, "output_dir": tmpdir},
        )

        train = [Sample(index=0, question="Q", target="A")]
        grader = MockGrader(score=0.5)

        mock_outputs = [ModelOutput(sample_index=0, prompt_sent="p", response="resp")]
        disc_resp = _make_mock_response(json.dumps({"preferred": 1, "feedback": "ok"}), "disc")
        opt_resp = _make_mock_response(json.dumps({"prompt": "new prompt"}), "opt")

        with patch("prefpo.optimize.generate_outputs", new_callable=AsyncMock, return_value=mock_outputs), \
             patch("prefpo.optimize.generate_prompt_variant", new_callable=AsyncMock, return_value="variant prompt"), \
             patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
                 {"preferred": 1, "feedback": "ok"},
                 [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": json.dumps({"preferred": 1, "feedback": "ok"})}],
                 disc_resp,
             )), \
             patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
                 {"prompt": "new prompt"}, opt_resp,
             )):

            result = await optimize_async(config, grader=grader, train=train)

        # Pool should have at least 3: seed_0, variant_0, improved_0
        assert len(result.final_pool["prompts"]) >= 3
@pytest.mark.asyncio
async def test_optimize_standalone_mode():
    """Standalone mode with mocked LLM calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PrefPOConfig(
            mode="standalone",
            task_model={"name": "openai/gpt-4o"},
            discriminator={"show_expected": True, "criteria": ["test criterion"]},
            pool={"initial_prompts": ["Write a poem", "Write a nice poem"]},
            run={"iterations": 1, "max_concurrent": 10, "output_dir": tmpdir},
        )

        grader = MockGrader(score=0.6)

        mock_outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="A nice poem")]
        disc_resp = _make_mock_response(json.dumps({"preferred": 2, "feedback": "better"}), "disc")
        opt_resp = _make_mock_response(json.dumps({"prompt": "Write a beautiful poem"}), "opt")

        with patch("prefpo.optimize.generate_standalone", new_callable=AsyncMock, return_value=mock_outputs), \
             patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
                 {"preferred": 2, "feedback": "better"},
                 [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": json.dumps({"preferred": 2, "feedback": "better"})}],
                 disc_resp,
             )), \
             patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
                 {"prompt": "Write a beautiful poem"}, opt_resp,
             )):

            result = await optimize_async(config, grader=grader)

        assert result.best_score == 0.6
        assert result.best_test_score is None  # standalone has no test set
        assert len(result.history) == 1
@pytest.mark.asyncio
async def test_optimize_instruction_mode_requires_train():
    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
    )
    grader = MockGrader()
    with pytest.raises(ValueError, match="train"):
        await optimize_async(config, grader=grader)
@pytest.mark.asyncio
async def test_optimize_standalone_rejects_train():
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"]},
    )
    grader = MockGrader()
    train = [Sample(index=0, question="Q", target="A")]
    with pytest.raises(ValueError, match="Standalone"):
        await optimize_async(config, grader=grader, train=train)
@pytest.mark.asyncio
async def test_optimize_standalone_rejects_system_role():
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["test"], "prompt_role": "system"},
    )
    grader = MockGrader()
    with pytest.raises(ValueError, match="prompt_role"):
        await optimize_async(config, grader=grader)
# --- Additional edge-case tests with real API calls ---

from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader
@pytest.mark.asyncio
@pytest.mark.live
async def test_optimize_val_none_scores_on_train():
    """When val=None, grader uses train samples for scoring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train, _, _ = load_bbh("disambiguation_qa", train_size=3, val_size=1, seed=42)
        grader = get_bbh_grader("disambiguation_qa")
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": [
                "Answer the question with the correct letter. End with ANSWER: <letter>.",
                "Think step by step. End with ANSWER: <letter>.",
            ]},
            run={"iterations": 1, "max_concurrent": 10, "output_dir": tmpdir},
        )
        result = await optimize_async(config, grader=grader, train=train, val=None)
        assert result.best_score >= 0.0
        assert result.best_score <= 1.0
        assert len(result.history) == 1
@pytest.mark.asyncio
@pytest.mark.live
async def test_optimize_with_test_set():
    """best_test_score populated when test set is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train, val, test = load_bbh("disambiguation_qa", train_size=3, val_size=3, test_size=5, seed=42)
        grader = get_bbh_grader("disambiguation_qa")
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": [
                "Answer the question. End with ANSWER: <letter>.",
                "Think carefully. End with ANSWER: <letter>.",
            ]},
            run={"iterations": 1, "max_concurrent": 10, "output_dir": tmpdir},
        )
        result = await optimize_async(config, grader=grader, train=train, val=val, test=test)
        assert result.best_test_score is not None
        assert 0.0 <= result.best_test_score <= 1.0
@pytest.mark.asyncio
@pytest.mark.live
async def test_optimize_replace_strategy():
    """Pool stays at 2 entries with replace strategy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train, val, _ = load_bbh("disambiguation_qa", train_size=3, val_size=3, seed=42)
        grader = get_bbh_grader("disambiguation_qa")
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={
                "initial_prompts": [
                    "Answer the question. End with ANSWER: <letter>.",
                    "Think step by step. End with ANSWER: <letter>.",
                ],
                "update_strategy": "replace",
            },
            run={"iterations": 2, "max_concurrent": 10, "output_dir": tmpdir},
        )
        result = await optimize_async(config, grader=grader, train=train, val=val)
        assert len(result.final_pool["prompts"]) == 2
@pytest.mark.asyncio
async def test_optimize_standalone_empty_prompt_rejected():
    """Raises ValueError for empty initial prompt in standalone mode."""
    config = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        pool={"initial_prompts": ["  "]},
    )
    grader = MockGrader()
    with pytest.raises(ValueError, match="non-empty"):
        await optimize_async(config, grader=grader)
@pytest.mark.asyncio
@pytest.mark.live
async def test_optimize_multi_trial_dispatch():
    """n_trials>1 dispatches to optimize_multi_trial and writes summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train, val, _ = load_bbh("disambiguation_qa", train_size=3, val_size=3, seed=42)
        grader = get_bbh_grader("disambiguation_qa")
        config = PrefPOConfig(
            mode="instruction",
            task_model={"name": "openai/gpt-4o"},
            pool={"initial_prompts": [
                "Answer the question. End with ANSWER: <letter>.",
                "Think carefully. End with ANSWER: <letter>.",
            ]},
            run={"iterations": 1, "n_trials": 2, "max_concurrent": 10, "output_dir": tmpdir},
        )
        result = await optimize_async(config, grader=grader, train=train, val=val)

        from pathlib import Path
        summaries = list(Path(tmpdir).glob("multi_trial_summary_*.json"))
        assert len(summaries) == 1
        data = json.loads(summaries[0].read_text())
        assert data["n_trials"] == 2

@pytest.mark.asyncio
async def test_system_prompt_with_system_role_raises():
    """system_prompt + prompt_role='system' is invalid — raises ValueError."""
    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o", "system_prompt": "You are helpful."},
        pool={"initial_prompts": ["test"], "prompt_role": "system"},
    )
    grader = MockGrader()
    train = [Sample(index=0, question="Q", target="A")]
    with pytest.raises(ValueError, match="Cannot set task_model.system_prompt"):
        await optimize_async(config, grader=grader, train=train)
