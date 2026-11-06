# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend validation of the per-request `reasoning_end_str` override."""

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import tokens_input
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

MODEL = "Qwen/Qwen3-0.6B"
THINK_END_ID = 151668  # `</think>` in the Qwen3 vocabulary


def _processor(reasoning_config: ReasoningConfig | None) -> InputProcessor:
    return InputProcessor(
        VllmConfig(
            model_config=ModelConfig(model=MODEL, max_model_len=256),
            reasoning_config=reasoning_config,
        )
    )


@pytest.fixture(scope="module")
def processor() -> InputProcessor:
    return _processor(ReasoningConfig(reasoning_parser="qwen3"))


def _process(processor: InputProcessor, reasoning_end_str: str) -> SamplingParams:
    params = SamplingParams(
        thinking_token_budget=8, reasoning_end_str=reasoning_end_str
    )
    request = processor.process_inputs(
        "req", tokens_input([1, 2, 3]), params, ("generate",)
    )
    assert request.sampling_params is not None
    return request.sampling_params


def test_reasoning_end_str_is_tokenized(processor):
    token_ids = _process(processor, "Time to answer.</think>").reasoning_end_token_ids
    assert token_ids is not None
    assert token_ids[-1] == THINK_END_ID
    assert THINK_END_ID not in token_ids[:-1]


@pytest.mark.parametrize(
    "reasoning_end_str,error",
    [
        ("Time to answer.", "must end with the reasoning end marker"),
        # Forcing would stop at the first marker.
        ("</think>Time to answer.</think>", "must end with the reasoning end marker"),
        # A forced start marker restarts the budget: reasoning never closes.
        ("<think>Time to answer.</think>", "must not contain the reasoning start"),
        ("go " * 200 + "</think>", "maximum is"),
    ],
)
def test_reasoning_end_str_is_rejected(processor, reasoning_end_str, error):
    with pytest.raises(VLLMValidationError, match=error):
        _process(processor, reasoning_end_str)


def test_reasoning_end_str_requires_reasoning_config():
    with pytest.raises(VLLMValidationError, match="reasoning_config is not"):
        _process(_processor(None), "Time to answer.</think>")


def test_reasoning_end_str_rejects_empty_string():
    with pytest.raises(VLLMValidationError, match="non-empty"):
        SamplingParams(reasoning_end_str="")
