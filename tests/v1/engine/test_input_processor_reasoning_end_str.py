# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend handling of the per-request `reasoning_end_str` override."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm import SamplingParams
from vllm.config.reasoning import MAX_REQUEST_REASONING_END_TOKENS
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

THINK_END = 91
_ENCODINGS = {
    "</think>": [THINK_END],
    "Time to answer.</think>": [10, 11, 12, THINK_END],
    "Time to answer.": [10, 11, 12],
    # The marker merges with the preceding text into a single token.
    "answer</think>": [13],
    "</think>Time to answer.</think>": [THINK_END, 10, 11, 12, THINK_END],
}


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        if text.startswith("long:"):
            return [10] * int(text.removeprefix("long:")) + [THINK_END]
        return _ENCODINGS[text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        assert list(ids) == [THINK_END]
        return "</think>"


def _reasoning_config():
    return SimpleNamespace(
        enabled=True,
        reasoning_start_token_ids=[90],
        reasoning_end_token_ids=[THINK_END],
        natural_reasoning_end_token_ids=[THINK_END],
    )


def _tokenize(reasoning_end_str: str, tokenizer=None) -> SamplingParams:
    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(reasoning_config=_reasoning_config()),
        tokenizer=_Tokenizer() if tokenizer is None else tokenizer,
    )
    params = SamplingParams(
        thinking_token_budget=8, reasoning_end_str=reasoning_end_str
    )
    InputProcessor._tokenize_reasoning_end_str(processor, params)
    return params


@pytest.mark.parametrize(
    "text,expected",
    [
        ("</think>", [THINK_END]),
        ("Time to answer.</think>", [10, 11, 12, THINK_END]),
    ],
)
def test_reasoning_end_str_is_tokenized(text, expected):
    assert _tokenize(text).reasoning_end_token_ids == expected


@pytest.mark.parametrize(
    "text",
    [
        "Time to answer.",  # missing the marker
        "answer</think>",  # marker merged into the preceding token
        "</think>Time to answer.</think>",  # marker before the end
    ],
)
def test_reasoning_end_str_must_end_with_reasoning_end_marker(text):
    with pytest.raises(VLLMValidationError, match="must end with the reasoning end"):
        _tokenize(text)


def test_reasoning_end_str_length_is_bounded():
    params = _tokenize(f"long:{MAX_REQUEST_REASONING_END_TOKENS - 1}")
    assert len(params.reasoning_end_token_ids) == MAX_REQUEST_REASONING_END_TOKENS
    with pytest.raises(VLLMValidationError, match="maximum is"):
        _tokenize(f"long:{MAX_REQUEST_REASONING_END_TOKENS}")


def test_reasoning_end_str_requires_tokenizer():
    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(reasoning_config=_reasoning_config()),
        tokenizer=None,
    )
    params = SamplingParams(reasoning_end_str="</think>")
    with pytest.raises(VLLMValidationError, match="requires a tokenizer"):
        InputProcessor._tokenize_reasoning_end_str(processor, params)


def test_reasoning_end_str_rejects_empty_string():
    with pytest.raises(VLLMValidationError, match="non-empty"):
        SamplingParams(reasoning_end_str="")


@pytest.mark.parametrize("reasoning_config", [None, SimpleNamespace(enabled=False)])
def test_reasoning_end_str_requires_reasoning_config(reasoning_config):
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            return_sampling_mask=False,
            enable_trace_replay=False,
            is_diffusion=False,
        ),
        vllm_config=SimpleNamespace(
            reasoning_config=reasoning_config,
            _check_supports_watermarking=lambda params: False,
        ),
        resolve_watermarking=lambda params: False,
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=None,
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
    )
    params = SamplingParams(reasoning_end_str="</think>")
    with (
        patch.object(SamplingParams, "verify"),
        pytest.raises(VLLMValidationError, match="reasoning_end_str is set"),
    ):
        InputProcessor._validate_params(processor, params, ("generate",))
