# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend validation of the per-request `reasoning_end_str` override."""

from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.config.reasoning import MAX_REQUEST_REASONING_END_TOKENS
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

THINK_START = 90
THINK_END = 91
VOCAB_SIZE = 100
_ENCODINGS = {
    "</think>": [THINK_END],
    "Time to answer.</think>": [10, 11, 12, THINK_END],
    "Time to answer.": [10, 11, 12],
    # The marker merges with the preceding text into a single token.
    "answer</think>": [13],
    "</think>Time to answer.</think>": [THINK_END, 10, 11, 12, THINK_END],
    "<think></think>": [THINK_START, THINK_END],
    # A tokenizer id the model's LM head does not cover.
    "added</think>": [VOCAB_SIZE, THINK_END],
}


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        if text.startswith("long:"):
            return [10] * int(text.removeprefix("long:")) + [THINK_END]
        return _ENCODINGS[text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return {THINK_START: "<think>", THINK_END: "</think>"}[ids[0]]


def _tokenize(reasoning_end_str: str, enabled: bool | None = True) -> SamplingParams:
    reasoning_config = (
        None
        if enabled is None
        else SimpleNamespace(
            enabled=enabled,
            reasoning_start_token_ids=[THINK_START],
            natural_reasoning_end_token_ids=[THINK_END],
        )
    )
    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(reasoning_config=reasoning_config),
        model_config=SimpleNamespace(get_vocab_size=lambda: VOCAB_SIZE),
        tokenizer=_Tokenizer(),
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
    "text,error",
    [
        ("Time to answer.", "must end with the reasoning end marker"),
        # Marker merged into the preceding token.
        ("answer</think>", "must end with the reasoning end marker"),
        # Forcing would stop at the first marker.
        ("</think>Time to answer.</think>", "must end with the reasoning end marker"),
        # A forced start marker restarts the budget: reasoning never closes.
        ("<think></think>", "must not contain the reasoning start marker"),
        # Would be used as an out-of-bounds logits column in the sampler.
        ("added</think>", "outside the model vocabulary"),
        (f"long:{MAX_REQUEST_REASONING_END_TOKENS}", "maximum is"),
    ],
)
def test_reasoning_end_str_is_rejected(text, error):
    with pytest.raises(VLLMValidationError, match=error):
        _tokenize(text)


@pytest.mark.parametrize("enabled", [None, False])
def test_reasoning_end_str_requires_reasoning_config(enabled):
    with pytest.raises(VLLMValidationError, match="reasoning_config is not"):
        _tokenize("</think>", enabled=enabled)


def test_reasoning_end_str_rejects_empty_string():
    with pytest.raises(VLLMValidationError, match="non-empty"):
        SamplingParams(reasoning_end_str="")
