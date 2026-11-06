# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import torch

from tests.v1.sample.utils import create_mock_reasoning_config
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        create_mock_reasoning_config([151667], [151668]),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


START = 100
END = 200
PHRASE_A = 201
PHRASE_B = 202
THINK = 60
CONTENT = 50
VOCAB_SIZE = 256


def _generate(
    holder: ThinkingBudgetStateHolder,
    params: list[SamplingParams],
    scripted: list[list[int]],
    num_steps: int,
) -> list[list[int]]:
    """Greedily decode: take a forced token if any, else the scripted token."""
    outputs: list[list[int]] = [[] for _ in params]
    holder.sync_batch(
        BatchUpdate(
            batch_size=len(params),
            removed=(),
            added=[(i, p, None, outputs[i]) for i, p in enumerate(params)],
            moved=(),
        )
    )
    for step in range(num_steps):
        holder.update_state(output_token_ids=outputs, spec_token_ids=None)
        logits = torch.zeros((len(params), VOCAB_SIZE))
        holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
        for i, output in enumerate(outputs):
            if logits[i].max() > 1.0e8:
                output.append(int(logits[i].argmax()))
            else:
                output.append(scripted[i][step] if step < len(scripted[i]) else THINK)
    return outputs


def test_per_request_reasoning_end_str_overrides_forced_sequence():
    """Requests in one batch force their own end sequence, or the default."""
    holder = ThinkingBudgetStateHolder(
        create_mock_reasoning_config([START], [END]),
        8,
        0,
        torch.device("cpu"),
        False,
    )
    override = SamplingParams(thinking_token_budget=3)
    override._reasoning_end_token_ids = [PHRASE_A, PHRASE_B, END]
    thinking = [START] + [THINK] * 10

    default_out, override_out = _generate(
        holder,
        [SamplingParams(thinking_token_budget=3), override],
        [thinking, thinking],
        num_steps=8,
    )

    assert default_out[:5] == [START, THINK, THINK, THINK, END]
    assert override_out[:7] == [START, THINK, THINK, THINK, PHRASE_A, PHRASE_B, END]
    # Forcing stops once the end sequence is complete.
    assert override_out[7] == THINK


def test_natural_end_detected_when_forced_sequence_has_transition_phrase():
    """A bare natural end marker ends reasoning even though the forced
    sequence carries a transition phrase, so the phrase is never injected
    into the answer once the budget would have run out."""
    holder = ThinkingBudgetStateHolder(
        create_mock_reasoning_config(
            [START], [PHRASE_A, PHRASE_B, END], natural_end_token_ids=[END]
        ),
        8,
        0,
        torch.device("cpu"),
        False,
    )

    (output,) = _generate(
        holder,
        [SamplingParams(thinking_token_budget=4)],
        [[START, THINK, THINK, END] + [CONTENT] * 12],
        num_steps=14,
    )

    assert output == [START, THINK, THINK, END] + [CONTENT] * 10
