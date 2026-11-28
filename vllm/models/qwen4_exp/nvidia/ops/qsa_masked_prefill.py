# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked-dense QSA prefill on Blackwell (sm_100/sm_103).

On long prefill chunks QSA's sparse kernel runs far below Blackwell's
tensor-core throughput. FA4's causal attention over each request's whole
prefix, with every row masked to its own QSA selection by a bitmap mask_mod,
computes the same softmax faster up to ~12K keys (4x at 3K, 1.3x at 12K).
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.vllm_flash_attn.cute.utils import ssa_to_scalar

# Below this many rows per request the FA4 launch cost outweighs the gain;
# beyond this many keys the masked dense work outgrows the sparse kernel.
MIN_ROWS = 1024
MAX_KEYS = 12288


@dataclass
class MaskedPrefillPlan:
    """Prefill requests attended densely, and their rows and prefix keys."""

    rows: torch.Tensor  # int64 [num_rows], batch rows of the dense requests
    first_row: int | None  # set when rows is one contiguous run
    num_sparse_rows: int  # rows of the other requests
    key_req: torch.Tensor  # int64 [num_keys], block-table row of each key
    key_pos: torch.Tensor  # int64 [num_keys], request position of each key
    cu_rows: torch.Tensor  # int32 [num_dense_requests + 1]
    cu_keys: torch.Tensor  # int32 [num_dense_requests + 1]
    max_rows: int
    max_keys: int
    compress_ratio: int


def plan_masked_prefill(
    cm: CommonAttentionMetadata, compress_ratio: int, device: torch.device
) -> MaskedPrefillPlan | None:
    """Plan the requests with at least MIN_ROWS new rows and at most MAX_KEYS
    keys (all prefills, so their CPU seq_lens upper bounds are exact)."""
    if cm.max_query_len < MIN_ROWS or cm.seq_lens_cpu_upper_bound is None:
        return None
    query_start_loc = cm.query_start_loc_cpu[: cm.num_reqs + 1].numpy()
    seq_lens = cm.seq_lens_cpu_upper_bound[: cm.num_reqs].numpy()
    query_lens = np.diff(query_start_loc)
    reqs = np.flatnonzero((query_lens >= MIN_ROWS) & (seq_lens <= MAX_KEYS))
    if reqs.size == 0:
        return None
    row_lens, key_lens = query_lens[reqs], seq_lens[reqs]
    rows = np.concatenate(
        [np.arange(query_start_loc[r], query_start_loc[r + 1]) for r in reqs]
    )
    parts = (
        rows,
        np.repeat(reqs, key_lens),
        np.concatenate([np.arange(n) for n in key_lens]),
        np.concatenate(([0], np.cumsum(row_lens))),
        np.concatenate(([0], np.cumsum(key_lens))),
    )
    packed = torch.from_numpy(np.concatenate(parts).astype(np.int64))
    rows_d, key_req, key_pos, cu_rows, cu_keys = (
        packed.pin_memory().to(device, non_blocking=True).split([len(p) for p in parts])
    )
    return MaskedPrefillPlan(
        rows=rows_d,
        first_row=int(rows[0]) if rows[-1] - rows[0] + 1 == len(rows) else None,
        num_sparse_rows=cm.num_actual_tokens - len(rows),
        key_req=key_req,
        key_pos=key_pos,
        cu_rows=cu_rows.to(torch.int32),
        cu_keys=cu_keys.to(torch.int32),
        max_rows=int(row_lens.max()),
        max_keys=int(key_lens.max()),
        compress_ratio=compress_ratio,
    )


@triton.jit
def _selection_bitmap_kernel(
    selected_ptr,
    rows_ptr,
    bitmap_ptr,
    stride_selected,
    stride_bitmap,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A selection (expand_qsa_block_indices) is count - count % RATIO tokens
    # in whole aligned groups of RATIO, then a causal tail of count % RATIO
    # tokens from an aligned start. RATIO divides 32, so each group or tail
    # sets adjacent bits of one word.
    i = tl.program_id(0)
    row = tl.load(rows_ptr + i)
    selected = selected_ptr + row * stride_selected
    count = tl.load(selected + WIDTH)
    tail = count % RATIO
    groups = (count - tail) // RATIO
    out = bitmap_ptr + i.to(tl.int64) * stride_bitmap
    group_bits = tl.full((BLOCK,), (1 << RATIO) - 1, tl.int32)
    for start in range(0, groups, BLOCK):
        g = start + tl.arange(0, BLOCK)
        valid = g < groups
        first = tl.load(selected + g * RATIO, mask=valid, other=-1)
        valid &= first >= 0
        tl.atomic_or(
            out + first // 32, group_bits << (first % 32), mask=valid, sem="relaxed"
        )
    if tail > 0:
        first = tl.load(selected + groups * RATIO)
        bits = ((1 << tail) - 1) << (first % 32)
        tl.atomic_or(out + first // 32, bits, sem="relaxed")


@triton.jit
def _gather_prefix_kv_kernel(
    kv_cache_ptr,
    block_table_ptr,
    key_req_ptr,
    key_pos_ptr,
    out_ptr,
    stride_block,
    stride_head,
    stride_token,
    stride_table,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    ROW_SIZE: tl.constexpr,
):
    key = tl.program_id(0)
    req = tl.load(key_req_ptr + key)
    pos = tl.load(key_pos_ptr + key)
    block = tl.load(block_table_ptr + req * stride_table + pos // PAGE_SIZE)
    heads = tl.arange(0, NUM_KV_HEADS)[:, None]
    cols = tl.arange(0, ROW_SIZE)[None, :]
    kv = tl.load(
        kv_cache_ptr
        + block.to(tl.int64) * stride_block
        + heads * stride_head
        + (pos % PAGE_SIZE) * stride_token
        + cols
    )
    tl.store(out_ptr + (key.to(tl.int64) * NUM_KV_HEADS + heads) * ROW_SIZE + cols, kv)


@triton.jit
def _gate_scatter_kernel(
    attn_ptr,
    gate_ptr,
    rows_ptr,
    out_ptr,
    stride_gate,
    stride_out,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    row = tl.load(rows_ptr + i)
    for start in tl.static_range(0, ROW_SIZE, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        valid = cols < ROW_SIZE
        attn = tl.load(attn_ptr + i.to(tl.int64) * ROW_SIZE + cols, mask=valid)
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=valid)
        gated = attn.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))
        tl.store(
            out_ptr + row * stride_out + cols,
            gated.to(out_ptr.dtype.element_ty),
            mask=valid,
        )


@cute.jit
def _bitmap_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    # Bits of keys kv_idx .. kv_idx + 127 in the query's bitmap row. FA4 also
    # evaluates its tile's padding rows (q_idx >= seqlen_q, output dropped);
    # those read the request's last row.
    q = cutlass.min(ssa_to_scalar(q_idx), seqlen_info.seqlen_q - 1)
    row = aux_tensors[0][seqlen_info.offset_q + q, None]
    words = cute.make_rmem_tensor(4, cutlass.Int32)
    cute.autovec_copy(cute.local_tile(row, (4,), (ssa_to_scalar(kv_idx) >> 7,)), words)
    return cute.recast_tensor(words, cutlass.Uint32).load()


_bitmap_mask_mod.__vec_size__ = 128


def selection_bitmap(selected: torch.Tensor, plan: MaskedPrefillPlan) -> torch.Tensor:
    """Bit p of row i: batch row plan.rows[i] selected request position p."""
    rows = plan.rows
    words = triton.cdiv(plan.max_keys, 128) * 4  # whole 128-key mask_mod reads
    bitmap = torch.zeros((rows.numel(), words), dtype=torch.int32, device=rows.device)
    _selection_bitmap_kernel[(rows.numel(),)](
        selected,
        rows,
        bitmap,
        selected.stride(0),
        bitmap.stride(0),
        WIDTH=selected.shape[1] - 1,
        RATIO=plan.compress_ratio,
        BLOCK=512,
    )
    return bitmap


def masked_prefill_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    output_gate: torch.Tensor,
    out: torch.Tensor,
    plan: MaskedPrefillPlan,
    bitmap: torch.Tensor,
    scale: float,
) -> None:
    """out[plan.rows] = sigmoid(gate) * attention over each row's selection.

    kv_cache is QSA's [blocks, kv_heads, page_size, 2 * head_dim] view.
    """
    _, num_kv_heads, page_size, row_size = kv_cache.shape
    head_dim = row_size // 2
    num_keys = plan.key_pos.numel()
    kv = kv_cache.new_empty((num_keys, num_kv_heads, row_size))
    _gather_prefix_kv_kernel[(num_keys,)](
        kv_cache,
        block_table,
        plan.key_req,
        plan.key_pos,
        kv,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        PAGE_SIZE=page_size,
        NUM_KV_HEADS=num_kv_heads,
        ROW_SIZE=row_size,
    )
    rows = plan.rows
    if plan.first_row is not None:
        q = query[plan.first_row : plan.first_row + rows.numel()]
    else:
        q = query.index_select(0, rows)
    attn = flash_attn_varlen_func(
        q,
        kv[..., :head_dim],
        kv[..., head_dim:],
        max_seqlen_q=plan.max_rows,
        cu_seqlens_q=plan.cu_rows,
        max_seqlen_k=plan.max_keys,
        cu_seqlens_k=plan.cu_keys,
        softmax_scale=scale,
        causal=True,
        fa_version=4,
        mask_mod=_bitmap_mask_mod,
        aux_tensors=[bitmap],
    )
    gate = output_gate.flatten(1)
    assert out.stride(2) == 1 and out.stride(1) == head_dim
    _gate_scatter_kernel[(rows.numel(),)](
        attn,
        gate,
        rows,
        out,
        gate.stride(0),
        out.stride(0),
        ROW_SIZE=gate.shape[1],
        BLOCK=2048,
        num_warps=8,
    )
