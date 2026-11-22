# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inkling 2 Mini's temporal operators on an AsyncGRPO TreeRow."""

from contextvars import ContextVar
from functools import wraps

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.flex_attention import compile_friendly_flex_attention
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.inkling2_loop import modeling_inkling2_loop as inkling


INKLING_TREE_ATTENTION = "trl_inkling_tree"
_building_row = ContextVar("trl_inkling_building_row", default=False)
_ordinary_distance = inkling.get_relative_distance


class InklingTree:
    """Shared ancestry for convolutions and full/sliding attention. Built once per packed forward."""

    def __init__(self, position_ids, tree, kernel_size):
        device = position_ids.device
        self.positions = position_ids[0]
        self.offsets = tree["tree_cu_seqlens_q"].tolist()
        self.k_offsets = tree["tree_cu_seqlens_k"].tolist()
        self.kv = tree["tree_kv_index"]
        self.parents = tree["tree_segment_parents"].tolist()
        self.masks = {}
        sizes = [b - a for a, b in zip(self.offsets[:-1], self.offsets[1:], strict=True)]
        ends = self.offsets[1:].copy()
        halos, retained = [], []
        for segment, size in enumerate(sizes):
            start = self.offsets[segment]
            parent = self.parents[segment]
            tail = []
            while parent >= 0 and len(tail) < kernel_size - 1:
                end = self.offsets[parent + 1]
                lo = max(self.offsets[parent], end - (kernel_size - 1 - len(tail)))
                tail = list(range(lo, end)) + tail
                parent = self.parents[parent]
            halos.extend(
                (
                    torch.tensor([-1] * (kernel_size - 1 - len(tail)) + tail, dtype=torch.long),
                    torch.arange(start, start + size),
                )
            )
            first = start + (segment + 1) * (kernel_size - 1)
            retained.append(torch.arange(first, first + size))
        for segment in reversed(range(len(sizes))):
            parent = self.parents[segment]
            if parent >= 0:
                ends[parent] = max(ends[parent], ends[segment])
        self.subtree_ends = torch.repeat_interleave(torch.tensor(ends), torch.tensor(sizes)).to(device)
        indices = torch.cat(halos)
        self.halo_zeros = torch.where(indices < 0)[0].to(device)
        self.with_halos = indices.clamp_min(0).to(device)
        self.keep = torch.cat(retained).to(device)

    def expand(self, states):
        # Discarding the halo outputs insulates neighboring segments in one native convolution.
        return states.index_select(1, self.with_halos).index_fill(1, self.halo_zeros, 0)

    def block_mask(self, window):
        if window in self.masks:
            return self.masks[window]
        tile = 128
        n = self.offsets[-1]
        q_tiles, k_tiles = (n + tile - 1) // tile, (n + tile) // tile
        # The final zero-value key is the learned sink. Each query attends to it.
        candidates = [{n // tile} for _ in range(q_tiles)]
        for seg, parent in enumerate(self.parents):
            start, end = self.offsets[seg : seg + 2]
            for q_block in range(start // tile, (end - 1) // tile + 1):
                first_q = max(start, q_block * tile)
                # Follow the segment's ancestors, stopping at the sliding-window boundary.
                left = window - (first_q - start) - 1 if window else None
                candidates[q_block].update(
                    range(max(start, first_q - window + 1 if window else start) // tile, q_block + 1)
                )
                ancestor = parent
                while ancestor >= 0 and (left is None or left > 0):
                    lo, hi = self.offsets[ancestor : ancestor + 2]
                    earliest = max(lo, hi - left) if left is not None else lo
                    candidates[q_block].update(range(earliest // tile, (hi - 1) // tile + 1))
                    if left is not None:
                        left -= hi - lo
                    ancestor = self.parents[ancestor]
        counts = torch.tensor([len(blocks) for blocks in candidates], dtype=torch.int32)
        columns = torch.zeros((q_tiles, k_tiles), dtype=torch.int32)
        for row, blocks in enumerate(candidates):
            columns[row, : len(blocks)] = torch.tensor(sorted(blocks), dtype=torch.int32)
        ends, pos = self.subtree_ends, self.positions

        def permitted(b, h, q, k):
            # Inductor can evaluate padded lanes, including the explicit sink.
            qi, ki = q.clamp(max=n - 1), k.clamp(max=n - 1)
            causal = (k <= q) & (q < ends[ki])
            if window:
                causal = causal & (pos[qi] - pos[ki] < window)
            return (k == n) | ((k < n) & causal)

        mask = BlockMask.from_kv_blocks(
            counts[None, None].to(pos.device),
            columns[None, None].to(pos.device),
            BLOCK_SIZE=tile,
            mask_mod=permitted,
            seq_lengths=(n, n + 1),
        )
        self.masks[window] = mask
        return mask


def _relative_distance(inputs_embeds, cache, is_sliding):
    # The stock helper always constructs an N x N tensor. On a tree, distances are differences of path depths.
    if _building_row.get():
        return None
    return _ordinary_distance(inputs_embeds, cache, is_sliding)


def _convolution(native):
    @wraps(native)
    def forward(hidden_states, dynamic_input=None, **kwargs):
        tree = kwargs.pop("inkling_tree", None)
        if tree is None:
            return native(hidden_states, dynamic_input=dynamic_input, **kwargs)
        extended = tree.expand(hidden_states)
        dynamic = tree.expand(dynamic_input) if dynamic_input is not None else None
        return native(extended, dynamic_input=dynamic, **kwargs).index_select(1, tree.keep)

    return forward


def _relative_bias(layer):
    native = layer.forward

    @wraps(native)
    def forward(hidden_states, relative_distance, position_ids):
        if relative_distance is not None:
            return native(hidden_states, relative_distance, position_ids)
        states = layer.linear(hidden_states).view(*hidden_states.shape[:-1], layer.num_heads, -1)
        return (layer.norm(states, position_ids) @ layer.proj).transpose(1, 2)

    return forward


def _attention(module, query, key, value, attention_mask, scaling, **kwargs):
    tree = kwargs.pop("inkling_tree", None)
    if tree is None or not module.is_causal:
        return inkling.eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
    bias, sinks = kwargs["position_bias"], kwargs["sinks"]
    window = kwargs.get("sliding_window")
    pos = tree.positions
    n, extent = pos.numel(), bias.size(-1)
    if query.device.type == "cuda":
        # Give the sink its own zero value so ordinary softmax and its backward include the sink probability.
        # Native Inkling materializes QK, its scaled value and then the biased logit in the query dtype.
        # Keep those rounding points before the fused softmax.
        dtype = query.dtype

        def score_mod(score, b, h, q, k):
            qi, ki = q.clamp(max=n - 1), k.clamp(max=n - 1)
            distance = pos[qi] - pos[ki]
            contribution = bias[b, h, qi, distance.clamp(0, extent - 1)]
            relative = torch.where((distance >= 0) & (distance < extent), contribution, 0.0)
            scaled = (score.to(dtype).float() * scaling).to(dtype)
            adjusted = (scaled.float() + relative.float()).to(dtype)
            return torch.where(k == n, sinks[b, h, qi, 0], adjusted).to(score.dtype)

        key, value = (F.pad(x, (0, 0, 0, 1)) for x in (key, value))
        output = compile_friendly_flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=tree.block_mask(window),
            scale=1.0,
            enable_gqa=query.size(1) != key.size(1),
            training=module.training,
        )
        return output.transpose(1, 2).contiguous(), None

    # CPU reference path. Bound the attention matrix to one tile of queries and the keys on its own path.
    result = []
    for i, (start, end) in enumerate(zip(tree.offsets[:-1], tree.offsets[1:], strict=True)):
        path = tree.kv[tree.k_offsets[i] : tree.k_offsets[i + 1]]
        ancestors = len(path) - (end - start)
        for q in range(start, end, 128):
            stop = min(q + 128, end)
            lo = max(0, ancestors + q - start - window + 1) if window else 0
            keys = path[lo : ancestors + stop - start]
            distance = pos[q:stop, None] - pos[keys][None, :]
            gather = distance.clamp(0, extent - 1).expand(*bias.shape[:2], -1, -1)
            relative = bias[:, :, q:stop].gather(-1, gather).masked_fill((distance < 0) | (distance >= extent), 0)
            visible = distance >= 0
            if window:
                visible = visible & (distance < window)
            mask = relative.masked_fill(~visible, -torch.inf)
            out, _ = inkling.eager_attention_forward(
                module,
                query[:, :, q:stop],
                key.index_select(2, keys),
                value.index_select(2, keys),
                mask,
                scaling=scaling,
                sinks=sinks[:, :, q:stop],
            )
            result.append(out)
    return torch.cat(result, dim=1), None


def install_inkling_tree(model):
    """Adapt the real Inkling text backbone to TRL's packed row, preserving native non-sequence attention."""
    text = next(layer for layer in model.modules() if isinstance(layer, inkling.Inkling2LoopTextModel))
    if text.config._attn_implementation == INKLING_TREE_ATTENTION:
        return
    AttentionInterface.register(INKLING_TREE_ATTENTION, _attention)
    AttentionMaskInterface.register(INKLING_TREE_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    text.config._attn_implementation = INKLING_TREE_ATTENTION
    inkling.get_relative_distance = _relative_distance
    for layer in text.modules():
        if isinstance(layer, (inkling.Inkling2LoopDynamicShortConvolution, inkling.Inkling2LoopShortConvolution)):
            layer.forward = _convolution(layer.forward)
        elif isinstance(layer, inkling.Inkling2LoopRelativePositionBias):
            layer.forward = _relative_bias(layer)
    native = text.forward

    @wraps(native)
    def forward(*args, **kwargs):
        if "tree_kv_index" not in kwargs:
            return native(*args, **kwargs)
        if kwargs.get("cache") is not None:
            raise ValueError("A training TreeRow cannot reuse an inference cache.")
        kwargs["use_cache"] = False
        kwargs["attention_mask"] = {"full_attention": None, "sliding_attention": None, "linear_attention": None}
        kwargs["inkling_tree"] = InklingTree(kwargs["position_ids"], kwargs, text.config.conv_kernel_size)
        token = _building_row.set(True)
        try:
            return native(*args, **kwargs)
        finally:
            _building_row.reset(token)

    text.forward = forward
