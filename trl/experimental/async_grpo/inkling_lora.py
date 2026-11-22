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

"""Inkling's seven text projections in the published Qwen harness recipe."""

import copy
from pathlib import Path

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import ParamWrapper
from safetensors.torch import save_file
from torch import nn
from transformers.integrations.moe import _grouped_linear
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopTextModel


ATTENTION = {"q_proj": "wq_du", "k_proj": "wk_dv", "v_proj": "wv_dv", "o_proj": "wo_ud"}
EXPERTS = ("gate_proj", "up_proj", "down_proj")
ATTENTION_TARGETS = {f"self_attn.{role}.linear" for role in ATTENTION}
EXPERT_TARGETS = [f"mlp.experts.{role}.weight" for role in EXPERTS]


def is_inkling_recipe_lora(config):
    return (
        isinstance(config, LoraConfig)
        and config.target_modules == ATTENTION_TARGETS
        and set(config.target_parameters or ()) == set(EXPERT_TARGETS)
    )


class _ExpertProjection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=weight.requires_grad)

    def forward(self, states, offsets):
        return _grouped_linear(states, self.weight, offsets)


def _project(projection, states, offsets):
    if not isinstance(projection, ParamWrapper) or projection.disable_adapters or projection.merged:
        return projection(states, offsets)
    # PEFT's generic ParamWrapper folds BA into an entire bank of expert weights for each call.
    # Use its own factors and scaling on the selected tokens, with grouped GEMMs on each bank.
    output = projection.get_base_layer()(states, offsets)
    for adapter in projection.active_adapters:
        left, right, scale = projection.get_delta_factors(adapter)
        low_rank = _grouped_linear(states, right.contiguous(), offsets)
        output = output + _grouped_linear(low_rank, left.contiguous(), offsets) * scale
    return output


class _SplitExperts(nn.Module):
    def __init__(self, experts):
        super().__init__()
        gate, up = experts.gate_up_proj.chunk(2, dim=1)
        self.gate_proj = _ExpertProjection(gate)
        self.up_proj = _ExpertProjection(up)
        self.down_proj = _ExpertProjection(experts.down_proj)
        self.act_fn = experts.act_fn
        self.num_experts = experts.num_experts

    def forward(self, states, indices, weights):
        # Sort once for the base, A and B of all three projections. A sentinel has zero routing weight.
        top_k = indices.size(1)
        ids = indices.flatten()
        weights = weights.flatten().masked_fill(ids == self.num_experts, 0)
        sorted_ids, perm = ids.clamp_max(self.num_experts - 1).sort()
        ends = torch.arange(1, self.num_experts + 1, device=ids.device)
        offsets = torch.searchsorted(sorted_ids, ends, out_int32=True)
        tokens = perm // top_k
        selected = states[tokens]
        gate = _project(self.gate_proj, selected, offsets)
        up = _project(self.up_proj, selected, offsets)
        down = _project(self.down_proj, self.act_fn(gate, up), offsets)
        output = torch.zeros_like(states, dtype=torch.float32)
        output.index_add_(0, tokens, (down * weights[perm, None]).float())
        return output.to(states.dtype)


def prepare_inkling_recipe_lora(model, config):
    """Resolve the reference LoRA targets on a loaded Inkling base before PEFT injection or checkpoint reload."""
    if not isinstance(config, LoraConfig):
        return config
    if config.target_modules == set(ATTENTION).union(EXPERTS) and not config.target_parameters:
        config = copy.deepcopy(config)
        config.target_modules = ATTENTION_TARGETS.copy()
        config.target_parameters = EXPERT_TARGETS.copy()
    if is_inkling_recipe_lora(config):
        text = next(module for module in model.modules() if isinstance(module, Inkling2LoopTextModel))
        for layer in text.layers:
            if not isinstance(layer.mlp.experts, _SplitExperts):
                layer.mlp.experts = _SplitExperts(layer.mlp.experts)
    return config


def save_inkling_serving_adapter(model, state, name, directory):
    """Write the attention checkpoint names and one pair of 2D factors per routed expert and projection."""
    config = copy.deepcopy(model.peft_config[name])
    if config.rank_pattern or config.alpha_pattern or config.use_rslora or config.lora_bias:
        raise ValueError("The harness recipe serves one rank and alpha for every projection, with bias-free LoRA.")
    backbone_name, text = next((n, m) for n, m in model.named_modules() if isinstance(m, Inkling2LoopTextModel))
    remaining = dict(state)
    weights = {}

    def factors(source):
        return (remaining.pop(f"{source}.lora_{ab}.{name}.weight") for ab in ("A", "B"))

    for idx, layer in enumerate(text.layers):
        source_layer = f"{backbone_name}.layers.{idx}"
        dest = f"base_model.model.model.llm.layers.{idx}"
        for role, checkpoint_name in ATTENTION.items():
            a, b = factors(f"{source_layer}.self_attn.{role}.linear")
            weights[f"{dest}.attn.{checkpoint_name}.lora_A.weight"] = a
            weights[f"{dest}.attn.{checkpoint_name}.lora_B.weight"] = b
        experts = layer.mlp.experts.num_experts
        for role in EXPERTS:
            a, b = factors(f"{source_layer}.mlp.experts.{role}")
            if a.shape[0] != experts * config.r or b.shape[1] != experts * config.r:
                raise ValueError(f"Unexpected expert factors for {source_layer}.{role}: {a.shape}, {b.shape}")
            # PEFT packs A in (expert, rank, input) and B in (output, rank, expert) order.
            a = a.reshape(experts, config.r, -1)
            b = b.reshape(b.shape[0], config.r, experts).permute(2, 0, 1)
            for expert in range(experts):
                key = f"{dest}.mlp.experts.{expert}.{role}"
                weights[f"{key}.lora_A.weight"] = a[expert].contiguous()
                weights[f"{key}.lora_B.weight"] = b[expert].contiguous()
    if remaining:
        raise ValueError(f"Unmapped Inkling adapter factors: {list(remaining)[:3]}")

    config.target_parameters = None
    config.target_modules = set(ATTENTION.values()).union(EXPERTS)
    config.inference_mode = True
    config.save_pretrained(directory)
    save_file(weights, str(Path(directory) / "adapter_model.safetensors"), metadata={"format": "pt"})
