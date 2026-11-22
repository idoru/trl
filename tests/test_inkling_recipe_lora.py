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

import copy
import json

import pytest
import torch
import torch.nn.functional as F

from .test_async_grpo_inkling import make_model


def policy(base):
    from peft import LoraConfig, get_peft_model

    from trl.experimental.async_grpo.inkling_lora import prepare_inkling_recipe_lora

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(base, prepare_inkling_recipe_lora(base, config))


def test_routed_lora_outputs_and_gradients_match_independent_experts():
    pytest.importorskip("peft")
    torch.manual_seed(451)
    torch.set_num_threads(1)
    base = make_model()
    native = copy.deepcopy(base.model.layers[0].mlp.experts)
    policy(base)
    adapted = base.model.layers[0].mlp.experts
    states = torch.randn(4, native.hidden_dim, requires_grad=True)
    # Expert 1 is unused. Expert 4 is the sentinel that native Inkling skips.
    ids = torch.tensor([[0, 2], [3, 0], [2, 4], [0, 3]])
    weights = torch.rand(ids.shape, requires_grad=True)
    torch.testing.assert_close(adapted(states, ids, weights), native(states, ids, weights))
    with torch.no_grad():
        for name, value in adapted.named_parameters():
            if "lora_B" in name:
                value.normal_(0, 0.08)

    def dense(role, original, expert, x):
        layer = getattr(adapted, role)
        a = layer.lora_A["default"].weight.reshape(native.num_experts, 32, -1)[expert]
        b = layer.lora_B["default"].weight.reshape(original.size(0), 32, native.num_experts)[:, :, expert]
        return F.linear(x, original + 2 * (b @ a))

    rows = []
    for token, choices in enumerate(ids.tolist()):
        y = torch.zeros_like(states[token])
        for choice, expert in enumerate(choices):
            if expert == native.num_experts:
                continue
            gate_w, up_w = native.gate_up_proj[expert].chunk(2)
            gate = dense("gate_proj", gate_w, expert, states[token])
            up = dense("up_proj", up_w, expert, states[token])
            down = dense("down_proj", native.down_proj[expert], expert, native.act_fn(gate, up))
            y = y + down * weights[token, choice]
        rows.append(y)
    expected = torch.stack(rows)
    actual = adapted(states, ids, weights)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    inputs = (states, weights, *(p for p in adapted.parameters() if p.requires_grad))
    cotangent = torch.randn_like(actual)
    for got, want in zip(
        torch.autograd.grad(actual, inputs, cotangent), torch.autograd.grad(expected, inputs, cotangent), strict=True
    ):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)


def test_split_training_weights_cannot_use_the_unconverted_merged_sync(tmp_path):
    from trl.experimental.async_grpo import AsyncGRPOConfig
    from trl.experimental.async_grpo.async_grpo_trainer import select_adapter_sync

    model = policy(make_model())
    args = AsyncGRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none")
    for lora, dp in ((None, 1), ({"max_loras": 18, "max_lora_rank": 32}, 2)):
        with pytest.raises(ValueError, match="versioned"):
            select_adapter_sync({"lora_config": lora, "parallel_config": {"data_parallel_size": dp}}, model, args)


def test_updated_recipe_lora_roundtrip_and_versioned_export(tmp_path):
    from accelerate import Accelerator
    from peft import PeftConfig, PeftModel
    from safetensors.torch import load_file

    from trl.experimental.async_grpo.async_grpo_trainer import save_lora_adapter
    from trl.experimental.async_grpo.inkling_lora import prepare_inkling_recipe_lora

    torch.manual_seed(610)
    torch.set_num_threads(1)
    base = make_model(hidden_size=64, moe_intermediate_size=48).eval()
    fresh = copy.deepcopy(base)
    model = policy(base).eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-5)
    ids = torch.tensor([[3, 19, 27, 6, 1, 4, 21]])
    frozen = fresh(ids).logits.detach()
    # A valid gate/up pair can have combined rank 64. Give expert zero disjoint input subspaces.
    bank = base.model.layers[0].mlp.experts
    with torch.no_grad():
        for role, start in (("gate_proj", 0), ("up_proj", 32)):
            proj = getattr(bank, role)
            proj.lora_A["default"].weight[:32].copy_(torch.eye(64)[start : start + 32])
            b = proj.lora_B["default"].weight.reshape(48, 32, bank.num_experts)
            b[:, :, 0].copy_(torch.eye(48, 32))
    logits = model(ids).logits
    original = logits.detach().clone()
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    optimizer.step()
    expected = model(ids).logits.detach()
    assert (expected - original).abs().max() > 1e-5
    training = tmp_path / "checkpoint-1"
    model.save_pretrained(training)
    config = PeftConfig.from_pretrained(training)
    prepare_inkling_recipe_lora(fresh, config)
    reloaded = PeftModel.from_pretrained(fresh, training, config=config).eval()
    torch.testing.assert_close(reloaded(ids).logits, expected, rtol=0, atol=0)
    with reloaded.disable_adapter():
        torch.testing.assert_close(reloaded(ids).logits, frozen, rtol=2e-5, atol=2e-6)

    version = tmp_path / "inkling-v1"
    save_lora_adapter(model, Accelerator(cpu=True), "default", str(version))
    served = json.loads((version / "adapter_config.json").read_text())
    assert served["r"] == 32 and served["lora_alpha"] == 64
    assert set(served["target_modules"]) == {"wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_proj", "up_proj", "down_proj"}
    tensors = load_file(version / "adapter_model.safetensors")
    names = set()
    for i, layer in enumerate(base.model.layers):
        prefix = f"base_model.model.model.llm.layers.{i}"
        for role, checkpoint_name in zip(
            ("q_proj", "k_proj", "v_proj", "o_proj"), ("wq_du", "wk_dv", "wv_dv", "wo_ud"), strict=True
        ):
            proj = getattr(layer.self_attn, role).linear
            for factor in ("A", "B"):
                key = f"{prefix}.attn.{checkpoint_name}.lora_{factor}.weight"
                torch.testing.assert_close(tensors[key], getattr(proj, f"lora_{factor}")["default"].weight)
                names.add(key)
        bank = layer.mlp.experts
        for role in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(bank, role)
            for e in range(bank.num_experts):
                key = f"{prefix}.mlp.experts.{e}.{role}.lora_"
                a, b = tensors[key + "A.weight"], tensors[key + "B.weight"]
                assert a.size(0) == b.size(1) == 32
                torch.testing.assert_close((2 * b) @ a, proj.get_delta_weight("default")[e])
                names.update((key + "A.weight", key + "B.weight"))
    assert set(tensors) == names
    prefix = "base_model.model.model.llm.layers.0.mlp.experts.0"
    deltas = [
        tensors[f"{prefix}.{role}.lora_B.weight"] @ tensors[f"{prefix}.{role}.lora_A.weight"]
        for role in ("gate_proj", "up_proj")
    ]
    assert torch.linalg.matrix_rank(torch.cat(deltas)).item() == 64

    optimizer.zero_grad()
    F.cross_entropy(model(ids).logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
    optimizer.step()
    model.save_pretrained(tmp_path / "checkpoint-2")
    second = tmp_path / "inkling-v2"
    save_lora_adapter(model, Accelerator(cpu=True), "default", str(second))
    updated = load_file(second / "adapter_model.safetensors")
    first_again = load_file(version / "adapter_model.safetensors")
    assert all(torch.equal(tensors[k], first_again[k]) for k in names)
    assert any(not torch.equal(tensors[k], updated[k]) for k in names)
