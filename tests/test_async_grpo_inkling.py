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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trl.experimental.async_grpo.packing import TreeRow, unpad_tree


def make_model(head_dim=16):
    pytest.importorskip("transformers.models.inkling2_loop")
    from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

    config = Inkling2LoopTextConfig(
        vocab_size=64,
        unpadded_vocab_size=64,
        eos_token_id=63,
        hidden_size=32,
        num_hidden_layers=8,
        layer_types=["hybrid_sliding", "hybrid"] * 4,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=6,
        loop_num_passes=3,
        loop_gate_rank=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=head_dim,
        sliding_window_size=4,
        rel_extent=6,
        d_rel=4,
        moe_intermediate_size=24,
        n_routed_experts=4,
        num_experts_per_tok=2,
        conv_kernel_size=4,
        use_cache=False,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    model = Inkling2LoopForCausalLM(config)
    # A new model initializes several learned operations to zero, which could hide a missing path.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith((".static_weight", ".query", ".relative_position_bias.proj")):
                p.normal_(0.0, 0.2)
    return model


def pack(branches, device):
    row = TreeRow()
    for i, ids in enumerate(branches):
        mask = np.arange(len(ids)) > 0
        row.add(
            {
                "group_id": i // 2,
                "input_ids": np.array(ids),
                "completion_mask": mask,
                "old_log_probs": np.zeros(len(ids)),
                "sampling_mask": [[] for _ in ids],
                "sampler_ids": np.array(ids)[:, None],
                "sampler_log_probs": np.zeros((len(ids), 1)),
                "advantage": 1.0,
            }
        )
    data = {k: v[None].to(device) for k, v in row.pack().items()}
    return data, unpad_tree(data)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_inkling_packed_logits_and_gradients(device):
    from trl.experimental.async_grpo.inkling_tree import install_inkling_tree

    torch.manual_seed(17)
    torch.set_num_threads(1)
    branches = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 9], [1, 2, 12], [31, 32, 33, 34], [1, 2, 12]]
    # Exercise the released checkpoint's head dimension on the compiled CUDA path.
    model = make_model(head_dim=128 if device == "cuda" else 16).to(device)
    if device == "cuda":
        model.to(torch.bfloat16)
        for name, parameter in model.named_parameters():
            if any(key in name for key in model._keep_in_fp32_modules_strict):
                parameter.data = parameter.data.float()
    packed_model = copy.deepcopy(model)
    install_inkling_tree(packed_model)
    data, tree = pack(branches, device)
    expected_outputs = []
    for ids in branches:
        sequence = torch.tensor([ids], device=device)
        out = model(sequence, use_cache=False).logits[0, :-1]
        expected_outputs.append(out)
    expected = torch.cat(expected_outputs)
    actual = packed_model(data["input_ids"], position_ids=data["position_ids"], use_cache=False, **tree).logits[0]
    actual = actual[data["token_index"][0]]
    rtol, atol = (0.06, 0.011) if device == "cuda" else (2e-4, 2e-5)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    cotangent = torch.randn_like(expected) / expected.numel()
    (expected * cotangent).sum().backward()
    (actual * cotangent).sum().backward()
    for (name, p), (other_name, q) in zip(model.named_parameters(), packed_model.named_parameters(), strict=True):
        assert name == other_name and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=rtol, atol=atol * 0.04, msg=lambda s, n=name: f"{n}: {s}")


def test_inkling_tree_mask_at_block_boundaries():
    from trl.experimental.async_grpo.inkling_tree import InklingTree

    prefix = list(range(1, 154))
    branches = [prefix + list(range(160, 329)), prefix + [350, 351], [400, 401], [400, 402]]
    data, tree = pack(branches, "cpu")
    layout = InklingTree(data["position_ids"], tree, 4)
    size = data["input_ids"].numel()
    q, k = torch.arange(size)[:, None], torch.arange(size + 1)[None, :]
    for window in (None, 4, 128):
        mask = layout.block_mask(window)
        expected = torch.zeros(size, size + 1, dtype=torch.bool)
        for i, (start, end) in enumerate(zip(layout.offsets[:-1], layout.offsets[1:], strict=True)):
            path = layout.kv[layout.k_offsets[i] : layout.k_offsets[i + 1]]
            for j in range(start, end):
                limit = len(path) - end + j + 1
                expected[j, path[max(0, limit - window) if window else 0 : limit]] = True
        expected[:, -1] = True
        torch.testing.assert_close(mask.mask_mod(0, 0, q, k), expected)
        for block, count in enumerate(mask.kv_num_blocks[0, 0].tolist()):
            allowed = set(mask.kv_indices[0, 0, block, :count].tolist())
            required = set((expected[128 * block : 128 * (block + 1)].any(0).nonzero().flatten() // 128).tolist())
            assert required <= allowed


def test_inkling_mask_at_reference_token_budget():
    from trl.experimental.async_grpo.inkling_tree import InklingTree

    # Two 16k continuations share a 32k prefix. This is one complete reference-sized row.
    n, fork = 65_536, 32_768
    midpoint = fork + (n - fork) // 2
    positions = torch.cat((torch.arange(midpoint), torch.arange(fork, midpoint)))[None]
    kv = torch.cat((torch.arange(fork), torch.arange(midpoint), torch.arange(fork), torch.arange(midpoint, n)))
    layout = InklingTree(
        positions,
        {
            "tree_cu_seqlens_q": torch.tensor([0, fork, midpoint, n]),
            "tree_cu_seqlens_k": torch.tensor([0, fork, fork + midpoint, len(kv)]),
            "tree_segment_parents": torch.tensor([-1, 0, 0]),
            "tree_kv_index": kv,
        },
        kernel_size=4,
    )
    full, sliding = layout.block_mask(None), layout.block_mask(1024)
    query = torch.tensor([n - 1] * 5)
    keys = torch.tensor([fork - 1, midpoint - 1, n - 1024, n - 1025, n])
    assert full.mask_mod(0, 0, query, keys).tolist() == [True, False, True, True, True]
    assert sliding.mask_mod(0, 0, query, keys).tolist() == [False, False, True, False, True]
    assert full.kv_indices.numel() == (n // 128) * (n // 128 + 1)
    assert layout.with_halos.numel() == n + 3 * 3


def test_inkling_async_grpo_loss(tmp_path, monkeypatch):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from trl.experimental.async_grpo import AsyncGRPOConfig, async_grpo_trainer
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
    from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer

    torch.manual_seed(18)
    torch.set_num_threads(1)
    # The real trainer's throughput logger probes a CUDA device even in CPU mode.
    monkeypatch.setattr(async_grpo_trainer, "get_peak_flops_per_device", lambda accelerator, dtype: None)
    # This fixture supplies the sampler's measured probabilities directly. It starts no external vLLM server.
    monkeypatch.setattr(async_grpo_trainer, "WeightTransferClient", lambda **kwargs: None)
    monkeypatch.setattr(
        async_grpo_trainer, "AsyncRolloutWorker", lambda *args, **kwargs: SimpleNamespace(rollout_buffer=[])
    )
    model = make_model()
    tokens = {str(i): i for i in range(64)}
    processor = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(tokens, unk_token="0")), pad_token="0", eos_token="63"
    )
    args = AsyncGRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        max_steps=1,
        temperature=0.8,
        score_centering=True,
        gradient_checkpointing=False,
    )
    samples = []
    with torch.no_grad():
        for i, ids in enumerate(([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 11, 12], [1, 2, 3, 10])):
            ids = torch.tensor(ids)
            log_probs = (model(ids[None], use_cache=False).logits[0].float() / 0.8).log_softmax(-1)
            scores = torch.zeros(len(ids))
            scores[1:] = log_probs[:-1].gather(1, ids[1:, None])[:, 0]
            top = log_probs.topk(8)
            sampler_ids = torch.zeros_like(top.indices)
            sampler_ids[1:] = top.indices[:-1]
            sampler_logps = torch.zeros_like(top.values)
            sampler_logps[1:] = top.values[:-1]
            samples.append(
                {
                    "group_id": 0,
                    "input_ids": ids.numpy(),
                    "completion_mask": np.arange(len(ids)) >= 2,
                    "old_log_probs": scores.numpy(),
                    "sampling_mask": [[] for _ in ids],
                    "sampler_ids": sampler_ids.numpy(),
                    "sampler_log_probs": sampler_logps.numpy(),
                    "advantage": [1.0, -0.4, 0.15][i],
                    "reward": float(i),
                    "reward_std": 1.0,
                    "metrics": {},
                }
            )

    def fixture_session(**kwargs):
        raise AssertionError("This numerical check does not request rollouts")

    trainer = HarnessAsyncGRPOTrainer(
        model,
        args=args,
        processing_class=processor,
        harness_session_factory=fixture_session,
        train_dataset=Dataset.from_dict({"prompt": ["1", "2"]}),
    )
    # Trainer.train sets this for each group of accumulated microbatches before calling compute_loss.
    trainer.current_gradient_accumulation_steps = args.gradient_accumulation_steps
    collator = DataCollatorForRollout(pad_token_id=0)

    def loss(examples):
        row = TreeRow()
        for example in examples:
            row.add(example)
        batch = collator([[row]])
        return trainer.compute_loss(trainer.model, batch), int(batch["global_n_tokens"][0])

    actual, count = loss(samples)
    actual.backward()
    packed_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    expected = sum(value * length / count for value, length in (loss([s]) for s in samples))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    expected.backward()
    assert max(g.abs().max().item() for g in packed_grads.values()) > 1e-5
    for name, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(
                packed_grads[name], p.grad, rtol=2e-4, atol=2e-6, msg=lambda s, n=name: f"{n}: {s}"
            )
