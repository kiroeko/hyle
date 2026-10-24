"""Independent local Torch oracle, never called by the C++ candidate.

The installed Qwen forward resets conv/GDN state on cached multi-token input.
Use its explicit recurrent formula for every block, with the actual prior state
and the last three projected convolution rows. No Hyle state is read or filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
import types

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")


def read_schedule(path: Path, vocabulary: int, capacity: int) -> list[dict]:
    """Read the consumer's prompt/decode pairs before importing/loading Torch."""
    if not 0 < vocabulary <= 0x7fffffff or not 0 < capacity <= 0x7fffffff:
        raise ValueError("schedule requires positive vocabulary/capacity within I32 positions")
    phases = []
    total = turn = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = shlex.split(line)
        expected = "prompt" if len(phases) % 2 == 0 else "decode"
        if len(fields) != 2 or fields[0] != expected or not fields[1]:
            raise ValueError(f"{path}:{line_number}: expected {expected} IDSFILE")
        source = path.parent / fields[1]
        ids = []
        for token in source.read_text(encoding="utf-8").split():
            if not re.fullmatch(r"-?[0-9]+", token):
                raise ValueError(f"{source}: malformed token ID")
            value = int(token)
            if not 0 <= value < vocabulary:
                raise ValueError(f"{source}: token ID outside vocabulary")
            if len(ids) >= capacity - total:
                raise ValueError(f"{path}:{line_number}: schedule exceeds context capacity")
            ids.append(value)
        if not ids:
            raise ValueError(f"{source}: phase requires nonempty token IDs")
        if expected == "prompt":
            turn += 1
        phases.append(dict(kind=expected, ids=ids, turn=turn))
        total += len(ids)
    if not phases or phases[-1]["kind"] != "decode":
        raise ValueError("schedule requires complete prompt/decode turns")
    return phases


def phase_name(phase: dict, local_index: int) -> str:
    prefix = f"round{phase['turn']}"
    return prefix + ("_prefill" if phase["kind"] == "prompt" else f"_decode{local_index + 1}")


def fixed_recurrent_forward(self, hidden_states, cache_params=None,
                            cache_position=None, attention_mask=None):
    import torch
    import torch.nn.functional as f
    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
    del cache_position
    hidden_states = impl.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, tokens, _ = hidden_states.shape
    projected = self.in_proj_qkv(hidden_states).transpose(1, 2)
    old_conv = None if cache_params is None else cache_params.conv_states[self.layer_idx]
    old_state = None if cache_params is None else cache_params.recurrent_states[self.layer_idx]
    if old_conv is None or old_conv.numel() == 0:
        prefix = projected.new_zeros(batch, projected.shape[1], self.conv_kernel_size - 1)
    else:
        prefix = old_conv[:, :, -(self.conv_kernel_size - 1):]
    samples = torch.cat((prefix, projected), dim=-1)
    convolved = f.conv1d(samples, self.conv1d.weight, self.conv1d.bias,
                        groups=self.conv1d.groups)
    activated = f.silu(convolved).transpose(1, 2)
    query, key, value = torch.split(activated, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch, tokens, -1, self.head_k_dim)
    key = key.reshape(batch, tokens, -1, self.head_k_dim)
    value = value.reshape(batch, tokens, -1, self.head_v_dim)
    if self.num_v_heads != self.num_k_heads:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    beta = self.in_proj_b(hidden_states).sigmoid()
    decay = -self.A_log.float().exp() * f.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    if old_state is not None and old_state.numel() == 0:
        old_state = None
    output, state = impl.torch_recurrent_gated_delta_rule(query, key, value, decay, beta,
        initial_state=old_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    if cache_params is not None:
        # Keep the native reference's four-row rolling state; the C++ consumer
        # stores exactly the three rows needed before the next causal window.
        cache_params.conv_states[self.layer_idx] = samples[:, :, -self.conv_kernel_size:].contiguous()
        cache_params.recurrent_states[self.layer_idx] = state
    z = self.in_proj_z(hidden_states).reshape(-1, self.head_v_dim)
    output = self.norm(output.reshape(-1, self.head_v_dim), z)
    return self.out_proj(output.reshape(batch, tokens, -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--trace-level", choices=("full", "decisions"), default="full")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("reference output must be new")
    schedule = None
    if args.schedule is not None:
        if args.continuation is not None or args.steps is not None:
            raise ValueError("--schedule is mutually exclusive with --continuation/explicit --steps")
        schedule = read_schedule(args.schedule, 248320, args.capacity)
        if (args.inputs / "input.json").exists():
            source_id = json.loads((args.inputs / "input.json").read_bytes())["source_id"]
        else:
            manifest = (args.inputs / "weights.tsv").read_text(encoding="utf-8").splitlines()
            if len(manifest) < 3 or manifest[0] != "HYLE_QWEN35_TEXT_V1":
                raise ValueError("reference inputs require input.json source_id or an original weights.tsv")
            source_id = manifest[2]
    if args.steps is None:
        args.steps = 2
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    config = Qwen3_5TextConfig(**json.loads((args.model / "config.json").read_bytes())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = impl.Qwen3_5ForCausalLM(config)
    model.model.rotary_emb = impl.Qwen3_5TextRotaryEmbedding(config, device="cpu")
    parameters = {}
    shard = args.model / "model.safetensors-00001-of-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as source:
        for name in source.keys():
            if name.startswith("model.language_model."):
                parameters[name.replace("model.language_model.", "model.", 1)] = source.get_tensor(name)
    assert len(parameters) == 320
    parameters["lm_head.weight"] = parameters["model.embed_tokens.weight"]
    model.load_state_dict(parameters, strict=True, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    del parameters
    model.eval().to("cuda")
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.forward = types.MethodType(fixed_recurrent_forward, layer.linear_attn)
    args.output.mkdir(parents=True)
    captured = []
    model.model.norm.register_forward_hook(lambda module, inputs, output: captured.append(output[:, -1:].detach()))
    cache = None
    steps = []
    generated = []
    visible_tokens = 0
    def raw(path, tensor):
        path.write_bytes(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    def evaluate(name, ids):
        nonlocal cache, visible_tokens
        directory = args.output / name
        directory.mkdir()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(input_ids=torch.tensor([ids], dtype=torch.long, device="cuda"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits = out.logits[0, -1].float()
        token = int(logits.argmax().item())
        assert torch.isfinite(logits).all()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        raw(directory / "logits.f32", logits)
        raw(directory / "hidden.bf16", captured.pop())
        if args.trace_level == "full":
            for layer in range(24):
                prefix = directory / f"layer{layer}"
                if layer % 4 == 3:
                    raw(prefix.with_suffix(".key.bf16"), cache.key_cache[layer][0].transpose(0, 1))
                    raw(prefix.with_suffix(".value.bf16"), cache.value_cache[layer][0].transpose(0, 1))
                else:
                    raw(prefix.with_suffix(".conv.bf16"), cache.conv_states[layer][:, :, -3:].transpose(1, 2))
                    raw(prefix.with_suffix(".gdn.f32"), cache.recurrent_states[layer].transpose(-1, -2))
        visible_tokens += len(ids)
        steps.append(dict(name=name, input_ids=ids, next_id=token, seconds=seconds))
        if schedule is not None:
            steps[-1]["visible_tokens"] = visible_tokens
        print(name, len(ids), token, flush=True)
        return token
    with torch.inference_mode():
        if schedule is not None:
            for phase in schedule:
                calls = 1 if phase["kind"] == "prompt" else len(phase["ids"])
                for local in range(calls):
                    ids = phase["ids"] if phase["kind"] == "prompt" else [phase["ids"][local]]
                    generated.append(evaluate(phase_name(phase, local), ids))
        else:
            prompt = list(map(int, (args.inputs / "prompt.ids").read_text().split()))
            token = evaluate("prefill", prompt)
            if args.continuation:
                token = evaluate("continuation", list(map(int, args.continuation.read_text().split())))
            generated.append(token)
            for step in range(1, args.steps):
                if token == config.eos_token_id:
                    break
                token = evaluate(f"decode{step}", [token])
                generated.append(token)
    if schedule is None:
        source_id = json.loads((args.inputs / "input.json").read_bytes())["source_id"]
    receipt = dict(kind="independent_torch_reference_not_hyle_execution", source_id=source_id,
        text_tensors=320, text_weight_bytes=sum(x.numel() * x.element_size() for x in model.parameters()),
        model_source=str(impl.__file__), model_source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        state_formula="explicit prior conv+recurrent state for every block; no cached multi-token reset",
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, steps=steps, generated_ids=generated)
    if schedule is not None:
        receipt.update(schedule=str(args.schedule.resolve()), capacity=args.capacity,
                       phases=schedule, processed_tokens=visible_tokens,
                       trace_level=args.trace_level, teacher_forced=True,
                       independent_quality_acceptance=False)
    (args.output / "reference.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
