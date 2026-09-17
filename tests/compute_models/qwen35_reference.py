"""Offline reference consumer for the frozen, local Qwen3.5-2B text model.

This script produces oracle data, not HYLE execution evidence or performance results.
It deliberately does not load vision/MTP, sample tokens, or provide a serving API.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from local_inputs import read_inventory, sha256, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    inventory = read_inventory(args.inventory)
    input_receipt = verify(args.source, inventory)
    cases = json.loads(args.cases.read_bytes())
    if cases["source_id"] != inventory["artifact_identity_sha256"]:
        raise ValueError("case/input identity mismatch")
    if args.output.exists():
        raise ValueError("reference output must be a new directory")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "safetensors", "numpy")
    }
    if versions["transformers"] != "5.3.0":
        raise ValueError("this qualified reference recipe requires Transformers 5.3.0")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    config = Qwen3_5TextConfig(**json.loads((args.source / "config.json").read_bytes())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = implementation.Qwen3_5ForCausalLM(config)
    # Nonpersistent positional buffers are formula-derived, not checkpoint weights.
    model.model.rotary_emb = implementation.Qwen3_5TextRotaryEmbedding(config, device="cpu")
    parameters = {}
    consumed = []
    for shard in inventory["shards"]:
        with safe_open(args.source / shard["path"], framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith("model.language_model."):
                    parameters[name.replace("model.language_model.", "model.", 1)] = source.get_tensor(name)
                    consumed.append(name)
    parameters["lm_head.weight"] = parameters["model.embed_tokens.weight"]
    model.load_state_dict(parameters, strict=True, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    del parameters
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("unbound model weights")
    model.eval()
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        if free < weight_bytes + 1024**3:
            raise RuntimeError(f"insufficient free GPU memory: {free}; weights {weight_bytes} + 1 GiB reserve")
        torch.cuda.reset_peak_memory_stats()
    model.to(args.device)
    # The oracle uses the auditable Torch fallback formulas, never optional FLA/conv kernels.
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            attention = layer.linear_attn
            attention.causal_conv1d_fn = None
            attention.causal_conv1d_update = implementation.torch_causal_conv1d_update
            attention.chunk_gated_delta_rule = implementation.torch_chunk_gated_delta_rule
            attention.recurrent_gated_delta_rule = implementation.torch_recurrent_gated_delta_rule

    args.output.mkdir(parents=True)
    shutil.copyfile(args.cases, args.output / "cases.json")
    shutil.copyfile(__file__, args.output / "reference_script.py")
    shutil.copyfile(implementation.__file__, args.output / "modeling_qwen3_5.py")
    receipt = {
        "schema_version": 1,
        "kind": "external_reference_execution_not_hyle_verdict",
        "source_id": inventory["artifact_identity_sha256"],
        "input_receipt": input_receipt,
        "case_sha256": sha256(args.cases),
        "reference_inputs_sha256": hashlib.sha256(json.dumps(
            {"source_id": cases["source_id"], "reference_cases": cases["reference_cases"]},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": args.device,
        "model_source_sha256": sha256(Path(implementation.__file__)),
        "model_source_path": str(implementation.__file__),
        "branch": "24_layer_text_backbone_tied_head_only",
        "profile": "B16",
        "attention": "eager",
        "linear_attention": "torch_chunk_prefill_and_recurrent_decode",
        "weights_consumed": sorted(consumed),
        "excluded": ["vision", "MTP"],
        "cases": [],
    }
    layer_outputs = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def capture(module, inputs, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            layer_outputs[f"layer_{index:02d}"] = value[:, -1].detach().float().cpu().contiguous()
        handles.append(layer.register_forward_hook(capture))

    with torch.inference_mode():
        for case in cases["reference_cases"]:
            print(f"REFERENCE {case['id']} begin", flush=True)
            tensors = {}
            cache = None
            step_summaries = []
            steps = [case["prompt_ids"]] + [[token] for token in case["decode_ids"]]
            for step, token_ids in enumerate(steps):
                inputs = torch.tensor([token_ids], dtype=torch.long, device=args.device)
                output = model(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                logits = output.logits[:, -1].detach().float().cpu().contiguous()
                if not torch.isfinite(logits).all():
                    raise ValueError(f"nonfinite logits: {case['id']} step {step}")
                tensors[f"step_{step:02d}.logits"] = logits
                for name, value in layer_outputs.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"nonfinite layer: {case['id']} step {step} {name}")
                    tensors[f"step_{step:02d}.{name}"] = value
                top = torch.topk(logits[0], 2)
                step_summaries.append({
                    "step": step,
                    "tokens_consumed": len(token_ids),
                    "argmax": top.indices[0].item(),
                    "top1_margin": (top.values[0] - top.values[1]).item(),
                    "logits_min": logits.min().item(),
                    "logits_max": logits.max().item(),
                })
            state_dtypes = {}
            for family in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
                for index, state in enumerate(getattr(cache, family)):
                    if state is not None:
                        name = f"final.{family}.{index:02d}"
                        state_dtypes[name] = str(state.dtype)
                        value = state.detach().float().cpu().contiguous()
                        if not torch.isfinite(value).all():
                            raise ValueError(f"nonfinite state: {name}")
                        tensors[name] = value
            artifact = args.output / f"{case['id']}.safetensors"
            save_file(tensors, artifact, metadata={"profile": "B16", "source_id": receipt["source_id"]})
            receipt["cases"].append({
                "id": case["id"],
                "artifact": artifact.name,
                "sha256": sha256(artifact),
                "bytes": artifact.stat().st_size,
                "steps": step_summaries,
                "state_dtypes": state_dtypes,
                "result": "REFERENCE_OUTPUT_GENERATED",
            })
            del output, cache, tensors
            print(f"REFERENCE {case['id']} saved", flush=True)
    for handle in handles:
        handle.remove()
    if args.device == "cuda":
        torch.cuda.synchronize()
        receipt["gpu"] = torch.cuda.get_device_name()
        receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(receipt["cases"]), "result": "REFERENCE_OUTPUT_GENERATED"}))


if __name__ == "__main__":
    main()
