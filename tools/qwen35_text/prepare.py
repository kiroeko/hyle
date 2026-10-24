"""Prepare local Qwen3.5-2B text inputs; no candidate model mathematics.

The C++ consumer reads the original checkpoint itself. This script only checks
its identity, writes tensor offsets, and invokes the installed local tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]

MTP_SHAPES = {
    "mtp.fc.weight": (2048, 4096),
    "mtp.layers.0.input_layernorm.weight": (2048,),
    "mtp.layers.0.mlp.down_proj.weight": (2048, 6144),
    "mtp.layers.0.mlp.gate_proj.weight": (6144, 2048),
    "mtp.layers.0.mlp.up_proj.weight": (6144, 2048),
    "mtp.layers.0.post_attention_layernorm.weight": (2048,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2048),
    "mtp.layers.0.self_attn.o_proj.weight": (2048, 2048),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (4096, 2048),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2048),
    "mtp.norm.weight": (2048,),
    "mtp.pre_fc_norm_embedding.weight": (2048,),
    "mtp.pre_fc_norm_hidden.weight": (2048,),
}


def select_weights(header: dict, header_bytes: int, file_bytes: int,
                   mtp: bool = False) -> list:
    selected = []
    mtp_names = set()
    for name, tensor in sorted(header.items()):
        if not name.startswith("model.language_model.") and not (mtp and name.startswith("mtp.")):
            continue
        dtype, shape, (begin, end) = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        count = 1
        for dimension in shape:
            if dimension <= 0:
                raise ValueError("empty/negative checkpoint tensor")
            count *= dimension
        if dtype not in ("BF16", "F32") or end - begin != count * {"BF16": 2, "F32": 4}[dtype]:
            raise ValueError(f"unsupported original tensor: {name}")
        if begin < 0 or end < begin or header_bytes + 8 + end > file_bytes:
            raise ValueError(f"tensor exceeds original shard: {name}")
        if name.startswith("mtp."):
            if dtype != "BF16" or tuple(shape) != MTP_SHAPES.get(name):
                raise ValueError(f"identified MTP tensor differs: {name}")
            mtp_names.add(name)
        selected.append((name, dtype, shape, header_bytes + 8 + begin, end - begin))
    if mtp and mtp_names != MTP_SHAPES.keys():
        raise ValueError("identified MTP tensor inventory differs")
    expected_count, expected_bytes = (335, 3885311680) if mtp else (320, 3763655360)
    if len(selected) != expected_count or sum(item[4] for item in selected) != expected_bytes:
        raise ValueError("identified text+MTP tensor inventory differs" if mtp else
                         "identified text-only tensor inventory differs")
    return selected


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Please calculate 2 + 3. Answer briefly.")
    parser.add_argument("--continuation", default="", help="optional raw text cached continuation")
    parser.add_argument("--mtp", action="store_true",
                        help="include the original 15 MTP weights; does not execute MTP")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output directory must be new")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    inventory = json.loads((ROOT / "docs/implementation/tasks/compute/qwen35-2b-local-input.json").read_text())
    records = []
    for item in inventory["files"]:
        path = args.model / item["path"]
        if path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"original local input identity differs: {path}")
        records.append(item)
    config = json.loads((args.model / "config.json").read_bytes())["text_config"]
    required = dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=24,
                    num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                    linear_num_key_heads=16, linear_num_value_heads=16,
                    linear_key_head_dim=128, linear_value_head_dim=128,
                    linear_conv_kernel_dim=4, vocab_size=248320, tie_word_embeddings=True)
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("this consumer requires the identified Qwen3.5-2B text geometry")
    shard = (args.model / inventory["shards"][0]["path"]).resolve()
    with shard.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > shard.stat().st_size - 8:
            raise ValueError("invalid safetensors header extent")
        header = json.loads(stream.read(length))
    selected = select_weights(header, length, shard.stat().st_size, args.mtp)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True)
    prompt = encoded["input_ids"]
    continuation = tokenizer.encode(args.continuation, add_special_tokens=False)
    args.output.mkdir(parents=True)
    signature = "HYLE_QWEN35_TEXT_MTP_V1" if args.mtp else "HYLE_QWEN35_TEXT_V1"
    with (args.output / "weights.tsv").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{signature}\n{shard.as_posix()}\n{inventory['artifact_identity_sha256']}\n")
        for name, dtype, shape, offset, count in selected:
            stream.write("\t".join((name, dtype, str(len(shape)), *(str(d) for d in shape), str(offset), str(count))) + "\n")
    for name, values in (("prompt.ids", prompt), ("continuation.ids", continuation)):
        (args.output / name).write_text(" ".join(map(str, values)) + "\n", encoding="ascii")
    receipt = dict(kind="qwen35_text_input_preparation_no_model_execution",
        source_id=inventory["artifact_identity_sha256"], original_files=records,
        text_tensor_count=320, text_weight_bytes=3763655360,
        prompt=args.prompt, prompt_ids=prompt, continuation_ids=continuation,
        checkpoint=str(shard), script_sha256=digest(Path(__file__)))
    if args.mtp:
        receipt.update(weight_selection="text+mtp", manifest_signature=signature,
            selected_tensor_count=len(selected), selected_weight_bytes=sum(item[4] for item in selected),
            mtp_tensor_count=15, mtp_weight_bytes=121656320)
    (args.output / "input.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"text_tensors": 320, "prompt_tokens": len(prompt), "output": str(args.output)}
    if args.mtp:
        summary.update(mtp_tensors=15, selected_tensors=len(selected))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
