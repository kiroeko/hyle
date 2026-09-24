"""Extract a bounded Qwen3.5-2B D5 fixture from the user's verified checkpoint.

This is a test consumer, not a model loader. It copies complete selected tensors
without changing their bytes and creates one deterministic BF16 MatMul oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/implementation/tasks/compute/qwen35-2b-local-input.json"
SHARD = "model.safetensors-00001-of-00001.safetensors"
TENSORS = (
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.conv1d.weight",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "mtp.fc.weight",
)
EMBEDDING = "model.language_model.embed_tokens.weight"
EMBEDDING_ROWS = (0, 42, 151643, 151644, 151645, 248318, 248319)
DTYPE_BYTES = {"BF16": 2, "F32": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    return name.replace(".", "_") + ".bin"


def checked_header(stream, shard_size: int) -> tuple[int, dict]:
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("safetensors shard has no complete header length")
    header_bytes = struct.unpack("<Q", raw)[0]
    if header_bytes == 0 or header_bytes + 8 > shard_size:
        raise ValueError("invalid safetensors header size")
    header = json.loads(stream.read(header_bytes))
    return 8 + header_bytes, header


def extract(model_dir: Path, output: Path) -> dict:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    expected = next(item for item in inventory["files"] if item["path"] == SHARD)
    shard = model_dir / SHARD
    if not shard.is_file() or shard.stat().st_size != expected["size_bytes"]:
        raise ValueError("local Qwen shard is missing or has the wrong byte size")

    output.mkdir(parents=True, exist_ok=True)
    tensors_dir = output / "tensors"
    tensors_dir.mkdir(exist_ok=True)
    records = []
    with shard.open("rb") as stream:
        payload_base, header = checked_header(stream, shard.stat().st_size)
        for name in TENSORS:
            item = header.get(name)
            if not isinstance(item, dict):
                raise ValueError(f"required tensor missing: {name}")
            dtype = item["dtype"]
            shape = item["shape"]
            begin, end = item["data_offsets"]
            elements = 1
            for dimension in shape:
                if not isinstance(dimension, int) or dimension <= 0:
                    raise ValueError(f"invalid shape for {name}")
                elements *= dimension
            if dtype not in DTYPE_BYTES or end - begin != elements * DTYPE_BYTES[dtype]:
                raise ValueError(f"invalid storage extent for {name}")
            if payload_base + end > shard.stat().st_size:
                raise ValueError(f"tensor exceeds shard: {name}")
            stream.seek(payload_base + begin)
            payload = stream.read(end - begin)
            if len(payload) != end - begin:
                raise ValueError(f"short read for {name}")
            destination = tensors_dir / safe_name(name)
            destination.write_bytes(payload)
            records.append({
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "source_data_offsets": [begin, end],
                "bytes": len(payload),
                "file": destination.relative_to(output).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })

        embedding = header.get(EMBEDDING)
        if not isinstance(embedding, dict) or embedding.get("dtype") != "BF16" or \
                embedding.get("shape") != [248320, 2048]:
            raise ValueError("Qwen embedding anchor changed")
        embedding_begin, embedding_end = embedding["data_offsets"]
        row_bytes = 2048 * DTYPE_BYTES["BF16"]
        selected = bytearray()
        source_ranges = []
        for row in EMBEDDING_ROWS:
            begin = embedding_begin + row * row_bytes
            end = begin + row_bytes
            if end > embedding_end:
                raise ValueError("selected embedding row exceeds tensor")
            stream.seek(payload_base + begin)
            payload = stream.read(row_bytes)
            if len(payload) != row_bytes:
                raise ValueError("short embedding row read")
            selected.extend(payload)
            source_ranges.append([begin, end])
        embedding_file = output / "embedding_rows_7x2048.bf16"
        embedding_file.write_bytes(selected)

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required to build the independent oracle") from error

    projection_name = "model.language_model.layers.0.linear_attn.out_proj.weight"
    projection_record = next(item for item in records if item["name"] == projection_name)
    if projection_record["dtype"] != "BF16" or projection_record["shape"] != [2048, 2048]:
        raise ValueError("Qwen projection anchor changed")
    weight_bits = np.fromfile(output / projection_record["file"], dtype="<u2").reshape(2048, 2048)
    weight = (weight_bits.astype(np.uint32) << 16).view(np.float32)
    source = np.array([((index * 17) % 101 - 50) / 256.0 for index in range(2048)], dtype=np.float32)
    bits = source.view(np.uint32)
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    activation_bits = ((bits + rounding) >> 16).astype("<u2")
    activation = (activation_bits.astype(np.uint32) << 16).view(np.float32)
    expected_output = (weight @ activation).astype("<f4")
    projection_activation_file = output / "activation_1x2048.bf16"
    projection_expected_file = output / "expected_out_proj_1x2048.f32"
    activation_bits.tofile(projection_activation_file)
    expected_output.tofile(projection_expected_file)

    mtp_name = "mtp.fc.weight"
    mtp_record = next(item for item in records if item["name"] == mtp_name)
    if mtp_record["dtype"] != "BF16" or mtp_record["shape"] != [2048, 4096]:
        raise ValueError("Qwen MTP projection anchor changed")
    mtp_bits = np.fromfile(output / mtp_record["file"], dtype="<u2").reshape(2048, 4096)
    mtp_weight = (mtp_bits.astype(np.uint32) << 16).view(np.float32)
    mtp_oracles = []
    for seed in (3, 11):
        source = np.array(
            [((index * 13 + seed * 7) % 127 - 63) / 512.0 for index in range(4096)],
            dtype=np.float32)
        bits = source.view(np.uint32)
        rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
        activation_bits = ((bits + rounding) >> 16).astype("<u2")
        activation = (activation_bits.astype(np.uint32) << 16).view(np.float32)
        mtp_expected = (mtp_weight @ activation).astype("<f4")
        activation_file = output / f"mtp_activation_seed{seed}_1x4096.bf16"
        expected_file = output / f"mtp_expected_seed{seed}_1x2048.f32"
        activation_bits.tofile(activation_file)
        mtp_expected.tofile(expected_file)
        mtp_oracles.append({
            "seed": seed,
            "activation": activation_file.name,
            "activation_sha256": sha256(activation_file),
            "expected": expected_file.name,
            "expected_sha256": sha256(expected_file),
        })

    receipt = {
        "schema_version": 1,
        "kind": "d5-bounded-real-checkpoint-fixture",
        "source_inventory": INVENTORY.relative_to(ROOT).as_posix(),
        "source_id": inventory["artifact_identity_sha256"],
        "source_shard": SHARD,
        "source_shard_expected_sha256": expected["sha256"],
        "source_shard_size": expected["size_bytes"],
        "source_full_hash_reused_from_verified_inventory": True,
        "payload_base": payload_base,
        "tensors": records,
        "slices": [{
            "name": EMBEDDING,
            "dtype": "BF16",
            "source_shape": [248320, 2048],
            "selected_rows": list(EMBEDDING_ROWS),
            "source_data_ranges": source_ranges,
            "shape": [len(EMBEDDING_ROWS), 2048],
            "bytes": len(selected),
            "file": embedding_file.name,
            "sha256": sha256(embedding_file),
        }],
        "oracle": {
            "operation": "BF16 [1,2048] x transpose(BF16 [2048,2048])",
            "region": "R03",
            "activation": projection_activation_file.name,
            "activation_sha256": sha256(projection_activation_file),
            "expected": projection_expected_file.name,
            "expected_sha256": sha256(projection_expected_file),
            "accumulation": "NumPy float32 matrix-vector product after exact BF16 decode",
        },
        "mtp_oracles": mtp_oracles,
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path,
                        default=os.environ.get("HYLE_QWEN35_MODEL_DIR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model_dir is None:
        parser.error("--model-dir or HYLE_QWEN35_MODEL_DIR is required")
    receipt = extract(args.model_dir.resolve(), args.output.resolve())
    print(json.dumps({"source_id": receipt["source_id"],
                      "tensors": len(receipt["tensors"]),
                      "bytes": sum(item["bytes"] for item in receipt["tensors"]) +
                               sum(item["bytes"] for item in receipt["slices"]),
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
