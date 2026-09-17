"""Derive fixed quantized weights once from local data; no model downloads."""

import argparse
import ctypes
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from local_inputs import read_inventory, sha256, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--profile", choices=("Q4-S128", "Q8-B32", "F8-B128"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    inventory = read_inventory(args.inventory)
    verify(args.source, inventory)
    contract = json.loads(args.contract.read_bytes())
    if contract["source_id"] != inventory["artifact_identity_sha256"]:
        raise ValueError("quantization contract/input identity mismatch")
    if args.output.exists():
        raise ValueError("output must be a new directory; reuse by verified receipt")

    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.numpy import save_file
    import quantization

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    np.seterr(over="raise", invalid="raise", divide="raise")
    encode = {"Q4-S128": quantization.q4_s128, "Q8-B32": quantization.q8_b32,
              "F8-B128": quantization.f8_b128}[args.profile]
    index = json.loads((args.source / "model.safetensors.index.json").read_bytes())["weight_map"]
    output = {}
    summaries = []
    for item in contract["quantized_tensors"]:
        name = item["name"]
        with safe_open(args.source / index[name], framework="pt", device="cpu") as source:
            tensor = source.get_tensor(name)
            if list(tensor.shape) != item["shape"] or str(tensor.dtype) != "torch.bfloat16":
                raise ValueError(f"unexpected source tensor: {name}")
            original = tensor.float().numpy()
        codes, scales = encode(original)
        decoded = quantization.dequantize(args.profile, codes, scales, original.shape)
        error = decoded.astype(np.float64) - original
        norm = np.sqrt(np.sum(original.astype(np.float64)**2))
        summaries.append({"name": name, "shape": item["shape"],
                          "codes_bytes": codes.nbytes, "scales_bytes": scales.nbytes,
                          "max_abs_weight_error": float(np.max(np.abs(error))),
                          "relative_l2_weight_error": float(np.sqrt(np.sum(error**2)) / norm) if norm else 0.0})
        output[name + ".codes"] = np.ascontiguousarray(codes)
        output[name + ".scales"] = np.ascontiguousarray(scales)
        del tensor, original, codes, scales, decoded, error
        if len(summaries) % 24 == 0:
            print(f"{args.profile}: {len(summaries)} / {len(contract['quantized_tensors'])}", flush=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    artifact = partial / "quantized.safetensors"
    save_file(output, artifact, metadata={"source_id": contract["source_id"], "profile": args.profile})
    receipt = {"schema_version": 1, "kind": "offline_quantized_input_not_compute_verdict",
               "source_id": contract["source_id"], "profile": args.profile,
               "contract_sha256": sha256(args.contract), "script_sha256": sha256(Path(__file__)),
               "codec_sha256": sha256(Path(quantization.__file__)),
               "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")},
               "artifact": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact),
               "tensors": summaries, "preserved_weights": "borrow original model using same source_id; no duplicates",
               "loss_scope": "weight representation only; not model-quality or HYLE correctness acceptance"}
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(args.contract, partial / "contract.json")
    shutil.copyfile(__file__, partial / "prepare_quantized.py")
    shutil.copyfile(quantization.__file__, partial / "quantization.py")
    partial.rename(args.output)
    print(json.dumps({"profile": args.profile, "directory": str(args.output),
                      "bytes": receipt["bytes"], "sha256": receipt["sha256"], "tensors": len(summaries)}))


if __name__ == "__main__":
    main()
