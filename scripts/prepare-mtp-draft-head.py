#!/usr/bin/env python3
"""Prepare a source-bound full-vocabulary Q4_K proposal head from local weights.

Only this one matrix is converted. The target head and embeddings stay in their
original prepared source. Quantization runs offline through an explicitly named
local ggml library, using its Q4_K reference quantizer without an importance
matrix. No model, library, tokenizer or data is downloaded.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import re
import shlex


SIGNATURE = "HYLE_QWEN_DRAFT_HEAD_Q4_K_V1"


def source_head(inputs: Path) -> dict:
    manifest = inputs / "weights.tsv"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4 or lines[0] != "HYLE_QWEN_ROLE_FILES_V2":
        raise ValueError("an existing dense HF Qwen V2 prepared source is required")
    identity = lines[1]
    if not re.fullmatch("[0-9a-f]{64}", identity):
        raise ValueError("invalid prepared source identity")
    geometry = shlex.split(lines[2])
    if len(geometry) < 4 or geometry[0] != "geometry":
        raise ValueError("missing prepared geometry")
    hidden, vocabulary = int(geometry[1]), int(geometry[3])
    if hidden <= 0 or hidden % 256 or not 0 < vocabulary <= 2**31 - 1:
        raise ValueError("Q4_K head requires positive bounded rows and complete K blocks")
    names = ("lm_head.weight", "model.language_model.embed_tokens.weight")
    found = {}
    for line in lines[3:]:
        fields = shlex.split(line)
        if not fields or fields[0] not in names:
            continue
        if fields[0] in found:
            raise ValueError("duplicate original head role")
        found[fields[0]] = fields
    selected = next((found[name] for name in names if name in found), None)
    if selected is None:
        raise ValueError("prepared source has no original dense vocabulary head")
    if len(selected) != 8 or int(selected[2]) != 2:
        raise ValueError("head must be one original plain matrix without source codec operands")
    dtype, shape = selected[1], tuple(map(int, selected[3:5]))
    path = Path(selected[5])
    if not path.is_absolute():
        path = inputs / path
    offset, size = map(int, selected[6:8])
    if dtype not in ("BF16", "F16", "F32") or shape != (vocabulary, hidden):
        raise ValueError("head dtype or logical shape differs from source geometry")
    unit = 4 if dtype == "F32" else 2
    if offset < 0 or size != vocabulary * hidden * unit or not path.is_file() or offset + size > path.stat().st_size:
        raise ValueError("head byte range differs from its original local source")
    return dict(identity=identity, hidden=hidden, vocabulary=vocabulary,
                dtype=dtype, path=path.resolve(), offset=offset, bytes=size)


class Quantizer:
    def __init__(self, library: Path):
        if not library.is_file():
            raise ValueError("existing local ggml library required")
        self.directory = os.add_dll_directory(str(library.parent)) if os.name == "nt" else None
        self.library = ctypes.CDLL(str(library))
        lib = self.library
        lib.ggml_type_size.argtypes = [ctypes.c_int]
        lib.ggml_type_size.restype = ctypes.c_size_t
        lib.ggml_blck_size.argtypes = [ctypes.c_int]
        lib.ggml_blck_size.restype = ctypes.c_int64
        lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.ggml_validate_row_data.restype = ctypes.c_bool
        lib.ggml_quantize_free.argtypes = []
        lib.ggml_quantize_free.restype = None
        if lib.ggml_type_size(12) != 144 or lib.ggml_blck_size(12) != 256 or lib.ggml_quantize_requires_imatrix(12):
            raise ValueError("local quantizer does not expose the required Q4_K reference format")

    def rows(self, values, columns: int) -> bytes:
        import numpy as np
        if columns <= 0 or columns % 256 or not values.size or values.size % columns:
            raise ValueError("quantizer requires complete nonempty Q4_K rows")
        if values.dtype != np.float32 or not values.flags.c_contiguous or not np.isfinite(values).all():
            raise ValueError("quantizer requires finite contiguous original F32 rows")
        count = values.size // columns
        size = count * (columns // 256) * 144
        result = ctypes.create_string_buffer(size)
        actual = self.library.ggml_quantize_chunk(12, values.ctypes.data, result, 0, count, columns, None)
        if actual != size or not self.library.ggml_validate_row_data(12, result, size):
            raise ValueError("quantizer returned invalid or incomplete Q4_K rows")
        return result.raw

    def close(self):
        self.library.ggml_quantize_free()
        if self.directory is not None:
            self.directory.close()


def prepare(inputs: Path, library: Path, version: str, output: Path) -> dict:
    import numpy as np
    if not re.fullmatch(r"llama\.cpp@[0-9a-f]{40}", version):
        raise ValueError("record the actual local quantizer commit as llama.cpp@<40 hex>")
    head = source_head(inputs)
    quantizer = Quantizer(library)
    try:
        output.mkdir(parents=True, exist_ok=False)
        payload = output / "draft-head-q4_k.bin"
        source_digest, encoded_digest = hashlib.sha256(), hashlib.sha256()
        total = 0
        with head["path"].open("rb") as source, payload.open("xb") as target:
            source.seek(head["offset"])
            unit = 4 if head["dtype"] == "F32" else 2
            chunk_rows = max(1, min(1024, (8 * 1024 * 1024) // (head["hidden"] * 4)))
            for begin in range(0, head["vocabulary"], chunk_rows):
                rows = min(chunk_rows, head["vocabulary"] - begin)
                raw = source.read(rows * head["hidden"] * unit)
                if len(raw) != rows * head["hidden"] * unit:
                    raise ValueError("original head became truncated during preparation")
                source_digest.update(raw)
                if head["dtype"] == "BF16":
                    values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                else:
                    values = np.frombuffer(raw, dtype="<f4" if unit == 4 else "<f2").astype(np.float32)
                encoded = quantizer.rows(values, head["hidden"])
                target.write(encoded)
                encoded_digest.update(encoded)
                total += len(encoded)
        expected = head["vocabulary"] * (head["hidden"] // 256) * 144
        if total != expected or payload.stat().st_size != expected:
            raise ValueError("complete encoded head byte extent differs")
        # Publish metadata only after the complete payload has been written.
        metadata = (f'{SIGNATURE}\n{head["identity"]}\n{encoded_digest.hexdigest()}\n'
                    f'{version}:Q4_K:reference:no-imatrix\n'
                    f'{head["vocabulary"]} {head["hidden"]}\n'
                    f'"{payload.name}" {expected}\n')
        with (output / "mtp-draft-head.tsv").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(metadata)
        return dict(rows=head["vocabulary"], columns=head["hidden"], bytes=expected,
                    source_head_sha256=source_digest.hexdigest(), payload_sha256=encoded_digest.hexdigest())
    finally:
        quantizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--quantizer-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.inputs.resolve(), args.ggml_library.resolve(), args.quantizer_version, args.output.resolve())
    print("Prepared full Q4_K draft head:", result)


if __name__ == "__main__":
    main()
