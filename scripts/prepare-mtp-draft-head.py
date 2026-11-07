#!/usr/bin/env python3
"""Prepare a source-bound full or compact Q4_K proposal head from local weights.

Only this one matrix is converted. The target head and embeddings stay in their
original prepared source. Quantization runs offline through an explicitly named
local ggml library, using its Q4_K reference quantizer without an importance
matrix. A source-bound vocabulary can select the original rows before
quantization. No model, library, tokenizer or data is downloaded.
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
SUBSET_SIGNATURE = "HYLE_QWEN_DRAFT_HEAD_Q4_K_SUBSET_V1"
DEFAULT_VOCABULARY = "mtp-draft-vocabulary.ids"


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


def selected_vocabulary(vocabulary: Path, head: dict) -> tuple[list[str], list[int]]:
    """Validate source-bound original IDs for an independently compact head."""
    if vocabulary.stat().st_size > 256 + head["vocabulary"] * 32:
        raise ValueError("draft vocabulary exceeds its original text byte bound")
    text = vocabulary.read_text(encoding="utf-8");mapping = text.splitlines()
    if (len(mapping) < 6 or mapping[:3] != ['hyle-draft-vocabulary-v2',head['identity'],str(head['vocabulary'])]):
        raise ValueError("draft vocabulary source identity or original geometry differs")
    recommendation = mapping[3].split()
    if (len(recommendation) != 2 or recommendation[0] != 'recommended_drafts' or
            (recommendation[1] != 'none' and
             (not recommendation[1].isdigit() or not 0 < int(recommendation[1]) <= (1 << 64) - 1)) or
            mapping[4] not in ('head original', 'head prepared')):
        raise ValueError("draft vocabulary preference metadata is invalid")
    words = ' '.join(mapping[5:]).split()
    if any(not re.fullmatch(r'[0-9]+', word) for word in words):
        raise ValueError("draft vocabulary requires original nonnegative token IDs")
    ids = list(map(int, words))
    if not 0 < len(ids) < head['vocabulary'] or len(set(ids)) != len(ids) or any(n >= head['vocabulary'] for n in ids):
        raise ValueError("compact head requires distinct original vocabulary rows forming a nonempty subset")
    return mapping, sorted(ids)


def default_k3_inputs(inputs: Path, vocabulary: Path, head: dict) -> dict[str, bytes]:
    """Borrow original weight ranges; publish an explicit ready-input default."""
    lines = (inputs / "weights.tsv").read_text(encoding="utf-8").splitlines()
    geometry = shlex.split(lines[2])
    if (head["dtype"] != "BF16" or len(geometry) != 21 or geometry[0] != 'geometry' or
            int(geometry[1]) != head['hidden'] or int(geometry[3]) != head['vocabulary'] or
            geometry[18] != '1'):
        raise ValueError("automatic K3 preparation requires a dense BF16 source with one NextN layer")
    mapping, ids = selected_vocabulary(vocabulary, head)
    normalized = lines[:3];mtp = set()
    for line in lines[3:]:
        fields = shlex.split(line)
        if fields == ['kv_cache','BF16','omit_original_calibration']:
            normalized.append(line);continue
        if len(fields) < 6 or not fields[2].isdigit():
            raise ValueError("automatic K3 requires original dense BF16 role records")
        rank = int(fields[2]);path_at = 3 + rank
        if (len(fields) != rank + 6 or fields[1] not in ('BF16','F32') or
                (fields[1] == 'F32' and rank > 1)):
            raise ValueError("automatic K3 cannot select another source arithmetic policy")
        source = Path(fields[path_at])
        if not source.is_absolute():source = inputs / source
        source = source.resolve()
        if any(c in source.as_posix() for c in '\r\n\0"'):
            raise ValueError("original weight path contains an unsupported control character")
        if not source.is_file():raise ValueError("original weight source is not available locally")
        normalized.append('\t'.join([*fields[:path_at], '"'+source.as_posix()+'"', *fields[path_at+1:]]))
        if fields[0].startswith('mtp.'):mtp.add(fields[0])
    if len(mtp) != 15:
        raise ValueError("automatic K3 requires the original complete 15-role NextN inventory")
    result = {'weights.tsv': ('\n'.join(normalized)+'\n').encode('utf-8'),
              DEFAULT_VOCABULARY: ('\n'.join(mapping[:3])+ '\nrecommended_drafts 3\nhead prepared\n' +
                                      ' '.join(map(str,sorted(ids)))+'\n').encode('utf-8')}
    for name in ('prompt.ids','continuation.ids'):
        if (inputs/name).is_file():result[name]=(inputs/name).read_bytes()
    return result


def prepare(inputs: Path, library: Path, version: str, output: Path, default_vocabulary: Path | None = None,
            *, compact_vocabulary: Path | None = None) -> dict:
    import numpy as np
    if not re.fullmatch(r"llama\.cpp@[0-9a-f]{40}", version):
        raise ValueError("record the actual local quantizer commit as llama.cpp@<40 hex>")
    if default_vocabulary is not None and compact_vocabulary is not None:
        raise ValueError("choose either automatic K3 inputs or a separate compact head")
    head = source_head(inputs)
    ready_inputs = default_k3_inputs(inputs, default_vocabulary, head) if default_vocabulary is not None else {}
    selected_ids = (selected_vocabulary(compact_vocabulary, head)[1] if compact_vocabulary is not None else
                    list(map(int,ready_inputs[DEFAULT_VOCABULARY].decode('utf-8').splitlines()[5].split()))
                    if ready_inputs else None)
    mapping_identity = (hashlib.sha256(np.asarray(selected_ids,dtype='<i4').tobytes()).hexdigest()
                        if selected_ids is not None else None)
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
                if selected_ids is not None:
                    if not np.isfinite(values).all():
                        raise ValueError("quantizer requires finite original source rows")
                    import bisect
                    first=bisect.bisect_left(selected_ids,begin);last=bisect.bisect_left(selected_ids,begin+rows)
                    if first == last:
                        continue
                    at=np.asarray(selected_ids[first:last],dtype=np.int64)-begin
                    # Q4_K reference quantization is row-independent. Avoid
                    # quantizing source rows that the compact head will omit.
                    values=values.reshape(rows,head['hidden'])[at].copy(order='C')
                encoded = quantizer.rows(values, head["hidden"])
                target.write(encoded)
                encoded_digest.update(encoded)
                total += len(encoded)
        output_rows = len(selected_ids) if selected_ids is not None else head["vocabulary"]
        expected = output_rows * (head["hidden"] // 256) * 144
        if total != expected or payload.stat().st_size != expected:
            raise ValueError("complete encoded head byte extent differs")
        for name, data in ready_inputs.items():
            with (output/name).open('xb') as stream:stream.write(data)
        # Publish metadata only after the complete payload has been written.
        metadata = (f'{SUBSET_SIGNATURE if selected_ids is not None else SIGNATURE}\n{head["identity"]}\n{encoded_digest.hexdigest()}\n'
                    f'{version}:Q4_K:reference:no-imatrix\n'
                    f'{output_rows} {head["hidden"]}\n'
                    f'"{payload.name}" {expected}\n')
        if mapping_identity is not None:metadata+=mapping_identity+'\n'
        with (output / "mtp-draft-head.tsv").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(metadata)
        return dict(rows=output_rows, columns=head["hidden"], bytes=expected,
                    source_head_sha256=source_digest.hexdigest(), payload_sha256=encoded_digest.hexdigest())
    finally:
        quantizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ggml-library", type=Path, required=True)
    parser.add_argument("--quantizer-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    vocabulary = parser.add_mutually_exclusive_group()
    vocabulary.add_argument("--default-k3-vocabulary", type=Path,
                        help="prepare runnable BF16 inputs with these source-bound encoded rows; enables automatic CUDA K3")
    vocabulary.add_argument("--draft-vocabulary", type=Path,
                        help="prepare only these source-bound rows without setting an automatic draft depth")
    args = parser.parse_args()
    result = prepare(args.inputs.resolve(), args.ggml_library.resolve(), args.quantizer_version, args.output.resolve(),
                     args.default_k3_vocabulary.resolve() if args.default_k3_vocabulary is not None else None,
                     compact_vocabulary=args.draft_vocabulary.resolve() if args.draft_vocabulary else None)
    print("Prepared automatic K3 inputs:" if args.default_k3_vocabulary else
          "Prepared compact Q4_K draft head:" if args.draft_vocabulary else "Prepared full Q4_K draft head:", result)


if __name__ == "__main__":
    main()
