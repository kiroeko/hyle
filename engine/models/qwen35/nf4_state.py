"""Bounded bitsandbytes NF4 double-quant source metadata, not an executor."""
import hashlib
import json
import math
import struct

MAX_STATE_BYTES = 512
STATE_SUFFIX = ".quant_state.bitsandbytes__nf4"
SUFFIXES = ("", ".absmax", ".nested_absmax", ".nested_quant_map", ".quant_map", STATE_SUFFIX)
READER = "bitsandbytes-default@833649043474794b8fe7a4136e0c40faf077b2e0"
POLICY = dict(reader=READER, quant_type="nf4", block_size=64, nested_block_size=256,
              decode_dtype="F16", compute_dtype="F16", nested_dtype="F32",
              nibble_order="high_first", offset_dtype="F32")
POLICY_LINE = "\t".join(("nf4_double_quant", "64", "256", "F16", "F16", "F32", "high_first", READER))


def parse(raw: bytes, shape=None) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_STATE_BYTES:
        raise ValueError("original NF4 quant_state byte limit differs")
    if b'\\' in raw or any(ch > 127 for ch in raw):
        raise ValueError("invalid original NF4 quant_state string representation")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate original NF4 quant_state field")
            result[key] = value
        return result
    try:
        state = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("invalid original NF4 quant_state JSON") from error
    keys = {"quant_type", "blocksize", "dtype", "shape", "nested_blocksize", "nested_dtype", "nested_offset"}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError("original NF4 quant_state requires exactly seven fields")
    if (state["quant_type"] != "nf4" or state["dtype"] != "float16" or state["nested_dtype"] != "float32"
            or type(state["blocksize"]) is not int or state["blocksize"] != 64
            or type(state["nested_blocksize"]) is not int or state["nested_blocksize"] != 256):
        raise ValueError("original NF4 quant_state decode/block contract differs")
    dims = state["shape"]
    if (not isinstance(dims, list) or len(dims) != 2
            or any(type(d) is not int or not 0 < d <= (1 << 63)-1 for d in dims)
            or dims[0] > ((1 << 63)-1)//dims[1] or (shape is not None and dims != list(shape))):
        raise ValueError("original NF4 quant_state logical shape differs")
    offset = state["nested_offset"]
    if type(offset) not in (float, int):
        raise ValueError("original NF4 nested offset must be finite F32")
    try:
        if not math.isfinite(offset):
            raise ValueError("original NF4 nested offset must be finite F32")
        encoded = struct.pack("<f", offset)
    except (OverflowError, struct.error) as error:
        raise ValueError("original NF4 nested offset must be finite F32") from error
    if not math.isfinite(struct.unpack("<f", encoded)[0]):
        raise ValueError("original NF4 nested offset must be finite F32")
    return dict(logical_shape=dims, offset_f32_bits=encoded[::-1].hex(),
                original_json_sha256=hashlib.sha256(raw).hexdigest())


def groups(roles, config, expected):
    bases = set()
    for name in roles:
        for suffix in SUFFIXES[1:]:
            if name.endswith(suffix):
                bases.add(name[:-len(suffix)])
    result = {}
    for name in sorted(bases):
        if name not in expected or not name.endswith(".weight") or len(expected[name]) != 2:
            raise ValueError("unknown original NF4 logical role: " + name)
        if any(name+s not in roles for s in SUFFIXES):
            raise ValueError("incomplete original NF4 six-role group: " + name)
        n,k = expected[name]
        count = n*k
        if count % 2:
            raise ValueError("original NF4 flattened packed extent is odd: " + name)
        g = (count+63)//64
        shapes = ([count//2,1], [g], [(g+255)//256], [256], [16])
        types = ("U8", "U8", "F32", "F32", "F32")
        for suffix,dtype,dims in zip(SUFFIXES,types,shapes):
            role = roles[name+suffix]
            if role["dtype"] != dtype or role["shape"] != dims:
                raise ValueError("original NF4 physical type/shape differs: " + name+suffix)
        state = roles[name+STATE_SUFFIX]
        if state["dtype"] != "U8" or len(state["shape"]) != 1 or not 0 < state["shape"][0] <= MAX_STATE_BYTES:
            raise ValueError("original NF4 quant_state byte limit differs: " + name)
        result[name] = dict(logical_shape=[n,k])
    if result:
        q = config.get("quantization_config", {})
        required = dict(quant_method="bitsandbytes", bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True, bnb_4bit_quant_storage="uint8", bnb_4bit_compute_dtype="float16")
        text = config.get("text_config", config)
        if (any(type(q.get(key)) is not type(value) or q.get(key) != value for key,value in required.items())
                or text.get("dtype", text.get("torch_dtype")) != "float16"):
            raise ValueError("original NF4 source decode/compute F16 policy differs")
        for name,role in roles.items():
            if name in expected and name not in result and role["dtype"] != "F16":
                raise ValueError("original NF4 unquantized role must retain F16: " + name)
    return result
