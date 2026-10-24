"""Prepare local source-derived Qwen text inputs; no candidate model mathematics.

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

ROOT = Path(__file__).resolve().parents[3]

if __package__:
    from . import geometry, nf4_state
else:
    import geometry
    import nf4_state


FP8_READER = "sglang-06992c92"
FP8_POLICY = dict(reader="sglang@06992c92ad4494fe9f9636e75be3e35040512447",
    activation_scale="divide", activation_floor_f32_bits="2edbe6ff",
    input_dtype="BF16", accumulation_dtype="F32", output_dtype="BF16",
    weight_scale_conversion="bf16_to_f32_exact", block_size=[128,128])
FP8_POLICY_LINE = "\t".join(("fp8_block128", "divide", "2edbe6ff", "BF16", "F32", "BF16",
                              "bf16_to_f32_exact", FP8_POLICY["reader"]))
CT_MIXED_READER = "compressed-tensors-0.17.2a20260716"
CT_MIXED_POLICY = dict(reader="compressed-tensors@0.17.2a20260716",
    channel_token="BF16_typed_divide_zero_only_epsilon_and_per_weight_product",
    local_nvfp4_input_rounding="F32", cache_dtype="BF16")
CT_MIXED_POLICY_LINE = "\t".join(("ct_mixed", "channel_token_bf16_zero_only",
    "local_nvfp4_f32", CT_MIXED_POLICY["reader"]))
BF16_CACHE_POLICY_LINE = "kv_cache\tBF16\tomit_original_calibration"
CT_ASYMMETRIC_POLICY = dict(reader="compressed-tensors@ac8e2ba82f4e0d5eaa40069f8fc642738f124cd4",
    group_size=32, scale_dtype="BF16", zero_dtype="I32", zero_packing_axis="N",
    arithmetic="scale_dtype_subtract_then_multiply")
CT_ASYMMETRIC_POLICY_LINE = "\t".join(("ct_asymmetric_int4", "32", "BF16", "packed_N",
    CT_ASYMMETRIC_POLICY["reader"]))


def ct_mixed_groups(roles: dict[str, dict], config: dict, model_geometry: dict,
                    reader: str | None, kv_cache_dtype: str | None) -> dict:
    """Identify the fixed CT reader's original groups and optional BF16-KV omission."""
    if kv_cache_dtype not in (None, "bf16"):
        raise ValueError("only explicit BF16 KV storage is currently supported")
    channel, local = {}, {}
    for name, role in roles.items():
        if name.endswith(".weight") and role["dtype"] == "F8_E4M3" and name[:-7]+".weight_scale" in roles:
            base = name[:-7]
            sf = roles[base+".weight_scale"]
            if len(role["shape"]) != 2 or sf["dtype"] != "BF16" or sf["shape"] != [role["shape"][0],1]:
                raise ValueError("original CT channel/token FP8 geometry differs: "+base)
            if any(base+s in roles for s in (".weight_scale_inv", ".weight_packed", ".qweight", ".input_scale", ".weight_scale_2")):
                raise ValueError("ambiguous original CT channel/token projection: "+base)
            channel[base] = dict(logical_shape=list(role["shape"]))
    local_bases = {n[:-14] for n,r in roles.items() if n.endswith(".weight_packed") and r["dtype"] == "U8"}
    for suffix in (".weight_global_scale", ".input_global_scale"):
        local_bases.update(n[:-len(suffix)] for n in roles if n.endswith(suffix))
    for base in sorted(local_bases):
        names = [base+s for s in (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")]
        if any(n not in roles for n in names):
            raise ValueError("incomplete original CT local NVFP4 group: "+base)
        codes,sf,wg,ig = (roles[n] for n in names)
        if codes["dtype"] != "U8" or len(codes["shape"]) != 2:
            raise ValueError("original CT local NVFP4 packed geometry differs: "+base)
        n,packed_k = codes["shape"];k=packed_k*2
        if k%16 or sf["dtype"] != "F8_E4M3" or sf["shape"] != [n,k//16] or any(r["dtype"] != "F32" or r["shape"] != [1] for r in (wg,ig)):
            raise ValueError("original CT local NVFP4 scale geometry differs: "+base)
        if any(base+s in roles for s in (".weight", ".weight_shape", ".qweight", ".qzeros", ".scales", ".g_idx", ".input_scale", ".weight_scale_2")):
            raise ValueError("ambiguous original CT local NVFP4 storage: "+base)
        local[base] = dict(logical_shape=[n,k])
    omitted = []
    for index,kind in enumerate(model_geometry["layers"]):
        if kind != "A":
            continue
        names = [f"model.language_model.layers.{index}.self_attn.{which}_scale" for which in ("k","v")]
        if not any(n in roles for n in names):
            continue
        if any(n not in roles or roles[n]["dtype"] != "BF16" or roles[n]["shape"] != [1] for n in names):
            raise ValueError("invalid original BF16 cache calibration pair: "+names[0])
        omitted.extend(names)
    if channel or local or omitted:
        q = config.get("quantization_config", {})
        if reader != CT_MIXED_READER or q.get("quant_method") != "compressed-tensors" or q.get("version") != "0.17.2.a20260716":
            raise ValueError("CT mixed roles require explicit --fp8-reader "+CT_MIXED_READER+" and its fixed producer config")
        groups = q.get("config_groups", {}).values()
        def matches(group, fmt, bits, strategy, dynamic, group_size=None):
            w,a = group.get("weights",{}), group.get("input_activations",{})
            return (group.get("format") == fmt and all(w.get(k)==v for k,v in dict(num_bits=bits,type="float",symmetric=True,dynamic=False,strategy=strategy[0],group_size=group_size).items()) and all(a.get(k)==v for k,v in dict(num_bits=bits,type="float",symmetric=True,dynamic=dynamic,strategy=strategy[1],group_size=group_size).items()))
        if channel and not any(matches(g,"float-quantized",8,("channel","token"),True) for g in groups):
            raise ValueError("CT channel/token reader arithmetic differs from source config")
        if local and not any(matches(g,"nvfp4-pack-quantized",4,("tensor_group","tensor_group"),"local",16) for g in groups):
            raise ValueError("CT local NVFP4 reader arithmetic differs from source config")
        if omitted and kv_cache_dtype != "bf16":
            raise ValueError("source KV calibration requires explicit --kv-cache-dtype bf16; FP8 KV execution is not implemented")
    elif reader == CT_MIXED_READER:
        raise ValueError("selected CT reader has no matching original weights")
    return dict(channel=channel,local=local,omitted_cache_calibration=omitted)


def float8_block_groups(roles: dict[str, dict], config: dict, reader: str | None = None) -> dict:
    """Validate the original two-role block128 storage, without reading payloads."""
    groups = {}
    for name, role in roles.items():
        if not (name.endswith(".weight_scale_inv") or role["dtype"] == "F8_E4M3"):
            continue
        # NVFP4's F8_E4M3 .weight_scale is a distinct codec, never block128.
        if name.endswith(".weight_scale"):
            continue
        base = name[:-17] if name.endswith(".weight_scale_inv") else name[:-7] if name.endswith(".weight") else None
        if base is None or base + ".weight" not in roles or base + ".weight_scale_inv" not in roles:
            raise ValueError("incomplete original FP8 block128 projection: " + name)
        weight, scale = roles[base + ".weight"], roles[base + ".weight_scale_inv"]
        if weight["dtype"] != "F8_E4M3" or len(weight["shape"]) != 2:
            raise ValueError("original FP8 block128 weight type/shape differs: " + base)
        n,k = weight["shape"]
        if k % 128 or scale["dtype"] != "BF16" or scale["shape"] != [(n+127)//128,k//128]:
            raise ValueError("original FP8 block128 scale geometry differs: " + base)
        if any(base + suffix in roles for suffix in (".qweight", ".weight_packed", ".weight_scale", ".weight_scale_2", ".input_scale")):
            raise ValueError("ambiguous original FP8 block128 projection: " + base)
        groups[base] = dict(logical_shape=[n,k])
    if groups:
        q = config.get("quantization_config", {})
        if any(q.get(key) != value for key,value in dict(quant_method="fp8",fmt="e4m3",
                activation_scheme="dynamic",weight_block_size=[128,128]).items()) or q.get("scale_fmt") is not None:
            raise ValueError("original FP8 block128 config differs from selected reader contract")
        if reader != FP8_READER:
            raise ValueError("FP8 block128 requires explicit --fp8-reader " + FP8_READER)
    elif reader is not None:
        raise ValueError("selected FP8 reader has no block128 weights")
    return groups


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


ROLE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}


def ct_packed_groups(roles: dict[str, dict]) -> dict[str, dict]:
    """Infer original CT group geometry; neither inspect values nor decode weights."""
    bases = {name[:-14] for name in roles if name.endswith(".weight_packed") and roles[name]["dtype"] != "U8"}
    bases.update(name[:-13] for name in roles if name.endswith(".weight_shape"))
    bases.update(name[:-18] for name in roles if name.endswith(".weight_zero_point"))
    result = {}
    for base in sorted(bases):
        names = [base + suffix for suffix in (".weight_packed", ".weight_scale", ".weight_shape")]
        if any(name not in roles for name in names):
            raise ValueError(f"incomplete original CT symmetric role group: {base}")
        codes, scales, shape = (roles[name] for name in names)
        if codes["dtype"] != "I32" or len(codes["shape"]) != 2 or any(type(v) is not int or v <= 0 for v in codes["shape"]):
            raise ValueError(f"original CT packed layout differs: {base}")
        n, packed_k = codes["shape"]
        k = packed_k * 8
        if k > (1 << 63) - 1 or scales["dtype"] not in ("BF16", "F16", "F32") or len(scales["shape"]) != 2:
            raise ValueError(f"original CT scale geometry differs: {base}")
        scale_n, groups = scales["shape"]
        if type(scale_n) is not int or type(groups) is not int or scale_n != n or groups <= 0 or k % groups or k // groups > (1 << 32) - 1:
            raise ValueError(f"original CT scale group geometry differs: {base}")
        if shape["dtype"] != "I64" or shape["shape"] != [2]:
            raise ValueError(f"original CT shape descriptor differs: {base}")
        if any(base + suffix in roles for suffix in (".weight", ".qweight", ".qzeros", ".scales", ".g_idx", ".input_scale", ".weight_scale_2")):
            raise ValueError(f"ambiguous original CT projection storage: {base}")
        result[base] = dict(logical_shape=[n, k], group_size=k // groups, scale_dtype=scales["dtype"])
        if base + ".weight_zero_point" in roles:
            zero = roles[base + ".weight_zero_point"]
            if zero["dtype"] != "I32" or zero["shape"] != [(n + 7) // 8, groups]:
                raise ValueError(f"original CT zero-point output-axis geometry differs: {base}")
            result[base]["asymmetric"] = True
    return result


def ct_asymmetric_policy(groups: dict[str, dict], config: dict) -> dict | None:
    """Select the original g32/BF16 reader, never infer symmetry from absent Z."""
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict):
        raise ValueError("original CT asymmetric reader config differs")
    configured = quant.get("config_groups") or {}
    if (not isinstance(configured, dict) or any(not isinstance(group, dict) or
            not isinstance(group.get("weights"), dict) for group in configured.values())):
        raise ValueError("original CT asymmetric reader config differs")
    declared = any(group["weights"].get("symmetric") is False for group in configured.values())
    if not declared and not any(group.get("asymmetric") for group in groups.values()):
        return None
    if quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "pack-quantized" or len(configured) != 1:
        raise ValueError("original CT asymmetric reader config differs")
    group = next(iter(configured.values()))
    weights = group.get("weights", {})
    required = dict(num_bits=4, type="int", strategy="group", group_size=32, symmetric=False,
                    dynamic=False, actorder=None, block_structure=None, scale_dtype=None, zp_dtype="torch.int8")
    if (any(type(weights.get(key)) is not type(value) or weights.get(key) != value for key, value in required.items())
            or group.get("format") != "pack-quantized" or group.get("targets") != ["Linear"]
            or group.get("input_activations") is not None or group.get("output_activations") is not None
            or quant.get("kv_cache_scheme") is not None):
        raise ValueError("original CT asymmetric reader config differs")
    if not groups or any(not value.get("asymmetric") or value["group_size"] != 32 or value["scale_dtype"] != "BF16"
                         for value in groups.values()):
        raise ValueError("original CT asymmetric g32/BF16 requires every packed zero-point role")
    return dict(CT_ASYMMETRIC_POLICY)


def validate_ct_shape_values(raw: bytes, geometry: dict, name: str) -> None:
    if len(raw) != 16 or list(struct.unpack("<qq", raw)) != geometry["logical_shape"]:
        raise ValueError(f"original CT weight_shape values differ: {name}")


def logical_roles(names, omitted_cache_calibration=()) -> set[str]:
    auxiliary = (".qzeros", ".scales", ".g_idx", ".input_scale", ".weight_scale", ".weight_zero_point", ".weight_scale_2", ".weight_scale_inv", ".weight_shape", ".weight_global_scale", ".input_global_scale", *nf4_state.SUFFIXES[1:])
    result = set()
    for name in names:
        if name in omitted_cache_calibration or name.endswith(auxiliary):
            continue
        logical = name[:-8] + ".weight" if name.endswith(".qweight") else name[:-14] + ".weight" if name.endswith(".weight_packed") else name
        if logical in result:
            raise ValueError(f"ambiguous original logical projection: {logical}")
        result.add(logical)
    return result


def validate_role_geometry(roles: dict[str, dict], model_geometry: dict, omitted_cache_calibration=(), nf4_groups=None) -> None:
    # Config can declare a trained NextN layer omitted by this artifact. Keep
    # that original geometry identity; validate all NextN roles if any exist.
    expected = geometry.role_shapes(model_geometry, include_mtp=any(name.startswith('mtp.') for name in roles))
    if logical_roles(roles, omitted_cache_calibration) != set(expected):
        raise ValueError("complete original text/MTP logical role inventory differs from source geometry")
    for name, shape in expected.items():
        if nf4_groups and name in nf4_groups:
            actual = nf4_groups[name]['logical_shape']
        elif name in roles:
            row = roles[name]
            actual = list(row['shape'])
            if row['dtype'] == 'U8' and len(actual) == 2 and name[:-7]+'.weight_scale_2' in roles:
                actual[1] *= 2  # Original packed two-nibble NVFP4 storage.
        elif name[:-7]+'.qweight' in roles:
            row = roles[name[:-7]+'.qweight']
            if row['dtype'] != 'I32' or len(row['shape']) != 2:
                raise ValueError('invalid original GPTQ projection axes: '+name)
            actual = [row['shape'][1], row['shape'][0]*8]
        else:
            row = roles[name[:-7]+'.weight_packed']
            if row['dtype'] not in ('I32','U8') or len(row['shape']) != 2:
                raise ValueError('invalid original CT projection axes: '+name)
            actual = [row['shape'][0], row['shape'][1]*(2 if row['dtype']=='U8' else 8)]
        if actual != list(shape):
            raise ValueError('original role shape differs from source geometry: '+name)


def _read_role_source(role: dict) -> tuple[Path, bytes | None, str]:
    """Hash one borrowed range; retain only bounded CT/NF4 source metadata."""
    name, count, offset = role["name"], role["bytes"], role["offset"]
    source = (ROOT / role["path"]).resolve()
    if type(offset) is not int or offset < 0 or offset + count > source.stat().st_size:
        raise ValueError(f"short original role source: {name}")
    received, remaining = hashlib.sha256(), count
    is_nf4_state = name.endswith(nf4_state.STATE_SUFFIX)
    if (is_nf4_state and not 0 < count <= nf4_state.MAX_STATE_BYTES) or (name.endswith('.weight_shape') and count != 16):
        raise ValueError("original CT/NF4 metadata byte limit differs: " + name)
    shape_value = bytearray() if name.endswith(".weight_shape") or is_nf4_state else None
    with source.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"short original role read: {name}")
            received.update(chunk)
            if shape_value is not None:
                shape_value.extend(chunk)
            remaining -= len(chunk)
    if any(ch in source.as_posix() for ch in "\r\n\0"):
        raise ValueError("role file path contains an unsupported control character")
    return source, bytes(shape_value) if shape_value is not None else None, received.hexdigest()


def verify_role_source(role: dict) -> tuple[Path, bytes | None]:
    source, values, sha = _read_role_source(role)
    if sha != role["sha256"]:
        raise ValueError(f"original role SHA differs: {role['name']}")
    return source, values


def prepare_role_files(path: Path, mtp: bool, config: dict, fp8_reader: str | None = None,
                       kv_cache_dtype: str | None = None) -> tuple[list[str], dict]:
    """Bind original cached physical roles without copying their storage."""
    original = json.loads(path.read_bytes())
    if not original.get("complete") or not original.get("repository") or len(original.get("revision", "")) != 40:
        raise ValueError("role source requires a complete fixed repository/revision record")
    return _prepare_role_records(original, mtp, config, verify_role_source, fp8_reader, kv_cache_dtype)


def _prepare_role_records(original, mtp, config, authenticate, fp8_reader=None, kv_cache_dtype=None):
    if fp8_reader not in (None, FP8_READER, CT_MIXED_READER):
        raise ValueError("unknown fixed FP8 reference reader")
    model_geometry = geometry.from_hf(config)
    if mtp and not model_geometry['mtp_layers']:
        raise ValueError('source geometry has no NextN layer')
    all_roles = {}
    for role in original["roles"]:
        name, dtype, shape = role["name"], role["dtype"], role["shape"]
        if (not name.startswith(("model.language_model.", "mtp.")) and not name.startswith("lm_head.")) or name in all_roles:
            raise ValueError("role source contains a duplicate or non-text role")
        if dtype not in ROLE_SIZES or len(shape) > 3:
            raise ValueError(f"unsupported original role dtype/rank: {name}")
        count = ROLE_SIZES[dtype]
        for dimension in shape:
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(f"invalid original role shape: {name}")
            count *= dimension
            if count > (1 << 64) - 1:
                raise ValueError(f"original role byte extent overflows: {name}")
        if type(role["bytes"]) is not int or count != role["bytes"]:
            raise ValueError(f"original role byte extent differs: {name}")
        all_roles[name] = role
    if mtp and not any(name.startswith('mtp.') for name in all_roles):
        raise ValueError('source has no original NextN roles')
    mixed = ct_mixed_groups(all_roles, config, model_geometry, fp8_reader, kv_cache_dtype)
    omitted = set(mixed["omitted_cache_calibration"])
    nf4_groups = nf4_state.groups(all_roles, config, geometry.role_shapes(model_geometry))
    validate_role_geometry(all_roles, model_geometry, omitted, nf4_groups)
    channel_physical = {base+suffix for base in mixed['channel'] for suffix in ('.weight','.weight_scale')}
    fp8_groups = float8_block_groups({name:role for name,role in all_roles.items() if name not in channel_physical},
        config, fp8_reader if fp8_reader == FP8_READER else None)
    ct_groups = ct_packed_groups(all_roles)
    asymmetric_policy = ct_asymmetric_policy(ct_groups, config)
    checked_shapes = {}
    for base, group_geometry in ct_groups.items():
        name = base + ".weight_shape"
        source, raw = authenticate(all_roles[name])
        validate_ct_shape_values(raw, group_geometry, name)
        checked_shapes[name] = source
    for name, group_geometry in nf4_groups.items():
        state_name = name + nf4_state.STATE_SUFFIX
        source, raw = authenticate(all_roles[state_name])
        state = nf4_state.parse(raw, group_geometry['logical_shape'])
        if state['original_json_sha256'] != all_roles[state_name]['sha256']:
            raise ValueError("original NF4 quant_state SHA differs: " + name)
        group_geometry.update(state)
        checked_shapes[state_name] = source
    lines, identities = [], []
    for role in original["roles"]:
        name, dtype, shape = role["name"], role["dtype"], role["shape"]
        count, offset = role["bytes"], role["offset"]
        source = checked_shapes[name] if name in checked_shapes else authenticate(role)[0]
        filename = source.as_posix()
        quoted = '"' + filename.replace('\\', '\\\\').replace('"', '\\"') + '"'
        lines.append("\t".join((name, dtype, str(len(shape)), *(str(d) for d in shape),
                                 quoted, str(offset), str(count))))
        identities.append({key: role[key] for key in ("name", "dtype", "shape", "bytes", "sha256")})
    identity = dict(repository=original["repository"], revision=original["revision"],
                    geometry=model_geometry, roles=sorted(identities, key=lambda item: item["name"]))
    if fp8_groups:
        identity["fp8_block128_policy"] = dict(FP8_POLICY)
    if nf4_groups:
        identity['nf4_double_quant_policy'] = dict(nf4_state.POLICY)
        identity['nf4_states'] = nf4_groups
    if asymmetric_policy:
        identity['ct_asymmetric_int4_policy'] = asymmetric_policy
    has_mixed = bool(mixed['channel'] or mixed['local'])
    if has_mixed:
        identity["ct_mixed_policy"] = dict(CT_MIXED_POLICY)
    if omitted:
        identity["cache_calibration_omission"] = dict(cache_dtype="BF16",reason="original calibration unused by explicitly selected BF16 KV storage",roles=sorted(omitted))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    chosen = [item for item in identities if item['name'] not in omitted and (mtp or not item['name'].startswith('mtp.'))]
    policy_lines = [FP8_POLICY_LINE] if fp8_groups else []
    if nf4_groups:
        policy_lines.append(nf4_state.POLICY_LINE)
        policy_lines.extend('\t'.join(('nf4_state', name, state['original_json_sha256'], state['offset_f32_bits']))
                            for name,state in sorted(nf4_groups.items()))
    if has_mixed:
        policy_lines.append(CT_MIXED_POLICY_LINE)
    if asymmetric_policy:
        policy_lines.append(CT_ASYMMETRIC_POLICY_LINE)
    if omitted:
        policy_lines.append(BF16_CACHE_POLICY_LINE)
    receipt = dict(
        source_id=fingerprint, source_repository=original["repository"], source_revision=original["revision"],
        geometry=model_geometry, metadata_tensor_count=len(lines),
        physical_roles=len(chosen), selected_tensor_count=len(chosen),
        selected_weight_bytes=sum(item["bytes"] for item in chosen),
        text_weight_bytes=sum(item["bytes"] for item in identities if item['name'] not in omitted and not item["name"].startswith("mtp.")),
        mtp_weight_bytes=sum(item["bytes"] for item in identities if item["name"].startswith("mtp.")),
        original_role_records=identity["roles"], storage="borrowed original cached files/ranges; no copied checkpoint")
    if fp8_groups:
        receipt["fp8_block128_policy"] = dict(FP8_POLICY)
        receipt["fp8_block128_groups"] = sum(mtp or not name.startswith("mtp.") for name in fp8_groups)
        receipt["metadata_fp8_block128_groups"] = len(fp8_groups)
    if nf4_groups:
        receipt['nf4_double_quant_policy'] = dict(nf4_state.POLICY)
        receipt['nf4_states'] = nf4_groups
        receipt['nf4_double_quant_groups'] = sum(mtp or not name.startswith('mtp.') for name in nf4_groups)
        receipt['metadata_nf4_double_quant_groups'] = len(nf4_groups)
        receipt['nf4_execution_scope'] = 'cold metadata only; full F16 activation/compute engine path not yet implemented'
    if has_mixed:
        receipt['ct_mixed_policy'] = dict(CT_MIXED_POLICY)
        receipt['ct_channel_token_groups'] = len(mixed['channel'])
        receipt['ct_local_nvfp4_groups'] = len(mixed['local'])
    if asymmetric_policy:
        receipt['ct_asymmetric_int4_policy'] = asymmetric_policy
        receipt['ct_asymmetric_int4_groups'] = sum(mtp or not name.startswith('mtp.') for name in ct_groups)
        receipt['metadata_ct_asymmetric_int4_groups'] = len(ct_groups)
        receipt['ct_asymmetric_execution_scope'] = 'metadata preparation only; original consumer projection binding is available, full-model execution is not verified'
    if omitted:
        receipt['cache_calibration_omission'] = dict(identity['cache_calibration_omission'],
            original_roles=[r for r in identities if r['name'] in omitted],bytes=sum(r['bytes']for r in identities if r['name']in omitted))
    return ["HYLE_QWEN_ROLE_FILES_V2", fingerprint, geometry.line(model_geometry), *policy_lines, *sorted(lines)], receipt


def prepare_safetensors(model: Path, mtp: bool, config: dict, fp8_reader: str | None = None,
                        kv_cache_dtype: str | None = None) -> tuple[list[str], dict]:
    """Index complete local text roles, hash only their original ranges, no copies."""
    model = model.resolve()
    index = model / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_bytes()).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('invalid local safetensors index')
        shard_names = sorted({filename for name,filename in mapping.items()
                              if name.startswith(('model.language_model.','mtp.','lm_head.'))})
    else:
        mapping, shard_names = None, ['model.safetensors']
    roles, seen = [], set()
    for filename in shard_names:
        if not isinstance(filename,str) or Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('unsafe local safetensors shard path')
        shard = (model / filename).resolve()
        if not shard.is_relative_to(model):
            raise ValueError('local safetensors shard escapes model directory')
        size = shard.stat().st_size
        with shard.open('rb') as stream:
            first = stream.read(8)
            if len(first) != 8:
                raise ValueError('short safetensors preamble')
            length = struct.unpack('<Q',first)[0]
            if length > size-8 or length > 64*1024*1024:
                raise ValueError('invalid safetensors header extent')
            header = json.loads(stream.read(length))
        for name,row in sorted(header.items()):
            if not name.startswith(('model.language_model.','mtp.','lm_head.')):
                continue
            if name in seen or (mapping is not None and mapping.get(name) != filename):
                raise ValueError('duplicate or misplaced local safetensors role: '+name)
            seen.add(name)
            begin,end = row['data_offsets']
            if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= size-8-length:
                raise ValueError('local role exceeds original safetensors shard: '+name)
            roles.append(dict(name=name,dtype=row['dtype'],shape=row['shape'],path=str(shard),
                              offset=8+length+begin,bytes=end-begin))
    original = dict(repository='local:'+model.as_posix(), revision=None, roles=roles)

    def discover(role):
        source,values,sha = _read_role_source(role)
        role['sha256'] = sha
        return source,values

    return _prepare_role_records(original,mtp,config,discover,fp8_reader,kv_cache_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Please calculate 2 + 3. Answer briefly.")
    parser.add_argument("--continuation", default="", help="optional raw text cached continuation")
    parser.add_argument("--mtp", action="store_true",
                        help="select the source NextN layer; does not execute MTP")
    weight_source = parser.add_mutually_exclusive_group()
    weight_source.add_argument("--role-files", type=Path,
                        help="prepared fixed-source text role record; borrow original quantized files/ranges")
    weight_source.add_argument("--gguf-role-files", type=Path,
                        help="complete acquired GGUF payload-source record; preserve original codec and converted parameters")
    parser.add_argument("--fp8-reader", choices=[FP8_READER, CT_MIXED_READER],
                        help="explicit fixed reference-reader arithmetic for original FP8/CT sources")
    parser.add_argument("--kv-cache-dtype", choices=['bf16'],
                        help="explicit BF16 KV storage; source cache calibration remains authenticated but is omitted from weight IO")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output directory must be new")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    config_path = args.model / "config.json"
    config = json.loads(config_path.read_bytes())
    records = [dict(path="config.json", size_bytes=config_path.stat().st_size, sha256=digest(config_path))]
    if args.gguf_role_files:
        if args.fp8_reader or args.kv_cache_dtype:
            raise ValueError("--fp8-reader describes original HF FP8 block128 roles, not GGUF")
        if __package__:
            from .gguf_roles import prepare_role_files as prepare_gguf_role_files
        else:
            from gguf_roles import prepare_role_files as prepare_gguf_role_files
        role_lines, role_receipt = prepare_gguf_role_files(args.gguf_role_files, args.mtp)
    elif args.role_files:
        role_lines, role_receipt = prepare_role_files(args.role_files, args.mtp, config, args.fp8_reader, args.kv_cache_dtype)
    else:
        role_lines, role_receipt = prepare_safetensors(args.model, args.mtp, config, args.fp8_reader, args.kv_cache_dtype)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True)
    prompt = encoded["input_ids"]
    continuation = tokenizer.encode(args.continuation, add_special_tokens=False)
    args.output.mkdir(parents=True)
    with (args.output / "weights.tsv").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(role_lines) + "\n")
    for name, values in (("prompt.ids", prompt), ("continuation.ids", continuation)):
        (args.output / name).write_text(" ".join(map(str, values)) + "\n", encoding="ascii")
    receipt = dict(kind="qwen_text_input_preparation_no_model_execution",
        original_files=records, tokenizer_directory=str(args.model.resolve()),
        prompt=args.prompt, prompt_ids=prompt, continuation_ids=continuation,
        script_sha256=digest(Path(__file__)), **role_receipt)
    receipt.update(weight_selection="text+mtp" if args.mtp else "text", manifest_signature=role_lines[0])
    (args.output / "input.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    expected = geometry.role_shapes(role_receipt["geometry"])
    summary = {"text_tensors":sum(not n.startswith("mtp.") for n in expected),
               "prompt_tokens":len(prompt), "output":str(args.output)}
    if args.mtp:
        summary.update(mtp_tensors=sum(n.startswith("mtp.") for n in expected),
                       selected_tensors=role_receipt["physical_roles"])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
