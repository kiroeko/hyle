"""Offline test-data codecs. These are not shipping HYLE implementations."""

import numpy as np


def q4_s128(values):
    values = np.asarray(values, dtype=np.float16)
    if values.ndim != 2 or values.shape[1] % 128 or not np.isfinite(values).all():
        raise ValueError("Q4-S128 requires finite [N,K] with K divisible by 128")
    groups = values.reshape(values.shape[0], -1, 128)
    scale = (np.max(np.abs(groups), axis=-1).astype(np.float32) * (2.0 / 15.0)).astype(np.float16)
    quotient = groups / np.where(scale[..., None] == 0, np.float16(1), scale[..., None])
    codes = (np.clip(np.rint(quotient), -8, 7).astype(np.int8) + 8).astype(np.uint8)
    codes = codes.reshape(values.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scale


def q8_b32(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 32 or not np.isfinite(values).all():
        raise ValueError("Q8-B32 requires finite [N,K] with K divisible by 32")
    groups = values.reshape(values.shape[0], -1, 32)
    with np.errstate(under="ignore"):
        scale = np.max(np.abs(groups), axis=-1) / np.float32(127)
        stored_scale = scale.astype(np.float16)
    # Underflowed F16 scales decode to zero regardless of code. Canonicalize
    # their codes to zero, avoiding an infinite reciprocal of a tiny F32 scale.
    inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=stored_scale != 0)
    quotient = groups * inverse[..., None]
    # C roundf, including half ties away from zero; not numpy's nearest-even rint.
    rounded = np.copysign(np.floor(np.abs(quotient) + np.float32(0.5)), quotient)
    return rounded.astype(np.int8).reshape(values.shape), stored_scale


def encode_e4m3fn(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("offline E4M3 input must be finite")
    negative = np.signbit(values)
    absolute = np.minimum(np.abs(values), np.float32(448))
    # E4M3 subnormal quantum is 2^-9; normal significands have three fraction bits.
    exponent = np.floor(np.log2(np.maximum(absolute, np.float32(2**-6)))).astype(np.int32)
    quantum = np.exp2(np.maximum(exponent - 3, -9)).astype(np.float32)
    significand = np.rint(absolute / quantum).astype(np.int32)
    carry = significand == 16
    exponent = exponent + carry
    significand = np.where(carry, 8, significand)
    normal = ((exponent + 7) << 3) + significand - 8
    subnormal = np.rint(absolute * np.float32(512)).astype(np.int32)
    codes = np.where(absolute < np.float32(2**-6), subnormal, normal)
    return np.minimum(codes, 126).astype(np.uint8) | (negative.astype(np.uint8) << 7)


def decode_e4m3fn(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    exponent = (codes >> 3) & 15
    fraction = codes & 7
    magnitude = np.where(exponent == 0, fraction.astype(np.float32) * (2**-9),
                         (1 + fraction.astype(np.float32) / 8) * np.exp2(exponent.astype(np.int32) - 7))
    result = np.where(codes & 128, -magnitude, magnitude).astype(np.float32)
    return np.where((codes & 127) == 127, np.float32(np.nan), result)


def f8_b128(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or any(size % 128 for size in values.shape) or not np.isfinite(values).all():
        raise ValueError("F8-B128 requires finite [N,K] with N and K divisible by 128")
    n, k = values.shape
    blocks = values.reshape(n // 128, 128, k // 128, 128)
    scale = np.max(np.abs(blocks), axis=(1, 3)) / np.float32(448)
    scale = np.where(scale == 0, np.float32(1), scale)
    codes = encode_e4m3fn(blocks / scale[:, None, :, None]).reshape(n, k)
    return codes, scale


def dequantize(profile, codes, scales, shape):
    n, k = shape
    if profile == "Q4-S128":
        unpacked = np.empty((n, k), dtype=np.float32)
        unpacked[:, 0::2] = (codes & 15).astype(np.int32) - 8
        unpacked[:, 1::2] = (codes >> 4).astype(np.int32) - 8
        # W4A16 consumes the dequantized values at F16 precision.
        return (unpacked.reshape(n, -1, 128) * scales.astype(np.float32)[..., None]).astype(np.float16).astype(np.float32).reshape(n, k)
    if profile == "Q8-B32":
        return (codes.reshape(n, -1, 32).astype(np.float32) * scales.astype(np.float32)[..., None]).astype(np.float16).astype(np.float32).reshape(n, k)
    if profile == "F8-B128":
        decoded = decode_e4m3fn(codes).reshape(n // 128, 128, k // 128, 128)
        return (decoded * scales[:, None, :, None]).reshape(n, k)
    raise ValueError(f"unknown numerical profile: {profile}")
