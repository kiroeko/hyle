#pragma once

#include <ATen/ATen.h>

namespace hyle::compute::torch_shared {

// Original ATen conversion followed by public NaN/finite-overflow semantics.
// All payload transformations below are original Tensor kernels, not HYLE
// loops.
inline at::Tensor storage_cast(const at::Tensor &input, at::ScalarType target) {
  if (input.scalar_type() == target)
    return input;
  auto value = input.to(at::kFloat);
  if (target == at::kFloat) {
    // Restore the declared widening payload bits, including signalling NaNs.
    // Native arithmetic conversions can quiet them before a storage write.
    const bool half = input.scalar_type() == at::kHalf;
    const bool bf16 = input.scalar_type() == at::kBFloat16;
    const bool e4 = input.scalar_type() == at::kFloat8_e4m3fn;
    auto bits = input.flatten()
                    .view(half || bf16 ? at::kShort : at::kByte)
                    .to(at::kInt);
    auto sign = bits.bitwise_and(half || bf16 ? 0x8000 : 0x80)
                    .bitwise_left_shift(half || bf16 ? 16 : 24);
    at::Tensor nan, payload;
    if (half) {
      nan = bits.bitwise_and(0x7c00).eq(0x7c00).logical_and(
          bits.bitwise_and(0x3ff).ne(0));
      payload = bits.bitwise_and(0x3ff)
                    .bitwise_left_shift(13)
                    .bitwise_or(0x7f800000)
                    .bitwise_or(sign);
    } else if (bf16) {
      nan = bits.bitwise_and(0x7f80).eq(0x7f80).logical_and(
          bits.bitwise_and(0x7f).ne(0));
      payload = bits.bitwise_and(0xffff).bitwise_left_shift(16);
    } else if (e4) {
      nan = bits.bitwise_and(0x7f).eq(0x7f);
      payload = sign.bitwise_or(0x7fc00000);
    } else {
      nan = bits.bitwise_and(0x7c).eq(0x7c).logical_and(
          bits.bitwise_and(3).ne(0));
      payload = bits.bitwise_and(3)
                    .bitwise_left_shift(21)
                    .bitwise_or(0x7f800000)
                    .bitwise_or(sign);
    }
    auto output = at::where(nan, payload, value.flatten().view(at::kInt));
    return output.view(at::kFloat).view(input.sizes());
  }
  const bool e4 = target == at::kFloat8_e4m3fn;
  const bool e5 = target == at::kFloat8_e5m2;
  auto converted = value;
  if (e4 || e5) {
    const double maximum = e4 ? 448 : 57344;
    converted = value.clamp(-maximum, maximum);
    if (e5)
      converted = at::where(value.isinf(), value, converted);
  }
  auto output = converted.to(target);
  const auto storage = e4 || e5 ? at::kByte : at::kShort;
  auto bits = output.flatten().view(storage);
  const int positive = e4                    ? 0x7f
                       : e5                  ? 0x7e
                       : target == at::kHalf ? 0x7fff
                                             : 0x7fc0;
  const int negative = e4 ? 0xff : e5 ? 0xfe : target == at::kHalf ? -1 : -64;
  // Arithmetic widening can canonicalize a negative signalling NaN on CUDA.
  // The storage Cast contract instead preserves the original input sign.
  const auto source_type = input.scalar_type();
  const bool byte_source = source_type == at::kFloat8_e4m3fn || source_type == at::kFloat8_e5m2;
  const auto source_bits = input.flatten().view(byte_source ? at::kByte
      : source_type == at::kFloat ? at::kInt : at::kShort);
  const auto negative_source = byte_source ? source_bits.bitwise_and(0x80).ne(0)
                                          : source_bits.lt(0);
  auto nan = at::where(negative_source, at::full_like(bits, negative),
                       at::full_like(bits, positive));
  if (target == at::kBFloat16) {
    // Match SIMD's upper-payload preservation plus quiet bit. Derive payload
    // from source storage before an arithmetic widening can canonicalize NaNs.
    const auto raw = source_bits.to(at::kInt);
    auto payload = source_type == at::kFloat
        ? raw.bitwise_right_shift(16).bitwise_and(0x7f)
        : source_type == at::kHalf
        ? raw.bitwise_right_shift(3).bitwise_and(0x7f)
        : source_type == at::kFloat8_e5m2
        ? raw.bitwise_and(3).bitwise_left_shift(5)
        : at::zeros_like(raw); // E4M3's declared F32 NaN has only its quiet bit.
    payload = payload.bitwise_or(0x7fc0);
    nan = at::where(negative_source, payload.bitwise_or(0x8000), payload)
              .to(storage);
  }
  bits = at::where(value.flatten().isnan(), nan, bits);
  return bits.view(target).view(input.sizes());
}

} // namespace hyle::compute::torch_shared
