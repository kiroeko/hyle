#pragma once

#include <cstdint>

#include "hyle/compute/tensor_desc.h"

namespace hyle::compute {

enum class Float8ScaleMode : std::uint8_t {
  kFloat32Block128,
  kUnsignedExponent8Group32,
};

// Encoding identities are independent of the executing provider. FP8 payloads
// use E4M3FN. All matrices below are positive and contain complete scale
// blocks.
inline constexpr StorageFormatId kFloat8ActivationFloat32Block128{
    0x48594c454c4f5750ULL, 1};
inline constexpr StorageFormatId kFloat8WeightFloat32Block128{
    0x48594c454c4f5750ULL, 2};
inline constexpr StorageFormatId kFloat8ActivationUnsignedExponent8Group32{
    0x48594c454c4f5750ULL, 3};
inline constexpr StorageFormatId kFloat8WeightUnsignedExponent8Group32{
    0x48594c454c4f5750ULL, 4};
inline constexpr StorageFormatId kMxfp4Weight{0x48594c454c4f5750ULL, 5};

// Logical F32 [M,K]. Physical storage is row-major E4M3FN [M,K], followed
// immediately by row-major scales: F32 [M,K/128] or UE8M0 [M,K/32].
HYLE_COMPUTE_OPS_API Result<TensorDesc>
float8_activation_description(std::int64_t m, std::int64_t k,
                              Float8ScaleMode mode) noexcept;

// Logical F32 [K,N], for ordinary MatMul. Physical storage is E4M3FN [N,K],
// followed by F32 [N/128,K/128] or UE8M0 [N,K/32] scales. F32 mode requires
// both N and K divisible by 128; UE8 mode requires K divisible by 32.
// F32 scales use the host's native float representation and 4-byte alignment.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
float8_weight_description(std::int64_t k, std::int64_t n,
                          Float8ScaleMode mode) noexcept;

// Logical F16 [K,N], physical row-major [N,K] in 32-value/17-byte blocks.
// Each block is one E8M0 scale byte followed by 16 packed E2M1 bytes: the
// low nibbles encode values 0..15, the high nibbles encode values 16..31.
// K is divisible by 32. This describes existing content; it is not a quantizer.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
mxfp4_weight_description(std::int64_t k, std::int64_t n) noexcept;

} // namespace hyle::compute
