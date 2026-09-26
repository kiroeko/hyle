#pragma once

#include <cstdint>
#include <optional>

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
inline constexpr StorageFormatId kFloat8GroupedWeightFloat32Block128{
    0x48594c454c4f5750ULL, 6};

// GGUF/ggml native block encodings. These identities describe existing model
// bytes and are independent of the executing route. They deliberately do not
// expose a quantizer or promise that every backend accepts every format.
enum class GgufQuantizedType : std::uint8_t {
  kQ3K,
  kQ4K,
  kQ5K,
  kQ6K,
  kQ8_0,
  kIq3Xxs,
  kIq3S,
  kIq4Nl,
  kIq4Xs,
  kMxfp4,
};

inline constexpr StorageFormatId kGgufQ3K{0x48594c4547475546ULL, 1};
inline constexpr StorageFormatId kGgufQ4K{0x48594c4547475546ULL, 2};
inline constexpr StorageFormatId kGgufQ5K{0x48594c4547475546ULL, 3};
inline constexpr StorageFormatId kGgufQ6K{0x48594c4547475546ULL, 4};
inline constexpr StorageFormatId kGgufQ8_0{0x48594c4547475546ULL, 5};
inline constexpr StorageFormatId kGgufIq3Xxs{0x48594c4547475546ULL, 6};
inline constexpr StorageFormatId kGgufIq3S{0x48594c4547475546ULL, 7};
inline constexpr StorageFormatId kGgufIq4Nl{0x48594c4547475546ULL, 8};
inline constexpr StorageFormatId kGgufIq4Xs{0x48594c4547475546ULL, 9};

HYLE_COMPUTE_OPS_API std::optional<GgufQuantizedType>
gguf_quantized_type(StorageFormatId id) noexcept;
HYLE_COMPUTE_OPS_API std::optional<GgufQuantizedType>
gguf_quantized_type(const StorageFormat &format) noexcept;

// Logical [K,N], physical ggml [N,K] with K-native blocks. logical_type is the
// floating activation/output type seen by MatMul; it is not the packed scalar
// width. The payload must already contain exact GGUF bytes.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
gguf_weight_description(GgufQuantizedType type, DataType logical_type,
                        std::int64_t k, std::int64_t n) noexcept;

// Logical and physical [E,N,K] for ggml MUL_MAT_ID. The expert and row axes
// remain outside each native K block.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
gguf_grouped_weight_description(GgufQuantizedType type, DataType logical_type,
                                std::int64_t experts, std::int64_t n,
                                std::int64_t k) noexcept;

// Logical F32 [M,K]. Physical storage is row-major E4M3FN [M,K], followed
// immediately by row-major scales: F32 [M,K/128] or UE8M0 [M,K/32].
HYLE_COMPUTE_OPS_API Result<TensorDesc>
float8_activation_description(std::int64_t m, std::int64_t k,
                              Float8ScaleMode mode) noexcept;

// Logical F32 [K,N], for ordinary MatMul. Physical storage is E4M3FN [N,K],
// followed by F32 [ceil(N/128),K/128] or UE8M0 [N,K/32] scales. F32 mode
// requires K divisible by 128; its last N scale block may be partial.
// UE8 mode requires K divisible by 32.
// F32 scales use the host's native float representation and 4-byte alignment.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
float8_weight_description(std::int64_t k, std::int64_t n,
                          Float8ScaleMode mode) noexcept;

// Logical F32 [E,N,K]. All E4M3FN codes [E,N,K] precede all native F32
// scales [E,ceil(N/128),K/128]; experts do not interleave codes and scales.
// Positive dimensions, complete K blocks, and 4-byte alignment are required.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
float8_grouped_weight_description(std::int64_t experts, std::int64_t n,
                                  std::int64_t k) noexcept;

// Logical F16 [K,N], physical row-major [N,K] in 32-value/17-byte blocks.
// Each block is one E8M0 scale byte followed by 16 packed E2M1 bytes: the
// low nibbles encode values 0..15, the high nibbles encode values 16..31.
// K is divisible by 32. This describes existing content; it is not a quantizer.
HYLE_COMPUTE_OPS_API Result<TensorDesc>
mxfp4_weight_description(std::int64_t k, std::int64_t n) noexcept;

} // namespace hyle::compute
