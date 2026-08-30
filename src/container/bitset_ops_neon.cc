#include <arm_neon.h>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace hyle::container::detail {

std::size_t bitset_count_neon(const std::uint64_t* words,
                              std::size_t count) noexcept {
  std::size_t result = 0;
  std::size_t i = 0;
#if defined(__ARM_FEATURE_DOTPROD)
  // Keep byte popcounts byte-wide and accumulate four groups at once with
  // UDOT. Converting every 64-bit lane back to uint64_t costs three pairwise
  // widening instructions per vector and was measurably slower on Apple M4.
  // Bound each block so the combined uint32 accumulators cannot overflow.
  constexpr std::size_t kWordsPerBatch = 8;
  constexpr std::size_t kMaxBatchesPerBlock = 1u << 20;
  const uint8x16_t ones = vdupq_n_u8(1);
  while (i + kWordsPerBatch <= count) {
    const std::size_t available = (count - i) / kWordsPerBatch;
    const std::size_t batches =
        available < kMaxBatchesPerBlock ? available : kMaxBatchesPerBlock;
    const std::size_t end = i + batches * kWordsPerBatch;
    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);
    uint32x4_t sum2 = vdupq_n_u32(0);
    uint32x4_t sum3 = vdupq_n_u32(0);
    for (; i < end; i += kWordsPerBatch) {
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(words + i);
      sum0 = vdotq_u32(sum0, vcntq_u8(vld1q_u8(bytes)), ones);
      sum1 = vdotq_u32(sum1, vcntq_u8(vld1q_u8(bytes + 16)), ones);
      sum2 = vdotq_u32(sum2, vcntq_u8(vld1q_u8(bytes + 32)), ones);
      sum3 = vdotq_u32(sum3, vcntq_u8(vld1q_u8(bytes + 48)), ones);
    }
    const uint32x4_t sum =
        vaddq_u32(vaddq_u32(sum0, sum1), vaddq_u32(sum2, sum3));
    result += static_cast<std::size_t>(vaddlvq_u32(sum));
  }
#endif
  // Baseline Armv8 has CNT+UADDLV even when the target contract does not
  // guarantee dot-product. This also handles the bounded tail of the UDOT
  // loop without reading past the logical block array.
  for (; i + 2 <= count; i += 2) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(words + i);
    result += static_cast<std::size_t>(vaddlvq_u8(vcntq_u8(vld1q_u8(bytes))));
  }
  for (; i < count; ++i) result += std::popcount(words[i]);
  return result;
}

}  // namespace hyle::container::detail
