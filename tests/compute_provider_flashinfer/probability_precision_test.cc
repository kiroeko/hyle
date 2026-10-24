#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/providers/flashinfer.h"

namespace {
namespace c = hyle::compute;
std::uint16_t bfloat_word(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
float bfloat_value(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16U);
}
} // namespace

TEST(FlashinferProbabilityPrecision, BfloatProbabilityTileIsIndependentOfSplitKvPartials) {
  // Exact BF16 Q/K give scores [0,1] for the second causal query. Cancellation
  // in V makes the intermediate probability boundary visible after BF16 output.
  // Both references are scalar mathematics, with no model/native reference data.
  const double exponential = std::exp(-1.);
  const double mathematical = (exponential - .375) / (exponential + 1.);
  const double stored_exponential = bfloat_value(bfloat_word(static_cast<float>(exponential)));
  const double staged = (stored_exponential - .375) / (stored_exponential + 1.);
  const auto expected = bfloat_word(static_cast<float>(staged));
  ASSERT_NE(expected, bfloat_word(static_cast<float>(mathematical)));

  auto context = c::Context::create(c::cuda::device(0).value(), c::flashinfer::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto host = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost).value();
  auto queue = context.create_queue().value();
  const auto plain = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  const c::BatchCapacity capacity{
      .storage = c::BatchStorage::kRagged,
      .query_type = c::DataType::kBFloat16,
      .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain,
      .max_requests = 1, .max_query_rows = 2, .max_kv_rows = 2,
      .max_pages = 0, .max_page_refs = 0,
      .query_heads = 1, .kv_heads = 1, .qk_head_dim = 64, .value_head_dim = 64, .page_size = 1};
  auto schema = c::BatchSchema::create(capacity).value();
  const std::array<std::int32_t, 2> pointers{0, 2};
  auto layout = schema.version({pointers, pointers, {}, {}}).value();
  std::array<std::uint16_t, 128> q{}, k{}, v{};
  q[0] = q[64] = k[64] = bfloat_word(1.F);
  for (std::size_t i = 0; i < 64; ++i) { v[i] = bfloat_word(1.F); v[64 + i] = bfloat_word(-.375F); }
  const auto load = [&](std::span<const std::uint16_t> values) {
    auto buffer = hyle::mem::Buffer::create(memory.ref(), values.size_bytes(), 256).value();
    EXPECT_TRUE(buffer.view().write(std::as_bytes(values)).ok());
    return c::ReadLease::adopt(std::move(buffer)).value();
  };
  const std::array inputs{load(q), load(k), load(v)};
  auto program = c::BatchProgram::prepare(context, schema,
      {.resource = memory.ref(), .host_workspace = host.ref(), .scale = 1.F, .causal = true}).value();
  auto slot = c::BatchInvocation::create(program, queue).value();
  auto prepared = slot.prepare_batch(layout).value();
  ASSERT_FALSE(slot.report().bf16_partials);
  auto done = prepared.run(inputs).value();
  ASSERT_TRUE(done.wait().ok());
  auto result = slot.output().value();
  std::array<std::uint16_t, 128> actual{};
  ASSERT_TRUE(result.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(actual[i], bfloat_word(1.F));
    EXPECT_EQ(actual[64 + i], expected) << i;
  }
  result.reset(); done = {}; prepared = {};
  ASSERT_TRUE(slot.close().ok());
  ASSERT_TRUE(context.drain().ok());
}
