#include "../../engine/inference/output_pool_capacity.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/mem/resource.h"
#include "hyle/mem/range_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace {
namespace c = hyle::compute;
namespace i = hyle::inference;
namespace m = hyle::mem;
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class T> T take(hyle::Result<T> value) {
  if (!value) throw std::runtime_error("unexpected original lease/capacity error");
  return std::move(*value);
}
void check(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error("unexpected original lease operation error");
}
void invalid_and_overflow() {
  const std::array one{std::size_t{1}}, zero{std::size_t{0}};
  for (const auto alignment : {std::size_t{0}, std::size_t{3}}) {
    const auto result = i::checked_output_pool_capacity({}, 0, alignment);
    require(!result && result.error().IsInvalidArgument() && !result.error().has_native_error(),
        "invalid explicit alignment was hidden by the empty branch");
  }
  const auto empty = take(i::checked_output_pool_capacity({}, 0, 64));
  require(!empty.backing_bytes && !empty.max_blocks, "empty outputs acquired a pool requirement");
  const auto no_copies = i::checked_output_pool_capacity(one, 0, 256);
  const auto no_bytes = i::checked_output_pool_capacity(zero, 1, 256);
  require(!no_copies && no_copies.error().IsInvalidArgument() &&
      !no_bytes && no_bytes.error().IsInvalidArgument(), "zero output bytes/copies did not reject");
  constexpr auto max_size = (std::numeric_limits<std::size_t>::max)();
  const std::array align_overflow{max_size};
  const std::array sum_overflow{max_size, std::size_t{1}};
  const std::array product_overflow{max_size / 2 + 1};
  for (const auto &result : {
      i::checked_output_pool_capacity(align_overflow, 1, 256),
      i::checked_output_pool_capacity(sum_overflow, 1, 1),
      i::checked_output_pool_capacity(product_overflow, 2, 1),
      i::checked_output_pool_capacity(one, max_size, 1),
      i::checked_output_pool_capacity(one, (std::numeric_limits<std::uint32_t>::max)(), 1)})
    require(!result && result.error().IsOutOfRange() && !result.error().has_native_error(),
        "capacity overflow was narrowed, softened or given native provenance");
  const std::array pair{std::size_t{1}, std::size_t{1}};
  const auto arithmetic_uint32_bound = take(i::checked_output_pool_capacity(pair,
      ((std::numeric_limits<std::uint32_t>::max)() - std::size_t{1}) / 2, 1));
  require(arithmetic_uint32_bound.max_blocks == (std::numeric_limits<std::uint32_t>::max)() - 1u,
      "representable uint32 arithmetic block bound was rejected");
  // A representable count is not a promise that the original range metadata
  // can represent it. This rejection occurs before allocating its nodes.
  const auto range_decline = m::RangeAllocator::create(1,
      {.max_allocations = arithmetic_uint32_bound.max_blocks});
  require(!range_decline && range_decline.error().IsInvalidArgument(),
      "arithmetic capacity bypassed the original range metadata bound");
}
void exact_boundary() {
  const std::array bytes{std::size_t{256}};
  const auto plan = take(i::checked_output_pool_capacity(bytes, 2, 256));
  auto exact = take(c::LeasePool::create(m::system_resource(), plan.backing_bytes,
      {.max_blocks = plan.max_blocks, .backing_alignment = 256}));
  auto first = take(exact.allocate(256));
  auto second = take(exact.allocate(256));
  require(exact.stats().live_blocks == 2, "exact cold capacity did not fit its last actual block");
  auto short_pool = take(c::LeasePool::create(m::system_resource(), plan.backing_bytes - 1,
      {.max_blocks = plan.max_blocks, .backing_alignment = 256}));
  auto held = take(short_pool.allocate(256));
  const auto refused = short_pool.allocate(256);
  require(!refused && refused.error().IsResourceExhausted() && short_pool.stats().live_blocks == 1,
      "one-byte-short original pool fabricated capacity or consumed the live peer");
  (void)first; (void)second; (void)held;
}
void unaligned_outputs_keep_original_readers() {
  const std::array bytes{std::size_t{260}}; // F32 [1,65], not a padded model vocabulary.
  const auto plan = take(i::checked_output_pool_capacity(bytes, 2, 256));
  auto pool = take(c::LeasePool::create(m::system_resource(), plan.backing_bytes,
      {.max_blocks = plan.max_blocks, .backing_alignment = 256}));
  auto writer_a = take(pool.allocate(260)); check(take(writer_a.view()).fill(std::byte{0x1a}));
  auto a = take(writer_a.publish()); auto retained_a = a;
  auto writer_b = take(pool.allocate(260)); check(take(writer_b.view()).fill(std::byte{0x2b}));
  auto b = take(writer_b.publish());
  const auto view_a = take(a.view()), view_b = take(b.view());
  require(view_a.offset() % 256 == 0 && view_b.offset() % 256 == 0 &&
      (view_a.offset() + 260 <= view_b.offset() || view_b.offset() + 260 <= view_a.offset()),
      "original output writers overlap or ignore their requested allocation alignment");
  a.reset();
  const auto still_held = pool.allocate(260);
  require(!still_held && still_held.error().IsResourceExhausted(), "a copied reader stopped owning its original block");
  retained_a.reset();
  auto recovered = take(pool.allocate(260));
  std::array<std::byte, 260> observed{}; check(take(b.view()).read(observed));
  for (const auto value : observed) require(value == std::byte{0x2b}, "peer output changed during actual block recovery");
  auto old_formula = take(c::LeasePool::create(m::system_resource(), 260 * 2 + 1,
      {.max_blocks = 2, .backing_alignment = 256}));
  auto old_first = take(old_formula.allocate(260));
  const auto old_second = old_formula.allocate(260);
  require(!old_second && old_second.error().IsResourceExhausted(),
      "unaligned-row regression fixture did not exercise the actual inter-block alignment gap");
  const auto smaller = take(i::checked_output_pool_capacity(std::array{std::size_t{65}}, 2, 64));
  auto explicit_pool = take(c::LeasePool::create(m::system_resource(), smaller.backing_bytes,
      {.max_blocks = smaller.max_blocks, .backing_alignment = 64}));
  auto explicit_a = take(explicit_pool.allocate(65, 64));
  auto explicit_b = take(explicit_pool.allocate(65, 64));
  require(explicit_pool.stats().live_blocks == 2, "capacity helper silently imposed a fixed256 alignment");
  (void)recovered; (void)old_first; (void)explicit_a; (void)explicit_b;
}
void original_budget_outlives_pool_through_reader() {
  const auto resource = m::system_resource();
  const std::array limits{c::ResourceBudgetLimit{resource, 1u << 20}};
  auto budget = take(c::ResourceBudget::create({.limits = limits}));
  const auto before = take(budget.usage(resource)).live_bytes;
  c::ReadLease escaped;
  {
    const auto plan = take(i::checked_output_pool_capacity(std::array{std::size_t{260}}, 2, 256));
    auto pool = take(c::LeasePool::create(resource, plan.backing_bytes,
        {.max_blocks = plan.max_blocks, .backing_alignment = 256, .resource_budget = budget}));
    auto writer = take(pool.allocate(260));
    check(take(writer.view()).fill(std::byte{0x37})); escaped = take(writer.publish());
  }
  require(take(budget.usage(resource)).live_bytes > before,
      "destroying the pool released backing still owned by a retained original reader");
  std::array<std::byte, 260> observed{}; check(take(escaped.view()).read(observed));
  for (const auto value : observed) require(value == std::byte{0x37}, "escaped original pool backing was replaced");
  escaped.reset();
  require(take(budget.usage(resource)).live_bytes == before,
      "final original reader release leaked its pool budget/metadata charge");
}
} // namespace

int main() {
  try {
    invalid_and_overflow(); exact_boundary(); unaligned_outputs_keep_original_readers();
    original_budget_outlives_pool_through_reader();
    std::puts("{\"status\":\"PASS\",\"actual_lease_pool\":true,\"explicit_alignment\":true,\"checked_overflow\":true,\"unaligned_rows\":true,\"retained_reader_recovery\":true,\"budget_lifetime\":true,\"native_submissions\":0}");
    return 0;
  } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
