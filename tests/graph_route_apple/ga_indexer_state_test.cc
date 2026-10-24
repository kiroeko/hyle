#include "../compute_models/ga_indexer_state_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

namespace {
namespace f = ga_indexer_state_test;
namespace a = hyle::compute::apple;

template<class Run> void run_apple(Run run) {
  auto context = f::take(a::Context::create(), "Apple context");
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = f::take(a::MemoryResource::create(context), "Apple shared memory");
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::apple::execution_source(context), "Apple execution source"),
      f::g::StorageDomainId{0}}};
  run(memory.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {domains, targets}, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation,
         std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}
} // namespace

TEST(AppleGaIndexer, EmptySelectionKeepsTwoShapesAndIndependentResults) {
  run_apple([](auto... args) { ga_indexer_state_test::run_empty(args...); });
}
TEST(AppleGaIndexer, PositiveSelectionKeepsExistingValuesAndIds) {
  run_apple([](auto... args) { ga_indexer_state_test::run_positive(args...); });
}

TEST(AppleGaIndexer, OriginalIntegerCandidatesKeepEveryBitAndOldResults) {
  run_apple([](auto... args) { ga_indexer_state_test::run_mapping(args...); });
}
TEST(AppleGaIndexer, BfloatCacheScoresSelectionAndVersionedIntegerGeometry) {
  run_apple([](auto... args) { ga_indexer_state_test::run_roles(args...); });
}
