#include "../compute_models/mlx_moe_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

TEST(AppleMlxMoeGraph, MixedBitMoeKeepsTypedMetadataAndIndependentSlots) {
  namespace f = mlx_moe_test;
  namespace a = hyle::compute::apple;
  auto context = a::Context::create().value();
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = a::MemoryResource::create(context).value();
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::g::apple::execution_source(context).value(), f::g::StorageDomainId{0}}};
  f::run(memory.ref(), {},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {domains, targets}, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> bytes) {
        return destination.write(bytes);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> bytes) {
        return source.read(bytes);
      });
}
