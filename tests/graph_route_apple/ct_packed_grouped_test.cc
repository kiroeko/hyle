#include "../compute_models/ct_packed_grouped_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"

namespace {
namespace f = ct_packed_grouped_test;
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
      [&](const f::g::Graph &graph, f::g::CompileOptions options, bool) {
        return f::g::compile(graph, {domains, targets}, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation,
         std::span<const f::g::QueueBinding> bindings, bool) {
        return invocation.prepare_replay(bindings);
      });
}
} // namespace

TEST(AppleCtPackedGrouped, FlatBanksKeepTypedRoundsRealStridesAndOwnedSlots) {
  run_apple([](auto resource, auto queues, auto compile, auto write,
               auto read, auto prepare) {
    f::run(resource, queues, compile, write, read, prepare);
  });
}
TEST(AppleCtPackedGrouped, CompleteIndexValidationPrecedesExpertAddressAndRecovers) {
  run_apple([](auto resource, auto queues, auto compile, auto write,
               auto read, auto prepare) {
    f::run_domain(resource, queues, compile, write, read, prepare);
  });
}
TEST(AppleCtPackedGrouped, PublishedFlatDecodeKeepsOriginalDecomposition) {
  run_apple([](auto resource, auto queues, auto compile, auto write,
               auto read, auto prepare) {
    f::run_escape(resource, queues, compile, write, read, prepare);
  });
}

TEST(AppleCtPackedGrouped, OverlappingAndGappedReadOnlyBorrowKeepsRealCopyOrder) {
  run_apple([](auto resource, auto queues, auto compile, auto write,
               auto read, auto prepare) {
    const auto description = f::take(f::c::TensorDesc::strided_bytes(f::Type::kFloat32,
        std::array<std::int64_t, 3>{2, 2, 2},
        std::array<std::int64_t, 3>{4, 4, 20}), "overlap/hole description");
    auto source = f::take(f::m::Buffer::create(resource, 32, 256), "physical eight floats");
    const std::array<float, 8> payload{1, 2, 3, 4, 5, 6, 7, 8};
    f::check(write(source.view(), std::as_bytes(std::span(payload))), "write physical span");
    const auto tensor = f::take(f::c::ConstTensorView::create(description, source.view()),
        "read-only overlapping source");
    const auto borrowed = f::take(a::detail::MemoryAccess::borrow_tensor(tensor),
        "borrow original strided Metal bytes");
    EXPECT_FALSE(borrowed.flags().contiguous);
    EXPECT_FALSE(borrowed.flags().row_contiguous);
    EXPECT_FALSE(borrowed.flags().col_contiguous);
    EXPECT_EQ(borrowed.strides(), (mlx::core::Strides{1, 1, 5}));
    EXPECT_EQ(borrowed.data_size(), 8u);
    auto builder = f::take(f::g::GraphBuilder::create(), "metadata Copy builder");
    const auto input = f::take(builder.input(description), "strided read-only input");
    const auto dense_description = f::dense(f::Type::kFloat32, {8});
    const auto dense_input = f::take(builder.input(dense_description), "dense read-only input");
    const auto dense_output = f::take(builder.add(f::c::Copy{}, dense_input), "dense snapshot Copy");
    f::check(builder.publish(dense_output), "publish dense snapshot");
    const auto output = f::take(builder.add(f::c::Copy{}, input), "ordinary native Copy");
    f::check(builder.publish(output), "publish dense Copy");
    const auto graph = f::take(std::move(builder).finish(), "finish metadata Copy graph");
    auto compiled = f::take(compile(graph, f::g::CompileOptions{}, false), "compile metadata Copy");
    auto invocation = f::take(compiled.executable.create_invocation(
        {.resources = f::g::ResourceOptions{}}), "metadata Copy invocation");
    f::check(invocation.bind(input, tensor), "bind original overlapping source");
    f::check(invocation.bind(dense_input, f::take(f::c::ConstTensorView::create(
        dense_description, source.view()), "dense physical source")), "bind dense source");
    f::c::LeasePool pool;
    if (f::take(compiled.executable.output_ownership(), "Copy output ownership") == f::g::OutputOwnership::kCaller) {
      pool = f::take(f::c::LeasePool::create(resource, 512, {.max_blocks = 2}), "Copy output pool");
      for (const auto value : {output, dense_output}) {
        const auto desc = f::take(graph.value_description(value), "Copy dense output");
        f::check(invocation.bind(value, desc,
            f::take(pool.allocate(desc.required_bytes().value()), "Copy output writer")), "bind Copy output");
      }
    }
    f::check(prepare(invocation, queues, false), "prepare metadata Copy");
    auto work = f::take(invocation.submit(queues), "submit metadata Copy");
    f::check(work.wait(), "metadata Copy completion");
    auto result = f::take(invocation.result(output), "metadata Copy result");
    auto dense_result = f::take(invocation.result(dense_output), "dense Copy result");
    f::expect_output(result, std::vector<float>{1, 6, 2, 7, 2, 7, 3, 8}, read);
    f::expect_output(dense_result, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, read);
    const std::array<float, 8> next{11, 12, 13, 14, 15, 16, 17, 18};
    f::check(write(source.view(), std::as_bytes(std::span(next))), "replace original source bytes");
    invocation = {};
    compiled = f::g::CompiledGraph{};
    f::expect_output(result, std::vector<float>{1, 6, 2, 7, 2, 7, 3, 8}, read);
    f::expect_output(dense_result, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, read);
    ::testing::Test::RecordProperty("actual_submissions", 1);
  });
}
