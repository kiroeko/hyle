#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "../compute/native_context_fixture.h"
#include "hyle/mem/buffer.h"
#include "prepared_state.h"
#include "provider_internal.h"

namespace {

using namespace hyle::compute;
using namespace hyle::compute::detail;
using hyle::Result;
using hyle::Status;
using hyle::mem::Buffer;

struct RmsDomain final {
  std::array<ShapeDimension, 3> dimensions{};
  std::array<ShapeGuard, 1> guards{
      {ShapeGuard{ShapeGuardKind::kLessEqual, ShapeOperand::value(0),
                  ShapeOperand::value(1)}}};
  std::array<TensorDesc, 2> inputs{};
  std::array<TensorDesc, 1> outputs{};
  std::array<ProviderTensorAxes, 2> input_axes{};
  std::array<ProviderTensorAxes, 1> output_axes{};
  std::size_t input_count = 1;

  explicit RmsDomain(bool weighted = false, std::int64_t width = 5) {
    dimensions[0].bounds = {0, 4, 1};
    dimensions[1].bounds = {0, 3, 1};
    dimensions[2] = {infer_shape_range(ShapeDimensionKind::kMultiply,
                                       dimensions[0].bounds,
                                       dimensions[1].bounds)
                         .value(),
                     ShapeDimensionKind::kMultiply, ShapeOperand::value(0),
                     ShapeOperand::value(1)};
    inputs[0] = TensorDesc::dense(DataType::kFloat32,
                                  std::array{std::int64_t{12}, width})
                    .value();
    inputs[1] =
        TensorDesc::dense(DataType::kFloat32, std::array{width}).value();
    outputs[0] = inputs[0];
    input_axes[0].values[0] = output_axes[0].values[0] = 2;
    input_count = weighted ? 2 : 1;
  }

  std::span<const TensorDesc> input_descriptions() const {
    return std::span(inputs).first(input_count);
  }
  ProviderShapeDomain view() const {
    return {dimensions, std::span(input_axes).first(input_count), output_axes,
            guards};
  }
};

Result<PreparedOperation>
prepare_domain(const Context &context, const RmsDomain &domain,
               std::optional<hyle::mem::AllocatorRef> allocator = std::nullopt,
               bool exact = false) {
  std::array<ResourceBindingRequirement, 2> inputs{};
  std::array<ResourceBindingRequirement, 1> outputs{};
  const auto input_requirements = std::span(inputs).first(domain.input_count);
  const Operation operation(RmsNorm{});
  const auto shape = exact ? ProviderShapeDomain{} : domain.view();
  HYLE_ASSIGN_OR_RETURN(auto proposal,
                        OperationPreparationAccess::propose(
                            context, 0, operation, domain.input_descriptions(),
                            domain.outputs, input_requirements, outputs, shape,
                            allocator));
  if (!exact) {
    EXPECT_EQ(proposal.shape_coverage, ProviderShapeCoverage::kEntireDomain);
  }
  const std::array resources{hyle::mem::system_resource(),
                             hyle::mem::system_resource()};
  HYLE_ASSIGN_OR_RETURN(auto attempt,
                        OperationPreparationAccess::prepare_candidate(
                            context, proposal, operation,
                            domain.input_descriptions(), domain.outputs,
                            input_requirements, outputs,
                            {std::span(resources).first(domain.input_count),
                             std::span(resources).first(1)},
                            shape, allocator));
  if (!attempt.succeeded())
    return std::unexpected(std::move(attempt.reason));
  return std::move(attempt.prepared);
}

struct FaultAllocator final {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t attempts = 0, fail = SIZE_MAX, live = 0;
  std::size_t bytes = 0, last_size = 0;
  struct Allocation {
    void *pointer = nullptr;
    std::size_t size = 0, alignment = 0;
  };
  std::array<Allocation, 256> allocations{};

  void record(void *pointer, std::size_t size, std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return;
    for (auto &entry : allocations) {
      if (entry.pointer == nullptr) {
        entry = {pointer, size, alignment};
        ++live;
        bytes += size;
        last_size = size;
        return;
      }
    }
    ADD_FAILURE() << "fixture allocation table exhausted";
  }
  void forget(void *pointer, std::size_t size, std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return;
    for (auto &entry : allocations) {
      if (entry.pointer == pointer) {
        EXPECT_EQ(entry.size, size);
        EXPECT_EQ(entry.alignment, alignment);
        bytes -= entry.size;
        entry = {};
        --live;
        return;
      }
    }
    ADD_FAILURE() << "allocation was freed through a different owner";
  }
  hyle::mem::AllocatorRef ref() noexcept {
    return {[](void *self, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (s.attempts++ == s.fail)
                return nullptr;
              auto *p = s.backing.allocate(size, alignment);
              s.record(p, size, alignment);
              return p;
            },
            [](void *self, void *p, std::size_t size,
               std::size_t alignment) noexcept {
              auto &s = *static_cast<FaultAllocator *>(self);
              s.forget(p, size, alignment);
              s.backing.deallocate(p, size, alignment);
            },
            [](void *self, void *p, std::size_t old_size, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &s = *static_cast<FaultAllocator *>(self);
              if (size != 0 && s.attempts++ == s.fail)
                return nullptr;
              auto *next = s.backing.reallocate(p, old_size, size, alignment);
              if (size == 0 || next != nullptr) {
                s.forget(p, old_size, alignment);
                s.record(next, size, alignment);
              }
              return next;
            },
            this};
  }
};

// Same ordinary exact invocation journey as direct Compute. Only the private
// cold root creation/instantiation belongs to the Graph factor seam.
void check_rms(PreparedOperation prepared, Queue &queue, bool weighted,
               FaultAllocator *allocations = nullptr,
               std::optional<hyle::mem::AllocatorRef> invocation_allocator =
                   std::nullopt) {
  const auto description = prepared.input_description().value();
  const auto bytes = description.required_bytes().value();
  const auto count = bytes / sizeof(float);
  ASSERT_LE(count, 60u);
  std::array<float, 60> values{}, actual{};
  for (std::size_t i = 0; i < count; ++i)
    values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.25F;
  auto input =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  auto output =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  ASSERT_TRUE(
      input.view().write(std::as_bytes(std::span(values).first(count))).ok());
  const std::array<float, 5> scale{2, -1, 0.5F, 0, 3};
  auto weights = Buffer::create(hyle::mem::system_resource(), sizeof(scale),
                                alignof(float))
                     .value();
  ASSERT_TRUE(weights.view().write(std::as_bytes(std::span(scale))).ok());
  std::array<ConstTensorView, 2> inputs{};
  inputs[0] = ConstTensorView::create(description, input.view()).value();
  if (weighted)
    inputs[1] = ConstTensorView::create(prepared.input_description(1).value(),
                                        weights.view())
                    .value();
  const auto destination =
      TensorView::create(description, output.view()).value();
  auto invocation = OperationPreparationAccess::create_invocation(
                        prepared, {}, invocation_allocator)
                        .value();
  prepared = {};
  const auto before = allocations == nullptr ? 0 : allocations->attempts;
  // The invocation, not the temporary exact handle, retains the compiled root.
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto completion = invocation.submit(
        queue, std::span(inputs).first(weighted ? 2 : 1), destination);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
  }
  if (allocations != nullptr) {
    EXPECT_EQ(allocations->attempts, before);
  }
  ASSERT_TRUE(output.view()
                  .read(std::as_writable_bytes(std::span(actual).first(count)))
                  .ok());
  const auto width = static_cast<std::size_t>(description.dimensions().back());
  for (std::size_t row = 0; width != 0 && row < count / width; ++row) {
    double sum = 0;
    for (std::size_t column = 0; column < width; ++column) {
      const auto value = values[row * width + column];
      sum += static_cast<double>(value) * value;
    }
    const double inverse =
        1.0 / std::sqrt(sum / static_cast<double>(width) + RmsNorm{}.epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      const auto index = row * width + column;
      EXPECT_NEAR(actual[index],
                  values[index] * inverse * (weighted ? scale[column] : 1.0F),
                  2.0e-5);
    }
  }
}

TEST(NativeProviderShape, DoesNotMistakeUpperPointSupportForABoundedArtifact) {
  using namespace hyle::compute;
  using namespace hyle::compute::detail;
  auto context = hyle::compute::test::native_context().value();
  const std::array<std::int64_t, 1> shape{8};
  const std::array descriptions{
      TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array dimensions{ShapeDimension{.bounds = {1, 8, 1}}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[0] = 0;
  const ProviderShapeDomain domain{dimensions, axes, axes};
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  const auto propose = [&]() {
    return OperationPreparationAccess::propose(
        context, 0, Operation(Unary{UnaryKind::kNeg}), descriptions,
        descriptions, inputs, outputs, domain);
  };
  auto bounded = propose();
  ASSERT_FALSE(bounded.has_value());
  EXPECT_TRUE(bounded.error().IsUnimplemented());
  dimensions[0].bounds.minimum = 8;
  auto exact = propose();
  ASSERT_TRUE(exact.has_value()) << exact.error().message();
  EXPECT_EQ(exact->shape_coverage, ProviderShapeCoverage::kExactPoint);
  EXPECT_EQ(
      exact->profile_key,
      ProviderRouteKey{static_cast<std::uint64_t>(OperationKind::kUnary) + 1});
}

TEST(NativeProviderShape,
     ExecutesEveryAdmittedDerivedPointWithAndWithoutScale) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  for (bool weighted : {false, true}) {
    RmsDomain domain(weighted);
    auto root_result = prepare_domain(context, domain);
    ASSERT_TRUE(root_result) << root_result.error().message();
    auto root = std::move(*root_result);
    auto unbound = root.create_invocation();
    ASSERT_FALSE(unbound);
    EXPECT_TRUE(unbound.error().IsFailedPrecondition());
    for (std::int64_t batch = 0; batch <= 4; ++batch) {
      for (std::int64_t tokens = 0; tokens <= 3; ++tokens) {
        SCOPED_TRACE(::testing::Message()
                     << weighted << ':' << batch << ':' << tokens);
        std::array point{batch, tokens, std::int64_t{-999}};
        auto instance = OperationPreparationAccess::instantiate(root, point);
        if (batch > tokens) {
          ASSERT_FALSE(instance);
          EXPECT_TRUE(instance.error().IsInvalidArgument());
          continue;
        }
        ASSERT_TRUE(instance) << instance.error().message();
        EXPECT_EQ(point[2], batch * tokens);
        const auto expected =
            TensorDesc::dense(DataType::kFloat32,
                              std::array{batch * tokens, std::int64_t{5}})
                .value();
        EXPECT_EQ(instance->input_description().value(), expected);
        EXPECT_EQ(instance->output_description().value(), expected);
        EXPECT_TRUE(instance->alias_support(0, 0).value().supported);
        check_rms(std::move(*instance), queue, weighted);
      }
    }
  }
}

TEST(NativeProviderShape, OwnsProgramAndAxesAfterSourceMutationAndDestruction) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  PreparedOperation root;
  {
    RmsDomain domain(true);
    root = prepare_domain(context, domain).value();
    domain.dimensions = {};
    domain.guards = {};
    domain.input_axes = {};
    domain.output_axes = {};
    std::array<std::int64_t, 3> point{2, 3, -1};
    auto instance = OperationPreparationAccess::instantiate(root, point);
    ASSERT_TRUE(instance) << instance.error().message();
    check_rms(std::move(*instance), queue, true);
  }
  std::array<std::int64_t, 3> point{3, 3, -1};
  auto instance = OperationPreparationAccess::instantiate(root, point);
  ASSERT_TRUE(instance) << instance.error().message();
  root = {};
  // Context remains the admission owner; dropping its last public handle
  // closes it. Releasing the domain handle alone must not invalidate children.
  check_rms(std::move(*instance), queue, true);
}

TEST(NativeProviderShape, RejectsInvalidPointsWithoutMetadataAllocation) {
  FaultAllocator allocator;
  {
    auto context = hyle::compute::test::native_context(
                       {.metadata_allocator = allocator.ref()})
                       .value();
    auto root = prepare_domain(context, RmsDomain{}).value();
    const auto before = allocator.attempts;
    for (auto point : {std::array<std::int64_t, 3>{-1, 2, 0},
                       std::array<std::int64_t, 3>{0, 4, 0}}) {
      auto invalid = OperationPreparationAccess::instantiate(root, point);
      ASSERT_FALSE(invalid);
      EXPECT_TRUE(invalid.error().IsOutOfRange());
    }
    std::array<std::int64_t, 2> short_point{1, 2};
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, short_point)
                    .error()
                    .IsInvalidArgument());
    std::array<std::int64_t, 3> guarded{4, 3, 0};
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, guarded)
                    .error()
                    .IsInvalidArgument());
    EXPECT_EQ(allocator.attempts, before);
    ASSERT_TRUE(context.close().ok());
    guarded = {1, 2, 0};
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, guarded)
                    .error()
                    .IsFailedPrecondition());
    EXPECT_EQ(allocator.attempts, before);
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(NativeProviderShape, EqualBoundsAreNotEqualSymbolsAndHiddenWidthIsFixed) {
  auto context = hyle::compute::test::native_context().value();
  RmsDomain domain(true);
  const auto rejected = [&]() {
    auto result = prepare_domain(context, domain);
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented()) << result.error().message();
  };
  // Two independent dimensions have identical ranges but may bind differently.
  domain.dimensions[0].bounds = domain.dimensions[1].bounds = {0, 12, 1};
  domain.dimensions[2] = ShapeDimension{.bounds = {0, 12, 1}};
  domain.input_axes[0].values[0] = 0;
  domain.output_axes[0].values[0] = 1;
  rejected();
  domain = RmsDomain(true);
  domain.dimensions[0].bounds = {1, 5, 1};
  domain.dimensions[1].bounds = {0, 12, 1};
  domain.dimensions[2] = ShapeDimension{.bounds = {0, 12, 1}};
  domain.input_axes[0].values[1] = domain.output_axes[0].values[1] = 0;
  rejected();
  domain = RmsDomain(true);
  domain.dimensions[0].bounds = {1, 5, 1};
  domain.dimensions[2] = ShapeDimension{.bounds = {0, 12, 1}};
  domain.input_axes[1].values[0] = 0;
  rejected();
}

TEST(NativeProviderShape, ZeroHiddenWidthRemainsAValidEmptyExecution) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto root = prepare_domain(context, RmsDomain(false, 0)).value();
  std::array<std::int64_t, 3> point{2, 3, 0};
  auto instance = OperationPreparationAccess::instantiate(root, point);
  ASSERT_TRUE(instance) << instance.error().message();
  check_rms(std::move(*instance), queue, false);
}

TEST(NativeProviderShape, BindsMultiplePrefixAxesWithoutChangingTheRowKernel) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  RmsDomain domain;
  domain.inputs[0] = TensorDesc::dense(DataType::kFloat32,
                                       std::array<std::int64_t, 3>{4, 3, 5})
                         .value();
  domain.outputs[0] = domain.inputs[0];
  domain.input_axes[0].values[0] = domain.output_axes[0].values[0] = 0;
  domain.input_axes[0].values[1] = domain.output_axes[0].values[1] = 1;
  auto root = prepare_domain(context, domain).value();
  for (std::int64_t batch = 0; batch <= 3; ++batch) {
    std::array<std::int64_t, 3> point{batch, 3, -1};
    auto instance = OperationPreparationAccess::instantiate(root, point);
    ASSERT_TRUE(instance) << instance.error().message();
    EXPECT_EQ(instance->input_description()->rank(), 3u);
    check_rms(std::move(*instance), queue, false);
  }
}

TEST(NativeProviderShape, ConcreteRowsReuseTheExistingExecutorPartition) {
  struct ParallelProbe {
    std::size_t episodes = 0, tasks = 0;
  } probe;
  const ParallelExecutorRef executor(
      &probe, 4,
      [](void *self, std::size_t count, ParallelTaskFn task,
         void *state) noexcept {
        auto &p = *static_cast<ParallelProbe *>(self);
        ++p.episodes;
        p.tasks += count;
        for (std::size_t index = 0; index < count; ++index)
          task(state, index, count);
      });
  auto context =
      hyle::compute::test::native_context({.executor = executor}).value();
  auto queue = context.create_queue().value();
  auto root = prepare_domain(context, RmsDomain(false, 8192)).value();
  for (auto point : {std::array<std::int64_t, 3>{1, 1, -1},
                     std::array<std::int64_t, 3>{1, 3, -1},
                     std::array<std::int64_t, 3>{3, 3, -1}}) {
    auto bounded = OperationPreparationAccess::instantiate(root, point).value();
    const auto description = bounded.input_description().value();
    const auto bytes = description.required_bytes().value();
    auto input =
        Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
            .value();
    auto output =
        Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
            .value();
    ASSERT_TRUE(input.view().fill(std::byte{0}).ok());
    auto exact = prepare(context, RmsNorm{}, description).value();
    ParallelProbe exact_cost;
    for (auto *prepared : {&exact, &bounded}) {
      probe = {};
      auto invocation = prepared->create_invocation().value();
      ASSERT_TRUE(output.view().fill(std::byte{0xff}).ok());
      ASSERT_TRUE(
          invocation
              .submit(
                  queue,
                  ConstTensorView::create(description, input.view()).value(),
                  TensorView::create(description, output.view()).value())
              ->wait()
              .ok());
      std::vector<float> values(bytes / sizeof(float));
      ASSERT_TRUE(
          output.view().read(std::as_writable_bytes(std::span(values))).ok());
      for (const auto value : values)
        EXPECT_EQ(value, 0.0F);
      if (prepared == &exact)
        exact_cost = probe;
      else {
        EXPECT_EQ(probe.episodes, exact_cost.episodes);
        EXPECT_EQ(probe.tasks, exact_cost.tasks);
      }
    }
  }
}

TEST(NativeProviderShape, OrdinaryExactArtifactsDoNotBecomeDomains) {
  auto context = hyle::compute::test::native_context().value();
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{2, 5})
          .value();
  auto exact = prepare(context, RmsNorm{}, description).value();
  std::array<std::int64_t, 1> point{2};
  EXPECT_TRUE(OperationPreparationAccess::instantiate(exact, point)
                  .error()
                  .IsFailedPrecondition());
  EXPECT_TRUE(OperationPreparationAccess::instantiate({}, point)
                  .error()
                  .IsFailedPrecondition());
  EXPECT_TRUE(exact.create_invocation().has_value());
}

TEST(NativeProviderShape, EveryColdAllocationFailureCleansUpAndCanRetry) {
  FaultAllocator allocator;
  {
    auto context = hyle::compute::test::native_context(
                       {.metadata_allocator = allocator.ref()})
                       .value();
    auto queue = context.create_queue().value();
    const RmsDomain domain(true);
    const auto live_context = allocator.live;
    const auto start = allocator.attempts;
    auto root = prepare_domain(context, domain).value();
    const auto root_allocations = allocator.attempts - start;
    root = {};
    ASSERT_EQ(allocator.live, live_context);
    for (std::size_t fail = 0; fail < root_allocations; ++fail) {
      SCOPED_TRACE(fail);
      allocator.fail = allocator.attempts + fail;
      auto failed = prepare_domain(context, domain);
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, live_context);
      allocator.fail = SIZE_MAX;
      EXPECT_TRUE(prepare_domain(context, domain).has_value());
      EXPECT_EQ(allocator.live, live_context);
    }
    root = prepare_domain(context, domain).value();
    const auto live_root = allocator.live;
    const auto instance_start = allocator.attempts;
    std::array<std::int64_t, 3> point{2, 3, -1};
    auto instance =
        OperationPreparationAccess::instantiate(root, point).value();
    const auto instance_allocations = allocator.attempts - instance_start;
    instance = {};
    EXPECT_EQ(allocator.live, live_root);
    for (std::size_t fail = 0; fail < instance_allocations; ++fail) {
      SCOPED_TRACE(fail);
      allocator.fail = allocator.attempts + fail;
      auto failed = OperationPreparationAccess::instantiate(root, point);
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, live_root);
      allocator.fail = SIZE_MAX;
      instance = OperationPreparationAccess::instantiate(root, point).value();
      check_rms(std::move(instance), queue, true, &allocator);
      EXPECT_EQ(allocator.live, live_root);
    }
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(NativeMetadataOwner, ExactAndDomainArtifactsUseTheirOwnAllocator) {
  for (const bool exact : {false, true}) {
    SCOPED_TRACE(exact);
    FaultAllocator context_owner, compile_owner, instance_owner, run_owner;
    {
      auto context = hyle::compute::test::native_context(
                         {.metadata_allocator = context_owner.ref()})
                         .value();
      auto queue = context.create_queue().value();
      const auto context_attempts = context_owner.attempts;
      const RmsDomain domain(true);
      auto root =
          prepare_domain(context, domain, compile_owner.ref(), exact).value();
      EXPECT_GT(compile_owner.live, 0u);
      auto instance = root;
      if (!exact) {
        std::array<std::int64_t, 3> point{2, 3, -1};
        instance = OperationPreparationAccess::instantiate(root, point,
                                                           instance_owner.ref())
                       .value();
        EXPECT_GT(instance_owner.live, 0u);
      }
      root = {};
      EXPECT_GT(compile_owner.live, 0u);
      check_rms(std::move(instance), queue, true, &run_owner, run_owner.ref());
      EXPECT_EQ(context_owner.attempts, context_attempts);
      EXPECT_EQ(compile_owner.live, 0u);
      EXPECT_EQ(instance_owner.live, 0u);
      EXPECT_EQ(run_owner.live, 0u);
      EXPECT_EQ(compile_owner.bytes + instance_owner.bytes + run_owner.bytes,
                0u);
    }
    EXPECT_EQ(context_owner.live, 0u);
  }
}

TEST(NativeMetadataOwner, EveryOverrideAllocationCanFailAndRetry) {
  FaultAllocator context_owner, compile_owner, instance_owner, run_owner;
  auto context = hyle::compute::test::native_context(
                     {.metadata_allocator = context_owner.ref()})
                     .value();
  const auto context_attempts = context_owner.attempts;
  const RmsDomain domain;
  auto root = prepare_domain(context, domain, compile_owner.ref()).value();
  const auto compile_count = compile_owner.attempts;
  root = {};
  for (std::size_t fail = 0; fail < compile_count; ++fail) {
    SCOPED_TRACE(fail);
    compile_owner.fail = compile_owner.attempts + fail;
    auto rejected = prepare_domain(context, domain, compile_owner.ref());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsResourceExhausted());
    EXPECT_EQ(compile_owner.live, 0u);
    compile_owner.fail = SIZE_MAX;
    EXPECT_TRUE(prepare_domain(context, domain, compile_owner.ref()));
    EXPECT_EQ(compile_owner.live, 0u);
  }
  root = prepare_domain(context, domain, compile_owner.ref()).value();
  const auto compile_bytes = compile_owner.bytes;
  std::array<std::int64_t, 3> point{2, 3, -1};
  auto instance =
      OperationPreparationAccess::instantiate(root, point, instance_owner.ref())
          .value();
  const auto instance_count = instance_owner.attempts;
  instance = {};
  for (std::size_t fail = 0; fail < instance_count; ++fail) {
    SCOPED_TRACE(fail);
    instance_owner.fail = instance_owner.attempts + fail;
    auto rejected = OperationPreparationAccess::instantiate(
        root, point, instance_owner.ref());
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsResourceExhausted());
    EXPECT_EQ(instance_owner.live, 0u);
    EXPECT_EQ(compile_owner.bytes, compile_bytes);
    instance_owner.fail = SIZE_MAX;
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point,
                                                        instance_owner.ref()));
    EXPECT_EQ(instance_owner.live, 0u);
  }
  instance =
      OperationPreparationAccess::instantiate(root, point, instance_owner.ref())
          .value();
  run_owner.fail = 0;
  auto rejected = OperationPreparationAccess::create_invocation(
      instance, {}, run_owner.ref());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(run_owner.live, 0u);
  run_owner.fail = SIZE_MAX;
  EXPECT_TRUE(OperationPreparationAccess::create_invocation(instance, {},
                                                            run_owner.ref()));
  EXPECT_EQ(run_owner.live, 0u);
  EXPECT_EQ(context_owner.attempts, context_attempts);
}

TEST(NativeMetadataOwner, InvalidOverrideIsNotSilentlyInherited) {
  auto context = hyle::compute::test::native_context().value();
  const RmsDomain domain;
  const hyle::mem::AllocatorRef invalid;
  EXPECT_TRUE(
      prepare_domain(context, domain, invalid).error().IsInvalidArgument());
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  const Operation operation(RmsNorm{});
  auto proposal = OperationPreparationAccess::propose(
                      context, 0, operation, domain.input_descriptions(),
                      domain.outputs, inputs, outputs, domain.view())
                      .value();
  const std::array resources{hyle::mem::system_resource()};
  auto rejected = OperationPreparationAccess::prepare_candidate(
      context, proposal, operation, domain.input_descriptions(), domain.outputs,
      inputs, outputs, {resources, resources}, domain.view(), invalid);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto root = prepare_domain(context, domain).value();
  std::array<std::int64_t, 3> point{2, 3, -1};
  EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point, invalid)
                  .error()
                  .IsInvalidArgument());
  auto instance = OperationPreparationAccess::instantiate(root, point).value();
  EXPECT_TRUE(
      OperationPreparationAccess::create_invocation(instance, {}, invalid)
          .error()
          .IsInvalidArgument());
}

TEST(NativeMetadataOwner, OrdinaryInvocationKeepsItsSizeAndAllocationCount) {
  // Structural baseline before allocator overrides; test-only, no product
  // compatibility type. The new cold flag fits existing trailing padding.
  struct PreviousLayout : hyle::util::RefCounted<PreviousLayout> {
    PreparedState *prepared = nullptr;
    void *provider_state = nullptr;
    CompletionLifetime lifetime{};
    std::atomic<bool> in_flight{false};
  };
  static_assert(sizeof(InvocationState) == sizeof(PreviousLayout));
  FaultAllocator owner, separate;
  auto context = hyle::compute::test::native_context().value();
  auto root = prepare_domain(context, RmsDomain{}, owner.ref(), true).value();
  const auto root_bytes = owner.bytes;
  const auto before = owner.attempts;
  auto invocation = root.create_invocation().value();
  EXPECT_EQ(owner.attempts, before + 1);
  EXPECT_EQ(owner.bytes - root_bytes, sizeof(PreviousLayout));
  invocation = {};
  EXPECT_EQ(owner.bytes, root_bytes);
  invocation =
      OperationPreparationAccess::create_invocation(root, {}, owner.ref())
          .value();
  EXPECT_EQ(owner.bytes - root_bytes, sizeof(PreviousLayout));
  invocation = {};
  invocation =
      OperationPreparationAccess::create_invocation(root, {}, separate.ref())
          .value();
  EXPECT_EQ(owner.bytes, root_bytes);
  EXPECT_EQ(separate.live, 1u);
  EXPECT_EQ(separate.bytes,
            sizeof(PreviousLayout) + sizeof(hyle::mem::AllocatorRef));
  root = {};
  EXPECT_GT(owner.bytes, 0u);
  invocation = {};
  EXPECT_EQ(owner.bytes, 0u);
  EXPECT_EQ(separate.bytes, 0u);
}

TEST(NativeMetadataOwner, ConcurrentInstancesHaveIndependentOwners) {
  auto context = hyle::compute::test::native_context().value();
  FaultAllocator compiled;
  auto root = prepare_domain(context, RmsDomain{}, compiled.ref()).value();
  const auto compile_attempts = compiled.attempts;
  const auto run = [&](std::int64_t batch) {
    FaultAllocator owner;
    auto queue = context.create_queue().value();
    for (int repeat = 0; repeat < 8; ++repeat) {
      std::array<std::int64_t, 3> point{batch, 3, -1};
      auto instance =
          OperationPreparationAccess::instantiate(root, point, owner.ref());
      ASSERT_TRUE(instance) << instance.error().message();
      check_rms(std::move(*instance), queue, false, &owner);
      EXPECT_EQ(owner.live, 0u);
    }
  };
  {
    std::jthread first(run, 1), second(run, 3);
  }
  EXPECT_EQ(compiled.attempts, compile_attempts);
  root = {};
  EXPECT_EQ(compiled.live, 0u);
}

TEST(NativeProviderShape, IndependentInstancesShareOneImmutableDomain) {
  auto context = hyle::compute::test::native_context().value();
  auto root = prepare_domain(context, RmsDomain{}).value();
  const auto run = [&](std::int64_t batch) {
    auto queue = context.create_queue().value();
    for (int repeat = 0; repeat < 8; ++repeat) {
      std::array<std::int64_t, 3> point{batch, 3, -1};
      auto instance = OperationPreparationAccess::instantiate(root, point);
      ASSERT_TRUE(instance) << instance.error().message();
      check_rms(std::move(*instance), queue, false);
    }
  };
  std::jthread first(run, 1), second(run, 3);
}

enum class BrokenInstance {
  kNone,
  kMissingHook,
  kRuntime,
  kWorkspaceSize,
  kWorkspaceAlignment,
  kAlias,
  kCallback,
  kRecursiveDomain,
  kPartialError,
  kUnownedPartial
};

struct DomainProbe final {
  BrokenInstance broken = BrokenInstance::kNone;
  std::size_t proposals = 0, preparations = 0, instances = 0;
  std::size_t roots_destroyed = 0, children_destroyed = 0;
  bool root_live = false;
  std::optional<std::size_t> root_private_bytes{}, instance_private_bytes{};
};

struct DomainArtifact final {
  DomainProbe *probe = nullptr;
  bool root = false;
};

void destroy_artifact(void *state, hyle::mem::AllocatorRef allocator) noexcept {
  auto *artifact = static_cast<DomainArtifact *>(state);
  auto &probe = *artifact->probe;
  EXPECT_TRUE(probe.root_live);
  if (artifact->root) {
    probe.root_live = false;
    ++probe.roots_destroyed;
  } else {
    ++probe.children_destroyed;
  }
  std::destroy_at(artifact);
  allocator.deallocate(artifact, sizeof(DomainArtifact),
                       alignof(DomainArtifact));
}

const ProviderPreparedVTable kDomainCallbacks{
    .destroy_prepared = &destroy_artifact,
    .alias_support =
        [](void *state, std::size_t output, std::size_t input) noexcept {
          const auto &artifact = *static_cast<DomainArtifact *>(state);
          return AliasSupport{output == 0 && input == 0,
                              !artifact.root && artifact.probe->broken ==
                                                    BrokenInstance::kAlias
                                  ? 4u
                                  : 0u};
        },
    .create_invocation =
        [](void *, hyle::mem::AllocatorRef, hyle::mem::BufferView,
           void **invocation) noexcept {
          *invocation = nullptr;
          return Status::Ok();
        },
    .destroy_invocation = [](void *, hyle::mem::AllocatorRef) noexcept {},
    .shared_submit =
        [](void *, void *, Queue &, std::span<const ConstTensorView>,
           std::span<const TensorView>) noexcept {
          return EnqueueResult::rejected(
              Status::Unimplemented("contract-only fixture"));
        }};

const ProviderPreparedVTable kMissingSubmit = [] {
  auto table = kDomainCallbacks;
  table.shared_submit = nullptr;
  return table;
}();

ProviderPrepared probe_artifact(DomainProbe &probe, bool root,
                                hyle::mem::AllocatorRef allocator) {
  void *storage =
      allocator.allocate(sizeof(DomainArtifact), alignof(DomainArtifact));
  if (storage == nullptr)
    return {};
  return {.vtable = &kDomainCallbacks,
          .state = ::new (storage) DomainArtifact{&probe, root},
          .workspace = {32, 16},
          .runtime = {.replay_class = ProviderReplayClass::kPrepared},
          .persistent_bytes = sizeof(DomainArtifact),
          .invocation_private_bytes =
              root ? probe.root_private_bytes : probe.instance_private_bytes};
}

Status probe_instantiate(void *root, std::span<const TensorDesc>,
                         std::span<const TensorDesc>,
                         hyle::mem::AllocatorRef allocator,
                         ProviderPrepared *instance) noexcept {
  auto &probe = *static_cast<DomainArtifact *>(root)->probe;
  ++probe.instances;
  EXPECT_TRUE(probe.root_live);
  if (probe.broken == BrokenInstance::kUnownedPartial) {
    // Deliberately malformed borrowed pointer: no unowned heap allocation is
    // created even though the tested provider contract is invalid.
    instance->state = &probe;
    return Status::DataLoss("malformed partial fixture");
  }
  *instance = probe_artifact(probe, false, allocator);
  if (instance->state == nullptr)
    return Status::ResourceExhausted("fixture allocation");
  switch (probe.broken) {
  case BrokenInstance::kRuntime:
    instance->runtime.replay_class = ProviderReplayClass::kManaged;
    break;
  case BrokenInstance::kWorkspaceSize:
    instance->workspace.size = 33;
    break;
  case BrokenInstance::kWorkspaceAlignment:
    instance->workspace.alignment = 32;
    break;
  case BrokenInstance::kCallback:
    instance->vtable = &kMissingSubmit;
    break;
  case BrokenInstance::kRecursiveDomain:
    instance->instantiate = &probe_instantiate;
    break;
  case BrokenInstance::kPartialError:
    return Status::DataLoss("owned partial fixture");
  default:
    break;
  }
  return Status::Ok();
}

const ProviderVTable kDomainProvider{
    .identity = {.id = ProviderId{0x48594c452d544553ULL, 0x542d444f4d41494eULL},
                 .name = "test-domain",
                 .version = "1"},
    .create_context =
        [](const Provider &provider, Device, void *, hyle::mem::AllocatorRef,
           void **context) noexcept {
          *context = ProviderAccess::payload<DomainProbe *>(provider);
          return Status::Ok();
        },
    .destroy_context = [](void *, hyle::mem::AllocatorRef) noexcept {},
    .propose =
        [](void *context, const ProviderOperationRequest &,
           std::span<ResourceBindingRequirement>,
           std::span<ResourceBindingRequirement>,
           ProviderOperationProposal *proposal) noexcept {
          ++static_cast<DomainProbe *>(context)->proposals;
          *proposal = {
              .runtime = {.replay_class = ProviderReplayClass::kPrepared},
              .cost = {.workspace = WorkspaceRequirement{32, 16},
                       .persistent_bytes = sizeof(DomainArtifact)},
              .shape_coverage = ProviderShapeCoverage::kEntireDomain};
          return Status::Ok();
        },
    .prepare =
        [](void *context, const ProviderPrepareRequest &,
           hyle::mem::AllocatorRef allocator,
           ProviderPrepared *prepared) noexcept {
          auto &probe = *static_cast<DomainProbe *>(context);
          ++probe.preparations;
          *prepared = probe_artifact(probe, true, allocator);
          if (prepared->state == nullptr)
            return ProviderPrepareOutcome::error(
                Status::ResourceExhausted("fixture allocation"));
          probe.root_live = true;
          if (probe.broken != BrokenInstance::kMissingHook)
            prepared->instantiate = &probe_instantiate;
          return ProviderPrepareOutcome::prepared();
        },
    .drain_context = [](void *) noexcept { return Status::Ok(); }};

Context probe_context(DomainProbe &probe) {
  const std::array providers{ProviderAccess::make(&kDomainProvider, &probe)};
  return Context::create(cpu::device(), {.providers = providers}).value();
}

TEST(PreparedDomainContract,
     RequiresInstantiationBeforeAcceptingADomainArtifact) {
  DomainProbe probe{.broken = BrokenInstance::kMissingHook};
  auto context = probe_context(probe);
  auto root = prepare_domain(context, RmsDomain{});
  ASSERT_FALSE(root);
  EXPECT_TRUE(root.error().IsInternal());
  EXPECT_EQ(probe.preparations, 1u);
  EXPECT_EQ(probe.roots_destroyed, 1u);
  EXPECT_FALSE(probe.root_live);
}

TEST(PreparedDomainContract, RejectsBrokenInstancesAndDestroysOwnedPartials) {
  for (auto mode :
       {BrokenInstance::kRuntime, BrokenInstance::kWorkspaceSize,
        BrokenInstance::kWorkspaceAlignment, BrokenInstance::kAlias,
        BrokenInstance::kCallback, BrokenInstance::kRecursiveDomain,
        BrokenInstance::kPartialError, BrokenInstance::kUnownedPartial}) {
    SCOPED_TRACE(static_cast<int>(mode));
    DomainProbe probe;
    auto context = probe_context(probe);
    auto root = prepare_domain(context, RmsDomain{}).value();
    probe.broken = mode;
    std::array<std::int64_t, 3> point{2, 3, -1};
    auto instance = OperationPreparationAccess::instantiate(root, point);
    ASSERT_FALSE(instance);
    EXPECT_TRUE(mode == BrokenInstance::kPartialError
                    ? instance.error().IsDataLoss()
                    : instance.error().IsInternal());
    EXPECT_EQ(probe.instances, 1u);
    EXPECT_EQ(probe.children_destroyed,
              mode == BrokenInstance::kUnownedPartial ? 0u : 1u);
    EXPECT_TRUE(probe.root_live);
    probe.broken = BrokenInstance::kNone;
    instance = OperationPreparationAccess::instantiate(root, point);
    ASSERT_TRUE(instance) << instance.error().message();
    EXPECT_EQ(probe.proposals, 1u);
    EXPECT_EQ(probe.preparations, 1u);
    root = {};
    EXPECT_TRUE(probe.root_live);
    instance = PreparedOperation{};
    EXPECT_FALSE(probe.root_live);
    EXPECT_EQ(probe.roots_destroyed, 1u);
  }
}

TEST(PreparedDomainContract,
     ChecksShapeBeforeCallbackAndInvocationRetainsRoot) {
  DomainProbe probe;
  auto context = probe_context(probe);
  auto root = prepare_domain(context, RmsDomain{}).value();
  std::array<std::int64_t, 3> point{4, 3, -1};
  EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point)
                  .error()
                  .IsInvalidArgument());
  point = {1, 4, -1};
  EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point)
                  .error()
                  .IsOutOfRange());
  EXPECT_EQ(probe.instances, 0u);
  point = {2, 3, -1};
  auto instance = OperationPreparationAccess::instantiate(root, point).value();
  auto workspace = Buffer::create(hyle::mem::system_resource(), 32, 16).value();
  auto invocation = instance.create_invocation(workspace.view()).value();
  root = {};
  instance = {};
  EXPECT_EQ(probe.roots_destroyed, 0u);
  EXPECT_EQ(probe.children_destroyed, 0u);
  invocation = {};
  EXPECT_EQ(probe.children_destroyed, 1u);
  EXPECT_EQ(probe.roots_destroyed, 1u);
}

TEST(PreparedDomainContract, ConcretePrivateStorageCannotExceedDomainBound) {
  DomainProbe probe;
  probe.root_private_bytes = 64;
  auto context = probe_context(probe);
  auto root = prepare_domain(context, RmsDomain{}).value();
  EXPECT_EQ(root.invocation_private_bytes(), 64U);
  for (auto bytes :
       {std::optional<std::size_t>{32}, std::optional<std::size_t>{65},
        std::optional<std::size_t>{}}) {
    std::array<std::int64_t, 3> point{1, 2, -1};
    probe.instance_private_bytes = bytes;
    auto instance = OperationPreparationAccess::instantiate(root, point);
    if (bytes && *bytes <= 64) {
      ASSERT_TRUE(instance);
      EXPECT_EQ(instance->invocation_private_bytes(), bytes);
    } else {
      ASSERT_FALSE(instance);
      EXPECT_TRUE(instance.error().IsInternal());
    }
  }
  EXPECT_EQ(probe.instances, 3U);
  EXPECT_EQ(probe.children_destroyed, 3U);
  root = {};
  EXPECT_EQ(probe.roots_destroyed, 1U);
}

TEST(PreparedDomainContract, SchemaIsRecheckedBeforeAConcreteProviderCallback) {
  DomainProbe probe;
  auto context = probe_context(probe);
  RmsDomain domain;
  // The deliberately lying fixture promises shape preservation for every
  // point, but its output remains the fixed envelope. Core must not execute it.
  domain.output_axes = {};
  auto root = prepare_domain(context, domain).value();
  std::array<std::int64_t, 3> point{1, 2, -1};
  auto instance = OperationPreparationAccess::instantiate(root, point);
  ASSERT_FALSE(instance);
  EXPECT_TRUE(instance.error().IsInvalidArgument());
  EXPECT_EQ(probe.instances, 0u);
}

TEST(PreparedDomainContract,
     RejectsPointOnlyAndInvalidCoverageBeforePreparation) {
  DomainProbe probe;
  auto context = probe_context(probe);
  const RmsDomain domain;
  const Operation operation(RmsNorm{});
  std::array<ResourceBindingRequirement, 1> inputs{}, outputs{};
  auto proposal = OperationPreparationAccess::propose(
                      context, 0, operation, domain.input_descriptions(),
                      domain.outputs, inputs, outputs, domain.view())
                      .value();
  const std::array resources{hyle::mem::system_resource()};
  for (auto coverage : {ProviderShapeCoverage::kExactPoint,
                        static_cast<ProviderShapeCoverage>(255)}) {
    proposal.shape_coverage = coverage;
    auto attempt = OperationPreparationAccess::prepare_candidate(
        context, proposal, operation, domain.input_descriptions(),
        domain.outputs, inputs, outputs, {resources, resources}, domain.view());
    ASSERT_FALSE(attempt);
    EXPECT_TRUE(attempt.error().IsInvalidArgument());
  }
  EXPECT_EQ(probe.preparations, 0u);
}

TEST(NativeProviderShape, SuppliedMetadataExecutesAndRetainsItsOwnDescriptors) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  for (bool weighted : {false, true}) {
    auto root = prepare_domain(context, RmsDomain(weighted)).value();
    for (std::int64_t rows = 0; rows <= 3; ++rows) {
      std::array<std::int64_t, 3> point{rows, 3, -1};
      auto input = TensorDesc::dense(DataType::kFloat32,
                                     std::array{rows * 3, std::int64_t{5}})
                       .value();
      std::array inputs{input, TensorDesc::dense(DataType::kFloat32,
                                                 std::array<std::int64_t, 1>{5})
                                   .value()};
      std::array outputs{input};
      auto instance = OperationPreparationAccess::instantiate(
          root, point, std::span(inputs).first(weighted ? 2 : 1), outputs);
      ASSERT_TRUE(instance) << instance.error().message();
      EXPECT_EQ(point[2], rows * 3);
      inputs = {};
      outputs = {};
      point = {};
      EXPECT_EQ(instance->input_description().value(), input);
      EXPECT_EQ(instance->output_description().value(), input);
      check_rms(std::move(*instance), queue, weighted);
    }
  }
}

TEST(NativeProviderShape, SuppliedMetadataAllocationFailureAndOwnerLifetime) {
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto root = prepare_domain(context, RmsDomain{}).value();
  const std::array descriptions{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{6, 5})
          .value()};
  std::array<std::int64_t, 3> point{2, 3, -1};
  FaultAllocator counted;
  {
    auto instance = OperationPreparationAccess::instantiate(
        root, point, descriptions, descriptions, counted.ref());
    ASSERT_TRUE(instance) << instance.error().message();
    EXPECT_GT(counted.live, 0u);
  }
  EXPECT_EQ(counted.live, 0u);
  const auto count = counted.attempts;
  ASSERT_GT(count, 0u);
  for (std::size_t failure = 0; failure < count; ++failure) {
    FaultAllocator owner;
    owner.fail = failure;
    auto failed = OperationPreparationAccess::instantiate(
        root, point, descriptions, descriptions, owner.ref());
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsResourceExhausted());
    EXPECT_EQ(owner.live, 0u);
    owner.fail = SIZE_MAX;
    {
      auto retry = OperationPreparationAccess::instantiate(
          root, point, descriptions, descriptions, owner.ref());
      ASSERT_TRUE(retry) << retry.error().message();
      check_rms(std::move(*retry), queue, false, &owner);
    }
    EXPECT_EQ(owner.live, 0u);
  }
  {
    auto instance = OperationPreparationAccess::instantiate(
                        root, point, descriptions, descriptions, counted.ref())
                        .value();
    root = {};
    check_rms(std::move(instance), queue, false, &counted);
  }
  EXPECT_EQ(counted.live, 0u);
}

TEST(NativeProviderShape, DenseEnvelopeDoesNotProveAStridedFamily) {
  auto context = hyle::compute::test::native_context().value();
  RmsDomain domain;
  domain.input_axes[0].layout = ProviderTensorLayout::kBoundedStrided;
  domain.input_axes[0].maximum_bytes =
      domain.inputs[0].required_bytes().value();
  auto input_family = prepare_domain(context, domain);
  ASSERT_FALSE(input_family);
  EXPECT_TRUE(input_family.error().IsUnimplemented());
  domain.input_axes[0].layout = ProviderTensorLayout::kDerived;
  domain.input_axes[0].maximum_bytes = 0;
  domain.output_axes[0].layout = ProviderTensorLayout::kBoundedStrided;
  domain.output_axes[0].maximum_bytes =
      domain.outputs[0].required_bytes().value();
  auto output_family = prepare_domain(context, domain);
  ASSERT_FALSE(output_family);
  EXPECT_TRUE(output_family.error().IsUnimplemented());
}

TEST(PreparedDomainContract, SuppliedMetadataIsCheckedBeforeTheProvider) {
  DomainProbe probe;
  auto context = probe_context(probe);
  auto root = prepare_domain(context, RmsDomain{}).value();
  const std::array<std::int64_t, 2> shape{6, 5};
  const std::array expected{
      TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<std::int64_t, 3> point{2, 3, -1};
  for (const auto &wrong :
       {TensorDesc{}, TensorDesc::dense(DataType::kInt32, shape).value(),
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{5, 5})
            .value(),
        TensorDesc::strided_bytes(DataType::kFloat32, shape,
                                  std::array<std::int64_t, 2>{24, 4})
            .value()}) {
    const std::array descriptors{wrong};
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point,
                                                        descriptors, expected)
                    .error()
                    .IsInvalidArgument());
    EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point, expected,
                                                        descriptors)
                    .error()
                    .IsInvalidArgument());
  }
  EXPECT_TRUE(OperationPreparationAccess::instantiate(root, point, {}, expected)
                  .error()
                  .IsInvalidArgument());
  point = {4, 3, -1};
  EXPECT_TRUE(
      OperationPreparationAccess::instantiate(root, point, expected, expected)
          .error()
          .IsInvalidArgument());
  EXPECT_EQ(probe.instances, 0u);
  point = {2, 3, -1};
  auto invalid_allocator =
      hyle::mem::AllocatorRef(nullptr, nullptr, nullptr, nullptr);
  EXPECT_TRUE(OperationPreparationAccess::instantiate(
                  root, point, expected, expected, invalid_allocator)
                  .error()
                  .IsInvalidArgument());
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(
      OperationPreparationAccess::instantiate(root, point, expected, expected)
          .error()
          .IsFailedPrecondition());
  EXPECT_EQ(probe.instances, 0u);
  EXPECT_TRUE(
      OperationPreparationAccess::instantiate({}, point, expected, expected)
          .error()
          .IsFailedPrecondition());
}

TEST(PreparedDomainContract, StridedMembershipDoesNotOverrideOperationSchema) {
  DomainProbe probe;
  auto context = probe_context(probe);
  RmsDomain domain;
  domain.output_axes[0].layout = ProviderTensorLayout::kBoundedStrided;
  domain.output_axes[0].maximum_bytes = 512;
  auto root = prepare_domain(context, domain).value();
  const std::array<std::int64_t, 2> shape{6, 5};
  const std::array inputs{TensorDesc::dense(DataType::kFloat32, shape).value()};
  const std::array outputs{
      TensorDesc::strided_bytes(DataType::kFloat32, shape,
                                std::array<std::int64_t, 2>{24, 4})
          .value()};
  std::array<std::int64_t, 3> point{2, 3, -1};
  ASSERT_TRUE(validate_shape_binding(domain.view(), domain.input_descriptions(),
                                     domain.outputs, point, inputs, outputs)
                  .ok());
  auto invalid =
      OperationPreparationAccess::instantiate(root, point, inputs, outputs);
  ASSERT_FALSE(invalid);
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_EQ(probe.instances, 0u);
  auto valid =
      OperationPreparationAccess::instantiate(root, point, inputs, inputs);
  ASSERT_TRUE(valid) << valid.error().message();
  EXPECT_EQ(probe.instances, 1u);
  root = {};
  EXPECT_TRUE(probe.root_live);
  valid = PreparedOperation{};
  EXPECT_FALSE(probe.root_live);
}

TEST(NativeProviderShape,
     BoundedCopyRequiresDenseOutputAndIdenticalAxisExpressions) {
  auto context = hyle::compute::test::native_context().value();
  std::array<ShapeDimension, 2> dimensions{};
  dimensions[0].bounds = dimensions[1].bounds = {0, 4, 1};
  std::array<ProviderTensorAxes, 1> input_axes{}, output_axes{};
  input_axes[0].values[0] = output_axes[0].values[0] = 0;
  input_axes[0].layout = ProviderTensorLayout::kBoundedStrided;
  const std::array inputs{
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 2>{4, 3},
                                std::array<std::int64_t, 2>{0, 4})
          .value()};
  const std::array outputs{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{4, 3})
          .value()};
  input_axes[0].maximum_bytes = inputs[0].required_bytes().value();
  const auto propose = [&] {
    std::array<ResourceBindingRequirement, 1> in{}, out{};
    return OperationPreparationAccess::propose(
        context, 0, Operation(Copy{}), inputs, outputs, in, out,
        ProviderShapeDomain{dimensions, input_axes, output_axes});
  };
  auto accepted = propose();
  ASSERT_TRUE(accepted) << accepted.error().message();
  EXPECT_EQ(accepted->shape_coverage, ProviderShapeCoverage::kEntireDomain);
  output_axes[0].values[0] = 1;
  auto unrelated = propose();
  ASSERT_FALSE(unrelated);
  EXPECT_TRUE(unrelated.error().IsUnimplemented());
  output_axes[0].values[0] = 0;
  output_axes[0].layout = ProviderTensorLayout::kBoundedStrided;
  output_axes[0].maximum_bytes = outputs[0].required_bytes().value();
  auto arbitrary_output = propose();
  ASSERT_FALSE(arbitrary_output);
  EXPECT_TRUE(arbitrary_output.error().IsUnimplemented());
}

} // namespace
