#include "hyle/compute/providers/onednn.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "brgemm_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"
#include "provider_internal.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::ContextOptions;
using hyle::compute::DataType;
using hyle::compute::MatMul;
using hyle::compute::PreparedOperation;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> deallocations{0};
  std::atomic<bool> fail{false};
  std::atomic<std::size_t> fail_at_attempt{0};
};

struct ParallelProbe final {
  std::atomic<std::size_t> episodes{0};
  std::atomic<std::size_t> tasks{0};
  bool in_parallel = false;
};

void probe_parallel_for(void *state, std::size_t count,
                        hyle::compute::ParallelTaskFn task,
                        void *task_context) noexcept {
  auto &probe = *static_cast<ParallelProbe *>(state);
  ++probe.episodes;
  probe.tasks += count;
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

bool probe_in_parallel(void *state) noexcept {
  return static_cast<ParallelProbe *>(state)->in_parallel;
}

void *probe_allocate(void *state, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(state);
  const std::size_t attempt = probe.attempts.fetch_add(1) + 1;
  if (probe.fail.load() || attempt == probe.fail_at_attempt.load())
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_deallocate(void *state, void *pointer, std::size_t size,
                      std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(state);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *state, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(state);
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_deallocate, &probe_reallocate,
                      &probe);
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array dimensions{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, dimensions).value();
}

void write_floats(hyle::mem::BufferView destination,
                  std::span<const float> values) {
  ASSERT_TRUE(destination.write(std::as_bytes(values)).ok());
}

std::vector<float> read_floats(hyle::mem::ConstBufferView source,
                               std::size_t count) {
  std::vector<float> values(count);
  EXPECT_TRUE(source.read(std::as_writable_bytes(std::span(values))).ok());
  return values;
}

Context create_context() {
  return Context::create(hyle::compute::cpu::device(),
                         hyle::compute::onednn::provider())
      .value();
}

struct BrgemmPlanOwner final {
  hyle::compute::onednn::internal::BrgemmPlan *plan = nullptr;
  AllocatorRef allocator{};

  ~BrgemmPlanOwner() {
    hyle::compute::onednn::internal::destroy_brgemm_plan(plan, allocator);
  }
};

struct PairPostProbe final {
  std::size_t calls = 0;
  std::size_t rows = 0;
  std::size_t leading_columns = 0;
  hyle::compute::onednn::internal::BrgemmTaskRange range{};
};

void record_pair_post(
    void *context, std::size_t rows, std::size_t leading_columns,
    hyle::compute::onednn::internal::BrgemmTaskRange range) noexcept {
  auto &probe = *static_cast<PairPostProbe *>(context);
  ++probe.calls;
  probe.rows = rows;
  probe.leading_columns = leading_columns;
  probe.range = range;
}

TEST(OneDnnProvider, ExposesStableIdentity) {
  const auto provider = hyle::compute::onednn::provider();
  EXPECT_TRUE(provider.valid());
  EXPECT_EQ(provider.id(), hyle::compute::onednn::kProviderId);
  EXPECT_EQ(provider.name(), "onednn");
  EXPECT_EQ(hyle::compute::onednn::version(), "3.13.2");

  auto context = Context::create(hyle::compute::cpu::device(), provider);
  EXPECT_TRUE(context.has_value());
  auto invalid =
      Context::create(hyle::compute::cpu::device(), hyle::compute::Provider{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  const hyle::compute::ParallelExecutorRef serial(nullptr, 8, nullptr);
  EXPECT_EQ(serial.concurrency(), 1U);
}

TEST(OneDnnProvider, DirectBrgemmDeclinesNonExactShapeDomains) {
  using namespace hyle::compute::detail;
  auto context = Context::create(hyle::compute::cpu::device(),
                                 hyle::compute::onednn::provider()).value();
  // Match the supported prefill shape below; tiny serial decode deliberately
  // declines to Native independently of the shape-domain contract.
  const std::array inputs{matrix(16, 5), matrix(5, 4)};
  const std::array outputs{matrix(16, 4)};
  std::array ranges{ProviderShapeRange{1, 16, 1}};
  std::array<ProviderTensorAxes, 2> input_axes{};
  std::array<ProviderTensorAxes, 1> output_axes{};
  input_axes[0].symbols[0] = output_axes[0].symbols[0] = 0;
  const ProviderShapeDomain domain{ranges, input_axes, output_axes};
  std::array<ResourceBindingRequirement, 2> input_requirements{};
  std::array<ResourceBindingRequirement, 1> output_requirements{};
  const auto propose = [&]() {
    return OperationPreparationAccess::propose(
        context, 0, hyle::compute::Operation(MatMul{}), inputs, outputs,
        input_requirements, output_requirements, domain);
  };
  auto bounded = propose();
  ASSERT_FALSE(bounded.has_value());
  EXPECT_TRUE(bounded.error().IsUnimplemented());
  ranges[0].minimum = 16;
  auto exact = propose();
  ASSERT_TRUE(exact.has_value()) << exact.error().message();
  EXPECT_EQ(exact->shape_coverage, ProviderShapeCoverage::kExactPoint);
}

TEST(OneDnnProvider, InternalPlanExecutesSharedLeftPairInOneEpisode) {
  namespace internal = hyle::compute::onednn::internal;
  constexpr std::size_t rows = 2;
  constexpr std::size_t inner = 3;
  constexpr std::size_t columns = 4;
  const AllocatorRef plan_allocator = hyle::mem::system_allocator();
  BrgemmPlanOwner owner{.allocator = plan_allocator};
  ASSERT_TRUE(internal::create_brgemm_plan(
                  {.rows = rows, .inner = inner, .columns = columns}, 1,
                  plan_allocator, &owner.plan)
                  .ok());
  ASSERT_NE(owner.plan, nullptr);
  EXPECT_EQ(internal::brgemm_task_count(owner.plan), 1u);

  const std::size_t scratchpad_size =
      internal::brgemm_scratchpad_size(owner.plan);
  ASSERT_GT(scratchpad_size, 0u);
  Buffer scratchpad =
      Buffer::create(hyle::mem::system_resource(), scratchpad_size,
                     internal::kBrgemmWorkspaceAlignment)
          .value();
  auto mapping = scratchpad.view().map_host(hyle::mem::HostAccess::kReadWrite);
  ASSERT_TRUE(mapping.has_value()) << mapping.error().message();
  auto scratch_bytes = mapping->writable_bytes();
  ASSERT_TRUE(scratch_bytes.has_value()) << scratch_bytes.error().message();
  ASSERT_TRUE(internal::validate_brgemm_scratchpad(
                  owner.plan, scratch_bytes->data(), scratch_bytes->size())
                  .ok());
  EXPECT_TRUE(internal::validate_brgemm_scratchpad(
                  owner.plan, scratch_bytes->data(), scratch_bytes->size() - 1)
                  .IsInvalidArgument());

  constexpr std::array<float, rows * inner> left{1, 2, 3, 4, 5, 6};
  constexpr std::array<float, inner * columns> first_right{1, 0, 0, 1, 0, 1,
                                                           1, 0, 1, 1, 0, 0};
  constexpr std::array<float, inner * columns> second_right{2, 0, 1, 0, 0, 2,
                                                            0, 1, 1, 0, 2, 0};
  std::array<float, rows * columns> first_output{};
  std::array<float, rows * columns> second_output{};
  PairPostProbe post{};
  ParallelProbe executor_probe{};
  const hyle::compute::ParallelExecutorRef executor(
      &executor_probe, 1, &probe_parallel_for, &probe_in_parallel);
  ASSERT_TRUE(internal::execute_brgemm_pair(
                  owner.plan, executor, left.data(), first_right.data(),
                  second_right.data(), first_output.data(),
                  second_output.data(), scratch_bytes->data(),
                  &record_pair_post, &post)
                  .ok());

  EXPECT_EQ(first_output,
            (std::array<float, rows * columns>{4, 5, 2, 1, 10, 11, 5, 4}));
  EXPECT_EQ(second_output,
            (std::array<float, rows * columns>{5, 4, 7, 2, 14, 10, 16, 5}));
  EXPECT_EQ(post.calls, 1u);
  EXPECT_EQ(post.rows, rows);
  EXPECT_EQ(post.leading_columns, columns);
  EXPECT_EQ(post.range.first_column, 0u);
  EXPECT_EQ(post.range.column_count, columns);
  EXPECT_EQ(executor_probe.episodes.load(), 1u);
  EXPECT_EQ(executor_probe.tasks.load(), 1u);
  EXPECT_TRUE(mapping->close().ok());
}

TEST(OneDnnProvider, PreparesAndRunsRowMajorFloat32MatMul) {
  constexpr std::size_t rows = 16;
  constexpr std::size_t inner = 5;
  constexpr std::size_t columns = 4;
  std::vector<float> left(rows * inner);
  std::vector<float> right(inner * columns);
  for (std::size_t index = 0; index < left.size(); ++index)
    left[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.25F;
  for (std::size_t index = 0; index < right.size(); ++index)
    right[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.5F;
  std::vector<float> expected(rows * columns, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t k = 0; k < inner; ++k) {
        expected[row * columns + column] +=
            left[row * inner + k] * right[k * columns + column];
      }
    }
  }

  const TensorDesc left_description = matrix(rows, inner);
  const TensorDesc right_description = matrix(inner, columns);
  const TensorDesc output_description = matrix(rows, columns);
  Buffer left_buffer = Buffer::create(hyle::mem::system_resource(),
                                      left.size() * sizeof(float), 64)
                           .value();
  Buffer right_buffer = Buffer::create(hyle::mem::system_resource(),
                                       right.size() * sizeof(float), 64)
                            .value();
  Buffer output_buffer = Buffer::create(hyle::mem::system_resource(),
                                        expected.size() * sizeof(float), 64)
                             .value();
  write_floats(left_buffer.view(), left);
  write_floats(right_buffer.view(), right);

  Context context = create_context();
  auto queue = context.create_queue().value();
  PreparedOperation prepared =
      hyle::compute::prepare(context, MatMul{}, left_description,
                             right_description, output_description)
          .value();
  EXPECT_EQ(prepared.provider().id, hyle::compute::onednn::kProviderId);

  const auto requirement = prepared.workspace_requirement();
  Buffer workspace;
  if (requirement.size != 0) {
    workspace = Buffer::create(hyle::mem::system_resource(), requirement.size,
                               requirement.alignment)
                    .value();
    auto absent = prepared.create_invocation();
    ASSERT_FALSE(absent.has_value());
    EXPECT_TRUE(absent.error().IsInvalidArgument());
  }
  auto invocation = prepared.create_invocation(workspace.view()).value();
  auto left_view =
      ConstTensorView::create(left_description, left_buffer.view()).value();
  auto right_view =
      ConstTensorView::create(right_description, right_buffer.view()).value();
  auto output_view =
      TensorView::create(output_description, output_buffer.view()).value();
  auto completion =
      invocation.submit(queue, left_view, right_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());

  const auto actual = read_floats(output_buffer.view(), expected.size());
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 2.0e-5F + 2.0e-5F);
  }
}

TEST(OneDnnProvider, RejectsContiguousKWeightsWithoutAnotherProvider) {
  constexpr std::int64_t rows = 2;
  constexpr std::int64_t inner = 3;
  constexpr std::int64_t columns = 4;
  const TensorDesc left = matrix(rows, inner);
  const TensorDesc physical_weights = matrix(columns, inner);
  const TensorDesc logical_weights = physical_weights.transposed().value();
  const TensorDesc output = matrix(rows, columns);

  Context context = create_context();
  auto prepared =
      hyle::compute::prepare(context, MatMul{}, left, logical_weights, output);
  ASSERT_FALSE(prepared.has_value());
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}

TEST(OneDnnProvider, RejectsEncodedWeightsBeforeCallingOneDnn) {
  constexpr StorageFormatId codec{0x48594c45U, 37U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const TensorDesc left = matrix(16, 64);
  const std::array<std::int64_t, 2> stored_shape{32, 64};
  const TensorDesc right =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();
  const TensorDesc output = matrix(16, 32);

  Context context = create_context();
  auto prepared =
      hyle::compute::prepare(context, MatMul{}, left, right, output);
  ASSERT_FALSE(prepared.has_value());
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}

TEST(OneDnnProvider, UsesTheBorrowedSynchronousExecutor) {
  constexpr std::size_t rows = 7;
  constexpr std::size_t inner = 17;
  constexpr std::size_t columns = 257;
  const TensorDesc left_description = matrix(rows, inner);
  const TensorDesc right_description = matrix(inner, columns);
  const TensorDesc output_description = matrix(rows, columns);
  const std::vector<float> left_values(rows * inner, 1.0F);
  const std::vector<float> right_values(inner * columns, 1.0F);
  Buffer left = Buffer::create(hyle::mem::system_resource(),
                               left_values.size() * sizeof(float), 64)
                    .value();
  Buffer right = Buffer::create(hyle::mem::system_resource(),
                                right_values.size() * sizeof(float), 64)
                     .value();
  Buffer output = Buffer::create(hyle::mem::system_resource(),
                                 rows * columns * sizeof(float), 64)
                      .value();
  write_floats(left.view(), left_values);
  write_floats(right.view(), right_values);

  ParallelProbe probe{.in_parallel = true};
  const hyle::compute::ParallelExecutorRef executor(
      &probe, 4, &probe_parallel_for, &probe_in_parallel);
  auto context =
      Context::create(hyle::compute::cpu::device(),
                      hyle::compute::onednn::provider({.executor = executor}))
          .value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, MatMul{}, left_description,
                                         right_description, output_description)
                      .value();
  Buffer workspace;
  const auto requirement = prepared.workspace_requirement();
  if (requirement.size != 0) {
    workspace = Buffer::create(hyle::mem::system_resource(), requirement.size,
                               requirement.alignment)
                    .value();
  }
  auto invocation = prepared.create_invocation(workspace.view()).value();
  auto left_view =
      ConstTensorView::create(left_description, left.view()).value();
  auto right_view =
      ConstTensorView::create(right_description, right.view()).value();
  auto output_view =
      TensorView::create(output_description, output.view()).value();
  auto completion =
      invocation.submit(queue, left_view, right_view, output_view);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());

  EXPECT_GT(probe.episodes.load(), 0U);
  EXPECT_EQ(probe.tasks.load(), 4U);
  for (float observed : read_floats(output.view(), rows * columns))
    EXPECT_EQ(observed, static_cast<float>(inner));
}

TEST(OneDnnProvider, RejectsSmallSerialDecodeWithoutAnotherProvider) {
  Context context = create_context();
  for (const std::int64_t rows : {1, 4}) {
    const TensorDesc left = matrix(rows, 17);
    const TensorDesc right = matrix(17, 65);
    const TensorDesc output = matrix(rows, 65);
    auto prepared =
        hyle::compute::prepare(context, MatMul{}, left, right, output);
    ASSERT_FALSE(prepared.has_value());
    EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
}

TEST(OneDnnProvider, ValidatesCallerOwnedWorkspaceSizeAndAlignment) {
  const TensorDesc description = matrix(32, 32);
  Context context = create_context();
  PreparedOperation prepared =
      hyle::compute::prepare(context, MatMul{}, description, description,
                             description)
          .value();
  ASSERT_EQ(prepared.provider().id, hyle::compute::onednn::kProviderId);
  const auto requirement = prepared.workspace_requirement();
  ASSERT_GT(requirement.size, 1U);
  ASSERT_EQ(requirement.alignment, 64U);

  Buffer storage =
      Buffer::create(hyle::mem::system_resource(), requirement.size + 64, 64)
          .value();
  auto too_small_view = storage.view().subview(0, requirement.size - 1).value();
  auto too_small = prepared.create_invocation(too_small_view);
  ASSERT_FALSE(too_small.has_value());
  EXPECT_TRUE(too_small.error().IsInvalidArgument());

  auto misaligned_view = storage.view().subview(1, requirement.size).value();
  auto misaligned = prepared.create_invocation(misaligned_view);
  ASSERT_FALSE(misaligned.has_value());
  EXPECT_TRUE(misaligned.error().IsInvalidArgument());

  auto exact_view = storage.view().subview(0, requirement.size).value();
  EXPECT_TRUE(prepared.create_invocation(exact_view).has_value());
}

TEST(OneDnnProvider, RejectsUnsupportedEmptyOperationsWithoutAnotherProvider) {
  const TensorDesc empty_left = matrix(0, 3);
  const TensorDesc right = matrix(3, 4);
  const TensorDesc empty_output = matrix(0, 4);
  Context context = create_context();
  auto prepared = hyle::compute::prepare(context, MatMul{}, empty_left, right,
                                         empty_output);
  ASSERT_FALSE(prepared.has_value());
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}

TEST(OneDnnProvider, SharesPreparedArtifactAcrossIndependentInvocations) {
  constexpr std::size_t extent = 32;
  const TensorDesc description = matrix(extent, extent);
  Context context = create_context();
  PreparedOperation prepared =
      hyle::compute::prepare(context, MatMul{}, description, description,
                             description)
          .value();
  PreparedOperation shared = prepared;
  std::atomic<int> failures{0};

  auto run = [&](const PreparedOperation &artifact, float value) {
    const std::vector<float> left(extent * extent, value);
    const std::vector<float> right(extent * extent, 1.0F);
    Buffer left_buffer = Buffer::create(hyle::mem::system_resource(),
                                        left.size() * sizeof(float), 64)
                             .value();
    Buffer right_buffer = Buffer::create(hyle::mem::system_resource(),
                                         right.size() * sizeof(float), 64)
                              .value();
    Buffer output_buffer = Buffer::create(hyle::mem::system_resource(),
                                          extent * extent * sizeof(float), 64)
                               .value();
    write_floats(left_buffer.view(), left);
    write_floats(right_buffer.view(), right);
    Buffer workspace;
    const auto requirement = artifact.workspace_requirement();
    if (requirement.size != 0) {
      workspace = Buffer::create(hyle::mem::system_resource(), requirement.size,
                                 requirement.alignment)
                      .value();
    }
    auto invocation = artifact.create_invocation(workspace.view());
    auto queue = context.create_queue();
    auto left_view = ConstTensorView::create(description, left_buffer.view());
    auto right_view = ConstTensorView::create(description, right_buffer.view());
    auto output_view = TensorView::create(description, output_buffer.view());
    if (!invocation || !queue || !left_view || !right_view || !output_view) {
      ++failures;
      return;
    }
    auto completion =
        invocation->submit(*queue, *left_view, *right_view, *output_view);
    if (!completion || !completion->wait().ok()) {
      ++failures;
      return;
    }
    const auto actual = read_floats(output_buffer.view(), extent * extent);
    for (float observed : actual) {
      if (observed != value * static_cast<float>(extent)) {
        ++failures;
        return;
      }
    }
  };

  std::thread first(run, std::cref(prepared), 1.0F);
  std::thread second(run, std::cref(shared), 2.0F);
  first.join();
  second.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST(OneDnnProvider, ReportsMetadataOomWithoutLeakingOwnedState) {
  AllocatorProbe probe;
  {
    ContextOptions options;
    options.metadata_allocator = allocator(probe);
    const std::array providers{hyle::compute::onednn::provider()};
    options.providers = providers;
    Context context =
        Context::create(hyle::compute::cpu::device(), options).value();
    const TensorDesc description = matrix(16, 16);

    probe.fail = true;
    auto failed_prepare = hyle::compute::prepare(context, MatMul{}, description,
                                                 description, description);
    ASSERT_FALSE(failed_prepare.has_value());
    EXPECT_TRUE(failed_prepare.error().IsResourceExhausted());

    probe.fail = false;
    probe.fail_at_attempt = probe.attempts.load() + 2;
    auto failed_wrapper = hyle::compute::prepare(context, MatMul{}, description,
                                                 description, description);
    ASSERT_FALSE(failed_wrapper.has_value());
    EXPECT_TRUE(failed_wrapper.error().IsResourceExhausted());
    EXPECT_EQ(failed_wrapper.error().message(),
              "oneDNN prepared wrapper allocation failed");
    probe.fail_at_attempt = 0;

    PreparedOperation prepared =
        hyle::compute::prepare(context, MatMul{}, description, description,
                               description)
            .value();
    Buffer workspace;
    const auto requirement = prepared.workspace_requirement();
    if (requirement.size != 0) {
      workspace = Buffer::create(hyle::mem::system_resource(), requirement.size,
                                 requirement.alignment)
                      .value();
    }
    probe.fail = true;
    auto failed_invocation = prepared.create_invocation(workspace.view());
    ASSERT_FALSE(failed_invocation.has_value());
    EXPECT_TRUE(failed_invocation.error().IsResourceExhausted());
    probe.fail = false;
  }
  EXPECT_EQ(probe.allocations.load(), probe.deallocations.load());
}

} // namespace
