#include "hyle/compute/providers/cublas.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/leased_operation.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/dtype.h"

namespace {

namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

struct LeaseMetadataCounter final {
  std::atomic<std::size_t> allocations{0};
  mem::AllocatorRef ref() noexcept {
    return {
        [](void *raw, std::size_t bytes, std::size_t alignment) noexcept {
          static_cast<LeaseMetadataCounter *>(raw)->allocations.fetch_add(1);
          return mem::system_allocator().allocate(bytes, alignment);
        },
        [](void *, void *pointer, std::size_t bytes, std::size_t alignment) noexcept {
          mem::system_allocator().deallocate(pointer, bytes, alignment);
        },
        [](void *raw, void *pointer, std::size_t old_bytes, std::size_t bytes,
           std::size_t alignment) noexcept {
          static_cast<LeaseMetadataCounter *>(raw)->allocations.fetch_add(1);
          return mem::system_allocator().reallocate(pointer, old_bytes, bytes, alignment);
        }, this};
  }
};

// Releases on every exit, including failed ASSERTs. The native stream remains
// the lifetime boundary; teardown observes the callback's final host access.
struct LeasedStreamGate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<LeasedStreamGate *>(raw);
    gate.released.wait(false, std::memory_order_acquire);
  }
  void release() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
  ~LeasedStreamGate() {
    release();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

compute::Context context() {
  return compute::Context::create(cuda::device(0).value(),
                                  compute::cublas::provider())
      .value();
}

compute::TensorDesc matrix(compute::DataType type, std::int64_t rows,
                           std::int64_t columns, bool column_major) {
  const std::array shape{rows, columns};
  const auto width = static_cast<std::int64_t>(compute::data_type_size(type));
  const std::array strides = column_major ? std::array{width, rows * width}
                                          : std::array{columns * width, width};
  return compute::TensorDesc::strided_bytes(type, shape, strides).value();
}

struct Matrix final {
  compute::TensorDesc description;
  mem::Buffer storage;
  compute::TensorView view;

  Matrix(compute::TensorDesc desc, mem::MemoryResourceRef resource)
      : description(desc),
        storage(
            mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                .value()),
        view(compute::TensorView::create(desc, storage.view()).value()) {}
};

std::size_t offset(const compute::TensorDesc &desc, std::size_t row,
                   std::size_t column) {
  return (row * static_cast<std::size_t>(desc.byte_strides()[0]) +
          column * static_cast<std::size_t>(desc.byte_strides()[1])) /
         compute::data_type_size(desc.data_type());
}

template <class T> float to_float(T value) { return static_cast<float>(value); }
template <> float to_float(__half value) { return __half2float(value); }
template <> float to_float(hyle::simd::bfloat16_t value) {
  return hyle::simd::bf16_to_f32(value);
}
template <class T> T from_float(float value) { return static_cast<T>(value); }
template <> __half from_float(float value) { return __float2half(value); }
template <> hyle::simd::bfloat16_t from_float(float value) {
  return hyle::simd::f32_to_bf16(value);
}

template <class T>
std::vector<T> values(const compute::TensorDesc &desc, int seed) {
  std::vector<T> result(desc.required_bytes().value() / sizeof(T));
  const auto rows = static_cast<std::size_t>(desc.dimensions()[0]);
  const auto columns = static_cast<std::size_t>(desc.dimensions()[1]);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < columns; ++col) {
      const int value = static_cast<int>((row * 13 + col * 7) % 19) - 9 + seed;
      result[offset(desc, row, col)] =
          from_float<T>(static_cast<float>(value) * 0.013173f);
    }
  }
  return result;
}

template <class T>
void check_product(const Matrix &left, const std::vector<T> &a,
                   const Matrix &right, const std::vector<T> &b,
                   Matrix &output) {
  std::vector<T> result(output.storage.size() / sizeof(T));
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  const auto m = static_cast<std::size_t>(left.description.dimensions()[0]);
  const auto k = static_cast<std::size_t>(left.description.dimensions()[1]);
  const auto n = static_cast<std::size_t>(right.description.dimensions()[1]);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      double expected = 0;
      for (std::size_t inner = 0; inner < k; ++inner)
        expected += static_cast<double>(
                        to_float(a[offset(left.description, row, inner)])) *
                    to_float(b[offset(right.description, inner, col)]);
      const double tolerance =
          std::is_same_v<T, hyle::simd::bfloat16_t>
              ? 0.008 * std::abs(expected) + 0.0001
              : (sizeof(T) == 2
                     ? 0.0007 * std::abs(expected) + 0.00004
                     : 0.000001 * std::abs(expected) + 0.000001);
      EXPECT_NEAR(to_float(result[offset(output.description, row, col)]),
                  expected, tolerance)
          << row << "," << col;
    }
  }
}

template <class T>
void run_matrix(compute::DataType type, bool left_column, bool right_column,
                bool output_column, std::int64_t m, std::int64_t k,
                std::int64_t n) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  std::array queues{owner.create_queue().value(), owner.create_queue().value()};
  for (auto &queue : queues)
    ASSERT_TRUE(queue.reserve_completions(1).ok());
  Matrix left(matrix(type, m, k, left_column), resource.ref());
  Matrix right(matrix(type, k, n, right_column), resource.ref());
  Matrix first(matrix(type, m, n, output_column), resource.ref());
  Matrix second(first.description, resource.ref());
  const std::array inputs{left.view.as_const(), right.view.as_const()};
  const std::array outputs{first.view};
  auto preparation = compute::prepare(
      owner, compute::Operation(compute::MatMul{}), inputs, outputs);
  ASSERT_TRUE(preparation.has_value()) << preparation.error().message();
  auto prepared = std::move(*preparation);
  EXPECT_EQ(prepared.provider().id, compute::cublas::kProviderId);
  EXPECT_TRUE(prepared.capture_compatible());
  EXPECT_FALSE(prepared.runtime_properties().blocks_host);
  EXPECT_FALSE(prepared.invocation_private_bytes().has_value());
  const auto needed = prepared.workspace_requirement();
  std::array workspaces{
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value(),
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value()};
  std::array lanes{prepared.create_invocation(workspaces[0].view()).value(),
                   prepared.create_invocation(workspaces[1].view()).value()};
  // Independent output/workspace/handle per lane; immutable weights are shared.
  for (int seed = 0; seed < 3; ++seed) {
    const auto a = values<T>(left.description, seed);
    const auto b = values<T>(right.description, seed + 2);
    ASSERT_TRUE(left.storage.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(right.storage.view().write(std::as_bytes(std::span(b))).ok());
    auto done_first = lanes[0].submit(queues[0], left.view.as_const(),
                                      right.view.as_const(), first.view);
    ASSERT_TRUE(done_first.has_value()) << done_first.error().message();
    auto done_second = lanes[1].submit(queues[1], left.view.as_const(),
                                       right.view.as_const(), second.view);
    ASSERT_TRUE(done_second.has_value()) << done_second.error().message();
    ASSERT_TRUE(done_first->wait().ok());
    ASSERT_TRUE(done_second->wait().ok());
    check_product(left, a, right, b, first);
    check_product(left, a, right, b, second);
  }
}

TEST(CublasProvider, Float32LayoutsAndChangedInputsMatchDoubleOracle) {
  for (int layouts = 0; layouts < 4; ++layouts) {
    SCOPED_TRACE(layouts);
    run_matrix<float>(compute::DataType::kFloat32, (layouts & 1) != 0,
                      (layouts & 2) != 0, false, 7, 17, 13);
  }
}

TEST(CublasProvider, HalfDecodeAndPrefillUseIndependentLanes) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_matrix<__half>(compute::DataType::kFloat16, false, false, false, rows,
                       128, 64);
    run_matrix<__half>(compute::DataType::kFloat16, false, true, false, rows,
                       128, 64);
  }
}

TEST(CublasProvider, Bfloat16DecodeAndPrefillUseIndependentLanes) {
  for (const auto rows : {1, 17}) {
    SCOPED_TRACE(rows);
    run_matrix<hyle::simd::bfloat16_t>(
        compute::DataType::kBFloat16, false, false, false, rows, 128, 64);
    run_matrix<hyle::simd::bfloat16_t>(
        compute::DataType::kBFloat16, false, true, false, rows, 128, 64);
  }
}

TEST(CublasProvider, BfloatLinearF32KeepsStoredWeightOrderAndSubBfloatScores) {
  using BFloat = hyle::simd::bfloat16_t;
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(1).ok());
  // All three axes differ, so treating stored [E,K] as [K,E] row-major
  // cannot accidentally pass. Input and output elements also differ in size.
  Matrix input(matrix(compute::DataType::kBFloat16, 3, 5, false), resource.ref());
  Matrix weights(matrix(compute::DataType::kBFloat16, 7, 5, false), resource.ref());
  Matrix output(matrix(compute::DataType::kFloat32, 3, 7, false), resource.ref());
  std::vector<BFloat> stored(35);
  for (std::size_t e = 0; e < 7; ++e)
    for (std::size_t k = 0; k < 5; ++k)
      stored[e * 5 + k] = from_float<BFloat>(
          static_cast<float>(static_cast<int>((e * 3 + k * 2) % 9) - 4) * .25F);
  for (std::size_t k = 0; k < 5; ++k)
    stored[k] = from_float<BFloat>(0);
  stored[0] = from_float<BFloat>(1);
  stored[1] = from_float<BFloat>(1.0F / 128);
  ASSERT_TRUE(weights.storage.view().write(std::as_bytes(std::span(stored))).ok());
  auto preparation = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{input.view.as_const(), weights.view.as_const()}, std::array{output.view});
  ASSERT_TRUE(preparation) << preparation.error().message();
  auto prepared = std::move(*preparation);
  EXPECT_EQ(prepared.provider().id, compute::cublas::kProviderId);
  EXPECT_TRUE(prepared.capture_compatible());
  const auto needed = prepared.workspace_requirement();
  EXPECT_LE(needed.size, std::size_t{32} * 1024 * 1024);
  auto workspace = mem::Buffer::create(resource.ref(), needed.size, needed.alignment).value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  for (int phase = 0; phase < 2; ++phase) {
    SCOPED_TRACE(phase);
    std::vector<BFloat> rows(15);
    for (std::size_t i = 0; i < rows.size(); ++i)
      rows[i] = from_float<BFloat>(
          static_cast<float>(static_cast<int>((i * 5 + static_cast<std::size_t>(phase)) % 11) - 5) * .5F);
    for (std::size_t k = 0; k < 5; ++k)
      rows[k] = from_float<BFloat>(0);
    rows[0] = from_float<BFloat>(1);
    rows[1] = from_float<BFloat>(.25F);
    ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(rows))).ok());
    auto completion = invocation.submit(queue, input.view.as_const(),
        weights.view.as_const(), output.view);
    ASSERT_TRUE(completion) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    std::array<float, 21> actual{};
    ASSERT_TRUE(output.storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
    for (std::size_t t = 0; t < 3; ++t)
      for (std::size_t e = 0; e < 7; ++e) {
        double expected = 0;
        for (std::size_t k = 0; k < 5; ++k)
          expected += static_cast<double>(to_float(rows[t * 5 + k])) *
              to_float(stored[e * 5 + k]);
        // The small dyadic products and sums are exact in F32. A broad GEMM
        // tolerance would hide a layout/byte-width or BF16-output regression.
        EXPECT_EQ(actual[t * 7 + e], expected) << t << "," << e;
      }
    EXPECT_EQ(actual[0], 1.001953125F);
    EXPECT_EQ(to_float(from_float<BFloat>(actual[0])), 1.0F);
  }
}

TEST(CublasProvider, ScoreOperationDoesNotRelaxOrdinaryMatMulDtypes) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto input = matrix(compute::DataType::kBFloat16, 3, 5, false);
  const auto stored = matrix(compute::DataType::kBFloat16, 7, 5, false);
  const auto rhs = matrix(compute::DataType::kBFloat16, 5, 7, false);
  const auto score = matrix(compute::DataType::kFloat32, 3, 7, false);
  const auto rounded = matrix(compute::DataType::kBFloat16, 3, 7, false);
  const auto wrong_weight = matrix(compute::DataType::kFloat32, 7, 5, false);
  auto ordinary = compute::prepare(owner, compute::Operation(compute::MatMul{}),
      std::array{input, rhs}, std::array{score}, resource.ref());
  ASSERT_FALSE(ordinary);
  EXPECT_TRUE(ordinary.error().IsInvalidArgument());
  auto wrong_score = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{input, stored}, std::array{rounded}, resource.ref());
  ASSERT_FALSE(wrong_score);
  EXPECT_TRUE(wrong_score.error().IsInvalidArgument());
  auto mixed_inputs = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{input, wrong_weight}, std::array{score}, resource.ref());
  ASSERT_FALSE(mixed_inputs);
  EXPECT_TRUE(mixed_inputs.error().IsUnimplemented());
}

TEST(CublasProvider, DenseHalfProjectionKeepsFloat32OutputAtOddInnerWidth) {
  auto owner = context();
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  Matrix input(matrix(compute::DataType::kFloat16, 3, 5, false), resource.ref());
  Matrix weights(matrix(compute::DataType::kFloat16, 7, 5, false), resource.ref());
  Matrix output(matrix(compute::DataType::kFloat32, 3, 7, false), resource.ref());
  std::array<__half, 15> a{}; std::array<__half, 35> w{};
  for (std::size_t row = 0; row < 3; ++row) {
    a[row * 5] = __float2half(1); a[row * 5 + 1] = __float2half(.25F);
  }
  for (std::size_t column = 0; column < 7; ++column) {
    w[column * 5] = __float2half(float(column + 1));
    w[column * 5 + 1] = __float2half(1.F / 2048);
  }
  ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(weights.storage.view().write(std::as_bytes(std::span(w))).ok());
  auto prepared = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{input.view.as_const(), weights.view.as_const()}, std::array{output.view});
  ASSERT_TRUE(prepared) << prepared.error().message();
  const auto need = prepared->workspace_requirement();
  auto workspace = mem::Buffer::create(resource.ref(), need.size, need.alignment).value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto done = invocation.submit(queue, input.view.as_const(), weights.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message(); ASSERT_TRUE(done->wait().ok());
  std::array<float, 21> actual{};
  ASSERT_TRUE(output.storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t row = 0; row < 3; ++row) for (std::size_t column = 0; column < 7; ++column)
    EXPECT_EQ(actual[row * 7 + column], float(column + 1) + 1.F / 8192);
  EXPECT_EQ(__half2float(__float2half(actual[0])), 1.F);
  const auto rejected = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{matrix(compute::DataType::kFloat32, 3, 5, false), weights.description},
      std::array{output.description}, resource.ref());
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsUnimplemented());
}

template<class T> void batched_dense_projection(compute::DataType type) {
  auto owner = context();
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto tensor = [](compute::DataType dtype, std::initializer_list<std::int64_t> shape) {
    return compute::TensorDesc::dense(dtype, std::span(shape.begin(), shape.size())).value();
  };
  Matrix input(tensor(type, {2, 3, 5}), resource.ref());
  Matrix weights(tensor(type, {2, 7, 5}), resource.ref());
  Matrix output(tensor(compute::DataType::kFloat32, {2, 3, 7}), resource.ref());
  std::vector<T> a(30, from_float<T>(0)), w(70, from_float<T>(0));
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t row = 0; row < 3; ++row) {
      const auto first = (batch * 3 + row) * 5;
      a[first] = from_float<T>(1.F + float(row + 1) / 262144); a[first + 1] = from_float<T>(-1);
      a[first + 2] = from_float<T>(1); a[first + 3] = from_float<T>(.25F);
    }
    const auto first = batch * 35;
    w[first] = w[first + 1] = from_float<T>(1);
    w[first + 5 + 2] = from_float<T>(1); w[first + 5 + 3] = from_float<T>(1.F / 2048);
    for (std::size_t column = 2; column < 7; ++column)
      w[first + column * 5 + 2] = from_float<T>(float((batch + 1) * column));
  }
  ASSERT_TRUE(input.storage.view().write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(weights.storage.view().write(std::as_bytes(std::span(w))).ok());
  auto prepared = compute::prepare(owner, compute::model_ops::dense_linear_f32(),
      std::array{input.view.as_const(), weights.view.as_const()}, std::array{output.view});
  ASSERT_TRUE(prepared) << prepared.error().message(); const auto need = prepared->workspace_requirement();
  auto workspace = mem::Buffer::create(resource.ref(), need.size, need.alignment).value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto done = invocation.submit(queue, input.view.as_const(), weights.view.as_const(), output.view);
  ASSERT_TRUE(done) << done.error().message(); ASSERT_TRUE(done->wait().ok());
  std::array<float, 42> actual{};
  ASSERT_TRUE(output.storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
  for (std::size_t batch = 0; batch < 2; ++batch) for (std::size_t row = 0; row < 3; ++row) {
    const auto first = (batch * 3 + row) * 7;
    EXPECT_EQ(actual[first], to_float(a[(batch * 3 + row) * 5]) - 1.F);
    EXPECT_EQ(actual[first + 1], 1.F + 1.F / 8192);
    for (std::size_t column = 2; column < 7; ++column)
      EXPECT_EQ(actual[first + column], float((batch + 1) * column));
  }
}
TEST(CublasProvider, DenseFloatingProjectionPreservesExactBatchAndFloatSource) {
  batched_dense_projection<__half>(compute::DataType::kFloat16);
  batched_dense_projection<float>(compute::DataType::kFloat32);
}

compute::TensorDesc batched_matrix(compute::DataType type,
                                   std::vector<std::int64_t> shape,
                                   bool column_major) {
  const auto width = static_cast<std::int64_t>(compute::data_type_size(type));
  std::vector<std::int64_t> strides(shape.size(), width);
  for (auto axis = shape.size() - 1; axis-- > 0;)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  if (column_major) {
    strides[shape.size() - 2] = width;
    strides[shape.size() - 1] = shape[shape.size() - 2] * width;
  }
  return compute::TensorDesc::strided_bytes(type, shape, strides).value();
}

// Independent logical broadcasting oracle: decode each output batch coordinate
// and index source dimensions directly, without the provider's affine runs.
std::size_t broadcast_batch_offset(const compute::TensorDesc &source,
                                   const compute::TensorDesc &output,
                                   std::size_t batch) {
  const auto source_rank = source.rank() - 2;
  const auto output_rank = output.rank() - 2;
  std::size_t offset_bytes = 0;
  for (auto axis = output_rank; axis-- > 0;) {
    const auto coordinate = batch %
        static_cast<std::size_t>(output.dimensions()[axis]);
    batch /= static_cast<std::size_t>(output.dimensions()[axis]);
    if (axis + source_rank < output_rank)
      continue;
    const auto source_axis = axis + source_rank - output_rank;
    if (source.dimensions()[source_axis] != 1)
      offset_bytes += coordinate *
          static_cast<std::size_t>(source.byte_strides()[source_axis]);
  }
  return offset_bytes;
}

std::size_t batched_element(const compute::TensorDesc &source,
                            const compute::TensorDesc &output,
                            std::size_t batch, std::size_t row,
                            std::size_t column) {
  return (broadcast_batch_offset(source, output, batch) +
          row * static_cast<std::size_t>(
              source.byte_strides()[source.rank() - 2]) +
          column * static_cast<std::size_t>(
              source.byte_strides()[source.rank() - 1])) /
         compute::data_type_size(source.data_type());
}

template <class T>
void run_batched_broadcast(compute::DataType type, bool multiple_resets) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  std::array queues{owner.create_queue().value(), owner.create_queue().value()};
  for (auto &queue : queues)
    ASSERT_TRUE(queue.reserve_completions(1).ok());
  // First domain broadcasts A within a run; the second broadcasts B and has
  // two outer coordinate axes. Odd small matrix sizes give real non-16-byte
  // batch offsets/strides for all three storage types.
  const auto left_desc = batched_matrix(type,
      multiple_resets ? std::vector<std::int64_t>{2, 1, 2, 2, 3}
                      : std::vector<std::int64_t>{2, 1, 2, 3},
      multiple_resets);
  const auto right_desc = batched_matrix(type,
      multiple_resets ? std::vector<std::int64_t>{1, 3, 1, 3, 5}
                      : std::vector<std::int64_t>{1, 3, 3, 5},
      !multiple_resets);
  const auto output_desc = batched_matrix(type,
      multiple_resets ? std::vector<std::int64_t>{2, 3, 2, 2, 5}
                      : std::vector<std::int64_t>{2, 3, 2, 5},
      false);
  const auto left_axis = multiple_resets ? std::size_t{2} : std::size_t{0};
  EXPECT_NE(left_desc.byte_strides()[left_axis] % 16, 0);
  EXPECT_NE(right_desc.byte_strides()[1] % 16, 0);
  EXPECT_NE(output_desc.byte_strides()[output_desc.rank() - 3] % 16, 0);
  Matrix left(left_desc, resource.ref()), right(right_desc, resource.ref()),
      first(output_desc, resource.ref()), second(output_desc, resource.ref());
  auto result = compute::prepare(owner, compute::Operation(compute::MatMul{}),
      std::array{left.view.as_const(), right.view.as_const()},
      std::array{first.view});
  ASSERT_TRUE(result) << result.error().message();
  auto prepared = std::move(*result);
  EXPECT_TRUE(prepared.capture_compatible());
  const auto needed = prepared.workspace_requirement();
  std::array workspaces{
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment).value(),
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment).value()};
  std::array lanes{prepared.create_invocation(workspaces[0].view()).value(),
                   prepared.create_invocation(workspaces[1].view()).value()};
  for (int phase = 0; phase < 2; ++phase) {
    std::vector<T> a(left.storage.size() / sizeof(T));
    std::vector<T> b(right.storage.size() / sizeof(T));
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = from_float<T>(float(int((i * 5 + std::size_t(phase * 2)) % 11) - 5) * .25F);
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = from_float<T>(float(int((i * 7 + std::size_t(phase * 3)) % 13) - 6) * .5F);
    ASSERT_TRUE(left.storage.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(right.storage.view().write(std::as_bytes(std::span(b))).ok());
    auto first_done = lanes[0].submit(queues[0], left.view.as_const(),
        right.view.as_const(), first.view);
    ASSERT_TRUE(first_done) << first_done.error().message();
    auto second_done = lanes[1].submit(queues[1], left.view.as_const(),
        right.view.as_const(), second.view);
    ASSERT_TRUE(second_done) << second_done.error().message();
    ASSERT_TRUE(first_done->wait().ok());
    ASSERT_TRUE(second_done->wait().ok());
    for (Matrix *output : {&first, &second}) {
      std::vector<T> actual(output->storage.size() / sizeof(T));
      ASSERT_TRUE(output->storage.view().read(
          std::as_writable_bytes(std::span(actual))).ok());
      const auto batch_count = output_desc.element_count().value() / 10;
      for (std::size_t batch = 0; batch < batch_count; ++batch)
        for (std::size_t row = 0; row < 2; ++row)
          for (std::size_t col = 0; col < 5; ++col) {
            double expected = 0;
            for (std::size_t k = 0; k < 3; ++k)
              expected += double(to_float(a[batched_element(left_desc,
                  output_desc, batch, row, k)])) *
                  double(to_float(b[batched_element(right_desc,
                      output_desc, batch, k, col)]));
            // Quarter/half inputs and a three-term dot are exactly represented
            // in all tested types, so no broad GEMM tolerance masks a map bug.
            EXPECT_EQ(to_float(actual[batched_element(output_desc,
                output_desc, batch, row, col)]), float(expected))
                << phase << "," << batch << "," << row << "," << col;
          }
    }
  }
}

TEST(CublasProvider, BatchedBroadcastUsesBothOperandsAndRealSubalignment) {
  for (bool multiple_resets : {false, true}) {
    SCOPED_TRACE(multiple_resets);
    run_batched_broadcast<float>(compute::DataType::kFloat32, multiple_resets);
    run_batched_broadcast<__half>(compute::DataType::kFloat16, multiple_resets);
    run_batched_broadcast<hyle::simd::bfloat16_t>(
        compute::DataType::kBFloat16, multiple_resets);
  }
  RecordProperty("finite_batched_domains", "6");
  RecordProperty("successful_submissions_expected", "24");
}

TEST(CublasProvider, RejectsUnsupportedTypesAndHostOnlyResources) {
  auto owner = context();
  auto device =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto pinned =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kPinnedHost)
          .value();
  const auto desc = matrix(compute::DataType::kFloat32, 16, 16, false);
  const std::array inputs{desc, desc};
  const std::array outputs{desc};
  auto host = compute::prepare(owner, compute::Operation(compute::MatMul{}),
                               inputs, outputs, pinned.ref());
  EXPECT_FALSE(host.has_value());
  const auto integer =
      compute::TensorDesc::dense(compute::DataType::kInt32,
                                 std::array<std::int64_t, 2>{16, 16})
          .value();
  auto unsupported = compute::prepare(
      owner, compute::Operation(compute::MatMul{}),
      std::array{integer, integer}, std::array{integer}, device.ref());
  EXPECT_FALSE(unsupported.has_value());
}

TEST(CublasProvider, CompletionRetainsReleasedPreparedAndInvocation) {
  auto owner = context();
  auto resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = matrix(compute::DataType::kFloat32, 32, 32, false);
  Matrix left(desc, resource.ref()), right(desc, resource.ref()),
      output(desc, resource.ref());
  const auto a = values<float>(desc, 0);
  const auto b = values<float>(desc, 3);
  ASSERT_TRUE(left.storage.view().write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(right.storage.view().write(std::as_bytes(std::span(b))).ok());
  auto prepared =
      compute::prepare(owner, compute::Operation(compute::MatMul{}),
                       std::array{left.view.as_const(), right.view.as_const()},
                       std::array{output.view})
          .value();
  const auto needed = prepared.workspace_requirement();
  auto workspace =
      mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
          .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  // Finish first-use SDK work before holding the stream. The second submit
  // must remain pending while the public artifact/lane handles are released.
  auto warm = invocation
                  .submit(queue, left.view.as_const(), right.view.as_const(),
                          output.view)
                  .value();
  ASSERT_TRUE(warm.wait().ok());
  struct Gate final {
    std::atomic<bool> released{false};
    static void CUDART_CB wait(void *raw) {
      auto &gate = *static_cast<Gate *>(raw);
      gate.released.wait(false, std::memory_order_acquire);
    }
    void release() {
      released.store(true, std::memory_order_release);
      released.notify_all();
    }
  } gate;
  struct Release final {
    Gate &gate;
    compute::Context &owner;
    ~Release() {
      gate.release();
      owner.drain().IgnoreError();
    }
  } release{gate, owner};
  const auto stream = cuda::stream(queue).value();
  ASSERT_EQ(cudaLaunchHostFunc(stream, &Gate::wait, &gate), cudaSuccess);
  auto done = invocation.submit(queue, left.view.as_const(),
                                right.view.as_const(), output.view);
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(done->poll().value(), compute::CompletionPoll::kPending);
  invocation = {};
  prepared = {};
  ASSERT_TRUE(owner.close().ok());
  gate.release();
  ASSERT_TRUE(done->wait().ok());
  check_product(left, a, right, b, output);
  EXPECT_FALSE(owner.create_queue().has_value());
}

struct MetadataAllocator final {
  mem::AllocatorRef backing = mem::system_allocator();
  std::size_t attempts = 0, fail_at = SIZE_MAX, live = 0;
  mem::AllocatorRef ref() noexcept {
    return {[](void *opaque, std::size_t size,
               std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (self.attempts++ == self.fail_at)
                return nullptr;
              void *result = self.backing.allocate(size, alignment);
              if (result != nullptr)
                ++self.live;
              return result;
            },
            [](void *opaque, void *pointer, std::size_t size,
               std::size_t alignment) noexcept {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (pointer != nullptr)
                --self.live;
              self.backing.deallocate(pointer, size, alignment);
            },
            [](void *opaque, void *pointer, std::size_t old_size,
               std::size_t size, std::size_t alignment) noexcept -> void * {
              auto &self = *static_cast<MetadataAllocator *>(opaque);
              if (size != 0 && self.attempts++ == self.fail_at)
                return nullptr;
              void *result =
                  self.backing.reallocate(pointer, old_size, size, alignment);
              if (pointer == nullptr && result != nullptr)
                ++self.live;
              else if (pointer != nullptr && size == 0)
                --self.live;
              return result;
            },
            this};
  }
};

TEST(CublasProvider,
     ColdMetadataFailuresRollBackAndReservedReplayDoesNotAllocate) {
  MetadataAllocator allocator;
  const std::array providers{
      hyle::compute::ProviderBinding{compute::cublas::provider(), 100}};
  const auto device = cuda::device(0).value();
  const auto create = [&] {
    return compute::Context::create(
        device,
        {.metadata_allocator = allocator.ref(), .providers = providers});
  };
  const auto context_begin = allocator.attempts;
  {
    auto owner = create();
    ASSERT_TRUE(owner.has_value());
  }
  const auto context_allocations = allocator.attempts - context_begin;
  ASSERT_GT(context_allocations, 0u);
  ASSERT_LT(context_allocations, 32u);
  EXPECT_EQ(allocator.live, 0u);
  for (std::size_t offset = 0; offset < context_allocations; ++offset) {
    SCOPED_TRACE(offset);
    allocator.fail_at = allocator.attempts + offset;
    auto owner = create();
    ASSERT_FALSE(owner.has_value());
    EXPECT_TRUE(owner.error().IsResourceExhausted());
    EXPECT_EQ(allocator.live, 0u);
  }
  allocator.fail_at = SIZE_MAX;
  {
    auto owner = create().value();
    auto resource =
        cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
    Matrix left(matrix(compute::DataType::kFloat32, 7, 128, false),
                resource.ref());
    Matrix right(matrix(compute::DataType::kFloat32, 128, 64, false),
                 resource.ref());
    Matrix output(matrix(compute::DataType::kFloat32, 7, 64, false),
                  resource.ref());
    const auto prepare = [&] {
      return compute::prepare(
          owner, compute::Operation(compute::MatMul{}),
          std::array{left.view.as_const(), right.view.as_const()},
          std::array{output.view});
    };
    const auto prepare_begin = allocator.attempts;
    {
      auto prepared = prepare();
      ASSERT_TRUE(prepared.has_value());
    }
    const auto prepare_allocations = allocator.attempts - prepare_begin;
    const auto prepare_live = allocator.live;
    ASSERT_GT(prepare_allocations, 0u);
    ASSERT_LT(prepare_allocations, 32u);
    for (std::size_t offset = 0; offset < prepare_allocations; ++offset) {
      SCOPED_TRACE(offset);
      allocator.fail_at = allocator.attempts + offset;
      auto prepared = prepare();
      ASSERT_FALSE(prepared.has_value());
      EXPECT_TRUE(prepared.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, prepare_live);
    }
    allocator.fail_at = SIZE_MAX;
    auto prepared = prepare().value();
    const auto needed = prepared.workspace_requirement();
    auto workspace =
        mem::Buffer::create(resource.ref(), needed.size, needed.alignment)
            .value();
    const auto lane_begin = allocator.attempts;
    {
      auto lane = prepared.create_invocation(workspace.view());
      ASSERT_TRUE(lane.has_value());
    }
    const auto lane_allocations = allocator.attempts - lane_begin;
    const auto lane_live = allocator.live;
    ASSERT_GT(lane_allocations, 0u);
    ASSERT_LT(lane_allocations, 32u);
    for (std::size_t offset = 0; offset < lane_allocations; ++offset) {
      SCOPED_TRACE(offset);
      allocator.fail_at = allocator.attempts + offset;
      auto lane = prepared.create_invocation(workspace.view());
      ASSERT_FALSE(lane.has_value());
      EXPECT_TRUE(lane.error().IsResourceExhausted());
      EXPECT_EQ(allocator.live, lane_live);
    }
    allocator.fail_at = SIZE_MAX;
    auto lane = prepared.create_invocation(workspace.view()).value();
    auto queue = owner.create_queue().value();
    ASSERT_TRUE(queue.reserve_completions(1).ok());
    const auto a = values<float>(left.description, 0);
    const auto b = values<float>(right.description, 3);
    ASSERT_TRUE(left.storage.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(right.storage.view().write(std::as_bytes(std::span(b))).ok());
    const auto replay_begin = allocator.attempts;
    for (int iteration = 0; iteration < 3; ++iteration) {
      auto done = lane.submit(queue, left.view.as_const(),
                              right.view.as_const(), output.view);
      ASSERT_TRUE(done.has_value());
      ASSERT_TRUE(done->wait().ok());
    }
    EXPECT_EQ(allocator.attempts, replay_begin);
    check_product(left, a, right, b, output);
  }
  EXPECT_EQ(allocator.live, 0u);
}

TEST(CublasProvider, LeasedDirectRetainsNativeUsesAndReplaysWithoutLeaseAllocation) {
  auto owner = context();
  auto resource = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(1).ok());
  LeaseMetadataCounter metadata;
  auto pool = compute::LeasePool::create(resource.ref(), 1024,
      {.max_blocks = 3, .allocator = metadata.ref()}).value();
  const auto desc = matrix(compute::DataType::kFloat32, 2, 2, false);
  auto left_write = pool.allocate(16).value();
  const auto primary = cuda::primary_context(owner).value();
  ASSERT_EQ(cuCtxPushCurrent(primary), CUDA_SUCCESS);
  struct PopPrimary final {
    CUcontext expected;
    ~PopPrimary() {
      CUcontext popped = nullptr;
      EXPECT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
      EXPECT_EQ(popped, expected);
    }
  } pop{primary};
  CUdeviceptr native_weight = 0;
  ASSERT_EQ(cuMemAlloc(&native_weight, 16), CUDA_SUCCESS);
  auto right_backing = cuda::ExternalBuffer::adopt(owner,
      reinterpret_cast<void *>(native_weight), 16).value();
  auto output = pool.allocate(16).value();
  const std::array<float, 4> a{1, 2, 3, 4}, identity{1, 0, 0, 1};
  ASSERT_TRUE(left_write.view()->write(std::as_bytes(std::span(a))).ok());
  ASSERT_TRUE(right_backing.view().write(std::as_bytes(std::span(identity))).ok());
  auto left = left_write.publish().value();
  EXPECT_FALSE(std::move(right_backing).into_read_lease({.allocator = {}}).has_value());
  EXPECT_TRUE(right_backing.valid());
  auto right = std::move(right_backing).into_read_lease({.allocator = metadata.ref()}).value();
  EXPECT_FALSE(right_backing.valid());
  EXPECT_FALSE(right.try_write().has_value());
  auto prepared = compute::prepare(owner, compute::Operation(compute::MatMul{}),
      std::array{left.tensor(desc).value(), right.tensor(desc).value()},
      std::array{output.tensor(desc).value()}).value();
  compute::LeasePool workspace_pool;
  compute::WriteLease workspace;
  const auto needed = prepared.workspace_requirement();
  if (needed.size) {
    const auto alignment = needed.alignment > 256 ? needed.alignment : 256;
    workspace_pool = compute::LeasePool::create(resource.ref(), needed.size,
        {.max_blocks = 1, .backing_alignment = alignment, .allocator = metadata.ref()}).value();
    workspace = workspace_pool.allocate(needed.size, alignment).value();
  }
  auto invocation = compute::LeasedOperationInvocation::create(prepared, std::move(workspace),
      {.allocator = metadata.ref()}).value();
  EXPECT_EQ(invocation.provider().id, compute::cublas::kProviderId);
  std::array inputs{compute::LeasedInput(desc, left), compute::LeasedInput(desc, right)};
  const std::array outputs{compute::LeasedOutput(desc, output)};

  // Rejection does not admit/poison a writer and the same lane remains usable.
  EXPECT_FALSE(invocation.submit(queue, {}, outputs).has_value());
  const auto wrong_desc = matrix(compute::DataType::kFloat32, 1, 4, false);
  EXPECT_FALSE(invocation.submit(queue, inputs,
      std::array{compute::LeasedOutput(wrong_desc, output)}).has_value());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  ASSERT_TRUE(output.view().has_value());
  auto warm = invocation.submit(queue, inputs, outputs).value();
  ASSERT_TRUE(warm.wait().ok()); // Mature SDK first-use work is outside replay.
  const auto cold_allocations = metadata.allocations.load();
  for (int iteration = 0; iteration < 8; ++iteration) {
    auto completion = invocation.submit(queue, inputs, outputs).value();
    ASSERT_TRUE(completion.wait().ok());
    auto ready = output.publish().value();
    auto old_result = ready;
    EXPECT_FALSE(ready.try_write().has_value());
    std::array<float, 4> observed{};
    ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(observed))).ok());
    EXPECT_EQ(observed, a);
    old_result.reset();
    output = ready.try_write().value();
  }
  EXPECT_EQ(metadata.allocations.load(), cold_allocations);

  {
    compute::LeasePool held_workspace_pool;
    compute::WriteLease held_workspace;
    if (needed.size) {
      const auto alignment = (std::max)(needed.alignment, std::size_t{256});
      held_workspace_pool = compute::LeasePool::create(resource.ref(), needed.size,
          {.max_blocks = 1, .backing_alignment = alignment, .allocator = metadata.ref()}).value();
      held_workspace = held_workspace_pool.allocate(needed.size, alignment).value();
    }
    auto held_run = compute::LeasedOperationInvocation::create(prepared,
        std::move(held_workspace), {.allocator = metadata.ref()}).value();
    auto held = held_run.submit(queue, inputs, outputs).value();
    ASSERT_TRUE(held_run.drain().ok());
    EXPECT_FALSE(held_run.in_flight());
    EXPECT_FALSE(held_run.submit(queue, inputs, outputs).has_value());
    ASSERT_TRUE(held.wait().ok());
    EXPECT_EQ(pool.stats().active_uses, 0u);
  }

  LeasedStreamGate gate;
  gate.stream = cuda::stream(queue).value();
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &LeasedStreamGate::wait, &gate), cudaSuccess);
  auto pending = invocation.submit(queue, inputs, outputs).value();
  EXPECT_EQ(pending.poll().value(), compute::CompletionPoll::kPending);
  EXPECT_TRUE(invocation.in_flight());
  EXPECT_FALSE(output.publish().has_value());
  EXPECT_FALSE(output.view().has_value());
  EXPECT_EQ(pending.request_cancel().code(), hyle::StatusCode::Unimplemented);
  EXPECT_EQ(pending.poll().value(), compute::CompletionPoll::kPending);
  EXPECT_EQ(pool.stats().active_uses, 2u);
  EXPECT_FALSE(output.publish().has_value());
  EXPECT_FALSE(invocation.submit(queue, inputs, outputs).has_value());
  // Neither public completion/lane nor business input handles own retirement.
  left.reset(); right.reset();
  for (auto &input : inputs) input.read.reset();
  invocation = {};
  pending = {};
  EXPECT_EQ(pool.stats().live_blocks, 2u);
  EXPECT_EQ(pool.stats().active_uses, 2u);
  EXPECT_FALSE(output.publish().has_value());
  gate.release();
  ASSERT_TRUE(pool.drain().ok());
  ASSERT_TRUE(workspace_pool.drain().ok());
  EXPECT_EQ(pool.stats().active_uses, 0u);
  EXPECT_EQ(pool.stats().live_blocks, 1u);
  auto ready = output.publish().value();
  std::array<float, 4> observed{};
  ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, a);
  ready.reset();
  EXPECT_EQ(pool.stats().live_blocks, 0u);
}

} // namespace
