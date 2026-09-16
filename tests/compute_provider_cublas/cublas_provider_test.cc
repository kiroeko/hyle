#include "hyle/compute/providers/cublas.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace {

namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;

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
template <class T> T from_float(float value) { return static_cast<T>(value); }
template <> __half from_float(float value) { return __float2half(value); }

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
      const double tolerance = sizeof(T) == 2
                                   ? 0.0007 * std::abs(expected) + 0.00004
                                   : 0.000001 * std::abs(expected) + 0.000001;
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

} // namespace
