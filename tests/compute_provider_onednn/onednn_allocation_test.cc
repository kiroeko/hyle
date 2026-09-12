#include "hyle/compute/providers/onednn.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include <gtest/gtest.h>

#include "hyle/compute/device.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

#if !defined(HYLE_STATIC)
#error "the oneDNN allocation probe requires static product ownership"
#endif

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> count{0};

void note() noexcept {
  if (enabled.load(std::memory_order_relaxed))
    count.fetch_add(1, std::memory_order_relaxed);
}

void *allocate(std::size_t size) {
  note();
  if (void *result = std::malloc(size == 0 ? 1 : size))
    return result;
  throw std::bad_alloc();
}

void *allocate_aligned(std::size_t size, std::size_t alignment) {
  note();
#if defined(_WIN32)
  if (void *result = _aligned_malloc(size == 0 ? 1 : size, alignment))
    return result;
#else
  void *result = nullptr;
  if (posix_memalign(&result, alignment, size == 0 ? 1 : size) == 0)
    return result;
#endif
  throw std::bad_alloc();
}

void deallocate_aligned(void *pointer) noexcept {
#if defined(_WIN32)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}
} // namespace allocation_probe

void *operator new(std::size_t size) {
  return allocation_probe::allocate(size);
}
void *operator new[](std::size_t size) {
  return allocation_probe::allocate(size);
}
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return allocation_probe::allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, const std::nothrow_t &) noexcept {
  return ::operator new(size, std::nothrow);
}
void *operator new(std::size_t size, std::align_val_t alignment) {
  return allocation_probe::allocate_aligned(
      size, static_cast<std::size_t>(alignment));
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void *operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return allocation_probe::allocate_aligned(
        size, static_cast<std::size_t>(alignment));
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  return ::operator new(size, alignment, std::nothrow);
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void *pointer, const std::nothrow_t &) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, const std::nothrow_t &) noexcept {
  std::free(pointer);
}
void operator delete(void *pointer, std::align_val_t) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}
void operator delete[](void *pointer, std::align_val_t) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}
void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}
void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}
void operator delete(void *pointer, std::align_val_t,
                     const std::nothrow_t &) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}
void operator delete[](void *pointer, std::align_val_t,
                       const std::nothrow_t &) noexcept {
  allocation_probe::deallocate_aligned(pointer);
}

namespace {

struct Shape final {
  std::size_t rows;
  std::size_t inner;
  std::size_t columns;
};

void serial_parallel_for(void *, std::size_t count,
                         hyle::compute::ParallelTaskFn task,
                         void *task_context) noexcept {
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

hyle::compute::TensorDesc matrix(std::size_t rows, std::size_t columns) {
  const std::array dimensions{static_cast<std::int64_t>(rows),
                              static_cast<std::int64_t>(columns)};
  return hyle::compute::TensorDesc::dense(hyle::compute::DataType::kFloat32,
                                          dimensions)
      .value();
}

TEST(OneDnnAllocation, WarmSubmitHasNoGlobalNewAcrossInferenceShapes) {
  constexpr std::array<Shape, 6> kShapes{{
      {1, 4096, 4096},
      {1, 5120, 13824},
      {4, 4096, 4096},
      {4, 5120, 13824},
      {32, 4096, 4096},
      {32, 5120, 13824},
  }};

  const hyle::compute::ParallelExecutorRef executor(nullptr, 4,
                                                    &serial_parallel_for);
  auto context = hyle::compute::Context::create(
                     hyle::compute::Device::cpu(),
                     hyle::compute::onednn::provider({.executor = executor}))
                     .value();
  auto queue = context.create_queue().value();

  for (const Shape shape : kShapes) {
    const auto left_description = matrix(shape.rows, shape.inner);
    const auto right_description = matrix(shape.inner, shape.columns);
    const auto output_description = matrix(shape.rows, shape.columns);
    auto prepared = hyle::compute::prepare(context, hyle::compute::MatMul{},
                                           left_description, right_description,
                                           output_description)
                        .value();
    ASSERT_EQ(prepared.provider(), hyle::compute::ProviderKind::kOneDnn);

    auto left =
        hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                  shape.rows * shape.inner * sizeof(float), 64)
            .value();
    auto right = hyle::mem::Buffer::create(
                     hyle::mem::system_resource(),
                     shape.inner * shape.columns * sizeof(float), 64)
                     .value();
    auto output = hyle::mem::Buffer::create(
                      hyle::mem::system_resource(),
                      shape.rows * shape.columns * sizeof(float), 64)
                      .value();
    ASSERT_TRUE(left.view().fill(std::byte{0}).ok());
    ASSERT_TRUE(right.view().fill(std::byte{0}).ok());

    const auto requirement = prepared.workspace_requirement();
    auto workspace =
        hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                  requirement.size, requirement.alignment)
            .value();
    auto invocation = prepared.create_invocation(workspace.view()).value();
    auto left_view =
        hyle::compute::ConstTensorView::create(left_description, left.view())
            .value();
    auto right_view =
        hyle::compute::ConstTensorView::create(right_description, right.view())
            .value();
    auto output_view =
        hyle::compute::TensorView::create(output_description, output.view())
            .value();

    auto warm = invocation.submit(queue, left_view, right_view, output_view);
    ASSERT_TRUE(warm.has_value());
    ASSERT_TRUE(warm->wait().ok());

    allocation_probe::count.store(0, std::memory_order_relaxed);
    allocation_probe::enabled.store(true, std::memory_order_release);
    auto measured =
        invocation.submit(queue, left_view, right_view, output_view);
    const bool submitted = measured.has_value();
    const bool completed = submitted && measured->wait().ok();
    allocation_probe::enabled.store(false, std::memory_order_release);

    ASSERT_TRUE(submitted);
    ASSERT_TRUE(completed);
    EXPECT_EQ(allocation_probe::count.load(std::memory_order_relaxed), 0U)
        << "shape=" << shape.rows << 'x' << shape.inner << 'x' << shape.columns;
  }
}

} // namespace
