#include "brgemm_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <new>

#include "oneapi/dnnl/dnnl_ukernel.h"

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#error "HYLE's oneDNN provider requires the locked BRGEMM ukernel API"
#endif

namespace hyle::compute::onednn::internal {
namespace {

static_assert(DNNL_VERSION_MAJOR == 3 && DNNL_VERSION_MINOR == 13 &&
                  DNNL_VERSION_PATCH == 2,
              "update the HYLE oneDNN provider identity with its source lock");

inline constexpr std::size_t kMinParallelTileColumns = 64;

bool try_align_scratchpad(std::size_t size, std::size_t *aligned) noexcept {
  constexpr std::size_t kPadding = kBrgemmWorkspaceAlignment - 1;
  if (aligned == nullptr ||
      size > (std::numeric_limits<std::size_t>::max)() - kPadding) {
    return false;
  }
  *aligned = (size + kPadding) & ~kPadding;
  return true;
}

bool is_cache_aligned(const void *pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) &
          (kBrgemmWorkspaceAlignment - 1)) == 0;
}

Status translate_status(dnnl_status_t status, const char *operation) noexcept {
  switch (status) {
  case dnnl_success:
    return Status::Ok();
  case dnnl_out_of_memory:
    return Status::ResourceExhausted(operation);
  case dnnl_invalid_arguments:
  case dnnl_invalid_graph:
  case dnnl_invalid_graph_op:
  case dnnl_invalid_shape:
  case dnnl_invalid_data_type:
    return Status::InvalidArgument(operation);
  case dnnl_unimplemented:
  case dnnl_last_impl_reached:
  case dnnl_not_required:
    return Status::Unimplemented(operation);
  case dnnl_runtime_error:
    return Status::Internal(operation);
  }
  return Status::Internal(operation);
}

void destroy_kernel(dnnl_brgemm_t &kernel) noexcept {
  if (kernel != nullptr) {
    dnnl_brgemm_destroy(kernel);
    kernel = nullptr;
  }
}

Status create_kernel(dnnl_brgemm_t *output, std::size_t rows, std::size_t inner,
                     std::size_t tile_columns, std::size_t leading_columns,
                     std::size_t *scratchpad_size) noexcept {
  if (output == nullptr || scratchpad_size == nullptr)
    return Status::InvalidArgument("oneDNN ukernel output is null");
  *output = nullptr;
  *scratchpad_size = 0;

  dnnl_brgemm_t kernel = nullptr;
  Status status = translate_status(
      dnnl_brgemm_create(&kernel, static_cast<dnnl_dim_t>(rows),
                         static_cast<dnnl_dim_t>(tile_columns),
                         static_cast<dnnl_dim_t>(inner), 1,
                         static_cast<dnnl_dim_t>(inner),
                         static_cast<dnnl_dim_t>(leading_columns),
                         static_cast<dnnl_dim_t>(leading_columns), dnnl_f32,
                         dnnl_f32, dnnl_f32),
      "oneDNN cannot create this BRGEMM ukernel");
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_set_add_C(kernel, 0),
                              "oneDNN rejected overwrite output semantics");
  }
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_finalize(kernel),
                              "oneDNN cannot finalize this BRGEMM ukernel");
  }
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_generate(kernel),
                              "oneDNN cannot generate this BRGEMM ukernel");
  }
  if (status.ok()) {
    status = translate_status(
        dnnl_brgemm_get_scratchpad_size(kernel, scratchpad_size),
        "oneDNN did not expose its BRGEMM scratchpad requirement");
  }
  if (!status.ok()) {
    destroy_kernel(kernel);
    return status;
  }
  *output = kernel;
  return Status::Ok();
}

} // namespace

struct BrgemmPlan final {
  dnnl_brgemm_t base_kernel = nullptr;
  dnnl_brgemm_t wide_kernel = nullptr;
  std::size_t rows = 0;
  std::size_t inner = 0;
  std::size_t columns = 0;
  std::size_t task_count = 1;
  std::size_t base_columns = 0;
  std::size_t wide_task_count = 0;
  std::size_t scratch_stride = 0;
  std::size_t scratchpad_size = 0;
};

namespace {

BrgemmTaskRange task_range(const BrgemmPlan &plan, std::size_t index) noexcept {
  const bool wide = index < plan.wide_task_count;
  const std::size_t first_column =
      wide ? index * (plan.base_columns + 1)
           : plan.wide_task_count * (plan.base_columns + 1) +
                 (index - plan.wide_task_count) * plan.base_columns;
  return BrgemmTaskRange{
      .first_column = first_column,
      .column_count = plan.base_columns + (wide ? 1 : 0),
  };
}

struct ExecutionEpisode final {
  const BrgemmPlan *plan = nullptr;
  const float *left = nullptr;
  const float *first_right = nullptr;
  const float *second_right = nullptr;
  float *first_output = nullptr;
  float *second_output = nullptr;
  std::byte *scratchpad = nullptr;
  BrgemmPairPostTaskFn post_task = nullptr;
  void *post_context = nullptr;
  std::atomic<int> first_error{static_cast<int>(dnnl_success)};
};

void record_error(ExecutionEpisode &episode, dnnl_status_t status) noexcept {
  if (status == dnnl_success)
    return;
  int expected = static_cast<int>(dnnl_success);
  episode.first_error.compare_exchange_strong(
      expected, static_cast<int>(status), std::memory_order_relaxed,
      std::memory_order_relaxed);
}

void execute_task(void *episode_value, std::size_t index,
                  std::size_t count) noexcept {
  auto &episode = *static_cast<ExecutionEpisode *>(episode_value);
  const BrgemmPlan &plan = *episode.plan;
  if (count != plan.task_count || index >= count)
    return;

  const BrgemmTaskRange range = task_range(plan, index);
  const dnnl_brgemm_t kernel =
      index < plan.wide_task_count ? plan.wide_kernel : plan.base_kernel;
  std::byte *task_scratchpad = episode.scratchpad + index * plan.scratch_stride;
  const std::array<dnnl_dim_t, 2> offsets{0, 0};

  dnnl_status_t status = dnnl_brgemm_set_hw_context(kernel);
  if (status == dnnl_success) {
    status = dnnl_brgemm_execute(
        kernel, episode.left, episode.first_right + range.first_column,
        offsets.data(), episode.first_output + range.first_column,
        task_scratchpad);
    if (status == dnnl_success && episode.second_right != nullptr) {
      status = dnnl_brgemm_execute(
          kernel, episode.left, episode.second_right + range.first_column,
          offsets.data(), episode.second_output + range.first_column,
          task_scratchpad);
    }
    const dnnl_status_t release = dnnl_brgemm_release_hw_context();
    if (status == dnnl_success)
      status = release;
  }
  record_error(episode, status);
  if (status == dnnl_success && episode.post_task != nullptr) {
    episode.post_task(episode.post_context, plan.rows, plan.columns, range);
  }
}

Status execute_impl(const BrgemmPlan *plan, ParallelExecutorRef executor,
                    const float *left, const float *first_right,
                    const float *second_right, float *first_output,
                    float *second_output, std::byte *scratchpad,
                    BrgemmPairPostTaskFn post_task,
                    void *post_context) noexcept {
  if (plan == nullptr || plan->base_kernel == nullptr || left == nullptr ||
      first_right == nullptr || first_output == nullptr ||
      scratchpad == nullptr ||
      ((second_right == nullptr) != (second_output == nullptr))) {
    return Status::Internal("oneDNN BRGEMM execution state is invalid");
  }
  ExecutionEpisode episode{
      .plan = plan,
      .left = left,
      .first_right = first_right,
      .second_right = second_right,
      .first_output = first_output,
      .second_output = second_output,
      .scratchpad = scratchpad,
      .post_task = post_task,
      .post_context = post_context,
  };
  executor.parallel_for(plan->task_count, &execute_task, &episode);
  return translate_status(static_cast<dnnl_status_t>(episode.first_error.load(
                              std::memory_order_relaxed)),
                          "oneDNN BRGEMM execution failed");
}

} // namespace

std::size_t recommended_brgemm_task_count(std::size_t columns,
                                          std::size_t concurrency) noexcept {
  const std::size_t max_tasks_by_width =
      (std::max)(std::size_t{1}, columns / kMinParallelTileColumns);
  return (std::min)((std::max)(std::size_t{1}, concurrency),
                    max_tasks_by_width);
}

Status create_brgemm_plan(BrgemmShape shape, std::size_t task_count,
                          mem::AllocatorRef allocator,
                          BrgemmPlan **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("oneDNN BRGEMM plan output is null");
  *output = nullptr;
  if (shape.rows == 0 || shape.inner == 0 || shape.columns == 0 ||
      task_count == 0 || task_count > shape.columns) {
    return Status::InvalidArgument("oneDNN BRGEMM plan shape is invalid");
  }
  constexpr auto kDimensionLimit =
      static_cast<std::uint64_t>((std::numeric_limits<dnnl_dim_t>::max)());
  if (shape.rows > kDimensionLimit || shape.inner > kDimensionLimit ||
      shape.columns > kDimensionLimit) {
    return Status::OutOfRange("oneDNN BRGEMM dimension overflows dnnl_dim_t");
  }

  const std::size_t base_columns = shape.columns / task_count;
  const std::size_t wide_task_count = shape.columns % task_count;
  std::size_t base_scratchpad = 0;
  dnnl_brgemm_t base_kernel = nullptr;
  Status status = create_kernel(&base_kernel, shape.rows, shape.inner,
                                base_columns, shape.columns, &base_scratchpad);
  dnnl_brgemm_t wide_kernel = nullptr;
  std::size_t wide_scratchpad = 0;
  if (status.ok() && wide_task_count != 0) {
    status = create_kernel(&wide_kernel, shape.rows, shape.inner,
                           base_columns + 1, shape.columns, &wide_scratchpad);
  }
  if (!status.ok()) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return status;
  }

  const std::size_t per_task_scratchpad =
      (std::max)(base_scratchpad, wide_scratchpad);
  std::size_t scratch_stride = 0;
  if (!try_align_scratchpad((std::max)(per_task_scratchpad, std::size_t{1}),
                            &scratch_stride) ||
      task_count > (std::numeric_limits<std::size_t>::max)() / scratch_stride) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return Status::OutOfRange("oneDNN MatMul workspace size overflows size_t");
  }

  void *storage = allocator.allocate(sizeof(BrgemmPlan), alignof(BrgemmPlan));
  if (storage == nullptr) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return Status::ResourceExhausted("oneDNN MatMul plan allocation failed");
  }
  *output = ::new (storage) BrgemmPlan{
      .base_kernel = base_kernel,
      .wide_kernel = wide_kernel,
      .rows = shape.rows,
      .inner = shape.inner,
      .columns = shape.columns,
      .task_count = task_count,
      .base_columns = base_columns,
      .wide_task_count = wide_task_count,
      .scratch_stride = scratch_stride,
      .scratchpad_size = task_count * scratch_stride,
  };
  return Status::Ok();
}

void destroy_brgemm_plan(BrgemmPlan *plan,
                         mem::AllocatorRef allocator) noexcept {
  if (plan == nullptr)
    return;
  destroy_kernel(plan->wide_kernel);
  destroy_kernel(plan->base_kernel);
  plan->~BrgemmPlan();
  allocator.deallocate(plan, sizeof(BrgemmPlan), alignof(BrgemmPlan));
}

std::size_t brgemm_task_count(const BrgemmPlan *plan) noexcept {
  return plan != nullptr ? plan->task_count : 0;
}

std::size_t brgemm_scratchpad_size(const BrgemmPlan *plan) noexcept {
  return plan != nullptr ? plan->scratchpad_size : 0;
}

Status validate_brgemm_scratchpad(const BrgemmPlan *plan,
                                  const void *scratchpad,
                                  std::size_t size) noexcept {
  if (plan == nullptr || plan->base_kernel == nullptr ||
      plan->scratchpad_size == 0) {
    return Status::Internal("oneDNN BRGEMM plan is invalid");
  }
  if (scratchpad == nullptr || size < plan->scratchpad_size ||
      !is_cache_aligned(scratchpad)) {
    return Status::InvalidArgument(
        "oneDNN workspace must satisfy its size and 64-byte alignment");
  }
  return Status::Ok();
}

Status execute_brgemm(const BrgemmPlan *plan, ParallelExecutorRef executor,
                      const float *left, const float *right, float *output,
                      std::byte *scratchpad) noexcept {
  return execute_impl(plan, executor, left, right, nullptr, output, nullptr,
                      scratchpad, nullptr, nullptr);
}

Status execute_brgemm_pair(const BrgemmPlan *plan, ParallelExecutorRef executor,
                           const float *left, const float *first_right,
                           const float *second_right, float *first_output,
                           float *second_output, std::byte *scratchpad,
                           BrgemmPairPostTaskFn post_task,
                           void *post_context) noexcept {
  if (second_right == nullptr || second_output == nullptr) {
    return Status::Internal("oneDNN BRGEMM pair state is invalid");
  }
  return execute_impl(plan, executor, left, first_right, second_right,
                      first_output, second_output, scratchpad, post_task,
                      post_context);
}

} // namespace hyle::compute::onednn::internal
