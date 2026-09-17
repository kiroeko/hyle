#include "reference.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "reference_spirv.h"
#include "vulkan_program_internal.h"

namespace hyle::bench::vulkan {
namespace d = compute::vulkan::detail;
namespace {
template <class T> T take(Result<T> result) {
  if (!result) {
    std::fprintf(stderr, "Vulkan reference: %.*s\n",
                 int(result.error().message().size()),
                 result.error().message().data());
    std::exit(1);
  }
  return std::move(*result);
}
void check(Status status) {
  if (!status.ok()) {
    std::fprintf(stderr, "Vulkan reference: %.*s\n",
                 int(status.message().size()), status.message().data());
    std::exit(1);
  }
}
compute::TensorDesc dense(int rows, int columns) {
  return take(compute::TensorDesc::dense(
      compute::DataType::kFloat32, std::array<std::int64_t, 2>{rows, columns}));
}
template <std::size_t N>
std::vector<std::byte> bytes(const std::array<std::uint32_t, N> &words) {
  std::vector<std::byte> result(sizeof(words));
  std::memcpy(result.data(), words.data(), sizeof(words));
  return result;
}
std::array<std::uint32_t, 2> divisor(std::uint32_t value) {
  const auto shift = std::uint32_t(std::bit_width(value - 1));
  const auto multiplier =
      ((std::uint64_t{1} << 32) * ((std::uint64_t{1} << shift) - value)) /
          value +
      1;
  return {std::uint32_t(multiplier), shift};
}

// Pinned ggml: load_shaders float tile recipe and get_dequantize_mul_mat_vec.
// Matrix values are native W[N,K], X[M,K], D[M,N]. No numerical conversion.
NativeStep matrix(const v::Context &context, std::uint32_t m, std::uint32_t k,
                  std::uint32_t n, std::uint32_t input, std::uint32_t weight,
                  std::uint32_t output) {
  const auto &facts = context.environment();
  NativeStep step;
  if (m <= 8) {
    const bool subgroup =
        facts.subgroup_arithmetic && facts.subgroup_size == 32;
    const bool large =
        subgroup && facts.vendor_id == 0x10de && n <= 8192 && k >= 1024;
    const std::span<const std::uint32_t> code =
        !subgroup ? std::span<const std::uint32_t>(spirv::vector_shared)
        : large   ? std::span<const std::uint32_t>(spirv::vector_hybrid)
                  : std::span<const std::uint32_t>(spirv::vector_subgroup);
    const std::array<v::SpecializationConstant, 3> spec{
        {{0, large ? 128u : 32u}, {1, 1}, {2, m}}};
    step.kernel = take(v::Kernel::create(context, {code, 5, 52, spec}));
    step.bindings = {weight, input, output, input, input};
    step.push = bytes(std::array<std::uint32_t, 13>{k, k, k, n, k * n, k, n, 0,
                                                    0, 1, 1, 1, 1});
    step.groups = {n, 1, 1};
  } else {
    std::uint32_t tile = std::min(m, n) <= 32   ? 32
                         : std::min(m, n) <= 64 ? 64
                                                : 128;
    // The DATA_A_F32 preprocessor branch fixes BK32, ignoring host ID3=16.
    while (tile > 32 && 2 * tile * 17 * 8 > facts.shared_memory_bytes)
      tile /= 2;
    // Exact-range handoff forbids upstream padding assumptions. This comparator
    // uses the same safe domain; the M16 workload has a partial tile.
    const auto &limits = d::Access::runtime(context)->device->properties.limits;
    const bool aligned = m % tile == 0 && n % tile == 0 && k % tile == 0 &&
                         limits.minStorageBufferOffsetAlignment >= 16;
    const std::array<std::uint32_t, 12> values{tile == 32 ? 32u : 128u,
                                               tile,
                                               tile,
                                               16,
                                               tile == 128 ? 64u : 32u,
                                               tile == 128 ? 64u : 32u,
                                               2,
                                               tile == 32 ? 2u : 4u,
                                               tile == 128 ? 4u : 2u,
                                               1,
                                               32,
                                               aligned ? 1u : 0u};
    std::array<v::SpecializationConstant, 12> spec{};
    for (std::uint32_t i = 0; i < spec.size(); ++i)
      spec[i] = {i, values[i]};
    step.kernel =
        take(v::Kernel::create(context, {spirv::matrix, 3, 64, spec}));
    step.bindings = {weight, input, output};
    step.push = bytes(std::array<std::uint32_t, 16>{
        n, m, k, k, k, n, k * n, k * m, m * n, 0, 1, k, 1, 1, 1, 1});
    step.groups = {(n + tile - 1) / tile, (m + tile - 1) / tile, 1};
  }
  return step;
}

NativeStep gate(const v::Context &context, std::uint32_t rows,
                std::uint32_t columns) {
  NativeStep step;
  step.kernel = take(v::Kernel::create(context, {spirv::gate, 3, 108, {}}));
  step.bindings = {4, 5, 6};
  const auto count = rows * columns;
  std::array<std::uint32_t, 27> push{};
  push[0] = count;
  push[1] = columns;
  push[2] = columns;
  push[3] = 2;
  for (std::uint32_t base : {6u, 10u, 14u}) {
    push[base] = 1;
    push[base + 1] = columns;
    push[base + 2] = count;
    push[base + 3] = count;
  }
  push[18] = rows;
  push[19] = 1;
  const auto outer = divisor(count), inner = divisor(columns);
  push[21] = outer[0];
  push[22] = outer[1];
  push[23] = outer[0];
  push[24] = outer[1];
  push[25] = inner[0];
  push[26] = inner[1];
  step.push = bytes(push);
  step.groups = {1, std::min((count + 511) / 512, 512u),
                 (count + 262143) / 262144};
  return step;
}
} // namespace

Reference::Reference(const v::Context &context, int rows, int inner, int hidden,
                     int output,
                     const std::array<std::vector<float>, 3> &host) {
  descriptions_[0] = dense(rows, inner);
  for (int i = 0; i < 3; ++i) {
    const int k = i == 2 ? hidden : inner, n = i == 2 ? output : hidden;
    descriptions_[i + 1] = take(compute::TensorDesc::strided_bytes(
        compute::DataType::kFloat32, std::array<std::int64_t, 2>{k, n},
        std::array<std::int64_t, 2>{4, std::int64_t(k) * 4}));
    std::vector<float> packed(host[i].size());
    for (int col = 0; col < n; ++col)
      for (int row = 0; row < k; ++row)
        packed[std::size_t(col) * k + row] =
            host[i][std::size_t(row) * n + col];
    weights_[i] = take(
        context.upload(descriptions_[i + 1], std::as_bytes(std::span(packed))));
  }
  descriptions_[4] = descriptions_[5] = descriptions_[6] = dense(rows, hidden);
  descriptions_[7] = dense(rows, output);
  steps_.push_back(matrix(context, rows, inner, hidden, 0, 1, 4));
  steps_.push_back(matrix(context, rows, inner, hidden, 0, 2, 5));
  steps_.push_back(gate(context, rows, hidden));
  steps_.push_back(matrix(context, rows, hidden, output, 6, 3, 7));
}

v::Program Reference::owning_program(const v::Context &context) const {
  // Only the lifecycle/resource adapter is shared with HYLE. This independently
  // authored native recipe bypasses typed preparation and Graph compilation.
  auto state = std::make_shared<d::ProgramState>();
  state->runtime = d::Access::runtime(context);
  for (std::size_t i = 0; i < descriptions_.size(); ++i)
    state->values.push_back(
        {descriptions_[i], i >= 1 && i <= 3 ? weights_[i - 1] : v::Tensor{}});
  state->inputs = {0};
  state->outputs = {7};
  state->input_descriptions = {descriptions_[0]};
  state->output_descriptions = {descriptions_[7]};
  for (const auto &step : steps_)
    state->dispatches.push_back({d::Access::pipeline(step.kernel),
                                 step.bindings, step.push, step.groups});
  check(d::plan_storage(*state));
  return d::Access::program(std::move(state));
}

std::array<v::Tensor, 8> Reference::values(const v::Context &context,
                                           const v::Tensor &input) const {
  std::array<v::Tensor, 8> result;
  result[0] = input;
  for (std::size_t i = 1; i < 4; ++i)
    result[i] = weights_[i - 1];
  for (std::size_t i = 4; i < 8; ++i)
    result[i] = take(context.allocate(descriptions_[i]));
  return result;
}

v::RecordedProgram
Reference::record(const v::Context &context,
                  const std::array<v::Tensor, 8> &values) const {
  auto recorder = take(v::Recorder::create(context));
  for (const auto &step : steps_) {
    std::vector<v::Tensor> bindings;
    for (const auto value : step.bindings)
      bindings.push_back(values[value]);
    check(recorder.dispatch(step.kernel, bindings, step.groups, step.push));
  }
  return take(recorder.finish());
}

struct RawProgram::State {
  const v::Context &context;
  std::shared_ptr<d::Run> run;
  std::function<Status(VkQueue)> submit;
  explicit State(const v::Context &c) : context(c) {}
};
RawProgram::RawProgram(const v::Context &context, const Reference &reference,
                       const std::array<v::Tensor, 8> &values)
    : state_(std::make_unique<State>(context)) {
  state_->run = take(d::create_run(d::Access::runtime(context)));
  auto recipe = std::make_shared<Reference>(reference);
  state_->run->recording_owner = recipe;
  std::vector<d::PlanValue> descriptions;
  for (const auto &value : values) {
    descriptions.push_back({value.description(), value});
    state_->run->retained_buffers.push_back(d::Access::buffer(value));
  }
  std::vector<d::Dispatch> dispatches;
  for (const auto &step : reference.steps())
    dispatches.push_back({d::Access::pipeline(step.kernel), step.bindings,
                          step.push, step.groups});
  check(d::record(state_->run, descriptions, dispatches));
  state_->submit = [state = state_.get()](VkQueue queue) -> Status {
    const auto &run = *state->run;
    const auto &device = *run.runtime->device;
    HYLE_RETURN_IF_ERROR(d::native_status(
        device.fn.ResetFences(device.native.device, 1, &run.fence),
        "raw reset"));
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &run.command;
    HYLE_RETURN_IF_ERROR(d::native_status(
        device.fn.QueueSubmit(queue, 1, &info, run.fence), "raw submit"));
    auto result =
        device.fn.WaitForFences(device.native.device, 1, &run.fence, VK_TRUE,
                                (std::numeric_limits<std::uint64_t>::max)());
    if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST)
      check(d::native_status(device.fn.QueueWaitIdle(queue),
                             "raw exceptional drain"));
    return d::native_status(result, "raw wait");
  };
}
RawProgram::~RawProgram() = default;
void RawProgram::run() { check(state_->context.with_queue(state_->submit)); }
} // namespace hyle::bench::vulkan
