#include "native_program.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include "metadata_arena.h"
#include "hyle/compute/model_ops.h"

// Private, locked resource seams. These symbols never form an installed C API.
extern "C" {
ggml_backend_buffer_t ggml_hyle_metal_import(ggml_backend_t, void *);
std::int64_t ggml_hyle_metal_checked_sync(ggml_backend_t);
bool ggml_hyle_metal_prepare_gdn(ggml_backend_t, const ggml_tensor *);
}

namespace hyle::compute::ggml_metal {
namespace {
struct BufferOwner {
  ggml_backend_buffer_t value = nullptr;
  BufferOwner() = default;
  BufferOwner(const BufferOwner &) = delete;
  BufferOwner &operator=(const BufferOwner &) = delete;
  ~BufferOwner() { ggml_backend_buffer_free(value); }
};

Status exception_status() noexcept {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ggml Metal resource allocation failed");
  } catch (const std::exception &e) {
    return Status::Internal(e.what());
  } catch (...) {
    return Status::Internal("ggml Metal native resource failure");
  }
}

Status native_status(std::int64_t code) noexcept {
  if (code == 0)
    return Status::Ok();
  // Keep the native NSError code in the existing Status detail channel.
  return Status::FromNative(StatusCode::Internal,
                            NativeError{NativeErrorDomain::Unknown, code},
                            "MTLCommandBufferErrorDomain: ggml Metal completion");
}

ggml_type native_type(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return GGML_TYPE_F32;
  case DataType::kFloat16:
    return GGML_TYPE_F16;
  case DataType::kInt32:
    return GGML_TYPE_I32;
  default:
    return GGML_TYPE_COUNT;
  }
}

bool mature_model_operation(Operation operation) noexcept {
  return operation.get_if<GatedDeltaNet>() ||
         operation.custom_as<model_ops::GroupedMatMul>() ||
         operation.custom_as<model_ops::HyperConnectionCombine>() ||
         operation.custom_as<model_ops::HyperConnectionPre>() ||
         operation.custom_as<model_ops::HyperConnectionPost>() ||
         operation.custom_as<model_ops::LightningIndexer>();
}
} // namespace

struct NativeProgram::State {
  ggml_runtime::MetadataArena arena;
  ggml_backend_t backend = nullptr;
  ggml_cgraph *graph = nullptr;
  std::array<ggml_tensor *, 7> tensors{};
  std::array<BufferOwner, 7> buffers;
  std::array<void *, 7> native_buffers{};
  std::array<std::size_t, 7> bytes{};
  std::vector<TensorDesc> outputs;
  std::vector<std::size_t> output_offsets;
  std::size_t port_count = 0;
  std::size_t output_port = 0;
  std::size_t scores = 0;
  bool failed = false;

  ~State() {
    if (backend != nullptr) {
      (void)ggml_hyle_metal_checked_sync(backend);
      // Imported buffers are released while the original device owner exists.
      for (auto &buffer : buffers)
        ggml_backend_buffer_free(std::exchange(buffer.value, nullptr));
      ggml_backend_free(backend);
    }
  }
};

NativeProgram::NativeProgram(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}
NativeProgram::~NativeProgram() = default;

Status NativeProgram::validate(Operation operation,
                               std::span<const TensorDesc> inputs) noexcept {
  if (!mature_model_operation(operation) || inputs.size() + 1 > 7)
    return Status::Unimplemented("ggml Metal model operation is not admitted");
  const auto output_count = operation_traits(operation).outputs;
  if (output_count > 2)
    return Status::Unimplemented(
        "ggml Metal model operation exceeds the native output limit");
  std::array<TensorDesc, 2> output_storage;
  const auto outputs = std::span(output_storage).first(output_count);
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, outputs));
  if (operation.get_if<GatedDeltaNet>()) {
    const auto q = inputs[0].dimensions();
    const auto v = inputs[2].dimensions();
    if (q[2] != v[2] || q[3] != v[3] ||
        (q[3] != 32 && q[3] != 64 && q[3] != 128) || q[1] > 32767)
      return Status::Unimplemented(
          "ggml Metal GatedDeltaNet requires equal heads, D32/64/128 and T<=32767");
  }
  for (const auto &input : inputs) {
    if (native_type(input.data_type()) == GGML_TYPE_COUNT || !input.is_dense() ||
        input.rank() > GGML_MAX_DIMS)
      return Status::Unimplemented(
          "ggml Metal model region requires dense F32/I32 inputs within rank four");
    HYLE_ASSIGN_OR_RETURN(auto bytes, input.required_bytes());
    if (bytes > std::size_t{INT32_MAX} * sizeof(float))
      return Status::OutOfRange("ggml Metal model input indexing exceeds I32");
  }
  std::size_t total = 0;
  for (const auto &output : outputs) {
    if (output.data_type() != DataType::kFloat32 || !output.is_dense())
      return Status::Unimplemented("ggml Metal model outputs must be dense F32");
    HYLE_ASSIGN_OR_RETURN(auto bytes, output.required_bytes());
    if (bytes > std::numeric_limits<std::size_t>::max() - total)
      return Status::OutOfRange("ggml Metal model output size overflows");
    total += bytes;
  }
  if (total / sizeof(float) > std::size_t{INT32_MAX})
    return Status::OutOfRange("ggml Metal model output exceeds I32");
  return Status::Ok();
}

Result<std::unique_ptr<NativeProgram>>
NativeProgram::create(Operation operation, std::span<const TensorDesc> inputs,
                      mem::AllocatorRef allocator) noexcept {
  HYLE_RETURN_IF_ERROR(validate(operation, inputs));
  try {
    auto state = std::make_unique<State>();
    HYLE_ASSIGN_OR_RETURN(state->arena,
                          ggml_runtime::MetadataArena::for_graph(12, 32, allocator));
    state->backend = ggml_backend_metal_init();
    if (state->backend == nullptr)
      return std::unexpected(Status::Unavailable("ggml Metal backend unavailable"));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto dims = inputs[i].dimensions();
      std::array<std::int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
      std::reverse_copy(dims.begin(), dims.end(), shape.begin());
      state->tensors[i] = ggml_new_tensor(
          state->arena.get(), native_type(inputs[i].data_type()),
          static_cast<int>((std::max)(std::size_t{1}, dims.size())),
          shape.data());
      if (state->tensors[i] == nullptr)
        return std::unexpected(Status::ResourceExhausted("ggml Metal tensor metadata exhausted"));
      HYLE_ASSIGN_OR_RETURN(state->bytes[i], inputs[i].required_bytes());
    }
    auto &t = state->tensors;
    ggml_tensor *result = nullptr;
    if (operation.get_if<GatedDeltaNet>()) {
      result = ggml_gated_delta_net(state->arena.get(), t[0], t[1], t[2],
                                    t[3], t[4], t[5], 1);
    } else if (operation.custom_as<model_ops::GroupedMatMul>()) {
      result = ggml_mul_mat_id(state->arena.get(), t[0], t[1], t[2]);
    } else if (const auto descriptor =
                   operation.custom_as<model_ops::HyperConnectionCombine>()) {
      result = ggml_dsv4_hc_comb(state->arena.get(), t[0], t[1], t[2],
                                 descriptor->epsilon, descriptor->iterations);
    } else if (operation.custom_as<model_ops::HyperConnectionPre>()) {
      result = ggml_dsv4_hc_pre(state->arena.get(), t[0], t[1]);
    } else if (operation.custom_as<model_ops::HyperConnectionPost>()) {
      result = ggml_dsv4_hc_post(state->arena.get(), t[0], t[1], t[2], t[3]);
    } else if (operation.custom_as<model_ops::LightningIndexer>()) {
      result = ggml_lightning_indexer(state->arena.get(), t[0], t[1], t[2],
                                      t[3]);
    }
    state->output_port = inputs.size();
    state->port_count = inputs.size() + 1;
    t[state->output_port] = result;
    if (result == nullptr)
      return std::unexpected(Status::ResourceExhausted("ggml Metal output metadata exhausted"));
    state->outputs.resize(operation_traits(operation).outputs);
    HYLE_RETURN_IF_ERROR(
        infer_output_descriptions(operation, inputs, state->outputs));
    std::size_t total = 0;
    for (const auto &output : state->outputs) {
      state->output_offsets.push_back(total);
      HYLE_ASSIGN_OR_RETURN(auto bytes, output.required_bytes());
      total += bytes;
    }
    state->scores = state->output_offsets.size() > 1
                        ? state->output_offsets[1]
                        : total;
    state->bytes[state->output_port] = total;
    if (!ggml_backend_supports_op(state->backend, result))
      return std::unexpected(Status::Unimplemented("ggml Metal model capability unavailable"));
    // Original specialization/cache, cold and shape-only; no user data or
    // mutable state is executed to manufacture a prepared replay claim.
    if (operation.get_if<GatedDeltaNet>() &&
        !ggml_hyle_metal_prepare_gdn(state->backend, result))
      return std::unexpected(Status::Unavailable("ggml Metal recurrent pipeline preparation failed"));
    state->graph = ggml_new_graph_custom(state->arena.get(), 32, false);
    if (state->graph == nullptr)
      return std::unexpected(Status::ResourceExhausted("ggml Metal graph metadata exhausted"));
    ggml_build_forward_expand(state->graph, result);
    return std::unique_ptr<NativeProgram>(new NativeProgram(std::move(state)));
  } catch (...) {
    return std::unexpected(exception_status());
  }
}

std::size_t NativeProgram::output_count() const noexcept {
  return state_->outputs.size();
}
const TensorDesc &
NativeProgram::output_description(std::size_t index) const noexcept {
  return state_->outputs[index];
}
std::size_t NativeProgram::output_offset(std::size_t index) const noexcept {
  return state_->output_offsets[index];
}
std::size_t NativeProgram::score_bytes() const noexcept { return state_->scores; }
std::size_t NativeProgram::output_bytes() const noexcept {
  return state_->bytes[state_->output_port];
}

Status NativeProgram::run(std::span<const BufferView> ports) noexcept {
  auto &s = *state_;
  if (s.failed)
    return Status::FailedPrecondition("failed ggml Metal program must be recreated");
  if (ports.size() != s.port_count)
    return Status::InvalidArgument("ggml Metal model buffer count mismatch");
  try {
    std::array<BufferOwner, 7> replacements;
    for (std::size_t i = 0; i < ports.size(); ++i) {
      if (ports[i].metal_buffer == nullptr || ports[i].offset % sizeof(float) != 0)
        return Status::InvalidArgument("ggml Metal requires an aligned native buffer");
      auto buffer = s.buffers[i].value;
      if (ports[i].metal_buffer != s.native_buffers[i]) {
        buffer = ggml_hyle_metal_import(s.backend, ports[i].metal_buffer);
        if (buffer == nullptr)
          return Status::ResourceExhausted("ggml Metal native buffer import failed");
        replacements[i].value = buffer;
      }
      const auto capacity = ggml_backend_buffer_get_size(buffer);
      if (ports[i].offset > capacity || s.bytes[i] > capacity - ports[i].offset)
        return Status::OutOfRange("ggml Metal tensor exceeds its native buffer");
    }
    for (std::size_t i = 0; i < ports.size(); ++i) {
      if (replacements[i].value != nullptr) {
        std::swap(replacements[i].value, s.buffers[i].value);
        s.native_buffers[i] = ports[i].metal_buffer;
      }
      s.tensors[i]->buffer = s.buffers[i].value;
      s.tensors[i]->data = static_cast<std::byte *>(
                              ggml_backend_buffer_get_base(s.buffers[i].value)) +
                          ports[i].offset;
    }
    const auto submitted = ggml_backend_graph_compute_async(s.backend, s.graph);
    const auto native_code = ggml_hyle_metal_checked_sync(s.backend);
    if (native_code > 0) {
      s.failed = true;
      return native_status(native_code);
    }
    if (submitted != GGML_STATUS_SUCCESS) {
      s.failed = true;
      return submitted == GGML_STATUS_ALLOC_FAILED
                 ? Status::ResourceExhausted("ggml Metal encoding allocation failed")
                 : Status::Internal("ggml Metal graph encoding failed");
    }
    if (native_code != 0) {
      s.failed = true;
      return native_status(native_code);
    }
    return Status::Ok();
  } catch (...) {
    s.failed = true;
    (void)ggml_hyle_metal_checked_sync(s.backend);
    return exception_status();
  }
}
} // namespace hyle::compute::ggml_metal
