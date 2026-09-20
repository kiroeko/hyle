#include "native_program.h"

#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <utility>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include "metadata_arena.h"

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
} // namespace

struct NativeProgram::State {
  ggml_runtime::MetadataArena arena;
  ggml_backend_t backend = nullptr;
  ggml_cgraph *graph = nullptr;
  std::array<ggml_tensor *, 7> tensors{};
  std::array<BufferOwner, 7> buffers;
  std::array<void *, 7> native_buffers{};
  std::array<std::size_t, 7> bytes{};
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

Status NativeProgram::validate(std::span<const TensorDesc> inputs) noexcept {
  std::array<TensorDesc, 2> outputs;
  HYLE_RETURN_IF_ERROR(
      infer_output_descriptions(Operation(GatedDeltaNet{}), inputs, outputs));
  const auto q = inputs[0].dimensions();
  const auto v = inputs[2].dimensions();
  if (q[2] != v[2] || q[3] != v[3] ||
      (q[3] != 32 && q[3] != 64 && q[3] != 128) || q[1] > 32767)
    return Status::Unimplemented(
        "ggml Metal GatedDeltaNet requires equal heads, D32/64/128 and T<=32767");
  for (const auto &input : inputs) {
    if (input.data_type() != DataType::kFloat32 || !input.is_dense())
      return Status::Unimplemented(
          "ggml Metal GatedDeltaNet currently admits dense F32 inputs");
    HYLE_ASSIGN_OR_RETURN(auto bytes, input.required_bytes());
    if (bytes / sizeof(float) > std::size_t{INT32_MAX})
      return Status::OutOfRange("ggml Metal recurrent indexing exceeds I32");
  }
  HYLE_ASSIGN_OR_RETURN(auto scores, outputs[0].required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto state, outputs[1].required_bytes());
  if ((scores + state) / sizeof(float) > std::size_t{INT32_MAX})
    return Status::OutOfRange("ggml Metal recurrent output exceeds I32");
  return Status::Ok();
}

Result<std::unique_ptr<NativeProgram>>
NativeProgram::create(std::span<const TensorDesc> inputs,
                      mem::AllocatorRef allocator) noexcept {
  HYLE_RETURN_IF_ERROR(validate(inputs));
  try {
    auto state = std::make_unique<State>();
    HYLE_ASSIGN_OR_RETURN(state->arena,
                          ggml_runtime::MetadataArena::for_graph(7, 16, allocator));
    state->backend = ggml_backend_metal_init();
    if (state->backend == nullptr)
      return std::unexpected(Status::Unavailable("ggml Metal backend unavailable"));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto dims = inputs[i].dimensions();
      state->tensors[i] = ggml_new_tensor_4d(state->arena.get(), GGML_TYPE_F32,
                                            dims[3], dims[2], dims[1], dims[0]);
      if (state->tensors[i] == nullptr)
        return std::unexpected(Status::ResourceExhausted("ggml Metal tensor metadata exhausted"));
      HYLE_ASSIGN_OR_RETURN(state->bytes[i], inputs[i].required_bytes());
    }
    auto &t = state->tensors;
    t[6] = ggml_gated_delta_net(state->arena.get(), t[0], t[1], t[2], t[3],
                                t[4], t[5], 1);
    if (t[6] == nullptr)
      return std::unexpected(Status::ResourceExhausted("ggml Metal output metadata exhausted"));
    state->scores = state->bytes[2];
    state->bytes[6] = state->scores + state->bytes[5];
    if (!ggml_backend_supports_op(state->backend, t[6]))
      return std::unexpected(Status::Unimplemented("ggml Metal recurrent capability unavailable"));
    // Original specialization/cache, cold and shape-only; no user data or
    // mutable state is executed to manufacture a prepared replay claim.
    if (!ggml_hyle_metal_prepare_gdn(state->backend, t[6]))
      return std::unexpected(Status::Unavailable("ggml Metal recurrent pipeline preparation failed"));
    state->graph = ggml_new_graph_custom(state->arena.get(), 16, false);
    if (state->graph == nullptr)
      return std::unexpected(Status::ResourceExhausted("ggml Metal graph metadata exhausted"));
    ggml_build_forward_expand(state->graph, t[6]);
    return std::unique_ptr<NativeProgram>(new NativeProgram(std::move(state)));
  } catch (...) {
    return std::unexpected(exception_status());
  }
}

std::size_t NativeProgram::score_bytes() const noexcept { return state_->scores; }
std::size_t NativeProgram::output_bytes() const noexcept { return state_->bytes[6]; }

Status NativeProgram::run(std::span<const BufferView> ports) noexcept {
  auto &s = *state_;
  if (s.failed)
    return Status::FailedPrecondition("failed ggml Metal program must be recreated");
  if (ports.size() != s.tensors.size())
    return Status::InvalidArgument("ggml Metal recurrent buffer count mismatch");
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
