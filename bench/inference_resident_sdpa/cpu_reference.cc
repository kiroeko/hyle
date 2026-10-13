#include "cpu_reference.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

#include "cpu_execution.h"
#include "ggml.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/custom_kernel.h"
#include "hyle/mem/buffer.h"
#include "metadata_arena.h"

namespace hyle::bench::resident {
namespace c = compute;
namespace native = c::ggml_runtime;
namespace {
constexpr c::ProviderRuntimeProperties runtime{
    .replay_class = c::ProviderReplayClass::kPrepared, .blocks_host = true};

struct State {
  native::MetadataArena arena;
  ggml_cgraph *graph = nullptr;
  std::array<ggml_tensor *, 3> input{};
  ggml_tensor *output = nullptr;
  std::vector<ggml_tensor *> views;
  std::vector<mem::Buffer> payloads;
  std::vector<mem::HostMapping> payload_maps;
  native::CpuExecution execution;
};

struct Kernel {
  std::array<c::TensorDesc, 3> descriptions;
  c::ScaledDotProductAttention operation;
  std::vector<ggml_fp16_t> mask;
  struct Invocation {
    std::unique_ptr<State> state;
  };

  Result<Invocation>
  create_invocation(mem::BufferView,
                    mem::AllocatorRef allocator) const noexcept {
    try {
      auto state = std::make_unique<State>();
      HYLE_ASSIGN_OR_RETURN(
          state->arena, native::MetadataArena::for_graph(24, 32, allocator));
      auto *context = state->arena.get();
      for (std::size_t port = 0; port < 3; ++port) {
        const auto dimensions = descriptions[port].dimensions();
        auto *tensor =
            ggml_new_tensor_4d(context, GGML_TYPE_BF16, dimensions[3],
                               dimensions[2], dimensions[1], dimensions[0]);
        for (std::size_t axis = 0; axis < 4; ++axis)
          tensor->nb[axis] = static_cast<std::size_t>(
              descriptions[port].byte_strides()[3 - axis]);
        state->input[port] = tensor;
        ggml_set_input(tensor);
      }
      auto *query = ggml_cast(context, state->input[0], GGML_TYPE_F32);
      ggml_tensor *mask_tensor = nullptr;
      if (operation.causal) {
        mask_tensor = ggml_new_tensor_2d(context, GGML_TYPE_F16,
                                             descriptions[1].dimensions()[2],
                                             descriptions[0].dimensions()[2]);
        mask_tensor->data = const_cast<ggml_fp16_t *>(mask.data());
      }
      auto *attention =
          ggml_flash_attn_ext(context, query, state->input[1], state->input[2],
                              mask_tensor, operation.scale, 0.F, 0.F);
      if (!ggml_prec_set_acc(attention, GGML_PREC_F32))
        return std::unexpected(
            Status::Unimplemented("reference F32 accumulation"));
      // Original flash output is [D,H,Q,B]. The public consumer is BHSD.
      state->output = ggml_cast(
          context,
          ggml_cont(context, ggml_permute(context, attention, 0, 2, 1, 3)),
          GGML_TYPE_BF16);
      ggml_set_output(state->output);
      state->graph = ggml_new_graph_custom(context, 32, false);
      ggml_build_forward_expand(state->graph, state->output);
      for (auto *tensor = ggml_get_first_tensor(context); tensor;
           tensor = ggml_get_next_tensor(context, tensor)) {
        if (tensor->view_src) {
          state->views.push_back(tensor);
          continue;
        }
        if (tensor->data || tensor == state->output ||
            tensor == state->input[0] || tensor == state->input[1] ||
            tensor == state->input[2])
          continue;
        HYLE_ASSIGN_OR_RETURN(auto buffer,
                              mem::Buffer::create(mem::system_resource(),
                                                  ggml_nbytes(tensor),
                                                  GGML_MEM_ALIGN));
        HYLE_ASSIGN_OR_RETURN(
            auto mapping, buffer.view().map_host(mem::HostAccess::kReadWrite));
        HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
        tensor->data = bytes.data();
        state->payloads.push_back(std::move(buffer));
        state->payload_maps.push_back(std::move(mapping));
      }
      HYLE_ASSIGN_OR_RETURN(state->execution, native::CpuExecution::prepare(
                                                  state->graph, 1, allocator));
      std::size_t payload_bytes = 0;
      for (const auto &buffer : state->payloads)
        payload_bytes += buffer.size();
      std::fprintf(stderr,
                   "# CPU_REFERENCE_COLD graph_metadata=%zu payload=%zu "
                   "plan_workspace=%zu mask=%zu private_pool_upper=UNKNOWN\n",
                   state->arena.capacity(), payload_bytes,
                   state->execution.workspace_bytes(),
                   mask.size() * sizeof(ggml_fp16_t));
      return Invocation{std::move(state)};
    } catch (const std::bad_alloc &) {
      return std::unexpected(
          Status::ResourceExhausted("CPU reference cold storage"));
    }
  }

  c::KernelCommandResult
  enqueue(Invocation &invocation, c::Queue &,
          std::span<const c::ConstTensorView> inputs,
          std::span<const c::TensorView> outputs) const noexcept {
    auto &state = *invocation.state;
    std::array<mem::HostMapping, 3> mappings;
    for (std::size_t port = 0; port < 3; ++port) {
      auto mapped = inputs[port].storage().map_host();
      if (!mapped)
        return c::KernelCommandResult::rejected(std::move(mapped).error());
      mappings[port] = std::move(*mapped);
      state.input[port]->data =
          const_cast<std::byte *>(mappings[port].bytes().data());
    }
    auto output = outputs[0].storage().map_host(mem::HostAccess::kWrite);
    if (!output)
      return c::KernelCommandResult::rejected(std::move(output).error());
    auto bytes = output->writable_bytes();
    if (!bytes)
      return c::KernelCommandResult::rejected(std::move(bytes).error());
    state.output->data = bytes->data();
    for (auto *view : state.views)
      view->data =
          static_cast<std::byte *>(view->view_src->data) + view->view_offs;
    auto status = state.execution.run();
    const auto close = [&](Status next) {
      if (status.ok() && !next.ok())
        status = std::move(next);
    };
    close(output->close());
    for (auto &mapping : mappings)
      close(mapping.close());
    return c::KernelCommandResult::accepted_command(std::move(status));
  }
};

struct Factory {
  Result<c::KernelProposal>
  propose(const c::KernelRequest &request,
          std::span<c::ResourceRequirement> inputs,
          std::span<c::ResourceRequirement> outputs) const noexcept {
    const auto *operation =
        request.operation.get_if<c::ScaledDotProductAttention>();
    if (!operation || request.inputs.size() != 3 ||
        request.outputs.size() != 1 ||
        operation->layout != c::AttentionLayout::kBatchHeadSequenceDepth ||
        operation->mask != c::MaskKind::kNone ||
        operation->accumulation_type != c::DataType::kFloat32 ||
        !std::isfinite(operation->scale))
      return std::unexpected(
          Status::Unimplemented("finite BF16 BHSD reference"));
    for (const auto &description : request.inputs)
      if (!description.is_plain() || description.rank() != 4 ||
          description.data_type() != c::DataType::kBFloat16 ||
          description.dimensions()[0] != 1 ||
          (description.dimensions()[3] != 256 && description.dimensions()[3] != 64) ||
          description.byte_strides()[3] != 2)
        return std::unexpected(
            Status::Unimplemented("finite BF16 reference view"));
    const auto q = request.inputs[0].dimensions();
    const auto k = request.inputs[1].dimensions();
    const bool primary = (q[2] == 1 || q[2] == 128) && q[3] == 256 &&
                         k[2] == 1024 && operation->causal;
    const bool holdout = q[2] == 17 && q[3] == 64 && k[2] == 257 &&
                         !operation->causal;
    if (!request.inputs[0].is_dense() || (!primary && !holdout) ||
        q[1] != 8 || k[1] != 2 || q[3] != k[3] ||
        request.inputs[1] != request.inputs[2] ||
        request.outputs[0] != request.inputs[0])
      return std::unexpected(
          Status::Unimplemented("finite reference GQA point"));
    for (const auto &description : request.inputs)
      for (std::size_t axis = 0; axis < 4; ++axis) {
        const auto stride = description.byte_strides()[axis];
        if (stride <= 0 || stride % 2 != 0 ||
            (axis < 3 && stride / description.dimensions()[axis + 1] <
                             description.byte_strides()[axis + 1]))
          return std::unexpected(Status::Unimplemented(
              "finite reference positive row-major padding"));
      }
    const c::ResourceRequirement storage{
        .required_properties = mem::MemoryProperty::kHostAccessible |
                               mem::MemoryProperty::kHostDeviceCoherent,
        .minimum_alignment = GGML_MEM_ALIGN};
    for (auto &input : inputs)
      input = storage;
    outputs[0] = storage;
    return c::KernelProposal{.runtime = runtime,
                             .cost = {.dispatch_count = 1},
                             .implementation = c::ProviderImplementationId{1}};
  }
  c::KernelPreparation prepare(const c::KernelPrepareRequest &request,
                               mem::AllocatorRef allocator) const noexcept {
    try {
      const auto operation =
          *request.operation.get_if<c::ScaledDotProductAttention>();
      const auto rows = request.inputs[0].dimensions()[2];
      const auto keys = request.inputs[1].dimensions()[2];
      Kernel kernel{{request.inputs[0], request.inputs[1], request.inputs[2]},
                    operation,
                    {}};
      if (operation.causal)
        kernel.mask.resize(static_cast<std::size_t>(rows * keys));
      for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t key = 0; operation.causal && key < keys; ++key)
          kernel.mask[static_cast<std::size_t>(row * keys + key)] =
              ggml_fp32_to_fp16(key <= keys - rows + row ? 0.F : -INFINITY);
      auto result = c::make_custom_kernel(std::move(kernel),
                                          {.runtime = runtime}, allocator);
      if (!result)
        return c::KernelPreparation::error(std::move(result).error());
      return c::KernelPreparation::prepared(std::move(*result));
    } catch (const std::bad_alloc &) {
      return c::KernelPreparation::error(
          Status::ResourceExhausted("CPU reference mask"));
    }
  }
};
} // namespace
Result<c::Provider> cpu_reference_provider() noexcept {
  return c::make_custom_provider({{0x48594c4552454650ULL, 1},
                                  "original-ggml-resident-cpu",
                                  "ffa4e8b80930029a35991f94e7c8a93cd67730ab"},
                                 c::cpu::kBackendId, Factory{});
}
} // namespace hyle::bench::resident
