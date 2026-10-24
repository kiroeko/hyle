#include "../compute_models/ct_packed_graph_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace f = ct_packed_graph_test;
struct Backend final {
  static constexpr bool supports_capture = true;
  static constexpr auto provider_id = f::c::torch_cuda::kProviderId;
  static f::c::Context create_context() {
    return f::take(f::g::cuda::create_context(), "CUDA Context");
  }
  static f::c::StorageOwner create_owner(const f::c::Context &context) {
    return f::take(f::c::StorageOwner::adopt(context,
        f::take(f::c::cuda::MemoryResource::create(context,
            f::c::cuda::MemoryKind::kDevice), "CUDA resource")), "CUDA StorageOwner");
  }
  static hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph,
      const f::c::Context &context, const f::c::StorageOwner &owner,
      f::g::CompileOptions options = {}) noexcept {
    return compile_unrestricted(graph, context, owner, options);
  }
  static hyle::Result<f::g::CompiledGraph> compile_unrestricted(const f::g::Graph &graph,
      const f::c::Context &context, const f::c::StorageOwner &owner,
      f::g::CompileOptions options = {}) noexcept {
    return f::g::cuda::compile(graph, context, owner, options);
  }
  static hyle::Status prepare(f::g::Invocation &slot, f::c::Queue &queue,
      bool capture) noexcept {
    return f::g::cuda::prepare_replay(slot, queue,
        {.mode = capture ? f::g::ReplayMode::kRequireCapture : f::g::ReplayMode::kCommands,
         .host_wait = f::g::HostWaitPolicy::kAllow});
  }
};

TEST(CudaCtPackedLinear, PublicChainsKeepSourceRoundsAndTypedCastWitnesses) {
  f::PublicChainsKeepSourceRoundsAndTypedCastWitnesses<Backend>();
}

TEST(CudaCtPackedLinear, CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults) {
  f::CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults<Backend>();
}

TEST(CudaCtPackedLinear, PublishedViewAndControlEscapesExecuteTheOriginalDecomposition) {
  f::PublishedViewAndControlEscapesExecuteTheOriginalDecomposition<Backend>();
}

TEST(CudaCtPackedLinear, OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank) {
  f::OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank<Backend>();
}
} // namespace
