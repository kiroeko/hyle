#include "../compute_models/ct_packed_graph_fixture.h"
#include "../compute_models/nf4_packed_graph_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"

namespace {
namespace f = ct_packed_graph_test;
struct Backend final {
  static constexpr bool supports_capture = true;
  static constexpr auto provider_id = f::c::cuda_native::kProviderId;
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

#if defined(HYLE_COMPUTE_TORCH_CUDA)
// This comparison also executes the optional independent ATen decoder.
TEST(CudaCtPackedLinear, PublicChainsKeepSourceRoundsAndTypedCastWitnesses) {
  f::PublicChainsKeepSourceRoundsAndTypedCastWitnesses<Backend>();
}
#endif

TEST(CudaCtPackedLinear, CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults) {
  f::CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults<Backend>();
}

#if defined(HYLE_COMPUTE_TORCH_CUDA)
TEST(CudaCtPackedLinear, PublishedViewAndControlEscapesExecuteTheOriginalDecomposition) {
  f::PublishedViewAndControlEscapesExecuteTheOriginalDecomposition<Backend>();
}
#else
TEST(CudaCtPackedLinear, MissingIndependentDecoderRejectsPublishedViewAndControlEscapes) {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  for (auto escape : {f::Escape::kPublishedDecode, f::Escape::kExtraView, f::Escape::kControl}) {
    const auto logical = f::build({true, f::Type::kBFloat16, f::Type::kFloat32,
        f::Type::kFloat32, 3}, 9, 64, escape);
    const auto compiled = Backend::compile_unrestricted(logical.graph, context, owner);
    ASSERT_FALSE(compiled);
    EXPECT_TRUE(compiled.error().status().IsUnimplemented()) << compiled.error().message();
  }
}
#endif

TEST(CudaCtPackedLinear, OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank) {
  f::OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank<Backend>();
}
TEST(CudaCtPackedLinear, Nf4SourceHalfRoundsAndMutableNestedOffset) {
  nf4_packed_graph_test::SourceHalfRoundsAndMutableNestedOffset<Backend>();
}

} // namespace
