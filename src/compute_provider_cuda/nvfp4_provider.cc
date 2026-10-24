#include "nvfp4_provider.h"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "nvfp4_kernel.h"

namespace hyle::compute::cuda_native::nvfp4 {
namespace {
constexpr std::array<std::size_t,6> kAlign{2,1,1,4,4,4};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true};
struct Prepared final {
  CUcontext primary = nullptr;
  Program program{};
  std::array<std::size_t,6> input_bytes{};
  std::size_t input_count = 0, output_bytes = 0, workspace_bytes = 0;
};
Result<Program> select(const Operation &operation, std::span<const TensorDesc> in,
                       std::span<const TensorDesc> out) noexcept {
  const auto local=operation.custom_as<model_ops::LocalNvfp4Linear>();
  const auto local_grouped=operation.custom_as<model_ops::GroupedLocalNvfp4Linear>();
  const auto mx_operation=operation.custom_as<model_ops::Mxfp4Linear>();
  const auto mx_grouped_operation=operation.custom_as<model_ops::GroupedMxfp4Linear>();
  const bool mx_grouped=mx_grouped_operation.has_value();
  const bool mx=mx_grouped || mx_operation.has_value();
  const bool grouped=mx_grouped || local_grouped.has_value() ||
      operation.custom_as<model_ops::GroupedNvfp4Linear>().has_value();
  if (!mx && !local && !grouped && !operation.custom_as<model_ops::Nvfp4Linear>())
    return std::unexpected(Status::Unimplemented("CUDA NVFP4 reference operation differs"));
  std::array<TensorDesc,1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation,in,inferred));
  if (out.size()!=1 || out[0]!=inferred[0])
    return std::unexpected(Status::InvalidArgument("CUDA NVFP4 reference output differs"));
  Program p;
  if (mx) {
    const auto activation_mode=mx_operation?mx_operation->activation_mode:
        mx_grouped_operation->activation_mode;
    p.contract=activation_mode==model_ops::Mxfp4ActivationMode::kBFloat16?
        Contract::kMxfp4BFloat16:Contract::kMxfp4;
  } else if (local || local_grouped) {
    const auto rounding=local?local->rounding:local_grouped->rounding;
    p.contract=rounding==model_ops::Fp4InputRounding::kBFloat16?
        Contract::kLocalBFloat16:Contract::kLocalFloat32;
  }
  p.rows=static_cast<std::uint64_t>(in[0].dimensions()[0]);
  p.columns=static_cast<std::uint64_t>(in[1].dimensions()[grouped?1:0]);
  p.depth=static_cast<std::uint64_t>(in[0].dimensions()[grouped?2:1]);
  p.quant_rows=p.rows;
  p.grouped=grouped;
  if (grouped) {
    p.slots=static_cast<std::uint64_t>(in[mx?3:5].dimensions()[1]);
    p.experts=static_cast<std::uint64_t>(in[1].dimensions()[0]);
    p.activation_slots=static_cast<std::uint64_t>(in[0].dimensions()[1]);
    p.activation_token_stride=p.rows>1?
        static_cast<std::uint64_t>(in[0].byte_strides()[0]/2):0;
    p.activation_slot_stride=static_cast<std::uint64_t>(in[0].byte_strides()[1]/2);
    if (p.rows>(std::numeric_limits<std::uint64_t>::max)()/p.slots)
      return std::unexpected(Status::OutOfRange("CUDA grouped NVFP4 rows overflow"));
    p.rows*=p.slots;
    if (!mx || (p.activation_slots!=1 && p.activation_slot_stride!=0))
      p.quant_rows=p.rows;
  }
  const auto limit=(std::numeric_limits<std::size_t>::max)()/2;
  if (p.depth>limit || p.quant_rows>limit/p.depth)
    return std::unexpected(Status::OutOfRange("CUDA NVFP4 quantized activation workspace overflows"));
  return p;
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  allocator.destroy(static_cast<Prepared *>(state));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef, mem::BufferView workspace,
                         void **output) noexcept {
  if (!output) return Status::InvalidArgument("CUDA NVFP4 invocation output is null");
  *output = nullptr;
  const auto &p = *static_cast<const Prepared *>(state);
  if (workspace.size()<p.workspace_bytes)
    return Status::InvalidArgument("CUDA NVFP4 activation workspace is too small");
  if (p.workspace_bytes==0) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(*output,cuda::detail::mutable_pointer(workspace,16,1));
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
bool bounded(const void *pointer, std::size_t bytes) noexcept {
  return pointer && bytes<=UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(pointer);
}
bool overlaps(const void *a, std::size_t na, const void *b, std::size_t nb) noexcept {
  const auto x=reinterpret_cast<std::uintptr_t>(a), y=reinterpret_cast<std::uintptr_t>(b);
  return x<y+nb && y<x+na;
}
detail::EnqueueResult submit(void *state, void *activation,
    detail::QueueSequence &sequence, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept {
  const auto &p=*static_cast<const Prepared *>(state);
  if (inputs.size()!=p.input_count || outputs.size()!=1)
    return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA NVFP4 binding counts differ"));
  auto stream=cuda::stream(sequence);
  if (!stream) return detail::EnqueueResult::rejected(stream.error());
  std::array<const void *,6> source{};
  for (std::size_t i=0;i<p.input_count;++i) {
    if (inputs[i].storage().size()<p.input_bytes[i])
      return detail::EnqueueResult::rejected(Status::OutOfRange("CUDA NVFP4 input span is too small"));
    auto pointer=cuda::detail::const_pointer(inputs[i].storage(),kAlign[i],1);
    if (!pointer) return detail::EnqueueResult::rejected(pointer.error());
    source[i]=*pointer;
  }
  if (outputs[0].storage().size()<p.output_bytes)
    return detail::EnqueueResult::rejected(Status::OutOfRange("CUDA NVFP4 output span is too small"));
  auto output=cuda::detail::mutable_pointer(outputs[0].storage(),2,1);
  if (!output) return detail::EnqueueResult::rejected(output.error());
  if (!bounded(*output,p.output_bytes) || (p.workspace_bytes!=0 &&
      (!bounded(activation,p.workspace_bytes) ||
       overlaps(*output,p.output_bytes,activation,p.workspace_bytes))))
    return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA NVFP4 output/workspace overlaps"));
  for (std::size_t i=0;i<p.input_count;++i)
    if (!bounded(source[i],p.input_bytes[i]) ||
        overlaps(source[i],p.input_bytes[i],*output,p.output_bytes) ||
        (p.workspace_bytes!=0 &&
         overlaps(source[i],p.input_bytes[i],activation,p.workspace_bytes)))
      return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA NVFP4 input/output/workspace overlaps"));
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  unsigned *word=nullptr;
  auto initialized=cuda::detail::acquire_device_status(sequence,&word);
  if (!initialized.status.ok()) return initialized;
  // The reset already owns a real Queue prefix, including on launch failure.
  return detail::EnqueueResult::accepted_command(cuda::status(
      launch(p.program,source.data(),activation,*output,word,*stream)));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared=&destroy,
    .alias_support=&alias,
    .create_invocation=&create_invocation,
    .destroy_invocation=&destroy_invocation,
    .shared_submit=&submit};
} // namespace
Status propose(const detail::ProviderOperationRequest &request,
    std::span<detail::ResourceBindingRequirement> inputs,
    std::span<detail::ResourceBindingRequirement> outputs,
    detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA NVFP4 reference needs an exact shape");
  HYLE_ASSIGN_OR_RETURN(const auto program,select(request.operation,request.inputs,request.outputs));
  if (!proposal || inputs.size()!=request.inputs.size() || outputs.size()!=1)
    return Status::InvalidArgument("CUDA NVFP4 proposal binding counts differ");
  for (std::size_t i=0;i<inputs.size();++i)
    inputs[i]={.required_properties=mem::MemoryProperty::kDeviceAccessible,
               .minimum_alignment=kAlign[i]};
  outputs[0]={.required_properties=mem::MemoryProperty::kDeviceAccessible,.minimum_alignment=2};
  const auto bytes=program.contract==Contract::kMxfp4BFloat16?std::size_t{0}:
      static_cast<std::size_t>(program.quant_rows*program.depth*2);
  *proposal={.route=kRoute,.runtime=kRuntime,
      .workspace_resource={.required_properties=mem::MemoryProperty::kDeviceAccessible,
                           .minimum_alignment=16},
      .cost={.workspace=WorkspaceRequirement{bytes,16},.persistent_bytes=sizeof(Prepared),
             .copy_bytes=0,.repack_bytes=bytes,.synchronization_count=0,
             .dispatch_count=program.grouped?4u:3u},
      .implementation=ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
    mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  if (!output) return ProviderPrepareOutcome::error(Status::InvalidArgument("CUDA NVFP4 prepared output is null"));
  *output={};
  auto program=select(request.operation,request.inputs,request.outputs);
  if (!program) return ProviderPrepareOutcome::error(program.error());
  const auto primary=static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  const auto configured=prepare_kernel(*program);
  if (configured!=cudaSuccess) return ProviderPrepareOutcome::error(cuda::status(configured));
  auto *p=allocator.construct<Prepared>();
  if (!p) return ProviderPrepareOutcome::error(Status::ResourceExhausted("CUDA NVFP4 prepared metadata"));
  p->primary=primary;p->program=*program;
  p->input_count=request.inputs.size();
  p->workspace_bytes=program->contract==Contract::kMxfp4BFloat16?std::size_t{0}:
      static_cast<std::size_t>(program->quant_rows*program->depth*2);
  for (std::size_t i=0;i<p->input_count;++i) p->input_bytes[i]=*request.inputs[i].required_bytes();
  p->output_bytes=*request.outputs[0].required_bytes();
  *output={.vtable=&kPrepared,.state=p,
           .workspace={.size=p->workspace_bytes,.alignment=16},.runtime=kRuntime,
           .persistent_bytes=sizeof(Prepared),.invocation_private_bytes=0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::nvfp4
