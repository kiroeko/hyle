#include "hyle/compute/providers/flashinfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>

#include "hyle/compute/route_policy.h"
#include "hyle/compute/model_ops.h"
#include "native_attention.h"
#include "native_batch.h"
#include "native_cache_encoder.h"
#include "hyle/compute/detail/batch_provider.h"
#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::flashinfer {
namespace {

constexpr std::size_t kAlignment = 16;
inline constexpr std::string_view kProviderVersion = "0.6.18+69ff11fc";
inline constexpr std::array kHardwarePredicates{
    RouteFactPredicate::number(route_fact::kArchitectureMajor, 8),
    RouteFactPredicate::number(route_fact::kArchitectureMinor, 9)};
inline constexpr RouteSelector kHardwareRule{
    .route = NativeRoute::kCuda,
    .provider = kProviderId,
    .provider_version = kProviderVersion,
    .environment = kHardwarePredicates};
inline constexpr std::array kFloat16AttentionPorts{
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 0,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 1,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 2,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                         .index = 0,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4}};
inline constexpr std::array kBFloat16AttentionPorts{
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 0,
                         .data_type = DataType::kBFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 1,
                         .data_type = DataType::kBFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 2,
                         .data_type = DataType::kBFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4},
    RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                         .index = 0,
                         .data_type = DataType::kBFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 4,
                         .maximum_rank = 4}};
inline constexpr RoutePolicyEntry kFloat16AttentionPolicy{
    .id = {0x48594c45464c5348ULL, 1, 1},
    .preference = 100,
    .provider = kProviderId,
    .implementation = ProviderImplementationId{1},
    .selector = {.route = NativeRoute::kCuda,
                 .provider = kProviderId,
                 .provider_version = kProviderVersion,
                 .operation = OperationKind::kScaledDotProductAttention,
                 .ports = kFloat16AttentionPorts}};
inline constexpr RoutePolicyEntry kBFloat16AttentionPolicy{
    .id = {0x48594c45464c5348ULL, 2, 1},
    .preference = 100,
    .provider = kProviderId,
    .implementation = ProviderImplementationId{1},
    .selector = {.route = NativeRoute::kCuda,
                 .provider = kProviderId,
                 .provider_version = kProviderVersion,
                 .operation = OperationKind::kScaledDotProductAttention,
                 .ports = kBFloat16AttentionPorts}};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

struct ProviderContext final {
  CUcontext primary = nullptr;
  std::uint64_t maximum_grid = 0;
};

struct BatchSlot final {
  CUcontext primary=nullptr;
  native::BatchSpec spec{};
  void*plan=nullptr;
  std::size_t plan_bytes=0;
  std::uint64_t generation=0;
  BatchLayoutVersion layout;
  BatchPlanReport report;
};
Status qualify_batch(void*opaque,BatchCapacity capacity,const BatchProgramOptions&options)noexcept {
  auto&context=*static_cast<ProviderContext*>(opaque);
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(capacity));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(options.causal,options.window_left));
  const bool fp8=capacity.key_storage.data_type()==DataType::kFloat8E4M3FN &&
                 capacity.value_storage.data_type()==DataType::kFloat8E4M3FN;
  const bool plain=capacity.key_storage.format().is_plain() && capacity.value_storage.format().is_plain() &&
      capacity.key_storage.data_type()==capacity.query_type && capacity.value_storage.data_type()==capacity.query_type;
  if(capacity.output_type!=capacity.query_type || (!fp8 && !plain) ||
     (capacity.query_type!=DataType::kFloat16 && capacity.query_type!=DataType::kBFloat16) ||
     !(capacity.value_head_dim==capacity.qk_head_dim || (capacity.qk_head_dim==64 && capacity.value_head_dim==128)))
    return Status::Unimplemented("FlashInfer batch requires same-type Q/output, plain or FP8 K/V and finite geometry");
  const auto effective_scale=options.scale*capacity.key_storage.scale();
  if(!std::isfinite(effective_scale) || std::abs(effective_scale)>std::numeric_limits<float>::max()/1.4426950408889634F)
    return Status::InvalidArgument("FlashInfer effective QK log2 scale must be finite");
  const auto group=capacity.query_heads/capacity.kv_heads;
  if(!context.maximum_grid ||
     (capacity.qk_head_dim!=64 && capacity.qk_head_dim!=128 && capacity.qk_head_dim!=256) ||
     (group!=1 && group!=2 && group!=4 && group!=8))return Status::Unimplemented("FlashInfer batch domain is not qualified");
  return Status::Ok();
}
Status create_batch_slot(void*opaque,BatchCapacity capacity,const BatchProgramOptions&options,
                         mem::AllocatorRef allocator,void**output)noexcept {
  *output=nullptr;HYLE_RETURN_IF_ERROR(qualify_batch(opaque,capacity,options));auto&context=*static_cast<ProviderContext*>(opaque);
  HYLE_ASSIGN_OR_RETURN(auto offsets,batch_metadata_offsets(capacity));
  auto*p=allocator.construct<BatchSlot>();if(!p)return Status::ResourceExhausted("FlashInfer batch slot metadata");
  p->primary=context.primary;p->plan_bytes=native::batch_state_bytes();p->plan=allocator.allocate(p->plan_bytes,16);
  if(!p->plan){allocator.destroy(p);return Status::ResourceExhausted("FlashInfer batch plan metadata");}
  native::initialize_batch_state(p->plan);
  p->spec={.query_heads=capacity.query_heads,.kv_heads=capacity.kv_heads,.qk_head_dim=capacity.qk_head_dim,
    .value_head_dim=capacity.value_head_dim,.page_size=capacity.page_size,
    .query_indptr_offset=offsets.query_indptr,.kv_indptr_offset=offsets.kv_indptr,.page_indices_offset=offsets.page_indices,.tail_offset=offsets.last_page_lengths,
    .scale=options.scale,.key_scale=capacity.key_storage.scale(),.value_scale=capacity.value_storage.scale(),
    .window_left=options.window_left,.paged=capacity.storage==BatchStorage::kPaged,.bfloat16=capacity.query_type==DataType::kBFloat16,
    .fp8=capacity.key_storage.data_type()==DataType::kFloat8E4M3FN,.causal=options.causal};
  *output=p;return Status::Ok();
}
void destroy_batch_slot(void*opaque,mem::AllocatorRef allocator)noexcept {
  auto*p=static_cast<BatchSlot*>(opaque);if(!p)return;
  native::destroy_batch_state(p->plan);allocator.deallocate(p->plan,p->plan_bytes,16);allocator.destroy(p);
}
detail::EnqueueResult plan_batch_slot(void*opaque,Queue&queue,const BatchLayoutVersion&layout,
                                    detail::BatchWorkspaceViews workspace,BatchPlanReport*report)noexcept {
  auto&slot=*static_cast<BatchSlot*>(opaque);slot.generation=0;
  auto stream=cuda::stream(queue);if(!stream)return detail::EnqueueResult::rejected(stream.error());
  auto floating=cuda::detail::mutable_pointer(workspace.float_workspace,16,1);if(!floating)return detail::EnqueueResult::rejected(floating.error());
  auto integer=cuda::detail::mutable_pointer(workspace.int_workspace,16,1);if(!integer)return detail::EnqueueResult::rejected(integer.error());
  const auto resource=workspace.host_workspace.resource();
  if(resource!=mem::system_resource() && (!resource.properties().has(mem::MemoryProperty::kHostPinned) ||
      resource.properties().domain.authority!=workspace.int_workspace.resource().properties().domain.authority))
    return detail::EnqueueResult::rejected(Status::Unimplemented("batch planning host workspace requires system or original CUDA pinned storage"));
  // These two admitted resources have persistent host maps; original CUDA
  // pinned storage is a host carrier, not a GPU tensor pointer.
  auto host=workspace.host_workspace.map_host();if(!host)return detail::EnqueueResult::rejected(host.error());
  auto hostbytes=host->writable_bytes();if(!hostbytes)return detail::EnqueueResult::rejected(hostbytes.error());
  void*host_pointer=hostbytes->data();
  cuda::detail::ScopedPrimaryContext scope(slot.primary);if(!scope)return detail::EnqueueResult::rejected(scope.status());
  auto spec=slot.spec;spec.query_rows=layout.facts().query_rows;spec.requests=layout.facts().requests;
  const auto metadata=layout.host();native::BatchFacts facts;
  try {
    auto status=cuda::status(native::query_batch_workspace(spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),*stream,facts));
    if(!status.ok())return detail::EnqueueResult::rejected(status);
    if(facts.float_bytes>workspace.float_workspace.size() || facts.int_bytes>workspace.int_workspace.size() || facts.int_bytes>workspace.host_workspace.size())
      return detail::EnqueueResult::rejected(Status::ResourceExhausted("FlashInfer batch workspace capacity exhausted"));
    // After this boundary preserve a possibly accepted original H2D prefix.
    try {
      status=cuda::status(native::plan_batch(slot.plan,spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),*floating,workspace.float_workspace.size(),*integer,host_pointer,workspace.int_workspace.size(),*stream,facts));
      if(status.ok()){slot.spec=spec;slot.generation=layout.generation();slot.layout=layout;*report={.layout_generation=layout.generation(),.float_workspace_bytes=facts.float_bytes,.int_workspace_bytes=facts.int_bytes,.cta_tile_q=facts.cta_tile_q,.padded_batch_size=facts.padded_batch_size,.split_kv=facts.split_kv};slot.report=*report;}
      return detail::EnqueueResult::accepted_command(status);
    }catch(const std::bad_alloc&){return detail::EnqueueResult::accepted_command(Status::ResourceExhausted("FlashInfer batch planning host allocation"));}
     catch(const std::exception&e){return detail::EnqueueResult::accepted_command(Status::Internal(e.what()));}
  }catch(const std::bad_alloc&){return detail::EnqueueResult::rejected(Status::ResourceExhausted("FlashInfer batch workspace query host allocation"));}
   catch(const std::exception&e){return detail::EnqueueResult::rejected(Status::Internal(e.what()));}
}
detail::EnqueueResult run_batch_slot(void*opaque,Queue&queue,std::span<const ConstTensorView>inputs,
                                   detail::BatchWorkspaceViews workspace)noexcept {
  auto&slot=*static_cast<BatchSlot*>(opaque);
  if(!slot.generation || inputs.size()!=3)return detail::EnqueueResult::rejected(Status::FailedPrecondition("FlashInfer batch slot has no prepared layout"));
  auto stream=cuda::stream(queue);if(!stream)return detail::EnqueueResult::rejected(stream.error());
  std::array<const void*,3> pointers{};
  for(std::size_t i=0;i<3;++i){auto p=cuda::detail::const_pointer(inputs[i].storage(),16,1);if(!p)return detail::EnqueueResult::rejected(p.error());pointers[i]=*p;}
  auto metadata=cuda::detail::const_pointer(mem::ConstBufferView(workspace.metadata),8,1);if(!metadata)return detail::EnqueueResult::rejected(metadata.error());
  auto floating=cuda::detail::mutable_pointer(workspace.float_workspace,16,1);if(!floating)return detail::EnqueueResult::rejected(floating.error());
  auto integer=cuda::detail::mutable_pointer(workspace.int_workspace,16,1);if(!integer)return detail::EnqueueResult::rejected(integer.error());
  auto output=cuda::detail::mutable_pointer(workspace.output,16,1);if(!output)return detail::EnqueueResult::rejected(output.error());
  cuda::detail::ScopedPrimaryContext scope(slot.primary);if(!scope)return detail::EnqueueResult::rejected(scope.status());
  try{return detail::EnqueueResult::accepted_command(cuda::status(native::run_batch(slot.plan,slot.spec,pointers[0],pointers[1],pointers[2],*metadata,*output,*floating,*integer,*stream)));}
  catch(const std::bad_alloc&){return detail::EnqueueResult::accepted_command(Status::ResourceExhausted("FlashInfer batch run host allocation"));}
  catch(const std::exception&e){return detail::EnqueueResult::accepted_command(Status::Internal(e.what()));}
}
Result<bool> update_batch_slot(void*opaque,const BatchLayoutVersion&layout,BatchPlanReport*report)noexcept{
  auto&slot=*static_cast<BatchSlot*>(opaque);if(!slot.generation || !slot.layout.schema().owns(layout))return false;
  const auto old=slot.layout.host(),next=layout.host();
  const auto equal=[](auto a,auto b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());};
  // Original PrefillPlan consumes these indptr values and the immutable spec,
  // not physical page IDs/tails. Ragged KV offsets are part of this full key.
  if(!equal(old.query_indptr,next.query_indptr)||!equal(old.kv_indptr,next.kv_indptr))return false;
  slot.layout=layout;slot.generation=layout.generation();*report=slot.report;report->layout_generation=slot.generation;report->reused_native_plan=true;
  slot.report=*report;return true;
}
const detail::BatchProviderVTable kBatchVTable{&qualify_batch,&create_batch_slot,&destroy_batch_slot,&update_batch_slot,&plan_batch_slot,&run_batch_slot};
const void*query_extension(void*,ProviderExtensionId id)noexcept{return id==detail::kBatchPlanningExtensionId?&kBatchVTable:nullptr;}

struct Prepared final {
  CUcontext primary = nullptr;
  native::Attention attention{};
  std::size_t batches = 0;
  std::size_t query_batch_bytes = 0;
  std::size_t kv_batch_bytes = 0;
  std::size_t workspace_bytes = 0;
  native::EncoderSpec encoder{};
  bool cache_encode=false;
};

struct Invocation final {
  void *workspace = nullptr;
  bool failed = false;
};

Status create_context(const Provider &provider, Device device,
                      void *backend_state, mem::AllocatorRef allocator,
                      void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("FlashInfer requires CUDA");
  const auto primary = cuda::detail::borrow_primary_context(backend_state);
  const auto *environment = cuda::detail::borrow_environment(backend_state);
  if (primary == nullptr || environment == nullptr)
    return Status::FailedPrecondition("FlashInfer CUDA context is invalid");
  const auto facts = cuda::detail::route_facts(*environment);
  const RouteMatchInput route_input{.platform = kBuildRoutePlatform,
                                    .route = NativeRoute::kCuda,
                                    .provider = provider.identity(),
                            .environment = facts.environment()};
  const bool qualified =
      match_route_selector(kHardwareRule, route_input).matched();
  std::uint64_t maximum_grid = 0;
  if (qualified) {
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;
    HYLE_RETURN_IF_ERROR(cuda::status(
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                               environment->device_ordinal)));
    HYLE_RETURN_IF_ERROR(cuda::status(cudaDeviceGetAttribute(
        &threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor,
        environment->device_ordinal)));
    if (multiprocessors <= 0 || threads_per_multiprocessor < 128)
      return Status::Internal("FlashInfer CUDA residency facts are invalid");
    // Both admitted upstream entry families use at least 128 threads/block.
    maximum_grid = static_cast<std::uint64_t>(multiprocessors) *
                   static_cast<std::uint64_t>(threads_per_multiprocessor / 128);
  }
  void *storage =
      allocator.allocate(sizeof(ProviderContext), alignof(ProviderContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("FlashInfer context allocation failed");
  *output = ::new (storage) ProviderContext{primary, maximum_grid};
  return Status::Ok();
}

void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<ProviderContext *>(state);
  if (context == nullptr)
    return;
  context->~ProviderContext();
  allocator.deallocate(context, sizeof(ProviderContext),
                       alignof(ProviderContext));
}

Status describe(const ProviderContext &context, Operation operation,
                std::span<const TensorDesc> inputs,
                std::span<const TensorDesc> outputs,
                Prepared &prepared) noexcept {
  if(const auto encode=operation.custom_as<model_ops::CacheEncode>()){
    if(!context.maximum_grid || encode->storage.data_type()!=DataType::kFloat8E4M3FN)
      return Status::Unimplemented("FlashInfer encoder requires the admitted FP8 CUDA domain");
    TensorDesc output;
    HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation,inputs,std::span(&output,1)));
    if(outputs.size()!=1 || outputs[0]!=output)
      return Status::InvalidArgument("FlashInfer cache encoder output carrier differs");
    const auto type=inputs[0].data_type();
    prepared={.primary=context.primary,.encoder={.elements=*inputs[0].required_bytes()/data_type_size(type),
      .scale=encode->storage.scale(),.input=type==DataType::kFloat32?native::EncoderInput::kFloat32:
          type==DataType::kBFloat16?native::EncoderInput::kBFloat16:native::EncoderInput::kFloat16},.cache_encode=true};
    return Status::Ok();
  }
  const auto *attention = operation.get_if<ScaledDotProductAttention>();
  const RouteMatchInput route_input{
      .platform = kBuildRoutePlatform,
      .route = NativeRoute::kCuda,
      .provider = {kProviderId, "flashinfer", kProviderVersion},
      .operation = operation,
      .inputs = inputs,
      .outputs = outputs};
  const bool typed_attention =
      match_route_policy(kFloat16AttentionPolicy, route_input).matched() ||
      match_route_policy(kBFloat16AttentionPolicy, route_input).matched();
  if (context.maximum_grid == 0 || !typed_attention || attention == nullptr ||
      inputs.size() != 3 || outputs.size() != 1 || !attention->causal ||
      attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
      attention->mask != MaskKind::kNone ||
      attention->accumulation_type != DataType::kFloat32)
    return Status::Unimplemented(
        "FlashInfer requires the admitted causal SM89 domain");
  const auto type = inputs[0].data_type();
  for (const auto &description : inputs) {
    if (!description.valid() || !description.is_plain() ||
        !description.is_dense() || description.rank() != 4 ||
        description.data_type() != type)
      return Status::Unimplemented(
          "FlashInfer requires dense matching BSHD tensors");
  }
  const auto q = inputs[0].dimensions();
  const auto k = inputs[1].dimensions();
  const auto v = inputs[2].dimensions();
  if (q[0] <= 0 || q[1] <= 0 || k[1] < q[1] || q[2] != 8 || k[2] != 2 ||
      q[3] != 256 || k[3] != 256 || v[3] != 256 ||
      q[1] > std::numeric_limits<std::int32_t>::max() / (8 * 256) ||
      k[1] > std::numeric_limits<std::int32_t>::max() / (2 * 256))
    return Status::Unimplemented(
        "FlashInfer head/length domain is not admitted");
  if (q[0] != k[0] || !std::equal(k.begin(), k.end(), v.begin()) ||
      outputs[0] != inputs[0] || !std::isfinite(attention->scale))
    return Status::InvalidArgument(
        "FlashInfer attention descriptions disagree");
  const auto query_length = static_cast<std::uint64_t>(q[1]);
  const auto kv_length = static_cast<std::uint64_t>(k[1]);
  const auto chunks_by_length = (kv_length + 255) / 256;
  // Upstream split-KV uses chunks of at least 256 tokens. For prefill,
  // chunks * q_length * q_heads <= maximum_grid * maximum_CTA_tile_q (128).
  // For decode the bound is maximum_grid * GQA_group_size (4). Take the
  // tighter shape/residency bound; batches reuse scratch in one ordered stream.
  const auto partial_rows =
      (std::min)(chunks_by_length * query_length * 8,
                 context.maximum_grid * (query_length == 1 ? 4 : 128));
  const auto scratch_bytes =
      partial_rows * (256 * sizeof(std::uint16_t) + sizeof(float));
  if (scratch_bytes > std::numeric_limits<std::size_t>::max())
    return Status::OutOfRange("FlashInfer workspace size overflows");
  prepared = {
      .primary = context.primary,
      .attention = {.query_length = static_cast<std::uint32_t>(q[1]),
                    .kv_length = static_cast<std::uint32_t>(k[1]),
                    .scale = attention->scale,
                    .bfloat16 = type == DataType::kBFloat16},
      .batches = static_cast<std::size_t>(q[0]),
      .query_batch_bytes = static_cast<std::size_t>(query_length * 8 * 256 * 2),
      .kv_batch_bytes = static_cast<std::size_t>(kv_length * 2 * 256 * 2),
      .workspace_bytes = static_cast<std::size_t>(scratch_bytes)};
  return Status::Ok();
}

Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("FlashInfer requires an exact prepared shape");
  if (inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument("FlashInfer proposal binding counts differ");
  Prepared prepared;
  HYLE_RETURN_IF_ERROR(describe(*static_cast<ProviderContext *>(state),
                                request.operation, request.inputs,
                                request.outputs, prepared));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = kAlignment};
  for (auto &input : inputs)
    input = binding;
  for (auto &output : outputs)
    output = binding;
  *proposal = {
      .route = 1,
      .runtime = kRuntime,
      .workspace_resource = binding,
      .cost = {.workspace =
                   WorkspaceRequirement{prepared.workspace_bytes, kAlignment},
               .persistent_bytes = std::nullopt,
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = std::nullopt},
      .implementation = ProviderImplementationId{1}};
  if(prepared.cache_encode){
    proposal->route=2;proposal->implementation=ProviderImplementationId{2};
    proposal->policy={0x48594c45464c5348ULL,3,1};proposal->preference=100;
    proposal->cost.dispatch_count=prepared.encoder.elements?1u:0u;
    proposal->cost.synchronization_count=std::nullopt; // original status reset/D2H/completion
    return Status::Ok();
  }
  proposal->policy = request.inputs.front().data_type() == DataType::kFloat16
                         ? kFloat16AttentionPolicy.id
                         : kBFloat16AttentionPolicy.id;
  proposal->preference = kFloat16AttentionPolicy.preference;
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}

Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  const auto &prepared = *static_cast<Prepared *>(state);
  if (workspace.size() < prepared.workspace_bytes)
    return Status::InvalidArgument("FlashInfer workspace is too small");
  void *pointer=nullptr;
  if(prepared.workspace_bytes){
    auto mapped=cuda::detail::mutable_pointer(workspace,kAlignment,1);
    if(!mapped)return std::move(mapped).error();
    pointer=*mapped;
  }
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("FlashInfer invocation allocation failed");
  *output = ::new (storage) Invocation{pointer};
  return Status::Ok();
}

void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (invocation == nullptr)
    return;
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}

detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<Prepared *>(state);
  auto &invocation = *static_cast<Invocation *>(lane);
  if (invocation.failed)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "FlashInfer invocation failed; create another"));
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  if(prepared.cache_encode){
    if(!prepared.encoder.elements)return detail::EnqueueResult::accepted_command();
    auto input=cuda::detail::const_pointer(inputs[0].storage(),kAlignment,1);
    if(!input)return detail::EnqueueResult::rejected(input.error());
    auto output=cuda::detail::mutable_pointer(outputs[0].storage(),kAlignment,1);
    if(!output)return detail::EnqueueResult::rejected(output.error());
    cuda::detail::ScopedPrimaryContext scope(prepared.primary);
    if(!scope)return detail::EnqueueResult::rejected(scope.status());
    unsigned *word=nullptr;auto initialized=cuda::detail::acquire_device_status(queue,&word);
    if(!initialized.status.ok())return initialized;
    auto status=cuda::status(native::encode_cache(prepared.encoder,*input,*output,word,*stream));
    invocation.failed=!status.ok();
    return detail::EnqueueResult::accepted_command(std::move(status));
  }
  std::array<const std::byte *, 3> pointers{};
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    auto pointer =
        cuda::detail::const_pointer(inputs[i].storage(), kAlignment, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[i] = static_cast<const std::byte *>(*pointer);
  }
  auto output =
      cuda::detail::mutable_pointer(outputs[0].storage(), kAlignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  Status terminal;
  try {
    for (std::size_t batch = 0; batch < prepared.batches; ++batch) {
      terminal = cuda::status(native::execute(
          prepared.attention, pointers[0] + batch * prepared.query_batch_bytes,
          pointers[1] + batch * prepared.kv_batch_bytes,
          pointers[2] + batch * prepared.kv_batch_bytes,
          static_cast<std::byte *>(*output) +
              batch * prepared.query_batch_bytes,
          invocation.workspace, *stream));
      if (!terminal.ok())
        break;
    }
  } catch (const std::bad_alloc &) {
    terminal = Status::ResourceExhausted("FlashInfer host allocation failed");
  } catch (const std::exception &error) {
    terminal = Status::Internal(error.what());
  }
  invocation.failed = !terminal.ok();
  // A failure may follow a partial native launch. The existing Queue terminal
  // owns the accepted prefix, Invocation and workspace until quiescence.
  return detail::EnqueueResult::accepted_command(std::move(terminal));
}

AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (!request.shape.exact())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("FlashInfer prepare route differs"));
  Prepared value;
  auto status =
      describe(*static_cast<ProviderContext *>(state), request.operation,
               request.inputs, request.outputs, value);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  if (request.route != (value.cache_encode ? 2u : 1u))
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("FlashInfer prepare route differs"));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("FlashInfer artifact allocation failed"));
  *output = {.vtable = &kPreparedVTable,
             .state = ::new (storage) Prepared(value),
             .workspace = {value.workspace_bytes, kAlignment},
             .runtime = kRuntime,
             .persistent_bytes = std::nullopt,
             .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "flashinfer",
                 .version = kProviderVersion},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
    .query_extension = &query_extension,
};

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}

} // namespace hyle::compute::flashinfer
