#include "hyle/compute/providers/flashinfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/route_policy.h"
#include "hyle/compute/model_ops.h"
#include "native_attention.h"
#include "native_batch.h"
#include "native_cache_encoder.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/batch_output_extent.h"
#include "hyle/compute/detail/queue_ref.h"
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

struct BatchWorkspaceQuery final {
  BatchLayoutVersion layout;
  detail::QueueRef queue;
  CUcontext primary=nullptr;
  native::BatchSpec spec{};
  native::BatchFacts facts{};
};
bool same_batch_spec(const native::BatchSpec&a,const native::BatchSpec&b)noexcept{
  return a.query_heads==b.query_heads && a.kv_heads==b.kv_heads && a.qk_head_dim==b.qk_head_dim &&
      a.value_head_dim==b.value_head_dim && a.page_size==b.page_size &&
      a.query_rows==b.query_rows && a.requests==b.requests &&
      a.query_indptr_offset==b.query_indptr_offset && a.kv_indptr_offset==b.kv_indptr_offset &&
      a.page_indices_offset==b.page_indices_offset && a.tail_offset==b.tail_offset &&
      a.scale==b.scale && a.key_scale==b.key_scale && a.value_scale==b.value_scale &&
      a.window_left==b.window_left && a.paged==b.paged && a.bfloat16==b.bfloat16 &&
      a.fp8==b.fp8 && a.causal==b.causal && a.allow_bf16_partials==b.allow_bf16_partials;
}
struct BatchSlot final {
  CUcontext primary=nullptr;
  native::BatchSpec spec{};
  void*plan=nullptr;
  std::size_t plan_bytes=0;
  std::uint64_t generation=0;
  std::uint32_t kv_chunk_rows=0;
  BatchLayoutVersion layout;
  BatchPlanReport report;
  // One successful fresh sizing result, consumed by the next plan attempt.
  // It is not a prepared plan or permission to submit, and never changes the
  // active generation/layout when sizing or common quota admission fails.
  BatchWorkspaceQuery workspace_query;
  bool mla=false;
};
Status qualify_batch(void*opaque,BatchCapacity capacity,const BatchProgramOptions&options)noexcept {
  auto&context=*static_cast<ProviderContext*>(opaque);
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(capacity));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(options.causal,options.window_left));
  // BatchPrefillPagedParams stores query strides in its signed IdType.
  if(std::uint64_t(capacity.query_heads)*capacity.qk_head_dim>std::uint64_t(INT32_MAX))
    return Status::OutOfRange("FlashInfer batch query stride exceeds native int32");
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
  // The original online softmax takes an unscaled maximum and masks with
  // -infinity before applying this scale. Zero/negative scales change those
  // semantics; decline instead of publishing incorrect probabilities.
  if(effective_scale<=0)
    return Status::Unimplemented("FlashInfer batch requires positive effective QK scale");
  const bool highdim=capacity.qk_head_dim==512 && capacity.value_head_dim==512 && plain;
  // Native packed-query/output and ragged tensor offsets use uint32 arithmetic.
  // Bounding the complete tensor also leaves room for CTA/group rounding.
  if(std::uint64_t(capacity.max_query_rows)*capacity.query_heads*
         (std::max)(capacity.qk_head_dim,capacity.value_head_dim)>UINT32_MAX ||
      (capacity.storage==BatchStorage::kRagged &&
       std::uint64_t(capacity.max_kv_rows)*capacity.kv_heads*
           (std::max)(capacity.qk_head_dim,capacity.value_head_dim)>UINT32_MAX))
    return Status::OutOfRange("FlashInfer batch offsets exceed native uint32");
  // The planner packs Q rows by the runtime head ratio; the original kernel
  // maps each packed row with uint_fastdiv rather than a fixed GQA template.
  // validate_batch_capacity already proves positive, divisible head counts.
  if(!context.maximum_grid ||
     (!highdim && capacity.qk_head_dim!=64 && capacity.qk_head_dim!=128 && capacity.qk_head_dim!=256))
    return Status::Unimplemented("FlashInfer batch domain is not qualified");
  return Status::Ok();
}
Status qualify_mla(void*opaque,BatchCapacity c,const BatchProgramOptions&o)noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  const auto&context=*static_cast<ProviderContext*>(opaque);
  const bool plain=c.key_storage.format().is_plain() && c.value_storage==c.key_storage &&
      c.key_storage.data_type()==c.query_type &&
      (c.query_type==DataType::kFloat16 || c.query_type==DataType::kBFloat16);
  const bool fp8=c.query_type==DataType::kBFloat16 && c.output_type==DataType::kBFloat16 &&
      c.key_storage.data_type()==DataType::kFloat8E4M3FN && c.value_storage==c.key_storage;
  if(!context.maximum_grid || c.kv_heads!=1 ||
      (c.qk_head_dim!=512 && c.qk_head_dim!=576) || c.value_head_dim!=512 || c.query_type!=c.output_type ||
      (!plain && !fp8) || o.window_left!=-1 || o.allow_bf16_partials)
    return Status::Unimplemented("FlashInfer MLA requires packed CKV512 with optional KPE64, plain matching Q or BF16 Q with one shared E4 cache, and full causal/noncausal attention");
  if(!std::isfinite(o.scale) || o.scale<=0 || o.scale>std::numeric_limits<float>::max()/1.4426950408889634F)
    return Status::Unimplemented("FlashInfer MLA requires positive finite log2 QK scale");
  const auto packed=std::uint64_t(c.max_query_rows)*c.query_heads;
  if(packed>(std::uint64_t(INT32_MAX)-128) || packed*c.qk_head_dim>UINT32_MAX ||
      std::uint64_t(c.max_pages)*c.page_size*c.qk_head_dim>UINT32_MAX ||
       (c.storage==BatchStorage::kRagged && std::uint64_t(c.max_kv_rows)*c.qk_head_dim>UINT32_MAX) ||
      std::uint64_t(c.max_page_refs)*c.page_size>UINT32_MAX || c.max_kv_rows>std::uint32_t(INT32_MAX/2))
    return Status::OutOfRange("FlashInfer MLA packed tensor or scheduler offsets exceed native integer fields");
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
    .fp8=capacity.key_storage.data_type()==DataType::kFloat8E4M3FN,.causal=options.causal,
    .allow_bf16_partials=options.allow_bf16_partials};
  *output=p;return Status::Ok();
}
Status create_mla_slot(void*opaque,BatchCapacity capacity,const BatchProgramOptions&options,
                         mem::AllocatorRef allocator,void**output)noexcept {
  *output=nullptr;HYLE_RETURN_IF_ERROR(qualify_mla(opaque,capacity,options));auto&context=*static_cast<ProviderContext*>(opaque);
  HYLE_ASSIGN_OR_RETURN(auto offsets,batch_metadata_offsets(capacity));
  auto*p=allocator.construct<BatchSlot>();if(!p)return Status::ResourceExhausted("FlashInfer batch slot metadata");
  p->primary=context.primary;p->plan_bytes=native::mla_state_bytes();p->plan=allocator.allocate(p->plan_bytes,16);
  if(!p->plan){allocator.destroy(p);return Status::ResourceExhausted("FlashInfer batch plan metadata");}
  native::initialize_mla_state(p->plan);p->mla=true;
  p->spec={.query_heads=capacity.query_heads,.kv_heads=capacity.kv_heads,.qk_head_dim=capacity.qk_head_dim,
    .value_head_dim=capacity.value_head_dim,.page_size=capacity.page_size,
    .query_indptr_offset=offsets.query_indptr,.kv_indptr_offset=offsets.kv_indptr,.page_indices_offset=offsets.page_indices,.tail_offset=offsets.last_page_lengths,
    .scale=options.scale,.key_scale=capacity.key_storage.scale(),.value_scale=capacity.value_storage.scale(),
    .window_left=options.window_left,.paged=capacity.storage==BatchStorage::kPaged,.bfloat16=capacity.query_type==DataType::kBFloat16,
    .fp8=capacity.key_storage.data_type()==DataType::kFloat8E4M3FN,.causal=options.causal,
    .allow_bf16_partials=options.allow_bf16_partials};
  *output=p;return Status::Ok();
}
void destroy_batch_slot(void*opaque,mem::AllocatorRef allocator)noexcept {
  auto*p=static_cast<BatchSlot*>(opaque);if(!p)return;
  if(p->mla)native::destroy_mla_state(p->plan);else native::destroy_batch_state(p->plan);allocator.deallocate(p->plan,p->plan_bytes,16);allocator.destroy(p);
}
// A read-only proof about this original prepared plan, not a new schedule
// cache. Sizing and update must use exactly the same topology qualification.
bool can_reuse_batch_plan(const BatchSlot&slot,const BatchLayoutVersion&layout)noexcept{
  if(!slot.generation || !slot.layout.origin_schema().owns(layout))return false;
  const auto old=slot.layout.host(),next=layout.host();
  const auto equal=[](auto a,auto b){return a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin());};
  if(!equal(old.query_indptr,next.query_indptr))return false;
  // MLA schedules bake KV lengths and offsets; only physical page IDs may
  // change without rebuilding that original plan.
  if(slot.mla && (!equal(old.kv_indptr,next.kv_indptr) || !equal(old.last_page_lengths,next.last_page_lengths)))return false;
  if(slot.spec.query_rows && !equal(old.kv_indptr,next.kv_indptr)){
    const std::uint64_t units=slot.spec.paged?slot.spec.page_size:1u;
    if(!slot.kv_chunk_rows || slot.kv_chunk_rows%units || !slot.report.cta_tile_q)return false;
    const auto chunk=slot.kv_chunk_rows/units;
    const auto chunks=[&](HostBatchLayout metadata,std::size_t request){
      auto length=std::uint64_t(metadata.kv_indptr[request+1]-metadata.kv_indptr[request]);
      if(slot.spec.window_left>=0){
        const auto window=std::uint64_t(slot.spec.window_left)+slot.report.cta_tile_q;
        length=std::min(length,(window+units-1u)/units);
      }
      return (std::max(length,std::uint64_t{1})+chunk-1u)/chunk;
    };
    // At the original chunk/CTA, equal active-request chunk counts preserve
    // every request/q/kv tile and output/merge offset. The existing kernels
    // read current KV offsets, page IDs and tails from the runtime metadata.
    for(std::size_t request=0;request+1<next.query_indptr.size();++request){
      if(next.query_indptr[request]==next.query_indptr[request+1])continue;
      if(chunks(old,request)!=chunks(next,request))return false;
    }
  }
  return true;
}
Status query_batch_layout(BatchSlot&slot,Queue&queue,const BatchLayoutVersion&layout,
                         bool allow_update,native::BatchFacts&facts) {
  slot.workspace_query={};
  auto stream=cuda::stream(queue);if(!stream)return stream.error();
  cuda::detail::ScopedPrimaryContext scope(slot.primary);if(!scope)return scope.status();
  // Queue and primary-context admission above still runs on a reuse hit.
  // An explicit fresh-plan request must retain the original scheduler query:
  // a different fresh chunk heuristic may require more than the old plan.
  if(allow_update && can_reuse_batch_plan(slot,layout)){
    facts={};facts.float_bytes=slot.report.float_workspace_bytes;facts.int_bytes=slot.report.int_workspace_bytes;
    return Status::Ok();
  }
  auto spec=slot.spec;spec.query_rows=layout.facts().query_rows;spec.requests=layout.facts().requests;
  HYLE_ASSIGN_OR_RETURN(auto queue_owner,detail::QueueRef::acquire(queue));
  const auto metadata=layout.host();std::vector<std::int32_t> lengths;
  if(slot.mla){lengths.reserve(spec.requests);for(std::size_t r=0;r<spec.requests;++r){
    const auto refs=metadata.kv_indptr[r+1]-metadata.kv_indptr[r];
    lengths.push_back(spec.paged?(refs?(refs-1)*std::int32_t(spec.page_size)+metadata.last_page_lengths[r]:0):refs);
  }}
  HYLE_RETURN_IF_ERROR(cuda::status(slot.mla?
      native::query_mla_workspace(spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),lengths.data(),*stream,facts):
      native::query_batch_workspace(spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),*stream,facts)));
  slot.workspace_query={layout,std::move(queue_owner),slot.primary,spec,facts};
  return Status::Ok();
}
Status batch_workspace(void*opaque,Queue&queue,const BatchLayoutVersion&layout,bool allow_update,BatchPlanReport*report)noexcept {
  try{
    native::BatchFacts facts;HYLE_RETURN_IF_ERROR(query_batch_layout(*static_cast<BatchSlot*>(opaque),queue,layout,allow_update,facts));
    *report={.layout_generation=layout.generation(),.float_workspace_bytes=facts.float_bytes,.int_workspace_bytes=facts.int_bytes};
    return Status::Ok();
  }catch(const std::bad_alloc&){return Status::ResourceExhausted("FlashInfer batch workspace query host allocation");}
   catch(const std::exception&e){return Status::Internal(e.what());}
}
detail::EnqueueResult plan_batch_slot(void*opaque,detail::QueueSequence &sequence,const BatchLayoutVersion&layout,
                                    detail::BatchWorkspaceViews workspace,BatchPlanReport*report)noexcept {
  auto&slot=*static_cast<BatchSlot*>(opaque);
  // Clear before any rejection or accepted H2D prefix. A direct capability
  // caller may omit sizing; an absent or mismatched result uses the original
  // query below instead of acquiring preparation from an old generation.
  auto query=std::move(slot.workspace_query);slot.workspace_query={};
  slot.generation=0;
  auto stream=cuda::stream(sequence);if(!stream)return detail::EnqueueResult::rejected(stream.error());
  auto spec=slot.spec;spec.query_rows=layout.facts().query_rows;spec.requests=layout.facts().requests;
  if(!spec.query_rows){
    slot.spec=spec;slot.generation=layout.generation();slot.layout=layout;
    *report={.layout_generation=layout.generation()};slot.report=*report;return detail::EnqueueResult::no_op();
  }
  Result<void*> floating=static_cast<void*>(nullptr);
  if(workspace.float_workspace.valid())floating=cuda::detail::mutable_pointer(workspace.float_workspace,16,1);
  if(!floating)return detail::EnqueueResult::rejected(floating.error());
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
  const auto metadata=layout.host();native::BatchFacts facts;
  try {
    std::vector<std::int32_t> lengths;
    if(slot.mla){lengths.reserve(spec.requests);for(std::size_t r=0;r<spec.requests;++r){
       const auto refs=metadata.kv_indptr[r+1]-metadata.kv_indptr[r];
       lengths.push_back(spec.paged?(refs?(refs-1)*std::int32_t(spec.page_size)+metadata.last_page_lengths[r]:0):refs);
    }}
    Status status;
    if(query.layout==layout && query.queue.same_owner(sequence) &&
        query.primary==slot.primary && same_batch_spec(query.spec,spec)){
      facts=query.facts;
    }else{
      status=cuda::status(slot.mla?
          native::query_mla_workspace(spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),lengths.data(),*stream,facts):
          native::query_batch_workspace(spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),*stream,facts));
    }
    if(!status.ok())return detail::EnqueueResult::rejected(status);
    if(facts.float_bytes>workspace.float_workspace.size() || facts.int_bytes>workspace.int_workspace.size() || facts.int_bytes>workspace.host_workspace.size())
      return detail::EnqueueResult::rejected(Status::ResourceExhausted("FlashInfer batch workspace capacity exhausted"));
    // After this boundary preserve a possibly accepted original H2D prefix.
    try {
      status=cuda::status(slot.mla?
          native::plan_mla(slot.plan,spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),lengths.data(),*floating,workspace.float_workspace.size(),*integer,host_pointer,workspace.int_workspace.size(),*stream,facts):
          native::plan_batch(slot.plan,spec,metadata.query_indptr.data(),metadata.kv_indptr.data(),*floating,workspace.float_workspace.size(),*integer,host_pointer,workspace.int_workspace.size(),*stream,facts));
      if(status.ok()){slot.kv_chunk_rows=facts.kv_chunk_rows;slot.spec=spec;slot.generation=layout.generation();slot.layout=layout;*report={.layout_generation=layout.generation(),.float_workspace_bytes=facts.float_bytes,.int_workspace_bytes=facts.int_bytes,.cta_tile_q=facts.cta_tile_q,.padded_batch_size=facts.padded_batch_size,.split_kv=facts.split_kv,.bf16_partials=facts.bf16_partials};slot.report=*report;}
      return spec.query_rows==0 && status.ok()?detail::EnqueueResult::no_op():detail::EnqueueResult::accepted_command(status);
    }catch(const std::bad_alloc&){return detail::EnqueueResult::accepted_command(Status::ResourceExhausted("FlashInfer batch planning host allocation"));}
     catch(const std::exception&e){return detail::EnqueueResult::accepted_command(Status::Internal(e.what()));}
  }catch(const std::bad_alloc&){return detail::EnqueueResult::rejected(Status::ResourceExhausted("FlashInfer batch workspace query host allocation"));}
   catch(const std::exception&e){return detail::EnqueueResult::rejected(Status::Internal(e.what()));}
}
detail::EnqueueResult run_batch_slot(void*opaque,detail::QueueSequence &sequence,std::span<const ConstTensorView>inputs,
                                   detail::BatchWorkspaceViews workspace)noexcept {
  auto&slot=*static_cast<BatchSlot*>(opaque);
  if(!slot.generation || inputs.size()!=(slot.mla?2u:3u))return detail::EnqueueResult::rejected(Status::FailedPrecondition("FlashInfer batch slot has no prepared layout"));
  auto stream=cuda::stream(sequence);if(!stream)return detail::EnqueueResult::rejected(stream.error());
  auto tail=detail::batch_output_tail(slot.layout.origin_schema().capacity(),slot.spec.query_rows,workspace.output);
  if(!tail)return detail::EnqueueResult::rejected(tail.error());
  Result<void*> tail_pointer=static_cast<void*>(nullptr);
  if(!tail->empty())tail_pointer=cuda::detail::mutable_pointer(*tail,1,1);
  if(!tail_pointer)return detail::EnqueueResult::rejected(tail_pointer.error());
  const auto clear_tail=[&]()noexcept->Status{
    return tail->empty()?Status::Ok():cuda::status(cudaMemsetAsync(*tail_pointer,0,tail->size(),*stream));
  };
  if(slot.spec.query_rows==0){
    if(tail->empty())return detail::EnqueueResult::no_op();
    cuda::detail::ScopedPrimaryContext scope(slot.primary);if(!scope)return detail::EnqueueResult::rejected(scope.status());
    // Empty active math still initializes the real output, without touching
    // Q/K/V, metadata or scratch. This uses the same admitted capture stream.
    return detail::EnqueueResult::accepted_command(clear_tail());
  }
  std::array<const void*,3> pointers{};
  for(std::size_t i=0;i<inputs.size();++i){auto p=cuda::detail::const_pointer(inputs[i].storage(),16,1);if(!p)return detail::EnqueueResult::rejected(p.error());pointers[i]=*p;}
  auto metadata=cuda::detail::const_pointer(mem::ConstBufferView(workspace.metadata),8,1);if(!metadata)return detail::EnqueueResult::rejected(metadata.error());
  Result<void*> floating=static_cast<void*>(nullptr);
  if(workspace.float_workspace.valid())floating=cuda::detail::mutable_pointer(workspace.float_workspace,16,1);
  if(!floating)return detail::EnqueueResult::rejected(floating.error());
  auto integer=cuda::detail::mutable_pointer(workspace.int_workspace,16,1);if(!integer)return detail::EnqueueResult::rejected(integer.error());
  auto output=cuda::detail::mutable_pointer(workspace.output,16,1);if(!output)return detail::EnqueueResult::rejected(output.error());
  cuda::detail::ScopedPrimaryContext scope(slot.primary);if(!scope)return detail::EnqueueResult::rejected(scope.status());
  try{if(slot.mla){
      auto status=cuda::status(native::run_mla(slot.plan,slot.spec,pointers[0],pointers[1],*metadata,*output,*floating,*integer,*stream));
      return detail::EnqueueResult::accepted_command(status.ok()?clear_tail():status);
    }
    const auto run=slot.spec.qk_head_dim==512?&native::run_highdim_batch:&native::run_batch;
    auto status=cuda::status(run(slot.plan,slot.spec,pointers[0],pointers[1],pointers[2],*metadata,*output,*floating,*integer,*stream));
    // A native failure keeps its accepted prefix and does not enqueue more
    // work. Successful math and the optional exact suffix share one stream.
    return detail::EnqueueResult::accepted_command(status.ok()?clear_tail():status);}
  catch(const std::bad_alloc&){return detail::EnqueueResult::accepted_command(Status::ResourceExhausted("FlashInfer batch run host allocation"));}
  catch(const std::exception&e){return detail::EnqueueResult::accepted_command(Status::Internal(e.what()));}
}
Result<bool> update_batch_slot(void*opaque,const BatchLayoutVersion&layout,BatchPlanReport*report)noexcept{
  auto&slot=*static_cast<BatchSlot*>(opaque);
  if(!can_reuse_batch_plan(slot,layout))return false;
  slot.workspace_query={};
  slot.layout=layout;slot.generation=layout.generation();*report=slot.report;report->layout_generation=slot.generation;report->reused_native_plan=true;
  slot.report=*report;return true;
}
ProviderRuntimeProperties batch_runtime(void *) noexcept {
  return {.mode=ProviderRuntimeMode::kSharedQueue,
      .workspace_owner=ProviderWorkspaceOwner::kProvider,
      .replay_class=ProviderReplayClass::kManaged,.blocks_host=false,
      .uses_private_threads=false,.supports_concurrent_invocations=true,
      // update_batch_slot changes only retained runtime-metadata contents
      // and host layout/generation. The exact original plan/launch topology
      // stays fixed; fresh plans and workspace/carrier changes are gated by
      // the common cold preparation bridge before their owners are replaced.
      .capture_compatible=true};
}
const detail::BatchProviderVTable kBatchVTable{.qualify=&qualify_batch,.create=&create_batch_slot,.destroy=&destroy_batch_slot,
  .update=&update_batch_slot,.plan=&plan_batch_slot,.run=&run_batch_slot,.supports_command_capture=true,.workspace=&batch_workspace,
  .runtime=&batch_runtime};
const detail::BatchProviderVTable kMlaVTable{.qualify=&qualify_mla,.create=&create_mla_slot,.destroy=&destroy_batch_slot,
  .update=&update_batch_slot,.plan=&plan_batch_slot,.run=&run_batch_slot,.supports_command_capture=true,.workspace=&batch_workspace,
  .runtime=&batch_runtime};
const void*query_extension(void*,ProviderExtensionId id)noexcept{
  if(id==detail::kBatchPlanningExtensionId)return &kBatchVTable;
  if(id==detail::kMlaBatchPlanningExtensionId)return &kMlaVTable;
  return nullptr;
}

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
  if (attention->scale <= 0)
    return Status::Unimplemented("FlashInfer attention requires positive QK scale");
  if (attention->scale > std::numeric_limits<float>::max()/1.4426950408889634F)
    return Status::Unimplemented("FlashInfer attention QK log2 scale exceeds native float");
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

detail::EnqueueResult submit(void *state, void *lane, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<Prepared *>(state);
  auto &invocation = *static_cast<Invocation *>(lane);
  if (invocation.failed)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "FlashInfer invocation failed; create another"));
  auto stream = cuda::stream(sequence);
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
    unsigned *word=nullptr;auto initialized=cuda::detail::acquire_device_status(sequence,&word);
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
    return ProviderPrepareOutcome::error(
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
