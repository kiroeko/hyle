#pragma once
#include "hyle/graph/executable.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/paged_arena.h"
#include <array>
#include <cstdint>
#include <string_view>

namespace hyle::graph::batch {
// Discover ordinary and typed batch candidates from these same real sources;
// the selected implementation owns its cold metadata and original slots.
HYLE_GRAPH_BATCH_API Result<CompiledGraph> compile(const Graph &, CompileEnvironment,
    CompileOptions = {}) noexcept;
struct NodeLayout final {NodeId node;compute::BatchLayoutVersion layout;};
// Cold native planning uses this same ordinary Graph invocation. Geometry and
// attention mathematics come from the typed node; these are resource limits,
// never a second copy of scale/causal/window/precision policy.
struct PrepareOptions final {
  mem::MemoryResourceRef host_workspace = mem::system_resource();
  std::size_t max_float_workspace_bytes = 64u * 1024u * 1024u;
  std::size_t max_int_workspace_bytes = 8u * 1024u * 1024u;
  mem::AllocatorRef allocator = mem::system_allocator();
  compute::ResourceBudget resource_budget{};
  bool update = true;
  // Producer plus accepted copy phases; zero keeps pending controls absent.
  std::size_t max_source_completions = 0;
};
HYLE_GRAPH_BATCH_API Status prepare(graph::Invocation &,
    std::span<const QueueBinding>, std::span<const NodeLayout>,
    PrepareOptions = {}) noexcept;
HYLE_GRAPH_BATCH_API Result<compute::BatchPlanReport> report(
    graph::Invocation &, NodeId) noexcept;
HYLE_GRAPH_BATCH_API Result<compute::NativePagedBatchReport> native_report(
    graph::Invocation &, NodeId) noexcept;
// Optional cold capture inside the already selected native slots. Bind every
// ordinary tensor first; the enclosing Invocation still owns replay/submit.
HYLE_GRAPH_BATCH_API Status prepare_replay(graph::Invocation &,
    std::span<const QueueBinding>) noexcept;
namespace detail {struct RowCacheAccess;}
// A small macro over two original ordinary Graph nodes: private request/page
// address translation followed by the existing checked row codec. IDs may be
// produced on device (for example by TopK). Only its typed preparation may
// bind the truthful physical capacity and owned metadata input.
class PagedRowCacheGather final {
public:
  PagedRowCacheGather()noexcept=default;
  bool valid()const noexcept{return schema_.valid() && decode_.valid();}
  ValueId bank_input()const noexcept{return bank_;}
  ValueId output()const noexcept{return output_;}
  NodeId node()const noexcept{return decode_;}
  compute::RowCacheBatchSchema schema()const noexcept{return schema_;}
private:
  friend struct detail::RowCacheAccess;
  friend Result<PagedRowCacheGather> add_row_cache_gather(GraphBuilder&,compute::RowCacheBatchSchema,ValueId)noexcept;
  compute::RowCacheBatchSchema schema_;
  NodeId map_,decode_;ValueId bank_,metadata_,ids_,mapped_,output_;
};
HYLE_GRAPH_BATCH_API Result<PagedRowCacheGather> add_row_cache_gather(
    GraphBuilder&,compute::RowCacheBatchSchema,ValueId logical_ids)noexcept;
// Padding-aware macro: only -1 or a legal score column outside its query's
// request-local [begin,end) becomes padding. It returns a 0/-Inf mask along
// with selected BF16 rows. Invalid columns/ranges/metadata still fail checked
// decode. Include reader() in the original invocation's row binding set.
class PagedRowCacheGatherPadded final {
public:
  bool valid()const noexcept{return reader_.valid();}
  const PagedRowCacheGather& reader()const noexcept{return reader_;}
  ValueId bank_input()const noexcept{return reader_.bank_input();}
  ValueId output()const noexcept{return reader_.output();}
  ValueId mask()const noexcept{return mask_;}
  NodeId node()const noexcept{return reader_.node();}
private:
  friend struct detail::RowCacheAccess;
  friend Result<PagedRowCacheGatherPadded> add_row_cache_gather_padded(GraphBuilder&,compute::RowCacheBatchSchema,ValueId,ValueId,std::uint32_t)noexcept;
  PagedRowCacheGather reader_;ValueId mask_;
};
HYLE_GRAPH_BATCH_API Result<PagedRowCacheGatherPadded> add_row_cache_gather_padded(
    GraphBuilder&,compute::RowCacheBatchSchema,ValueId logical_ids,ValueId ranges,std::uint32_t score_columns)noexcept;
// Bounded-tile FP4 row-inline scoring over exact typed page permissions.
// Five original ordinary-node ports are Q/physical bank/weights/ranges/owned
// metadata. Q/weights and rounding match RowCacheIndexerScores. Output [T,C]
// columns and ranges are request-local logical rows; C is a static score width
// covering each active request history, not a tile option or physical capacity.
class PagedRowCacheIndexer final {
public:
  bool valid()const noexcept{return reader_.valid();}
  ValueId bank_input()const noexcept{return reader_.bank_input();}
  ValueId output()const noexcept{return reader_.output();}
  NodeId node()const noexcept{return reader_.node();}
  const PagedRowCacheGather& reader()const noexcept{return reader_;}
  compute::RowCacheBatchSchema schema()const noexcept{return reader_.schema();}
private:
  friend struct detail::RowCacheAccess;
  friend Result<PagedRowCacheIndexer> add_row_cache_indexer(GraphBuilder&,compute::RowCacheBatchSchema,ValueId,ValueId,ValueId,std::uint32_t)noexcept;
  PagedRowCacheGather reader_;
};
HYLE_GRAPH_BATCH_API Result<PagedRowCacheIndexer> add_row_cache_indexer(
    GraphBuilder&,compute::RowCacheBatchSchema,ValueId query,ValueId weights,ValueId ranges,std::uint32_t score_columns)noexcept;
// As with the original Builder calls, a failure may follow successful input/
// node appends. Recreate the builder before retrying a failed macro.
struct RowCacheBinding final {
  PagedRowCacheGather reader;
  compute::RowCacheLayoutVersion layout;
};
// The immutable source is borrowed only for cold macro/private-port validation.
HYLE_GRAPH_BATCH_API Status prepare(graph::Invocation &, const Graph &,
    std::span<const QueueBinding>, std::span<const NodeLayout>,
    std::span<const RowCacheBinding>, PrepareOptions = {}) noexcept;
// Idle ready frames may replace a ready frame of the same prepared owned
// layout. Rejection preserves the old binding/replay; success requires the
// ordinary prepare_replay again. Pending source episodes require reset.
HYLE_GRAPH_BATCH_API Status bind(graph::Invocation &, ValueId, compute::PagedArenaRead) noexcept;
HYLE_GRAPH_BATCH_API Status bind(graph::Invocation &, ValueId, compute::PagedArenaPendingRead) noexcept;
HYLE_GRAPH_BATCH_API Status bind(graph::Invocation &, ValueId, compute::PagedRowCacheRead) noexcept;
HYLE_GRAPH_BATCH_API Status bind(graph::Invocation &, ValueId, compute::PagedRowCachePendingRead) noexcept;
// One source joins the original Graph queue sequence and public completion.
// Before acceptance, failure returns Update to its caller; accepted prefixes
// retain their true source controls and resource owners through retirement.
HYLE_GRAPH_BATCH_API Result<compute::Completion> submit(graph::Invocation &,
    std::span<const QueueBinding>, compute::CopyBatchUpdate &&) noexcept;
HYLE_GRAPH_BATCH_API Result<compute::CopyBatchUpdate> take_update(graph::Invocation &) noexcept;
// New attention cache rows use their schema's exact physical carrier for
// logical [R,KV-heads,depth], with blocks only inside the final depth axis.
// Destinations are independent page writers supplied at the cold bind boundary,
// never a mutable bank tensor.
// Append has no tensor output and must explicitly precede every cache reader
// with GraphBuilder::depends_on(reader, append). External writes are ordered
// by the ordinary Graph effect rules, including between distinct append nodes.
// R is inferred from the canonical one-row carrier and the complete input
// description, including strides. Q8_0 uses U8[R*kv_heads, block_row_bytes], not
// one carrier row per token. Native copy and reader capabilities are separate;
// this operation performs no encoding or dtype conversion and excludes RowCache.
struct PagedAppend final {
  compute::BatchSchema schema;
  compute::BatchCacheRole role=compute::BatchCacheRole::kKey;
  static constexpr compute::OperationId operation_id{
      0x48594c4547524241ULL,0x5041474544570001ULL};
  static constexpr std::string_view operation_name="paged-append";
  constexpr compute::OperationTraits operation_traits()const noexcept{
    return {{1,1},0,compute::AliasPolicy::kNone,compute::EffectKind::kExternalSideEffect,
        compute::PrecisionPolicy::kExact,compute::DataType::kUnknown};
  }
  HYLE_GRAPH_BATCH_API Status infer_output_descriptions(
      std::span<const compute::TensorDesc>,std::span<compute::TensorDesc>)const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<PagedAppend>);
HYLE_GRAPH_BATCH_API Result<PagedAppend> paged_append(compute::BatchSchema,
     compute::BatchCacheRole)noexcept;
namespace detail {struct PagedAppendData;struct PagedAppendAccess;}
struct PagedAppendOptions final {
  mem::AllocatorRef allocator=mem::system_allocator();
  compute::ResourceBudget resource_budget{};
};
// Cold owned transaction: old exact ready pages, immutable next layout and
// original dirty-page reservations. Construction validates append/prefix-COW
// mappings before moving any writer. No native work is started by this owner.
// Control is thread-compatible; the original resource/allocator lifetime stays
// unchanged. While bound, the original Invocation owns this transaction.
class PagedAppendPages final {
public:
  PagedAppendPages()noexcept=default;
  PagedAppendPages(const PagedAppendPages&)=delete;
  PagedAppendPages&operator=(const PagedAppendPages&)=delete;
  HYLE_GRAPH_BATCH_API PagedAppendPages(PagedAppendPages&&)noexcept;
  HYLE_GRAPH_BATCH_API PagedAppendPages&operator=(PagedAppendPages&&)noexcept;
  HYLE_GRAPH_BATCH_API ~PagedAppendPages();
  HYLE_GRAPH_BATCH_API static Result<PagedAppendPages> create(compute::PagedArena,
      compute::BatchLayoutVersion next,compute::PagedArenaRead previous,
      std::span<compute::PagedPageWriter> dirty,PagedAppendOptions={})noexcept;
  // Opt-in bounded publication. Reserved original tail authorities append in
  // place; fresh/COW partial pages return a new unique tail after ready().
  // Every fallible cold check precedes consuming either caller span.
  HYLE_GRAPH_BATCH_API static Result<PagedAppendPages> create(compute::PagedArena,
      compute::BatchLayoutVersion next,compute::PagedArenaRead previous,
      std::span<compute::PagedPageWriter> dirty,
      std::span<compute::PagedPageAppender> tails,PagedAppendOptions={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_GRAPH_BATCH_API compute::BatchLayoutVersion layout()const noexcept;
  HYLE_GRAPH_BATCH_API std::size_t page_count()const noexcept;
  // Available only while the transaction is owned by the caller. Failed or
  // cancelled native uses must actually retire before discard/reacquisition.
  HYLE_GRAPH_BATCH_API Result<compute::PagedPageWriter&> page(std::size_t)noexcept;
  HYLE_GRAPH_BATCH_API std::size_t appender_count()const noexcept;
  // Caller-owned only, including retired failure recovery. Successful ready()
  // leaves idle tails here for the continuation; snapshots copy no tail right.
  HYLE_GRAPH_BATCH_API Result<compute::PagedPageAppender&> appender(std::size_t)noexcept;
  // Only take_append after a successful whole Graph authorizes publication.
  // A failed ready-frame allocation may be retried; old snapshots stay ready.
  HYLE_GRAPH_BATCH_API Result<compute::PagedArenaRead> ready()noexcept;
private:
  friend struct detail::PagedAppendAccess;
  static Result<PagedAppendPages> create_impl(compute::PagedArena,
      compute::BatchLayoutVersion,compute::PagedArenaRead,
      std::span<compute::PagedPageWriter>,std::span<compute::PagedPageAppender>,
      bool appendable,PagedAppendOptions)noexcept;
  explicit PagedAppendPages(detail::PagedAppendData*p)noexcept:data_(p){}
  detail::PagedAppendData*data_=nullptr;
};
// Requires batch::prepare first. Failure leaves pages owned by the caller.
// The original Graph/selected artifacts prove write-before-read and admit only
// exact dirty pages to its existing lease-use/Completion lifetime.
HYLE_GRAPH_BATCH_API Status bind_append(Invocation&,const Graph&,NodeId append,
    ValueId cache_input,PagedAppendPages&&)noexcept;
// External-write-only variant, with no same-Graph cache reader or bank input.
// Requires batch::prepare with the selected append lane's original Queue.
// The exact page reads/writes and native controls join the original Graph's
// lease-use/Completion episode. No read permission or whole-bank write is made.
HYLE_GRAPH_BATCH_API Status bind_append(Invocation&,const Graph&,NodeId append,
     PagedAppendPages&&)noexcept;
// Retired success, failure or cancellation; no implicit wait. Revokes this
// binding/preparation before returning the real original page reservations.
HYLE_GRAPH_BATCH_API Result<PagedAppendPages> take_append(Invocation&,NodeId)noexcept;
// Atomically withdraw these original append transactions after actual whole-
// Graph retirement. Both spans have equal length and destinations are empty.
// Every node/owner is checked before any transaction is consumed. Failure or
// cancellation returns original reservations but never grants ready pages.
// Revokes replay/output observation exactly as take_append; no implicit wait.
HYLE_GRAPH_BATCH_API Status take_appends(Invocation&,std::span<const NodeId>,
    std::span<PagedAppendPages>)noexcept;
// Mathematical policy and an owned initial layout of one ordinary Graph node.
// This layout-only form uses its origin schema's cache interpretation.
// Storage variants use the typed compute::PagedAttention/PagedMlaAttention
// capacity form with the same owned metadata version; page rights stay separate.
// Implementations derive their own metadata and native plans during cold
// preparation. Page permissions remain separate from these immutable facts.
struct PagedAttentionOptions final {
  float scale = 1.0F;
  bool causal = true;
  std::int32_t window_left = -1;
  // Permit reduced split-KV intermediate storage; local accumulation is F32.
  // An implementation may retain the default, more accurate intermediates.
  bool allow_bf16_partials = false;
};
struct PagedAttention final {
  compute::BatchLayoutVersion layout;
  PagedAttentionOptions options;
  static constexpr compute::OperationId operation_id{
      0x48594c4547524241ULL, 0x5041474544410001ULL};
  static constexpr std::string_view operation_name = "paged-attention";
  constexpr compute::OperationTraits operation_traits() const noexcept {
    return {{3,3},1,compute::AliasPolicy::kNone,compute::EffectKind::kPure,
            options.allow_bf16_partials ? compute::PrecisionPolicy::kAccumulateInDeclaredType :
                compute::PrecisionPolicy::kExact,compute::DataType::kFloat32};
  }
  HYLE_GRAPH_BATCH_API Status infer_output_descriptions(
      std::span<const compute::TensorDesc>,std::span<compute::TensorDesc>) const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<PagedAttention>);
HYLE_GRAPH_BATCH_API Result<PagedAttention> paged_attention(
    compute::BatchLayoutVersion, PagedAttentionOptions = {}) noexcept;
struct PagedMlaAttention final {
  compute::BatchLayoutVersion layout;
  PagedAttentionOptions options;
  static constexpr compute::OperationId operation_id{0x48594c4547524241ULL,0x50414745444d0001ULL};
  static constexpr std::string_view operation_name="paged-mla-attention";
  constexpr compute::OperationTraits operation_traits()const noexcept{
    return {{2,2},1,compute::AliasPolicy::kNone,compute::EffectKind::kPure,
        options.allow_bf16_partials ? compute::PrecisionPolicy::kAccumulateInDeclaredType :
            compute::PrecisionPolicy::kExact,compute::DataType::kFloat32};
  }
  HYLE_GRAPH_BATCH_API Status infer_output_descriptions(
      std::span<const compute::TensorDesc>,std::span<compute::TensorDesc>)const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<PagedMlaAttention>);
HYLE_GRAPH_BATCH_API Result<PagedMlaAttention> paged_mla_attention(
    compute::BatchLayoutVersion, PagedAttentionOptions = {})noexcept;
struct MaterializedGraph final {
  Graph graph;
  std::array<ValueId,5> inputs{};
  ValueId output;
};
// Explicitly selected reference recipe over the original ordinary Graph.
// Native route compile/prepare still qualifies each operation and its cost.
// Bind owned same-version indices/mask; upload them with original F2 first.
HYLE_GRAPH_BATCH_API Result<MaterializedGraph> materialized_graph(
    const compute::MaterializedBatchProgram&,GraphBuilderOptions = {}) noexcept;
namespace detail {struct MaterializedInvocationData;}
struct MaterializedInvocationOptions final {
  // One cold output backing, including alignment padding; explicit byte cap.
  std::size_t max_output_bytes=0;
  CompileOptions compile{};
  ResourceOptions resources{};
};
// Host/system-memory reference over the original shared-Queue Graph. Owns the
// fixed recipe/compiled graph, exact metadata generation and one output slot.
// Native GPU reference uses its own typed resources and preparation contract.
// Control is thread-compatible. Borrowed Context/Queue and injected allocators
// outlive native work/results; readonly metadata and output leases retain their
// original owners. Independent controllers have separate compiled slots.
class MaterializedInvocation final {
public:
  MaterializedInvocation()noexcept=default;
  MaterializedInvocation(const MaterializedInvocation&)=delete;
  MaterializedInvocation&operator=(const MaterializedInvocation&)=delete;
  HYLE_GRAPH_BATCH_API MaterializedInvocation(MaterializedInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API MaterializedInvocation&operator=(MaterializedInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API ~MaterializedInvocation();
  HYLE_GRAPH_BATCH_API static Result<MaterializedInvocation> create(
      const compute::Context&,compute::MaterializedBatchProgram,compute::Queue&,
      MaterializedInvocationOptions)noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  // Explicit update boundary. Old results/completions/storage/native reads
  // prevent output reuse. Repeated prepare before execution is supported.
  HYLE_GRAPH_BATCH_API Status prepare_batch(compute::BatchLayoutVersion)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(compute::ReadLease query,compute::PagedArenaRead keys,
      compute::PagedArenaRead values)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(compute::ReadLease query,compute::ReadLease keys,
      compute::ReadLease values)noexcept;
  // Consumes this prepared episode; each subsequent run needs prepare/bind.
  HYLE_GRAPH_BATCH_API Result<compute::Completion> submit()noexcept;
  HYLE_GRAPH_BATCH_API Result<GraphOutput> output()noexcept;
  HYLE_GRAPH_BATCH_API compute::MaterializedBatchVersion version()const noexcept;
  HYLE_GRAPH_BATCH_API Status close()noexcept;
  HYLE_GRAPH_BATCH_API Status drain()noexcept;
private:
  explicit MaterializedInvocation(detail::MaterializedInvocationData*p)noexcept:data_(p){}
  detail::MaterializedInvocationData*data_=nullptr;
};
} // namespace hyle::graph::batch
