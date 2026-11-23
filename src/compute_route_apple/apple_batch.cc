#include "hyle/compute/apple_batch.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/apple_plan.h"
#include "apple_paged_three_rows.h"
#include "apple_runtime_internal.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <new>
#include <optional>
#include <cmath>

namespace hyle::compute::apple {
bool detail::PagedBatchAccess::cache_domain(const BatchCapacity& c) noexcept {
  const bool plain=c.key_storage==CacheStorage::plain(c.query_type).value_or(CacheStorage{}) &&
      c.value_storage==c.key_storage;
  const bool q8=c.key_storage==CacheStorage::q8_0().value() && c.value_storage==c.key_storage;
  const bool compact_shape=c.kv_heads==1 && c.value_head_dim==512 &&
      (c.qk_head_dim==512 || c.qk_head_dim==576) && c.key_storage==c.value_storage;
  const bool e4=(c.storage==BatchStorage::kPaged || compact_shape) &&
      c.key_storage.data_type()==DataType::kFloat8E4M3FN &&
      c.value_storage.data_type()==DataType::kFloat8E4M3FN;
  return (plain || q8 || e4) && c.output_type==c.query_type &&
      (c.query_type==DataType::kFloat16 || c.query_type==DataType::kBFloat16) &&
      ((c.qk_head_dim==c.value_head_dim &&
        (c.qk_head_dim==64 || c.qk_head_dim==128 || c.qk_head_dim==256)) ||
       (c.qk_head_dim==64 && c.value_head_dim==128) ||
       ((plain || (e4 && compact_shape)) && (c.qk_head_dim==512 || c.qk_head_dim==576) &&
        c.value_head_dim==512));
}
Status detail::PagedBatchAccess::validate_paged(
    const std::shared_ptr<Runtime>& runtime,const BatchLayoutVersion& layout,
    BatchCapacity capacity,bool causal,std::int32_t window_left) noexcept {
  if(!runtime || !layout.valid())
    return Status::InvalidArgument("Apple page validation needs its actual Runtime and owned layout");
  if(!compute::detail::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(),capacity))
    return Status::InvalidArgument("Apple page topology differs from its actual cache recipe");
  const auto c=capacity;
  if(c.storage!=BatchStorage::kPaged || c.page_size!=16 ||
      c.max_pages>std::uint32_t(INT32_MAX/16) || !cache_domain(c))
    return Status::Unimplemented("Apple page dtype/encoding/geometry is unsupported");
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,causal,window_left));
  const auto host=layout.host();
  for(std::size_t r=0;r<layout.facts().requests;++r){
    const auto nq=host.query_indptr[r+1]-host.query_indptr[r];
    const auto refs=host.kv_indptr[r+1]-host.kv_indptr[r];
    const auto nk=refs?std::uint64_t(refs-1)*c.page_size+std::uint32_t(host.last_page_lengths[r]):0;
    if(nq && (!nk || nk>INT32_MAX))
      return Status::Unimplemented("Apple nonempty page query needs bounded nonempty KV");
  }
  return Status::Ok();
}

namespace {
int cache_width(CacheStorage storage,std::uint32_t depth) noexcept {
  return int(storage==CacheStorage::q8_0().value()?depth/32*34:depth);
}

mlx::core::array grouped_decode_attention(
    const mlx::core::array& query, const mlx::core::array& keys, const mlx::core::array& values,
    const mlx::core::array& pages, int history, int first_page, float scale, mlx::core::Stream stream) {
  namespace mx = mlx::core;
  constexpr int heads = 8, kv_heads = 2, width = 256, lanes = 32;
  constexpr int partitions = detail::paged_vector_partitions;
  // Keep the native vector traversal and sum order for each query head.
  // The four heads in each group consume the same K/V values.
  static const auto partial = mx::fast::metal_kernel("hyle_four_head_vector_partials",
      {"query", "keys", "values", "pages", "config"}, {"numerator", "stats"}, R"METAL(
    constexpr uint group = 4, width = 256;
    const uint partition = threadgroup_position_in_grid.x;
    const uint kv_head = threadgroup_position_in_grid.y;
    const uint first_head = kv_head * group;
    const uint lane = thread_index_in_simdgroup;
    const uint channel = lane * 8;
    const int length = config[0];
    const float scale = as_type<float>(config[2]);
    float q[group][8], accum[group][8] = {};
    float largest[group], sums[group] = {};
    for (uint head = 0; head < group; ++head) {
      largest[head] = -FLT_MAX;
      for (uint d = 0; d < 8; ++d)
        q[head][d] = scale * float(query[(first_head + head) * width + channel + d]);
    }
    for (int token = int(partition); token < length; token += PARTITIONS) {
      const uint physical = uint(pages[config[1] + token / 16]) * 16 + token % 16;
      const ulong base = (ulong(physical) * KV_HEADS + kv_head) * width + channel;
      float score[group] = {}, value[8];
      for (uint d = 0; d < 8; ++d) {
        const float key = float(keys[base + d]);
        value[d] = float(values[base + d]);
        for (uint head = 0; head < group; ++head) score[head] += q[head][d] * key;
      }
      for (uint head = 0; head < group; ++head) {
        const float logit = simd_sum(score[head]);
        const float next = max(largest[head], logit);
        const float previous = fast::exp(largest[head] - next);
        const float weight = fast::exp(logit - next);
        largest[head] = next;
        sums[head] = sums[head] * previous + weight;
        for (uint d = 0; d < 8; ++d)
          accum[head][d] = accum[head][d] * previous + weight * value[d];
      }
    }
    for (uint head = 0; head < group; ++head) {
      const uint slice = (first_head + head) * PARTITIONS + partition;
      for (uint d = 0; d < 8; ++d) numerator[slice * width + channel + d] = accum[head][d];
      if (lane == 0) {
        stats[slice * 2] = largest[head];
        stats[slice * 2 + 1] = sums[head];
      }
    }
    )METAL");
  static const auto merge = mx::fast::metal_kernel("hyle_four_head_vector_merge",
      {"numerator", "stats"}, {"output"}, detail::paged_vector_merge_source);
  const std::array<std::int32_t, 3> config{history, first_page, std::bit_cast<std::int32_t>(scale)};
  const auto parts = partial({query, keys, values, pages, mx::array(config.data(), {3}, mx::int32)},
      {{heads,partitions,width},{heads,partitions,2}}, {mx::float32,mx::float32},
      {partitions * lanes,kv_heads,1}, {lanes,1,1}, {{"PARTITIONS",partitions},{"KV_HEADS",kv_heads}},
      std::nullopt, false, stream);
  return merge(parts, {{1,heads,width}}, {mx::bfloat16}, {heads * lanes,lanes,1}, {lanes,lanes,1},
      {{"PARTITIONS",partitions}}, std::nullopt, false, stream).front();
}

mlx::core::array dense_prefill_cache(const mlx::core::array& cache, const mlx::core::array& page_ids,
    int history, int heads, int width, mlx::core::Stream stream) {
  namespace mx = mlx::core;
  const auto physical = mx::reshape(cache, {-1, 16, heads, width}, stream);
  const auto ordered = mx::take(physical, page_ids, 0, stream);
  const auto rows = mx::reshape(ordered, {1, -1, heads, width}, stream);
  return mx::transpose(mx::slice(rows, {0, 0, 0, 0}, {1, history, heads, width}, stream),
      {0, 2, 1, 3}, stream);
}

std::vector<mlx::core::array> aligned_prefill_cache(
    const mlx::core::array& keys, const mlx::core::array& values, const mlx::core::array& page_ids,
    int history, int heads, mlx::core::Stream stream) {
  namespace mx = mlx::core;
  constexpr int page_size = 16, width = 256, vectors = width / 4;
  // Materialize only one owned pair. Never read beyond the published rows of
  // the last page, including when the physical page belongs to a shared cache.
  static const auto gather = mx::fast::metal_kernel("hyle_aligned_prefill_cache",
      {"keys", "values", "pages", "shape"}, {"key_cache", "value_cache"}, R"METAL(
    const uint column = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint head = thread_position_in_grid.z;
    const ulong target = (ulong(head) * shape[1] + row) * VECTORS + column;
    ushort4 key(0), value(0);
    if (row < uint(shape[0])) {
      const ulong physical_row = ulong(pages[row / 16]) * 16 + row % 16;
      const ulong source = (physical_row * HEADS + head) * VECTORS + column;
      key = reinterpret_cast<device const ushort4*>(keys)[source];
      value = reinterpret_cast<device const ushort4*>(values)[source];
    }
    reinterpret_cast<device ushort4*>(key_cache)[target] = key;
    reinterpret_cast<device ushort4*>(value_cache)[target] = value;
    )METAL");
  const int storage_rows = int(page_ids.size()) * page_size;
  const std::array<std::int32_t,2> shape{history,storage_rows};
  const mx::Shape output{1,heads,storage_rows,width};
  return gather({keys,values,page_ids,mx::array(shape.data(),{2},mx::int32)},
      {output,output}, {mx::bfloat16,mx::bfloat16}, {vectors,storage_rows,heads},
      {vectors,2,1}, {{"VECTORS",vectors},{"HEADS",heads}}, std::nullopt, false, stream);
}
}
Result<Program> detail::PagedBatchAccess::prepare_program(const Context& context,
    const NativePagedBatchProgram& recipe, BatchLayoutVersion layout,
    const WorkloadContext& workload) noexcept {
  return prepare_program(ContextAccess::runtime(context),recipe,std::move(layout),workload);
}
Result<Program> detail::PagedBatchAccess::prepare_program(std::shared_ptr<Runtime> runtime,
    const NativePagedBatchProgram& recipe, BatchLayoutVersion layout,
    const WorkloadContext& workload) noexcept {
  const auto query_rows=layout.facts().query_rows;
  return prepare_paged_graph(std::move(runtime),recipe,std::move(layout),query_rows,workload);
}
Result<Program> detail::PagedBatchAccess::prepare_paged_graph(std::shared_ptr<Runtime> runtime,
    const NativePagedBatchProgram& recipe, BatchLayoutVersion layout,
    std::uint32_t query_rows,const WorkloadContext& workload) noexcept {
  if(!runtime || !recipe.valid() || !recipe.schema().owns(layout))
    return std::unexpected(Status::InvalidArgument("Apple paged context/layout/schema differs"));
  const auto c=recipe.schema().capacity();
  if(query_rows>c.max_query_rows || layout.facts().query_rows>query_rows)
    return std::unexpected(Status::InvalidArgument("Apple page active queries exceed concrete Q/output extent"));
  const bool mla=recipe.schema().mla_layout()!=MlaCacheLayout::kNone;
  if(c.key_storage.data_type()==DataType::kFloat8E4M3FN && c.value_head_dim==512 && !mla)
    return std::unexpected(Status::Unimplemented("Apple high-dimensional E4 needs its single compact MLA cache"));
  HYLE_RETURN_IF_ERROR(validate_paged(runtime,layout,recipe.schema().capacity(),recipe.causal(),recipe.window_left()));
  try {
    std::array<InputSpec,4> specs;
    for(std::size_t i=0;i<recipe.input_count();++i){
      HYLE_ASSIGN_OR_RETURN(auto description,recipe.input_description(layout,i));
      if(!i){HYLE_ASSIGN_OR_RETURN(description,batch_query_description(c,query_rows));}
      HYLE_ASSIGN_OR_RETURN(specs[i],input_spec(description));
    }
    if(layout.facts().query_rows==0 && layout.facts().page_refs==0){
      return prepare_runtime_program(runtime,layout,std::span(specs).first(1),
          [heads=int(c.query_heads),depth=int(c.value_head_dim),query_rows](const BatchLayoutVersion&,
          const std::vector<mlx::core::array>& inputs,
          mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
        if(inputs.size()!=1)return std::unexpected(Status::InvalidArgument("empty Apple page episode needs only Q"));
        if(query_rows)return std::vector{mlx::core::zeros({int(query_rows),heads,depth},inputs[0].dtype(),stream)};
        HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));
        HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));
        return std::vector{mlx::core::reshape(sliced[0],{0,heads,depth},stream)};
      },workload,recipe.causal(),recipe.window_left());
    }
    const bool plain_bf16 = !mla && recipe.causal() && recipe.window_left() < 0 &&
        c.qk_head_dim == 256 && c.value_head_dim == 256 && c.query_type == DataType::kBFloat16 &&
        c.key_storage == CacheStorage::plain(c.query_type).value() && c.value_storage == c.key_storage;
    const auto& architecture = runtime->environment.architecture;
    // This GPU class uses 64 partitions for the native paged GQA vector path.
    const bool share_paged_kv = plain_bf16 && architecture.ends_with('g');
    const bool shared_three = share_paged_kv && c.query_heads > c.kv_heads;
    const bool grouped_decode = share_paged_kv && c.query_heads == 8 && c.kv_heads == 2;
    return prepare_runtime_program(runtime,layout,std::span(specs).first(mla?3:4),[recipe,query_rows,plain_bf16,shared_three,grouped_decode](const BatchLayoutVersion& layout,
        const std::vector<mlx::core::array>& inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      const bool mla=recipe.schema().mla_layout()!=MlaCacheLayout::kNone;
      if(inputs.size()!=(mla?3u:4u))return std::unexpected(Status::InvalidArgument("Apple paged native port count"));
      const auto c=recipe.schema().capacity();const auto host=layout.host();
      // Reshape and transpose are aliases of the truthful physical arena;
      // the private primitive owns the page table and logical row count.
      const mlx::core::Shape key_root{1,int(std::int64_t(c.max_pages)*16),int(c.kv_heads),cache_width(c.key_storage,c.qk_head_dim)};
      const mlx::core::Shape value_root{1,int(std::int64_t(c.max_pages)*16),int(c.kv_heads),cache_width(c.value_storage,c.value_head_dim)};
      const auto k=mlx::core::transpose(mlx::core::reshape(inputs[1],key_root,stream),{0,2,1,3},stream);
      // The value prefix is a strided alias of the same physical packed row.
      // The existing private kernel accepts independent K/V sequence pitches.
      const auto v=mla?mlx::core::slice(k,{0,0,0,0},{1,int(c.kv_heads),int(std::int64_t(c.max_pages)*16),512},stream):
          mlx::core::transpose(mlx::core::reshape(inputs[2],value_root,stream),{0,2,1,3},stream);
      std::vector<mlx::core::array> parts;
      for(std::size_t r=0;r<layout.facts().requests;++r){
        const auto nq=host.query_indptr[r+1]-host.query_indptr[r];if(!nq)continue;
        const auto refs=host.kv_indptr[r+1]-host.kv_indptr[r];
        const auto nk=std::uint64_t(refs-1)*16+std::uint32_t(host.last_page_lengths[r]);
        HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,host.query_indptr[r],nq));
        HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(slice),std::span(&inputs[0],1),stream));
        if (grouped_decode && nq == 1 && nk >= 8192 && nk <= 32768) {
          parts.push_back(grouped_decode_attention(sliced[0], inputs[1], inputs[2], inputs[3],
              int(nk), host.kv_indptr[r], recipe.scale(), stream));
          continue;
        }
        if (shared_three && nq == 3 && nk >= 4096 && nk <= 32768) {
          parts.push_back(detail::paged_three_row_attention(sliced[0], inputs[1], inputs[2], inputs[3],
              int(nk), host.kv_indptr[r], recipe.scale(), int(c.query_heads), int(c.kv_heads), stream));
          continue;
        }
        auto q=mlx::core::transpose(mlx::core::reshape(sliced[0],{1,nq,int(c.query_heads),int(c.qk_head_dim)},stream),{0,2,1,3},stream);
        // A prefill tile amortizes gathering the live BF16 pages. Bound the
        // two temporary dense caches to 64 MiB per request.
        const auto dense_cache_bytes = std::uint64_t(refs) * c.page_size * c.kv_heads *
            (c.qk_head_dim + c.value_head_dim) * sizeof(std::uint16_t);
        if (plain_bf16 && nq >= 128 && nq <= 512 && dense_cache_bytes <= 64u * 1024 * 1024) {
          namespace mx = mlx::core;
          const auto pages = mx::slice(inputs[3], {host.kv_indptr[r]}, {host.kv_indptr[r+1]}, stream);
          const auto result = [&] {
            if (nk % c.page_size != 0 && mx::fast::hyle_can_align_causal_prefill(q, stream)) {
              const auto reached = aligned_prefill_cache(inputs[1], inputs[2], pages, int(nk), int(c.kv_heads), stream);
              return mx::fast::hyle_padded_causal_prefill(q, reached[0], reached[1], int(nk), recipe.scale(), stream);
            }
            auto key = dense_prefill_cache(inputs[1], pages, int(nk), int(c.kv_heads), 256, stream);
            auto value = dense_prefill_cache(inputs[2], pages, int(nk), int(c.kv_heads), 256, stream);
            return mx::fast::scaled_dot_product_attention(q, key, value, recipe.scale(),
                "causal", std::nullopt, std::nullopt, true, stream);
          }();
          parts.push_back(mx::reshape(mx::transpose(result, {0,2,1,3}, stream),
              {nq, int(c.query_heads), 256}, stream));
          continue;
        }
        const bool e4=c.key_storage.data_type()==DataType::kFloat8E4M3FN;
        auto result=e4?mlx::core::fast::hyle_paged_attention(q,k,v,inputs[mla?2:3],int(nk),host.kv_indptr[r],recipe.scale(),recipe.causal(),stream,recipe.window_left(),c.key_storage.scale(),c.value_storage.scale()):
            mlx::core::fast::hyle_paged_attention(q,k,v,inputs[mla?2:3],int(nk),host.kv_indptr[r],recipe.scale(),recipe.causal(),stream,c.key_storage==CacheStorage::q8_0().value(),recipe.window_left());
        parts.push_back(mlx::core::reshape(mlx::core::transpose(result,{0,2,1,3},stream),{nq,int(c.query_heads),int(c.value_head_dim)},stream));
      }
      if(layout.facts().query_rows<query_rows)
        parts.push_back(mlx::core::zeros({int(query_rows-layout.facts().query_rows),
            int(c.query_heads),int(c.value_head_dim)},inputs[0].dtype(),stream));
      if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));return std::vector{mlx::core::reshape(sliced[0],{0,int(c.query_heads),int(c.value_head_dim)},stream)};}
      if(parts.size()==1)return parts;
      return lower(Operation(Concat{.axis=0}),parts,stream);
    },workload,recipe.causal(),recipe.window_left());
  } catch(const std::bad_alloc&) {
    return std::unexpected(Status::ResourceExhausted("Apple paged cold tracing metadata"));
  }
}
namespace {
Result<Program> prepare_ragged(const std::shared_ptr<detail::Runtime>&runtime, BatchLayoutVersion layout,
    RaggedAttention attention,const WorkloadContext&workload,
    std::optional<std::uint32_t> output_query_rows)noexcept{
  if(!layout.valid()||!compute::detail::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(),attention.capacity))
    return std::unexpected(Status::InvalidArgument("Apple ragged context/layout/domain differs"));
  const auto c=attention.capacity;
  const bool mla=layout.origin_schema().mla_layout()!=MlaCacheLayout::kNone;
  if(c.key_storage.data_type()==DataType::kFloat8E4M3FN && !mla)
    return std::unexpected(Status::Unimplemented("Apple ragged E4 needs its single compact MLA cache"));
  const auto query_rows=output_query_rows.value_or(c.max_query_rows);
  if(query_rows>c.max_query_rows || layout.facts().query_rows>query_rows)
    return std::unexpected(Status::InvalidArgument("Apple ragged active queries exceed concrete Q/output extent"));
  HYLE_RETURN_IF_ERROR(detail::PagedBatchAccess::validate_ragged(runtime,attention));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,attention.causal,attention.window_left));
  try{
    HYLE_ASSIGN_OR_RETURN(auto q,batch_query_description(c,query_rows));
    HYLE_ASSIGN_OR_RETURN(auto k,batch_cache_description(c,BatchCacheRole::kKey));
    std::array<InputSpec,3> inputs;HYLE_ASSIGN_OR_RETURN(inputs[0],input_spec(q));HYLE_ASSIGN_OR_RETURN(inputs[1],input_spec(k));
    if(!mla){HYLE_ASSIGN_OR_RETURN(auto v,batch_cache_description(c,BatchCacheRole::kValue));HYLE_ASSIGN_OR_RETURN(inputs[2],input_spec(v));}
    const auto host=layout.host();
    for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];
      if(nq&&(!nk||(attention.causal&&nk<nq)))return std::unexpected(Status::Unimplemented("Apple ragged nonempty Q needs sufficient KV"));}
    // The original MLX compiler retains the callable. Retain the immutable
    // owned version, never spans from the caller's metadata arrays.
    return detail::PagedBatchAccess::prepare_runtime_program(runtime,layout,std::span(inputs).first(mla?2u:3u),
        [attention,output_query_rows,mla](const BatchLayoutVersion& layout,const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=(mla?2u:3u))return std::unexpected(Status::InvalidArgument("Apple ragged payload count differs"));
      const auto c=attention.capacity;const auto host=layout.host();std::vector<mlx::core::array> parts;
      const auto view=[&](const mlx::core::array &input,std::int32_t start,std::int32_t rows,std::uint32_t heads,int width)->Result<mlx::core::array>{
        HYLE_ASSIGN_OR_RETURN(auto slice,View::slice(0,start,rows));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(slice),std::span(&input,1),stream));
        if(sliced[0].shape()!=mlx::core::Shape{rows,int(heads),width})
          return std::unexpected(Status::Internal("Apple ragged row view shape differs"));
        // Inserting an axis preserves the compact V prefix's CKV/KPE pitch.
        return mlx::core::expand_dims(sliced[0],0,stream);
      };
      for(std::size_t r=0;r<layout.facts().requests;++r){const auto nq=host.query_indptr[r+1]-host.query_indptr[r],nk=host.kv_indptr[r+1]-host.kv_indptr[r];if(!nq)continue;
        HYLE_ASSIGN_OR_RETURN(auto q,view(inputs[0],host.query_indptr[r],nq,c.query_heads,int(c.qk_head_dim)));
        const bool q8=c.key_storage==CacheStorage::q8_0().value();
        // U8 carriers are flattened by input_spec; first restore exact rows
        // and bytes, then slice only this request without decoding history.
        auto key_rows=mlx::core::reshape(inputs[1],{int(c.max_kv_rows),int(c.kv_heads),cache_width(c.key_storage,c.qk_head_dim)},stream);
        // Compact V aliases the CKV prefix with the original CKV/KPE row pitch.
        auto value_rows=mla?mlx::core::slice(key_rows,{0,0,0},{int(c.max_kv_rows),1,512},stream):
            mlx::core::reshape(inputs[2],{int(c.max_kv_rows),int(c.kv_heads),cache_width(c.value_storage,c.value_head_dim)},stream);
        HYLE_ASSIGN_OR_RETURN(auto k,view(key_rows,host.kv_indptr[r],nk,c.kv_heads,cache_width(c.key_storage,c.qk_head_dim)));
        HYLE_ASSIGN_OR_RETURN(auto v,view(value_rows,host.kv_indptr[r],nk,c.kv_heads,cache_width(c.value_storage,c.value_head_dim)));
        const auto qh=mlx::core::transpose(q,{0,2,1,3},stream),kh=mlx::core::transpose(k,{0,2,1,3},stream),vh=mlx::core::transpose(v,{0,2,1,3},stream);
        const bool e4=c.key_storage.data_type()==DataType::kFloat8E4M3FN;
        auto output=e4?mlx::core::fast::hyle_ragged_attention(qh,kh,vh,attention.scale,attention.causal,stream,attention.window_left,c.key_storage.scale(),c.value_storage.scale()):
            mlx::core::fast::hyle_ragged_attention(qh,kh,vh,attention.scale,attention.causal,stream,q8,attention.window_left);
        parts.push_back(mlx::core::reshape(mlx::core::transpose(output,{0,2,1,3},stream),{nq,int(c.query_heads),int(c.value_head_dim)},stream));
      }
      // Only the concrete Graph output suffix is initialized. Direct native
      // programs continue to expose active rows without physical padding.
      if(output_query_rows && layout.facts().query_rows<*output_query_rows)
        parts.push_back(mlx::core::zeros({int(*output_query_rows-layout.facts().query_rows),
            int(c.query_heads),int(c.value_head_dim)},inputs[0].dtype(),stream));
      if(parts.empty()){HYLE_ASSIGN_OR_RETURN(auto empty,View::slice(0,0,0));HYLE_ASSIGN_OR_RETURN(auto sliced,lower(Operation(empty),std::span(&inputs[0],1),stream));return std::vector{mlx::core::reshape(sliced[0],{0,int(c.query_heads),int(c.value_head_dim)},stream)};}
      if(parts.size()==1)return parts;
      return lower(Operation(Concat{.axis=0}),parts,stream);
    },workload,attention.causal,attention.window_left);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Apple ragged cold tracing metadata"));}
}
} // namespace
Status detail::PagedBatchAccess::validate_ragged(const std::shared_ptr<Runtime>&runtime,
    const RaggedAttention&attention)noexcept{
  if(!runtime || attention.capacity.storage!=BatchStorage::kRagged || !std::isfinite(attention.scale))
    return Status::InvalidArgument("Apple ragged Runtime/capacity/options differ");
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(attention.capacity));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(attention.causal,attention.window_left));
  if(!cache_domain(attention.capacity))
    return Status::Unimplemented("Apple ragged cache dtype/encoding/depth is outside the finite domain");
  HYLE_ASSIGN_OR_RETURN(auto query,batch_query_description(attention.capacity));
  HYLE_ASSIGN_OR_RETURN(auto key,batch_cache_description(attention.capacity,BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value,batch_cache_description(attention.capacity,BatchCacheRole::kValue));
  for(const auto&description:std::array{query,key,value})
    for(const auto dimension:description.dimensions())
      if(dimension>INT32_MAX)
        return Status::OutOfRange("Apple ragged input dimension exceeds native range");
  return Status::Ok();
}
Result<Program> prepare_ragged_batch(const Context&context,BatchLayoutVersion layout,
    RaggedAttention attention,const WorkloadContext&workload)noexcept{
  return prepare_ragged(detail::ContextAccess::runtime(context),std::move(layout),attention,workload,std::nullopt);
}
Result<Program> detail::PagedBatchAccess::prepare_ragged_graph(std::shared_ptr<Runtime>runtime,
    BatchLayoutVersion layout,RaggedAttention attention,std::uint32_t query_rows,
    const WorkloadContext&workload)noexcept{
  return prepare_ragged(runtime,std::move(layout),attention,workload,query_rows);
}
Result<Program> prepare_materialized_batch(const Context&context,
    const MaterializedBatchProgram&program,const WorkloadContext&workload)noexcept{
  if(!context.valid() || !program.valid())return std::unexpected(Status::InvalidArgument("Apple materialized batch context/program is empty"));
  try {
    std::array<InputSpec,5> inputs;const auto descriptions=program.descriptions();
    for(std::size_t i=0;i<inputs.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],input_spec(descriptions[i]));}
    // MLX retains the tracing callable. Copy its finite value records so it
    // does not borrow the common program's allocator/schema after prepare.
    // The retained native closure is separate, opaque native metadata.
    std::array<MaterializedBatchStep,19> recipe;const auto count=program.steps().size();
    std::copy(program.steps().begin(),program.steps().end(),recipe.begin());const auto output_value=program.output_value();
    return context.compile([recipe=std::move(recipe),count,output_value](const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=5)return std::unexpected(Status::InvalidArgument("Apple materialized batch needs five inputs"));
      std::array<std::optional<mlx::core::array>,24> values;
      for(std::size_t i=0;i<inputs.size();++i)values[i]=inputs[i];
      for(const auto&step:std::span(recipe).first(count)){
        std::vector<mlx::core::array> arguments;arguments.reserve(step.input_count);
        for(std::size_t i=0;i<step.input_count;++i)arguments.push_back(*values[step.inputs[i]]);
        HYLE_ASSIGN_OR_RETURN(auto result,lower(step.operation,arguments,stream));
        if(result.size()!=1)return std::unexpected(Status::Internal("Apple materialized recipe output arity"));
        values[step.output]=std::move(result[0]);
      }
      return std::vector<mlx::core::array>{*values[output_value]};
    },inputs,workload);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Apple materialized batch cold recipe metadata"));}
}
} // namespace hyle::compute::apple
