#include "hyle/compute/batch_ops.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace hyle::compute {
namespace {
struct MetadataHeader final {
  std::uint64_t encoding = 0x48594c45424c0001ULL;
  std::uint64_t generation = 0;
  BatchLayoutFacts facts{};
};
static_assert(sizeof(MetadataHeader)==32);
BatchMetadataOffsets active_offsets(BatchLayoutFacts facts,bool paged)noexcept {
  const auto ptrbytes=(std::size_t(facts.requests)+1)*sizeof(std::int32_t);
  const auto refbytes=std::size_t(facts.page_refs)*sizeof(std::int32_t);
  const auto tailbytes=paged?std::size_t(facts.requests)*sizeof(std::int32_t):0;
  return {sizeof(MetadataHeader),sizeof(MetadataHeader)+ptrbytes,
      sizeof(MetadataHeader)+2*ptrbytes,sizeof(MetadataHeader)+2*ptrbytes+refbytes,
      sizeof(MetadataHeader)+2*ptrbytes+refbytes+tailbytes};
}
Result<BatchLayoutFacts> validate_layout(detail::BatchLayoutGeometry c,HostBatchLayout h,bool row_cache) noexcept {
  if(h.query_indptr.empty() || h.query_indptr.size()>std::size_t(c.max_requests)+1 || h.kv_indptr.size()!=h.query_indptr.size())
    return std::unexpected(Status::InvalidArgument("batch indptr request extent"));
  const auto requests=h.query_indptr.size()-1;
  if(h.query_indptr[0]!=0 || h.kv_indptr[0]!=0)return std::unexpected(Status::InvalidArgument("batch indptr must start at zero"));
  if(c.storage==BatchStorage::kPaged){
    if(h.last_page_lengths.size()!=requests || h.page_indices.size()>c.max_page_refs)
      return std::unexpected(Status::InvalidArgument("batch paged metadata extent"));
    for(const auto id:h.page_indices)if(id<0 || std::uint32_t(id)>=c.max_pages)
      return std::unexpected(Status::OutOfRange("batch page ID outside physical arena"));
  }else if(!h.page_indices.empty() || !h.last_page_lengths.empty())
    return std::unexpected(Status::InvalidArgument("ragged layout cannot contain page tables"));
  std::uint64_t logical_kv=0;
  for(std::size_t i=0;i<requests;++i){
    if(h.query_indptr[i+1]<h.query_indptr[i] || h.kv_indptr[i+1]<h.kv_indptr[i])
      return std::unexpected(Status::InvalidArgument("batch indptr is not monotonic"));
    if(std::uint32_t(h.query_indptr[i+1])>c.max_query_rows || std::uint32_t(h.kv_indptr[i+1])>(c.storage==BatchStorage::kPaged?c.max_page_refs:c.max_kv_rows))
      return std::unexpected(Status::OutOfRange("batch active rows/references exceed capacity"));
    const auto q=std::uint32_t(h.query_indptr[i+1]-h.query_indptr[i]);
    const auto count=std::uint32_t(h.kv_indptr[i+1]-h.kv_indptr[i]);
    std::uint64_t kv=count;
    if(c.storage==BatchStorage::kPaged){
      const auto tail=h.last_page_lengths[i];
      if((count==0 && tail!=0) || (count!=0 && (tail<=0 || std::uint32_t(tail)>c.page_size)))
        return std::unexpected(Status::InvalidArgument("batch last page length"));
      kv=count?std::uint64_t(count-1)*c.page_size+std::uint32_t(tail):0;
    }
    if(!row_cache && q && !kv)return std::unexpected(Status::InvalidArgument("nonempty batch query requires KV rows"));
    logical_kv+=kv;
    if(logical_kv>c.max_kv_rows)return std::unexpected(Status::OutOfRange("batch logical KV rows exceed capacity"));
  }
  if(c.storage==BatchStorage::kPaged && std::size_t(h.kv_indptr.back())!=h.page_indices.size())
    return std::unexpected(Status::InvalidArgument("batch KV indptr end disagrees with page IDs"));
  return BatchLayoutFacts{std::uint32_t(requests),std::uint32_t(h.query_indptr.back()),std::uint32_t(logical_kv),
      c.storage==BatchStorage::kPaged?std::uint32_t(h.page_indices.size()):0u};
}
} // namespace
namespace detail {
struct BatchSchemaData final:util::RefCounted<BatchSchemaData> {
  BatchSchemaData(BatchCapacity c,BatchSchemaOptions o,mem::AllocatorRef a,BatchMetadataOffsets x)
      :capacity(c),options(std::move(o)),allocator(a),offsets(x){}
  void destroy()noexcept{const auto a=allocator;this->~BatchSchemaData();a.deallocate(this,sizeof(*this),alignof(BatchSchemaData));}
  BatchCapacity capacity;BatchSchemaOptions options;mem::AllocatorRef allocator;BatchMetadataOffsets offsets;
  RowCacheBatchCapacity row_capacity{};bool row_cache=false;
  MlaCacheLayout mla=MlaCacheLayout::kNone;
  std::atomic<std::uint64_t> next_generation{1};
};
struct BatchLayoutData final:util::RefCounted<BatchLayoutData> {
  BatchLayoutData(BatchSchema s,mem::AllocatorRef a):schema(std::move(s)),allocator(a),page_footprint(mem::StdAllocator<std::int32_t>(a)){}
  void destroy()noexcept{const auto a=allocator;this->~BatchLayoutData();a.deallocate(this,sizeof(*this),alignof(BatchLayoutData));}
  BatchSchema schema;mem::AllocatorRef allocator;
  ReadLease bytes;mem::HostMapping mapping; // mapping dies before its backing
  mutable std::mutex export_lock;
  mutable ReadLease exported;
  mutable std::vector<std::int32_t,mem::StdAllocator<std::int32_t>> page_footprint;
  std::uint64_t generation=0;BatchLayoutFacts facts{};
};
} // namespace detail

detail::BatchLayoutGeometry detail::BatchLayoutAccess::geometry(const BatchSchema&s)noexcept{
  if(!s.data_)return {};
  if(s.data_->row_cache){const auto c=s.data_->row_capacity;return {BatchStorage::kPaged,c.max_requests,c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.page_size};}
  const auto c=s.data_->capacity;return {c.storage,c.max_requests,c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.page_size};
}
bool detail::BatchLayoutAccess::is_row_cache(const BatchSchema&s)noexcept{return s.data_ && s.data_->row_cache;}

namespace {
Result<BatchMetadataOffsets> geometry_offsets(detail::BatchLayoutGeometry c)noexcept{
  const std::uint64_t ptrbytes=(std::uint64_t(c.max_requests)+1)*sizeof(std::int32_t);
  const std::uint64_t refbytes=std::uint64_t(c.max_page_refs)*sizeof(std::int32_t);
  const auto tailbytes=c.storage==BatchStorage::kPaged?std::uint64_t(c.max_requests)*sizeof(std::int32_t):0;
  const auto total=sizeof(MetadataHeader)+2*ptrbytes+refbytes+tailbytes;
  if(total>SIZE_MAX || total>std::uint64_t(INT64_MAX))return std::unexpected(Status::OutOfRange("batch metadata extent overflow"));
  return BatchMetadataOffsets{sizeof(MetadataHeader),std::size_t(sizeof(MetadataHeader)+ptrbytes),
      std::size_t(sizeof(MetadataHeader)+2*ptrbytes),std::size_t(sizeof(MetadataHeader)+2*ptrbytes+refbytes),std::size_t(total)};
}
Result<TensorDesc> layout_metadata_description(BatchMetadataOffsets offsets)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto encoding,StorageFormat::opaque({0x48594c4542415443ULL,0x4c41594f55540001ULL},8));
  return TensorDesc::opaque(DataType::kInt32,std::array<std::int64_t,1>{std::int64_t(offsets.bytes/4)},encoding,offsets.bytes);
}
}

Status validate_row_cache_batch_capacity(RowCacheBatchCapacity c)noexcept{
  if(!c.encoding.depth() || !c.encoding.row_bytes() || !c.max_requests || c.max_requests>=std::uint32_t(INT32_MAX) ||
     !c.max_query_rows || !c.max_kv_rows || !c.max_pages || !c.max_page_refs || !c.page_size)
    return Status::InvalidArgument("row cache capacities/encoding are invalid");
  for(const auto value:{c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.page_size})
    if(value>std::uint32_t(INT32_MAX))return Status::OutOfRange("row cache integer capacity overflow");
  // TensorDesc performs the remaining checked physical byte multiplication.
  auto description=TensorDesc::dense(DataType::kUInt8,std::array<std::int64_t,3>{c.max_pages,c.page_size,c.encoding.row_bytes()});
  if(!description)return description.error();
  HYLE_ASSIGN_OR_RETURN(auto bytes,description->required_bytes());
  return bytes>std::uint64_t(INT64_MAX)?Status::OutOfRange("row cache signed byte extent overflow"):Status::Ok();
}
Result<TensorDesc> row_cache_paged_description(RowCacheBatchCapacity c)noexcept{
  HYLE_RETURN_IF_ERROR(validate_row_cache_batch_capacity(c));
  return TensorDesc::dense(DataType::kUInt8,std::array<std::int64_t,3>{c.max_pages,c.page_size,c.encoding.row_bytes()});
}

Status validate_batch_capacity(BatchCapacity c) noexcept {
  if((c.storage!=BatchStorage::kPaged && c.storage!=BatchStorage::kRagged) ||
     (c.query_type!=DataType::kFloat16 && c.query_type!=DataType::kBFloat16) ||
     (c.output_type!=DataType::kFloat16 && c.output_type!=DataType::kBFloat16) ||
     !c.max_requests || c.max_requests>=std::uint32_t(INT32_MAX) || !c.max_query_rows || !c.max_kv_rows ||
     !c.query_heads || !c.kv_heads || !c.qk_head_dim || !c.value_head_dim || c.query_heads%c.kv_heads)
    return Status::InvalidArgument("batch capacities, dtype or GQA are invalid");
  HYLE_RETURN_IF_ERROR(validate_cache_storage(c.key_storage));
  HYLE_RETURN_IF_ERROR(validate_cache_storage(c.value_storage));
  for(const auto value:{c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.query_heads,c.kv_heads,c.qk_head_dim,c.value_head_dim,c.page_size})
    if(value>std::uint32_t(INT32_MAX))return Status::OutOfRange("batch native integer capacity overflow");
  if(c.storage==BatchStorage::kPaged){
    if(!c.max_pages || !c.max_page_refs || !c.page_size)return Status::InvalidArgument("paged capacity requires arena/ref/page extents");
    if(c.page_size>UINT32_MAX/c.kv_heads/std::max(c.qk_head_dim,c.value_head_dim))
      return Status::OutOfRange("paged native page stride overflow");
  }else{
    if(c.max_pages || c.max_page_refs || c.page_size!=1)return Status::InvalidArgument("ragged capacity requires packed rows without page extents");
  }
  if(std::uint64_t(c.query_heads)*std::max(c.qk_head_dim,c.value_head_dim)>UINT32_MAX ||
     std::uint64_t(c.kv_heads)*std::max(c.qk_head_dim,c.value_head_dim)>UINT32_MAX)
    return Status::OutOfRange("batch native row stride overflow");
  HYLE_ASSIGN_OR_RETURN(auto query,batch_query_description(c));
  HYLE_ASSIGN_OR_RETURN(auto key,batch_cache_description(c,BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto value,batch_cache_description(c,BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto output,batch_output_description(c,c.max_query_rows));
  for(const auto *description:{&query,&key,&value,&output}){
    HYLE_ASSIGN_OR_RETURN(auto bytes,description->required_bytes());
    if(bytes>std::uint64_t(INT64_MAX))return Status::OutOfRange("batch tensor signed extent overflow");
  }
  return Status::Ok();
}

Result<TensorDesc> batch_query_description(BatchCapacity c) noexcept {
  return batch_query_description(c,c.max_query_rows);
}
Result<TensorDesc> batch_query_description(BatchCapacity c,std::uint32_t rows) noexcept {
  if(rows>c.max_query_rows)return std::unexpected(Status::OutOfRange("batch query rows exceed capacity"));
  return TensorDesc::dense(c.query_type,std::array<std::int64_t,3>{rows,c.query_heads,c.qk_head_dim});
}
Result<BatchCapacity> mla_batch_capacity(MlaBatchCapacity c) noexcept {
  if(c.layout!=MlaCacheLayout::kCkv512 && c.layout!=MlaCacheLayout::kCkv512Rope64)
    return std::unexpected(Status::InvalidArgument("MLA cache layout is invalid"));
  HYLE_RETURN_IF_ERROR(validate_cache_storage(c.cache_storage));
  if(!c.cache_storage.format().is_plain())
    return std::unexpected(Status::Unimplemented("compact MLA requires scalar cache storage"));
  BatchCapacity result{.storage=c.storage,.query_type=c.query_type,.output_type=c.output_type,
      .key_storage=c.cache_storage,.value_storage=c.cache_storage,.max_requests=c.max_requests,
      .max_query_rows=c.max_query_rows,.max_kv_rows=c.max_kv_rows,
      .max_pages=c.max_pages,.max_page_refs=c.max_page_refs,.query_heads=c.query_heads,
      .kv_heads=1,.qk_head_dim=c.layout==MlaCacheLayout::kCkv512Rope64?576u:512u,
      .value_head_dim=512,.page_size=c.page_size};
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(result));return result;
}
Result<TensorDesc> batch_cache_description(const BatchSchema&schema,BatchCacheRole role) noexcept {
  if(!schema.valid())return std::unexpected(Status::InvalidArgument("empty batch schema"));
  if(detail::BatchLayoutAccess::is_row_cache(schema))return std::unexpected(Status::InvalidArgument("row cache is not attention storage"));
  if(schema.mla_layout()!=MlaCacheLayout::kNone){
    if(role!=BatchCacheRole::kMla)return std::unexpected(Status::InvalidArgument("MLA requires its single cache role"));
    return batch_cache_description(schema.capacity(),BatchCacheRole::kKey);
  }
  return batch_cache_description(schema.capacity(),role);
}
Result<TensorDesc> batch_cache_description(BatchCapacity c,BatchCacheRole role) noexcept {
  if(role!=BatchCacheRole::kKey && role!=BatchCacheRole::kValue)
    return std::unexpected(Status::InvalidArgument("invalid batch cache role"));
  const auto &storage=role==BatchCacheRole::kKey?c.key_storage:c.value_storage;
  const auto depth=role==BatchCacheRole::kKey?c.qk_head_dim:c.value_head_dim;
  if(c.storage==BatchStorage::kPaged)
    return cache_tensor_description(storage,std::array<std::int64_t,4>{c.max_pages,c.page_size,c.kv_heads,depth});
  if(c.storage==BatchStorage::kRagged)
    return cache_tensor_description(storage,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,depth});
  return std::unexpected(Status::InvalidArgument("invalid batch storage"));
}
Result<TensorDesc> batch_output_description(BatchCapacity c,std::uint32_t rows) noexcept {
  if(rows>c.max_query_rows)return std::unexpected(Status::OutOfRange("batch output rows exceed capacity"));
  return TensorDesc::dense(c.output_type,std::array<std::int64_t,3>{rows,c.query_heads,c.value_head_dim});
}
Status validate_batch_attention_options(bool causal,std::int32_t window_left) noexcept {
  if(window_left < -1 || (!causal && window_left!=-1))
    return Status::InvalidArgument("batch supports causal window or full noncausal attention");
  return Status::Ok();
}
Status validate_batch_attention_layout(const BatchLayoutVersion &layout,bool causal,
                                       std::int32_t window_left) noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(causal,window_left));
  if(!layout.valid())return Status::InvalidArgument("attention layout is empty");
  if(detail::BatchLayoutAccess::is_row_cache(layout.schema()))return Status::InvalidArgument("row cache is not an attention layout");
  if(!causal)return Status::Ok();
  const auto c=layout.schema().capacity();const auto host=layout.host();
  for(std::size_t r=0;r<layout.facts().requests;++r){
    const auto q=std::uint32_t(host.query_indptr[r+1]-host.query_indptr[r]);
    const auto count=std::uint32_t(host.kv_indptr[r+1]-host.kv_indptr[r]);
    const auto kv=c.storage==BatchStorage::kPaged?
        (count?std::uint64_t(count-1)*c.page_size+std::uint32_t(host.last_page_lengths[r]):0):count;
    if(q>kv)return Status::InvalidArgument("causal query requires right-aligned valid KV positions");
  }
  return Status::Ok();
}
Result<BatchMetadataOffsets> batch_metadata_offsets(BatchCapacity c) noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  return geometry_offsets({c.storage,c.max_requests,c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.page_size});
}
Result<TensorDesc> batch_metadata_description(BatchCapacity c) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto offsets,batch_metadata_offsets(c));
  return layout_metadata_description(offsets);
}
BatchSchema::BatchSchema(const BatchSchema&s)noexcept:data_(s.data_){if(data_)data_->incref();}
BatchSchema&BatchSchema::operator=(const BatchSchema&s)noexcept{if(this!=&s){BatchSchema next(s);std::swap(data_,next.data_);}return *this;}
BatchSchema::BatchSchema(BatchSchema&&s)noexcept:data_(std::exchange(s.data_,nullptr)){}
BatchSchema&BatchSchema::operator=(BatchSchema&&s)noexcept{if(this!=&s){BatchSchema next(std::move(s));std::swap(data_,next.data_);}return *this;}
BatchSchema::~BatchSchema(){if(data_)data_->decref();}
Result<BatchSchema> BatchSchema::create(BatchCapacity c,BatchSchemaOptions o)noexcept {
  HYLE_ASSIGN_OR_RETURN(auto offsets,batch_metadata_offsets(c));
  if(!o.allocator.valid())return std::unexpected(Status::InvalidArgument("batch allocator is invalid"));
  auto a=o.allocator;if(o.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,o.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::BatchSchemaData>(c,std::move(o),a,offsets);
  if(!p)return std::unexpected(Status::ResourceExhausted("batch schema metadata"));
  return BatchSchema(p);
}
BatchCapacity BatchSchema::capacity()const noexcept{
  if(data_ && data_->row_cache){BatchCapacity invalid;invalid.query_type=invalid.output_type=DataType::kUnknown;invalid.max_requests=0;return invalid;}
  return data_?data_->capacity:BatchCapacity{};
}
Result<BatchSchema> BatchSchema::create_mla(MlaBatchCapacity c,BatchSchemaOptions o)noexcept {
  HYLE_ASSIGN_OR_RETURN(auto capacity,mla_batch_capacity(c));
  HYLE_ASSIGN_OR_RETURN(auto result,create(capacity,std::move(o)));
  result.data_->mla=c.layout;return result;
}
MlaCacheLayout BatchSchema::mla_layout()const noexcept{return data_?data_->mla:MlaCacheLayout::kNone;}
BatchMetadataOffsets BatchSchema::metadata_offsets()const noexcept{return data_?data_->offsets:BatchMetadataOffsets{};}
Result<TensorDesc> BatchSchema::metadata_description()const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch schema is empty"));
  return layout_metadata_description(data_->offsets);
}
Result<BatchLayoutVersion> BatchSchema::version(HostBatchLayout h)const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch schema is empty"));
  const auto geometry=detail::BatchLayoutAccess::geometry(*this);
  HYLE_ASSIGN_OR_RETURN(auto facts,validate_layout(geometry,h,data_->row_cache));
  auto generation=data_->next_generation.load(std::memory_order_relaxed);
  do{if(generation==UINT64_MAX)return std::unexpected(Status::OutOfRange("batch layout generation exhausted"));}
  while(!data_->next_generation.compare_exchange_weak(generation,generation+1,std::memory_order_relaxed));
  auto*p=data_->allocator.construct<detail::BatchLayoutData>(*this,data_->allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("batch layout metadata"));
  BatchLayoutVersion result(p);p->generation=generation;p->facts=facts;
  const auto offsets=active_offsets(facts,geometry.storage==BatchStorage::kPaged);
  HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(mem::system_resource(),offsets.bytes,64));
  {
    HYLE_ASSIGN_OR_RETURN(auto mapping,buffer.view().map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());
    const MetadataHeader header{.generation=generation,.facts=facts};std::memcpy(bytes.data(),&header,sizeof(header));
    const auto copy=[&](std::size_t offset,std::span<const std::int32_t> values){if(!values.empty())std::memcpy(bytes.data()+offset,values.data(),values.size_bytes());};
    copy(offsets.query_indptr,h.query_indptr);copy(offsets.kv_indptr,h.kv_indptr);
    copy(offsets.page_indices,h.page_indices);copy(offsets.last_page_lengths,h.last_page_lengths);
    HYLE_RETURN_IF_ERROR(mapping.close());
  }
  HYLE_ASSIGN_OR_RETURN(p->bytes,ReadLease::adopt(std::move(buffer),{.allocator=data_->allocator,.resource_budget=data_->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(auto view,p->bytes.view());HYLE_ASSIGN_OR_RETURN(p->mapping,view.map_host());
  return result;
}
bool BatchSchema::owns(const BatchLayoutVersion&v)const noexcept{return data_ && v.data_ && v.data_->schema.data_==data_;}
Result<RowCacheBatchSchema> RowCacheBatchSchema::create(RowCacheBatchCapacity c,BatchSchemaOptions o)noexcept{
  HYLE_RETURN_IF_ERROR(validate_row_cache_batch_capacity(c));
  HYLE_ASSIGN_OR_RETURN(auto offsets,geometry_offsets({BatchStorage::kPaged,c.max_requests,c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.page_size}));
  if(!o.allocator.valid())return std::unexpected(Status::InvalidArgument("row cache allocator is invalid"));
  auto a=o.allocator;if(o.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,o.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::BatchSchemaData>(BatchCapacity{},std::move(o),a,offsets);
  if(!p)return std::unexpected(Status::ResourceExhausted("row cache schema metadata"));
  p->row_capacity=c;p->row_cache=true;return RowCacheBatchSchema(BatchSchema(p));
}
RowCacheBatchCapacity RowCacheBatchSchema::capacity()const noexcept{return owner_.data_?owner_.data_->row_capacity:RowCacheBatchCapacity{};}
Result<RowCacheLayoutVersion> RowCacheBatchSchema::version(HostBatchLayout h)const noexcept{
  HYLE_ASSIGN_OR_RETURN(auto version,owner_.version(h));return RowCacheLayoutVersion(std::move(version));
}
bool RowCacheBatchSchema::owns(const RowCacheLayoutVersion&v)const noexcept{return owner_.owns(v.owner_);}
RowCacheBatchSchema RowCacheLayoutVersion::schema()const noexcept{return RowCacheBatchSchema(owner_.schema());}
BatchLayoutVersion::BatchLayoutVersion(const BatchLayoutVersion&v)noexcept:data_(v.data_){if(data_)data_->incref();}
BatchLayoutVersion&BatchLayoutVersion::operator=(const BatchLayoutVersion&v)noexcept{if(this!=&v){BatchLayoutVersion next(v);std::swap(data_,next.data_);}return *this;}
BatchLayoutVersion::BatchLayoutVersion(BatchLayoutVersion&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
BatchLayoutVersion&BatchLayoutVersion::operator=(BatchLayoutVersion&&v)noexcept{if(this!=&v){BatchLayoutVersion next(std::move(v));std::swap(data_,next.data_);}return *this;}
BatchLayoutVersion::~BatchLayoutVersion(){if(data_)data_->decref();}
BatchSchema BatchLayoutVersion::schema()const noexcept{return data_?data_->schema:BatchSchema{};}
std::uint64_t BatchLayoutVersion::generation()const noexcept{return data_?data_->generation:0;}
BatchLayoutFacts BatchLayoutVersion::facts()const noexcept{return data_?data_->facts:BatchLayoutFacts{};}
HostBatchLayout BatchLayoutVersion::host()const noexcept {
  if(!data_)return {};
  const auto bytes=data_->mapping.bytes();
  const auto paged=detail::BatchLayoutAccess::geometry(data_->schema).storage==BatchStorage::kPaged;
  const auto offsets=active_offsets(data_->facts,paged);
  const auto span=[&](std::size_t offset,std::size_t count){return std::span<const std::int32_t>(reinterpret_cast<const std::int32_t*>(bytes.data()+offset),count);};
  const auto n=data_->facts.requests;
  return {span(offsets.query_indptr,std::size_t(n)+1),span(offsets.kv_indptr,std::size_t(n)+1),
      span(offsets.page_indices,data_->facts.page_refs),span(offsets.last_page_lengths,paged?n:0)};
}
Result<ReadLease> BatchLayoutVersion::metadata()const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch layout is empty"));
  std::lock_guard guard(data_->export_lock);
  if(data_->exported.valid())return data_->exported;
  HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(mem::system_resource(),data_->schema.metadata_offsets().bytes,64));
  {
    HYLE_ASSIGN_OR_RETURN(auto mapping,buffer.view().map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());
    std::fill(bytes.begin(),bytes.end(),std::byte{0});
    for(const auto&segment:detail::BatchLayoutAccess::segments(*this))
      if(!segment.bytes.empty())std::memcpy(bytes.data()+segment.offset,segment.bytes.data(),segment.bytes.size());
    HYLE_RETURN_IF_ERROR(mapping.close());
  }
  HYLE_ASSIGN_OR_RETURN(auto ready,ReadLease::adopt(std::move(buffer),
      {.allocator=data_->allocator,.resource_budget=data_->schema.data_->options.resource_budget}));
  data_->exported=std::move(ready);return data_->exported;
}
std::array<detail::BatchMetadataSegment,5> detail::BatchLayoutAccess::segments(const BatchLayoutVersion&version)noexcept {
  if(!version.data_)return {};
  const auto h=version.host();const auto offsets=version.schema().metadata_offsets();
  return {{{0,version.data_->mapping.bytes().first(sizeof(MetadataHeader))},
      {offsets.query_indptr,std::as_bytes(h.query_indptr)},
      {offsets.kv_indptr,std::as_bytes(h.kv_indptr)},
      {offsets.page_indices,std::as_bytes(h.page_indices)},
      {offsets.last_page_lengths,std::as_bytes(h.last_page_lengths)}}};
}
Result<std::span<const std::int32_t>> detail::BatchLayoutAccess::page_footprint(const BatchLayoutVersion&version)noexcept {
  if(!version.data_)return std::unexpected(Status::FailedPrecondition("batch layout is empty"));
  auto&data=*version.data_;
  std::lock_guard guard(data.export_lock);
  const auto refs=version.host().page_indices;
  if(data.page_footprint.empty() && !refs.empty()){
    // Build separately so an allocation failure leaves an empty, retryable
    // cache. Only active references consume the original metadata budget.
    decltype(data.page_footprint) next(mem::StdAllocator<std::int32_t>(data.allocator));
    try{next.assign(refs.begin(),refs.end());}
    catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("batch page footprint metadata"));}
    std::sort(next.begin(),next.end());next.erase(std::unique(next.begin(),next.end()),next.end());
    data.page_footprint.swap(next);
  }
  return std::span<const std::int32_t>(data.page_footprint);
}
bool BatchLayoutVersion::operator==(const BatchLayoutVersion&v)const noexcept{return data_==v.data_;}
Status infer_batch_attention(BatchCapacity c,BatchStorage storage,std::span<const TensorDesc> inputs,std::span<TensorDesc> outputs)noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  if(c.storage!=storage || inputs.size()!=4 || outputs.size()!=1)return Status::InvalidArgument("batch attention port/storage count");
  if(!inputs[0].valid() || inputs[0].rank()!=3)return Status::InvalidArgument("batch attention requires a packed rank-three query");
  const auto extent=inputs[0].dimensions()[0];
  if(extent<0 || std::uint64_t(extent)>c.max_query_rows)return Status::OutOfRange("batch query rows exceed capacity");
  const auto rows=static_cast<std::uint32_t>(extent);
  HYLE_ASSIGN_OR_RETURN(auto q,batch_query_description(c,rows));
  HYLE_ASSIGN_OR_RETURN(auto k,batch_cache_description(c,BatchCacheRole::kKey));
  HYLE_ASSIGN_OR_RETURN(auto v,batch_cache_description(c,BatchCacheRole::kValue));
  HYLE_ASSIGN_OR_RETURN(auto metadata,batch_metadata_description(c));
  if(inputs[0]!=q || inputs[1]!=k || inputs[2]!=v || inputs[3]!=metadata)return Status::InvalidArgument("batch attention capacity descriptions disagree");
  HYLE_ASSIGN_OR_RETURN(outputs[0],batch_output_description(c,rows));return Status::Ok();
}
} // namespace hyle::compute
