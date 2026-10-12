#include "hyle/compute/batch_ops.h"
#include "hyle/util/ref_counted.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>

namespace hyle::compute {
namespace {
struct MetadataHeader final {
  std::uint64_t encoding = 0x48594c45424c0001ULL;
  std::uint64_t generation = 0;
  BatchLayoutFacts facts{};
};
static_assert(sizeof(MetadataHeader)==32);
Status checked_extent(std::initializer_list<std::uint64_t> factors) noexcept {
  std::uint64_t bytes=1;
  for(const auto factor:factors){if(factor>SIZE_MAX/bytes)return Status::OutOfRange("batch tensor extent overflow");bytes*=factor;}
  if(bytes>std::uint64_t(INT64_MAX))return Status::OutOfRange("batch tensor signed extent overflow");
  return Status::Ok();
}
Result<BatchLayoutFacts> validate_layout(BatchCapacity c,HostBatchLayout h) noexcept {
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
    if(q>kv)return std::unexpected(Status::InvalidArgument("batch query requires right-aligned valid KV positions"));
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
  std::atomic<std::uint64_t> next_generation{1};
};
struct BatchLayoutData final:util::RefCounted<BatchLayoutData> {
  BatchLayoutData(BatchSchema s,mem::AllocatorRef a):schema(std::move(s)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~BatchLayoutData();a.deallocate(this,sizeof(*this),alignof(BatchLayoutData));}
  BatchSchema schema;mem::AllocatorRef allocator;
  ReadLease bytes;mem::HostMapping mapping; // mapping dies before its backing
  std::uint64_t generation=0;BatchLayoutFacts facts{};
};
} // namespace detail

Status validate_batch_capacity(BatchCapacity c) noexcept {
  if((c.storage!=BatchStorage::kPaged && c.storage!=BatchStorage::kRagged) ||
     (c.data_type!=DataType::kFloat16 && c.data_type!=DataType::kBFloat16) ||
     !c.max_requests || c.max_requests>=std::uint32_t(INT32_MAX) || !c.max_query_rows || !c.max_kv_rows ||
     !c.query_heads || !c.kv_heads || !c.head_dim || c.query_heads%c.kv_heads)
    return Status::InvalidArgument("batch capacities, dtype or GQA are invalid");
  for(const auto value:{c.max_query_rows,c.max_kv_rows,c.max_pages,c.max_page_refs,c.query_heads,c.kv_heads,c.head_dim,c.page_size})
    if(value>std::uint32_t(INT32_MAX))return Status::OutOfRange("batch native integer capacity overflow");
  if(c.storage==BatchStorage::kPaged){
    if(!c.max_pages || !c.max_page_refs || !c.page_size)return Status::InvalidArgument("paged capacity requires arena/ref/page extents");
    if(c.page_size>UINT32_MAX/c.kv_heads/c.head_dim)
      return Status::OutOfRange("paged native page stride overflow");
    HYLE_RETURN_IF_ERROR(checked_extent({c.max_pages,c.page_size,c.kv_heads,c.head_dim,2}));
  }else{
    if(c.max_pages || c.max_page_refs || c.page_size!=1)return Status::InvalidArgument("ragged capacity requires packed rows without page extents");
    HYLE_RETURN_IF_ERROR(checked_extent({c.max_kv_rows,c.kv_heads,c.head_dim,2}));
  }
  if(std::uint64_t(c.query_heads)*c.head_dim>UINT32_MAX || std::uint64_t(c.kv_heads)*c.head_dim>UINT32_MAX)
    return Status::OutOfRange("batch native row stride overflow");
  return checked_extent({c.max_query_rows,c.query_heads,c.head_dim,2});
}
Result<BatchMetadataOffsets> batch_metadata_offsets(BatchCapacity c) noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  const std::uint64_t ptrbytes=(std::uint64_t(c.max_requests)+1)*sizeof(std::int32_t);
  const std::uint64_t refbytes=std::uint64_t(c.max_page_refs)*sizeof(std::int32_t);
  const auto tailbytes=c.storage==BatchStorage::kPaged?std::uint64_t(c.max_requests)*sizeof(std::int32_t):0;
  const auto total=sizeof(MetadataHeader)+2*ptrbytes+refbytes+tailbytes;
  if(total>SIZE_MAX || total>std::uint64_t(INT64_MAX))return std::unexpected(Status::OutOfRange("batch metadata extent overflow"));
  return BatchMetadataOffsets{sizeof(MetadataHeader),std::size_t(sizeof(MetadataHeader)+ptrbytes),
      std::size_t(sizeof(MetadataHeader)+2*ptrbytes),std::size_t(sizeof(MetadataHeader)+2*ptrbytes+refbytes),std::size_t(total)};
}
Result<TensorDesc> batch_metadata_description(BatchCapacity c) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto offsets,batch_metadata_offsets(c));
  HYLE_ASSIGN_OR_RETURN(auto encoding,StorageFormat::opaque({0x48594c4542415443ULL,0x4c41594f55540001ULL},8));
  return TensorDesc::opaque(DataType::kInt32,std::array<std::int64_t,1>{std::int64_t(offsets.bytes/4)},encoding,offsets.bytes);
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
BatchCapacity BatchSchema::capacity()const noexcept{return data_?data_->capacity:BatchCapacity{};}
BatchMetadataOffsets BatchSchema::metadata_offsets()const noexcept{return data_?data_->offsets:BatchMetadataOffsets{};}
Result<TensorDesc> BatchSchema::metadata_description()const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch schema is empty"));
  return batch_metadata_description(data_->capacity);
}
Result<BatchLayoutVersion> BatchSchema::version(HostBatchLayout h)const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch schema is empty"));
  HYLE_ASSIGN_OR_RETURN(auto facts,validate_layout(data_->capacity,h));
  auto generation=data_->next_generation.load(std::memory_order_relaxed);
  do{if(generation==UINT64_MAX)return std::unexpected(Status::OutOfRange("batch layout generation exhausted"));}
  while(!data_->next_generation.compare_exchange_weak(generation,generation+1,std::memory_order_relaxed));
  auto*p=data_->allocator.construct<detail::BatchLayoutData>(*this,data_->allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("batch layout metadata"));
  BatchLayoutVersion result(p);p->generation=generation;p->facts=facts;
  HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(mem::system_resource(),data_->offsets.bytes,64));
  {
    HYLE_ASSIGN_OR_RETURN(auto mapping,buffer.view().map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());std::fill(bytes.begin(),bytes.end(),std::byte{0});
    const MetadataHeader header{.generation=generation,.facts=facts};std::memcpy(bytes.data(),&header,sizeof(header));
    const auto copy=[&](std::size_t offset,std::span<const std::int32_t> values){if(!values.empty())std::memcpy(bytes.data()+offset,values.data(),values.size_bytes());};
    copy(data_->offsets.query_indptr,h.query_indptr);copy(data_->offsets.kv_indptr,h.kv_indptr);
    copy(data_->offsets.page_indices,h.page_indices);copy(data_->offsets.last_page_lengths,h.last_page_lengths);
    HYLE_RETURN_IF_ERROR(mapping.close());
  }
  HYLE_ASSIGN_OR_RETURN(p->bytes,ReadLease::adopt(std::move(buffer),{.allocator=data_->allocator,.resource_budget=data_->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(auto view,p->bytes.view());HYLE_ASSIGN_OR_RETURN(p->mapping,view.map_host());
  return result;
}
bool BatchSchema::owns(const BatchLayoutVersion&v)const noexcept{return data_ && v.data_ && v.data_->schema.data_==data_;}
BatchLayoutVersion::BatchLayoutVersion(const BatchLayoutVersion&v)noexcept:data_(v.data_){if(data_)data_->incref();}
BatchLayoutVersion&BatchLayoutVersion::operator=(const BatchLayoutVersion&v)noexcept{if(this!=&v){BatchLayoutVersion next(v);std::swap(data_,next.data_);}return *this;}
BatchLayoutVersion::BatchLayoutVersion(BatchLayoutVersion&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
BatchLayoutVersion&BatchLayoutVersion::operator=(BatchLayoutVersion&&v)noexcept{if(this!=&v){BatchLayoutVersion next(std::move(v));std::swap(data_,next.data_);}return *this;}
BatchLayoutVersion::~BatchLayoutVersion(){if(data_)data_->decref();}
BatchSchema BatchLayoutVersion::schema()const noexcept{return data_?data_->schema:BatchSchema{};}
std::uint64_t BatchLayoutVersion::generation()const noexcept{return data_?data_->generation:0;}
BatchLayoutFacts BatchLayoutVersion::facts()const noexcept{return data_?data_->facts:BatchLayoutFacts{};}
HostBatchLayout BatchLayoutVersion::host()const noexcept {
  if(!data_)return {};const auto bytes=data_->mapping.bytes();const auto offsets=data_->schema.metadata_offsets();
  const auto span=[&](std::size_t offset,std::size_t count){return std::span<const std::int32_t>(reinterpret_cast<const std::int32_t*>(bytes.data()+offset),count);};
  const auto n=data_->facts.requests;const bool paged=data_->schema.capacity().storage==BatchStorage::kPaged;
  return {span(offsets.query_indptr,std::size_t(n)+1),span(offsets.kv_indptr,std::size_t(n)+1),
      span(offsets.page_indices,data_->facts.page_refs),span(offsets.last_page_lengths,paged?n:0)};
}
Result<ReadLease> BatchLayoutVersion::metadata()const noexcept {
  if(!data_)return std::unexpected(Status::FailedPrecondition("batch layout is empty"));return data_->bytes;
}
bool BatchLayoutVersion::operator==(const BatchLayoutVersion&v)const noexcept{return data_==v.data_;}
Status infer_batch_attention(BatchCapacity c,BatchStorage storage,std::span<const TensorDesc> inputs,std::span<TensorDesc> outputs)noexcept {
  HYLE_RETURN_IF_ERROR(validate_batch_capacity(c));
  if(c.storage!=storage || inputs.size()!=4 || outputs.size()!=1)return Status::InvalidArgument("batch attention port/storage count");
  HYLE_ASSIGN_OR_RETURN(auto q,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.head_dim}));
  TensorDesc kv;
  if(storage==BatchStorage::kPaged){HYLE_ASSIGN_OR_RETURN(kv,TensorDesc::dense(c.data_type,std::array<std::int64_t,4>{c.max_pages,c.page_size,c.kv_heads,c.head_dim}));}
  else{HYLE_ASSIGN_OR_RETURN(kv,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,c.head_dim}));}
  HYLE_ASSIGN_OR_RETURN(auto metadata,batch_metadata_description(c));
  if(inputs[0]!=q || inputs[1]!=kv || inputs[2]!=kv || inputs[3]!=metadata)return Status::InvalidArgument("batch attention capacity descriptions disagree");
  outputs[0]=q;return Status::Ok();
}
} // namespace hyle::compute
