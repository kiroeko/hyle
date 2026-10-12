#include "hyle/compute/batch_materialize.h"
#include "hyle/util/ref_counted.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hyle::compute {
namespace detail {
struct MaterializedBatchProgramData final:util::RefCounted<MaterializedBatchProgramData> {
  MaterializedBatchProgramData(BatchSchema s,MaterializedBatchOptions o,mem::AllocatorRef a)
      :schema(std::move(s)),options(std::move(o)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~MaterializedBatchProgramData();a.deallocate(this,sizeof(*this),alignof(MaterializedBatchProgramData));}
  BatchSchema schema;MaterializedBatchOptions options;mem::AllocatorRef allocator;
  std::array<TensorDesc,24> values{};std::array<MaterializedBatchStep,19> steps{};
  std::uint16_t value_count=5,step_count=0,output=0;
  MaterializedBatchCosts costs;
};
struct MaterializedBatchVersionData final:util::RefCounted<MaterializedBatchVersionData> {
  MaterializedBatchVersionData(MaterializedBatchProgram p,BatchLayoutVersion l,mem::AllocatorRef a)
      :program(std::move(p)),layout(std::move(l)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~MaterializedBatchVersionData();a.deallocate(this,sizeof(*this),alignof(MaterializedBatchVersionData));}
  MaterializedBatchProgram program;BatchLayoutVersion layout;mem::AllocatorRef allocator;ReadLease bytes;
};
} // namespace detail
namespace {
Result<std::size_t> product(std::initializer_list<std::uint64_t> factors)noexcept{
  std::size_t total=1;for(const auto factor:factors){if(!factor || factor>SIZE_MAX/total)return std::unexpected(Status::OutOfRange("materialized batch extent overflow"));total*=std::size_t(factor);}return total;
}
Result<std::uint16_t> add(detail::MaterializedBatchProgramData&p,Operation op,std::initializer_list<std::uint16_t> inputs)noexcept{
  if(p.step_count==p.steps.size() || p.value_count==p.values.size() || inputs.size()>2)
    return std::unexpected(Status::Internal("materialized batch fixed recipe capacity"));
  MaterializedBatchStep step;step.operation=std::move(op);step.output=p.value_count;step.input_count=std::uint8_t(inputs.size());
  std::array<TensorDesc,2> descriptions;std::size_t count=0;
  for(const auto id:inputs){step.inputs[count]=id;descriptions[count++]=p.values[id];}
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation,std::span(descriptions).first(count),std::span(&p.values[p.value_count],1)));
  HYLE_ASSIGN_OR_RETURN(auto bytes,p.values[p.value_count].required_bytes());
  if(bytes>SIZE_MAX-p.costs.logical_intermediate_bytes)return std::unexpected(Status::OutOfRange("materialized batch logical cost overflow"));
  p.costs.logical_intermediate_bytes+=bytes;p.steps[p.step_count++]=std::move(step);return p.value_count++;
}
Result<std::uint16_t> reshape(detail::MaterializedBatchProgramData&p,std::uint16_t input,std::initializer_list<std::int64_t> dims)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto shape,TensorShape::create(std::span(dims.begin(),dims.size())));
  HYLE_ASSIGN_OR_RETURN(auto view,View::reshape(shape));return add(p,Operation(view),{input});
}
std::uint32_t kv_length(BatchCapacity c,HostBatchLayout h,std::size_t r)noexcept{
  const auto count=std::uint32_t(h.kv_indptr[r+1]-h.kv_indptr[r]);
  return c.storage==BatchStorage::kPaged?(count?(count-1)*c.page_size+std::uint32_t(h.last_page_lengths[r]):0):count;
}
} // namespace
MaterializedBatchProgram::MaterializedBatchProgram(const MaterializedBatchProgram&p)noexcept:data_(p.data_){if(data_)data_->incref();}
MaterializedBatchProgram&MaterializedBatchProgram::operator=(const MaterializedBatchProgram&p)noexcept{if(this!=&p){MaterializedBatchProgram next(p);std::swap(data_,next.data_);}return *this;}
MaterializedBatchProgram::MaterializedBatchProgram(MaterializedBatchProgram&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
MaterializedBatchProgram&MaterializedBatchProgram::operator=(MaterializedBatchProgram&&p)noexcept{if(this!=&p){MaterializedBatchProgram next(std::move(p));std::swap(data_,next.data_);}return *this;}
MaterializedBatchProgram::~MaterializedBatchProgram(){if(data_)data_->decref();}
Result<MaterializedBatchProgram> MaterializedBatchProgram::prepare(BatchSchema schema,MaterializedBatchOptions options)noexcept{
  if(!schema.valid() || !options.allocator.valid() || !options.max_kv_per_request || !options.max_host_metadata_bytes || !options.max_logical_intermediate_bytes || !std::isfinite(options.scale))
    return std::unexpected(Status::InvalidArgument("materialized batch needs explicit finite capacities/budgets"));
  const auto c=schema.capacity();const auto n=c.max_query_rows,h=c.query_heads,d=c.head_dim,l=options.max_kv_per_request;
  if(l>c.max_kv_rows)return std::unexpected(Status::InvalidArgument("materialized per-request KV capacity exceeds schema"));
  HYLE_ASSIGN_OR_RETURN(auto rows,product({c.storage==BatchStorage::kPaged?c.max_pages:c.max_kv_rows,c.storage==BatchStorage::kPaged?c.page_size:1u,c.kv_heads}));
  if(rows>std::size_t(INT32_MAX))return std::unexpected(Status::OutOfRange("materialized gather flattened index exceeds I32"));
  MaterializedBatchCosts costs;
  HYLE_ASSIGN_OR_RETURN(costs.index_bytes,product({n,h,l,sizeof(std::int32_t)}));HYLE_ASSIGN_OR_RETURN(costs.mask_bytes,product({n,l,sizeof(float)}));
  if(costs.index_bytes>SIZE_MAX-costs.mask_bytes || costs.index_bytes+costs.mask_bytes>options.max_host_metadata_bytes)
    return std::unexpected(Status::ResourceExhausted("materialized host metadata byte capacity"));
  HYLE_ASSIGN_OR_RETURN(costs.gathered_kv_bytes,product({2u,n,h,l,d,2u}));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<detail::MaterializedBatchProgramData>(std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("materialized batch program metadata"));
  MaterializedBatchProgram result(p);p->costs=costs;
  HYLE_ASSIGN_OR_RETURN(p->values[0],TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{n,h,d}));
  if(c.storage==BatchStorage::kPaged){HYLE_ASSIGN_OR_RETURN(p->values[1],TensorDesc::dense(c.data_type,std::array<std::int64_t,4>{c.max_pages,c.page_size,c.kv_heads,d}));}
  else{HYLE_ASSIGN_OR_RETURN(p->values[1],TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,d}));}
  p->values[2]=p->values[1];
  HYLE_ASSIGN_OR_RETURN(p->values[3],TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,3>{n,h,l}));
  HYLE_ASSIGN_OR_RETURN(p->values[4],TensorDesc::dense(DataType::kFloat32,std::array<std::int64_t,4>{n,1,1,l}));
  HYLE_ASSIGN_OR_RETURN(auto k,reshape(*p,1,{std::int64_t(rows),d}));HYLE_ASSIGN_OR_RETURN(auto v,reshape(*p,2,{std::int64_t(rows),d}));
  // Native checked gather also rejects an invalid index if the ordinary
  // recipe is deliberately bound without its owned metadata version.
  const Operation gather(Gather{0,IndexBoundsPolicy::kChecked});
  HYLE_ASSIGN_OR_RETURN(k,add(*p,gather,{k,3}));HYLE_ASSIGN_OR_RETURN(v,add(*p,gather,{v,3}));
  HYLE_ASSIGN_OR_RETURN(k,add(*p,Operation(Cast{DataType::kFloat32}),{k}));HYLE_ASSIGN_OR_RETURN(v,add(*p,Operation(Cast{DataType::kFloat32}),{v}));
  HYLE_ASSIGN_OR_RETURN(auto permutation,View::permute(std::array<std::int32_t,4>{0,1,3,2}));HYLE_ASSIGN_OR_RETURN(k,add(*p,Operation(permutation),{k}));HYLE_ASSIGN_OR_RETURN(k,add(*p,Operation(Copy{}),{k}));
  HYLE_ASSIGN_OR_RETURN(auto q,add(*p,Operation(Cast{DataType::kFloat32}),{0}));HYLE_ASSIGN_OR_RETURN(q,reshape(*p,q,{n,h,1,d}));
  HYLE_ASSIGN_OR_RETURN(auto scores,add(*p,Operation(MatMul{}),{q,k}));
  HYLE_ASSIGN_OR_RETURN(auto probabilities,add(*p,Operation(Softmax{.axis=-1,.scale=p->options.scale,.mask=MaskKind::kAdditive}),{scores,4}));
  HYLE_ASSIGN_OR_RETURN(auto output,add(*p,Operation(MatMul{}),{probabilities,v}));HYLE_ASSIGN_OR_RETURN(output,add(*p,Operation(Cast{c.data_type}),{output}));HYLE_ASSIGN_OR_RETURN(output,reshape(*p,output,{n,h,d}));
  // The reference explicitly pays this copy to produce an independent caller
  // output, rather than exporting a reshape alias of private native scratch.
  HYLE_ASSIGN_OR_RETURN(p->output,add(*p,Operation(Copy{}),{output}));
  if(p->costs.logical_intermediate_bytes>p->options.max_logical_intermediate_bytes)
    return std::unexpected(Status::ResourceExhausted("materialized logical intermediate byte capacity"));
  return result;
}
std::span<const TensorDesc> MaterializedBatchProgram::descriptions()const noexcept{return data_?std::span(data_->values).first(data_->value_count):std::span<const TensorDesc>{};}
BatchSchema MaterializedBatchProgram::schema()const noexcept{return data_?data_->schema:BatchSchema{};}
std::span<const MaterializedBatchStep> MaterializedBatchProgram::steps()const noexcept{return data_?std::span(data_->steps).first(data_->step_count):std::span<const MaterializedBatchStep>{};}
std::uint16_t MaterializedBatchProgram::output_value()const noexcept{return data_?data_->output:0;}
MaterializedBatchCosts MaterializedBatchProgram::costs()const noexcept{return data_?data_->costs:MaterializedBatchCosts{};}
Result<MaterializedBatchVersion> MaterializedBatchProgram::version(BatchLayoutVersion layout)const noexcept{
  if(!data_ || !data_->schema.owns(layout))return std::unexpected(Status::InvalidArgument("materialized layout schema is foreign"));
  if(!layout.facts().query_rows)return std::unexpected(Status::Unimplemented("materialized empty active batch is not qualified"));
  const auto c=data_->schema.capacity();const auto host=layout.host();const auto l=data_->options.max_kv_per_request;
  for(std::size_t r=0;r<layout.facts().requests;++r)if(kv_length(c,host,r)>l)
    return std::unexpected(Status::OutOfRange("materialized request KV exceeds cold capacity"));
  auto*p=data_->allocator.construct<detail::MaterializedBatchVersionData>(*this,std::move(layout),data_->allocator);if(!p)return std::unexpected(Status::ResourceExhausted("materialized batch version metadata"));
  MaterializedBatchVersion result(p);const auto bytes=data_->costs.index_bytes+data_->costs.mask_bytes;
  HYLE_ASSIGN_OR_RETURN(auto buffer,mem::Buffer::create(mem::system_resource(),bytes,64));
  {
    HYLE_ASSIGN_OR_RETURN(auto mapping,buffer.view().map_host());HYLE_ASSIGN_OR_RETURN(auto writable,mapping.writable_bytes());
    auto*indices=reinterpret_cast<std::int32_t*>(writable.data());auto*mask=reinterpret_cast<float*>(writable.data()+data_->costs.index_bytes);
    std::fill_n(indices,data_->costs.index_bytes/sizeof(std::int32_t),0);std::fill_n(mask,data_->costs.mask_bytes/sizeof(float),-(std::numeric_limits<float>::infinity)());
    for(std::size_t row=p->layout.facts().query_rows;row<c.max_query_rows;++row)mask[row*l]=0;
    const auto groups=c.query_heads/c.kv_heads;
    for(std::size_t r=0;r<p->layout.facts().requests;++r){const auto length=kv_length(c,host,r);const auto qlength=std::uint32_t(host.query_indptr[r+1]-host.query_indptr[r]);
      for(std::uint32_t qi=0;qi<qlength;++qi){const auto row=std::uint32_t(host.query_indptr[r])+qi;const auto valid=data_->options.causal?length-qlength+qi+1:length;
        const auto first=c.storage==BatchStorage::kPaged?std::uint32_t(host.page_indices[std::uint32_t(host.kv_indptr[r])])*c.page_size:std::uint32_t(host.kv_indptr[r]);
        // Masked capacity entries gather a real valid row, never uninitialized
        // arena padding whose NaN could poison a masked score or 0*V term.
        for(std::uint32_t head=0;head<c.query_heads;++head)std::fill_n(indices+(std::size_t(row)*c.query_heads+head)*l,l,std::int32_t(std::uint64_t(first)*c.kv_heads+head/groups));
        for(std::uint32_t j=0;j<length;++j){const auto physical=c.storage==BatchStorage::kPaged?std::uint32_t(host.page_indices[std::uint32_t(host.kv_indptr[r])+j/c.page_size])*c.page_size+j%c.page_size:std::uint32_t(host.kv_indptr[r])+j;
          for(std::uint32_t head=0;head<c.query_heads;++head)indices[(std::size_t(row)*c.query_heads+head)*l+j]=std::int32_t(std::uint64_t(physical)*c.kv_heads+head/groups);
          if(j<valid)mask[std::size_t(row)*l+j]=0;
        }
      }
    }
    // Native Gather also evaluates unused Q capacity. These reads must stay
    // inside the exact page footprint; row zero is the first actual query,
    // even when preceding requests have no Q. Preserve its GQA head mapping.
    for(std::size_t row=p->layout.facts().query_rows;row<c.max_query_rows;++row)
      for(std::size_t head=0;head<c.query_heads;++head)
        std::fill_n(indices+(row*c.query_heads+head)*l,l,indices[head*l]);
    HYLE_RETURN_IF_ERROR(mapping.close());
  }
  HYLE_ASSIGN_OR_RETURN(p->bytes,ReadLease::adopt(std::move(buffer),{.allocator=data_->allocator,.resource_budget=data_->options.resource_budget}));return result;
}
MaterializedBatchVersion::MaterializedBatchVersion(const MaterializedBatchVersion&v)noexcept:data_(v.data_){if(data_)data_->incref();}
MaterializedBatchVersion&MaterializedBatchVersion::operator=(const MaterializedBatchVersion&v)noexcept{if(this!=&v){MaterializedBatchVersion next(v);std::swap(data_,next.data_);}return *this;}
MaterializedBatchVersion::MaterializedBatchVersion(MaterializedBatchVersion&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
MaterializedBatchVersion&MaterializedBatchVersion::operator=(MaterializedBatchVersion&&v)noexcept{if(this!=&v){MaterializedBatchVersion next(std::move(v));std::swap(data_,next.data_);}return *this;}
MaterializedBatchVersion::~MaterializedBatchVersion(){if(data_)data_->decref();}
BatchLayoutVersion MaterializedBatchVersion::layout()const noexcept{return data_?data_->layout:BatchLayoutVersion{};}
MaterializedBatchProgram MaterializedBatchVersion::program()const noexcept{return data_?data_->program:MaterializedBatchProgram{};}
Result<ReadLease> MaterializedBatchVersion::indices()const noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("materialized version is empty"));return data_->bytes.subview(0,data_->program.costs().index_bytes);}
Result<ReadLease> MaterializedBatchVersion::mask()const noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("materialized version is empty"));const auto costs=data_->program.costs();return data_->bytes.subview(costs.index_bytes,costs.mask_bytes);}
} // namespace hyle::compute
