#pragma once

#include <algorithm>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::detail {
// Common bounded permission metadata only. Native routes own their own
// invocation, view construction and submit; no queue/execution interface here.
class NativeLeaseInputs final {
public:
  explicit NativeLeaseInputs(mem::AllocatorRef allocator) : reads_(mem::StdAllocator<ReadLease>(allocator)) {}
  Status prepare(std::size_t count, LeaseImportOptions options) noexcept {
    if (count >= UINT32_MAX) return Status::ResourceExhausted("native lease input capacity");
    HYLE_ASSIGN_OR_RETURN(uses_, LeaseUsePool::create({.max_batches = 1,
        .max_resources_per_batch = static_cast<std::uint32_t>((std::max)(count, std::size_t{1})),
        .allocator = options.allocator, .resource_budget = options.resource_budget}));
    try { reads_.resize(count); }
    catch (...) { return Status::ResourceExhausted("native lease input metadata"); }
    return Status::Ok();
  }
  std::size_t size() const noexcept { return reads_.size(); }
  bool active() const noexcept { return uses_.active_batches() != 0; }
  bool closed() const noexcept { return closed_; }
  void retain(std::size_t index, const ReadLease &source) noexcept { reads_[index] = source; }
  Result<LeaseUseBatch> admit() noexcept {
    HYLE_ASSIGN_OR_RETURN(auto batch, uses_.begin());
    for (const auto &source : reads_) HYLE_RETURN_IF_ERROR(batch.read(source));
    return batch;
  }
  Status close() noexcept { closed_ = true; return uses_.close(); }
  Status drain() noexcept { closed_ = true; return uses_.drain(); }
private:
  std::vector<ReadLease, mem::StdAllocator<ReadLease>> reads_;
  LeaseUsePool uses_;
  bool closed_ = false;
};
// Cold native F2 source ownership only. One accepted result can expose several
// outputs while retaining exactly one original completion observer. No ready
// byte permission, new result slot or execution mechanism is created here.
template<class T>class NativeCopyResults final {
  struct Input final {
    OwnedResult<T> owner;
    std::size_t group=SIZE_MAX,output=0;
    TensorDesc description;
  };
public:
  explicit NativeCopyResults(mem::AllocatorRef a):inputs_(mem::StdAllocator<Input>(a)),
      ids_(mem::StdAllocator<CopyResourceId>(a)){}
  Status reserve(std::size_t count)noexcept{
    try{inputs_.resize(count);ids_.reserve(count);return Status::Ok();}
    catch(...){return Status::ResourceExhausted("native copy result source metadata");}
  }
  bool owns(std::size_t id)const noexcept{return id<inputs_.size() && inputs_[id].group!=SIZE_MAX;}
  std::span<const CopyResourceId> ids()const noexcept{return ids_;}
  template<class Row>Status validate(std::span<const Row> rows,std::span<const CopyResourceSpec> specs,
      const OwnedResult<T>&result)const noexcept{
    if(!result.valid() || rows.empty())return Status::InvalidArgument("native copy needs accepted result bindings");
    for(std::size_t i=0;i<rows.size();++i){const auto&row=rows[i];
      if(row.resource.value>=inputs_.size() || owns(row.resource.value) ||
          !row.description.valid() || !row.description.is_plain() || !row.description.is_dense())
        return Status::InvalidArgument("native result source requires an unused dense plain binding");
      HYLE_ASSIGN_OR_RETURN(auto bytes,row.description.required_bytes());
      if(!bytes || bytes>specs[row.resource.value].capacity_bytes)
        return Status::OutOfRange("native result source exceeds its nominated capacity");
      for(std::size_t j=0;j<i;++j)if(rows[j].resource==row.resource)
        return Status::InvalidArgument("duplicate native result source binding");
    }return Status::Ok();
  }
  template<class Row>Completion take(std::span<const Row> rows,OwnedResult<T>&&result)noexcept{
    const auto group=rows.front().resource.value;
    for(const auto&row:rows){auto&input=inputs_[row.resource.value];
      input.group=group;input.output=row.output_index;input.description=row.description;ids_.push_back(row.resource);
    }
    inputs_[group].owner=std::move(result);
    return ResultAccess::replace_completion(inputs_[group].owner,{});
  }
  const void*root(std::size_t id)const noexcept{return ResultAccess::storage(inputs_[inputs_[id].group].owner);}
  const T&payload(std::size_t id)const noexcept{
    return *static_cast<const T*>(result_payload(ResultAccess::storage(inputs_[inputs_[id].group].owner)));
  }
  std::size_t output(std::size_t id)const noexcept{return inputs_[id].output;}
  const TensorDesc&description(std::size_t id)const noexcept{return inputs_[id].description;}
  void reset()noexcept{for(auto&input:inputs_)input={};ids_.clear();}
private:
  std::vector<Input,mem::StdAllocator<Input>> inputs_;
  std::vector<CopyResourceId,mem::StdAllocator<CopyResourceId>> ids_;
};
} // namespace hyle::compute::detail
