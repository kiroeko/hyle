#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/compute/storage_owner.h"
#include "hyle/compute/tensor_desc.h"
#include "hyle/graph/executable.h"

// Shared ready-role identity, storage and original upload contract. Source
// authentication and native GGUF metadata remain in their existing adapter.
namespace original_named_role_source {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
struct Identity final {
  std::string_view repository,revision,variant,file;
};
struct ReadonlyRole final {
  std::string_view name;
  c::TensorDesc description;
  c::StorageOwner owner;
  c::ReadLease lease;
  const Identity *identity=nullptr;
  hyle::Result<g::ConstantBinding> binding(g::ValueId value) const {
    HYLE_ASSIGN_OR_RETURN(auto tensor,lease.tensor(description));
    return g::ConstantBinding{.value=value,.tensor=tensor,.owner=owner,.lease=lease};
  }
};
namespace detail {
template<class T> T take(hyle::Result<T> result,std::string_view where) {
  if(!result) throw std::runtime_error(std::string(where)+": "+std::string(result.error().message()));
  return std::move(*result);
}
inline ReadonlyRole bind(std::string_view name,const c::TensorDesc &description,
    const c::StorageOwner &owner,std::span<const std::byte> source,const Identity *identity) {
  if(source.size()!=take(description.required_bytes(),"original role extent"))
    throw std::runtime_error("original role byte extent differs from its exact typed description");
  auto storage=take(m::Buffer::create(owner.ref(),source.size(),256),"original role native backing");
  const auto status=storage.view().write(source);
  if(!status.ok()) throw std::runtime_error("original role upload: "+std::string(status.message()));
  auto ready=take(owner.adopt_buffer(std::move(storage)),"original role readonly owner");
  return {name,description,owner,std::move(ready),identity};
}
} // namespace detail

} // namespace original_named_role_source
