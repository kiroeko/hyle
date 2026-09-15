#include "result_owner_fixture.h"

#include "hyle/compute/detail/generated_result.h"

namespace hyle::compute::test {

SharedResultOwner::~SharedResultOwner() noexcept {
  if (destroyed != nullptr)
    ++*destroyed;
}

Result<OwnedResult<SharedResultOwner>>
make_shared_result(int value, int &destroyed) noexcept {
  HYLE_ASSIGN_OR_RETURN(
      auto slot,
      detail::ResultSlot<SharedResultOwner>::create(mem::system_allocator()));
  HYLE_ASSIGN_OR_RETURN(auto write, slot.begin());
  HYLE_RETURN_IF_ERROR(write.emplace(SharedResultOwner(value, destroyed)));
  return write.publish(Completion::completed());
}

} // namespace hyle::compute::test
