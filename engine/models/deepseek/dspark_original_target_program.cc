#include "dspark_original_target_program.h"

namespace dspark_original_target_program {
Value immutable(sem::Fragment &fragment,std::vector<Constant> &constants,
    const roles::ReadonlyRole &role,const c::TensorDesc &expected) {
  if(!role.owner||!role.lease||role.description!=expected)
    throw std::runtime_error("original target immutable role has a different layout or no ready lease");
  const auto value=fragment.constant(expected,{});
  constants.push_back({value,role});
  return value;
}
}
