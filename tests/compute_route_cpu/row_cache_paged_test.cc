#include "hyle/compute/cpu.h"
#include "../compute_models/row_cache_paged_fixture.h"
TEST(CpuPagedRowCache, ReadyFramesMapRequestLogicalIdsAndDecodeOnlySelectedRows){
  auto context=hyle::compute::cpu::create_context().value();
  row_cache_paged_test::exercise(context,hyle::mem::system_resource());
}
