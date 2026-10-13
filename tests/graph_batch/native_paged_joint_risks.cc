#include "native_paged_joint_lifecycle.h"
namespace {
namespace r=hyle_joint_page_risk;
using hyle::Status;
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto cpu,r::g::cpu::create_context());HYLE_ASSIGN_OR_RETURN(auto native,r::n::Context::create());
#if defined(HYLE_JOINT_PAGE_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,r::n::MemoryResource::create(native));HYLE_ASSIGN_OR_RETURN(auto staging,r::n::MemoryResource::create(native));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,r::n::MemoryResource::create(native,r::n::MemoryKind::kDevice));HYLE_ASSIGN_OR_RETURN(auto staging,r::n::MemoryResource::create(native,r::n::MemoryKind::kStaging));
#endif
  HYLE_ASSIGN_OR_RETURN(auto data,r::r::inputs(native,device,staging));
  HYLE_RETURN_IF_ERROR(r::domain_checks(cpu,native,device,staging,data));
  HYLE_RETURN_IF_ERROR(r::cold_failures(cpu,native,device,staging,data));
#if defined(HYLE_JOINT_PAGE_APPLE)
  for(unsigned mode:{0U,1U,2U,3U,5U,6U})HYLE_RETURN_IF_ERROR(r::lifecycle(mode));
#else
  for(unsigned mode=0;mode<6;++mode)HYLE_RETURN_IF_ERROR(r::lifecycle(mode));
#endif
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok()){std::fprintf(stderr,"joint risk FAIL: %.*s\n",int(status.message().size()),status.message().data());return 1;}return 0;}
