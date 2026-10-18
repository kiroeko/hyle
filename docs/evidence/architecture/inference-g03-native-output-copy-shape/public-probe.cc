#include <hyle/compute/apple_memory.h>
#include <array>
#include <cstdio>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace {
namespace c=hyle::compute;namespace n=c::apple;namespace mx=mlx::core;
using hyle::Status;
Status require(bool ok,const char* message){return ok?Status::Ok():Status::Internal(message);}
Status run(){
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  struct Close{n::Context& context;~Close(){context.close().IgnoreError();}}close{context};
  const std::array specs{n::InputSpec{{1,2,256},mx::float16}};
  HYLE_ASSIGN_OR_RETURN(auto producer,context.prepare(c::Unary{c::UnaryKind::kAbs},specs));
  HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{1,2,256}));
  HYLE_ASSIGN_OR_RETURN(auto slot,producer.create_invocation());
  const auto bytes=desc.required_bytes().value();
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(device.ref(),2*bytes+512));
  const std::array fixed_specs{c::CopyResourceSpec{device.ref(),bytes},c::CopyResourceSpec{device.ref(),bytes}};
  HYLE_ASSIGN_OR_RETURN(auto fixed,n::CopyBatchProgram::prepare(context,fixed_specs));
  HYLE_ASSIGN_OR_RETURN(auto fixed_slot,n::CopyBatchInvocation::create(fixed));
  c::ReadLease previous;
  for(unsigned step=0;step<2;++step){
    auto source=mx::full(mx::Shape{1,2,256},-float(step+1)/8.F,mx::float16);
    mx::async_eval({source});
    mx::synchronize(mx::default_stream(mx::Device::gpu));
    source.wait();
    const std::array inputs{source};HYLE_RETURN_IF_ERROR(slot.bind(inputs));
    c::ReadLease ready;
    {HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit());HYLE_ASSIGN_OR_RETURN(auto result,submitted.wait());
      HYLE_ASSIGN_OR_RETURN(ready,device.import_readonly(result->arrays()[0],desc));}
    const auto declared=fixed_slot.bind_source({0},ready);
    HYLE_RETURN_IF_ERROR(require(declared.IsInvalidArgument(),"native producer source unexpectedly used fixed device resource identity"));
    const auto resource=ready.view()->resource();
    HYLE_RETURN_IF_ERROR(require(resource!=device.ref() && (!previous.valid() || previous.view()->resource()!=resource),"native exports did not have distinct retained resource identities"));
    const std::array dynamic_specs{c::CopyResourceSpec{resource,bytes},c::CopyResourceSpec{device.ref(),bytes}};
    HYLE_ASSIGN_OR_RETURN(auto copy_program,n::CopyBatchProgram::prepare(context,dynamic_specs));
    HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(copy_program));
    HYLE_ASSIGN_OR_RETURN(auto destination,pool.allocate(bytes));
    HYLE_RETURN_IF_ERROR(copy.bind_source({0},ready));HYLE_RETURN_IF_ERROR(copy.bind_mutable({1},std::move(destination)));
    const std::array<c::CopyPhase,1> phases{{{}}};const std::array<c::CopyRange,1> ranges{{{{0},{1},{0},0,0,bytes}}};
    HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));
    {HYLE_ASSIGN_OR_RETURN(auto submitted,copy.submit());HYLE_RETURN_IF_ERROR(submitted.wait());
      HYLE_ASSIGN_OR_RETURN(auto output,submitted.output({0},{1},0,bytes));
      std::array<std::uint16_t,512> actual{};HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
      for(auto value:actual)HYLE_RETURN_IF_ERROR(require(value==(step?0x3400:0x3000),"actual native producer/copied bytes differ"));}
    previous=std::move(ready);
    std::fprintf(stderr,"actual Apple ABS producer step=%u native output/read lease/copy PASS; fixed resource binding rejects; per-export copy program+slot required, 1024 bytes device->device\n",step);
  }
  return Status::Ok();
}
}
int main(){const auto status=run();if(!status.ok())std::fprintf(stderr,"native producer shape: %.*s\n",int(status.message().size()),status.message().data());return status.ok()?0:1;}
