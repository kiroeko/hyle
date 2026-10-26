#if defined(HYLE_G01_PAGE_RISK_APPLE)
#include "native_apple_failure_fixture.h"
#include "hyle/compute/detail/apple_copy_stream_access.h"
#else
#include "native_vulkan_failure_fixture.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#endif
#include "allocation_fault_fixture.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>

namespace {
namespace c=hyle::compute;namespace g=hyle::graph;
#if defined(HYLE_G01_PAGE_RISK_APPLE)
namespace n=c::apple;
#else
namespace n=c::vulkan;
#endif
using hyle::Status;
Status require(bool good,const char*message){return good?Status::Ok():Status::Internal(message);}
enum class Mode{success,invalid_bindings,foreign,allocation,cancel_before,cancel_after,abandon,shape,unknown,producer_failure,alias_reads,budget,dtype,readonly_destination};
Status episode(Mode mode){
  hyle_f5_fault_fixture::Allocator fault;
  HYLE_ASSIGN_OR_RETURN(auto context,n::Context::create());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context));
#else
  HYLE_ASSIGN_OR_RETURN(auto device,n::MemoryResource::create(context,n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging,n::MemoryResource::create(context,n::MemoryKind::kStaging));
#endif
  struct Close{n::Context&context;~Close(){context.close().IgnoreError();}}close{context};
  HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{16}));
  HYLE_ASSIGN_OR_RETURN(auto half,c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,1>{8}));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input,builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto abs,builder.add(c::Unary{c::UnaryKind::kAbs},input));
  for(unsigned i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(auto slice,c::View::slice(0,8*i,8));
    HYLE_ASSIGN_OR_RETURN(auto out,builder.add(slice,abs));HYLE_RETURN_IF_ERROR(builder.publish(out));}
  HYLE_ASSIGN_OR_RETURN(auto graph,std::move(builder).finish());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto program,g::apple::compile(context,graph));
  HYLE_ASSIGN_OR_RETURN(auto native_input,device.allocate(desc));
#else
  HYLE_ASSIGN_OR_RETURN(auto program,g::vulkan::compile(context,graph));
#endif
  HYLE_ASSIGN_OR_RETURN(auto producer,program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto host_pool,c::LeasePool::create(staging.ref(),256));
  HYLE_ASSIGN_OR_RETURN(auto host_write,host_pool.allocate(64,64));
  std::array<float,16> words;for(unsigned i=0;i<16;++i)words[i]=-float(i+1);
  HYLE_RETURN_IF_ERROR(host_write.view()->write(std::as_bytes(std::span(words))));
  HYLE_ASSIGN_OR_RETURN(auto host,host_write.publish());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  HYLE_ASSIGN_OR_RETURN(auto tensor,host.tensor(desc));HYLE_RETURN_IF_ERROR(device.copy_from_host(native_input,tensor));
#else
  HYLE_ASSIGN_OR_RETURN(auto native_input,context.upload(desc,std::as_bytes(std::span(words))));
#endif
  const std::array args{native_input};HYLE_RETURN_IF_ERROR(producer.bind(args));
  c::ResourceBudget budget;
  if(mode==Mode::budget){
    const std::array limits{c::ResourceBudgetLimit{hyle::mem::system_resource(),32u<<20},
        c::ResourceBudgetLimit{device.ref(),1u<<20},c::ResourceBudgetLimit{staging.ref(),1u<<20}};
    HYLE_ASSIGN_OR_RETURN(budget,c::ResourceBudget::create({.limits=limits,.allocator=fault.ref(),.allow_unattributed=true}));
  }
  const std::array specs{c::CopyResourceSpec{device.ref(),32},c::CopyResourceSpec{device.ref(),32},c::CopyResourceSpec{staging.ref(),64}};
  HYLE_ASSIGN_OR_RETURN(auto copies,n::CopyBatchProgram::prepare(context,specs,{.allocator=fault.ref(),.resource_budget=budget}));
  HYLE_ASSIGN_OR_RETURN(auto copy,n::CopyBatchInvocation::create(copies));
  HYLE_ASSIGN_OR_RETURN(auto destination_pool,c::LeasePool::create(staging.ref(),256));
  HYLE_ASSIGN_OR_RETURN(auto destination,destination_pool.allocate(64,64));
  HYLE_RETURN_IF_ERROR(copy.bind_mutable({2},std::move(destination)));
#if defined(HYLE_G01_PAGE_RISK_APPLE)
  namespace mx=mlx::core;
  auto gate=std::make_shared<hyle_f5_apple_failure_fixture::Gate>();
  struct Release{decltype(gate) held;n::Context&context;~Release(){held->release();context.close().IgnoreError();}}release{gate,context};
  const std::array blocker_specs{n::InputSpec{{4},mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto blocker,context.compile([gate](const auto&inputs,mx::Stream stream){
    auto first=mx::square(inputs[0],stream);return std::vector{mx::array(first.shape(),first.dtype(),
        std::make_shared<hyle_f5_apple_failure_fixture::Checkpoint>(stream,gate),{first})};},blocker_specs));
  HYLE_ASSIGN_OR_RETURN(auto blocking,blocker.create_invocation());const std::array blocker_inputs{mx::array({1.F,2.F,3.F,4.F})};
  HYLE_RETURN_IF_ERROR(blocking.bind(blocker_inputs));
  {std::lock_guard lock(gate->mutex);gate->armed=true;gate->fail=mode==Mode::producer_failure;}
  HYLE_ASSIGN_OR_RETURN(auto blocking_work,blocking.submit());
  HYLE_RETURN_IF_ERROR(require(gate->await(),"late source risk missed original native worker checkpoint"));
  const auto release_native=[&]{gate->release();};
#else
  hyle_f5_vk_failure_fixture::Boundary boundary(context);
  static std::atomic<bool> masked{true};masked=true;
  boundary.device.fn.GetFenceStatus=[](VkDevice device,VkFence fence)->VkResult{
    auto&hook=*hyle_f5_vk_failure_fixture::Boundary::active;
    if(hook.fail_poll.exchange(false))return VK_ERROR_OUT_OF_HOST_MEMORY;
    return masked.load()?VK_NOT_READY:hook.original.GetFenceStatus(device,fence);
  };
  struct Release{n::Context&context;~Release(){masked=false;context.close().IgnoreError();}}release{context};
  const auto release_native=[&]{masked=false;};
#endif
  HYLE_ASSIGN_OR_RETURN(auto produced,producer.submit());
  const std::array<n::ResultCopyBinding,2> bindings{{{{0},0,half},{{1},mode==Mode::alias_reads?0u:1u,half}}};
  if(mode==Mode::budget){
    HYLE_ASSIGN_OR_RETURN(auto before,budget.usage(hyle::mem::system_resource()));
    HYLE_ASSIGN_OR_RETURN(auto full,budget.reserve(hyle::mem::system_resource(),before.limit_bytes-before.live_bytes));
    const auto refused=copy.bind_source(bindings,std::move(produced));
    HYLE_RETURN_IF_ERROR(require(refused.IsResourceExhausted() && produced.valid(),"late result source bypassed shared metadata budget"));
    full.reset();HYLE_ASSIGN_OR_RETURN(auto after,budget.usage(hyle::mem::system_resource()));
    HYLE_RETURN_IF_ERROR(require(after.live_bytes==before.live_bytes,"late result budget rejection leaked requested-byte charge"));
  }
  if(mode==Mode::cancel_before)HYLE_RETURN_IF_ERROR(produced.request_cancel());
  if(mode==Mode::invalid_bindings){
    const auto empty=copy.bind_source(std::span<const n::ResultCopyBinding>{},std::move(produced));
    HYLE_RETURN_IF_ERROR(require(empty.IsInvalidArgument() && produced.valid(),"empty result binding consumed accepted owner"));
    auto duplicate=bindings;duplicate[1].resource={0};
    HYLE_RETURN_IF_ERROR(require(copy.bind_source(duplicate,std::move(produced)).IsInvalidArgument() && produced.valid(),
        "duplicate result resource consumed accepted owner"));
    auto invalid=bindings;invalid[0].resource={99};
    HYLE_RETURN_IF_ERROR(require(copy.bind_source(invalid,std::move(produced)).IsInvalidArgument() && produced.valid(),
        "foreign result resource ID consumed accepted owner"));
    auto too_large=bindings;too_large[0].description=desc;
    HYLE_RETURN_IF_ERROR(require(copy.bind_source(too_large,std::move(produced)).IsOutOfRange() && produced.valid(),
        "result source nominated capacity exceeded and consumed owner"));
  }
  if(mode==Mode::allocation){
    const auto baseline=fault.live;bool complete=false;std::size_t count=0;
    for(std::size_t nth=0;nth<32;++nth){fault.fail=fault.attempts+nth;
      auto accepted=copy.bind_source(bindings,std::move(produced));fault.fail=SIZE_MAX;
      if(accepted.ok()){complete=true;count=nth;break;}
      HYLE_RETURN_IF_ERROR(require(accepted.IsResourceExhausted() && produced.valid() && fault.live==baseline,
          "late native source cold OOM consumed producer or leaked metadata"));}
    HYLE_RETURN_IF_ERROR(require(complete,"late native source allocation has no finite success boundary"));
    std::fprintf(stderr,"late native source all %zu cold binding allocation refusals preserve producer PASS\n",count);
  }else if(mode==Mode::foreign){
    HYLE_ASSIGN_OR_RETURN(auto other,n::Context::create());
#if defined(HYLE_G01_PAGE_RISK_APPLE)
    HYLE_ASSIGN_OR_RETURN(auto other_device,n::MemoryResource::create(other));
    HYLE_ASSIGN_OR_RETURN(auto other_staging,n::MemoryResource::create(other));
#else
    HYLE_ASSIGN_OR_RETURN(auto other_device,n::MemoryResource::create(other,n::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(auto other_staging,n::MemoryResource::create(other,n::MemoryKind::kStaging));
#endif
    const std::array other_specs{c::CopyResourceSpec{other_device.ref(),32},c::CopyResourceSpec{other_device.ref(),32},c::CopyResourceSpec{other_staging.ref(),64}};
    HYLE_ASSIGN_OR_RETURN(auto other_program,n::CopyBatchProgram::prepare(other,other_specs));
    HYLE_ASSIGN_OR_RETURN(auto other_copy,n::CopyBatchInvocation::create(other_program));
    const auto denied=other_copy.bind_source(bindings,std::move(produced));
    HYLE_RETURN_IF_ERROR(require(!denied.ok() && produced.valid(),"late source accepted a foreign original Runtime or consumed owner"));
    release_native();HYLE_RETURN_IF_ERROR(produced.completion().wait());HYLE_RETURN_IF_ERROR(other.close());
    std::fprintf(stderr,"late native source foreign Runtime refuses without consuming producer PASS\n");return Status::Ok();
  }else if(mode==Mode::shape || mode==Mode::dtype){
    auto bad=bindings;
    if(mode==Mode::shape)bad[1].output_index=99;
    else{HYLE_ASSIGN_OR_RETURN(bad[1].description,c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,1>{16}));}
    const auto accepted=copy.bind_source(bad,std::move(produced));
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
    HYLE_RETURN_IF_ERROR(require(!accepted.ok() && produced.valid(),"Vulkan cold output shape rejection consumed owner"));
    release_native();HYLE_RETURN_IF_ERROR(produced.completion().wait());
    std::fprintf(stderr,"Vulkan late source invalid native output rejected cold, original owner preserved PASS\n");return Status::Ok();
#else
    HYLE_RETURN_IF_ERROR(accepted); // Actual arrays exist only after worker execution.
#endif
  }else{
    const auto accepted=copy.bind_source(bindings,std::move(produced));
    if(mode==Mode::cancel_before){
      HYLE_RETURN_IF_ERROR(require(accepted.IsCancelled() && produced.valid(),"late source known cancellation consumed owner or accepted source"));
      release_native();produced.completion().wait().IgnoreError();
      std::fprintf(stderr,"late native source cancelled producer refused before transfer PASS\n");return Status::Ok();
    }
    HYLE_RETURN_IF_ERROR(accepted);
  }
  HYLE_RETURN_IF_ERROR(require(!produced.valid() && !producer.submit(),"late source lost original result/producer slot ownership"));
  HYLE_RETURN_IF_ERROR(require(!copy.bind_source({0},host).ok() && !copy.bind_mutable({0},c::WriteLease{}).ok(),
      "late source result ownership could be overwritten by ordinary bindings"));
  const std::array<c::CopyPhase,2> phases{{{}, {.dependency_mask=1}}};
  if(mode==Mode::readonly_destination){
    const std::array<c::CopyRange,1> forbidden{{{{1},{0},{0},0,0,32}}};
    HYLE_RETURN_IF_ERROR(require(!copy.prepare_replay(phases,forbidden).ok() && !producer.submit() &&
        destination_pool.stats().active_uses==0,"native result readonly alias acquired write permission or accepted copy work"));
    release_native();HYLE_RETURN_IF_ERROR(context.close());
    std::fprintf(stderr,"late native result readonly alias rejects destination writes before copy acceptance, retains original result PASS\n");
    return Status::Ok();
  }
  const std::array<c::CopyRange,2> ranges{{{{1},{2},{0},0,0,32},{{0},{2},{1},0,32,32}}};
  HYLE_RETURN_IF_ERROR(copy.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto work,copy.submit());
  HYLE_RETURN_IF_ERROR(require(!work.output({1},{2},0,64) && !copy.unbind_mutable({2}),"pending late source published ready destination"));
  if(mode==Mode::cancel_after)HYLE_RETURN_IF_ERROR(work.request_cancel());
#if !defined(HYLE_G01_PAGE_RISK_APPLE)
  if(mode==Mode::unknown){boundary.fail_poll=true;const auto diagnosis=n::detail::CopyUpdateAccess::observe(copy);
    HYLE_RETURN_IF_ERROR(require(!diagnosis.ok() && !boundary.fail_poll.load() &&
        destination_pool.stats().active_uses>0 && !work.output({1},{2},0,64),"late source unknown observation released actual destination use"));}
#endif
  if(mode==Mode::abandon){
    work={};copy={};HYLE_RETURN_IF_ERROR(require(destination_pool.stats().active_uses>0 && !producer.submit(),
        "late source abandonment released producer/storage before actual copy retirement"));
    release_native();HYLE_RETURN_IF_ERROR(context.close());
    HYLE_RETURN_IF_ERROR(require(destination_pool.stats().active_uses==0,"abandoned late source copy did not retire destination guards"));
  }else{
    release_native();const auto terminal=work.wait();
    const bool failed=mode==Mode::cancel_after || mode==Mode::shape || mode==Mode::dtype || mode==Mode::unknown || mode==Mode::producer_failure;
    HYLE_RETURN_IF_ERROR(require(failed?!terminal.ok():terminal.ok(),"late source final outcome did not preserve native error/success"));
    if(failed)HYLE_RETURN_IF_ERROR(require(!work.output({1},{2},0,64) && !copy.unbind_mutable({2}),
        "late source failed actual prefix published successful destination"));
    else{
      HYLE_ASSIGN_OR_RETURN(auto output,work.output({1},{2},0,64));std::array<float,16> actual;
      HYLE_RETURN_IF_ERROR(output.view()->read(std::as_writable_bytes(std::span(actual))));
      for(unsigned i=0;i<16;++i)HYLE_RETURN_IF_ERROR(require(actual[i]==float(mode==Mode::alias_reads?i%8+1:(i+8)%16+1),
          "late source multi-output alias/phase/root permutation differs"));
      HYLE_RETURN_IF_ERROR(require(!producer.submit(),"old source controls allowed another producer result before reset"));
    }
    HYLE_RETURN_IF_ERROR(require(destination_pool.stats().active_uses==0,"late source final observation did not retire actual guards"));
    work={};copy={};
  }
  std::fprintf(stderr,"late native source mode=%u actual owner/phase/error/retirement PASS\n",unsigned(mode));return Status::Ok();
}
}
int main(){for(auto mode:{Mode::success,Mode::invalid_bindings,Mode::foreign,Mode::allocation,Mode::cancel_before,Mode::cancel_after,Mode::abandon,Mode::shape,Mode::alias_reads,Mode::budget,Mode::dtype,Mode::readonly_destination,
#if defined(HYLE_G01_PAGE_RISK_APPLE)
    Mode::producer_failure
#else
    Mode::unknown
#endif
  }){const auto status=episode(mode);if(!status.ok()){std::fprintf(stderr,"late native source risk mode=%u: %.*s\n",unsigned(mode),int(status.message().size()),status.message().data());return 1;}}
  return 0;
}
