#include <hyle/compute/cuda_weights.h>
#include <hyle/compute/prepared_operation.h>
#include <array>
#include <vector>
#include <thread>
#include <latch>
#include <cstdio>
#include <cmath>
namespace c=hyle::compute;namespace cu=c::cuda;namespace m=hyle::mem;
hyle::Status run(){
 HYLE_ASSIGN_OR_RETURN(auto ctx,cu::create_context());
 HYLE_ASSIGN_OR_RETURN(auto host,cu::MemoryResource::create(ctx,cu::MemoryKind::kPinnedHost));
 HYLE_ASSIGN_OR_RETURN(auto dev,cu::MemoryResource::create(ctx,cu::MemoryKind::kDevice));
 HYLE_ASSIGN_OR_RETURN(auto copy,ctx.create_queue());HYLE_ASSIGN_OR_RETURN(auto math,ctx.create_queue());
 HYLE_RETURN_IF_ERROR(copy.reserve_completions(8));HYLE_RETURN_IF_ERROR(math.reserve_completions(32));
 constexpr int n=2048;constexpr size_t bytes=size_t(n)*n*4;
 HYLE_ASSIGN_OR_RETURN(auto desc,c::TensorDesc::dense(c::DataType::kFloat32,std::array<int64_t,2>{n,n}));
 HYLE_ASSIGN_OR_RETURN(auto src,m::Buffer::create(m::system_resource(),bytes,256));
 HYLE_ASSIGN_OR_RETURN(auto xb,m::Buffer::create(dev.ref(),bytes,256));HYLE_ASSIGN_OR_RETURN(auto yb,m::Buffer::create(dev.ref(),bytes,256));
 HYLE_ASSIGN_OR_RETURN(auto source,c::ConstTensorView::create(desc,src.view()));HYLE_ASSIGN_OR_RETURN(auto x,c::ConstTensorView::create(desc,xb.view()));HYLE_ASSIGN_OR_RETURN(auto y,c::TensorView::create(desc,yb.view()));
 std::vector<float> vals(size_t(n)*n,1.0f/n);HYLE_RETURN_IF_ERROR(src.view().write(std::as_bytes(std::span(vals))));
 std::fill(vals.begin(),vals.end(),1.0f);HYLE_RETURN_IF_ERROR(xb.view().write(std::as_bytes(std::span(vals))));
 const std::array specs{cu::WeightSpec{7,desc}};HYLE_ASSIGN_OR_RETURN(auto pool,cu::WeightSlots::create(ctx,host,dev,specs,bytes));
 const std::array selected{cu::WeightFragment{7,source}};HYLE_ASSIGN_OR_RETURN(auto initial,pool.upload(0,selected,copy));HYLE_RETURN_IF_ERROR(initial.wait());
 HYLE_ASSIGN_OR_RETURN(auto w,initial.binding(7));HYLE_ASSIGN_OR_RETURN(auto program,c::prepare(ctx,c::MatMul{},x.description(),w.description(),dev.ref()));
 const auto need=program.workspace_requirement();std::array<m::Buffer,8> scratch;std::array<c::OperationInvocation,8> ops;std::array<c::Completion,8> done;
 for(size_t i=0;i<8;++i){HYLE_ASSIGN_OR_RETURN(scratch[i],m::Buffer::create(dev.ref(),need.size,need.alignment));HYLE_ASSIGN_OR_RETURN(ops[i],program.create_invocation(scratch[i].view()));}
 std::latch accepted(1);hyle::Status status;
 std::jthread worker([&]()noexcept{
  bool notified=false;
  for(size_t i=0;i<8;++i){auto result=ops[i].submit(math,std::array{x,w},std::array{y});if(!result){status=result.error();break;}done[i]=std::move(*result);if(!notified){accepted.count_down();notified=true;}}
  if(!notified)accepted.count_down();
  for(auto &finish:done)if(finish.valid()){auto end=finish.wait();if(status.ok())status=end;}
 });
 accepted.wait();auto migrated=pool.upload(1,selected,copy);worker.join();if(!status.ok())return status;
 if(!migrated)return migrated.error();HYLE_RETURN_IF_ERROR(migrated->wait());
 HYLE_RETURN_IF_ERROR(yb.view().read(std::as_writable_bytes(std::span(vals))));for(auto v:vals)if(std::abs(v-1.0f)>0.0002f)return hyle::Status::Internal("overlap math differs");
 puts("Native overlap probe: 8 mature FP32 GEMMs and independent selected-weight DMA completed correctly");return hyle::Status::Ok();
}
int main(){auto s=run();if(!s.ok()){fprintf(stderr,"%.*s\n",int(s.message().size()),s.message().data());return 1;}return 0;}
