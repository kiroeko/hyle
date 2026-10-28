#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/cuda.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/prepared_operation.h>
#include <hyle/compute/providers/cuda.h>
#include <hyle/mem/buffer.h>

namespace {
namespace c=hyle::compute;
namespace mem=hyle::mem;
template<class T> T take(hyle::Result<T> result) {
  if(!result)throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if(!status.ok())throw std::runtime_error(std::string(status.message()));
}
void run() {
  constexpr std::int64_t n=9,k=128,group=64;
  auto context=take(c::cuda::create_context());
  auto memory=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));
  auto queue=take(context.create_queue());
  const std::array descriptions{
    take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{1,k})),
    take(c::TensorDesc::dense(c::DataType::kUInt32,std::array{n,k/8})),
    take(c::TensorDesc::dense(c::DataType::kFloat32,std::array{n,k/group})),
    take(c::TensorDesc::dense(c::DataType::kFloat32,std::array{n,k/group}))};
  std::array<mem::Buffer,4> buffers;
  std::array<c::ConstTensorView,4> views;
  for(std::size_t i=0;i<buffers.size();++i) {
    buffers[i]=take(mem::Buffer::create(memory.ref(),take(descriptions[i].required_bytes()),256));
    views[i]=take(c::ConstTensorView::create(descriptions[i],buffers[i].view()));
  }
  std::vector<std::uint32_t> words(n*k/8);
  for(int row=0;row<n;++row)for(int j=0;j<k/8;++j)words[row*(k/8)+j]=0x11111111U*unsigned(row+1);
  const std::vector<float> scales(n*k/group,.5F),biases(scales.size(),-.25F);
  require(buffers[1].view().write(std::as_bytes(std::span(words))));
  require(buffers[2].view().write(std::as_bytes(std::span(scales))));
  require(buffers[3].view().write(std::as_bytes(std::span(biases))));
  const auto operation=c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine,4,group);
  std::array<c::TensorDesc,1> output_desc;
  require(c::infer_output_descriptions(operation,descriptions,output_desc));
  auto program=take(c::prepare(context,operation,descriptions,output_desc,memory.ref()));
  if(program.runtime_properties().blocks_host)throw std::runtime_error("SDK did not select native packed execution");
  auto first=take(program.create_invocation()),second=take(program.create_invocation());
  std::array<mem::Buffer,3> outputs;
  for(auto& buffer:outputs)buffer=take(mem::Buffer::create(memory.ref(),n*sizeof(float),256));
  std::vector<float> activation(k),retained;
  for(int step=0;step<64;++step) {
    activation[0]=1.F+float(step)*.125F;activation[1]=-.25F;
    require(buffers[0].view().write(std::as_bytes(std::span(activation))));
    const auto bank=step==0?0:1+step%2;
    const std::array destination{take(c::TensorView::create(output_desc[0],outputs[bank].view()))};
    auto work=take((step%2?second:first).submit(queue,views,destination));require(work.wait());
    std::vector<float> actual(n);require(outputs[bank].view().read(std::as_writable_bytes(std::span(actual))));
    for(int row=0;row<n;++row)
      if(actual[row]!=(activation[0]+activation[1])*(float(row+1)*.5F-.25F))
        throw std::runtime_error("resident packed SDK independent oracle differs");
    if(step==0)retained=actual;
  }
  std::vector<float> old(n);require(outputs[0].view().read(std::as_writable_bytes(std::span(old))));
  if(old!=retained)throw std::runtime_error("resident packed SDK changed old output");
}
} // namespace
int main() {
  try {run();std::puts("PASS CUDA public resident packed consumer: 64 steps, two slots, retained output");return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
