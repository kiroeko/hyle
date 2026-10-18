#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>
#include "hyle/compute/cpu.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
template<class T> std::vector<std::byte> bytes(const std::vector<T>& data) {
  const auto view=std::as_bytes(std::span(data));return {view.begin(),view.end()};
}
c::TensorDesc dense(c::DataType type,std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type,{shape.begin(),shape.size()}).value();
}
void require(bool ok) {if(!ok)throw std::runtime_error("CPU delivery consumer failed");}
std::vector<std::byte> execute(bool graph,c::Operation op,const std::vector<c::TensorDesc>& descs,
    const std::vector<std::vector<std::byte>>& data) {
  std::vector<m::Buffer> owners;std::vector<c::ConstTensorView> inputs;
  for(std::size_t i=0;i<descs.size();++i) {
    owners.push_back(m::Buffer::create(m::system_resource(),data[i].size(),64).value());
    require(owners.back().view().write(data[i]).ok());
    inputs.push_back(c::ConstTensorView::create(descs[i],owners.back().view()).value());
  }
  std::array<c::TensorDesc,1> inferred;require(c::infer_output_descriptions(op,descs,inferred).ok());
  auto owner=m::Buffer::create(m::system_resource(),inferred[0].required_bytes().value(),64).value();
  const auto output=c::TensorView::create(inferred[0],owner.view()).value();
  if(graph) {
    auto context=g::cpu::create_context().value();auto builder=g::GraphBuilder::create().value();
    std::vector<g::ValueId> values;for(const auto& desc:descs)values.push_back(builder.input(desc).value());
    auto node=builder.add_node(op,values).value();auto result=builder.result(node).value();
    require(builder.publish(result).ok());auto compiled=g::compile(std::move(builder).finish().value(),context).value();
    auto invocation=compiled.executable.create_invocation().value();
    for(std::size_t i=0;i<inputs.size();++i)require(invocation.bind(values[i],inputs[i]).ok());
    require(invocation.bind(result,output).ok());auto queue=context.create_queue().value();
    const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
    require(invocation.prepare_replay(queues).ok());
    for(int i=0;i<2;++i)require(invocation.submit(queue).value().wait().ok());
  }else{
    auto context=c::cpu::create_context().value();auto prepared=c::prepare(context,op,inputs,std::span(&output,1)).value();
    auto invocation=prepared.create_invocation().value();auto queue=context.create_queue().value();
    for(int i=0;i<2;++i)require(invocation.submit(queue,inputs,std::span(&output,1)).value().wait().ok());
  }
  std::vector<std::byte> result(inferred[0].required_bytes().value());require(owner.view().read(result).ok());return result;
}
void floats(const std::vector<std::byte>& output,float expected) {
  for(std::size_t i=0;i<output.size();i+=4){float value;std::memcpy(&value,output.data()+i,4);if(std::abs(value-expected)>=0.0001f)std::cerr<<"float actual="<<value<<" expected="<<expected<<'\n';require(std::abs(value-expected)<0.0001f);}
}
void put_half(std::vector<std::byte>& data,std::size_t offset,std::uint16_t word) {std::memcpy(data.data()+offset,&word,2);}
int main(){try{
  using D=c::DataType;using Q=c::GgufQuantizedType;
  using F=c::model_ops::WeightArtifactFormat;using M=c::model_ops::MlxWeightFormat;
  int checks=0;
  for(bool graph:{false,true}) {
    for(auto type:{Q::kQ4_0,Q::kQ4_1,Q::kQ2K,Q::kIq2Xxs,Q::kIq2Xs,Q::kIq2S,
                   Q::kIq1S,Q::kIq1M,Q::kIq1Xxxs,Q::kIq1Xxs,Q::kIq1Xs}) {
      std::cout<<"graph="<<graph<<" GGUF="<<int(type)<<std::endl;
      const auto k=type==Q::kQ4_0||type==Q::kQ4_1?64:256;
      const auto w=c::gguf_weight_description(type,D::kFloat32,k,2).value();
      std::vector<std::byte> packed(w.required_bytes().value());float expected=0;
      if(type==Q::kQ4_0||type==Q::kQ4_1) {
        const auto block=type==Q::kQ4_0?18:20;
        for(std::size_t base=0;base<packed.size();base+=block){
          put_half(packed,base,0x3c00);
          if(type==Q::kQ4_1)put_half(packed,base+2,0x3800);
          std::fill_n(packed.begin()+base+(type==Q::kQ4_0?2:4),16,std::byte(type==Q::kQ4_0?0x99:0x11));
        }
        // Original Q8_0/Q8_1 activation uses half(1/127) = 0x2008.
        // Q8_1's separately stored half sum is exactly 32 per block.
        expected=type==Q::kQ4_0?63.99609375f:95.99609375f;
      }else if(type==Q::kQ2K){
        for(std::size_t base=0;base<packed.size();base+=84){
          std::fill_n(packed.begin()+base,16,std::byte{0x11});
          std::fill_n(packed.begin()+base+16,64,std::byte{0x55});
          put_half(packed,base+80,0x3c00);put_half(packed,base+82,0x3400);
        }expected=192;
      }
      floats(execute(graph,c::Operation(c::MatMul{}), {dense(D::kFloat32,{1,k}),w},
                     {bytes(std::vector<float>(k,1)),packed}),expected);++checks;
    }
    for(auto format:{F::kNf4,F::kNf4DoubleQuant,F::kAwqGemm4,F::kCompressedTensorsSymmetric4}) {
      std::cout<<"graph="<<graph<<" artifact="<<int(format)<<std::endl;
      const auto group=format==F::kNf4||format==F::kNf4DoubleQuant?64:128;
      std::vector<c::TensorDesc> descs;std::vector<std::vector<std::byte>> payload;
      if(format==F::kNf4||format==F::kNf4DoubleQuant){
        descs={dense(D::kUInt8,{2,128})};payload={bytes(std::vector<std::uint8_t>(256,0x11))};
        if(format==F::kNf4){descs.push_back(dense(D::kFloat32,{8}));payload.push_back(bytes(std::vector<float>(8,1)));}
        else{
          descs.insert(descs.end(),{dense(D::kUInt8,{8}),dense(D::kFloat32,{1}),dense(D::kFloat32,{256}),dense(D::kFloat32,{})});
          payload.insert(payload.end(),{bytes(std::vector<std::uint8_t>(8,1)),bytes(std::vector<float>{1}),
                         bytes(std::vector<float>(256,1)),bytes(std::vector<float>{0})});
        }
        descs.push_back(dense(D::kFloat32,{16}));payload.push_back(bytes(std::vector<float>(16,1)));
      }else if(format==F::kAwqGemm4){
        descs={dense(D::kInt32,{256,1}),dense(D::kInt32,{2,1}),dense(D::kFloat16,{2,8})};
        payload={bytes(std::vector<std::int32_t>(256,0x11111111)),bytes(std::vector<std::int32_t>(2,0)),bytes(std::vector<std::uint16_t>(16,0x3c00))};
      }else{
        descs={dense(D::kInt32,{2,32}),dense(D::kBFloat16,{2,2})};
        payload={bytes(std::vector<std::int32_t>(64,-1717986919)),bytes(std::vector<std::uint16_t>(4,0x3f80))};
      }
      auto op=c::model_ops::artifact_weight_decode(format,group);
      std::array<c::TensorDesc,1> inferred;require(c::infer_output_descriptions(op,descs,inferred).ok());
      auto decoded=execute(graph,op,descs,payload);
      const auto dtype=inferred[0].data_type();const auto n=inferred[0].dimensions()[0];
      const auto one=dtype==D::kFloat16?0x3c00:0x3f80;
      const auto projected=execute(graph,c::Operation(c::MatMul{}), {dense(dtype,{1,256}),inferred[0].transposed().value()},
                    {bytes(std::vector<std::uint16_t>(256,std::uint16_t(one))),decoded});
      require(projected.size()==std::size_t(n*2));
      for(std::size_t i=0;i<projected.size();i+=2){std::uint16_t value;std::memcpy(&value,projected.data()+i,2);if(value!=(dtype==D::kFloat16?0x5c00:0x4380))std::cerr<<"half actual="<<value<<" dtype="<<int(dtype)<<'\n';require(value==(dtype==D::kFloat16?0x5c00:0x4380));}
      ++checks;
    }
    for(auto format:{M::kAffine4,M::kMxfp4,M::kMxfp8}){
      const bool affine=format==M::kAffine4;
      const auto words=format==M::kMxfp8?16:8;
      const auto code=format==M::kMxfp8?0x38383838u:affine?0x11111111u:0x22222222u;
      std::vector<c::TensorDesc> descs{dense(D::kFloat32,{1,2,64}),dense(D::kUInt32,{2,3,words}),
            dense(affine?D::kFloat32:D::kUInt8,{2,3,affine?1:2})};
      std::vector<std::vector<std::byte>> payload{bytes(std::vector<float>(128,1)),bytes(std::vector<std::uint32_t>(6*words,code)),
          affine?bytes(std::vector<float>(6,1)):bytes(std::vector<std::uint8_t>(12,127))};
      if(affine){descs.push_back(descs.back());payload.push_back(bytes(std::vector<float>(6,0)));}
      descs.push_back(dense(D::kInt32,{1,2}));payload.push_back(bytes(std::vector<std::int32_t>{1,1}));
      floats(execute(graph,c::model_ops::mlx_grouped_matmul(format,affine?64:32),descs,payload),64);++checks;
    }
  }
  std::cout<<"ordinary CPU Direct/Graph delivery checks="<<checks<<" PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
