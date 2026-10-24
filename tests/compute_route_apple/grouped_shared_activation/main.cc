#include <hyle/compute/apple.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/low_precision.h>
#include <hyle/graph/apple.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <array>
#include <bit>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../compute_models/grouped_shared_activation_fixture.h"
#include "../../../src/compute_provider_ggml_metal/native_program.h"

namespace c=hyle::compute;
namespace a=c::apple;
namespace g=hyle::graph;
namespace mx=mlx::core;
template<class T>T take(hyle::Result<T> x) {
  if (!x) throw std::runtime_error(std::string(x.error().message()));
  return std::move(*x);
}
void check(hyle::Status x) {
  if (!x.ok()) throw std::runtime_error(std::string(x.message()));
}
mx::array ready(mx::array x) {
  mx::async_eval({x});mx::synchronize(mx::default_stream(mx::Device::gpu));x.wait();return x;
}
c::TensorDesc dense(c::DataType type,std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(type,{dims.begin(),dims.size()}));
}
void expect(const mx::array &x,const std::vector<float> &expected) {
  if (x.size()!=expected.size()) throw std::runtime_error("output size");
  for (std::size_t i=0;i<expected.size();++i)
    if (std::bit_cast<std::uint32_t>(x.data<float>()[i])!=std::bit_cast<std::uint32_t>(expected[i]))
      throw std::runtime_error("exact mismatch at "+std::to_string(i));
}

void domain(bool graph,int tokens,int slots,int mode,bool packed=false) {
  // mode0: per-slot; mode1: dense A1; mode2: existing slot-broadcast View.
  grouped_shared_test::Fixture f(tokens,slots);
  const bool shared=mode!=0;
  const auto wd=packed?take(c::gguf_grouped_weight_description(
      c::GgufQuantizedType::kQ8_0,c::DataType::kFloat32,f.experts,f.rows,f.columns)):
      dense(c::DataType::kFloat32,{f.experts,f.rows,f.columns});
  std::vector<std::uint8_t> encoded;
  if (packed) {
    // Original GGUF Q8_0: F16 scale1/4 and 32 exact signed integer codes.
    for (std::size_t block=0;block<f.weights.size()/32;++block) {
      encoded.push_back(0);encoded.push_back(0x34);
      for (std::size_t k=0;k<32;++k)
        encoded.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(f.weights[block*32+k]*4)));
    }
  }
  auto physical=dense(c::DataType::kFloat32,{f.tokens,shared?1:f.slots,f.columns});
  const bool singleton=mode==2 && tokens==1 && slots==1;
  if (singleton) physical=dense(c::DataType::kFloat32,{f.columns});
  const auto idd=dense(c::DataType::kInt32,{f.tokens,f.slots});
  auto logical=physical;
  if (mode==2) {
    const std::array shape{f.tokens,f.slots,f.columns};
    const std::array<std::int64_t,3> strides{tokens==1?0:f.columns*4,0,4};
    logical=take(c::TensorDesc::strided_bytes(c::DataType::kFloat32,shape,strides));
  }
  if (!graph && tokens==32 && mode==0 && !packed) {
    const std::array descriptions{wd,logical,idd};
    auto native=take(c::ggml_metal::NativeProgram::create(c::model_ops::grouped_matmul(),descriptions));
    const auto logical_bytes=std::size_t(f.tokens*f.slots*f.rows)*sizeof(float);
    if (native->output_bytes()<=logical_bytes)
      throw std::runtime_error("MMID backend metadata span was not reserved");
    std::printf("MMID output logical=%zu physical=%zu\n",logical_bytes,native->output_bytes());
  }
  auto context=take(a::Context::create());
  auto program=[&]() {
    if (!graph) {
      const std::array specs{take(a::input_spec(wd)),take(a::input_spec(logical)),take(a::input_spec(idd))};
      return take(context.prepare(c::model_ops::grouped_matmul(),specs));
    }
    auto b=take(g::GraphBuilder::create());
    const auto w=take(b.input(wd));auto x=take(b.input(physical));const auto ids=take(b.input(idd));
    if (mode==2) {
      const std::array shape{f.tokens,f.slots,f.columns};
      x=take(b.add(take(c::View::broadcast_to(take(c::TensorShape::create(shape)))),x));
    }
    const auto y=take(b.add(c::model_ops::grouped_matmul(),w,x,ids));
    check(b.publish(y));
    return take(g::apple::compile(context,take(std::move(b).finish())));
  }();
  auto retained=take(program.create_invocation());
  auto active=take(program.create_invocation());
  std::optional<c::ResultLease<a::Results>> old;
  std::vector<float> old_values;
  for (int pass=0;pass<3;++pass) {
    const auto input=f.activations(shared,pass);
    const auto ids=f.ids(pass==1,pass);
    const mx::Shape shape=singleton?mx::Shape{int(f.columns)}:
        mx::Shape{tokens,shared?1:slots,int(f.columns)};
    auto base=ready(mx::array(input.data(),shape,mx::float32));
    auto x=base;
    if (!graph && mode==2) {
      x=ready(mx::broadcast_to(base,{tokens,slots,int(f.columns)}));
      if (x.strides()[1]!=0 || x.data<float>()!=base.data<float>())
        throw std::runtime_error("broadcast must remain a storage view");
    }
    const auto weight=packed?ready(mx::array(encoded.data(),take(a::input_spec(wd)).shape,mx::uint8)):
        ready(mx::array(f.weights.data(),{int(f.experts),int(f.rows),int(f.columns)},mx::float32));
    const std::array args{weight,
      x,ready(mx::array(ids.data(),{tokens,slots},mx::int32))};
    auto &slot=pass==0?retained:active;
    check(slot.bind(args));auto work=take(slot.submit());auto result=take(work.wait());
    const auto expected=f.expected(shared,input,ids);
    expect(result.get()->arrays()[0],expected);
    if (old) expect(old->get()->arrays()[0],old_values);
    if (pass==0) {old.emplace(std::move(result));old_values=expected;}
  }
  if (!graph && mode==2 && !singleton) {
    const auto wrong=f.activations(false);
    const auto indices=f.ids(false);
    const std::array args{
      ready(mx::array(f.weights.data(),{int(f.experts),int(f.rows),int(f.columns)},mx::float32)),
      ready(mx::array(wrong.data(),{tokens,slots,int(f.columns)},mx::float32)),
      ready(mx::array(indices.data(),{tokens,slots},mx::int32))};
    auto bad=take(program.create_invocation());
    auto bound=bad.bind(args);
    if (bound.ok()) {
      auto work=bad.submit();
      if (work) {
        auto result=work->wait();
        if (result || !result.error().IsInvalidArgument()) throw std::runtime_error("wrong actual stride accepted");
      } else if (!work.error().IsInvalidArgument()) throw std::runtime_error("wrong stride failure type");
    } else if (!bound.IsInvalidArgument()) throw std::runtime_error("wrong stride bind failure type");
    expect(old->get()->arrays()[0],old_values);
  }
  check(context.close());expect(old->get()->arrays()[0],old_values);
  std::printf("PASS %s T%d S%d mode%d %s: repeated/unique, rebind, retained output\n",graph?"Graph":"Direct",tokens,slots,mode,packed?"Q8_0":"F32");
}
int main() {
  try {
    for (bool graph:{false,true}) {
      for (int tokens:{31,32}) for (int mode:{0,1,2}) domain(graph,tokens,2,mode);
      domain(graph,1,1,2);
      domain(graph,32,2,1,true);
    }
    std::puts("PASS 16 public grouped shared-activation domains");return 0;
  } catch(const std::exception &e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
