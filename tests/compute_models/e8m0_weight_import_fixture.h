#pragma once

#include <cstring>
#include "nvfp4_reference_fixture.h"
#include "hyle/compute/weight_import.h"

namespace e8m0_weight_import_test {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace scalar=nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type=c::DataType;

struct Case final {
  static constexpr std::int64_t rows=2, columns=129, depth=256;
  c::TensorDesc weight=take(c::float8_weight_description(depth,columns,
      c::Float8ScaleMode::kFloat32Block128), "canonical weight");
  c::TensorDesc activation=dense(Type::kBFloat16,{rows,depth});
  c::TensorDesc output=dense(Type::kBFloat16,{rows,columns});
  std::vector<std::byte> canonical;
  bool stored_float32 = false;
  g::Graph graph;
  g::ValueId a{},w{},y{};

  explicit Case(bool raw_scales = false) : stored_float32(raw_scales) {
    const auto codes_description=dense(Type::kFloat8E4M3FN,{columns,depth});
    const auto scales_description=dense(stored_float32?Type::kFloat32:Type::kUInt8,{2,2});
    auto codes=take(m::Buffer::create(m::system_resource(),columns*depth,64), "original E4 codes");
    auto scales=take(m::Buffer::create(m::system_resource(),scales_description.required_bytes().value(),64), "original scale grid");
    auto prepared=take(m::Buffer::create(m::system_resource(),weight.required_bytes().value(),64), "cold canonical output");
    std::vector<std::uint8_t> raw(std::size_t(columns*depth),0);
    for (std::int64_t n=0;n<columns;++n) {
      raw[std::size_t(n*depth+1)]=n%2?0xb0:0x38;
      constexpr std::array<std::uint8_t,3> second{0x28,0xb8,0x40};
      raw[std::size_t(n*depth+129)]=second[std::size_t(n%3)];
    }
    constexpr std::array<std::uint8_t,4> grid{126,128,127,129};
    check(codes.view().write(std::as_bytes(std::span(raw))), "stored E4");
    constexpr std::array<float,4> raw_grid{1.0039F,.7F,.3333F,1.997F};
    if (stored_float32)
      check(scales.view().write(std::as_bytes(std::span(raw_grid))), "stored F32 macroblock128 scales");
    else
      check(scales.view().write(std::as_bytes(std::span(grid))), "stored E8M0 macroblock128 scales");
    check(c::import_float8_block128_weight(
        take(c::ConstTensorView::create(codes_description,codes.view()), "code tensor"),
        take(c::ConstTensorView::create(scales_description,scales.view()), "scale tensor"),
        take(c::TensorView::create(weight,prepared.view()), "canonical tensor")), "cold import once");
    canonical.resize(prepared.size());
    check(prepared.view().read(canonical), "imported physical bytes");
    std::vector<std::byte> independent(canonical.size());
    std::memcpy(independent.data(),raw.data(),raw.size());
    for (unsigned i=0;i<4;++i) {
      const float value=stored_float32?raw_grid[i]:std::ldexp(1.F,int(grid[i])-127);
      std::memcpy(independent.data()+raw.size()+i*4,&value,4);
    }
    if (canonical!=independent) throw std::runtime_error("import changed code bytes or macroblock scale order");
    auto builder=take(g::GraphBuilder::create(), "import consumer builder");
    a=take(builder.input(activation), "BF16 A");
    w=take(builder.constant(weight), "original compressed weight");
    y=take(builder.add(c::model_ops::float8_block_linear(stored_float32?
        c::model_ops::Float8ScalePolicy::kDivide:c::model_ops::Float8ScalePolicy::kPowerOfTwo,1.e-4F),a,w), "existing dynamic FP8 linear");
    check(builder.publish(y), "publish imported consumer");
    graph=take(std::move(builder).finish(), "existing consumer graph");
  }
  std::vector<std::uint16_t> input(unsigned phase) const {
    std::vector<std::uint16_t> result(std::size_t(rows*depth),0);
    for (std::int64_t row=0;row<rows;++row) for (unsigned group=0;group<2;++group) {
      const auto base=std::size_t(row*depth+group*128);
      const auto scale=float(1u<<((unsigned(row)+group+phase)%2));
      constexpr std::array<float,4> values{.125F,.5F,1.F,2.F};
      constexpr std::array<float,4> raw_values{.50390625F,.25390625F,1.0078125F,2.015625F};
      result[base]=scalar::bf16(448.F*scale);
      result[base+1]=scalar::bf16((stored_float32?raw_values:values)[(unsigned(row)+group+phase)%4]*scale*(phase%2?-1.F:1.F));
    }
    return result;
  }
  std::vector<std::uint16_t> oracle(std::span<const std::uint16_t> input,
                                  bool apply_stored_qdq=true) const {
    std::vector<std::uint16_t> result(std::size_t(rows*columns));
    const std::array<float,4> grid=stored_float32?std::array{1.0039F,.7F,.3333F,1.997F}:
        std::array{.5F,2.F,1.F,4.F};
    constexpr std::array<float,3> second{.25F,-1.F,2.F};
    for (std::int64_t row=0;row<rows;++row) for (std::int64_t n=0;n<columns;++n) {
      const auto block=std::size_t(n/128)*2;
      const auto quantized_activation=[&](std::int64_t group) {
        const auto base=std::size_t(row*depth+group*128);
        const float value=scalar::wide(input[base+1]);
        if (!stored_float32 || !apply_stored_qdq) return value;
        // Each group's independent maximum is448*2^q, so its dynamic scale
        // is exact. Literal E4 nearest-even conversion deliberately changes A.
        const float scale=scalar::wide(input[base])/448.F;
        const auto code=scalar::encode_scale(std::fabs(value)/scale);
        return std::copysign(scalar::multiply(scalar::e4m3(code),scale),value);
      };
      const auto first=scalar::multiply(quantized_activation(0),n%2?-.5F:1.F);
      const auto last=scalar::multiply(quantized_activation(1),second[std::size_t(n%3)]);
      result[std::size_t(row*columns+n)]=scalar::bf16(scalar::f32(double(scalar::multiply(first,grid[block]))+
          double(scalar::multiply(last,grid[block+1]))));
    }
    return result;
  }
};

template<class Compile,class Write,class Read,class Prepare>
void run(m::MemoryResourceRef resource,std::span<const g::QueueBinding> queues,
         Compile compile,Write write,Read read,Prepare prepare,c::StorageOwner owner={},
         bool stored_float32=false) {
  Case shape(stored_float32);
  if (stored_float32) EXPECT_NE(shape.oracle(shape.input(0)),shape.oracle(shape.input(0),false));
  auto buffer=take(m::Buffer::create(resource,shape.canonical.size(),256), "compressed constant");
  check(write(buffer.view(),shape.canonical), "upload canonical bytes");
  auto ready=owner?take(owner.adopt_buffer(std::move(buffer)), "owned constant")
      :take(c::ReadLease::adopt(std::move(buffer)), "constant lease");
  std::array constants{g::ConstantBinding{shape.w,take(ready.tensor(shape.weight), "canonical binding"),
      g::ConstantPolicy::kMaterialize,owner,owner?ready:c::ReadLease{}}};
  auto compiled=take(compile(shape.graph,g::CompileOptions{.constants=constants}), "compile original linear");
  const auto ownership=take(compiled.executable.output_ownership(), "output ownership");
  std::array runs{take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}), "slot0"),
                  take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}), "slot1")};
  constants={};ready={};compiled={};shape.graph={};
  const auto size=shape.output.required_bytes().value();
  auto pool=take(c::LeasePool::create(resource,2*((size+255)/256*256),{.max_blocks=2}), "independent outputs");
  std::array<m::Buffer,2> inputs;
  for (unsigned slot=0;slot<2;++slot) {
    inputs[slot]=take(m::Buffer::create(resource,shape.activation.required_bytes().value(),256), "activation");
    check(runs[slot].bind(shape.a,take(c::ConstTensorView::create(shape.activation,inputs[slot].view()), "A view")), "bind A");
    if (ownership==g::OutputOwnership::kCaller)
      check(runs[slot].bind(shape.y,shape.output,take(pool.allocate(size), "Y writer")), "bind Y");
    check(prepare(runs[slot],queues), "prepare original replay");
  }
  std::array<g::GraphOutput,2> retained;
  std::array<std::vector<std::uint16_t>,2> expected;
  for (unsigned phase=0;phase<2;++phase) for (unsigned slot=0;slot<2;++slot) {
    if (phase) {
      scalar::expect_result(retained[slot],expected[slot],read);
      EXPECT_FALSE(runs[slot].submit(queues));retained[slot].reset();
    }
    const auto data=shape.input(phase*2+slot);
    check(write(inputs[slot].view(),std::as_bytes(std::span(data))), "current activation");
    auto done=take(runs[slot].submit(queues), "existing linear submit");check(done.wait(), "existing Completion");
    retained[slot]=take(runs[slot].result(shape.y), "imported result");expected[slot]=shape.oracle(data);
    scalar::expect_result(retained[slot],expected[slot],read);
    if (retained[1-slot].valid())scalar::expect_result(retained[1-slot],expected[1-slot],read);
  }
  runs={};for(unsigned slot=0;slot<2;++slot)scalar::expect_result(retained[slot],expected[slot],read);
  ::testing::Test::RecordProperty("cold_weight_imports",1);
  ::testing::Test::RecordProperty("scale_source",stored_float32?"stored_F32":"E8M0_U8");
  ::testing::Test::RecordProperty("successful_submissions",4);
}
} // namespace e8m0_weight_import_test
