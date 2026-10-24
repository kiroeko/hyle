#pragma once

#include <cstring>
#include <iomanip>
#include "nvfp4_reference_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/graph/batch.h"

namespace paged_e4_kv_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace scalar = nvfp4_reference_test;
using scalar::take;
using scalar::check;
using scalar::dense;
using Type = c::DataType;

inline std::uint8_t encode(float value, float scale) {
  const auto normalized = scalar::f32(double(value) / double(scale));
  return std::uint8_t(scalar::encode_scale(std::abs(normalized)) |
      (std::signbit(normalized) ? 0x80u : 0u));
}

// The oracle enumerates all finite E4 codes. In particular, it neither calls
// the device codec nor treats E4 storage as a floating arithmetic operand.
template<class Compile, class Write, class Read>
void encoder(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
             Compile compile, Write write, Read read, bool fail = false) {
  const auto run = [&](Type type, float scale, std::vector<float> values, bool invalid) {
    const auto input_desc = dense(type, {2, 8});
    std::vector<std::byte> input(values.size() * (type == Type::kFloat32 ? 4 : 2));
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (type == Type::kFloat32) std::memcpy(input.data() + i * 4, &values[i], 4);
      else {
        const auto bits = type == Type::kBFloat16 ? scalar::bf16(values[i])
            : half_test::encode<hyle::simd::float16_t>(values[i]).bits;
        values[i] = type == Type::kBFloat16 ? scalar::wide(bits)
            : float(half_test::decode(hyle::simd::float16_t::from_bits(bits)));
        std::memcpy(input.data() + i * 2, &bits, 2);
      }
    }
    auto builder = take(g::GraphBuilder::create(), "E4 encoder builder");
    const auto x = take(builder.input(input_desc), "E4 source");
    const auto y = take(builder.add(c::Operation::custom(c::model_ops::CacheEncode{
        take(c::CacheStorage::float8(scale), "fixed cache scale")}), x), "E4 encoder");
    check(builder.publish(y), "publish E4 bytes");
    auto graph = take(std::move(builder).finish(), "E4 encoder graph");
    auto compiled = take(compile(graph, {}), "compile E4 encoder");
    auto invocation = take(compiled.executable.create_invocation({.resources = g::ResourceOptions{}}),
        "E4 encoder invocation");
    auto pool = take(c::LeasePool::create(resource, input.size() + 128), "encoder inputs");
    auto output_pool = take(c::LeasePool::create(resource, 128), "encoder output");
    auto writer = take(pool.allocate(input.size(), 64), "source storage");
    check(write(take(writer.view(), "source view"), input), "write source");
    auto source = take(writer.publish(), "publish source");
    check(invocation.bind(x, input_desc, source), "bind E4 source");
    if (take(compiled.executable.output_ownership(), "encoder ownership") == g::OutputOwnership::kCaller) {
      const auto description = take(c::cache_tensor_description(
          take(c::CacheStorage::float8(scale), "output interpretation"), input_desc.dimensions()), "encoded description");
      check(invocation.bind(y, description, take(output_pool.allocate(values.size(), 64), "encoded writer")), "bind encoded output");
    }
    check(invocation.prepare_replay(queues), "prepare E4 encoder");
    auto submitted = invocation.submit(queues);
    if (!submitted) {
      ASSERT_TRUE(invalid) << submitted.error().message();
      EXPECT_TRUE(submitted.error().IsInvalidArgument()) << submitted.error().message();
      EXPECT_FALSE(invocation.result(y));
      return;
    }
    auto done = std::move(*submitted);
    const auto status = done.wait();
    if (invalid) {
      EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
      EXPECT_FALSE(invocation.result(y));
      EXPECT_FALSE(invocation.prepare_replay(queues).ok());
      return;
    }
    check(status, "encode completion");
    auto result = take(invocation.result(y), "encoded result");
    auto bytes = take(result.storage_lease(), "encoded bytes lease");
    std::vector<std::byte> actual(values.size());
    check(read(*bytes.view(), actual), "read encoded bytes");
    for (std::size_t i = 0; i < values.size(); ++i)
      ASSERT_EQ(std::to_integer<unsigned>(actual[i]), unsigned(encode(values[i], scale)))
          << "type=" << int(type) << " scale=" << std::hexfloat << scale << " lane=" << i;
  };
  if (fail) {
    std::vector<float> bad(16, .125F);
    bad[7] = std::numeric_limits<float>::quiet_NaN();
    run(Type::kFloat32, .5F, bad, true);
    // A fresh ordinary invocation must work after the failed producer retires.
    bad[7] = -.125F;
    run(Type::kFloat32, .5F, bad, false);
    return;
  }
  const std::vector<float> values{0.F, -0.F, .53125F, -.53125F, .59375F, -.59375F,
      std::nextafter(.53125F, 0.F), std::nextafter(.53125F, 1.F),
      1000.F, -1000.F, .00048828125F, -.00048828125F, 1.F, -1.F,
      std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
  for (auto type : {Type::kFloat32, Type::kFloat16, Type::kBFloat16})
    run(type, .5F, values, false);
  for (float scale : {std::numeric_limits<float>::denorm_min(), std::ldexp(1.F, -140),
                     std::numeric_limits<float>::max()}) {
    auto edge = values;
    edge[2] = scale;
    edge[3] = -scale;
    edge[4] = scalar::f32(double(scale) * .5);
    edge[5] = -edge[4];
    run(Type::kFloat32, scale, edge, false);
  }
}

struct Case final {
  static constexpr unsigned pages = 5, page_rows = 16, heads = 4, kv_heads = 2, dk = 64, dv = 128;
  static constexpr float key_scale = .5F, value_scale = 2.F, attention_scale = .125F;
  bool single;
  c::BatchSchema schema;
  std::array<c::BatchLayoutVersion, 2> layouts;
  std::array<c::TensorDesc, 3> descriptions;
  c::TensorDesc output_description;
  std::array<g::ValueId, 3> inputs{};
  g::NodeId node{};
  g::ValueId output{};
  g::Graph graph;
  std::array<std::vector<std::uint8_t>, 2> banks;
  std::vector<std::uint16_t> query;

  explicit Case(bool q1) : single(q1) {
    const c::BatchCapacity capacity{.query_type=Type::kBFloat16, .output_type=Type::kBFloat16,
        .key_storage=take(c::CacheStorage::float8(key_scale), "K scale"),
        .value_storage=take(c::CacheStorage::float8(value_scale), "V scale"),
        .max_requests=2, .max_query_rows=4, .max_kv_rows=64, .max_pages=pages, .max_page_refs=4,
        .query_heads=heads, .kv_heads=kv_heads, .qk_head_dim=dk, .value_head_dim=dv};
    schema = take(c::BatchSchema::create(capacity), "E4 paged schema");
    const std::array<std::int32_t, 3> qp{0,1,4}, kp{0,2,4};
    const std::array<std::int32_t, 4> ids{2,0,1,0};
    const std::array<std::int32_t, 2> first{1,7}, second{1,15}, q1_second{15,15};
    const auto count = single ? 2u : 3u, ref_count = single ? 2u : 4u;
    layouts[0] = take(schema.version({std::span(qp).first(count), std::span(kp).first(count),
        std::span(ids).first(ref_count), std::span(first).first(count-1)}), "initial page layout");
    layouts[1] = take(schema.version({std::span(qp).first(count), std::span(kp).first(count),
        std::span(ids).first(ref_count), std::span(single?q1_second:second).first(count-1)}), "grown page layout");
    descriptions = {dense(Type::kBFloat16, {single?1:4,heads,dk}),
        take(c::batch_cache_description(schema,c::BatchCacheRole::kKey), "physical K"),
        take(c::batch_cache_description(schema,c::BatchCacheRole::kValue), "physical V")};
    output_description = dense(Type::kBFloat16, {single?1:4,heads,dv});
    auto builder = take(g::GraphBuilder::create(), "paged E4 builder");
    for (unsigned p = 0; p < 3; ++p) inputs[p] = take(builder.input(descriptions[p]), "paged port");
    const auto operation = take(g::batch::paged_attention(layouts[0],
        {.scale=attention_scale, .causal=true, .window_left=16}), "paged E4 policy");
    node = take(builder.add_node(c::Operation::custom(operation), inputs), "paged E4 node");
    output = take(builder.result(node), "paged output");
    check(builder.publish(output), "publish attention");
    graph = take(std::move(builder).finish(), "paged E4 graph");
    for (unsigned role = 0; role < 2; ++role) {
      const unsigned depth = role ? dv : dk;
      auto &bank = banks[role];
      bank.resize(pages*page_rows*kv_heads*depth, 0x7f);
      for (unsigned page = 0; page < 3; ++page)
        for (unsigned row = 0; row < (page ? 16u : 15u); ++row)
          for (unsigned h = 0; h < kv_heads; ++h) for (unsigned d = 0; d < depth; ++d) {
            const auto base = float(int((page*3+row+h)%7)-3)*.5F;
            const auto value = base + float(int(d%5)-2)*.0625F;
            bank[((page*page_rows+row)*kv_heads+h)*depth+d] = encode(value, role?value_scale:key_scale);
          }
    }
    query.resize((single?1u:4u)*heads*dk);
    for (unsigned t = 0; t < (single?1u:4u); ++t) for (unsigned h = 0; h < heads; ++h)
      for (unsigned d = 0; d < dk; ++d)
        query[(t*heads+h)*dk+d] = scalar::bf16(.125F*float((t+h)%3+1));
  }

  std::vector<float> oracle(unsigned epoch, bool ignore_key_scale=false, bool ignore_value_scale=false) const {
    const std::array<unsigned, 3> qp{0,1,4};
    const std::array<unsigned, 4> ids{2,0,1,0};
    const unsigned rows = single?1:4;
    std::vector<float> result(rows*heads*dv);
    for (unsigned request=0; request<(single?1u:2u); ++request) {
      const unsigned tail = single?(epoch?15:1):(request?(epoch?15:7):1);
      const unsigned length = 16+tail, nq = qp[request+1]-qp[request];
      for (unsigned t=qp[request]; t<qp[request+1]; ++t) for (unsigned h=0; h<heads; ++h) {
        const unsigned position=length-nq+t-qp[request], begin=position>16?position-16:0;
        std::vector<double> logits(position-begin+1); double maximum=-std::numeric_limits<double>::infinity();
        for (unsigned row=begin; row<=position; ++row) {
          const unsigned physical=ids[request*2+row/16]*16+row%16, kh=h/(heads/kv_heads);
          double dot=0;
          for (unsigned d=0; d<dk; ++d) dot+=double(scalar::wide(query[(t*heads+h)*dk+d]))*
              double(scalar::e4m3(banks[0][(physical*kv_heads+kh)*dk+d]))*(ignore_key_scale?1.:key_scale);
          logits[row-begin]=dot*attention_scale; maximum=std::max(maximum,logits[row-begin]);
        }
        double denominator=0; for (auto &logit:logits) {logit=std::exp(logit-maximum);denominator+=logit;}
        for (unsigned d=0;d<dv;++d) {
          double sum=0;
          for (unsigned row=begin;row<=position;++row) {
            const unsigned physical=ids[request*2+row/16]*16+row%16, kh=h/(heads/kv_heads);
            sum+=logits[row-begin]*double(scalar::e4m3(banks[1][(physical*kv_heads+kh)*dv+d]))*
                (ignore_value_scale?1.:value_scale);
          }
          result[(t*heads+h)*dv+d]=scalar::rounded_bf16(scalar::f32(sum/denominator));
        }
      }
    }
    return result;
  }
};

template<class Compile, class Write, class Read>
void paged(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
           Compile compile, Write write, Read read, bool single=false,
           m::MemoryResourceRef host_workspace=m::system_resource()) {
  Case shape(single);
  auto keys=take(c::PagedArena::create(shape.schema,c::BatchCacheRole::kKey,resource,
      {.max_backing_bytes=1u<<20}), "key arena");
  auto values=take(c::PagedArena::create(shape.schema,c::BatchCacheRole::kValue,resource,
      {.max_backing_bytes=1u<<20}), "value arena");
  std::array<std::array<c::ReadLease,Case::pages>,2> ready;
  for (unsigned role=0;role<2;++role) for (unsigned p=0;p<Case::pages;++p) {
    auto &arena=role?values:keys;auto writer=take(arena.acquire(p), "physical page");
    const auto page_bytes=arena.page_bytes();
    check(write(take(writer.lease().view(), "page view"), std::as_bytes(std::span(shape.banks[role]).subspan(
        p*page_bytes,page_bytes))), "upload only original E4 bytes");
    ready[role][p]=take(writer.publish(p?16:15), "initialized page prefix");
  }
  auto compiled=take(compile(shape.graph,{}), "compile ordinary E4 paged graph");
  const auto ownership=take(compiled.executable.output_ownership(), "attention ownership");
  const auto output_bytes=shape.output_description.required_bytes().value();
  auto output_pool=take(c::LeasePool::create(resource,output_bytes*4+256,{.max_blocks=4}), "attention output pool");
  std::array<g::Invocation,3> slots;
  for (auto &slot:slots) slot=take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}), "independent paged slot");
  auto pool=take(c::LeasePool::create(resource,shape.query.size()*2+128), "query pool");
  auto writer=take(pool.allocate(shape.query.size()*2,64), "query bytes");
  check(write(take(writer.view(), "query view"), std::as_bytes(std::span(shape.query))), "upload query");
  auto query=take(writer.publish(), "ready query");
  std::array<c::ReadLease,3> held;
  std::array<g::GraphOutput,3> observers;
  std::array<std::vector<std::uint16_t>,3> saved;
  const auto observe=[&](const c::ReadLease &lease) {
    std::vector<std::uint16_t> words(shape.output_description.element_count().value());
    check(read(*lease.view(),std::as_writable_bytes(std::span(words))), "read attention");
    return words;
  };
  for (unsigned epoch=0;epoch<2;++epoch) {
    const auto expected=shape.oracle(epoch), wrong_k=shape.oracle(epoch,true), wrong_v=shape.oracle(epoch,false,true);
    float key_witness=0,value_witness=0;
    for (std::size_t i=0;i<expected.size();++i) {key_witness=std::max(key_witness,std::abs(expected[i]-wrong_k[i]));value_witness=std::max(value_witness,std::abs(expected[i]-wrong_v[i]));}
    ASSERT_GT(key_witness,.02F);ASSERT_GT(value_witness,.02F);
    for (unsigned index=0;index<(single && epoch?3u:2u);++index) {
      auto &slot=slots[index];observers[index].reset();held[index]={};
      const auto layout=shape.layouts[epoch];const std::array versions{g::batch::NodeLayout{shape.node,layout}};
      const g::batch::PrepareOptions preparation{.host_workspace=host_workspace};
      check(g::batch::prepare(slot,queues,versions,preparation), "prepare current page layout");
      if (single && epoch && index<2) {
        auto common=g::batch::report(slot,shape.node);
        const bool reused=common?common->reused_native_plan:take(g::batch::native_report(slot,shape.node), "native reuse report").reused_native_plan;
        EXPECT_TRUE(reused) << "Q1 growth must update the original native reader";
      }
      const std::array krefs{ready[0][0],ready[0][2],ready[0][1]}, vrefs{ready[1][0],ready[1][2],ready[1][1]};
      auto kframe=take(keys.read(layout,std::span(krefs).first(single?2:3)), "exact K pages");
      auto vframe=take(values.read(layout,std::span(vrefs).first(single?2:3)), "exact V pages");
      check(slot.bind(shape.inputs[0],shape.descriptions[0],query), "bind Q");
      check(g::batch::bind(slot,shape.inputs[1],kframe), "bind K permissions");
      check(g::batch::bind(slot,shape.inputs[2],vframe), "bind V permissions");
      if (ownership==g::OutputOwnership::kCaller)
        check(slot.bind(shape.output,shape.output_description,take(output_pool.allocate(output_bytes,64), "attention writer")), "bind attention output");
      check(slot.prepare_replay(queues), "prepare original replay");
      auto done=take(slot.submit(queues), "submit paged attention");check(done.wait(), "paged completion");
      observers[index]=take(slot.result(shape.output), "published attention");
      held[index]=take(observers[index].storage_lease(), "retained result");
      saved[index]=observe(held[index]);
      for (std::size_t i=0;i<expected.size();++i) {
        const auto actual=scalar::wide(saved[index][i]);
        ASSERT_TRUE(std::isfinite(actual));
        ASSERT_NEAR(actual,expected[i],.008F) << "epoch=" << epoch << " slot=" << index << " lane=" << i;
      }
      EXPECT_FALSE(g::batch::prepare(slot,queues,versions,preparation).ok()) << "old result must prevent overwrite";
      if (index) EXPECT_EQ(observe(held[0]),saved[0]) << "second slot changed the first result";
      if (index==2) EXPECT_EQ(saved[2],saved[0]) << "warm Q1 layout disagrees with a fresh invocation";
    }
  }
  slots={};compiled={};shape.graph={};
  for (unsigned index=0;index<3;++index) if (held[index].valid())
    EXPECT_EQ(observe(held[index]),saved[index]) << "result lost original owner";
}
} // namespace paged_e4_kv_test
