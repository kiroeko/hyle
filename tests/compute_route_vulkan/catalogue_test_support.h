#pragma once

#include <algorithm>
#include <bit>
#include <span>
#include <vector>

#include "tensor_test_support.h"
#include "../compute_models/cpu_c5_non_gguf_fixture.h"
#include "../compute_models/cpu_c5_plain_fixture.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
#include "hyle/graph/vulkan.h"
#endif

namespace vulkan_catalogue_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace t = vulkan_tensor_test;

inline std::vector<float> floats(const v::Context &context, const v::Tensor &tensor) {
  std::vector<float> result;
  if (tensor.description().data_type() == c::DataType::kInt32) {
    for (auto raw : t::bits(context,tensor))
      result.push_back(static_cast<float>(std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(raw))));
  } else for (auto value : t::values(context,tensor)) result.push_back(static_cast<float>(value));
  return result;
}

inline void execute(const v::Context &context, sem::Fragment fixture,
                    std::vector<v::Tensor> *capture = nullptr) {
  SCOPED_TRACE("region=" + std::to_string(fixture.region));
  std::vector<v::Tensor> prepared, values(fixture.descriptions.size());
  if (fixture.preparation) {
    execute(context,*fixture.preparation,&prepared);
    ASSERT_EQ(prepared.size(),fixture.prepared_constants.size());
  }
  std::vector<v::Tensor> constants;
  for (std::size_t i=0;i<fixture.constants.size();++i) {
    const auto found = std::ranges::find(fixture.prepared_constants,i);
    v::Tensor tensor;
    if (found != fixture.prepared_constants.end()) tensor=prepared[static_cast<std::size_t>(found-fixture.prepared_constants.begin())];
    else {
      auto uploaded=context.upload(fixture.description(fixture.constants[i]),fixture.constant_bytes[i]);
      ASSERT_TRUE(uploaded) << uploaded.error().message();tensor=std::move(*uploaded);
    }
    ASSERT_EQ(tensor.description(),fixture.description(fixture.constants[i]));
    values[fixture.constants[i].ordinal()]=tensor;constants.push_back(std::move(tensor));
  }
  fixture.constant_bytes.clear();prepared.clear();
  const auto workload = d5_qwen38_official_fixture::workload(*fixture.profile,fixture.phase,fixture.region);
#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
  auto graph=sem::make_graph(fixture);hyle::graph::vulkan::CompileReport report;
  auto compiled=hyle::graph::vulkan::compile(context,graph.graph,constants,{.report=&report,.workload=workload});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->workload(),workload);
  auto slot=compiled->create_invocation();ASSERT_TRUE(slot) << slot.error().message();
#else
  std::vector<v::Program> calls;std::vector<v::Invocation> slots;
  for (const auto &node:fixture.nodes) {
    if (node.operation.kind()==c::OperationKind::kView) continue;
    std::vector<v::Argument> arguments;
    for (auto id:node.inputs)
      if (id.kind()==d5_qwen38_24t_full_router::ValueKind::kConstant) arguments.emplace_back(values[id.ordinal()]);
      else arguments.emplace_back(fixture.description(id));
    auto program=v::prepare(context,node.operation,arguments,workload);
    ASSERT_TRUE(program) << c::operation_name(node.operation) << ": " << program.error().message();
    calls.push_back(std::move(*program));
  }
#endif
  sem::StateFeedback feedback{fixture};
  for (const auto &replay:fixture.replays) {
#if !defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
    // A direct chain's consumers retain its previous intermediate result
    // leases. Reuse immutable Programs and renew these mutable request slots;
    // Graph and the separate single-op tests exercise persistent-slot replay.
    slots.clear();
    for (const auto &program:calls) {
      auto slot=program.create_invocation();ASSERT_TRUE(slot) << slot.error().message();
      slots.push_back(std::move(*slot));
    }
#endif
    for (const auto &node:fixture.nodes) for (auto id:node.outputs) values[id.ordinal()]={};
    for (std::size_t i=0;i<fixture.inputs.size();++i) {
      auto uploaded=context.upload(fixture.description(fixture.inputs[i]),feedback.input(replay,i));
      ASSERT_TRUE(uploaded) << uploaded.error().message();values[fixture.inputs[i].ordinal()]=std::move(*uploaded);
    }
#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
    std::vector<v::Tensor> inputs;
    for (auto id:fixture.inputs) inputs.push_back(values[id.ordinal()]);
    ASSERT_TRUE(slot->bind(inputs).ok());auto work=slot->submit();ASSERT_TRUE(work) << work.error().message();
    auto result=work->wait();ASSERT_TRUE(result) << result.error().message();
    const auto outputs=result->get()->tensors();ASSERT_EQ(outputs.size(),fixture.outputs.size());
    for (std::size_t i=0;i<outputs.size();++i) values[fixture.outputs[i].ordinal()]=outputs[i];
#else
    std::size_t call=0;
    for (const auto &node:fixture.nodes) {
      if (node.operation.kind()==c::OperationKind::kView) {
        auto view=values[node.inputs[0].ordinal()].view(*node.operation.get_if<c::View>());
        ASSERT_TRUE(view) << view.error().message();values[node.outputs[0].ordinal()]=std::move(*view);continue;
      }
      std::vector<v::Tensor> inputs;
      for (auto id:node.inputs) if (id.kind()!=d5_qwen38_24t_full_router::ValueKind::kConstant) inputs.push_back(values[id.ordinal()]);
      ASSERT_TRUE(slots[call].bind(inputs).ok());auto work=slots[call++].submit();ASSERT_TRUE(work) << work.error().message();
      auto result=work->wait();ASSERT_TRUE(result) << result.error().message();const auto outputs=result->get()->tensors();
      ASSERT_EQ(outputs.size(),node.outputs.size());
      for (std::size_t i=0;i<outputs.size();++i) values[node.outputs[i].ordinal()]=outputs[i];
    }
#endif
    for (std::size_t i=0;i<fixture.outputs.size();++i) {
      SCOPED_TRACE(i);const auto actual=floats(context,values[fixture.outputs[i].ordinal()]);feedback.observe(i,actual);
      sem::expect_output(fixture,actual,replay.outputs[i],replay.native_outputs.empty()?std::span<const float>{}:replay.native_outputs[i],
          fixture.exact_output(i),fixture.id_validation(i),fixture.bfloat_chain_output(i),i);
    }
    if (capture) {capture->clear();for(auto id:fixture.outputs) capture->push_back(values[id.ordinal()]);}
  }
  if (fixture.invalid_index_input) {
    const auto port=*fixture.invalid_index_input;auto description=fixture.description(fixture.inputs[port]);
    std::vector<std::byte> invalid(*description.required_bytes());const std::int32_t bad=-1;
    std::memcpy(invalid.data(),&bad,sizeof(bad));auto uploaded=context.upload(description,invalid).value();
    const auto id=fixture.inputs[port];values[id.ordinal()]=uploaded;
    std::vector<std::vector<std::byte>> before;
    for (auto output:fixture.outputs) before.push_back(t::bytes(context,values[output.ordinal()]));
#if defined(HYLE_TEST_GRAPH_ROUTE_VULKAN)
    auto failed=compiled->create_invocation().value();std::vector<v::Tensor> inputs;
    for (auto input:fixture.inputs) inputs.push_back(values[input.ordinal()]);
    ASSERT_TRUE(failed.bind(inputs).ok());auto work=failed.submit().value();auto result=work.wait();
    ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange()) << result.error().message();
#else
    std::size_t call=0;bool checked=false;
    for (const auto &node:fixture.nodes) {
      if (node.operation.kind()==c::OperationKind::kView) continue;
      if (std::ranges::find_if(node.inputs,[&](auto input){return input.ordinal()==id.ordinal();})!=node.inputs.end()) {
        auto failed=calls[call].create_invocation().value();std::vector<v::Tensor> inputs;
        for (auto input:node.inputs) if(input.kind()!=d5_qwen38_24t_full_router::ValueKind::kConstant) inputs.push_back(values[input.ordinal()]);
        ASSERT_TRUE(failed.bind(inputs).ok());auto work=failed.submit().value();auto result=work.wait();
        ASSERT_FALSE(result);EXPECT_TRUE(result.error().IsOutOfRange()) << result.error().message();checked=true;break;
      }
      ++call;
    }
    EXPECT_TRUE(checked);
#endif
    for (std::size_t i=0;i<fixture.outputs.size();++i) EXPECT_EQ(t::bytes(context,values[fixture.outputs[i].ordinal()]),before[i]);
  }
}
inline void run(sem::Fragment fixture) {
  auto context=vulkan_test::context();ASSERT_TRUE(context) << context.error().message();
  execute(*context,std::move(fixture));
}

inline void plain(const cpu_c5_plain_inputs::Case &item) {
  namespace f=cpu_c5_plain_fixture;const auto profile=f::profile(item);const std::string_view kind=item.kind;
  SCOPED_TRACE(item.id);
  SCOPED_TRACE(profile.artifact_repository);
  if (kind=="projection" || kind=="head") {
    const auto rows=kind=="head"?4096:item.shape[0];
    for (std::int64_t first=0;first<item.shape[0];first+=rows) {
      run(f::projection(item,&profile,std::min(rows,item.shape[0]-first),first));ASSERT_FALSE(::testing::Test::HasFailure());
    }
  } else if (kind=="embedding" || kind=="position_lookup" || kind=="integer_lookup") run(f::lookup(item,&profile));
  else if (kind=="norm" || kind=="norm_zero_centered") run(f::norm(item,&profile));
  else if (kind=="conv") run(f::convolution(item,&profile));
  else run(f::vector_parameter(item,&profile));
}
inline void packed(const cpu_c5_non_gguf_inputs::Case &item) {
  namespace f=cpu_c5_non_gguf_fixture;namespace in=cpu_c5_non_gguf_inputs;
  const auto profile=f::profile(item);
  SCOPED_TRACE(item.id);
  SCOPED_TRACE(profile.artifact_repository);
  std::printf("VULKAN_CATALOGUE_PACKED id=%s N=%lld K=%lld originalE=%lld executedE=%lld\n",item.id,
      static_cast<long long>(item.n),static_cast<long long>(item.k),static_cast<long long>(item.e),static_cast<long long>(std::min<std::int64_t>(item.e,8)));
  if (item.format==in::Format::kNf4 || item.format==in::Format::kAwq || item.format==in::Format::kCt) run(f::artifact(item,&profile));
  else if (item.n>32768&&!item.embedding) {
    for (std::int64_t first=0;first<item.n;first+=4096) {
      run(f::mlx(item,&profile,std::min<std::int64_t>(4096,item.n-first),first));ASSERT_FALSE(::testing::Test::HasFailure());
    }
  } else run(f::mlx(item,&profile,item.n));
}
} // namespace vulkan_catalogue_test
