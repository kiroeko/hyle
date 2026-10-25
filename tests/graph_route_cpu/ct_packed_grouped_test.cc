#include "../compute_models/ct_packed_grouped_fixture.h"
#include "../compute_models/kimi_ct_int4_source_roles.h"
#include "../compute_models/nvfp4_reference_fixture.h"
#include "../compute_models/original_role_input.h"
#include "hyle/graph/cpu.h"

namespace {
namespace f = ct_packed_grouped_test;
void exercise(unsigned journey) {
  auto context = f::take(f::g::cpu::create_context(), "CPU Context");
  auto queue = f::take(context.create_queue(), "CPU Queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  const auto compile = [&](const f::g::Graph &graph, f::g::CompileOptions options,
                           bool) {
    return f::g::cpu::compile(graph, context, options);
  };
  const auto write = [](f::m::BufferView destination, std::span<const std::byte> bytes) {
    return destination.write(bytes);
  };
  const auto read = [](f::m::ConstBufferView source, std::span<std::byte> bytes) {
    return source.read(bytes);
  };
  const auto prepare = [](f::g::Invocation &slot,
                          std::span<const f::g::QueueBinding> bindings, bool) {
    return slot.prepare_replay(bindings,
        {.mode = f::g::ReplayMode::kCommands, .host_wait = f::g::HostWaitPolicy::kAllow});
  };
  if (journey == 0) f::run(f::m::system_resource(), queues, compile, write, read, prepare);
  else if (journey == 1) f::run_domain(f::m::system_resource(), queues, compile, write, read, prepare);
  else f::run_escape(f::m::system_resource(), queues, compile, write, read, prepare);
}
TEST(CpuCtPackedGrouped, OriginalBanksKeepRoundingStridesSlotsAndResults) { exercise(0); }
TEST(CpuCtPackedGrouped, CompleteIdsValidateBeforeSelectedBankAccess) { exercise(1); }
TEST(CpuCtPackedGrouped, PublishedDecodeRetainsTheOriginalGraph) { exercise(2); }


TEST(CpuCtPackedGrouped, KimiInt4IndependentSourceRolesKeepSelectedBanksAndBfloatStore) {
  namespace r=kimi_ct_int4_source_roles;
  namespace s=model_plane_resident_weights;
  namespace nv=nvfp4_reference_test;
  namespace c=f::c;namespace g=f::g;namespace m=f::m;
  using Type=f::Type;
  auto context=f::take(g::cpu::create_context(),"Kimi CT CPU context");
  auto queue=f::take(context.create_queue(),"Kimi CT CPU queue");
  struct Drain{c::Context &value;~Drain(){value.drain().IgnoreError();}}drain{context};
  struct SystemResource{m::MemoryResourceRef ref()const noexcept{return m::system_resource();}};
  auto owner=f::take(c::StorageOwner::adopt(context,SystemResource{}),"external source CPU owner");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto layout=f::take(s::Layout::create(r::source_descriptions(),r::domain,r::capacity),"original independent Q/SF planes");
  auto supply=f::take(s::CpuSupply::create(owner,layout),"existing two-generation CPU supply");
  struct Selection{std::vector<std::int32_t> globals,map;};
  const auto selection=[](unsigned generation) {
    Selection result;result.globals={std::int32_t(generation*2),std::int32_t(generation*2+1)};
    result.map.assign(std::size_t(r::domain),-1);
    for(std::size_t i=0;i<result.globals.size();++i)result.map[std::size_t(result.globals[i])]=std::int32_t(i);
    return result;
  };
  const auto scale=[](std::int32_t expert,std::int64_t row,std::int64_t group) {
    return std::uint16_t(0x3f81u+unsigned((expert+row+group)%3));
  };
  const auto code=[](std::int32_t expert,std::int64_t row,std::int64_t group) {
    return 3+int((expert+row+group)%3)*2;
  };
  const auto source=[&](const Selection &selected) {
    std::array<r::ExternalExpert,2> result;
    for(std::size_t i=0;i<result.size();++i) {
      const auto global=selected.globals[i];const auto &spec=r::specs[std::size_t(global)];
      result[i].global=global;
      std::vector<std::uint32_t> packed(std::size_t(r::n*r::k/8),0x88888888u);
      std::vector<std::uint16_t> sf(std::size_t(r::n*r::k/r::group));
      for(std::int64_t row=0;row<r::n;++row) {
        for(std::int64_t group=0;group<r::k/r::group;++group)
          sf[std::size_t(row*(r::k/r::group)+group)]=scale(global,row,group);
        for(std::int64_t group=0;group<2;++group) {
          const auto at=std::size_t(row*(r::k/8)+group*r::group/8);
          packed[at]=(packed[at]&~15u)|unsigned(8+code(global,row,group));
        }
      }
      const std::array<std::int64_t,2> logical{r::n,r::k};
      const std::array<std::span<const std::byte>,3> bodies{
          std::as_bytes(std::span(packed)),std::as_bytes(std::span(logical)),std::as_bytes(std::span(sf))};
      for(std::size_t role=0;role<3;++role)result[i].roles[role]={
          r::named::detail::bind(spec.roles[role].name,spec.roles[role].description(),owner,bodies[role],&r::identity),
          spec.roles[role].data_begin,spec.roles[role].data_end};
    }
    return result;
  };
  auto builder=f::take(g::GraphBuilder::create(),"Kimi CT source graph");
  const auto adesc=f::dense(Type::kBFloat16,{2,1,r::k}),idesc=f::dense(Type::kInt32,{2,2});
  const auto activation=f::take(builder.input(adesc),"original BF16 A");
  const auto ids=f::take(builder.input(idesc),"actual global IDs");
  std::array<g::ValueId,3> banks;
  for(std::size_t p=0;p<2;++p)banks[p]=f::take(builder.input(layout.resident[p]),"selected compressed plane");
  banks[2]=f::take(builder.input(layout.map_description()),"same-generation map");
  const auto local=f::take(builder.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},banks[2],ids),"checked actual local IDs");
  const auto reshape=[&](g::ValueId value,std::initializer_list<std::int64_t> dimensions) {
    const auto shape=f::take(c::TensorShape::create({dimensions.begin(),dimensions.size()}),"original source shape");
    return f::take(builder.add(f::take(c::View::reshape(shape),"original source View"),value),"original source reshape");
  };
  const auto q=reshape(banks[0],{r::capacity*r::n,r::k/8});
  const auto sf=reshape(banks[1],{r::capacity*r::n,r::k/r::group});
  const auto decoded=f::take(builder.add(c::model_ops::artifact_weight_decode(
      c::model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4,128,Type::kBFloat16),q,sf),"original BF16 CT decode boundary");
  const auto weights=reshape(decoded,{r::capacity,r::n,r::k});
  const auto widened=f::take(builder.add(c::Cast{Type::kFloat32},activation),"explicit widened BF16 A");
  const auto dot=f::take(builder.add(c::model_ops::grouped_matmul(),weights,widened,local),"original selected BF16 weight operand");
  const auto output=f::take(builder.add(c::Cast{Type::kBFloat16},dot),"original projection BF16 store");
  f::check(builder.publish(output),"publish actual projection store");
  auto graph=f::take(std::move(builder).finish(),"finite selected Kimi source graph");
  auto compiled=f::take(g::cpu::compile(graph,context),"compile existing grouped packed path");
  bool packed=false;
  for(std::size_t step=0;step<compiled.report.step_count();++step) {
    const auto record=f::take(compiled.report.step(step),"actual Kimi source step");
    if(record.policy==ct_packed_graph_test::kPackedPolicy) {
      packed=true;EXPECT_LT(record.workspace.size,std::size_t(r::capacity*r::n*r::k*2));
    }
  }
  ASSERT_TRUE(packed)<<"Existing CT grouped contraction must retain compressed banks until selected access";
  const auto odesc=f::take(graph.value_description(output),"actual BF16 projection output");
  auto pool=f::take(c::LeasePool::create(owner.ref(),256u<<10,{.max_blocks=8}),"finite original input/output pool");
  const auto upload=[&](std::span<const std::byte> bytes) {
    auto writer=f::take(pool.allocate(bytes.size(),64),"dynamic source bytes");
    f::check(writer.view()->write(bytes),"write dynamic source");return f::take(writer.publish(),"ready dynamic source");
  };
  std::array<s::CpuBatch,2> generations;
  std::array<g::Invocation,2> runs;
  std::array<c::ReadLease,2> results;
  std::array<std::vector<std::uint16_t>,2> saved;
  constexpr std::array<std::array<float,2>,2> coefficients{{{1.F,-1.F},{.5F,1.F}}};
  std::array<int,2> weight_store_witnesses{};
  for(unsigned generation=0;generation<2;++generation) {
    const auto selected=selection(generation);auto external=source(selected);
    if(!generation) {
      auto bad=external;bad[0].roles[0].data_begin+=4;
      EXPECT_FALSE(r::Source::accept(bad));
      const std::array<std::int64_t,2> wrong{r::k,r::n};
      bad=external;const auto &spec=r::specs[0].roles[1];
      bad[0].roles[1].ready=r::named::detail::bind(spec.name,spec.description(),owner,std::as_bytes(std::span(wrong)),&r::identity);
      EXPECT_FALSE(r::Source::accept(bad));
      bad=external;bad[0].roles[2].ready.description=f::dense(Type::kFloat16,{r::n,r::k/r::group});
      EXPECT_FALSE(r::Source::accept(bad));
      EXPECT_TRUE(external[0].roles[0].ready.lease.valid());
    }
    auto accepted=f::take(r::Source::accept(external),"actual external shape/Q/SF roles");
    // ReadonlyRole and scoped mappings retain the original external leases.
    external={};
    std::array<std::string,2> expected_hash;
    for(std::size_t p=0;p<2;++p) {
      std::vector<std::byte> bytes;
      for(const auto global:selected.globals) {
        const auto plane=accepted.plane_bytes(p,global);bytes.insert(bytes.end(),plane.begin(),plane.end());
      }
      expected_hash[p]=original_role_input::sha256(bytes);
    }
    generations[generation]=f::take(supply.upload(selected,accepted),"existing same-generation compressed Q/SF/map upload");
    f::check(accepted.close(),"release original source mappings/leases before Graph");
    for(std::size_t p=0;p<2;++p) {
      std::vector<std::byte> bytes(generations[generation].banks[p].size());
      f::check(generations[generation].banks[p].view()->read(bytes),"read copied original plane bytes");
      EXPECT_EQ(original_role_input::sha256(bytes),expected_hash[p]);
    }
    auto &run=runs[generation];run=f::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"original source Invocation");
    for(std::size_t p=0;p<banks.size();++p)f::check(run.bind(banks[p],p<2?layout.resident[p]:layout.map_description(),
        generations[generation].banks[p]),"same-generation original bank/map bind");
    std::vector<std::uint16_t> a(std::size_t(2*r::k),0);
    for(std::size_t query=0;query<coefficients.size();++query)for(std::size_t term=0;term<2;++term)
      a[query*std::size_t(r::k)+term*std::size_t(r::group)]=nv::bf16(coefficients[query][term]);
    const std::array<std::int32_t,4> route{selected.globals[1],selected.globals[0],selected.globals[0],selected.globals[1]};
    f::check(run.bind(activation,adesc,upload(std::as_bytes(std::span(a)))),"actual A lease");
    f::check(run.bind(ids,idesc,upload(std::as_bytes(std::span(route)))),"actual current global IDs");
    if(f::take(compiled.executable.output_ownership(),"Kimi source output ownership")==g::OutputOwnership::kCaller)
      f::check(run.bind(output,odesc,f::take(pool.allocate(f::take(odesc.required_bytes(),"BF16 output bytes"),64),"original result writer")),"bind output writer");
    f::check(run.prepare_replay(queues),"prepare original selected Graph");
    {auto done=f::take(run.submit(queues),"submit original selected Graph");f::check(done.wait(),"actual projection Completion");}
    results[generation]=f::take(f::take(run.result(output),"original projection result").storage_lease(),"retain old projection storage");
    saved[generation].resize(std::size_t(4*r::n));
    f::check(results[generation].view()->read(std::as_writable_bytes(std::span(saved[generation]))),"read actual BF16 store");
    for(std::size_t item=0;item<route.size();++item)for(std::int64_t row=0;row<r::n;++row) {
      const auto query=item/2;float sum=0,without_weight_store=0;
      for(std::int64_t group=0;group<2;++group) {
        const auto raw=nv::f32(double(code(route[item],row,group))*nv::wide(scale(route[item],row,group)));
        const auto weight=nv::wide(nv::bf16(raw));
        const auto a_value=double(coefficients[query][std::size_t(group)]);
        sum=nv::f32(double(sum)+double(nv::f32(double(weight)*a_value)));
        // Forbidden omission of the original BF16 weight-product boundary.
        // Compare FINAL BF16 outputs: one-term dots cannot witness this gate.
        without_weight_store=nv::f32(double(without_weight_store)+double(nv::f32(double(raw)*a_value)));
      }
      const auto expected=nv::bf16(sum),alternative=nv::bf16(without_weight_store);
      weight_store_witnesses[query]+=int(expected!=alternative);
      EXPECT_EQ(saved[generation][item*std::size_t(r::n)+std::size_t(row)],expected);
    }
  }
  EXPECT_GT(weight_store_witnesses[0],0)<<"Cancellation query must distinguish the intermediate BF16 weight store";
  EXPECT_GT(weight_store_witnesses[1],0)<<"Second query must distinguish the intermediate BF16 weight store";
  EXPECT_NE(saved[0],saved[1]);
  EXPECT_EQ(supply.stats().live_blocks,6u);
  {
    const auto selected=selection(0);auto external=source(selected);
    auto accepted=f::take(r::Source::accept(external),"third cold valid source");
    EXPECT_FALSE(supply.upload(selected,accepted))<<"Two original retained generations exhaust the existing finite pool";
    f::check(accepted.close(),"close third original source mappings");
  }
  generations={};EXPECT_EQ(supply.stats().live_blocks,6u); // Invocations still own Q/SF/map.
  runs={};EXPECT_EQ(supply.stats().live_blocks,0u);
  compiled={};
  for(std::size_t generation=0;generation<results.size();++generation) {
    std::vector<std::uint16_t> actual(saved[generation].size());
    f::check(results[generation].view()->read(std::as_writable_bytes(std::span(actual))),"old BF16 output after all bank owners retire");
    EXPECT_EQ(actual,saved[generation]);
  }
  ::testing::Test::RecordProperty("source_repository",std::string(r::repository));
  ::testing::Test::RecordProperty("source_revision",std::string(r::revision));
  ::testing::Test::RecordProperty("source_value_scope","generated fixture only; publisher Q/shape/SF body not cached or verified");
  ::testing::Test::RecordProperty("source_generations",2);
  ::testing::Test::RecordProperty("selected_experts_per_generation",2);
  ::testing::Test::RecordProperty("actual_projection_rows",3072);
  ::testing::Test::RecordProperty("actual_projection_depth",3584);
  ::testing::Test::RecordProperty("bf16_weight_store_final_output_witnesses_query0",weight_store_witnesses[0]);
  ::testing::Test::RecordProperty("bf16_weight_store_final_output_witnesses_query1",weight_store_witnesses[1]);
}

} // namespace
