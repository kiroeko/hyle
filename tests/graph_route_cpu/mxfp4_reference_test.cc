#include "../compute_models/mxfp4_reference_fixture.h"
#include "../compute_models/inferact_mxfp4_source_roles.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU MXFP4 context");
  auto queue=f::take(context.create_queue(),"CPU MXFP4 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuMxfp4ReferenceGraph, OriginalPackedBanksSurviveSharedAndIndependentSlots) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuMxfp4ReferenceGraph, TinyOperandsRuntimeFailuresAndIeeeRecovery) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuMxfp4ReferenceGraph, WeightOnlyKeepsOriginalActivationAndResultLifetimes) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare,{},
        mxfp4_reference_test::ActivationMode::kBFloat16);
  });
}
TEST(CpuMxfp4ReferenceGraph, WeightOnlyTinyValuesCheckedFailuresAndPrescaledWeights) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare,
        mxfp4_reference_test::ActivationMode::kBFloat16);
  });
}
TEST(CpuMxfp4ReferenceDirect, QuantizesSharedRowsOnceAndRejectsWorkspaceAliasing) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU MX direct context");
  auto queue=f::take(context.create_queue(),"CPU MX direct queue");
  for(const auto layout:{f::Layout::kDense,f::Layout::kShared,f::Layout::kPerSlot,f::Layout::kSharedView}) {
    f::Case sample(layout);const auto data=sample.data();const auto payload=data.payloads();
    const auto operation=sample.grouped()?f::c::model_ops::grouped_mxfp4_linear():f::c::model_ops::mxfp4_linear();
    auto prepared=f::take(f::c::prepare(context,operation,std::span(sample.descriptions).first(sample.port_count()),
        std::array{sample.output_description},f::m::system_resource()),"prepare MX direct");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    const auto requirement=prepared.workspace_requirement();
    EXPECT_EQ(requirement.size,std::size_t(2*sample.tokens*sample.depth*(layout==f::Layout::kPerSlot?sample.slots:1)));
    std::array<f::m::Buffer,4> buffers;std::array<f::c::ConstTensorView,4> inputs;
    for(std::size_t i=0;i<sample.port_count();++i) {
      buffers[i]=f::take(f::m::Buffer::create(f::m::system_resource(),std::max(payload[i].size(),requirement.size),64),"original MX direct input");
      f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"upload original input");
      inputs[i]=f::take(f::c::ConstTensorView::create(sample.descriptions[i],buffers[i].view()),"original view");
    }
    const auto size=sample.output_description.required_bytes().value();
    auto destination=f::take(f::m::Buffer::create(f::m::system_resource(),size,64),"direct output");
    const std::vector<std::byte> poison(size,std::byte{0x5a});f::check(destination.view().write(poison),"poison output");
    const std::array outputs{f::take(f::c::TensorView::create(sample.output_description,destination.view()),"output view")};
    for(std::size_t alias=0;alias<sample.port_count();++alias) {
      auto aliased=f::take(prepared.create_invocation(buffers[alias].view()),"aliased workspace");
      auto done=aliased.submit(queue,std::span(inputs).first(sample.port_count()),outputs);ASSERT_TRUE(done)<<done.error().message();
      EXPECT_TRUE(done->wait().IsInvalidArgument());
      for(std::size_t i=0;i<sample.port_count();++i) {
        std::vector<std::byte> actual(payload[i].size());
        f::check(buffers[i].view().subview(0,payload[i].size()).value().read(actual),"unchanged original bytes");EXPECT_EQ(actual,payload[i]);
      }
      std::vector<std::byte> actual(size);f::check(destination.view().read(actual),"unchanged output");EXPECT_EQ(actual,poison);
    }
    auto scratch=f::take(f::m::Buffer::create(f::m::system_resource(),requirement.size,64),"distinct workspace");
    auto run=f::take(prepared.create_invocation(scratch.view()),"direct valid invocation");
    auto done=f::take(run.submit(queue,std::span(inputs).first(sample.port_count()),outputs),"direct valid submit");f::check(done.wait(),"direct completion");
    std::vector<std::uint16_t> q(requirement.size/2);f::check(scratch.view().read(std::as_writable_bytes(std::span(q))),"actual QDQ bytes");
    EXPECT_EQ(q[2],0x8000u)<<"source negative zero survives actual quantization";
  }
}
TEST(CpuMxfp4ReferenceDirect, WeightOnlyNeedsNoActivationWorkspace) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU W4A16 context");
  for(const auto layout:{f::Layout::kDense,f::Layout::kShared,f::Layout::kPerSlot,f::Layout::kSharedView}) {
    const f::Case sample(layout,3,64,f::ActivationMode::kBFloat16);
    const auto operation=sample.grouped()?f::c::model_ops::grouped_mxfp4_linear(sample.mode)
        :f::c::model_ops::mxfp4_linear(sample.mode);
    auto prepared=f::take(f::c::prepare(context,operation,std::span(sample.descriptions).first(sample.port_count()),
        std::array{sample.output_description},f::m::system_resource()),"prepare W4A16");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size,0u);
    EXPECT_TRUE(prepared.create_invocation());
  }
}
TEST(CpuMxfp4ReferenceSchema, RejectsWrongScaleAndIncompleteGroupWithoutGlobals) {
  namespace f=mxfp4_reference_test;f::Case sample;std::array<f::c::TensorDesc,1> output;
  auto input=sample.descriptions;
  EXPECT_TRUE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input[2]=f::dense(f::Type::kFloat8E4M3FN,{sample.columns,sample.depth/32});
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input=sample.descriptions;input[0]=f::dense(f::Type::kBFloat16,{sample.tokens,16});
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(sample.descriptions,output).ok());
  f::Case grouped(f::Layout::kShared);input=grouped.descriptions;input[0]=f::dense(f::Type::kBFloat16,{grouped.tokens,2,grouped.depth});
  EXPECT_FALSE(f::c::model_ops::GroupedMxfp4Linear{}.infer_output_descriptions(input,output).ok());
  const auto bad_mode=static_cast<f::ActivationMode>(99);
  input=sample.descriptions;
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{bad_mode}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input=grouped.descriptions;
  EXPECT_FALSE(f::c::model_ops::GroupedMxfp4Linear{bad_mode}.infer_output_descriptions(input,output).ok());
}


TEST(CpuMxfp4ReferenceGraph, InferactFusedHalvesKeepOriginalScaleImportAndSelectedBankGeneration) {
  namespace r=inferact_mxfp4_source_roles;
  namespace s=model_plane_resident_weights;
  namespace f=mxfp4_reference_test;
  namespace nv=nvfp4_reference_test;
  namespace c=f::c;namespace g=f::g;namespace m=f::m;
  using Type=c::DataType;
  constexpr std::int64_t rows=3;
  auto context=f::take(g::cpu::create_context(),"Inferact source CPU context");
  auto queue=f::take(context.create_queue(),"Inferact source CPU queue");
  struct Drain{c::Context &context;~Drain(){context.drain().IgnoreError();}}drain{context};
  struct SystemResource{m::MemoryResourceRef ref()const noexcept{return m::system_resource();}};
  auto owner=f::take(c::StorageOwner::adopt(context,SystemResource{}),"Inferact source owner");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&queue}};
  const auto planes=f::take(r::source_descriptions(rows),"Inferact original half prefix descriptors");
  const auto layout=f::take(s::Layout::create(planes,r::domain,r::capacity),"Inferact same-generation planes/map");
  EXPECT_EQ(layout.logical_bytes,54272u);
  auto supply=f::take(s::CpuSupply::create(owner,layout),"Inferact existing bounded two-generation supply");
  struct Selection{std::vector<std::int32_t> globals,map;};
  const auto selection=[](unsigned epoch) {
    Selection v;v.globals=epoch?std::vector<std::int32_t>{1,511}:std::vector<std::int32_t>{0,17};
    v.map.assign(std::size_t(r::domain),-1);
    for(std::size_t i=0;i<v.globals.size();++i)v.map[std::size_t(v.globals[i])]=std::int32_t(i);
    return v;
  };
  const auto code=[](std::int32_t id,std::int64_t row,std::size_t half,std::size_t group) {
    return unsigned(1+(id+row+std::int64_t(half))%6)|(group?8u:0u);
  };
  // These generated bodies have the ORIGINAL full4096-row dimensions. The
  // adapter alone selects each2048-row half's first3 rows, at distinct W/SF
  // byte strides. No short6-row fake source is labelled an original role.
  const auto source=[&](const Selection &selected) {
    std::array<r::ExternalExpert,2> result;
    for(std::size_t i=0;i<result.size();++i) {
      const auto id=selected.globals[i];const auto at=std::ranges::find(r::specs,id,&r::ExpertSpec::global);
      if(at==r::specs.end())throw std::runtime_error("uncached fixture source ID");
      result[i].global=id;
      std::vector<std::uint8_t> codes(std::size_t(r::full_rows*r::k/2),0x66);
      std::vector<float> raw_sf(std::size_t(r::full_rows*r::k/32),1.F);
      for(std::size_t half=0;half<2;++half)for(std::int64_t row=0;row<rows;++row)for(std::size_t group=0;group<2;++group) {
        const auto original_row=std::int64_t(half)*r::half_rows+row;
        const auto position=std::size_t(original_row*(r::k/2))+group*16;
        codes[position]=std::uint8_t((codes[position]&0xf0u)|code(id,row,half,group));
        raw_sf[std::size_t(original_row*(r::k/32))+group]=(half==group)?1.25F:1.5F;
      }
      const std::array<std::span<const std::byte>,2> bytes{std::as_bytes(std::span(codes)),std::as_bytes(std::span(raw_sf))};
      for(std::size_t role=0;role<2;++role)result[i].roles[role]={
          r::named::detail::bind(at->roles[role].name,at->roles[role].description(),owner,bytes[role],&r::identity),
          at->roles[role].data_begin,at->roles[role].data_end};
    }
    return result;
  };
  auto builder=f::take(g::GraphBuilder::create(),"Inferact selected source graph");
  const auto adesc=f::dense(Type::kBFloat16,{2,1,r::k}),idesc=f::dense(Type::kInt32,{2,2});
  const auto a=f::take(builder.input(adesc),"unchanged BF16 activation"),ids=f::take(builder.input(idesc),"actual global IDs");
  std::array<g::ValueId,5> banks;
  for(std::size_t p=0;p<4;++p)banks[p]=f::take(builder.input(layout.resident[p]),"original selected plane");
  banks.back()=f::take(builder.input(layout.map_description()),"same-generation remap");
  const auto local=f::take(builder.add(c::Gather{0,c::IndexBoundsPolicy::kChecked},banks.back(),ids),"checked global remap");
  std::array<g::ValueId,2> outputs;
  for(std::size_t half=0;half<2;++half) {
    outputs[half]=f::take(builder.add(c::model_ops::grouped_mxfp4_linear(c::model_ops::Mxfp4ActivationMode::kBFloat16),
        a,banks[half*2],banks[half*2+1],local),"existing BF16 weight-only grouped reader");
    f::check(builder.publish(outputs[half]),"publish actual gate/up projection");
  }
  const auto graph=f::take(std::move(builder).finish(),"Inferact ordinary source graph");
  auto compiled=f::take(g::cpu::compile(graph,context),"compile existing CPU reader");
  const auto odesc=f::dense(Type::kBFloat16,{2,2,rows});
  auto pool=f::take(c::LeasePool::create(owner.ref(),256u<<10,{.max_blocks=16}),"bounded dynamic inputs/results");
  const auto upload=[&](std::span<const std::byte> bytes){auto w=f::take(pool.allocate(bytes.size(),64),"dynamic writer");f::check(w.view()->write(bytes),"dynamic write");return f::take(w.publish(),"dynamic readonly input");};
  constexpr std::array<std::array<float,2>,2> coefficients{{{1.25F,-.75F},{.5F,1.5F}}};
  std::array<s::CpuBatch,2> batches;
  std::array<g::Invocation,2> runs;
  std::array<std::array<c::ReadLease,2>,2> results;
  std::array<std::array<std::vector<std::uint16_t>,2>,2> saved;
  for(unsigned epoch=0;epoch<2;++epoch) {
    const auto selected=selection(epoch);auto external=source(selected);
    if(!epoch) {
      auto bad=external;bad[0].roles[0].data_begin+=4096;
      EXPECT_FALSE(r::Source::accept(bad,rows));
      std::vector<float> invalid(std::size_t(r::full_rows*r::k/32),1.F);invalid[std::size_t(r::half_rows*(r::k/32))]=0.F;
      const auto &spec=r::specs[0].roles[1];bad=external;
      bad[0].roles[1].ready=r::named::detail::bind(spec.name,spec.description(),owner,std::as_bytes(std::span(invalid)),&r::identity);
      EXPECT_FALSE(r::Source::accept(bad,rows))<<"Strict importer must not clamp invalid raw F32 SF";
      EXPECT_TRUE(external[0].roles[1].ready.lease.valid());
    }
    auto ready=f::take(r::Source::accept(external,rows),"actual fused W/F32-SF source composition");external={};
    for(const auto global:selected.globals)for(std::size_t half=0;half<2;++half) {
      const auto sf=ready.plane_bytes(half*2+1,global);
      EXPECT_EQ(sf[0],std::byte(half?128:127));EXPECT_EQ(sf[1],std::byte(half?127:128));
      const auto w=ready.plane_bytes(half*2,global);
      EXPECT_EQ(std::to_integer<unsigned>(w[0])&15u,code(global,0,half,0));
    }
    batches[epoch]=f::take(supply.upload(selected,ready),"same-generation W/SF/map publication");
    f::check(ready.close(),"original sources retire after synchronous staging");
    auto &run=runs[epoch];run=f::take(compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"actual selected Invocation");
    for(std::size_t p=0;p<banks.size();++p)f::check(run.bind(banks[p],p<4?layout.resident[p]:layout.map_description(),batches[epoch].banks[p]),"one-generation selected banks/map");
    std::vector<std::uint16_t> activation(std::size_t(2*r::k),0);
    for(std::size_t query=0;query<2;++query)for(std::size_t term=0;term<2;++term)activation[query*std::size_t(r::k)+term*32]=nv::bf16(coefficients[query][term]);
    const std::array<std::int32_t,4> route{selected.globals[1],selected.globals[0],selected.globals[1],selected.globals[1]};
    f::check(run.bind(a,adesc,upload(std::as_bytes(std::span(activation)))),"actual BF16 A without QDQ");
    f::check(run.bind(ids,idesc,upload(std::as_bytes(std::span(route)))),"actual global route");
    for(std::size_t half=0;half<2;++half)if(f::take(compiled.executable.output_ownership(half),"projection output ownership")==g::OutputOwnership::kCaller)
      f::check(run.bind(outputs[half],odesc,f::take(pool.allocate(24,64),"actual projection result writer")),"bind result");
    f::check(run.prepare_replay(queues),"prepare original weight-only Graph");
    {auto done=f::take(run.submit(queues),"submit actual selected gate/up");f::check(done.wait(),"selected completion");}
    for(std::size_t half=0;half<2;++half) {
      results[epoch][half]=f::take(f::take(run.result(outputs[half]),"actual original projection").storage_lease(),"old result reader");
      saved[epoch][half].resize(12);f::check(results[epoch][half].view()->read(std::as_writable_bytes(std::span(saved[epoch][half]))),"read actual gate/up BF16");
      for(std::size_t item=0;item<route.size();++item)for(std::int64_t row=0;row<rows;++row) {
        double sum=0;
        for(std::size_t term=0;term<2;++term) {
          const float scale=half==term?1.F:2.F; // independent known GRS-RNE(1.25/1.5)
          const float weight=nv::wide(nv::bf16(nv::e2m1[code(route[item],row,half,term)]*scale));
          sum+=double(weight)*double(coefficients[item/2][term]);
        }
        EXPECT_EQ(saved[epoch][half][item*std::size_t(rows)+std::size_t(row)],nv::bf16(float(sum)));
      }
    }
  }
  EXPECT_EQ(supply.stats().live_blocks,10u);
  {auto external=source(selection(0));auto ready=f::take(r::Source::accept(external,rows),"third valid source");EXPECT_FALSE(supply.upload(selection(0),ready));f::check(ready.close(),"close rejected third source");}
  batches={};EXPECT_EQ(supply.stats().live_blocks,10u);runs={};EXPECT_EQ(supply.stats().live_blocks,0u);compiled={};
  for(std::size_t epoch=0;epoch<2;++epoch)for(std::size_t half=0;half<2;++half) {
    std::vector<std::uint16_t> actual(12);f::check(results[epoch][half].view()->read(std::as_writable_bytes(std::span(actual))),"escaped old projection after bank retirement");EXPECT_EQ(actual,saved[epoch][half]);
  }
}
