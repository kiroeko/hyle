// Isolated current ordinary BF16 projection with one fixed original A row.
// This is not an instrumented default model graph or a task-quality test.
#include "../../engine/models/qwen35/main.cc"
#include "../../engine/models/qwen35/probe_geometry.h"
#include "qwen_geometry_fixture.h"
#include <cstring>
namespace q=qwen35;
namespace {
std::vector<std::byte> exact_file(const std::filesystem::path &path,std::size_t bytes,std::uint64_t offset=0) {
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("cannot read retained original range");
  input.seekg(static_cast<std::streamoff>(offset));std::vector<std::byte> result(bytes);
  if(!input.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(bytes)))throw std::runtime_error("retained original range is truncated");return result;
}
void save(const std::filesystem::path &path,std::span<const std::byte> bytes) {
  std::ofstream output(path,std::ios::binary);output.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!output)throw std::runtime_error("cannot save bounded projection result");
}
int check_packed_head(const std::filesystem::path &component,const std::filesystem::path &output) {
  std::filesystem::create_directories(output);
  const auto codes=exact_file(component/"codes.bin",7680),scales=exact_file(component/"scales.bin",960),globals=exact_file(component/"globals.bin",8);
  auto backend=q::take(q::Backend::create());const auto geometry=qwen_test::q27(false);
  std::map<std::string,q::LoadedWeight> bank;
  const auto add=[&](const std::string &name,q::Type type,std::initializer_list<std::int64_t> dims,std::span<const std::byte> data){
    auto &value=bank.try_emplace(name,q::LoadedWeight{q::Tensor(backend.device(),q::dense(type,dims))}).first->second;value.tensor.write<std::byte>(data);
  };
  add("lm_head.weight",q::Type::kUInt8,{3,2560},codes);bank.at("lm_head.weight").codec=q::prepared::ProjectionCodec::kNvfp4;
  add("lm_head.weight_scale",q::Type::kFloat8E4M3FN,{3,320},scales);
  add("lm_head.input_scale",q::Type::kFloat32,{},std::span(globals).first(4));
  add("lm_head.weight_scale_2",q::Type::kFloat32,{},std::span(globals).subspan(4));
  // Resolve the malformed source independently before any valid binding or
  // Recipe borrows pointers into this bank.
  auto missing=std::move(bank.at("lm_head.input_scale"));bank.erase("lm_head.input_scale");bool rejected=false;
  try{q::ProjectionBindings invalid;invalid.emplace("lm_head.weight",q::resolve_projection(bank,"lm_head.weight"));}catch(const std::exception&){rejected=true;}
  bank.emplace("lm_head.input_scale",std::move(missing));if(!rejected)throw std::runtime_error("packed head resolver silently ignored its missing original input scale");
  q::ProjectionBindings projections;projections.emplace("lm_head.weight",q::resolve_projection(bank,"lm_head.weight"));
  q::GraphAssembly assembly(backend);
  q::Recipe recipe(bank,projections,assembly,geometry,2);const auto x=recipe.input(q::dense(q::Type::kBFloat16,{2,5120}));const auto y=recipe.scores(x,"lm_head.weight");recipe.publish(y);
  const auto graph=q::take(std::move(recipe.builder).finish());std::size_t packed=0,dense=0;
  for(std::size_t index=0;index<graph.node_count();++index){const auto op=q::take(graph.operation(q::take(graph.node_id(index))));const auto name=q::c::operation_name(op);packed+=name=="nvfp4-reference-linear";dense+=name=="dense-linear-f32";}
  if(packed!=1||dense)throw std::runtime_error("packed head did not preserve its explicit NVFP4 projection");
  // Recipe registers its packed-row dimension even when these head ports are
  // fixed. Bind that actual dimension through the normal exact child API.
  const std::array shape{q::g::ShapeBinding{recipe.last_row,1}};
  const std::array variants{q::g::ExactShapeVariant{shape}};
  auto compiled=q::take(backend.compile(graph,assembly.constants,assembly.options({.shape_policy=q::g::CompileShapePolicy::kFiniteExact,.exact_variants=variants,.constants=assembly.constants})));
  compiled.executable=q::take(compiled.executable.exact_variant(0));compiled.report=q::take(compiled.report.exact_variant(0));
  for(std::size_t i=0;i<compiled.report.physical_value_count();++i){
    const auto value=q::take(compiled.report.physical_value(i));const auto &d=value.description;
    if((d.data_type()==q::Type::kBFloat16||d.data_type()==q::Type::kFloat32)&&d.rank()==2&&d.dimensions()[0]==3&&d.dimensions()[1]==5120)
      throw std::runtime_error("packed head materialized a whole floating weight bank");
  }
  const auto output_facts=q::take(hyle::inference::prepare_output(compiled,y));
  const auto yd=output_facts.description;if(yd.data_type()!=q::Type::kFloat32||yd.dimensions()[0]!=2||yd.dimensions()[1]!=3)throw std::runtime_error("packed head logits contract changed");
  std::vector<std::uint16_t> av(2*5120);av[0]=0x3fa0;av[5120+33]=0x40d0; // exact BF161.25,6.5, one nonzero per token
  q::Tensor a(backend.device(),q::dense(q::Type::kBFloat16,{2,5120}));a.write<std::uint16_t>(av);
  auto run=q::take(compiled.executable.create_invocation());q::check(run.prepare_resources());q::check(run.bind(x,a.read()));q::Tensor caller;
  if(output_facts.caller_owned()){caller=q::Tensor(q::take(backend.resource_for(output_facts.storage)),yd);q::check(run.bind(y,caller.view()));}
  auto queues=q::take(hyle::inference::prepare_queue_bindings(backend.queue_bindings(),compiled.report));const auto bindings=hyle::inference::queue_span(queues);q::check(run.prepare_replay(bindings,{.mode=q::g::ReplayMode::kCommands,.host_wait=q::g::HostWaitPolicy::kAllow}));auto done=q::take(run.submit(bindings));q::check(done.wait());
  std::vector<std::byte> result(q::take(yd.required_bytes()));if(caller.buffer)q::check(backend.read(caller.buffer.view(),result));else{auto lease=q::take(hyle::inference::retain_output(run,output_facts));q::check(backend.read(q::take(lease.view()),result));}
  save(output/"logits.f32",result);save(output/"activation.bf16",std::as_bytes(std::span(av)));
  std::ofstream report(output/"observation.json");report<<"{\"score_policy\":"<<std::quoted(std::string(q::score_policy(projections.at("lm_head.weight"), q::PrecisionProfile{})))<<",\"score_dtype\":\"F32\",\"underlying_projection_dtype\":\"BF16\",\"packed_projection_nodes\":"<<packed<<",\"dense_f32_projection_nodes\":"<<dense<<",\"logical_head_shape\":[3,5120],\"original_bytes\":8648,\"checked_output_values\":6,\"whole_floating_head_bank\":false,\"missing_input_scale_rejected\":true,\"declared_graph_workspace_bytes\":"<<compiled.report.workspace_bytes()<<"}\n";
  std::cout<<"{\"packed_head_source_bytes\":8648,\"score_policy\":"<<std::quoted(std::string(q::score_policy(projections.at("lm_head.weight"), q::PrecisionProfile{})))<<",\"graph_calls\":1,\"missing_input_scale_rejected\":true}\n";return 0;
}
// Generated nonzero component fixture. This checks the actual cold binding and
// Recipe entry, not catalog18 publisher values or a complete MoE model.
int check_ct_block128() {
  auto backend=q::take(q::Backend::create()); const auto geometry=qwen_test::q35(false);
  std::map<std::string,q::LoadedWeight> bank;
  const auto add=[&](const std::string &name,q::Type type,std::initializer_list<std::int64_t> dims,
      const auto &data,q::prepared::ProjectionCodec codec=q::prepared::ProjectionCodec::kUnspecified,std::uint32_t group=0) {
    auto &weight=bank.try_emplace(name,q::LoadedWeight{q::Tensor(backend.device(),q::dense(type,dims)),codec,group}).first->second;
    using Element=typename std::remove_cvref_t<decltype(data)>::value_type;
    weight.tensor.write<Element>(std::span(data));
  };
  const std::string name="fixture.block.weight",base="fixture.block";
  const std::vector<std::uint8_t> codes(129*256,0x38);
  const std::array<std::uint16_t,4> sf{0x3f80,0x3f00,0x4000,0x3f80};
  add(name,q::Type::kFloat8E4M3FN,{129,256},codes,q::prepared::ProjectionCodec::kCtFloat8Block128,128);
  add(base+".weight_scale",q::Type::kBFloat16,{2,2},sf);
  q::ProjectionBindings projections;projections.emplace(name,q::resolve_projection(bank,name));
  if(!std::holds_alternative<q::CtBlock128Projection>(projections.at(name)))throw std::runtime_error("CT block128 source codec lost its reader equation");
  q::GraphAssembly assembly(backend);q::Recipe recipe(bank,projections,assembly,geometry,3);
  const auto x=recipe.input(q::Recipe::dense_spec(q::Type::kBFloat16,{recipe.token_rows,q::f(256)}));
  const auto y=recipe.linear_absolute(x,name);recipe.publish(y);const auto graph=q::take(std::move(recipe.builder).finish());
  if(graph.node_count()!=21)throw std::runtime_error("CT block128 Recipe changed the shared QDQ node sequence");
  auto source_read=bank.at(name).tensor.ready;
  const std::array shapes{std::array{q::g::ShapeBinding{recipe.last_row,0}},std::array{q::g::ShapeBinding{recipe.last_row,2}}};
  const std::array variants{q::g::ExactShapeVariant{shapes[0]},q::g::ExactShapeVariant{shapes[1]}};
  auto compiled=q::take(backend.compile(graph,assembly.constants,assembly.options({.shape_policy=q::g::CompileShapePolicy::kFiniteExact,.exact_variants=variants,.constants=assembly.constants})));
  q::c::ReadLease previous;std::vector<std::byte> previous_bytes;
  const auto bf16=[](float v){auto bits=std::bit_cast<std::uint32_t>(v);bits+=0x7fffU+((bits>>16)&1U);return static_cast<std::uint16_t>(bits>>16);};
  for(std::size_t index=0;index<2;++index) {
    const auto rows=index?3:1;const auto child=q::take(compiled.executable.exact_variant(index));const auto report=q::take(compiled.report.exact_variant(index));
    const auto facts=q::take(hyle::inference::prepare_output(q::g::CompiledGraph{child,report},y));
    q::Tensor input(backend.device(),q::dense(q::Type::kBFloat16,{rows,256}));
    std::vector<std::uint16_t> a(static_cast<std::size_t>(rows)*256);
    for(int row=0;row<rows;++row)for(int k=0;k<256;++k)a[static_cast<std::size_t>(row)*256+k]=bf16(k%128==127?448.F:float(row+1));
    input.write<std::uint16_t>(a);auto run=q::take(child.create_invocation());q::check(run.prepare_resources());q::check(run.bind(x,input.read()));
    q::Tensor caller;if(facts.caller_owned()){caller=q::Tensor(q::take(backend.resource_for(facts.storage)),facts.description);q::check(run.bind(y,caller.view()));}
    auto queues=q::take(hyle::inference::prepare_queue_bindings(backend.queue_bindings(),report));const auto bindings=hyle::inference::queue_span(queues);
    q::check(run.prepare_replay(bindings,{.mode=q::g::ReplayMode::kCommands,.host_wait=q::g::HostWaitPolicy::kAllow}));auto done=q::take(run.submit(bindings));q::check(done.wait());
    q::c::ReadLease result;if(facts.caller_owned()){caller.freeze(backend);result=caller.ready;}else result=q::take(hyle::inference::retain_output(run,facts));
    std::vector<std::byte> raw(q::take(facts.description.required_bytes()));q::check(backend.read(q::take(result.view()),raw));
    for(int row=0;row<rows;++row)for(int n=0;n<129;++n){
      // E4 values and scale1 make A QDQ exact. All products/sums are exact F32
      // integers or halves; only the public BF16 endpoint rounds this result.
      const auto sum=127*(row+1)+448;const auto expected=bf16(float(sum)*(n==128?3.F:1.5F));
      std::uint16_t actual=0;std::memcpy(&actual,raw.data()+(static_cast<std::size_t>(row)*129+n)*2,2);
      if(actual!=expected)throw std::runtime_error("CT Recipe block-row/scale or BF16 result differs");
    }
    if(previous){std::vector<std::byte> held(previous_bytes.size());q::check(backend.read(q::take(previous.view()),held));if(held!=previous_bytes)throw std::runtime_error("old CT output was overwritten");}
    previous=result;previous_bytes=raw;
  }
  std::vector<std::uint8_t> held_codes(codes.size());q::check(backend.read(q::take(source_read.view()),std::as_writable_bytes(std::span(held_codes))));
  if(held_codes!=codes)throw std::runtime_error("original CT weight read lease changed");
  const std::array<float,1> global{1.F};const std::array<std::uint8_t,16> fp4{};const std::array<std::uint8_t,2> fp4_sf{0x38,0x38};
  add("fixture.local.weight_packed",q::Type::kUInt8,{2,8},fp4,q::prepared::ProjectionCodec::kCtLocalNvfp4Bfloat16,16);
  add("fixture.local.weight_scale",q::Type::kFloat8E4M3FN,{2,1},fp4_sf);
  add("fixture.local.weight_global_scale",q::Type::kFloat32,{1},global);add("fixture.local.input_global_scale",q::Type::kFloat32,{1},global);
  const auto local=q::resolve_projection(bank,"fixture.local.weight");
  if(std::get<q::LocalNvfp4Projection>(local).equation.rounding!=q::c::model_ops::Fp4InputRounding::kBFloat16)
    throw std::runtime_error("CT block128 source policy lost local BF16 prerounding");
  std::cout<<"{\"fixture\":\"generated_ct_block128\",\"recipe_calls\":2,\"rows\":[1,3],\"columns\":129,\"depth\":256,\"checked_values\":516,\"old_weight_and_output_lease\":true,\"local_bf16_resolver\":true,\"original_payload_numeric\":false}\n";
  return 0;
}

}
int main(int argc,char**argv){try{
  if(argc==2&&std::string_view(argv[1])=="--ct-block128")return check_ct_block128();
  if(argc==4&&std::string_view(argv[1])=="--packed-head")return check_packed_head(argv[2],argv[3]);
  if(argc!=4)throw std::runtime_error("qwen_mtp_projection_width_probe PREPARED_Q35_INPUTS FIXED_BF16_A OUTPUT_DIRECTORY");
  const std::filesystem::path inputs=argv[1],fixed_a=argv[2],output=argv[3];std::filesystem::create_directories(output);
  const auto geometry=q::probe::read_two_b_geometry(inputs);auto roles=q::read_weight_roles(inputs/"weights.tsv",false);
  const std::string name="model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
  const auto found=std::ranges::find(roles.selected,name,&q::prepared::Role::name);
  if(found==roles.selected.end()||found->dtype!="BF16"||found->shape!=std::vector<std::int64_t>{6144,2048}||found->bytes!=25165824)throw std::runtime_error("fixed projection requires the original BF16 Q35 physical role");
  const auto weight_raw=exact_file(found->source,found->bytes,found->offset),a_raw=exact_file(fixed_a,4096);
  auto backend=q::take(q::Backend::create());std::map<std::string,q::LoadedWeight> weights;
  auto &weight=weights.try_emplace(name,q::LoadedWeight{q::Tensor(backend.device(),q::dense(q::Type::kBFloat16,{6144,2048}))}).first->second;weight.tensor.write<std::byte>(weight_raw);
  q::ProjectionBindings projections;projections.emplace(name,q::resolve_projection(weights,name));
  q::GraphAssembly assembly(backend);
  q::Recipe recipe(weights,projections,assembly,geometry,5);
  const auto x=recipe.input(q::Recipe::dense_spec(q::Type::kBFloat16,{recipe.packed_rows,q::f(2048)}));
  const auto y=recipe.linear_absolute(x,name);recipe.publish(y);const auto graph=q::take(std::move(recipe.builder).finish());
  const std::array shapes{std::array{q::g::ShapeBinding{recipe.last_row,0}},std::array{q::g::ShapeBinding{recipe.last_row,2}},std::array{q::g::ShapeBinding{recipe.last_row,4}}};
  const std::array variants{q::g::ExactShapeVariant{shapes[0]},q::g::ExactShapeVariant{shapes[1]},q::g::ExactShapeVariant{shapes[2]}};
  auto compiled=q::take(backend.compile(graph,assembly.constants,assembly.options({.shape_policy=q::g::CompileShapePolicy::kFiniteExact,.exact_variants=variants,.constants=assembly.constants})));
  const std::array<std::size_t,3> widths{1,3,5};
  for(std::size_t index=0;index<widths.size();++index){
    const auto rows=widths[index];const auto child=q::take(compiled.executable.exact_variant(index));const auto report=q::take(compiled.report.exact_variant(index));
    const auto output_facts=q::take(hyle::inference::prepare_output(q::g::CompiledGraph{child,report},y));
    const auto desc=output_facts.description;q::Tensor input(backend.device(),q::dense(q::Type::kBFloat16,{static_cast<std::int64_t>(rows),2048}));
    std::vector<std::byte> data;for(std::size_t row=0;row<rows;++row)data.insert(data.end(),a_raw.begin(),a_raw.end());input.write<std::byte>(data);
    auto run=q::take(compiled.executable.create_invocation({.shape=shapes[index]}));q::check(run.prepare_resources());q::check(run.bind(x,input.read()));q::Tensor result;
    if(output_facts.caller_owned()){result=q::Tensor(q::take(backend.resource_for(output_facts.storage)),desc);q::check(run.bind(y,result.view()));}
    auto queues=q::take(hyle::inference::prepare_queue_bindings(backend.queue_bindings(),report));const auto bindings=hyle::inference::queue_span(queues);q::check(run.prepare_replay(bindings,{.mode=q::g::ReplayMode::kCommands,.host_wait=q::g::HostWaitPolicy::kAllow}));auto done=q::take(run.submit(bindings));q::check(done.wait());
    std::vector<std::byte> raw(q::take(desc.required_bytes()));
    if(result.buffer)q::check(backend.read(result.buffer.view(),raw));else{auto lease=q::take(hyle::inference::retain_output(run,output_facts));q::check(backend.read(q::take(lease.view()),raw));}
    save(output/("T"+std::to_string(rows)+"-row0.bf16"),std::span(raw).first(6144*2));
    std::ofstream route(output/("T"+std::to_string(rows)+"-route.tsv"));route<<"step\tprovider\timplementation\toperation\n";
    for(std::size_t step=0;step<report.step_count();++step){const auto s=q::take(report.step(step));route<<step<<'\t'<<s.provider.name<<'\t'<<s.implementation.value();for(std::size_t n=0;n<s.node_count;++n){const auto node=q::take(report.step_node(step,n));route<<'\t'<<q::c::operation_name(q::take(graph.operation_kind(node)));}route<<'\n';}
  }
  save(output/"fixed-A.bf16",a_raw);
  std::ofstream facts(output/"observation.json");facts<<"{\"status\":\"FIXED_A_T1_T3_T5_COMPLETE\",\"source_identity\":"<<std::quoted(roles.identity)<<",\"source_file\":"<<std::quoted(found->source.string())<<",\"source_offset\":"<<found->offset<<",\"source_bytes\":"<<found->bytes<<",\"graph_calls\":3,\"rows\":[1,3,5],\"A_row_bytes\":4096,\"output_row_elements\":6144,\"full_model_claim\":false,\"default_hot_graph_instrumented\":false,\"cross_width_equality_required\":false}\n";
  std::cout<<"{\"projection_graph_calls\":3,\"widths\":[1,3,5],\"default_hot_graph_instrumented\":false}\n";return 0;
}catch(const std::exception&error){std::cerr<<"fixed projection: "<<error.what()<<'\n';return 1;}}
