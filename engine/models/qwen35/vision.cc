#include "vision.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "../../inference/artifact_digest.h"
#include "../../inference/program_carrier.h"
#include "../../inference/program_io.h"
#include "hyle/simd/dtype.h"

namespace qwen35 {
namespace inference = hyle::inference;
namespace {
std::vector<std::string> fields(const std::string &line) {
  std::vector<std::string> result;
  std::size_t begin = 0;
  for (;;) {
    const auto end = line.find('\t', begin);
    result.push_back(line.substr(begin, end - begin));
    if (end == std::string::npos) return result;
    begin = end + 1;
  }
}
std::size_t integer(const std::string &word) {
  std::size_t value = 0;
  const auto parsed = std::from_chars(word.data(), word.data() + word.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != word.data() + word.size())
    throw std::runtime_error("invalid vision asset integer");
  return value;
}
std::map<std::string, std::vector<std::int64_t>> expected_roles(const VisionGeometry &v) {
  using Shape = std::vector<std::int64_t>;
  const auto h = static_cast<std::int64_t>(v.hidden), ffn = static_cast<std::int64_t>(v.intermediate);
  const auto merged = static_cast<std::int64_t>(cache::multiply(v.hidden, cache::multiply(v.merge, v.merge)));
  std::map<std::string, Shape> roles;
  const auto add = [&](std::string name, Shape shape) { roles.emplace("model.visual." + name, std::move(shape)); };
  add("patch_embed.proj.weight", {h, static_cast<std::int64_t>(cache::multiply(3,
      cache::multiply(v.temporal,cache::multiply(v.patch,v.patch))))});
  add("patch_embed.proj.bias", {h}); add("pos_embed.weight", {static_cast<std::int64_t>(v.positions), h});
  add("merger.norm.weight", {h}); add("merger.norm.bias", {h});
  add("merger.linear_fc1.weight", {merged, merged}); add("merger.linear_fc1.bias", {merged});
  add("merger.linear_fc2.weight", {static_cast<std::int64_t>(v.output), merged});
  add("merger.linear_fc2.bias", {static_cast<std::int64_t>(v.output)});
  for (std::size_t i = 0; i < v.layers; ++i) {
    const auto prefix = "blocks." + std::to_string(i) + ".";
    for (const auto &[name, shape] : std::map<std::string, Shape>{
        {"attn.qkv.weight", {3*h,h}}, {"attn.qkv.bias", {3*h}},
        {"attn.proj.weight", {h,h}}, {"attn.proj.bias", {h}},
        {"mlp.linear_fc1.weight", {ffn,h}}, {"mlp.linear_fc1.bias", {ffn}},
        {"mlp.linear_fc2.weight", {h,ffn}}, {"mlp.linear_fc2.bias", {h}},
        {"norm1.weight", {h}}, {"norm1.bias", {h}}, {"norm2.weight", {h}}, {"norm2.bias", {h}}})
      add(prefix + name, shape);
  }
  return roles;
}
} // namespace

VisionSource read_vision_source(const std::filesystem::path &path, const prepared::Roles &target) {
  if(std::filesystem::file_size(path)>(8u<<20))throw std::runtime_error("vision asset exceeds the bounded metadata reader");
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open vision asset: " + path.string());
  const auto line = [&] { std::string value; if (!prepared::read_manifest_line(input, value))
      throw std::runtime_error("short vision asset"); if (value.ends_with('\r')) value.pop_back(); return value; };
  std::array<std::string, 6> header;
  for (auto &value : header) value = line();
  if (header[0] != "HYLE_QWEN_VISION_V1" || header[1] != target.identity)
    throw std::runtime_error("vision asset is not bound to the loaded language source");
  prepared::validate_identity(header[5]);
  const auto geometry = fields(header[2]), special = fields(header[3]), rope = fields(header[4]);
  if (geometry.size() != 9 || special.size() != 3 || rope.size() != 3)
    throw std::runtime_error("invalid vision asset geometry");
  VisionSource result;
  auto &v = result.geometry;
  std::array<std::size_t*, 9> destinations{&v.layers,&v.hidden,&v.intermediate,&v.heads,&v.patch,&v.temporal,&v.merge,&v.positions,&v.output};
  for (std::size_t i = 0; i < destinations.size(); ++i) {
    const auto value = integer(geometry[i]);
    if (!value || value > INT32_MAX)
      throw std::runtime_error("vision dimensions exceed the tensor index range");
    *destinations[i] = value;
  }
  std::array<std::int32_t*, 3> ids{&v.image_id,&v.start_id,&v.end_id};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto id = integer(special[i]);
    if (id >= static_cast<std::size_t>(target.geometry.vocabulary)) throw std::runtime_error("vision token outside vocabulary");
    *ids[i] = static_cast<std::int32_t>(id); v.mrope[i] = integer(rope[i]);
    if (v.mrope[i] > static_cast<std::size_t>(target.geometry.rotary) / 2)
      throw std::runtime_error("vision rotary section exceeds the language rotary prefix");
  }
  if(v.layers>(4096-9)/12)throw std::runtime_error("vision tensor inventory exceeds the bounded source reader");
  if (!v.layers || !v.hidden || !v.intermediate || !v.heads || v.hidden % v.heads ||
      (v.hidden / v.heads) % 4 || !v.patch || !v.temporal || !v.merge ||
      !v.positions || static_cast<std::size_t>(std::sqrt(static_cast<double>(v.positions))) *
          static_cast<std::size_t>(std::sqrt(static_cast<double>(v.positions))) != v.positions ||
      v.output != static_cast<std::size_t>(target.geometry.hidden) ||
      2 * (v.mrope[0] + v.mrope[1] + v.mrope[2]) != static_cast<std::size_t>(target.geometry.rotary))
    throw std::runtime_error("vision source differs from the supported Qwen image equation");
  auto expected = expected_roles(v);
  std::string identity_text;
  for (std::size_t i = 0; i < 5; ++i) identity_text += header[i] + '\n';
  result.roles.geometry = target.geometry;
  std::string row;
  while (prepared::read_manifest_line(input, row)) {
    if (row.ends_with('\r')) row.pop_back();
    const auto parts = fields(row);
    if (parts.size() != 7) throw std::runtime_error("invalid vision tensor record");
    prepared::Role role;
    role.name = parts[0]; role.dtype = parts[1]; role.source = std::filesystem::path(std::u8string(parts[3].begin(),parts[3].end()));
    role.offset = integer(parts[4]); role.bytes = integer(parts[5]);
    std::istringstream shape(parts[2]); std::string word;
    while (std::getline(shape, word, ',')) role.shape.push_back(static_cast<std::int64_t>(integer(word)));
    const auto found = expected.find(role.name);
    const auto *encoding=gguf_capabilities(role.dtype);
    const bool encoded=encoding && role.shape.size()==2 && role.name.ends_with(".weight") &&
        role.name!="model.visual.pos_embed.weight";
    const auto physical=encoded?take(take(c::gguf_weight_description(encoding->encoding,Type::kFloat32,
        role.shape[1],role.shape[0])).required_bytes()):prepared::physical_bytes(role.dtype,role.shape);
    if (found == expected.end() || found->second != role.shape || physical != role.bytes)
      throw std::runtime_error("invalid vision tensor: " + role.name);
    expected.erase(found); prepared::validate_identity(parts[6]);
    std::ifstream source(role.source, std::ios::binary);
    if (!source || role.offset > static_cast<std::uint64_t>((std::numeric_limits<std::streamoff>::max)()))
      throw std::runtime_error("cannot read visual weight source");
    const auto available=std::filesystem::file_size(role.source);
    if(role.offset>available || role.bytes>available-role.offset)throw std::runtime_error("short visual weight source");
    std::vector<std::byte> original(role.bytes);
    source.seekg(static_cast<std::streamoff>(role.offset));
    source.read(reinterpret_cast<char*>(original.data()), static_cast<std::streamsize>(original.size()));
    if (!source || inference::artifact::sha256(original) != parts[6])
      throw std::runtime_error("visual weight source hash differs: " + role.name);
    result.roles.bytes = cache::add(result.roles.bytes, role.bytes);
    result.roles.selected.push_back(std::move(role)); identity_text += row + '\n';
  }
  if (!expected.empty() || inference::artifact::sha256(std::as_bytes(std::span(identity_text))) != header[5])
    throw std::runtime_error("vision asset inventory or identity differs");
  result.identity = header[5]; result.roles.identity = result.identity;
  return result;
}

V apply_position_angles(Recipe &r, V value, V cosine, V sine, std::size_t rotary) {
  const auto original = r.spec(value);
  if (original.rank() != 4 || !original.dimensions()[3].is_fixed() ||
      rotary > static_cast<std::size_t>(original.dimensions()[3].fixed_value()) || !rotary || rotary % 2)
    throw std::runtime_error("explicit position angles require [B,T,H,D] and an even rotary prefix");
  const auto half = f(static_cast<std::int64_t>(rotary / 2));
  const auto first = r.cast(r.slice(value, 3, f(0), half), Type::kFloat32);
  const auto second = r.cast(r.slice(value, 3, half, half), Type::kFloat32);
  const auto left = r.binary(c::BinaryKind::kSubtract, r.binary(c::BinaryKind::kMultiply, first, cosine),
      r.binary(c::BinaryKind::kMultiply, second, sine));
  const auto right = r.binary(c::BinaryKind::kAdd, r.binary(c::BinaryKind::kMultiply, second, cosine),
      r.binary(c::BinaryKind::kMultiply, first, sine));
  std::vector<V> parts{r.cast(left, original.data_type()), r.cast(right, original.data_type())};
  const auto tail = original.dimensions()[3].fixed_value() - static_cast<std::int64_t>(rotary);
  if (tail) parts.push_back(r.slice(value, 3, f(static_cast<std::int64_t>(rotary)), f(tail)));
  return r.copy(r.concat(parts, 3, original));
}

struct VisionEncoder::Program final {
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  inference::ProgramIo io;
  inference::ProgramCarrier<VisionEncoder> carrier;
  Tensor pixels, indices, coefficients, cosine, sine;
  V output;
  inference::ProgramIo::Output result;
  c::LeasePool output_pool;
};

VisionEncoder::VisionEncoder(inference::Backend &backend, const prepared::Roles &target,
    const std::filesystem::path &path, const c::ResourceBudget &budget)
    : backend_(take(backend.share_endpoint())), budget_(budget), source_(read_vision_source(path, target)) {
  load_selected_weight_roles(source_.roles, backend_, weights_, pools_, FfnPlacement(source_.roles, 0), budget_);
  for (auto &[name, weight] : weights_)
    if (name.ends_with(".weight") && weight.tensor.description.rank() == 2 && name != "model.visual.pos_embed.weight")
      projections_.emplace(name, StoredProjection{&weight.tensor});
}
VisionEncoder::~VisionEncoder() {
  for (auto &[key, program] : programs_) {
    (void)key; (void)program->carrier.retire(backend_);
    program->carrier.release_invocation_root();
  }
}
void VisionEncoder::validate(const inference::PromptAttachment &part) const {
  const auto &v = source_.geometry;
  if (part.format != "qwen35.image.patches.f32.v1" || part.shape.size() != 3 || part.shape[0] != 1 ||
      !part.shape[1] || !part.shape[2] || part.shape[1] % v.merge || part.shape[2] % v.merge)
    throw std::invalid_argument("image patch payload differs from the prepared vision recipe");
  const auto patches = cache::multiply(part.shape[1], part.shape[2]);
  if (part.rows != patches / cache::multiply(v.merge, v.merge) ||
      part.payload.size() != cache::multiply(cache::multiply(patches, 3 * v.temporal * v.patch * v.patch), sizeof(float)))
    throw std::invalid_argument("image patch rows or bytes differ from their exact shape");
  for (std::size_t at = 0; at < part.payload.size(); at += sizeof(float)) {
    float value; std::memcpy(&value, part.payload.data() + at, sizeof(value));
    if (!std::isfinite(value) || value < -1 || value > 1)
      throw std::invalid_argument("image pixels must be finite normalized RGB values");
  }
}

std::unique_ptr<VisionEncoder::Program> VisionEncoder::prepare(std::size_t height, std::size_t width, Type output_type) {
  auto program = std::make_unique<Program>(); auto &p = *program;
  const auto &v = source_.geometry;
  const auto n = static_cast<std::int64_t>(cache::multiply(height, width));
  const auto h = static_cast<std::int64_t>(v.hidden), head = h / static_cast<std::int64_t>(v.heads);
  GraphAssembly assembly(backend_, PrecisionProfile::float32_activations(), budget_);
  Recipe r(weights_, projections_, assembly, source_.roles.geometry, 1);
  const auto input = r.input(dense(Type::kFloat32, {n, static_cast<std::int64_t>(3*v.temporal*v.patch*v.patch)}));
  const auto indices = r.input(dense(Type::kInt32, {4*n}));
  const auto coefficients = r.input(dense(Type::kFloat32, {4*n,1}));
  const auto cosine = r.input(dense(Type::kFloat32, {1,n,1,head/2}));
  const auto sine = r.input(dense(Type::kFloat32, {1,n,1,head/2}));
  const auto linear = [&](V x, const std::string &name) {
    // The source Linear/Conv epilogue adds bias before narrowing. Narrowing
    // the product first accumulates a different rounding error over 24 blocks.
    const auto stored=r.weight_absolute(name+".weight");
    V product;
    if(weights_.at(name+".weight").tensor.description.is_plain()){
      constexpr std::array<std::int32_t,2> axes{1,0};
      const auto operand=r.one(take(c::View::permute(axes)),{r.cast(stored,Type::kFloat32)});
      product=r.one(c::MatMul{},{x,operand},r.dense_spec(Type::kFloat32,
          {r.spec(x).dimensions()[0],r.spec(stored).dimensions()[0]}));
    }else product=r.scores(x,name+".weight");
    return r.cast(r.binary(c::BinaryKind::kAdd, product,
        r.cast(r.weight_absolute(name + ".bias"), Type::kFloat32)), r.activation_type());
  };
  const auto norm = [&](V x, const std::string &name) {
    const auto f32 = r.cast(x, Type::kFloat32);
    return r.cast(r.same(c::LayerNorm{.epsilon=1e-6F}, {f32,
        r.cast(r.weight_absolute(name + ".weight"), Type::kFloat32),
        r.cast(r.weight_absolute(name + ".bias"), Type::kFloat32)}, f32), r.activation_type());
  };
  auto x = linear(r.cast(input, r.activation_type()), "model.visual.patch_embed.proj");
  const auto table = r.cast(r.weight_absolute("model.visual.pos_embed.weight"),r.activation_type());
  const auto table_type = r.spec(table).data_type();
  auto positions = r.one(c::Gather{.axis=0}, {table, indices});
  positions = r.binary(c::BinaryKind::kMultiply, positions, r.cast(coefficients, table_type));
  auto position_sum = r.slice(positions, 0, f(0), f(n));
  for (std::int64_t i = 1; i < 4; ++i)
    position_sum = r.binary(c::BinaryKind::kAdd, position_sum, r.slice(positions, 0, f(i*n), f(n)));
  x = r.binary(c::BinaryKind::kAdd, x, r.cast(position_sum, r.activation_type()));
  for (std::size_t i = 0; i < v.layers; ++i) {
    const auto prefix = "model.visual.blocks." + std::to_string(i) + ".";
    const auto qkv = linear(norm(x, prefix + "norm1"), prefix + "attn.qkv");
    const auto q = apply_position_angles(r, r.reshape(r.copy(r.slice(qkv, 1, f(0), f(h))),
        {f(1),f(n),f(static_cast<std::int64_t>(v.heads)),f(head)}), cosine, sine, static_cast<std::size_t>(head));
    const auto k = apply_position_angles(r, r.reshape(r.copy(r.slice(qkv, 1, f(h), f(h))),
        {f(1),f(n),f(static_cast<std::int64_t>(v.heads)),f(head)}), cosine, sine, static_cast<std::size_t>(head));
    const auto value = r.reshape(r.copy(r.slice(qkv, 1, f(2*h), f(h))),
        {f(1),f(n),f(static_cast<std::int64_t>(v.heads)),f(head)});
    const auto attended = r.one(c::ScaledDotProductAttention{.scale=1.F/std::sqrt(static_cast<float>(head))}, {q,k,value},
        r.dense_spec(r.activation_type(), {f(1),f(n),f(static_cast<std::int64_t>(v.heads)),f(head)}));
    x = r.binary(c::BinaryKind::kAdd, x, linear(r.reshape(attended, {f(n),f(h)}), prefix + "attn.proj"));
    auto ff = linear(norm(x, prefix + "norm2"), prefix + "mlp.linear_fc1");
    ff = r.unary(c::UnaryKind::kGeluTanh, ff);
    x = r.binary(c::BinaryKind::kAdd, x, linear(ff, prefix + "mlp.linear_fc2"));
  }
  x = r.reshape(norm(x, "model.visual.merger.norm"),
      {f(n / static_cast<std::int64_t>(v.merge*v.merge)), f(h*static_cast<std::int64_t>(v.merge*v.merge))});
  x = linear(r.unary(c::UnaryKind::kGelu, linear(x, "model.visual.merger.linear_fc1")), "model.visual.merger.linear_fc2");
  p.output = r.copy(r.cast(x, output_type)); r.publish(p.output);
  p.graph = take(std::move(r.builder).finish());
  const std::array bindings{g::ShapeBinding{r.last_row,0}};
  auto compile = backend_.compile(p.graph, assembly.constants(), assembly.options({.shape=bindings,
      .constants=assembly.constants(), .resource_budget=budget_}));
  if (!compile) throw std::runtime_error("vision graph preparation: " + std::string(compile.error().message()));
  p.compiled = std::move(*compile);
  p.literals = std::move(assembly.literals);
  p.pixels = Tensor(backend_.device(), dense(Type::kFloat32, {n,static_cast<std::int64_t>(3*v.temporal*v.patch*v.patch)}), true, budget_);
  p.indices = Tensor(backend_.device(), dense(Type::kInt32,{4*n}), true, budget_);
  p.coefficients = Tensor(backend_.device(), dense(Type::kFloat32,{4*n,1}), true, budget_);
  p.cosine = Tensor(backend_.device(),dense(Type::kFloat32,{1,n,1,head/2}),true,budget_);
  p.sine = Tensor(backend_.device(),dense(Type::kFloat32,{1,n,1,head/2}),true,budget_);
  check(p.io.prepare(p.compiled,5,0,0,1));
  check(p.io.add_fixed(input,p.pixels.read())); check(p.io.add_fixed(indices,p.indices.read()));
  check(p.io.add_fixed(coefficients,p.coefficients.read())); check(p.io.add_fixed(cosine,p.cosine.read()));
  check(p.io.add_fixed(sine,p.sine.read())); p.result=take(p.io.add_output(p.output)); check(p.io.seal());
  const auto bytes = take(p.io.output_facts(p.result).description.required_bytes());
  p.output_pool = take(c::LeasePool::create(backend_.device(), bytes, {.resource_budget=budget_}));
  check(p.carrier.prepare(p.compiled,backend_.queue_bindings()));
  return program;
}

Tensor VisionEncoder::encode(const inference::PromptAttachment &part, Type output_type) {
  validate(part);
  const auto height=part.shape[1],width=part.shape[2]; const auto &v=source_.geometry;
  const std::array key{height,width,static_cast<std::size_t>(output_type)};
  if (!programs_.contains(key)) {
    // A small bounded inventory, not a Cartesian product of model/backend/
    // resolutions. Eviction only concerns retired encoder programs; escaped
    // result leases retain their original output pool independently.
    if (programs_.size()==2) {
      auto old=programs_.begin(); const auto retired=old->second->carrier.retire(backend_);
      check(retired.status); if (!retired.retired) throw std::runtime_error("visual program has not retired");
      programs_.erase(old);
    }
    programs_.emplace(key,prepare(height,width,output_type));
  }
  auto &p=*programs_.at(key);
  if (!p.carrier.initialized()) check(p.carrier.initialize({.io=p.io,.invocation={.resources=g::ResourceOptions{.resource_budget=budget_}}}));
  const auto n=height*width,side=static_cast<std::size_t>(std::sqrt(static_cast<double>(v.positions)));
  const auto half=v.hidden/v.heads/2;
  std::vector<std::int32_t> indices(4*n); std::vector<float> coefficients(4*n),cosine(n*half),sine(n*half);
  std::size_t at=0;
  for(std::size_t by=0;by<height;by+=v.merge)for(std::size_t bx=0;bx<width;bx+=v.merge)
    for(std::size_t dy=0;dy<v.merge;++dy)for(std::size_t dx=0;dx<v.merge;++dx,++at){
      const auto y=by+dy,x=bx+dx;
      const float fy=height==1?0.F:static_cast<float>(y)*static_cast<float>(side-1)/static_cast<float>(height-1);
      const float fx=width==1?0.F:static_cast<float>(x)*static_cast<float>(side-1)/static_cast<float>(width-1);
      const auto y0=static_cast<std::size_t>(fy),x0=static_cast<std::size_t>(fx);
      const auto y1=(std::min)(y0+1,side-1),x1=(std::min)(x0+1,side-1);
      const auto yh=fy-static_cast<float>(y0),xw=fx-static_cast<float>(x0);
      const std::array ids{y0*side+x0,y0*side+x1,y1*side+x0,y1*side+x1};
      const std::array weights{(1-yh)*(1-xw),(1-yh)*xw,yh*(1-xw),yh*xw};
      for(std::size_t j=0;j<4;++j){indices[j*n+at]=static_cast<std::int32_t>(ids[j]);coefficients[j*n+at]=weights[j];}
      for(std::size_t j=0;j<half;++j){
        const auto coordinate=j<half/2?y:x;
        const auto freq=j%(half/2);
        const auto angle=static_cast<float>(coordinate)*std::pow(10000.F,-static_cast<float>(2*freq)/static_cast<float>(half));
        cosine[at*half+j]=std::cos(angle);sine[at*half+j]=std::sin(angle);
      }
    }
  p.indices.write<std::int32_t>(indices); p.coefficients.write<float>(coefficients);
  p.cosine.write<float>(cosine);p.sine.write<float>(sine);check(p.pixels.storage().write(part.payload));
  auto &run=*take(p.carrier.borrow());
  check(p.io.bind_output(run,p.result,p.output_pool));
  try {
    check(run.prepare_replay(p.carrier.queues(),{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kAllow}));
    auto done=take(p.carrier.submit(shared_from_this()));check(done.wait());
    // The encoder carrier is reusable while several images remain in a turn.
    // Retaining its physical result alone must never authorize overwriting it.
    // Make one endpoint-local immutable handoff; no feature visits host memory.
    const auto description=p.io.output_facts(p.result).description;
    Tensor result(backend_.device(),description,true,budget_);
    auto produced=take(p.io.retain(run,p.result,budget_));
    check(result.storage().copy_from(take(produced.view())));
    produced.reset();
    check(p.io.release_inputs(p.carrier));
    result.freeze(backend_);
    return result;
  } catch (...) {
    const auto retired=p.carrier.retire(backend_);
    if(retired.retired)check(p.carrier.reset_retired());
    throw;
  }
}

ConditionedPrompt image_position_plan(const VisionGeometry &v, std::span<const std::int32_t> ids,
    std::span<const inference::PromptAttachment> images, std::size_t pending_rows,
    std::int64_t initial_position) {
  ConditionedPrompt result; result.positions.resize(ids.size());
  if(!v.merge)throw std::invalid_argument("image positions require a spatial merge geometry");
  std::size_t at=0; auto position=initial_position;
  for (const auto &image:images) {
    if(image.shape.size()!=3 || image.shape[0]!=1 || !image.shape[1] || !image.shape[2] ||
        image.shape[1]%v.merge || image.shape[2]%v.merge ||
        image.rows!=cache::multiply(image.shape[1]/v.merge,image.shape[2]/v.merge))
      throw std::invalid_argument("image position grid differs from its language rows");
    const auto begin=cache::add(image.begin,pending_rows);
    if(begin<at || begin==0 || begin>=ids.size() || image.rows>=ids.size()-begin ||
        ids[begin-1]!=v.start_id || ids[begin+image.rows]!=v.end_id ||
        !std::ranges::all_of(ids.subspan(begin,image.rows),[&](auto id){return id==v.image_id;}))
      throw std::invalid_argument("image payload does not match its exact language placeholders");
    for(;at<begin;++at,++position) {
      if(ids[at]==v.image_id)throw std::invalid_argument("image token has no conditioning payload");
      if(position<0 || position>INT32_MAX)throw std::out_of_range("image rotary position overflows");
      result.positions[at].fill(static_cast<std::int32_t>(position));
    }
    const auto height=image.shape[1]/v.merge,width=image.shape[2]/v.merge;
    if(position<0 || position>INT32_MAX-static_cast<std::int64_t>((std::max)(height,width)))
      throw std::out_of_range("image rotary extent overflows");
    for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x)
      result.positions[at++]={static_cast<std::int32_t>(position),static_cast<std::int32_t>(position+y),static_cast<std::int32_t>(position+x)};
    position+=static_cast<std::int64_t>((std::max)(height,width));
    result.images.push_back({begin,image.rows,{}});
  }
  for(;at<ids.size();++at,++position){
    if(ids[at]==v.image_id)throw std::invalid_argument("image token has no conditioning payload");
    if(position<0 || position>INT32_MAX)throw std::out_of_range("image rotary position overflows");
    result.positions[at].fill(static_cast<std::int32_t>(position));
  }
  return result;
}

ConditionedPrompt prepare_image_prompt(VisionEncoder &encoder, std::span<const std::int32_t> ids,
    std::span<const inference::PromptAttachment> images, std::size_t pending_rows,
    std::int64_t initial_position, Type activation_type) {
  for(const auto &image:images)encoder.validate(image);
  auto result=image_position_plan(encoder.geometry(),ids,images,pending_rows,initial_position);
  for(std::size_t i=0;i<images.size();++i)result.images[i].features=encoder.encode(images[i],activation_type);
  return result;
}
} // namespace qwen35
