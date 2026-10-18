#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/graph/cpu.h"

namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
template<class T> T take(hyle::Result<T> r) {
  if (!r) { std::fprintf(stderr, "%.*s\n", int(r.error().message().size()), r.error().message().data()); std::exit(2); }
  return std::move(*r);
}
void require(hyle::Status s) {
  if (!s.ok()) { std::fprintf(stderr, "%.*s\n", int(s.message().size()), s.message().data()); std::exit(2); }
}
int main() {
  const auto d = take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t,2>{1,8}));
  const auto rhs = take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t,2>{8,1}));
  const auto out = take(c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t,2>{1,1}));
  auto make = [](std::size_t n) { return take(m::Buffer::create(m::system_resource(), n, 64)); };
  auto a = make(32), b = make(32), y = make(32), weights = make(32), z = make(4);
  auto ctx = take(c::cpu::create_context());
  auto q = take(ctx.create_queue());
  const std::array inputs{take(c::ConstTensorView::create(d,a.view())), take(c::ConstTensorView::create(d,b.view()))};
  const auto dest = take(c::TensorView::create(d,y.view()));
  const c::GatedActivation op{.activation=c::GatedActivationKind::kSilu, .inputs=c::GatedInputKind::kSeparate};
  auto p = take(c::prepare(ctx,c::Operation{op},inputs,std::span(&dest,1)));
  auto lane = take(p.create_invocation());
  auto unary = take(c::prepare(ctx,c::Unary{c::UnaryKind::kSigmoid},d));
  auto ui = take(unary.create_invocation());
  auto builder = take(g::GraphBuilder::create());
  const auto ga = take(builder.input(d)), gb = take(builder.input(d)), gw = take(builder.input(rhs));
  const auto activated = take(builder.add(op,ga,gb));
  const auto projected = take(builder.add(c::MatMul{},activated,gw));
  require(builder.publish(projected));
  auto graph = take(std::move(builder).finish());
  auto gc = take(g::cpu::create_context());
  auto compiled = take(g::compile(graph,gc));
  auto gq = take(gc.create_queue());
  auto gi = take(compiled.executable.create_invocation());
  require(gi.bind(ga,inputs[0])); require(gi.bind(gb,inputs[1]));
  require(gi.bind(gw,take(c::ConstTensorView::create(rhs,weights.view()))));
  require(gi.bind(projected,take(c::TensorView::create(out,z.view()))));
  const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&gq}};
  require(gi.prepare_replay(queues));
  std::array<float,8> w{}; w.fill(1);
  require(weights.view().write(std::as_bytes(std::span(w))));
  std::printf("Direct gate=%.*s sigmoid=%.*s Graph steps=%zu first=%.*s\n",int(p.provider().name.size()),p.provider().name.data(),int(unary.provider().name.size()),unary.provider().name.data(),compiled.report.step_count(),int(compiled.report.step(0)->provider.name.size()),compiled.report.step(0)->provider.name.data());
  int mismatches = 0;
  for (float x : {-80.F,-90.F,-100.F}) for (float v : {1e38F,std::numeric_limits<float>::infinity()}) {
    std::array<float,8> gate{},values{}; gate[0]=x; values[0]=v;
    require(a.view().write(std::as_bytes(std::span(gate)))); require(b.view().write(std::as_bytes(std::span(values))));
    require(take(lane.submit(q,inputs,std::span(&dest,1))).wait());
    std::array<float,8> actual{}; require(y.view().read(std::as_writable_bytes(std::span(actual))));
    require(take(gi.submit(gq)).wait()); float projected_actual=0; require(z.view().read(std::as_writable_bytes(std::span(&projected_actual,1))));
    const auto e = std::exp(double(x));
    const float activation = static_cast<float>(double(x)*(e/(1+e)));
    const float expected = activation*v;
    const auto matches = [&](float value) { return std::isinf(expected) ? value==expected : std::abs(value-expected)<=3e-6*std::abs(expected)+4*std::numeric_limits<float>::denorm_min(); };
    const bool good = matches(actual[0]) && matches(projected_actual);
    mismatches += !good;
    std::printf("x=%g value=%g direct=%+.9g downstream=%+.9g oracle=%+.9g %s\n",x,v,actual[0],projected_actual,expected,good?"MATCH":"MISMATCH");
    require(take(ui.submit(q,inputs[0],dest)).wait()); require(y.view().read(std::as_writable_bytes(std::span(actual))));
    std::printf("sigmoid x=%g actual=%+.9g oracle=%+.9g\n",x,actual[0],static_cast<float>(e/(1+e)));
  }
  require(ctx.close()); require(gc.close());
  std::printf("finite-tail diagnostic mismatches=%d (no timing, no Native comparison)\n",mismatches);
  for (auto dtype : {c::DataType::kFloat16,c::DataType::kBFloat16}) {
    auto fresh=take(c::cpu::create_context());auto queue=take(fresh.create_queue());
    auto pcast=take(c::prepare(fresh,c::Cast{dtype},d));
    auto result=make(16);auto ci=take(pcast.create_invocation());
    std::array<float,8> source{};source[0]=std::bit_cast<float>(std::uint32_t{0x7fc12345});source[1]=std::bit_cast<float>(std::uint32_t{0xffc12345});
    require(a.view().write(std::as_bytes(std::span(source))));
    require(take(ci.submit(queue,inputs[0],take(c::TensorView::create(take(pcast.output_description(0)),result.view())))).wait());
    std::array<std::uint16_t,8> bits{};require(result.view().read(std::as_writable_bytes(std::span(bits))));
    std::printf("Cast target=%d original-SDK-provider=%.*s NaN bits=%04x/%04x\n",int(dtype),int(pcast.provider().name.size()),pcast.provider().name.data(),unsigned(bits[0]),unsigned(bits[1]));
    require(fresh.close());
  }
}
