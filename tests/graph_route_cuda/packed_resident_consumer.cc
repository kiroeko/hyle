#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/model_ops.h>
#include <hyle/compute/providers/torch_cuda.h>
#include <hyle/graph/cuda.h>
#include <hyle/mem/buffer.h>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template<class T> void write(mem::Buffer &buffer, const std::vector<T> &values) {
  require(buffer.view().write(std::as_bytes(std::span(values))));
}
std::vector<float> read(mem::Buffer &buffer) {
  std::vector<float> result(buffer.size() / sizeof(float));
  require(buffer.view().read(std::as_writable_bytes(std::span(result))));
  return result;
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}));
}
void run(std::int64_t rows) {
  constexpr std::int64_t n = 9, k = 128, group = 64, experts = 2, slots = 2;
  auto context = take(g::cuda::create_context());
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  auto queue = take(context.create_queue());
  auto second_queue = take(context.create_queue());
  const std::array desc{
      dense(c::DataType::kFloat32, {rows,k}),
      dense(c::DataType::kFloat32, {1,slots,k}),
      dense(c::DataType::kFloat32, {1,slots,3}),
      dense(c::DataType::kUInt32, {n,k/8}),
      dense(c::DataType::kFloat32, {n,k/group}),
      dense(c::DataType::kFloat32, {n,k/group}),
      dense(c::DataType::kUInt32, {experts,n,k/8}),
      dense(c::DataType::kFloat32, {experts,n,k/group}),
      dense(c::DataType::kFloat32, {experts,n,k/group})};
  // Linear and grouped inputs share the same original packed backing.
  std::array<mem::Buffer,6> inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &d = desc[i < 3 ? i : i+3];
    inputs[i] = take(mem::Buffer::create(memory.ref(), take(d.required_bytes()), 256));
  }
  std::array<c::ConstTensorView,9> views;
  for (std::size_t i = 0; i < views.size(); ++i) {
    const auto backing = i < 3 ? i : i < 6 ? i : i-3;
    auto range = take(inputs[backing].view().subview(0, take(desc[i].required_bytes())));
    views[i] = take(c::ConstTensorView::create(desc[i], range));
  }
  const std::array out_desc{dense(c::DataType::kFloat32,{rows,n}),
                            dense(c::DataType::kFloat32,{1,slots,n})};
  std::array<mem::Buffer,3> linear, selected;
  for (std::size_t i = 0; i < linear.size(); ++i) {
    linear[i] = take(mem::Buffer::create(memory.ref(), take(out_desc[0].required_bytes()), 256));
    selected[i] = take(mem::Buffer::create(memory.ref(), take(out_desc[1].required_bytes()), 256));
  }
  std::vector<std::uint32_t> words(experts*n*k/8);
  std::vector<float> scales(experts*n*k/group), biases(scales.size());
  std::vector<float> activation(rows*k), grouped(slots*k), scores(slots*3);
  std::vector<float> retained_linear, retained_selected;
  {
    auto builder = take(g::GraphBuilder::create());
    std::array<g::ValueId,9> ids;
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = take(builder.input(desc[i]));
    const auto projection = take(builder.add(
        c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine4,group),
        ids[0],ids[3],ids[4],ids[5]));
    const auto top = take(builder.add_node(c::Operation(c::TopK{.count=1}), std::array{ids[2]}));
    const auto produced = take(builder.result(top,1));
    const auto chosen = take(builder.add(take(c::View::reshape(std::array<std::int64_t,2>{1,slots})), produced));
    const auto product = take(builder.add(
        c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine4,group),
        ids[1],ids[6],ids[7],ids[8],chosen));
    require(builder.publish(projection));
    require(builder.publish(product));
    auto logical = take(std::move(builder).finish());
    auto compiled = take(g::cuda::compile(logical, context, memory.ref()));
    std::size_t packed_steps = 0;
    for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
      const auto step = take(compiled.report.step(i));
      if (step.provider.id == c::torch_cuda::kProviderId) {
        ++packed_steps;
        check(!step.runtime.blocks_host, "Graph selected a host-blocking packed implementation");
      }
    }
    check(packed_steps == 2, "Graph did not route both packed operations to the actual Provider");
    auto first = take(compiled.executable.create_invocation());
    auto second = take(compiled.executable.create_invocation());
    for (auto *slot : {&first,&second})
      for (std::size_t i = 0; i < ids.size(); ++i) require(slot->bind(ids[i], views[i]));
    const auto bind = [&](g::Invocation &slot, c::Queue &target, std::size_t bank) {
      require(slot.bind(projection, take(c::TensorView::create(out_desc[0],linear[bank].view()))));
      require(slot.bind(product, take(c::TensorView::create(out_desc[1],selected[bank].view()))));
      require(g::cuda::prepare_replay(slot, target,
          {.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}));
    };
    bind(first,queue,0); bind(second,second_queue,1);
    const auto oracle = [&](const std::vector<float> &a, std::int64_t row,
                            std::int64_t expert, std::int64_t column) {
      double result = 0;
      for (std::int64_t j = 0; j < k; ++j) {
        const auto wr = expert*n+column, scale_index = wr*(k/group)+j/group;
        const auto code = (words[wr*(k/8)+j/8] >> (4*(j%8))) & 15U;
        result += double(a[row*k+j])*(double(code)*scales[scale_index]+biases[scale_index]);
      }
      return float(result);
    };
    for (int step = 0; step <= 64; ++step) {
      for (std::size_t i = 0; i < words.size(); ++i)
        words[i] = 0x11111111U*std::uint32_t((i+std::size_t(step))%16);
      for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = .125F*float(1+(i+std::size_t(step))%4);
        biases[i] = .125F*float(int(i%3)-1);
      }
      for (std::int64_t row = 0; row < rows; ++row) {
        activation[row*k] = float(1+row%5)+float(step)*.125F;
        activation[row*k+1] = -.25F; activation[row*k+65] = .5F;
      }
      std::array<std::int64_t,slots> expected_ids{};
      for (std::int64_t slot = 0; slot < slots; ++slot) {
        grouped[slot*k] = float(slot+1); grouped[slot*k+1] = -.5F;
        grouped[slot*k+65] = float(step)*.125F;
        expected_ids[slot] = step%3 == 0 ? 1 : (step+slot)%2;
        for (std::int64_t e = 0; e < 3; ++e)
          scores[slot*3+e] = e == expected_ids[slot] ? 10.F : -1.F;
      }
      write(inputs[0],activation); write(inputs[1],grouped); write(inputs[2],scores);
      write(inputs[3],words); write(inputs[4],scales); write(inputs[5],biases);
      auto &slot = step%2 ? second : first;
      const std::size_t bank = step == 0 ? 0 : step%2 ? 1 : 2;
      auto done = take(slot.submit(step%2 ? second_queue : queue));
      if (step != 0 && step%16 == 0) {
        // Two accepted streams read the same backing; both retire before
        // the caller changes inputs for the next frame.
        auto concurrent = take(second.submit(second_queue));
        require(done.wait()); require(concurrent.wait());
        check(read(selected[1]) == read(selected[2]), "Independent Graph slots interfered");
      } else require(done.wait());
      const auto actual = read(linear[bank]), picked = read(selected[bank]);
      for (std::int64_t row = 0; row < rows; ++row)
        for (std::int64_t column = 0; column < n; ++column)
          check(actual[row*n+column] == oracle(activation,row,0,column), "Graph linear stored-byte oracle differs");
      for (std::int64_t s = 0; s < slots; ++s)
        for (std::int64_t column = 0; column < n; ++column)
          check(picked[s*n+column] == oracle(grouped,s,expected_ids[s],column), "Graph selected stored-byte oracle differs");
      if (step == 0) {
        retained_linear = actual; retained_selected = picked;
        bind(first,queue,2); // One explicit cold rebind; subsequent runs keep two fixed banks.
      }
    }
    const auto before = read(selected[2]);
    for (std::int64_t s = 0; s < slots; ++s) {
      scores[s*3] = scores[s*3+1] = -1.F; scores[s*3+2] = 10.F;
    }
    write(inputs[2],scores); // TopK now produces expert 2 on device; bank has only two.
    auto failed = take(first.submit(queue));
    check(failed.wait().IsOutOfRange(), "Device-produced invalid expert was not checked");
    check(!first.output(product), "Failed Graph published a usable result");
    check(read(selected[2]) == before, "Checked selection changed output on invalid IDs");
  }
  check(read(linear[0]) == retained_linear && read(selected[0]) == retained_selected,
        "Graph teardown changed caller-owned prior outputs");
  std::printf("PASS CUDA Graph packed M=%lld:65 steps/two slots/shared backing/device IDs/checked failure/retained outputs\n",
              static_cast<long long>(rows));
}
} // namespace
int main() {
  try { for (const std::int64_t rows : {1,17,64}) run(rows); return 0; }
  catch (const std::exception &e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
