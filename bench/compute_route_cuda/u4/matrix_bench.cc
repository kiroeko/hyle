#include "matrix_request.h"

#include <nanobench.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include "hyle/core/environment.h"
#include "../../common/perf_process.h"

namespace nb = ankerl::nanobench;
std::string environment(const char *name) {
  const auto *value = std::getenv(name);
  return value ? value : "";
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
c::Provider provider(const std::string &name) {
  if (name == "hyle") return c::cuda_native::provider();
  if (name == "ggml") return c::ggml_cuda::provider();
  if (name == "sglang") return c::cuda_native::sglang_provider();
  if (name == "torch") return c::torch_cuda::provider();
  if (name == "cccl") return c::cuda_native::cccl_provider();
  throw std::runtime_error("unknown Provider in frozen pair file");
}
c::DataType type(const std::string &name) {
  if (name == "f32") return c::DataType::kFloat32;
  if (name == "f16") return c::DataType::kFloat16;
  if (name == "bf16") return c::DataType::kBFloat16;
  throw std::runtime_error("unknown dtype in frozen pair file");
}
class Session {
public:
  Session(Request r, const std::string &name)
      : recipe_(std::move(r)), context_(take(c::Context::create(take(c::cuda::device(0)), provider(name)))),
        memory_(take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice))),
        queue_(take(context_.create_queue())), prepared_(take(c::prepare(context_, recipe_.operation,
            recipe_.inputs, std::span(&recipe_.output, 1), memory_.ref()))) {
    require(queue_.reserve_completions(2));
    for (int seed : {5, 17}) {
      Slot slot;
      const auto values = payload(recipe_, seed);
      slot.expected = oracle(recipe_, values);
      for (std::size_t i = 0; i < recipe_.inputs.size(); ++i) {
        const auto &desc = recipe_.inputs[i];
        slot.buffers.push_back(take(hyle::mem::Buffer::create(memory_.ref(), take(desc.required_bytes()), 256)));
        require(slot.buffers.back().view().write(pack(desc, values[i])));
        slot.inputs.push_back(take(c::ConstTensorView::create(desc, slot.buffers.back().view())));
      }
      slot.output = take(hyle::mem::Buffer::create(memory_.ref(), take(recipe_.output.required_bytes()), 256));
      slot.outputs.push_back(take(c::TensorView::create(recipe_.output, slot.output.view())));
      const auto workspace = prepared_.workspace_requirement();
      slot.workspace = take(hyle::mem::Buffer::create(memory_.ref(), workspace.size, workspace.alignment));
      slot.invocation = take(workspace.size ? prepared_.create_invocation(slot.workspace.view())
                                           : prepared_.create_invocation());
      slots_.push_back(std::move(slot));
    }
  }
  void run() {
    auto &slot = slots_[next_++ % slots_.size()];
    auto done = take(slot.invocation.submit(queue_, slot.inputs, slot.outputs));
    require(done.wait()); // All accepted work and retained refs settle before return.
  }
  void verify() {
    for (auto &slot : slots_) {
      {
        auto done = take(slot.invocation.submit(queue_, slot.inputs, slot.outputs));
        require(done.wait());
      }
      std::vector<std::byte> bytes(slot.output.size()); require(slot.output.view().read(bytes));
      const auto values = unpack(recipe_.output.data_type(), bytes);
      const double tolerance = recipe_.output.data_type() == c::DataType::kBFloat16 ? .008
          : recipe_.output.data_type() == c::DataType::kFloat16 ? .001 : 3.e-5;
      for (std::size_t i = 0; i < values.size(); ++i)
        if (!std::isfinite(values[i]) || std::abs(values[i] - slot.expected[i]) > tolerance * (1 + std::abs(slot.expected[i])))
          throw std::runtime_error("complete matrix output oracle failed");
    }
  }
private:
  struct Slot {
    std::vector<hyle::mem::Buffer> buffers;
    hyle::mem::Buffer output, workspace;
    std::vector<c::ConstTensorView> inputs;
    std::vector<c::TensorView> outputs;
    std::vector<double> expected;
    c::OperationInvocation invocation;
  };
  Request recipe_;
  c::Context context_;
  c::cuda::MemoryResource memory_;
  c::Queue queue_;
  c::PreparedOperation prepared_;
  std::vector<Slot> slots_;
  std::size_t next_ = 0;
};
int main(int argc, char **argv) {
  try {
    (void)argc;
    const bool verify = !environment("HYLE_BENCH_VERIFY_ONLY").empty();
    const auto aa = environment("HYLE_BENCH_AA_ROUTE");
    if (!aa.empty() && aa != "selected" && aa != "peer") throw std::runtime_error("invalid AA route");
    const auto base = std::filesystem::absolute(argv[0]).parent_path();
    if (!verify && !environment("HYLE_PERF_RUN_NONCE").empty() && aa.empty()) {
      std::string ready; std::ifstream(base / "matrix-aa.ready") >> ready;
      if (ready != "PASS") throw std::runtime_error("matrix has no passing workload AA receipt");
    }
    const auto affinity = hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
#if HYLE_OS_WINDOWS
    DWORD_PTR mask = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &mask, &system)) return 2;
    const auto priority = GetPriorityClass(GetCurrentProcess());
    std::fprintf(stderr, "# HYLE_CUDA_MATRIX_RUN execution_order=%s allocation_order=%s aa_route=%s paired_mode=nanobench-compare affinity_topology=%s caller_cpu=%s affinity_ok=%s parent_mask=0x%llx priority=%s run_nonce=%s verify_only=%d\n",
        environment("HYLE_BENCH_RIVAL_FIRST").empty() ? "candidate-first" : "peer-first",
        environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty() ? "candidate-first" : "peer-first",
        aa.empty() ? "off" : aa.c_str(), affinity.topology, affinity.cpu, affinity.ok,
        static_cast<unsigned long long>(mask), priority == ABOVE_NORMAL_PRIORITY_CLASS ? "above-normal" : "other",
        hyle::bench::perf::run_nonce(), verify ? 1 : 0);
#else
    (void)affinity;
#endif
    std::ifstream input(HYLE_CUDA_MATRIX_PAIRS);
    if (!input) throw std::runtime_error("frozen matrix input is unavailable");
    const bool first_allocation = !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream row(line);
      std::string id, left, right, op, source, output;
      int tokens = 0, hidden = 0;
      if (!(row >> id >> left >> right >> op >> source >> output >> tokens >> hidden))
        throw std::runtime_error("invalid frozen matrix row");
      const auto recipe = request(op, type(source), tokens, hidden, type(output));
      std::unique_ptr<Session> candidate, peer;
      if (first_allocation) peer = std::make_unique<Session>(recipe, aa == "selected" ? left : right);
      candidate = std::make_unique<Session>(recipe, aa == "peer" ? right : left);
      if (!first_allocation) peer = std::make_unique<Session>(recipe, aa == "selected" ? left : right);
      candidate->verify(); peer->verify();
      if (verify) { std::fprintf(stderr, "# MATRIX_VERIFY %s PASS\n", id.c_str()); continue; }
      // Established resident CUDA warmup; no candidate-only work is omitted.
      for (int i = 0; i < 2048; ++i) { candidate->run(); peer->run(); }
      nb::Bench bench; bench.title(id).relative(true).minEpochTime(std::chrono::milliseconds(20));
      auto selected = [&] { candidate->run(); };
      auto reference = [&] { peer->run(); };
      if (!environment("HYLE_BENCH_RIVAL_FIRST").empty())
        (void)bench.compare("Peer", reference, "Candidate", selected);
      else
        (void)bench.compare("Candidate", selected, "Peer", reference);
      candidate->verify(); peer->verify();
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what()); return 2;
  }
}
