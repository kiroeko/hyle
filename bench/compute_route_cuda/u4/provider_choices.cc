#include <nanobench.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/core/environment.h"
#include "../../common/perf_process.h"

namespace c = hyle::compute;
namespace m = hyle::mem;
namespace nb = ankerl::nanobench;

template <class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
std::string environment(const char *name) {
  const auto *value = std::getenv(name);
  return value == nullptr ? "" : value;
}
std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}
float decoded(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}
struct Recipe {
  const char *title;
  std::int64_t tokens, hidden;
  bool half, gate, weighted;
  c::Provider mature;
};
class Route {
public:
  Route(const Recipe &recipe, bool mature)
      : recipe_(recipe), context_(take(c::Context::create(
            take(c::cuda::device(0)), mature ? recipe.mature : c::cuda_native::provider()))),
        memory_(take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice))),
        queue_(take(context_.create_queue())) {
    const auto type = recipe.half ? c::DataType::kBFloat16 : c::DataType::kFloat32;
    const auto width = recipe.hidden * (recipe.gate ? 2 : 1);
    const auto input = take(c::TensorDesc::dense(type, std::array{recipe.tokens, width}));
    const auto output = take(c::TensorDesc::dense(type, std::array{recipe.tokens, recipe.hidden}));
    std::vector<c::TensorDesc> descriptions{input};
    if (recipe.weighted)
      descriptions.push_back(take(c::TensorDesc::dense(type, std::array{recipe.hidden})));
    if (recipe.weighted) {
      weight_ = take(m::Buffer::create(memory_.ref(), static_cast<std::size_t>(recipe.hidden) * 2, 256));
      std::vector<std::uint16_t> weights(static_cast<std::size_t>(recipe.hidden), bf16(1.0F));
      require(weight_.view().write(std::as_bytes(std::span(weights))));
    }
    const c::Operation operation = recipe.gate
        ? c::Operation(c::GatedActivation{.activation = c::GatedActivationKind::kSilu,
                                         .inputs = c::GatedInputKind::kPackedHalves})
        : c::Operation(c::RmsNorm{.epsilon = 1.0e-6F, .accumulation_type = c::DataType::kFloat32});
    prepared_ = take(c::prepare(context_, operation, descriptions, std::span(&output, 1), memory_.ref()));
    const auto facts = take(c::cuda::environment(context_));
    std::fprintf(stderr, "# U4_CUDA_ENV device=%s sm=%d%d driver=%s driver_api=%d runtime=%d toolkit=%d provider=%.*s version=%.*s\n",
        facts.device_name.data(), facts.sm_major, facts.sm_minor, facts.driver_release.data(),
        facts.driver_cuda_version, facts.runtime_version, facts.build_toolkit_version,
        static_cast<int>(prepared_.provider().name.size()), prepared_.provider().name.data(),
        static_cast<int>(prepared_.provider().version.size()), prepared_.provider().version.data());
    for (std::size_t i = 0; i < 2; ++i) {
      const auto value = i == 0 ? 0.25F : -0.5F;
      Slot slot;
      slot.input = take(m::Buffer::create(memory_.ref(), take(input.required_bytes()), 256));
      slot.output = take(m::Buffer::create(memory_.ref(), take(output.required_bytes()), 256));
      if (recipe.half) {
        std::vector<std::uint16_t> values(take(input.element_count()), bf16(value));
        require(slot.input.view().write(std::as_bytes(std::span(values))));
      } else {
        std::vector<float> values(take(input.element_count()), value);
        require(slot.input.view().write(std::as_bytes(std::span(values))));
      }
      slot.inputs.push_back(take(c::ConstTensorView::create(input, slot.input.view())));
      if (recipe.weighted)
        slot.inputs.push_back(take(c::ConstTensorView::create(descriptions[1], weight_.view())));
      slot.outputs.push_back(take(c::TensorView::create(output, slot.output.view())));
      slot.invocation = take(prepared_.create_invocation());
      const auto exact = recipe.gate ? value * value / (1.0F + std::exp(-value))
                                    : value / std::sqrt(value * value + 1.0e-6F);
      slot.expected = recipe.half ? decoded(bf16(exact)) : exact;
      slots_.push_back(std::move(slot));
    }
  }
  void run() {
    auto &slot = slots_[next_++ % slots_.size()];
    auto completion = take(slot.invocation.submit(queue_, slot.inputs, slot.outputs));
    require(completion.wait());
    // Completion destruction and retained run-reference settlement are charged.
  }
  void verify() {
    for (auto &slot : slots_) {
      {
        auto completion = take(slot.invocation.submit(queue_, slot.inputs, slot.outputs));
        require(completion.wait());
      }
      const auto count = static_cast<std::size_t>(recipe_.tokens * recipe_.hidden);
      if (recipe_.half) {
        std::vector<std::uint16_t> values(count);
        require(slot.output.view().read(std::as_writable_bytes(std::span(values))));
        for (auto value : values)
          if (std::abs(decoded(value) - slot.expected) > 0.0078125F)
            throw std::runtime_error("BF16 complete output oracle failed");
      } else {
        std::vector<float> values(count);
        require(slot.output.view().read(std::as_writable_bytes(std::span(values))));
        for (auto value : values)
          if (std::abs(value - slot.expected) > 2.0e-5F)
            throw std::runtime_error("F32 complete output oracle failed");
      }
    }
  }
private:
  struct Slot {
    m::Buffer input, output;
    std::vector<c::ConstTensorView> inputs;
    std::vector<c::TensorView> outputs;
    c::OperationInvocation invocation;
    float expected = 0;
  };
  Recipe recipe_;
  c::Context context_;
  c::cuda::MemoryResource memory_;
  c::Queue queue_;
  m::Buffer weight_;
  c::PreparedOperation prepared_;
  std::vector<Slot> slots_;
  std::size_t next_ = 0;
};
void measure(const Recipe &recipe, bool verify_only) {
  const auto start = std::chrono::steady_clock::now();
  const auto aa = environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "selected" && aa != "peer") throw std::runtime_error("invalid A/A route");
  std::unique_ptr<Route> candidate, peer;
  const bool allocation_first = !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
  if (allocation_first) peer = std::make_unique<Route>(recipe, aa == "selected");
  candidate = std::make_unique<Route>(recipe, aa != "peer");
  if (!allocation_first) peer = std::make_unique<Route>(recipe, aa == "selected");
  candidate->verify(); peer->verify();
  std::fprintf(stderr, "# U4_SETUP title=%s seconds=%.6f\n", recipe.title,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  if (verify_only) return;
  nb::Bench bench;
  bench.title(recipe.title).relative(true).minEpochTime(std::chrono::milliseconds(8));
  auto selected = [&] { candidate->run(); };
  auto reference = [&] { peer->run(); };
  if (!environment("HYLE_BENCH_RIVAL_FIRST").empty())
    (void)bench.compare("HYLE native", reference, "Mature provider", selected);
  else
    (void)bench.compare("Mature provider", selected, "HYLE native", reference);
  candidate->verify(); peer->verify();
}
int main() {
  try {
    const auto affinity = hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
#if HYLE_OS_WINDOWS
    DWORD_PTR parent = 0, system = 0;
    if (!::GetProcessAffinityMask(::GetCurrentProcess(), &parent, &system)) return 2;
    const auto priority = ::GetPriorityClass(::GetCurrentProcess());
#else
    const unsigned long long parent = 0;
    const unsigned long priority = 0;
#endif
    const bool verify = !environment("HYLE_BENCH_VERIFY_ONLY").empty();
    std::fprintf(stderr, "# HYLE_U4_CUDA_RUN execution_order=%s allocation_order=%s aa_route=%s paired_mode=nanobench-compare affinity_topology=%s caller_cpu=%s affinity_ok=%s parent_mask=0x%llx priority=%s run_nonce=%s verify_only=%d\n",
        environment("HYLE_BENCH_RIVAL_FIRST").empty() ? "candidate-first" : "peer-first",
        environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty() ? "candidate-first" : "peer-first",
        environment("HYLE_BENCH_AA_ROUTE").empty() ? "off" : environment("HYLE_BENCH_AA_ROUTE").c_str(),
        affinity.topology, affinity.cpu, affinity.ok, static_cast<unsigned long long>(parent),
        priority == 32768 ? "above-normal" : "other", hyle::bench::perf::run_nonce(), verify ? 1 : 0);
    const std::array recipes{
        Recipe{"Qwen BF16 RMSNorm T1 H2048", 1, 2048, true, false, true, c::cuda_native::sglang_provider()},
        Recipe{"Qwen BF16 RMSNorm T32 H2048", 32, 2048, true, false, true, c::cuda_native::sglang_provider()},
        Recipe{"Qwen BF16 packed SiLU T1 H6144", 1, 6144, true, true, false, c::cuda_native::sglang_provider()},
        Recipe{"Qwen BF16 packed SiLU T32 H6144", 32, 6144, true, true, false, c::cuda_native::sglang_provider()},
        Recipe{"Qwen F32 RMSNorm T1 H2048", 1, 2048, false, false, false, c::ggml_cuda::provider()},
        Recipe{"Qwen F32 RMSNorm T32 H2048", 32, 2048, false, false, false, c::ggml_cuda::provider()}};
    for (const auto &recipe : recipes) measure(recipe, verify);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 2;
  }
}
