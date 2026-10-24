// Diagnostic only: original same-DAG math, original queues/completions, and
// calibrated GPU dispatch brackets. Does not replace the immutable P1 contract.
#define main hyle_g09_ordinary_vulkan_main
#include "native_joint_pipeline.cc"
#undef main
#include "vulkan_pipeline_trace.h"
#include <hyle/platform/dynlib.h>
#include <atomic>
#include <bit>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string_view>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif

namespace g09_vulkan {
struct Cpu { unsigned phase; std::uint64_t begin, end; };
struct Episode { g09_vk::Calibration before; g09_vk::Observation native; };
std::mutex monitor;
std::vector<Cpu> cpu;
std::vector<Episode> episodes;
std::atomic<unsigned> phase{0};
unsigned phase_base = 0;
g09_vk::DeviceInfo facts;
g09_vk::Begin begin;
g09_vk::End end;
Episode current;
std::uint64_t host_clock() noexcept {
#if defined(_WIN32)
  LARGE_INTEGER ticks{}; QueryPerformanceCounter(&ticks); return ticks.QuadPart;
#else
  timespec ticks{}; clock_gettime(CLOCK_MONOTONIC_RAW, &ticks);
  return std::uint64_t(ticks.tv_sec) * 1000000000 + ticks.tv_nsec;
#endif
}
void cpu_parallel(void *, std::size_t count, c::ParallelTaskFn task, void *context) noexcept {
  const auto observed = phase.load(std::memory_order_acquire);
  const auto start = observed ? host_clock() : 0;
  for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  if (observed) {
    const auto finish = host_clock(); std::lock_guard lock(monitor);
    cpu.push_back({observed, start, finish});
  }
}
Status observe(unsigned step, bool before) {
  if (before) {
    current = {};
    const auto id = step ? phase_base + step : 0;
    HYLE_RETURN_IF_ERROR(require(begin(id, &current.before), "original calibrated timestamps unavailable"));
    phase.store(id, std::memory_order_release);
  } else {
    phase.store(0, std::memory_order_release);
    HYLE_RETURN_IF_ERROR(require(end(&current.native), "completed original Neg queries unavailable"));
    if (step) episodes.push_back(current);
  }
  return Status::Ok();
}
long double host_ns(std::uint64_t ticks) { return static_cast<long double>(ticks) * 1e9L / facts.host_frequency; }
std::int64_t delta(std::uint64_t ticks, std::uint64_t reference) {
  auto difference = ticks - reference;
  if (facts.timestamp_bits == 64) return std::bit_cast<std::int64_t>(difference);
  const auto mask = (std::uint64_t{1} << facts.timestamp_bits) - 1;
  difference &= mask;
  const auto half = std::uint64_t{1} << (facts.timestamp_bits - 1);
  return difference < half ? std::int64_t(difference) : std::int64_t(difference) - std::int64_t(mask) - 1;
}
Status verdict() {
  std::array<unsigned, 2> inside{}, queries{};
  constexpr long double guard_ns = 1000;
  std::lock_guard lock(monitor);
  for (const auto &c : cpu)
    std::printf("{\"cpu_task\":true,\"phase\":%u,\"begin_ticks\":%llu,\"end_ticks\":%llu,\"begin_ns\":%.3Lf,\"end_ns\":%.3Lf}\n",
        c.phase, static_cast<unsigned long long>(c.begin), static_cast<unsigned long long>(c.end), host_ns(c.begin), host_ns(c.end));
  for (const auto &e : episodes) {
    const auto &a = e.before, &b = e.native.after;
    const auto drift = std::abs(host_ns(b.host_ticks) - host_ns(a.host_ticks) -
        static_cast<long double>(delta(b.device_ticks, a.device_ticks)) * facts.timestamp_period_ns);
    std::printf("{\"calibration\":true,\"before\":[%llu,%llu,%llu],\"after\":[%llu,%llu,%llu],\"offset_drift_ns\":%.3Lf,\"observed_dispatches\":%u}\n",
        static_cast<unsigned long long>(a.device_ticks), static_cast<unsigned long long>(a.host_ticks), static_cast<unsigned long long>(a.max_deviation_ns),
        static_cast<unsigned long long>(b.device_ticks), static_cast<unsigned long long>(b.host_ticks), static_cast<unsigned long long>(b.max_deviation_ns), drift, e.native.count);
    for (unsigned i = 0; i < e.native.count; ++i) {
      const auto &d = e.native.dispatches[i]; const auto order = d.phase / 100;
      HYLE_RETURN_IF_ERROR(require(order < 2 && delta(d.end, d.begin) > 0, "invalid original dispatch timestamp span"));
      ++queries[order];
      const auto uncertainty = static_cast<long double>(std::max(a.max_deviation_ns, b.max_deviation_ns)) + drift +
          facts.timestamp_period_ns + 1e9L / facts.host_frequency;
      const auto mapped = [&](std::uint64_t tick, const g09_vk::Calibration &c) {
        return host_ns(c.host_ticks) + static_cast<long double>(delta(tick, c.device_ticks)) * facts.timestamp_period_ns;
      };
      const auto lower = std::min(mapped(d.begin, a), mapped(d.begin, b)) - uncertainty;
      const auto upper = std::max(mapped(d.end, a), mapped(d.end, b)) + uncertainty;
      bool contained = false;
      for (const auto &c : cpu) if (c.phase == d.phase && host_ns(c.begin) + guard_ns < lower && upper + guard_ns < host_ns(c.end)) {
        contained = true; ++inside[order]; break;
      }
      std::printf("{\"original_neg_dispatch\":true,\"phase\":%u,\"query\":%u,\"ticks\":[%llu,%llu],\"mapped_envelope_ns\":[%.3Lf,%.3Lf],\"uncertainty_ns\":%.3Lf,\"whole_gpu_inside_cpu\":%s}\n",
          d.phase, d.query, static_cast<unsigned long long>(d.begin), static_cast<unsigned long long>(d.end), lower, upper, uncertainty, contained ? "true" : "false");
    }
  }
  std::printf("{\"single_dag\":true,\"warm_episodes_per_order\":8,\"timestamp_bits\":%u,\"host_domain\":%u,\"host_frequency\":%llu,\"period_ns\":%.9g,\"matched_neg_shader_modules\":%u,\"matched_neg_pipelines\":%u,\"observed_negative_dispatches\":[%u,%u],\"whole_gpu_inside_cpu_task\":[%u,%u]}\n",
      facts.timestamp_bits, facts.host_domain, static_cast<unsigned long long>(facts.host_frequency), facts.timestamp_period_ns,
      facts.matched_shader_modules, facts.matched_pipelines, queries[0], queries[1], inside[0], inside[1]);
  return require(queries[0] == 8 && queries[1] == 8 && inside[0] && inside[1] && facts.matched_shader_modules && facts.matched_pipelines,
      "both insertion orders need a calibrated original Neg interval inside the independent CPU Exp task");
}
Status exercise(const char *loader_path, const char *shader_path) {
  HYLE_ASSIGN_OR_RETURN(auto loader, hyle::platform::DynLib::open(loader_path));
  HYLE_ASSIGN_OR_RETURN(auto configure, loader.symbol_as<g09_vk::Configure>("g09_trace_configure"));
  HYLE_ASSIGN_OR_RETURN(auto ready, loader.symbol_as<g09_vk::Ready>("g09_trace_ready"));
  HYLE_ASSIGN_OR_RETURN(begin, loader.symbol_as<g09_vk::Begin>("g09_trace_begin"));
  HYLE_ASSIGN_OR_RETURN(end, loader.symbol_as<g09_vk::End>("g09_trace_end"));
  std::ifstream file(shader_path, std::ios::binary);
  std::vector<char> reference((std::istreambuf_iterator<char>(file)), {});
  const auto derived_path = std::string(shader_path) + ".actual.spv";
  HYLE_RETURN_IF_ERROR(require(!reference.empty() && configure(reference.data(), reference.size(), derived_path.c_str()), "missing exact compiled Neg shader reference"));
  cpu.reserve(128); episodes.reserve(16);
  HYLE_ASSIGN_OR_RETURN(auto host, g::cpu::create_context({.executor = c::ParallelExecutorRef{&phase, 1, &cpu_parallel}}));
  HYLE_ASSIGN_OR_RETURN(auto native, n::Context::create({.loader_path = loader_path}));
  HYLE_RETURN_IF_ERROR(require(ready(&facts), "original device lacks qualified timestamp support"));
  HYLE_ASSIGN_OR_RETURN(auto device, n::MemoryResource::create(native, n::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, n::MemoryResource::create(native, n::MemoryKind::kStaging));
  const PipelineResources environment{host,native,device,&staging};
  HYLE_RETURN_IF_ERROR(branches(environment, true, 2, true, 9, 1, &observe, true));
  phase_base = 100;
  HYLE_RETURN_IF_ERROR(branches(environment, false, 2, true, 9, 1, &observe, true));
  HYLE_RETURN_IF_ERROR(require(ready(&facts), "original native trace lost its bounded state"));
  return verdict();
}
} // namespace g09_vulkan

int main(int argc, char **argv) {
  if (argc == 1) return hyle_g09_ordinary_vulkan_main();
  if (argc != 4 || std::string_view(argv[1]) != "--trace") return 2;
  const auto status = g09_vulkan::exercise(argv[2], argv[3]);
  if (status.ok()) return 0;
  std::fprintf(stderr, "%.*s\n", int(status.message().size()), status.message().data()); return 1;
}
