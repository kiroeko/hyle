// compute bench (compute-06 #16): sync-bounded iterative throughput, launch vs replay.
//
// record/replay (RecordedLaunch) records a command buffer ONCE and re-submits it; a plain
// launch re-allocates + re-records + frees a command buffer EVERY call. For an iterative
// workload (same kernel, same buffers, thousands of iterations) the recorded path skips all
// that per-iteration CPU work.
//
// Method: a trivially small saxpy (tiny N) so GPU execution is small, over a batch of 64
// submits per timed op, then ONE sync to drain. The comparison keeps kernel, buffers,
// dispatch, submit count and synchronization cadence equal; only re-record vs reuse differs.
//
// INTERPRETATION LIMIT: nanobench reports wall-clock ns/submit after dividing one 64-submit
// epoch. That number includes host recording/submission, GPU execution, queueing and one
// sync amortized over 64 submits. It is not a pure CPU-submit measurement and this single
// tiny-saxpy workload is not an architecture certificate. Device execution timing belongs
// to Event::create_timing/elapsed_ns and should be reported separately by consumer benches.
//
// Skips cleanly if no Vulkan device (record/replay is Vulkan-only; CUDA/host have no separate
// record cost to amortize -> their RecordedLaunch::record returns Unimplemented by design).

#include <nanobench.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/stream.h"
#include "saxpy.spv.h"  // hyle_test_shaders::kSaxpySpirv (shared with the test, embedded)

using namespace hyle;
using namespace hyle::compute;

namespace {
namespace nb = ankerl::nanobench;

struct SaxpyParams { std::uint32_t n; float alpha; float beta; float bias; };

bool find_vulkan(Device& out) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n; ++i)
    if (std::strcmp(devs[i].backend(), "vulkan") == 0) { out = devs[i]; return true; }
  return false;
}
}  // namespace

int main() {
  Device vk;
  if (!find_vulkan(vk)) {
    // No Vulkan device: nothing to compare (record/replay is Vulkan-only). Report + exit 0.
    nb::Bench().title("compute record/replay -- SKIPPED (no Vulkan device)").run("skip", [] {});
    return 0;
  }
  auto s = Stream::create(vk);
  if (!s.has_value()) return 0;

  // Load saxpy from the same embedded SPIR-V the test uses.
  auto k = Kernel::load_spirv(vk, hyle_test_shaders::kSaxpySpirv, "main");
  if (!k.has_value()) return 0;

  // Tiny fixed workload: useful for exposing command-record overhead, but still measured
  // end-to-end with queue drain; do not label the result as pure CPU time.
  constexpr int kN = 256;
  const std::size_t bytes = kN * sizeof(float);
  BufferType bt = device_buffer_type(vk);
  auto bx = Buffer::create(bt, bytes);
  auto by = Buffer::create(bt, bytes);
  auto bo = Buffer::create(bt, bytes);
  if (!bx.has_value() || !by.has_value() || !bo.has_value()) return 0;
  std::vector<float> hx(kN, 1.0f), hy(kN, 2.0f);
  (void)bx->set(0, hx.data(), bytes);
  (void)by->set(0, hy.data(), bytes);

  SaxpyParams p{kN, 2.0f, 3.0f, 0.5f};
  LaunchConfig cfg;
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  void* bufs[3] = {bx->native_handle(), by->native_handle(), bo->native_handle()};
  ScalarField field{0, sizeof(p)};
  KernelArgs ka{bufs, 3, &p, sizeof(p), &field, 1};

  // Batch of submits per timed op, so the per-submit CPU cost isn't lost in fixed overhead.
  constexpr int kSubmits = 64;

  nb::Bench b;
  b.title("compute iterative throughput: launch vs replay, 64 submits + sync/op")
      .relative(true)
      .batch(kSubmits)
      .minEpochIterations(50);  // submit path + a sync/epoch is slow-ish; 50 keeps err% low

  // Baseline: plain launch re-records a fresh command buffer every call.
  b.run("launch x64 (re-record each)", [&] {
    for (int i = 0; i < kSubmits; ++i) (void)launch(*s, *k, cfg, ka);
    (void)s->sync();  // drain so in-flight command buffers are reclaimed each epoch
  });

  // Recorded: record ONCE outside the timed loop, replay (reuse the cb) inside.
  auto rl = RecordedLaunch::record(*s, *k, cfg, ka);
  if (rl.has_value()) {
    b.run("replay x64 (reuse recording)", [&] {
      for (int i = 0; i < kSubmits; ++i) (void)rl->replay(*s);
      (void)s->sync();
    });
  }
  return 0;
}
