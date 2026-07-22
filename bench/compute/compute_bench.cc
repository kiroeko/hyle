// compute bench (compute-06 #16): the CPU cost of SUBMITTING a kernel, launch vs replay.
//
// record/replay (RecordedLaunch) records a command buffer ONCE and re-submits it; a plain
// launch re-allocates + re-records + frees a command buffer EVERY call. For an iterative
// workload (same kernel, same buffers, thousands of iterations) the recorded path skips all
// that per-iteration CPU. This bench measures exactly that CPU: the host-side submit cost,
// NOT the GPU's execution time.
//
// Method: a trivially small saxpy (tiny N) so GPU execution is small, over a batch of 64
// submits per timed op, then ONE sync to drain. The comparison is HONEST: same kernel, same
// buffers, same dispatch; only "re-record each time (launch)" vs "reuse recording (replay)"
// differs. Measured ~1.12x for replay on an RTX 4070 (Win).
//
// HONEST CAVEAT (don't over-read the number): each op ends with a vkQueueWaitIdle sync whose
// cost (GPU drain) is FIXED across both rows, so it DILUTES the CPU-record difference replay
// actually saves -- the true per-submit CPU saving is larger than the 1.12x end-to-end ratio
// shows. A real iterative consumer that pipelines many replays before syncing (not sync-per-
// batch like here) sees more of it. We report the measured end-to-end ratio, not an inflated
// CPU-only microfigure, because that is the number a sync-bounded workload actually gets.
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

  // Tiny problem so GPU time is negligible and the CPU submit path dominates the measurement.
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
  b.title("compute submit CPU cost: launch (re-record) vs replay (reuse), 64 submits/op")
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
