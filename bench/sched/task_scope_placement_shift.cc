#include <cstdint>

#if !defined(_MSC_VER)
#error "The Sched v8 placement control is currently Windows/MSVC-only"
#endif

namespace {

volatile std::uint64_t g_placement_seed = 0x9e3779b97f4a7c15ULL;

} // namespace

// COFF merges .text$* contributions into .text and sorts them by the suffix.
// This small, retained, setup-only function shifts later code addresses without
// adding a hot-patch prefix to every function or changing the timed route.
#pragma code_seg(push, ".text$AAA")
extern "C" __declspec(noinline) void
hyle_sched_v8_placement_shift() noexcept {
  std::uint64_t value = g_placement_seed;
  for (unsigned round = 0; round < 17; ++round) {
    value ^= value >> 7;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value << 11;
  }
  g_placement_seed = value;
}
#pragma code_seg(pop)
