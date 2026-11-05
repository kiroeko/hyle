#pragma once

// CPU encoding metadata only. Each actual native stream borrows a disjoint
// slice of the invocation workspace; same-stream FIFO permits reuse across
// sibling projections. A new encoding resets the bits, so a captured CUDA
// graph contains the quantizer and recomputes it on every replay.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hyle_shared_mmvq {
struct Arena {
  unsigned stream;
  void *data;
  std::size_t bytes;
};
struct Plan {
  const void *context;
  const void *input;
  std::size_t bytes;
  std::vector<const void *> nodes;
  std::vector<Arena> arenas;
  std::array<bool, 64> initialized{};
  std::atomic_flag encoding = ATOMIC_FLAG_INIT;
  bool poisoned = false;

  Plan(const void *ctx, const void *src, std::size_t size, std::vector<const void *> selected,
       std::vector<Arena> storage)
      : context(ctx), input(src), bytes(size), nodes(std::move(selected)),
        arenas(std::move(storage)) {
    if (!context || !input || !bytes || nodes.size() < 2 || arenas.empty())
      throw std::invalid_argument("shared MMVQ requires a complete cold plan");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (!nodes[i])
        throw std::invalid_argument("missing MMVQ node");
      for (std::size_t j = 0; j < i; ++j)
        if (nodes[i] == nodes[j])
          throw std::invalid_argument("duplicate MMVQ node");
    }
    for (std::size_t i = 0; i < arenas.size(); ++i) {
      const auto &a = arenas[i];
      const auto start = reinterpret_cast<std::uintptr_t>(a.data);
      if (a.stream >= 64 || !a.data || a.bytes < bytes || a.bytes > UINTPTR_MAX - start)
        throw std::invalid_argument("invalid stream workspace");
      for (std::size_t j = 0; j < i; ++j) {
        const auto &b = arenas[j];
        const auto other = reinterpret_cast<std::uintptr_t>(b.data);
        if (a.stream == b.stream || (start < other + b.bytes && other < start + a.bytes))
          throw std::invalid_argument("stream workspaces overlap");
      }
    }
  }
};
struct Selection {
  void *data = nullptr;
  bool quantize = false;
};
// Accessed only by the native MMVQ translation unit through its C bridge.
inline thread_local Plan *active = nullptr;

inline void begin(Plan *plan) {
  if (!plan)
    return;
  if (active || plan->encoding.test_and_set(std::memory_order_acquire))
    throw std::logic_error("shared MMVQ invocation already encoding");
  if (plan->poisoned) {
    plan->encoding.clear(std::memory_order_release);
    throw std::logic_error("shared MMVQ invocation is poisoned");
  }
  plan->initialized.fill(false);
  active = plan;
}
inline void end(Plan *plan) noexcept {
  if (!plan)
    return;
  if (active != plan) {
    plan->poisoned = true;
    plan->encoding.clear(std::memory_order_release);
    return;
  }
  active = nullptr;
  plan->encoding.clear(std::memory_order_release);
}
inline Selection select(const void *context, const void *node, const void *input, unsigned stream,
                        std::size_t bytes) {
  if (!active || active->context != context)
    return {};
  bool registered = false;
  for (const auto *item : active->nodes)
    registered |= item == node;
  if (!registered)
    return {};
  if (active->input != input || active->bytes != bytes || stream >= 64)
    throw std::logic_error("shared MMVQ binding changed after cold preparation");
  for (const auto &arena : active->arenas)
    if (arena.stream == stream) {
      const bool first = !active->initialized[stream];
      active->initialized[stream] = true;
      return {arena.data, first};
    }
  // A native stream absent from the cold plan uses the original unshared path.
  return {};
}
struct Scope {
  explicit Scope(Plan *value) : plan(value) { begin(plan); }
  ~Scope() { end(plan); }
  void check() const {
    if (plan && (active != plan || plan->poisoned))
      throw std::logic_error("shared MMVQ encoder contract failed");
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  Plan *plan;
};
} // namespace hyle_shared_mmvq
