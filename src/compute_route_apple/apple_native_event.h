#pragma once

#include <span>
#include <mlx/array.h>
#include <mlx/backend/metal/event.h>

namespace hyle::compute::apple::detail {

// Ready public inputs were checked at bind. Pending raw ports carry the
// original native producer's event. Its MLX allocation stream is not ggml's
// queue: explicitly encode this bridge instead of MLX's same-stream shortcut.
inline void encode_native_input_waits(std::span<const mlx::core::array> inputs,
                                      mlx::core::Stream stream) {
  namespace mx = mlx::core;
  const void *last = nullptr;
  std::uint64_t last_value = 0;
  for (const auto &input : inputs) {
    auto &event = input.event();
    if (!event.valid() || input.is_available()) continue;
    const auto *native = event.cast<mx::metal::EventImpl>().mtl_event();
    if (native == last && event.value() == last_value) continue;
    event.wait(stream);
    last = native; last_value = event.value();
  }
}

} // namespace hyle::compute::apple::detail
