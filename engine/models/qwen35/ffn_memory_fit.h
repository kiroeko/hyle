#pragma once

#include <optional>

#include "weights.h"

namespace qwen35 {

struct PlacementBytes final {
  std::size_t device_weights = 0, host_weights = 0;
};

inline PlacementBytes placement_weight_bytes(const prepared::Roles &roles,
    const hyle::inference::CpuFfnLayers &layers) {
  const FfnPlacement placement(roles, layers);
  PlacementBytes result;
  for (const auto &role : roles.selected) {
    auto &bytes = placement.role_on_cpu(role.name) ? result.host_weights : result.device_weights;
    bytes = cache::add(bytes, streamed_weight_capacity(role.bytes));
  }
  return result;
}

inline hyle::inference::CpuFfnLayers fit_cpu_ffn_layers(const prepared::Roles &roles,
    const std::optional<hyle::inference::CpuFfnLayers> &requested,
    std::size_t device_runtime_bytes, std::size_t device_limit, std::size_t minimum_layers = 0) {
  const auto required = [&](const hyle::inference::CpuFfnLayers &layers) {
    return cache::add(placement_weight_bytes(roles, layers).device_weights, device_runtime_bytes);
  };
  const auto insufficient = [&](std::size_t bytes, const char *placement) {
    throw std::runtime_error("--device-budget-mib allows " + std::to_string(device_limit) +
        " managed device bytes; " + placement + " requires " + std::to_string(bytes) +
        " planned bytes with the requested execution capacity");
  };
  if (requested) {
    const auto bytes = required(*requested);
    if (bytes > device_limit) insufficient(bytes, "explicit FFN placement");
    return *requested;
  }
  if (required(minimum_layers) <= device_limit) return minimum_layers;
  std::size_t first = minimum_layers + 1, last = roles.geometry.layers.size();
  const auto minimum = required(last);
  if (minimum > device_limit) insufficient(minimum, "placing all target FFNs on CPU");
  while (first < last) {
    const auto middle = first + (last - first) / 2;
    if (required(middle) <= device_limit) last = middle;
    else first = middle + 1;
  }
  return first;
}

} // namespace qwen35
