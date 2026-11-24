#pragma once

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../inference/cpu_ffn_layers.h"
#include "prepared_roles.h"

namespace qwen35 {
// One cold requested placement, shared by weight loading and graph assembly.
// BackendSelection still owns actual endpoint availability and resource rules.
// Roles have already been selected and authenticated by the source reader; this
// only binds their placement, including every codec component of a linear group.
class FfnPlacement final {
 public:
  static void validate(const hyle::inference::CpuFfnLayers &layers, std::size_t target_layers) {
    if (!layers.fits(target_layers))
      throw std::runtime_error("--cpu-ffn-layers exceeds the source target layer count");
  }
  FfnPlacement(const prepared::Roles &roles, const hyle::inference::CpuFfnLayers &layers)
      : target_layers_(roles.geometry.layers.size()), layers_(layers) {
    validate(layers_, target_layers_);
    if (layers_.empty()) return;
    const auto inventory = prepared::parameter_inventory(roles.geometry.mtp_layers != 0, roles.geometry);
    for (const auto &[name, parameter] : inventory) {
      if (!parameter.target_ffn_layer || !layers_.contains(*parameter.target_ffn_layer)) continue;
      // Linear physical components share the principal's group root. The
      // reader owns codec suffix/shape validation; placement has no codec list.
      const auto group = std::string_view(name).substr(0, name.size() - std::string_view("weight").size());
      for (const auto &role : roles.selected)
        if (role.name == name || (parameter.linear && role.name.starts_with(group)))
          host_roles_.insert(role.name);
    }
  }
  std::size_t target_layers() const noexcept { return target_layers_; }
  std::size_t cpu_layers() const noexcept { return layers_.count(); }
  bool ffn_on_cpu(std::size_t layer) const noexcept { return layers_.contains(layer); }
  const hyle::inference::CpuFfnLayers &layers() const noexcept { return layers_; }
  bool role_on_cpu(std::string_view name) const { return host_roles_.contains(name); }

 private:
  const std::size_t target_layers_;
  const hyle::inference::CpuFfnLayers layers_;
  std::set<std::string, std::less<>> host_roles_;
};
} // namespace qwen35
