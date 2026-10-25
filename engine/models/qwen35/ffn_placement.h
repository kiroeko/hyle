#pragma once

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include "prepared_roles.h"

namespace qwen35 {
// One cold requested placement, shared by weight loading and graph assembly.
// BackendSelection still owns actual endpoint availability and resource rules.
// Roles have already been selected and authenticated by the source reader; this
// only binds their placement, including every codec component of a linear group.
class FfnPlacement final {
 public:
  static void validate(std::size_t cpu_layers, std::size_t target_layers) {
    if (cpu_layers > target_layers)
      throw std::runtime_error("--cpu-ffn-layers exceeds the source target layer count");
  }
  FfnPlacement(const prepared::Roles &roles, std::size_t cpu_layers)
      : target_layers_(roles.geometry.layers.size()), cpu_layers_(cpu_layers) {
    validate(cpu_layers_, target_layers_);
    if (!cpu_layers_) return;
    const auto inventory = prepared::parameter_inventory(roles.geometry.mtp_layers != 0, roles.geometry);
    for (const auto &[name, parameter] : inventory) {
      if (!parameter.target_ffn_layer || !ffn_on_cpu(*parameter.target_ffn_layer)) continue;
      // Linear physical components share the principal's group root. The
      // reader owns codec suffix/shape validation; placement has no codec list.
      const auto group = std::string_view(name).substr(0, name.size() - std::string_view("weight").size());
      for (const auto &role : roles.selected)
        if (role.name == name || (parameter.linear && role.name.starts_with(group)))
          host_roles_.insert(role.name);
    }
  }
  std::size_t target_layers() const noexcept { return target_layers_; }
  std::size_t cpu_layers() const noexcept { return cpu_layers_; }
  bool ffn_on_cpu(std::size_t layer) const noexcept { return layer < cpu_layers_; }
  bool role_on_cpu(std::string_view name) const { return host_roles_.contains(name); }

 private:
  const std::size_t target_layers_, cpu_layers_;
  std::set<std::string, std::less<>> host_roles_;
};
} // namespace qwen35
