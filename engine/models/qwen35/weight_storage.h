#pragma once

#include "weights.h"
#include "precision_profile.h"
#include "../../inference/backend.h"

#include <chrono>
#include <iostream>
#include <memory>

namespace qwen35 {

// Original physical assets only. Each Model keeps its own mutable cold root,
// programs and sessions. The endpoint must outlive the original pools/leases.
struct WeightStorage final {
  std::optional<hyle::inference::Backend> endpoint;
  std::vector<hyle::compute::LeasePool> pools;
  std::map<std::string, LoadedWeight> weights;
  WeightMeanings meanings;
  std::string identity;
  std::optional<WeightLoadReport> inventory;
  std::optional<prepared::Roles> source_roles;
  std::size_t cpu_ffn_layers = 0;
};

inline bool same_weight_reader(const prepared::Roles &a, const prepared::Roles &b) {
  return a.identity == b.identity && a.geometry == b.geometry && a.original == b.original &&
      a.representation == b.representation && a.cache_policy == b.cache_policy &&
      a.float8_policy.enabled == b.float8_policy.enabled && a.ct_mixed_policy.kind == b.ct_mixed_policy.kind &&
      a.nf4_policy.enabled == b.nf4_policy.enabled && a.ct_asymmetric_policy.enabled == b.ct_asymmetric_policy.enabled &&
      a.mlx_policy.bits == b.mlx_policy.bits && a.static_fp8_policy.enabled == b.static_fp8_policy.enabled &&
      PrecisionProfile::from_roles(a) == PrecisionProfile::from_roles(b);
}

// Analyze all selected physical roles before loading any profile. A matching
// source ID alone does not waive role codec, scale, placement or reader checks.
inline prepared::Roles shared_weight_roles(std::span<const PreparedWeightSource *const> sources) {
  if (sources.empty() || !sources.front())
    throw std::invalid_argument("shared weight preparation requires source roles");
  auto roles = sources.front()->roles;
  std::map<std::string, std::size_t> selected;
  for (std::size_t i = 0; i < roles.selected.size(); ++i)
    if (!selected.emplace(roles.selected[i].name, i).second)
      throw std::invalid_argument("shared weight source contains duplicate physical roles");
  for (const auto *source : sources.subspan(1)) {
    if (!source || !same_weight_reader(roles, source->roles))
      throw std::invalid_argument("execution profiles require different weight sources or reader contracts");
    for (const auto &role : source->roles.selected) {
      const auto [found, inserted] = selected.emplace(role.name, roles.selected.size());
      if (inserted) {
        roles.selected.push_back(role);
        roles.bytes = cache::add(roles.bytes, role.bytes);
      } else if (roles.selected[found->second] != role)
        throw std::invalid_argument("shared physical role has conflicting encoding, scale or source: " + role.name);
    }
    for (const auto &role : source->roles.cache_calibration) {
      const auto found = std::ranges::find(roles.cache_calibration, role.name, &prepared::Role::name);
      if (found == roles.cache_calibration.end()) roles.cache_calibration.push_back(role);
      else if (*found != role)
        throw std::invalid_argument("shared source has conflicting cache calibration: " + role.name);
    }
  }
  return roles;
}

inline std::shared_ptr<WeightStorage> prepare_shared_weights(hyle::inference::Backend &endpoint,
    prepared::Roles roles, std::size_t cpu_ffn_layers = 0,
    const hyle::compute::ResourceBudget &budget = {}) {
  auto bank = std::make_shared<WeightStorage>();
  bank->endpoint.emplace(take(endpoint.share_endpoint()));
  bank->cpu_ffn_layers = cpu_ffn_layers;
  const auto started = std::chrono::steady_clock::now();
  auto loading = load_selected_weight_roles(roles, *bank->endpoint, bank->weights, bank->pools,
      FfnPlacement(roles, cpu_ffn_layers), budget);
  std::cerr << "shared_weight_assets source=" << loading.identity << " physical_bytes=" << loading.bytes
      << " load_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
      << " io_calls=" << loading.stream.io_calls << " copied_physical_bytes=" << loading.stream.copied_physical_bytes << '\n';
  bank->identity = loading.identity;
  bank->meanings = loading.meanings;
  bank->inventory = std::move(loading);
  bank->source_roles = std::move(roles);
  return bank;
}

inline void validate_shared_weights(const WeightStorage &bank, const hyle::inference::Backend &endpoint,
    const prepared::Roles &roles, std::size_t cpu_ffn_layers) {
  if (!bank.endpoint || !bank.inventory || !bank.source_roles ||
      !endpoint.same_endpoint(*bank.endpoint) || !same_weight_reader(*bank.source_roles, roles) ||
      bank.identity != roles.identity || bank.cpu_ffn_layers != cpu_ffn_layers)
    throw std::invalid_argument("shared weight bank differs from its source, reader, placement or endpoint");
  for (const auto &role : roles.selected) {
    const auto original = std::ranges::find(bank.source_roles->selected, role.name, &prepared::Role::name);
    const auto loaded = bank.weights.find(role.name);
    if (original == bank.source_roles->selected.end() || *original != role || loaded == bank.weights.end() ||
        !loaded->second.tensor.ready || loaded->second.tensor.writer)
      throw std::invalid_argument("shared weight bank lacks an immutable matching role: " + role.name);
  }
  for (const auto &role : roles.cache_calibration) {
    const auto original = std::ranges::find(bank.source_roles->cache_calibration, role.name, &prepared::Role::name);
    if (original == bank.source_roles->cache_calibration.end() || *original != role)
      throw std::invalid_argument("shared source lacks matching cache calibration: " + role.name);
  }
}

} // namespace qwen35
