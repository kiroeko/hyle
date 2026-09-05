#pragma once

#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/platform/cpu.h"

namespace hyle::platform {
namespace cpu_convenience_detail {

template <class T, class Query>
Result<std::vector<T>> query_vector(Query query) noexcept {
  auto count = query(nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    try {
      std::vector<T> values(*count);
      auto filled = query(values.data(), values.size());
      if (!filled) return std::unexpected(std::move(filled).error());
      if (*filled <= values.size()) {
        values.resize(*filled);
        return values;
      }
      count = *filled;
    } catch (const std::bad_alloc&) {
      return std::unexpected(
          Status::ResourceExhausted("CPU snapshot allocation failed"));
    } catch (const std::length_error&) {
      return std::unexpected(Status::OutOfRange("CPU snapshot is too large"));
    }
  }
  return std::unexpected(
      Status::Aborted("available CPU set changed repeatedly"));
}

}  // namespace cpu_convenience_detail

inline Result<std::vector<unsigned>> available_cpu_ids(
    CpuOrder order = CpuOrder::kOsDefault) noexcept {
  return cpu_convenience_detail::query_vector<unsigned>(
      [order](unsigned* dst, std::size_t cap) {
        return available_cpu_ids(order, dst, cap);
      });
}

inline Result<std::vector<CpuInfo>> available_cpu_info() noexcept {
  return cpu_convenience_detail::query_vector<CpuInfo>(
      [](CpuInfo* dst, std::size_t cap) {
        return available_cpu_info(dst, cap);
      });
}

inline Result<std::vector<CpuCacheClassInfo>>
available_cpu_cache_classes() noexcept {
  return cpu_convenience_detail::query_vector<CpuCacheClassInfo>(
      [](CpuCacheClassInfo* dst, std::size_t cap) {
        return available_cpu_cache_classes(dst, cap);
      });
}

inline Result<std::vector<CpuFrequencyInfo>>
available_cpu_frequencies() noexcept {
  return cpu_convenience_detail::query_vector<CpuFrequencyInfo>(
      [](CpuFrequencyInfo* dst, std::size_t cap) {
        return available_cpu_frequencies(dst, cap);
      });
}

}  // namespace hyle::platform
