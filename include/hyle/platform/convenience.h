#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/error/result.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/fs.h"

// Opt-in owning conveniences for platform control/query paths. The syscall
// layer in fs.h/cpu.h deliberately keeps caller-owned buffer-fill output so
// arena users can choose the returned storage.
// Implementations may still
// allocate checked transient scratch for unbounded OS paths/encoding; such OOM
// is reported through Status. This header is the everyday C++ shape: it returns
// standard-library owners, retries a changing snapshot, and reports allocation/
// size failures through the shared error model.

namespace hyle::platform {
namespace convenience_detail {

inline Status owning_allocation_error() noexcept {
  return Status::ResourceExhausted("platform owning query allocation failed");
}

inline Status owning_size_error() noexcept {
  return Status::OutOfRange("platform owning query result is too large");
}

inline Status changing_query_error() noexcept {
  return Status::Aborted("platform query changed repeatedly while copying");
}

template <class Query>
Result<std::string> query_string(Query query) noexcept {
  auto count = query(nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());

  // cwd, environment and symlink state can change between the count and fill.
  // Four retries bound adversarial process-global churn without returning a
  // silently truncated value.
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    if (*count == (std::numeric_limits<std::size_t>::max)()) {
      return std::unexpected(owning_size_error());
    }
    try {
      std::string value(*count, '\0');
      auto filled = query(value.data(), value.size() + 1);
      if (!filled) return std::unexpected(std::move(filled).error());
      if (*filled <= value.size()) {
        value.resize(*filled);
        return value;
      }
      count = *filled;
    } catch (const std::bad_alloc&) {
      return std::unexpected(owning_allocation_error());
    } catch (const std::length_error&) {
      return std::unexpected(owning_size_error());
    }
  }
  return std::unexpected(changing_query_error());
}

template <class T, class Query>
Result<std::vector<T>> query_vector(Query query) noexcept {
  auto count = query(nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());

  // CPU availability is a live process view. Retry if affinity/cpuset changed
  // after sizing rather than returning a partial placement list.
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
      return std::unexpected(owning_allocation_error());
    } catch (const std::length_error&) {
      return std::unexpected(owning_size_error());
    }
  }
  return std::unexpected(changing_query_error());
}

}  // namespace convenience_detail

// Owned UTF-8 snapshots. Overloads keep the natural core names; including this
// header adds the zero/one-argument forms while the buffer-fill forms remain
// available for caller-provided storage.
inline Result<std::string> current_directory() noexcept {
  return convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return current_directory(dst, cap); });
}

inline Result<std::string> temp_directory() noexcept {
  return convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return temp_directory(dst, cap); });
}

inline Result<std::string> home_directory() noexcept {
  return convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return home_directory(dst, cap); });
}

inline Result<std::string> get_env_var(const char* name) noexcept {
  return convenience_detail::query_string(
      [name](char* dst, std::size_t cap) {
        return get_env_var(name, dst, cap);
      });
}

inline Result<std::string> read_symlink(const char* path) noexcept {
  return convenience_detail::query_string(
      [path](char* dst, std::size_t cap) {
        return read_symlink(path, dst, cap);
      });
}

// Owned live CPU views for initialization/reconfiguration code. The ordered
// overload retains every available CPU and changes only its selection order.
inline Result<std::vector<unsigned>> available_cpu_ids(
    CpuOrder order = CpuOrder::kOsDefault) noexcept {
  return convenience_detail::query_vector<unsigned>(
      [order](unsigned* dst, std::size_t cap) {
        return available_cpu_ids(order, dst, cap);
      });
}

inline Result<std::vector<CpuInfo>> available_cpu_info() noexcept {
  return convenience_detail::query_vector<CpuInfo>(
      [](CpuInfo* dst, std::size_t cap) {
        return available_cpu_info(dst, cap);
      });
}

}  // namespace hyle::platform
