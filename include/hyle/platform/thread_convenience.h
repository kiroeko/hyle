#pragma once

#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/platform/thread.h"

namespace hyle::platform::this_thread {

// Owned affinity snapshot for save/apply/restore control paths. The buffer-fill
// overload in thread.h remains the allocation-free mechanism.
inline Result<std::vector<unsigned>> affinity() noexcept {
  auto count = affinity(nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    try {
      std::vector<unsigned> cpus(*count);
      auto filled = affinity(cpus.data(), cpus.size());
      if (!filled) return std::unexpected(std::move(filled).error());
      if (*filled <= cpus.size()) {
        cpus.resize(*filled);
        return cpus;
      }
      count = *filled;
    } catch (const std::bad_alloc&) {
      return std::unexpected(Status::ResourceExhausted(
          "thread affinity snapshot allocation failed"));
    } catch (const std::length_error&) {
      return std::unexpected(
          Status::OutOfRange("thread affinity snapshot is too large"));
    }
  }
  return std::unexpected(
      Status::Aborted("thread affinity changed repeatedly"));
}

}  // namespace hyle::platform::this_thread
