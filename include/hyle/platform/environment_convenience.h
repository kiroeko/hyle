#pragma once

#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "hyle/platform/environment.h"

namespace hyle::platform {

inline Result<std::string> get_env_var(const char* name) noexcept {
  auto count = get_env_var(name, nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    if (*count == (std::numeric_limits<std::size_t>::max)()) {
      return std::unexpected(
          Status::OutOfRange("environment value is too large"));
    }
    try {
      std::string value(*count, '\0');
      auto filled = get_env_var(name, value.data(), value.size() + 1);
      if (!filled) return std::unexpected(std::move(filled).error());
      if (*filled <= value.size()) {
        value.resize(*filled);
        return value;
      }
      count = *filled;
    } catch (const std::bad_alloc&) {
      return std::unexpected(Status::ResourceExhausted(
          "environment value allocation failed"));
    } catch (const std::length_error&) {
      return std::unexpected(
          Status::OutOfRange("environment value is too large"));
    }
  }
  return std::unexpected(
      Status::Aborted("environment value changed repeatedly"));
}

}  // namespace hyle::platform
