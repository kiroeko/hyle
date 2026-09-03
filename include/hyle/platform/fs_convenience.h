#pragma once

#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "hyle/platform/fs.h"

namespace hyle::platform {
namespace fs_convenience_detail {

template <class Query>
Result<std::string> query_string(Query query) noexcept {
  auto count = query(nullptr, 0);
  if (!count) return std::unexpected(std::move(count).error());
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    if (*count == (std::numeric_limits<std::size_t>::max)()) {
      return std::unexpected(Status::OutOfRange("filesystem path is too large"));
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
      return std::unexpected(Status::ResourceExhausted(
          "filesystem path allocation failed"));
    } catch (const std::length_error&) {
      return std::unexpected(Status::OutOfRange(
          "filesystem path is too large"));
    }
  }
  return std::unexpected(
      Status::Aborted("filesystem query changed repeatedly"));
}

}  // namespace fs_convenience_detail

inline Result<std::string> current_directory() noexcept {
  return fs_convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return current_directory(dst, cap); });
}

inline Result<std::string> temp_directory() noexcept {
  return fs_convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return temp_directory(dst, cap); });
}

inline Result<std::string> home_directory() noexcept {
  return fs_convenience_detail::query_string(
      [](char* dst, std::size_t cap) { return home_directory(dst, cap); });
}

inline Result<std::string> read_symlink(const char* path) noexcept {
  return fs_convenience_detail::query_string(
      [path](char* dst, std::size_t cap) {
        return read_symlink(path, dst, cap);
      });
}

}  // namespace hyle::platform
