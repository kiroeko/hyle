#include "hyle/platform/environment.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace hyle::platform {
namespace {

bool valid_env_name(const char* name) noexcept {
  return name != nullptr && name[0] != '\0' &&
         std::strchr(name, '=') == nullptr;
}

std::size_t fill(char* dst, std::size_t cap, const char* src,
                 std::size_t size) noexcept {
  if (dst == nullptr) cap = 0;
  if (cap == 0) return size;
  const std::size_t copied = size < cap - 1 ? size : cap - 1;
  if (copied != 0) std::memcpy(dst, src, copied);
  dst[copied] = '\0';
  return size;
}

void clear_output(char* dst, std::size_t cap) noexcept {
  if (dst != nullptr && cap != 0) dst[0] = '\0';
}

#if HYLE_OS_WINDOWS

struct WideText {
  wchar_t stack[256];
  wchar_t* data = stack;
  bool heap = false;

  ~WideText() {
    if (heap) std::free(data);
  }
};

Status to_wide(const char* text, WideText& output,
               const char* operation) noexcept {
  const int required = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                              text, -1, nullptr, 0);
  if (required <= 0) {
    return status_from_win32(::GetLastError(), operation);
  }
  if (static_cast<std::size_t>(required) > std::size(output.stack)) {
    if (static_cast<std::size_t>(required) >
        (std::numeric_limits<std::size_t>::max)() / sizeof(wchar_t)) {
      return Status::OutOfRange("environment UTF-16 value is too large");
    }
    output.data = static_cast<wchar_t*>(
        std::malloc(static_cast<std::size_t>(required) * sizeof(wchar_t)));
    if (output.data == nullptr) {
      return Status::ResourceExhausted(
          "environment UTF-16 conversion allocation failed");
    }
    output.heap = true;
  }
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1,
                            output.data, required) != required) {
    return status_from_win32(::GetLastError(), operation);
  }
  return Status::Ok();
}

Result<std::size_t> fill_utf8(char* dst, std::size_t cap,
                              const wchar_t* text, int length) noexcept {
  if (length == 0) return fill(dst, cap, "", 0);
  ::SetLastError(ERROR_SUCCESS);
  const int required = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                              text, length, nullptr, 0,
                                              nullptr, nullptr);
  const DWORD size_error = ::GetLastError();
  if (required == 0 && (length != 0 || size_error != ERROR_SUCCESS)) {
    return std::unexpected(status_from_win32(
        size_error, "environment UTF-16 value conversion"));
  }
  char stack[512];
  char* utf8 = stack;
  if (static_cast<std::size_t>(required) > sizeof(stack)) {
    utf8 = static_cast<char*>(std::malloc(static_cast<std::size_t>(required)));
    if (utf8 == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "environment UTF-8 conversion allocation failed"));
    }
  }
  if (required != 0 &&
      ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text, length,
                            utf8, required, nullptr, nullptr) != required) {
    const DWORD code = ::GetLastError();
    if (utf8 != stack) std::free(utf8);
    return std::unexpected(status_from_win32(
        code, "environment UTF-16 value conversion"));
  }
  const std::size_t result = fill(dst, cap, utf8,
                                  static_cast<std::size_t>(required));
  if (utf8 != stack) std::free(utf8);
  return result;
}

#endif

}  // namespace

Result<std::size_t> get_env_var(const char* name, char* dst,
                                std::size_t cap) noexcept {
  clear_output(dst, cap);
  if (!valid_env_name(name)) {
    return std::unexpected(Status::InvalidArgument(
        "get environment variable: invalid name"));
  }
#if HYLE_OS_WINDOWS
  WideText wide_name;
  Status converted =
      to_wide(name, wide_name, "environment variable name conversion");
  if (!converted.ok()) return std::unexpected(std::move(converted));

  wchar_t stack[512];
  for (unsigned attempt = 0; attempt != 4; ++attempt) {
    ::SetLastError(ERROR_SUCCESS);
    const DWORD required =
        ::GetEnvironmentVariableW(wide_name.data, nullptr, 0);
    const DWORD size_error = ::GetLastError();
    if (required == 0) {
      if (size_error == ERROR_ENVVAR_NOT_FOUND) {
        return std::unexpected(Status::NotFound(
            "get environment variable: variable is not set"));
      }
      if (size_error == ERROR_SUCCESS) return std::size_t{0};
      return std::unexpected(status_from_win32(
          size_error, "GetEnvironmentVariableW(size)"));
    }
    wchar_t* value = stack;
    if (required > std::size(stack)) {
      value = static_cast<wchar_t*>(
          std::malloc(static_cast<std::size_t>(required) * sizeof(wchar_t)));
      if (value == nullptr) {
        return std::unexpected(Status::ResourceExhausted(
            "environment value allocation failed"));
      }
    }
    ::SetLastError(ERROR_SUCCESS);
    const DWORD written =
        ::GetEnvironmentVariableW(wide_name.data, value, required);
    const DWORD value_error = ::GetLastError();
    if (written == 0 && value_error == ERROR_ENVVAR_NOT_FOUND) {
      if (value != stack) std::free(value);
      return std::unexpected(Status::NotFound(
          "get environment variable: variable is not set"));
    }
    if (written >= required) {
      if (value != stack) std::free(value);
      continue;
    }
    if (written == 0 && value_error != ERROR_SUCCESS) {
      if (value != stack) std::free(value);
      return std::unexpected(
          status_from_win32(value_error, "GetEnvironmentVariableW"));
    }
    auto result = fill_utf8(dst, cap, value, static_cast<int>(written));
    if (value != stack) std::free(value);
    return result;
  }
  return std::unexpected(Status::Aborted(
      "environment variable changed repeatedly while reading"));
#else
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::unexpected(Status::NotFound(
        "get environment variable: variable is not set"));
  }
  return fill(dst, cap, value, std::strlen(value));
#endif
}

Status set_env_var(const char* name, const char* value) noexcept {
  if (!valid_env_name(name) || value == nullptr) {
    return Status::InvalidArgument(
        "set environment variable: invalid name or null value");
  }
#if HYLE_OS_WINDOWS
  WideText wide_name;
  WideText wide_value;
  Status status =
      to_wide(name, wide_name, "environment variable name conversion");
  if (!status.ok()) return status;
  status = to_wide(value, wide_value, "environment variable value conversion");
  if (!status.ok()) return status;
  if (::SetEnvironmentVariableW(wide_name.data, wide_value.data)) {
    return Status::Ok();
  }
  return status_from_win32(::GetLastError(), "SetEnvironmentVariableW");
#else
  if (::setenv(name, value, 1) == 0) return Status::Ok();
  return status_from_errno(errno, "setenv");
#endif
}

Status unset_env_var(const char* name) noexcept {
  if (!valid_env_name(name)) {
    return Status::InvalidArgument("unset environment variable: invalid name");
  }
#if HYLE_OS_WINDOWS
  WideText wide_name;
  Status status =
      to_wide(name, wide_name, "environment variable name conversion");
  if (!status.ok()) return status;
  if (::SetEnvironmentVariableW(wide_name.data, nullptr)) return Status::Ok();
  const DWORD code = ::GetLastError();
  return code == ERROR_ENVVAR_NOT_FOUND
             ? Status::Ok()
             : status_from_win32(code, "SetEnvironmentVariableW(unset)");
#else
  if (::unsetenv(name) == 0) return Status::Ok();
  return status_from_errno(errno, "unsetenv");
#endif
}

}  // namespace hyle::platform
