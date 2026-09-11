#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <fcntl.h>
#  include <io.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <spawn.h>
#  include <unistd.h>
extern char** environ;
#endif

namespace {

bool spawn_grandchild(const char* executable) {
#if defined(_WIN32)
  (void)executable;
  wchar_t path[32768]{};
  const DWORD length =
      ::GetModuleFileNameW(nullptr, path, static_cast<DWORD>(std::size(path)));
  if (length == 0 || length == std::size(path)) return false;
  wchar_t command[33000]{};
  if (_snwprintf_s(command, std::size(command), _TRUNCATE,
                   L"\"%ls\" sleep 30000", path) <= 0) {
    return false;
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!::CreateProcessW(path, command, nullptr, nullptr, TRUE, 0, nullptr,
                        nullptr, &startup, &process)) {
    return false;
  }
  (void)::CloseHandle(process.hThread);
  (void)::CloseHandle(process.hProcess);
  return true;
#else
  char* arguments[] = {const_cast<char*>(executable),
                       const_cast<char*>("sleep"),
                       const_cast<char*>("30000"), nullptr};
  pid_t child = -1;
  return ::posix_spawn(&child, executable, nullptr, nullptr, arguments,
                       environ) == 0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  // The helper validates an arbitrary-byte process pipe, not the MSVCRT text
  // convention where 0x1A ends stdin and newline bytes are translated.
  if (::_setmode(::_fileno(stdin), _O_BINARY) < 0 ||
      ::_setmode(::_fileno(stdout), _O_BINARY) < 0 ||
      ::_setmode(::_fileno(stderr), _O_BINARY) < 0) {
    return 100;
  }
#endif
  if (argc < 2) return 101;
  const std::string_view mode = argv[1];
  if (mode == "exit") {
    if (argc != 3) return 102;
    return std::atoi(argv[2]);
  }
  if (mode == "sleep") {
    if (argc != 3) return 103;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(std::atoi(argv[2])));
    return 0;
  }
  if (mode == "echo") {
    char buffer[4096];
    for (;;) {
      const std::size_t count = std::fread(buffer, 1, sizeof(buffer), stdin);
      if (count != 0) {
        if (std::fwrite(buffer, 1, count, stdout) != count) return 104;
        std::fflush(stdout);
      }
      if (count != sizeof(buffer)) {
        if (std::ferror(stdin)) return 105;
        break;
      }
    }
    std::fputs("child-stderr", stderr);
    std::fflush(stderr);
    return 0;
  }
  if (mode == "environment") {
    if (argc != 3) return 106;
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&value, &value_size, argv[2]) != 0 || value == nullptr)
      return 107;
    const int written = std::fputs(value, stdout);
    std::free(value);
    if (written < 0) return 112;
#else
    const char* value = std::getenv(argv[2]);
    if (value == nullptr) return 107;
    std::fputs(value, stdout);
#endif
    std::fflush(stdout);
    return 0;
  }
  if (mode == "environment_absent") {
    if (argc != 3) return 113;
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t value_size = 0;
    const int result = _dupenv_s(&value, &value_size, argv[2]);
    const bool absent = result == 0 && value == nullptr;
    std::free(value);
    return absent ? 0 : 114;
#else
    return std::getenv(argv[2]) == nullptr ? 0 : 114;
#endif
  }
  if (mode == "arguments") {
    if (argc != 6) return 109;
    return std::string_view(argv[2]).empty() &&
                   std::string_view(argv[3]) == "a b" &&
                   std::string_view(argv[4]) == "\"" &&
                   std::string_view(argv[5]) == "trail\\"
               ? 0
               : 110;
  }
  if (mode == "tree") {
    if (!spawn_grandchild(argv[0])) return 111;
    std::fputs("ready", stdout);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return 0;
  }
  if (mode == "streams") {
    std::fputs("out", stdout);
    std::fflush(stdout);
    std::fputs("err", stderr);
    std::fflush(stderr);
    return 0;
  }
  if (mode == "cwd_marker") {
    std::FILE* marker = nullptr;
#if defined(_WIN32)
    if (::fopen_s(&marker, "hyle-child-cwd-marker", "wb") != 0)
      marker = nullptr;
#else
    marker = std::fopen("hyle-child-cwd-marker", "wb");
#endif
    if (marker == nullptr) return 115;
    const int closed = std::fclose(marker);
    return closed == 0 ? 0 : 116;
  }
  if (mode == "unrelated_handle_absent") {
    if (argc != 3) return 117;
#if defined(_WIN32)
    const auto raw = static_cast<std::uintptr_t>(
        std::strtoull(argv[2], nullptr, 10));
    DWORD flags = 0;
    ::SetLastError(ERROR_SUCCESS);
    const BOOL visible = ::GetHandleInformation(
        reinterpret_cast<HANDLE>(raw), &flags);
    return visible == FALSE && ::GetLastError() == ERROR_INVALID_HANDLE
               ? 0
               : 118;
#else
    const int descriptor = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    errno = 0;
    return ::fcntl(descriptor, F_GETFD) < 0 && errno == EBADF ? 0 : 118;
#endif
  }
  return 108;
}
