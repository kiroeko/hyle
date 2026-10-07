#include "driver_release.h"
#include "hyle/core/environment.h"

#include <algorithm>
#include <iterator>

#include <nvml.h>

#include "hyle/platform/dynlib.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace hyle::compute::cuda::detail {
namespace {

#if HYLE_OS_WINDOWS
struct LoaderErrorMode final {
  DWORD previous = 0;
  bool changed = false;

  LoaderErrorMode() noexcept {
    changed = ::SetThreadErrorMode(
                  ::GetThreadErrorMode() | SEM_FAILCRITICALERRORS |
                      SEM_NOOPENFILEERRORBOX,
                  &previous) != FALSE;
  }
  ~LoaderErrorMode() {
    if (changed)
      (void)::SetThreadErrorMode(previous, nullptr);
  }
};
#endif

Result<platform::DynLib> open_driver_library() noexcept {
#if HYLE_OS_WINDOWS
  std::array<wchar_t, 512> directory{};
  const auto length = ::GetSystemDirectoryW(directory.data(),
                                            static_cast<UINT>(directory.size()));
  constexpr wchar_t suffix[] = L"\\nvml.dll";
  if (length == 0 || length >= directory.size() - std::size(suffix))
    return std::unexpected(Status::Unavailable("NVML system path unavailable"));
  std::copy(std::begin(suffix), std::end(suffix), directory.begin() + length);
  std::array<char, 2048> path{};
  if (::WideCharToMultiByte(CP_UTF8, 0, directory.data(), -1, path.data(),
                          static_cast<int>(path.size()), nullptr, nullptr) == 0)
    return std::unexpected(Status::Unavailable("NVML system path unavailable"));
  return platform::DynLib::open(path.data());
#else
  return platform::DynLib::open("libnvidia-ml.so.1");
#endif
}

} // namespace

std::array<char, 80> read_driver_release() noexcept {
  static_assert(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE <= 80);
  std::array<char, 80> release{};
#if HYLE_OS_WINDOWS
  LoaderErrorMode silent_loader;
#endif
  auto library = open_driver_library();
  if (!library)
    return release;
  auto initialize = library->symbol_as<decltype(&nvmlInit_v2)>("nvmlInit_v2");
  auto query = library->symbol_as<decltype(&nvmlSystemGetDriverVersion)>(
      "nvmlSystemGetDriverVersion");
  auto shutdown = library->symbol_as<decltype(&nvmlShutdown)>("nvmlShutdown");
  if (!initialize || !query || !shutdown || !*initialize || !*query ||
      !*shutdown || (*initialize)() != NVML_SUCCESS)
    return release;
  // NVML counts successful initialization references. Balance this one before
  // unloading our library reference; no GPU enumeration or global cache.
  const auto result = (*query)(release.data(),
                                static_cast<unsigned int>(release.size()));
  (void)(*shutdown)();
  if (result != NVML_SUCCESS)
    release.fill('\0');
  release.back() = '\0';
  return release;
}

} // namespace hyle::compute::cuda::detail
