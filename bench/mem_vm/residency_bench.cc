#if !defined(_WIN32)
#  error "mem_vm residency qualification currently has a W1-WIN contract"
#endif

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/mem/mapping.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kMiB = 1024u * 1024u;
constexpr std::size_t kFileBytes = 16u * kMiB;
constexpr std::size_t kStepBytes = 4u * kMiB;

[[noreturn]] void fail(const char* operation) {
  std::fprintf(stderr, "hyle_mem_vm_residency_bench: %s failed (%lu)\n",
               operation, static_cast<unsigned long>(::GetLastError()));
  std::abort();
}

bool environment_present(const char* name) {
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

class RawMapping {
 public:
  RawMapping() = default;
  explicit RawMapping(const std::filesystem::path& path) { open(path); }
  ~RawMapping() { reset(); }

  RawMapping(const RawMapping&) = delete;
  RawMapping& operator=(const RawMapping&) = delete;
  RawMapping(RawMapping&& other) noexcept { swap(other); }
  RawMapping& operator=(RawMapping&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void lock_replay() noexcept {
    for (std::size_t offset = 0; offset < kFileBytes;
         offset += kStepBytes) {
      if (!::VirtualLock(static_cast<std::byte*>(data_) + offset,
                         kStepBytes)) {
        fail("VirtualLock(peer)");
      }
    }
    if (!::VirtualUnlock(data_, kFileBytes)) fail("VirtualUnlock(peer)");
  }

 private:
  void open(const std::filesystem::path& path) {
    file_ = ::CreateFileW(path.c_str(), GENERIC_READ,
                          FILE_SHARE_READ | FILE_SHARE_WRITE |
                              FILE_SHARE_DELETE,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                          nullptr);
    if (file_ == INVALID_HANDLE_VALUE) fail("CreateFileW(peer)");
    mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0,
                                    nullptr);
    if (mapping_ == nullptr) fail("CreateFileMappingW(peer)");
    data_ = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, kFileBytes);
    if (data_ == nullptr) fail("MapViewOfFile(peer)");
  }

  void reset() noexcept {
    if (data_ != nullptr) ::UnmapViewOfFile(data_);
    if (mapping_ != nullptr) ::CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
    data_ = nullptr;
    mapping_ = nullptr;
    file_ = INVALID_HANDLE_VALUE;
  }

  void swap(RawMapping& other) noexcept {
    std::swap(file_, other.file_);
    std::swap(mapping_, other.mapping_);
    std::swap(data_, other.data_);
  }

  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
  void* data_ = nullptr;
};

class ResidencyPair {
 public:
  explicit ResidencyPair(bool peer_allocation_first) {
    path_ = std::filesystem::temp_directory_path() /
             ("hyle_mem_vm_residency_" +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
             ".bin");
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) std::abort();
    constexpr std::size_t kChunk = 64u * 1024u;
    const std::string zeros(kChunk, '\0');
    for (std::size_t written = 0; written < kFileBytes; written += kChunk) {
      output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }
    output.close();
    if (!output) std::abort();

    // The measured contract is progressive lock/unlock after the application
    // has admitted residency. Process-limit policy is a one-time global control
    // action and is deliberately completed before either timed path.
    SIZE_T minimum = 0;
    SIZE_T maximum = 0;
    HANDLE process = ::GetCurrentProcess();
    if (!::GetProcessWorkingSetSize(process, &minimum, &maximum)) {
      fail("GetProcessWorkingSetSize(setup)");
    }
    constexpr SIZE_T kBudget = 2u * kFileBytes + 4u * kMiB;
    if (!::SetProcessWorkingSetSize(process, minimum + kBudget,
                                    maximum + kBudget)) {
      fail("SetProcessWorkingSetSize(setup)");
    }

    if (peer_allocation_first) {
      peer_.emplace(path_);
      open_hyle();
    } else {
      open_hyle();
      peer_.emplace(path_);
    }
  }

  ~ResidencyPair() {
    hyle_.reset();
    peer_.reset();
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  void hyle_replay() noexcept {
    hyle::mem::ResidencyOptions options;
    for (std::size_t target = kStepBytes; target <= kFileBytes;
         target += kStepBytes) {
      if (!hyle_->lock_prefix(target, options).ok()) std::abort();
    }
    if (hyle_->locked_size() != kFileBytes) std::abort();
    if (!hyle_->unlock().ok()) std::abort();
  }

  void peer_replay() noexcept { peer_->lock_replay(); }

 private:
  void open_hyle() {
    const std::string utf8 = path_.string();
    auto mapping = hyle::mem::FileMapping::open(utf8.c_str());
    if (!mapping || mapping->size() != kFileBytes) std::abort();
    hyle_.emplace(std::move(*mapping));
  }

  std::filesystem::path path_;
  std::optional<hyle::mem::FileMapping> hyle_;
  std::optional<RawMapping> peer_;
};

}  // namespace

int main() {
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  std::fprintf(
      stderr,
      "# HYLE_MEM_VM_RESIDENCY_RUN execution_order=%s allocation_order=%s "
      "affinity_topology=%s caller_cpu=%s affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  ResidencyPair pair(peer_allocation_first);
  pair.hyle_replay();
  pair.peer_replay();

  nb::Bench bench;
  bench.title("model loader progressive residency 4x4MiB")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE lock_prefix 4x4MiB",
           [&] { pair.hyle_replay(); }, "Win32 VirtualLock 4x4MiB",
           [&] { pair.peer_replay(); });
}
