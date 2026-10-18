#include "hyle/io/detail/proactor.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

using hyle::io::detail::Completion;
using hyle::io::detail::FileStat;
using hyle::io::detail::Proactor;

[[noreturn]] void fail(const char* what) {
  std::fprintf(stderr, "batch: %s\n", what);
  std::exit(1);
}

void episode(Proactor& p, int fd, std::size_t count, int mode,
             std::uint64_t generation, std::vector<FileStat>& stats,
             std::vector<std::array<unsigned char, 8>>& bytes,
             std::vector<bool>& seen) {
  const std::uint64_t base = generation * 2048;
  const std::size_t reads = mode == 2 ? count : (mode == 1 ? count / 2 : 0);
  for (std::size_t i = 0; i < count; ++i) {
    stats[i] = FileStat{};
    seen[i] = false;
    const bool submitted = i < reads
        ? p.submit_read(fd, bytes[i].data(), bytes[i].size(), 0, base + i)
        : p.submit_statx(fd, &stats[i], base + i);
    if (!submitted) fail("submit rejected within declared capacity");
  }
  std::array<Completion, 128> completions{};
  std::size_t completed = 0;
  while (completed < count) {
    const std::size_t got = p.reap(completions.data(), completions.size(), -1);
    for (std::size_t j = 0; j < got; ++j) {
      const auto& c = completions[j];
      if (c.user_data < base || c.user_data >= base + count)
        fail("unexpected completion token");
      const std::size_t i = c.user_data - base;
      if (seen[i]) fail("duplicate completion");
      seen[i] = true;
      if (i < reads) {
        if (c.result != 8 || bytes[i] != std::array<unsigned char, 8>{1,2,3,4,5,6,7,8})
          fail("read completion/data mismatch");
      } else if (c.result != 0 || stats[i].size != 8 || stats[i].is_directory) {
        fail("stat completion/data mismatch");
      }
      ++completed;
    }
  }
  if (p.in_flight()) fail("episode did not fully retire");
}

int main(int argc, char** argv) {
  if (argc != 3) fail("usage: batch label fixture-path");
  const int fd = ::open(argv[2], O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0) fail("open fixture");
  constexpr std::array<unsigned char,8> payload{1,2,3,4,5,6,7,8};
  if (::write(fd, payload.data(), payload.size()) != 8) fail("write fixture");
  Proactor p;
  if (!p.init(2048)) fail("init");
  if (p.backend_kind() != hyle::io::IoBackend::kIoUring)
    fail("requires native io_uring; fallback is not measured");
  std::vector<FileStat> stats(1024);
  std::vector<std::array<unsigned char,8>> bytes(1024);
  std::vector<bool> seen(1024);
  std::uint64_t generation = 1;
  for (int mode : {0, 1, 2}) {
    for (std::size_t count : {16u, 256u, 1024u}) {
      if (mode == 2 && count != 1024) continue;
      for (int i = 0; i < 16; ++i)
        episode(p, fd, count, mode, generation++, stats, bytes, seen);
      const std::size_t rounds = 262144 / count;
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t round = 0; round < rounds; ++round)
        episode(p, fd, count, mode, generation++, stats, bytes, seen);
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      std::printf("{\"label\":\"%s\",\"kind\":\"%s\",\"batch\":%zu,"
                  "\"operations\":%zu,\"seconds\":%.9f,\"ns_per_op\":%.3f,"
                  "\"backend\":\"%s\",\"validated\":true}\n",
                  argv[1], mode == 2 ? "read-only" : (mode == 1 ? "read-first-half-statx" : "statx"),
                  count, rounds * count, seconds,
                  seconds * 1e9 / (rounds * count), p.backend_name());
    }
  }
  if (::close(fd) != 0) fail("close fixture");
}
