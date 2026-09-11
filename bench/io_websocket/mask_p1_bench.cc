// RFC 6455 bulk mask P1: identical mutable bytes, prepared four-byte key and
// frame completion. HTTP, sockets, schedulers and randomness are deliberately
// outside this primitive contract.

#include <nanobench.h>

#include <boost/asio/buffer.hpp>
#include <boost/beast/websocket/detail/mask.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include "websocket_mask_internal.h"

namespace beast_websocket = boost::beast::websocket;

namespace {

constexpr std::array<std::byte, 4> kHyleKey{std::byte{0x37}, std::byte{0xfa},
                                            std::byte{0x21}, std::byte{0x3d}};
constexpr std::size_t kBytesPerEpoch = 16u * 1024u * 1024u;

void run_case(std::size_t size, const char *title, bool peer_first) {
  std::vector<std::byte> hyle_bytes(size, std::byte{0x5a});
  std::vector<unsigned char> beast_bytes(size, 0x5a);
  beast_websocket::detail::prepared_key beast_key{};
  beast_websocket::detail::prepare_key(beast_key, 0x3d21fa37u);
  const std::size_t rounds = std::max<std::size_t>(1, kBytesPerEpoch / size);

  auto run_hyle = [&] {
    for (std::size_t i = 0; i != rounds; ++i)
      hyle::io::detail::websocket_apply_mask(hyle_bytes, kHyleKey);
    ankerl::nanobench::doNotOptimizeAway(hyle_bytes.data());
  };
  auto run_beast = [&] {
    for (std::size_t i = 0; i != rounds; ++i) {
      beast_websocket::detail::mask_inplace(boost::asio::buffer(beast_bytes),
                                            beast_key);
    }
    ankerl::nanobench::doNotOptimizeAway(beast_bytes.data());
  };

  run_hyle();
  run_beast();
  if (!std::equal(hyle_bytes.begin(), hyle_bytes.end(), beast_bytes.begin(),
                  [](std::byte left, unsigned char right) {
                    return std::to_integer<unsigned char>(left) == right;
                  })) {
    std::abort();
  }

  ankerl::nanobench::Bench bench;
  bench.title(title)
      .relative(true)
      .unit("byte")
      .batch(static_cast<std::uint64_t>(rounds * size))
      .minEpochIterations(1)
      .epochs(11);
  if (peer_first) {
    bench.run("Boost.Beast prepared-key mask", run_beast);
    bench.run("HYLE prepared-key mask", run_hyle);
  } else {
    bench.run("HYLE prepared-key mask", run_hyle);
    bench.run("Boost.Beast prepared-key mask", run_beast);
  }
}

} // namespace

int main() {
  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  const char *run_nonce = std::getenv("HYLE_PERF_RUN_NONCE");
  std::fprintf(stderr,
               "# HYLE_IO_WEBSOCKET_MASK_RUN execution_order=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               run_nonce != nullptr ? run_nonce : "unmanaged");
  run_case(124, "WebSocket mask control-size 124B", peer_first);
  run_case(4 * 1024, "WebSocket mask token/audio chunk 4KiB", peer_first);
  run_case(64 * 1024, "WebSocket mask binary chunk 64KiB", peer_first);
  run_case(1024 * 1024, "WebSocket mask large frame 1MiB", peer_first);
  return 0;
}
