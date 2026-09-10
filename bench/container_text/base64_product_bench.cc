#include <nanobench.h>
#include <simdutf.h>

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/container/base64.h"
#include "hyle/core/compiler.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kDecodeBytes = 256 * 1024;
constexpr std::size_t kDecodeMeasurementUnitBytes = 64;
constexpr std::size_t kDecodeMeasurementUnits =
    kDecodeBytes / kDecodeMeasurementUnitBytes;
constexpr std::size_t kEncodeRecords = 4096;
constexpr std::size_t kDigestBytes = 20;
constexpr std::size_t kDigestBase64Bytes = 28;

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
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

nb::Bench make_bench(const char* title, std::size_t batch) {
  nb::Bench bench;
  bench.title(title).relative(true).batch(batch).minEpochTime(
      std::chrono::milliseconds(8));
  return bench;
}

bool ranges_overlap(const void* first, std::size_t first_size,
                    const void* second, std::size_t second_size) noexcept {
  if (first_size == 0 || second_size == 0) return false;
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
  if (first_size > kMax - first_begin || second_size > kMax - second_begin)
    return true;
  return first_begin < second_begin + second_size &&
         second_begin < first_begin + first_size;
}

hyle::container::Base64Status map_provider_error(
    simdutf::error_code error) noexcept {
  using hyle::container::Base64Status;
  switch (error) {
    case simdutf::SUCCESS: return Base64Status::kSuccess;
    case simdutf::INVALID_BASE64_CHARACTER:
      return Base64Status::kInvalidCharacter;
    case simdutf::BASE64_INPUT_REMAINDER:
      return Base64Status::kInvalidLength;
    case simdutf::BASE64_EXTRA_BITS:
      return Base64Status::kNonCanonicalBits;
    case simdutf::OUTPUT_BUFFER_TOO_SMALL:
      return Base64Status::kOutputTooSmall;
    default: return Base64Status::kOther;
  }
}

// Direct-provider adapters reproduce the fixed public contract, including the
// checks charged to the HYLE call. They differ only in bypassing the exported
// HYLE seam before entering the same vendored simdutf implementation.
HYLE_NOINLINE hyle::container::Base64Result peer_decode(
    std::span<const char> input, std::span<std::byte> output,
    hyle::container::Base64DecodeOptions options = {}) noexcept {
  using namespace hyle::container;
  if (!detail::valid_base64_decode_options(options) ||
      (input.data() == nullptr && !input.empty()) ||
      (output.data() == nullptr && !output.empty()) ||
      ranges_overlap(input.data(), input.size(), output.data(), output.size())) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }
  if (input.empty()) return {};
  for (std::size_t i = 0; i < input.size(); ++i) {
    const char value = input[i];
    if (value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
        value == '\f') {
      return {Base64Status::kWhitespaceRejected, i, 0};
    }
  }
  char empty_output = 0;
  char* output_data = output.empty()
                          ? &empty_output
                          : reinterpret_cast<char*>(output.data());
  std::size_t output_count = output.size();
  const simdutf::result result = simdutf::base64_to_binary_safe(
      input.data(), input.size(), output_data, output_count,
      simdutf::base64_default,
      simdutf::last_chunk_handling_options::strict, false);
  return {map_provider_error(result.error), result.count, output_count};
}

HYLE_NOINLINE hyle::container::Base64Result peer_encode(
    std::span<const std::byte> input, std::span<char> output,
    hyle::container::Base64EncodeOptions options = {}) noexcept {
  using namespace hyle::container;
  if (!detail::valid_base64_encode_options(options) ||
      (input.data() == nullptr && !input.empty()) ||
      (output.data() == nullptr && !output.empty()) ||
      ranges_overlap(input.data(), input.size(), output.data(), output.size())) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }
  const Base64SizeResult size = base64_encoded_size(input.size(), options);
  if (!size) return {size.status, size.input_count, 0};
  if (output.size() < size.output_size)
    return {Base64Status::kOutputTooSmall, 0, 0};
  if (input.empty()) return {};
  const std::size_t written = simdutf::binary_to_base64(
      reinterpret_cast<const char*>(input.data()), input.size(), output.data(),
      simdutf::base64_default);
  return {Base64Status::kSuccess, input.size(), written};
}

std::uint64_t sample_checksum(std::span<const std::byte> bytes) noexcept {
  if (bytes.empty()) return 0;
  return static_cast<std::uint64_t>(bytes.front()) * UINT64_C(257) +
         static_cast<std::uint64_t>(bytes[bytes.size() / 2]) * UINT64_C(17) +
         static_cast<std::uint64_t>(bytes.back()) + bytes.size();
}

std::uint64_t sample_checksum(std::span<const char> bytes) noexcept {
  if (bytes.empty()) return 0;
  return static_cast<unsigned char>(bytes.front()) * UINT64_C(257) +
         static_cast<unsigned char>(bytes[bytes.size() / 2]) * UINT64_C(17) +
         static_cast<unsigned char>(bytes.back()) + bytes.size();
}

class DecodeReplay {
 public:
  DecodeReplay() : raw_(kDecodeBytes), output_(kDecodeBytes) {
    for (std::size_t i = 0; i < raw_.size(); ++i) {
      raw_[i] = std::byte{static_cast<unsigned char>(
          (i * UINT64_C(1315423911) + (i >> 3) + 29) & 0xff)};
    }
    const auto size = hyle::container::base64_encoded_size(raw_.size());
    if (!size) std::abort();
    encoded_.resize(size.output_size);
    const auto result =
        peer_encode(raw_, std::span<char>{encoded_.data(), encoded_.size()});
    if (!result || result.output_count != encoded_.size()) std::abort();
  }

  HYLE_NOINLINE std::uint64_t candidate() noexcept {
    const auto result = hyle::container::base64_decode(encoded_, output_);
    if (!result || result.output_count != raw_.size()) std::abort();
    return sample_checksum(output_) + result.input_count;
  }

  HYLE_NOINLINE std::uint64_t peer() noexcept {
    const auto result = peer_decode(encoded_, output_);
    if (!result || result.output_count != raw_.size()) std::abort();
    return sample_checksum(output_) + result.input_count;
  }

  [[nodiscard]] bool valid() noexcept {
    const auto candidate_result = hyle::container::base64_decode(encoded_, output_);
    const bool candidate_ok = candidate_result && output_ == raw_;
    const auto peer_result = peer_decode(encoded_, output_);
    return candidate_ok && peer_result && output_ == raw_ &&
           candidate_result.status == peer_result.status &&
           candidate_result.input_count == peer_result.input_count &&
           candidate_result.output_count == peer_result.output_count;
  }

 private:
  std::vector<std::byte> raw_;
  std::vector<char> encoded_;
  std::vector<std::byte> output_;
};

class EncodeReplay {
 public:
  EncodeReplay()
      : input_(kEncodeRecords * kDigestBytes),
        output_(kEncodeRecords * kDigestBase64Bytes) {
    for (std::size_t i = 0; i < input_.size(); ++i) {
      input_[i] = std::byte{static_cast<unsigned char>(
          (i * UINT64_C(11400714819323198485) + 7) & 0xff)};
    }
  }

  HYLE_NOINLINE std::uint64_t candidate() noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t record = 0; record < kEncodeRecords; ++record) {
      const auto input = std::span<const std::byte>{input_}.subspan(
          record * kDigestBytes, kDigestBytes);
      const auto output =
          std::span<char>{output_}.subspan(record * kDigestBase64Bytes,
                                          kDigestBase64Bytes);
      const auto result = hyle::container::base64_encode(input, output);
      if (!result || result.output_count != output.size()) std::abort();
      checksum += static_cast<unsigned char>(output.front()) +
                  static_cast<unsigned char>(output.back()) +
                  result.output_count;
    }
    return checksum;
  }

  HYLE_NOINLINE std::uint64_t peer() noexcept {
    std::uint64_t checksum = 0;
    for (std::size_t record = 0; record < kEncodeRecords; ++record) {
      const auto input = std::span<const std::byte>{input_}.subspan(
          record * kDigestBytes, kDigestBytes);
      const auto output =
          std::span<char>{output_}.subspan(record * kDigestBase64Bytes,
                                          kDigestBase64Bytes);
      const auto result = peer_encode(input, output);
      if (!result || result.output_count != output.size()) std::abort();
      checksum += static_cast<unsigned char>(output.front()) +
                  static_cast<unsigned char>(output.back()) +
                  result.output_count;
    }
    return checksum;
  }

  [[nodiscard]] bool valid() noexcept {
    const std::uint64_t candidate_checksum = candidate();
    const std::vector<char> candidate_output = output_;
    const std::uint64_t peer_checksum = peer();
    return candidate_checksum == peer_checksum && output_ == candidate_output;
  }

 private:
  std::vector<std::byte> input_;
  std::vector<char> output_;
};

template <class Function>
std::uintptr_t function_address(Function function) noexcept {
  return reinterpret_cast<std::uintptr_t>(function);
}

int validate_contract() {
  DecodeReplay decode;
  EncodeReplay encode;

  std::array<std::byte, 8> candidate_output{};
  std::array<std::byte, 8> peer_output{};
  constexpr std::string_view kWhitespace = "Z g==";
  const auto candidate_whitespace = hyle::container::base64_decode(
      std::span<const char>{kWhitespace}, candidate_output);
  const auto peer_whitespace_result = peer_decode(
      std::span<const char>{kWhitespace}, peer_output);
  constexpr std::string_view kInvalid = "Zm#v";
  const auto candidate_invalid = hyle::container::base64_decode(
      std::span<const char>{kInvalid}, candidate_output);
  const auto peer_invalid =
      peer_decode(std::span<const char>{kInvalid}, peer_output);

  const bool failure_equal =
      candidate_whitespace.status == peer_whitespace_result.status &&
      candidate_whitespace.input_count == peer_whitespace_result.input_count &&
      candidate_whitespace.output_count ==
          peer_whitespace_result.output_count &&
      candidate_invalid.status == peer_invalid.status &&
      candidate_invalid.input_count == peer_invalid.input_count &&
      candidate_invalid.output_count == peer_invalid.output_count;
  const bool passed = decode.valid() && encode.valid() && failure_equal;

  using DecodeFunction = hyle::container::Base64Result (*)(
      std::span<const char>, std::span<std::byte>,
      hyle::container::Base64DecodeOptions) noexcept;
  using EncodeFunction = hyle::container::Base64Result (*)(
      std::span<const std::byte>, std::span<char>,
      hyle::container::Base64EncodeOptions) noexcept;
  const auto hyle_decode = static_cast<DecodeFunction>(
      &hyle::container::base64_decode);
  const auto hyle_encode = static_cast<EncodeFunction>(
      &hyle::container::base64_encode);

  std::printf(
      "# HYLE_CONTAINER_TEXT_BASE64_CONTRACT schema=1 decode=%s encode=%s "
      "failures=%s decode_bytes=%zu encode_records=%zu digest_bytes=%zu "
      "verdict=%s\n",
      decode.valid() ? "PASS" : "FAIL", encode.valid() ? "PASS" : "FAIL",
      failure_equal ? "PASS" : "FAIL", kDecodeBytes, kEncodeRecords,
      kDigestBytes, passed ? "PASS" : "FAIL");
  std::printf(
      "# HYLE_CONTAINER_TEXT_BASE64_LAYOUT hyle_decode=%" PRIxPTR
      " peer_decode=%" PRIxPTR " hyle_encode=%" PRIxPTR
      " peer_encode=%" PRIxPTR "\n",
      function_address(hyle_decode), function_address(&peer_decode),
      function_address(hyle_encode), function_address(&peer_encode));
  return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  if (argc == 2 && std::string_view(argv[1]) == "--validate-contract")
    return validate_contract();

  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_CONTAINER_TEXT_BASE64_RUN schema=1 execution_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s decode_bytes=%zu encode_records=%zu digest_bytes=%zu "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok, kDecodeBytes,
      kEncodeRecords, kDigestBytes, hyle::bench::perf::run_nonce());

  DecodeReplay decode;
  nb::Bench decode_bench = make_bench(
      "container_text multimodal Base64 strict decode 64-byte units N=4096",
      kDecodeMeasurementUnits);
  run_pair(
      decode_bench, peer_first,
      "HYLE Base64 strict decode 64-byte units N=4096",
      [&] { nb::doNotOptimizeAway(decode.candidate()); },
      "simdutf semantic Base64 strict decode 64-byte units N=4096",
      [&] { nb::doNotOptimizeAway(decode.peer()); });

  EncodeReplay encode;
  nb::Bench encode_bench = make_bench(
      "container_text WebSocket-accept Base64 encode records N=4096",
      kEncodeRecords);
  run_pair(
      encode_bench, peer_first,
      "HYLE Base64 20-byte encode records N=4096",
      [&] { nb::doNotOptimizeAway(encode.candidate()); },
      "simdutf semantic 20-byte encode records N=4096",
      [&] { nb::doNotOptimizeAway(encode.peer()); });
  return 0;
}
