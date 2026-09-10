#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "hyle/container/visibility.h"

// Allocation-free contiguous Base64/Base64URL codecs. The caller owns both
// buffers; output is length-delimited and is never NUL-terminated implicitly.
// Defaults deliberately describe canonical RFC 4648 Base64 rather than a
// permissive data-cleaning policy.

namespace hyle::container {

enum class Base64Alphabet : std::uint8_t {
  kStandard,
  kUrl,
};

enum class Base64EncodePadding : std::uint8_t {
  kInclude,
  kOmit,
};

// Every mode requires canonical unused tail bits. kAllowed accepts either the
// canonical padded or canonical unpadded spelling; it never accepts malformed
// padding or arbitrary trailing garbage.
enum class Base64DecodePadding : std::uint8_t {
  kRequired,
  kAllowed,
  kForbidden,
};

// ASCII whitespace is exactly space, tab, LF, CR and form-feed, matching the
// WHATWG/simdutf Base64 definition. Other non-alphabet bytes are never ignored.
enum class Base64Whitespace : std::uint8_t {
  kReject,
  kAllowAscii,
};

struct Base64EncodeOptions {
  Base64Alphabet alphabet = Base64Alphabet::kStandard;
  Base64EncodePadding padding = Base64EncodePadding::kInclude;
};

struct Base64DecodeOptions {
  Base64Alphabet alphabet = Base64Alphabet::kStandard;
  Base64DecodePadding padding = Base64DecodePadding::kRequired;
  Base64Whitespace whitespace = Base64Whitespace::kReject;
};

enum class Base64Status : std::uint8_t {
  kSuccess,
  kInvalidArgument,
  kInvalidCharacter,
  kInvalidPadding,
  kInvalidLength,
  kNonCanonicalBits,
  kWhitespaceRejected,
  kOutputTooSmall,
  kSizeOverflow,
  kOther,
};

struct Base64SizeResult {
  Base64Status status = Base64Status::kSuccess;
  // On success, the complete input length. On malformed decoded input, the
  // first offending offset (or input.size() for an incomplete tail).
  std::size_t input_count = 0;
  std::size_t output_size = 0;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return status == Base64Status::kSuccess;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
};

struct Base64Result {
  Base64Status status = Base64Status::kSuccess;
  // Number of input units consumed and output bytes actually written. Failed
  // decode output is uncommitted and must be discarded even when non-empty.
  std::size_t input_count = 0;
  std::size_t output_count = 0;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return status == Base64Status::kSuccess;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
};

namespace detail {

constexpr bool valid_base64_encode_options(
    Base64EncodeOptions options) noexcept {
  return (options.alphabet == Base64Alphabet::kStandard ||
          options.alphabet == Base64Alphabet::kUrl) &&
         (options.padding == Base64EncodePadding::kInclude ||
          options.padding == Base64EncodePadding::kOmit);
}

constexpr bool valid_base64_decode_options(
    Base64DecodeOptions options) noexcept {
  return (options.alphabet == Base64Alphabet::kStandard ||
          options.alphabet == Base64Alphabet::kUrl) &&
         (options.padding == Base64DecodePadding::kRequired ||
          options.padding == Base64DecodePadding::kAllowed ||
          options.padding == Base64DecodePadding::kForbidden) &&
         (options.whitespace == Base64Whitespace::kReject ||
          options.whitespace == Base64Whitespace::kAllowAscii);
}

}  // namespace detail

// Exact encoded size with overflow and option validation. This performs no
// allocation and is O(1).
[[nodiscard]] constexpr Base64SizeResult base64_encoded_size(
    std::size_t input_size, Base64EncodeOptions options = {}) noexcept {
  if (!detail::valid_base64_encode_options(options)) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }

  const std::size_t groups = input_size / 3;
  const std::size_t tail = input_size % 3;
  if (groups > std::numeric_limits<std::size_t>::max() / 4) {
    return {Base64Status::kSizeOverflow, input_size, 0};
  }
  std::size_t output_size = groups * 4;
  if (tail != 0) {
    const std::size_t tail_size =
        options.padding == Base64EncodePadding::kInclude ? 4 : tail + 1;
    if (output_size >
        std::numeric_limits<std::size_t>::max() - tail_size) {
      return {Base64Status::kSizeOverflow, input_size, 0};
    }
    output_size += tail_size;
  }
  return {Base64Status::kSuccess, input_size, output_size};
}

// Safe upper bound from the encoded storage extent alone. Whitespace and
// padding can only reduce the decoded size. Use base64_decoded_size when an
// exact size plus detailed validation is worth an additional O(n) pass.
[[nodiscard]] constexpr std::size_t base64_max_decoded_size(
    std::size_t encoded_size) noexcept {
  const std::size_t groups = encoded_size / 4;
  const std::size_t tail = encoded_size % 4;
  return groups * 3 + (tail * 3) / 4;
}

// Cold checked-size/diagnostic route. It validates the complete input and
// returns a stable first-invalid offset without writing output.
[[nodiscard]] HYLE_CONTAINER_TEXT_API Base64SizeResult base64_decoded_size(
    std::span<const char> input, Base64DecodeOptions options = {}) noexcept;

// Bulk codec routes. They never allocate. Encode is all-or-nothing when the
// output is too small. Decode may have written partial, uncommitted bytes on
// any failure; discard the complete output span in that case. Input and output
// storage must not overlap.
[[nodiscard]] HYLE_CONTAINER_TEXT_API Base64Result base64_encode(
    std::span<const std::byte> input, std::span<char> output,
    Base64EncodeOptions options = {}) noexcept;
[[nodiscard]] HYLE_CONTAINER_TEXT_API Base64Result base64_decode(
    std::span<const char> input, std::span<std::byte> output,
    Base64DecodeOptions options = {}) noexcept;

// Zero-copy conveniences for the two common byte-storage spellings. These are
// inline adapters over the same exported byte-span core, not extra codecs.
[[nodiscard]] inline Base64Result base64_encode(
    std::span<const char> input, std::span<char> output,
    Base64EncodeOptions options = {}) noexcept {
  return base64_encode(std::as_bytes(input), output, options);
}

[[nodiscard]] inline Base64Result base64_decode(
    std::span<const char> input, std::span<char> output,
    Base64DecodeOptions options = {}) noexcept {
  return base64_decode(input, std::as_writable_bytes(output), options);
}

}  // namespace hyle::container
