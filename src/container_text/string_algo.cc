#include "hyle/container/string_algo.h"

#include "hyle/container/base64.h"

#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
#  include "simdutf.h"

static_assert(simdutf::SIMDUTF_VERSION_MAJOR == 9 &&
                  simdutf::SIMDUTF_VERSION_MINOR == 1 &&
                  simdutf::SIMDUTF_VERSION_REVISION == 0,
              "HYLE's reviewed simdutf contract requires v9.1.0");
#endif

// string_algo.cc -- runtime-dispatched UTF-8 validate/count entries. The shipping
// container_text factor directly embeds simdutf's verified SIMD cores
// ("借核不手搓").
// The conditional keeps this TU usable in stripped/standalone builds; the scalar
// Utf8Decoder is also the constexpr/reference fallback.
//
// CONTRACT MATCH: utf8_length returns the code-point count, or -1 on malformed
// input -- i.e. VALIDATE + COUNT. simdutf splits these (validate_utf8 +
// count_utf8), so the implementation does both: validate first, count only if
// valid. Calling count_utf8 alone would violate the public malformed-input
// contract because that provider operation intentionally skips validation.

namespace hyle::container::detail {

bool utf8_validate_runtime(StringView s) noexcept {
  if (s.data() == nullptr) return s.size() == 0;
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  return simdutf::validate_utf8(s.data(), s.size());
#else
  Utf8Decoder decoder(s);
  while (!decoder.is_end()) {
    decoder.next();
    if (!decoder.valid()) return false;
  }
  return true;
#endif
}

std::int64_t utf8_length_runtime(StringView s) noexcept {
  if (s.data() == nullptr) return s.size() == 0 ? 0 : -1;
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  // Borrowed SIMD core: validate then count (matches utf8_length's validate+count
  // contract). simdutf runs its own internal runtime ISA dispatch.
  if (!simdutf::validate_utf8(s.data(), s.size())) return -1;
  return static_cast<std::int64_t>(simdutf::count_utf8(s.data(), s.size()));
#else
  // Scalar fallback: identical to the header's compile-time path.
  Utf8Decoder d(s);
  std::int64_t n = 0;
  while (!d.is_end()) {
    d.next();
    if (!d.valid()) return -1;
    ++n;
  }
  return n;
#endif
}

}  // namespace hyle::container::detail

namespace hyle::container {
namespace {

#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
using UtfProvider = simdutf::implementation;

// Resolve the private runtime-dispatch provider once per public operation. A
// safe conversion performs validation, sizing and conversion; routing every
// stage through simdutf's free functions would reload the same atomic active
// implementation three times. Keeping the pointer operation-local observes a
// provider change on the next call without adding global/TLS state.
HYLE_FORCE_INLINE const UtfProvider* active_utf_provider() noexcept {
  return simdutf::get_active_implementation();
}

HYLE_FORCE_INLINE simdutf::result provider_validate_utf16(
    const UtfProvider* provider, const char16_t* input,
    std::size_t size) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->validate_utf16be_with_errors(input, size);
#  else
  return provider->validate_utf16le_with_errors(input, size);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_convert_utf8_to_utf16(
    const UtfProvider* provider, const char* input, std::size_t size,
    char16_t* output) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->convert_valid_utf8_to_utf16be(input, size, output);
#  else
  return provider->convert_valid_utf8_to_utf16le(input, size, output);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_utf8_length_from_utf16(
    const UtfProvider* provider, const char16_t* input,
    std::size_t size) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->utf8_length_from_utf16be(input, size);
#  else
  return provider->utf8_length_from_utf16le(input, size);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_convert_utf16_to_utf8(
    const UtfProvider* provider, const char16_t* input, std::size_t size,
    char* output) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->convert_valid_utf16be_to_utf8(input, size, output);
#  else
  return provider->convert_valid_utf16le_to_utf8(input, size, output);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_utf32_length_from_utf16(
    const UtfProvider* provider, const char16_t* input,
    std::size_t size) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->utf32_length_from_utf16be(input, size);
#  else
  return provider->utf32_length_from_utf16le(input, size);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_convert_utf16_to_utf32(
    const UtfProvider* provider, const char16_t* input, std::size_t size,
    char32_t* output) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->convert_valid_utf16be_to_utf32(input, size, output);
#  else
  return provider->convert_valid_utf16le_to_utf32(input, size, output);
#  endif
}

HYLE_FORCE_INLINE std::size_t provider_convert_utf32_to_utf16(
    const UtfProvider* provider, const char32_t* input, std::size_t size,
    char16_t* output) noexcept {
#  if SIMDUTF_IS_BIG_ENDIAN
  return provider->convert_valid_utf32_to_utf16be(input, size, output);
#  else
  return provider->convert_valid_utf32_to_utf16le(input, size, output);
#  endif
}

UtfStatus from_simdutf(simdutf::error_code error) noexcept {
  switch (error) {
    case simdutf::SUCCESS: return UtfStatus::kSuccess;
    case simdutf::HEADER_BITS: return UtfStatus::kHeaderBits;
    case simdutf::TOO_SHORT: return UtfStatus::kTooShort;
    case simdutf::TOO_LONG: return UtfStatus::kTooLong;
    case simdutf::OVERLONG: return UtfStatus::kOverlong;
    case simdutf::TOO_LARGE: return UtfStatus::kTooLarge;
    case simdutf::SURROGATE: return UtfStatus::kSurrogate;
    case simdutf::OUTPUT_BUFFER_TOO_SMALL:
      return UtfStatus::kOutputTooSmall;
    default: return UtfStatus::kOther;
  }
}

UtfResult validation_result(simdutf::result result) noexcept {
  if (result.error == simdutf::SUCCESS)
    return {UtfStatus::kSuccess, result.count, 0};
  return {from_simdutf(result.error), result.count, 0};
}
#else
struct UtfProvider {};

HYLE_FORCE_INLINE const UtfProvider* active_utf_provider() noexcept {
  return nullptr;
}
#endif

template <class T>
bool invalid_span(std::span<T> value) noexcept {
  return value.data() == nullptr && !value.empty();
}

HYLE_FORCE_INLINE UtfResult validate_utf8_core(
    StringView input, const UtfProvider* provider) noexcept {
  if (input.data() == nullptr) {
    return input.empty()
               ? UtfResult{}
               : UtfResult{UtfStatus::kInvalidArgument, 0, 0};
  }
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  return validation_result(
      provider->validate_utf8_with_errors(input.data(), input.size()));
#else
  (void)provider;
  Utf8Decoder decoder(input);
  while (!decoder.is_end()) {
    decoder.next();
    if (!decoder.valid()) return {UtfStatus::kOther, input.size(), 0};
  }
  return {UtfStatus::kSuccess, input.size(), 0};
#endif
}

HYLE_FORCE_INLINE UtfResult validate_utf16_core(
    std::span<const char16_t> input, const UtfProvider* provider) noexcept {
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  return validation_result(
      provider_validate_utf16(provider, input.data(), input.size()));
#else
  (void)provider;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const char16_t value = input[i];
    if (value >= 0xD800 && value <= 0xDBFF) {
      if (++i == input.size() || input[i] < 0xDC00 || input[i] > 0xDFFF)
        return {UtfStatus::kSurrogate, i - 1, 0};
    } else if (value >= 0xDC00 && value <= 0xDFFF) {
      return {UtfStatus::kSurrogate, i, 0};
    }
  }
  return {UtfStatus::kSuccess, input.size(), 0};
#endif
}

HYLE_FORCE_INLINE UtfResult validate_utf32_core(
    std::span<const char32_t> input, const UtfProvider* provider) noexcept {
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  return validation_result(
      provider->validate_utf32_with_errors(input.data(), input.size()));
#else
  (void)provider;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const char32_t value = input[i];
    if (value > 0x10FFFF) return {UtfStatus::kTooLarge, i, 0};
    if (value >= 0xD800 && value <= 0xDFFF)
      return {UtfStatus::kSurrogate, i, 0};
  }
  return {UtfStatus::kSuccess, input.size(), 0};
#endif
}

}  // namespace

UtfResult utf8_validate_detailed(StringView input) noexcept {
  return validate_utf8_core(input, active_utf_provider());
}

UtfResult utf16_validate(std::span<const char16_t> input) noexcept {
  return validate_utf16_core(input, active_utf_provider());
}

UtfResult utf32_validate(std::span<const char32_t> input) noexcept {
  return validate_utf32_core(input, active_utf_provider());
}

UtfResult utf8_to_utf16(StringView input,
                        std::span<char16_t> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (input.data() == nullptr) {
    return input.empty()
               ? UtfResult{}
               : UtfResult{UtfStatus::kInvalidArgument, 0, 0};
  }
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider->validate_utf8_with_errors(input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider->utf16_length_from_utf8(input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider_convert_utf8_to_utf16(
        provider, input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf8_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

UtfResult utf8_to_utf32(StringView input,
                        std::span<char32_t> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (input.data() == nullptr) {
    return input.empty()
               ? UtfResult{}
               : UtfResult{UtfStatus::kInvalidArgument, 0, 0};
  }
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider->validate_utf8_with_errors(input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider->utf32_length_from_utf8(input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider->convert_valid_utf8_to_utf32(
        input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf8_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

UtfResult utf16_to_utf8(std::span<const char16_t> input,
                        std::span<char> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
  if (input.empty()) return {};
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider_validate_utf16(provider, input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider_utf8_length_from_utf16(provider, input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider_convert_utf16_to_utf8(
        provider, input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf16_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

UtfResult utf16_to_utf32(std::span<const char16_t> input,
                         std::span<char32_t> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
  if (input.empty()) return {};
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider_validate_utf16(provider, input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider_utf32_length_from_utf16(provider, input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider_convert_utf16_to_utf32(
        provider, input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf16_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

UtfResult utf32_to_utf8(std::span<const char32_t> input,
                        std::span<char> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
  if (input.empty()) return {};
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider->validate_utf32_with_errors(input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider->utf8_length_from_utf32(input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider->convert_valid_utf32_to_utf8(
        input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf32_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

UtfResult utf32_to_utf16(std::span<const char32_t> input,
                         std::span<char16_t> output) noexcept {
  if (invalid_span(output)) return {UtfStatus::kInvalidArgument, 0, 0};
#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  if (invalid_span(input)) return {UtfStatus::kInvalidArgument, 0, 0};
  if (input.empty()) return {};
  const UtfProvider* provider = active_utf_provider();
  const simdutf::result validation =
      provider->validate_utf32_with_errors(input.data(), input.size());
  if (validation.error != simdutf::SUCCESS)
    return validation_result(validation);
  const std::size_t required =
      provider->utf16_length_from_utf32(input.data(), input.size());
  if (output.size() < required)
    return {UtfStatus::kOutputTooSmall, input.size(), required};
  std::size_t produced = 0;
  if (required != 0) {
    produced = provider_convert_utf32_to_utf16(
        provider, input.data(), input.size(), output.data());
  }
  return {UtfStatus::kSuccess, input.size(), produced};
#else
  UtfResult validation = validate_utf32_core(input, active_utf_provider());
  if (!validation) return validation;
  (void)output;
  return {UtfStatus::kOther, 0, 0};
#endif
}

}  // namespace hyle::container

namespace hyle::container {
namespace {

constexpr bool is_base64_ascii_whitespace(char value) noexcept {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
         value == '\f';
}

constexpr int base64_value(char value, Base64Alphabet alphabet) noexcept {
  const auto byte = static_cast<unsigned char>(value);
  if (byte >= static_cast<unsigned char>('A') &&
      byte <= static_cast<unsigned char>('Z')) {
    return byte - static_cast<unsigned char>('A');
  }
  if (byte >= static_cast<unsigned char>('a') &&
      byte <= static_cast<unsigned char>('z')) {
    return 26 + byte - static_cast<unsigned char>('a');
  }
  if (byte >= static_cast<unsigned char>('0') &&
      byte <= static_cast<unsigned char>('9')) {
    return 52 + byte - static_cast<unsigned char>('0');
  }
  if (alphabet == Base64Alphabet::kStandard) {
    if (value == '+') return 62;
    if (value == '/') return 63;
  } else {
    if (value == '-') return 62;
    if (value == '_') return 63;
  }
  return -1;
}

template <class T>
bool invalid_base64_span(std::span<T> value) noexcept {
  return value.data() == nullptr && !value.empty();
}

bool byte_ranges_overlap(const void* first, std::size_t first_size,
                         const void* second,
                         std::size_t second_size) noexcept {
  if (first_size == 0 || second_size == 0) return false;
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
  if (first_size > kMax - first_begin || second_size > kMax - second_begin) {
    return true;
  }
  const auto first_end = first_begin + first_size;
  const auto second_end = second_begin + second_size;
  return first_begin < second_end && second_begin < first_end;
}

Base64Result base64_fast_policy_check(
    std::span<const char> input, Base64DecodeOptions options) noexcept {
  // Keep the canonical default route free of the tail/alphabet bookkeeping
  // needed by the two unpadded policies.  Besides being the normal protocol
  // shape, this gives the compiler a simple whitespace-membership loop that it
  // can lower independently of the policy matrix below.
  if (options.padding == Base64DecodePadding::kRequired) {
    if (options.whitespace == Base64Whitespace::kAllowAscii) return {};
    for (std::size_t i = 0; i < input.size(); ++i) {
      if (is_base64_ascii_whitespace(input[i])) {
        return {Base64Status::kWhitespaceRejected, i, 0};
      }
    }
    return {};
  }

  std::size_t data_count = 0;
  std::size_t last_data_offset = 0;
  int last_value = 0;
  bool saw_invalid_character = false;

  for (std::size_t i = 0; i < input.size(); ++i) {
    const char value = input[i];
    if (is_base64_ascii_whitespace(value)) {
      if (options.whitespace == Base64Whitespace::kReject) {
        return {Base64Status::kWhitespaceRejected, i, 0};
      }
      continue;
    }
    if (value == '=') {
      if (options.padding == Base64DecodePadding::kForbidden) {
        return {Base64Status::kInvalidPadding, i, 0};
      }
      continue;
    }
    const int decoded = base64_value(value, options.alphabet);
    if (decoded < 0) {
      // Let simdutf return progress for malformed alphabet input. Tail checks
      // are meaningful only after every data symbol was recognized.
      saw_invalid_character = true;
      continue;
    }
    ++data_count;
    last_data_offset = i;
    last_value = decoded;
  }

  if (!saw_invalid_character) {
    const std::size_t remainder = data_count % 4;
    if ((remainder == 2 && (last_value & 0x0f) != 0) ||
        (remainder == 3 && (last_value & 0x03) != 0)) {
      return {Base64Status::kNonCanonicalBits, last_data_offset, 0};
    }
  }
  return {};
}

#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
simdutf::base64_options to_simdutf(Base64EncodeOptions options) noexcept {
  if (options.alphabet == Base64Alphabet::kUrl) {
    return options.padding == Base64EncodePadding::kInclude
               ? simdutf::base64_url_with_padding
               : simdutf::base64_url;
  }
  return options.padding == Base64EncodePadding::kInclude
             ? simdutf::base64_default
             : simdutf::base64_default_no_padding;
}

simdutf::base64_options to_simdutf(Base64DecodeOptions options) noexcept {
  return options.alphabet == Base64Alphabet::kUrl ? simdutf::base64_url
                                                   : simdutf::base64_default;
}

simdutf::last_chunk_handling_options to_simdutf_last_chunk(
    Base64DecodeOptions options) noexcept {
  return options.padding == Base64DecodePadding::kRequired
             ? simdutf::last_chunk_handling_options::strict
             : simdutf::last_chunk_handling_options::loose;
}

Base64Status from_simdutf_base64(simdutf::error_code error) noexcept {
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
#endif

}  // namespace

Base64SizeResult base64_decoded_size(
    std::span<const char> input, Base64DecodeOptions options) noexcept {
  if (!detail::valid_base64_decode_options(options) ||
      invalid_base64_span(input)) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }

  std::size_t data_count = 0;
  std::size_t padding_count = 0;
  std::size_t first_padding = input.size();
  std::size_t last_data_offset = 0;
  int last_value = 0;
  bool saw_padding = false;

  for (std::size_t i = 0; i < input.size(); ++i) {
    const char value = input[i];
    if (is_base64_ascii_whitespace(value)) {
      if (options.whitespace == Base64Whitespace::kReject) {
        return {Base64Status::kWhitespaceRejected, i, 0};
      }
      continue;
    }
    if (value == '=') {
      if (options.padding == Base64DecodePadding::kForbidden) {
        return {Base64Status::kInvalidPadding, i, 0};
      }
      if (!saw_padding) first_padding = i;
      saw_padding = true;
      ++padding_count;
      if (padding_count > 2) {
        return {Base64Status::kInvalidPadding, i, 0};
      }
      continue;
    }
    if (saw_padding) {
      return {Base64Status::kInvalidPadding, i, 0};
    }
    const int decoded = base64_value(value, options.alphabet);
    if (decoded < 0) {
      return {Base64Status::kInvalidCharacter, i, 0};
    }
    ++data_count;
    last_data_offset = i;
    last_value = decoded;
  }

  const std::size_t remainder = data_count % 4;
  if (remainder == 1) {
    return {Base64Status::kInvalidLength, input.size(), 0};
  }
  const std::size_t required_padding =
      remainder == 2 ? 2 : (remainder == 3 ? 1 : 0);

  if (padding_count != 0 && padding_count != required_padding) {
    return {Base64Status::kInvalidPadding, first_padding, 0};
  }
  if (options.padding == Base64DecodePadding::kRequired &&
      padding_count != required_padding) {
    return {Base64Status::kInvalidPadding,
            padding_count == 0 ? input.size() : first_padding, 0};
  }

  if ((remainder == 2 && (last_value & 0x0f) != 0) ||
      (remainder == 3 && (last_value & 0x03) != 0)) {
    return {Base64Status::kNonCanonicalBits, last_data_offset, 0};
  }

  const std::size_t output_size =
      (data_count / 4) * 3 + (remainder == 0 ? 0 : remainder - 1);
  return {Base64Status::kSuccess, input.size(), output_size};
}

Base64Result base64_encode(std::span<const std::byte> input,
                           std::span<char> output,
                           Base64EncodeOptions options) noexcept {
  if (!detail::valid_base64_encode_options(options) ||
      invalid_base64_span(input) || invalid_base64_span(output) ||
      byte_ranges_overlap(input.data(), input.size(), output.data(),
                          output.size())) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }
  const Base64SizeResult required = base64_encoded_size(input.size(), options);
  if (!required) return {required.status, required.input_count, 0};
  if (output.size() < required.output_size) {
    return {Base64Status::kOutputTooSmall, 0, 0};
  }
  if (input.empty()) return {};

#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  const std::size_t written = simdutf::binary_to_base64(
      reinterpret_cast<const char*>(input.data()), input.size(), output.data(),
      to_simdutf(options));
  return {Base64Status::kSuccess, input.size(), written};
#else
  (void)output;
  return {Base64Status::kOther, 0, 0};
#endif
}

Base64Result base64_decode(std::span<const char> input,
                           std::span<std::byte> output,
                           Base64DecodeOptions options) noexcept {
  if (!detail::valid_base64_decode_options(options) ||
      invalid_base64_span(input) || invalid_base64_span(output) ||
      byte_ranges_overlap(input.data(), input.size(), output.data(),
                          output.size())) {
    return {Base64Status::kInvalidArgument, 0, 0};
  }
  if (input.empty()) return {};

  const Base64Result policy = base64_fast_policy_check(input, options);
  if (!policy) return policy;

#if defined(HYLE_CONTAINER_TEXT_SIMDUTF)
  char empty_output = 0;
  char* output_data = output.empty()
                          ? &empty_output
                          : reinterpret_cast<char*>(output.data());
  std::size_t output_count = output.size();
  const simdutf::result result = simdutf::base64_to_binary_safe(
      input.data(), input.size(), output_data, output_count,
      to_simdutf(options), to_simdutf_last_chunk(options), false);
  return {from_simdutf_base64(result.error), result.count, output_count};
#else
  (void)output;
  return {Base64Status::kOther, 0, 0};
#endif
}

}  // namespace hyle::container
