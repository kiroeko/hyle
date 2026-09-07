#include "hyle/container/string_algo.h"

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
