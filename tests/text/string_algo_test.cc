#include "hyle/container/string_algo.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>

#include "hyle/container/string_view.h"

using namespace hyle::container;
using namespace hyle::container::literals;

TEST(StringAlgo, Classifiers) {
  EXPECT_TRUE(is_ascii_space(' '));
  EXPECT_TRUE(is_ascii_space('\t'));
  EXPECT_FALSE(is_ascii_space('x'));
  EXPECT_TRUE(is_ascii_digit('7'));
  EXPECT_FALSE(is_ascii_digit('a'));
  EXPECT_TRUE(is_ascii_print('A'));
  EXPECT_FALSE(is_ascii_print('\n'));
  EXPECT_TRUE(is_hex_digit('f'));
  EXPECT_TRUE(is_hex_digit('F'));
  EXPECT_FALSE(is_hex_digit('g'));
}

// clay's hex_to_int returned 0 for invalid input (hiding bad data); ours -1.
TEST(StringAlgo, HexValueSignalsInvalid) {
  EXPECT_EQ(hex_value('0'), 0);
  EXPECT_EQ(hex_value('9'), 9);
  EXPECT_EQ(hex_value('a'), 10);
  EXPECT_EQ(hex_value('F'), 15);
  EXPECT_EQ(hex_value('g'), -1);  // not 0 -- a real signal
  EXPECT_EQ(hex_value('\0'), -1);
}

TEST(StringAlgo, CaseConvertBufferFill) {
  char buf[16];
  EXPECT_EQ(to_lower("HeLLo"_sv, buf, sizeof(buf)), 5u);
  EXPECT_STREQ(buf, "hello");
  EXPECT_EQ(to_upper("HeLLo"_sv, buf, sizeof(buf)), 5u);
  EXPECT_STREQ(buf, "HELLO");
}

TEST(StringAlgo, CaseConvertTruncates) {
  char buf[4];
  // 5 chars into a 4-byte buffer: writes 3 + NUL, returns full length 5.
  EXPECT_EQ(to_lower("ABCDE"_sv, buf, sizeof(buf)), 5u);
  EXPECT_STREQ(buf, "abc");
}

TEST(StringAlgo, CaseConvertLongAndInPlace) {
  constexpr std::size_t kSize = 257;
  std::array<char, kSize + 1> input{};
  std::array<char, kSize + 1> output{};
  for (std::size_t i = 0; i < kSize; ++i)
    input[i] = (i % 3 == 0) ? 'A' : ((i % 3 == 1) ? 'z' : '-');

  EXPECT_EQ(to_lower({input.data(), kSize}, output.data(),
                     static_cast<std::uint32_t>(output.size())),
            kSize);
  for (std::size_t i = 0; i < kSize; ++i)
    EXPECT_EQ(output[i], (i % 3 == 0) ? 'a' : ((i % 3 == 1) ? 'z' : '-'));
  EXPECT_EQ(output[kSize], '\0');

  EXPECT_EQ(to_upper({output.data(), kSize}, output.data(),
                     static_cast<std::uint32_t>(output.size())),
            kSize);
  for (std::size_t i = 0; i < kSize; ++i)
    EXPECT_EQ(output[i], (i % 3 == 0) ? 'A' : ((i % 3 == 1) ? 'Z' : '-'));
  EXPECT_EQ(output[kSize], '\0');
}

TEST(StringAlgo, HexEncode) {
  char buf[16];
  const unsigned char bytes[] = {0x00, 0xAB, 0xFF, 0x10};
  EXPECT_EQ(hex_encode(bytes, 4, buf, sizeof(buf)), 8u);
  EXPECT_STREQ(buf, "00abff10");
}

TEST(StringAlgo, HexDecodeAndValidation) {
  unsigned char bytes[4] = {};
  EXPECT_EQ(hex_decode("00aBff10"_sv, bytes, sizeof(bytes)), 4u);
  EXPECT_EQ(bytes[0], 0x00);
  EXPECT_EQ(bytes[1], 0xAB);
  EXPECT_EQ(bytes[2], 0xFF);
  EXPECT_EQ(bytes[3], 0x10);
  EXPECT_EQ(hex_decode("abc"_sv, bytes, sizeof(bytes)), SIZE_MAX);
  EXPECT_EQ(hex_decode("zz"_sv, bytes, sizeof(bytes)), SIZE_MAX);
}

TEST(StringAlgo, Escape) {
  char buf[32];
  std::size_t n = c_escape("a\tb\nc\\d"_sv, buf, sizeof(buf));
  EXPECT_STREQ(buf, "a\\tb\\nc\\\\d");
  EXPECT_EQ(n, 10u);  // a \t b \n c \\ d = 1+2+1+2+1+2+1

  const char binary[] = {'A', '\0', '\x1f', static_cast<char>(0xff)};
  n = c_escape({binary, sizeof(binary)}, buf, sizeof(buf));
  EXPECT_EQ(n, 13u);
  EXPECT_STREQ(buf, "A\\000\\037\\377");
}

TEST(StringAlgo, EscapeReportsTruncation) {
  char buf[4];
  // Full escaped length reported even when the buffer can't hold it.
  std::size_t n = c_escape("\n\n\n"_sv, buf, sizeof(buf));
  EXPECT_EQ(n, 6u);  // 3 * "\n" = 6
  EXPECT_STREQ(buf, "\\n");

  char tiny[2];
  n = c_escape("\nA"_sv, tiny, sizeof(tiny));
  EXPECT_EQ(n, 3u);
  EXPECT_STREQ(tiny, "");  // truncation is a prefix; it never skips a token
}

TEST(StringAlgo, Utf8LengthAscii) {
  EXPECT_TRUE(utf8_validate("hello"_sv));
  EXPECT_TRUE(utf8_validate(""_sv));
  EXPECT_EQ(utf8_length("hello"_sv), 5);
  EXPECT_EQ(utf8_length(""_sv), 0);
}

TEST(StringAlgo, Utf8RuntimeRejectsNullNonemptyView) {
  const StringView empty_null{nullptr, 0};
  EXPECT_TRUE(utf8_validate(empty_null));
  EXPECT_EQ(utf8_length(empty_null), 0);

  const StringView invalid_null{nullptr, 1};
  EXPECT_FALSE(utf8_validate(invalid_null));
  EXPECT_EQ(utf8_length(invalid_null), -1);
}

TEST(StringAlgo, Utf8DecoderRejectsNullNonemptyView) {
  Utf8Decoder empty(StringView{nullptr, 0});
  EXPECT_TRUE(empty.is_end());
  EXPECT_TRUE(empty.valid());
  EXPECT_EQ(empty.next(), 0);

  Utf8Decoder invalid(StringView{nullptr, 1});
  EXPECT_TRUE(invalid.is_end());
  EXPECT_FALSE(invalid.valid());
  EXPECT_EQ(invalid.next(), 0);
  EXPECT_FALSE(invalid.valid());
}

TEST(StringAlgo, Utf8LengthMultibyte) {
  // "héllo": é is 2 bytes (U+00E9 -> 0xC3 0xA9) -> 5 code points, 6 bytes.
  StringView s("h\xC3\xA9llo");
  EXPECT_EQ(s.size(), 6u);
  EXPECT_TRUE(utf8_validate(s));
  EXPECT_EQ(utf8_length(s), 5);

  // "你好" = 2 code points, 6 bytes (each CJK char is 3 bytes).
  StringView cjk("\xE4\xBD\xA0\xE5\xA5\xBD");
  EXPECT_EQ(cjk.size(), 6u);
  EXPECT_TRUE(utf8_validate(cjk));
  EXPECT_EQ(utf8_length(cjk), 2);
}

TEST(StringAlgo, Utf8LengthInvalid) {
  // A lone continuation byte 0x80 is not valid UTF-8.
  StringView bad("\x80");
  EXPECT_FALSE(utf8_validate(bad));
  EXPECT_EQ(utf8_length(bad), -1);
  // Truncated 2-byte sequence (0xC3 with no continuation).
  StringView truncated("\xC3");
  EXPECT_FALSE(utf8_validate(truncated));
  EXPECT_EQ(utf8_length(truncated), -1);
  const char overlong[] = {static_cast<char>(0xC0), static_cast<char>(0xAF)};
  EXPECT_FALSE(utf8_validate(StringView{overlong, 2}));
  EXPECT_EQ(utf8_length(StringView{overlong, 2}), -1);
  const char surrogate[] = {static_cast<char>(0xED), static_cast<char>(0xA0),
                            static_cast<char>(0x80)};
  EXPECT_FALSE(utf8_validate(StringView{surrogate, 3}));
  EXPECT_EQ(utf8_length(StringView{surrogate, 3}), -1);
  const char too_large[] = {static_cast<char>(0xF4), static_cast<char>(0x90),
                            static_cast<char>(0x80), static_cast<char>(0x80)};
  EXPECT_FALSE(utf8_validate(StringView{too_large, 4}));
  EXPECT_EQ(utf8_length(StringView{too_large, 4}), -1);
}

TEST(StringAlgo, Utf8LengthDispatchedBackendHandlesBlockTails) {
  // Exercise the product's out-of-line simdutf runtime dispatcher at every
  // byte tail around several SIMD blocks. Embedded NUL is a valid U+0000 in an
  // explicitly-sized view and must be counted rather than ending the scan.
  for (std::size_t size = 0; size <= 257; ++size) {
    std::string ascii(size, 'a');
    if (size != 0) ascii[size / 2] = '\0';
    EXPECT_TRUE(utf8_validate(StringView{ascii}))
        << "ASCII tail size=" << size;
    EXPECT_EQ(utf8_length(StringView{ascii}),
              static_cast<std::int64_t>(size))
        << "ASCII tail size=" << size;
  }

  const char suffix[] = {
      static_cast<char>(0xC3), static_cast<char>(0xA9),  // U+00E9
      static_cast<char>(0xE4), static_cast<char>(0xB8),
      static_cast<char>(0xAD),                          // U+4E2D
      static_cast<char>(0xF0), static_cast<char>(0x9F),
      static_cast<char>(0x98), static_cast<char>(0x80)  // U+1F600
  };
  for (std::size_t prefix = 0; prefix <= 257; ++prefix) {
    std::string mixed(prefix, 'x');
    mixed.append(suffix, sizeof(suffix));
    EXPECT_TRUE(utf8_validate(StringView{mixed}))
        << "multibyte tail prefix=" << prefix;
    EXPECT_EQ(utf8_length(StringView{mixed}),
              static_cast<std::int64_t>(prefix + 3))
        << "multibyte tail prefix=" << prefix;
  }
}

TEST(StringAlgo, Utf8LengthDispatchedBackendRejectsTruncatedTails) {
  const char truncated_three[] = {static_cast<char>(0xE2),
                                  static_cast<char>(0x82)};
  const char truncated_four[] = {
      static_cast<char>(0xF0), static_cast<char>(0x9F),
      static_cast<char>(0x98)};
  const char bad_continuation[] = {static_cast<char>(0xE2), 'A',
                                   static_cast<char>(0x82)};

  for (std::size_t prefix = 0; prefix <= 257; ++prefix) {
    std::string input(prefix, 'x');
    input.append(truncated_three, sizeof(truncated_three));
    EXPECT_FALSE(utf8_validate(StringView{input}))
        << "truncated 3-byte tail prefix=" << prefix;
    EXPECT_EQ(utf8_length(StringView{input}), -1)
        << "truncated 3-byte tail prefix=" << prefix;

    input.assign(prefix, 'x');
    input.append(truncated_four, sizeof(truncated_four));
    EXPECT_FALSE(utf8_validate(StringView{input}))
        << "truncated 4-byte tail prefix=" << prefix;
    EXPECT_EQ(utf8_length(StringView{input}), -1)
        << "truncated 4-byte tail prefix=" << prefix;

    input.assign(prefix, 'x');
    input.append(bad_continuation, sizeof(bad_continuation));
    EXPECT_FALSE(utf8_validate(StringView{input}))
        << "bad continuation tail prefix=" << prefix;
    EXPECT_EQ(utf8_length(StringView{input}), -1)
        << "bad continuation tail prefix=" << prefix;
  }
}

TEST(StringAlgo, ScalarUtf8DecoderRejectsNonUnicodeSequences) {
  const char overlong[] = {static_cast<char>(0xC0), static_cast<char>(0xAF)};
  Utf8Decoder a(StringView{overlong, 2});
  EXPECT_EQ(a.next(), 0);
  EXPECT_FALSE(a.valid());

  const char surrogate[] = {static_cast<char>(0xED), static_cast<char>(0xA0),
                            static_cast<char>(0x80)};
  Utf8Decoder b(StringView{surrogate, 3});
  EXPECT_EQ(b.next(), 0);
  EXPECT_FALSE(b.valid());

  const char maximum[] = {static_cast<char>(0xF4), static_cast<char>(0x8F),
                          static_cast<char>(0xBF), static_cast<char>(0xBF)};
  Utf8Decoder c(StringView{maximum, 4});
  EXPECT_EQ(c.next(), 0x10FFFF);
  EXPECT_TRUE(c.valid());
}

TEST(StringAlgo, UtfDetailedValidationReportsTypedOffsets) {
  const char truncated[] = {static_cast<char>(0xE2),
                            static_cast<char>(0x82)};
  const UtfResult utf8 = utf8_validate_detailed({truncated, 2});
  EXPECT_EQ(utf8.status, UtfStatus::kTooShort);
  EXPECT_EQ(utf8.input_offset, 0u);

  const std::array<char16_t, 2> bad16 = {u'a', 0xD800};
  const UtfResult utf16 = utf16_validate(bad16);
  EXPECT_EQ(utf16.status, UtfStatus::kSurrogate);
  EXPECT_EQ(utf16.input_offset, 1u);

  const std::array<char32_t, 2> bad32 = {U'a', 0x110000};
  const UtfResult utf32 = utf32_validate(bad32);
  EXPECT_EQ(utf32.status, UtfStatus::kTooLarge);
  EXPECT_EQ(utf32.input_offset, 1u);
}

TEST(StringAlgo, UtfCallerBufferConversionsCoverEveryPair) {
  const char utf8_bytes[] = {'A', static_cast<char>(0xC3),
                             static_cast<char>(0xA9),
                             static_cast<char>(0xF0),
                             static_cast<char>(0x9F),
                             static_cast<char>(0x98),
                             static_cast<char>(0x80)};
  const StringView utf8{utf8_bytes, sizeof(utf8_bytes)};
  const std::array<char16_t, 4> expected16 = {u'A', 0x00E9, 0xD83D,
                                              0xDE00};
  const std::array<char32_t, 3> expected32 = {U'A', U'\u00E9', 0x1F600};

  std::array<char16_t, 4> as16{};
  UtfResult result = utf8_to_utf16(utf8, as16);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.input_offset, utf8.size());
  EXPECT_EQ(result.output_count, expected16.size());
  EXPECT_EQ(as16, expected16);

  std::array<char32_t, 3> as32{};
  result = utf8_to_utf32(utf8, as32);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.output_count, expected32.size());
  EXPECT_EQ(as32, expected32);

  std::array<char, sizeof(utf8_bytes)> from16{};
  result = utf16_to_utf8(expected16, from16);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.output_count, sizeof(utf8_bytes));
  EXPECT_TRUE(std::equal(from16.begin(), from16.end(),
                         std::begin(utf8_bytes)));

  std::array<char32_t, 3> from16_as32{};
  result = utf16_to_utf32(expected16, from16_as32);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(from16_as32, expected32);

  std::array<char, sizeof(utf8_bytes)> from32{};
  result = utf32_to_utf8(expected32, from32);
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(std::equal(from32.begin(), from32.end(),
                         std::begin(utf8_bytes)));

  std::array<char16_t, 4> from32_as16{};
  result = utf32_to_utf16(expected32, from32_as16);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(from32_as16, expected16);
}

TEST(StringAlgo, UtfSizeQueryAndFailureDoNotPartiallyWrite) {
  const StringView input{"A\xF0\x9F\x98\x80"};
  const UtfResult query = utf8_to_utf16(input, {});
  EXPECT_EQ(query.status, UtfStatus::kOutputTooSmall);
  EXPECT_EQ(query.input_offset, input.size());
  EXPECT_EQ(query.output_count, 3u);

  std::array<char16_t, 2> too_small = {u'x', u'y'};
  const UtfResult short_result = utf8_to_utf16(input, too_small);
  EXPECT_EQ(short_result.status, UtfStatus::kOutputTooSmall);
  EXPECT_EQ(short_result.output_count, 3u);
  EXPECT_EQ(too_small[0], u'x');
  EXPECT_EQ(too_small[1], u'y');

  const char malformed[] = {'A', static_cast<char>(0x80)};
  std::array<char32_t, 4> untouched = {U'x', U'y', U'z', U'w'};
  const UtfResult invalid = utf8_to_utf32({malformed, 2}, untouched);
  EXPECT_FALSE(invalid.ok());
  EXPECT_NE(invalid.status, UtfStatus::kOutputTooSmall);
  EXPECT_EQ(untouched[0], U'x');
  EXPECT_EQ(untouched[3], U'w');

  EXPECT_TRUE(utf8_to_utf16(StringView{}, {}).ok());
  EXPECT_TRUE(utf16_to_utf8({}, {}).ok());
  EXPECT_TRUE(utf32_to_utf16({}, {}).ok());
}
