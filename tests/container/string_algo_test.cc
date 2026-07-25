#include "hyle/container/string_algo.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "hyle/container/string_ref.h"

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
  EXPECT_EQ(to_lower("HeLLo"_sr, buf, sizeof(buf)), 5u);
  EXPECT_STREQ(buf, "hello");
  EXPECT_EQ(to_upper("HeLLo"_sr, buf, sizeof(buf)), 5u);
  EXPECT_STREQ(buf, "HELLO");
}

TEST(StringAlgo, CaseConvertTruncates) {
  char buf[4];
  // 5 chars into a 4-byte buffer: writes 3 + NUL, returns full length 5.
  EXPECT_EQ(to_lower("ABCDE"_sr, buf, sizeof(buf)), 5u);
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

TEST(StringAlgo, Capitalize) {
  char a[] = "rect_width";
  capitalize(a);
  EXPECT_STREQ(a, "Rect Width");

  char b[] = "ui_scale";  // "ui" acronym at a word start -> "UI"
  capitalize(b);
  EXPECT_STREQ(b, "UI Scale");

  char c[] = "hello world";
  capitalize(c);
  EXPECT_STREQ(c, "Hello World");
}

TEST(StringAlgo, HexEncode) {
  char buf[16];
  const unsigned char bytes[] = {0x00, 0xAB, 0xFF, 0x10};
  EXPECT_EQ(hex_encode(bytes, 4, buf, sizeof(buf)), 8u);
  EXPECT_STREQ(buf, "00abff10");
}

TEST(StringAlgo, HexDecodeAndValidation) {
  unsigned char bytes[4] = {};
  EXPECT_EQ(hex_decode("00aBff10"_sr, bytes, sizeof(bytes)), 4u);
  EXPECT_EQ(bytes[0], 0x00);
  EXPECT_EQ(bytes[1], 0xAB);
  EXPECT_EQ(bytes[2], 0xFF);
  EXPECT_EQ(bytes[3], 0x10);
  EXPECT_EQ(hex_decode("abc"_sr, bytes, sizeof(bytes)), SIZE_MAX);
  EXPECT_EQ(hex_decode("zz"_sr, bytes, sizeof(bytes)), SIZE_MAX);
}

TEST(StringAlgo, Escape) {
  char buf[32];
  std::size_t n = escape("a\tb\nc\\d"_sr, buf, sizeof(buf));
  EXPECT_STREQ(buf, "a\\tb\\nc\\\\d");
  EXPECT_EQ(n, 10u);  // a \t b \n c \\ d = 1+2+1+2+1+2+1
}

TEST(StringAlgo, EscapeReportsTruncation) {
  char buf[4];
  // Full escaped length reported even when the buffer can't hold it.
  std::size_t n = escape("\n\n\n"_sr, buf, sizeof(buf));
  EXPECT_EQ(n, 6u);  // 3 * "\n" = 6
  EXPECT_LT(std::strlen(buf), 4u);
}

TEST(StringAlgo, Utf8LengthAscii) {
  EXPECT_EQ(utf8_length("hello"_sr), 5);
  EXPECT_EQ(utf8_length(""_sr), 0);
}

TEST(StringAlgo, Utf8LengthMultibyte) {
  // "héllo": é is 2 bytes (U+00E9 -> 0xC3 0xA9) -> 5 code points, 6 bytes.
  string_ref s("h\xC3\xA9llo");
  EXPECT_EQ(s.size, 6u);
  EXPECT_EQ(utf8_length(s), 5);

  // "你好" = 2 code points, 6 bytes (each CJK char is 3 bytes).
  string_ref cjk("\xE4\xBD\xA0\xE5\xA5\xBD");
  EXPECT_EQ(cjk.size, 6u);
  EXPECT_EQ(utf8_length(cjk), 2);
}

TEST(StringAlgo, Utf8LengthInvalid) {
  // A lone continuation byte 0x80 is not valid UTF-8.
  string_ref bad("\x80");
  EXPECT_EQ(utf8_length(bad), -1);
  // Truncated 2-byte sequence (0xC3 with no continuation).
  string_ref truncated("\xC3");
  EXPECT_EQ(utf8_length(truncated), -1);
  const char overlong[] = {static_cast<char>(0xC0), static_cast<char>(0xAF)};
  EXPECT_EQ(utf8_length(string_ref{overlong, 2}), -1);
  const char surrogate[] = {static_cast<char>(0xED), static_cast<char>(0xA0),
                            static_cast<char>(0x80)};
  EXPECT_EQ(utf8_length(string_ref{surrogate, 3}), -1);
  const char too_large[] = {static_cast<char>(0xF4), static_cast<char>(0x90),
                            static_cast<char>(0x80), static_cast<char>(0x80)};
  EXPECT_EQ(utf8_length(string_ref{too_large, 4}), -1);
}

TEST(StringAlgo, ScalarUtf8DecoderRejectsNonUnicodeSequences) {
  const char overlong[] = {static_cast<char>(0xC0), static_cast<char>(0xAF)};
  Utf8Decoder a(string_ref{overlong, 2});
  EXPECT_EQ(a.next(), 0);
  EXPECT_FALSE(a.valid());

  const char surrogate[] = {static_cast<char>(0xED), static_cast<char>(0xA0),
                            static_cast<char>(0x80)};
  Utf8Decoder b(string_ref{surrogate, 3});
  EXPECT_EQ(b.next(), 0);
  EXPECT_FALSE(b.valid());

  const char maximum[] = {static_cast<char>(0xF4), static_cast<char>(0x8F),
                          static_cast<char>(0xBF), static_cast<char>(0xBF)};
  Utf8Decoder c(string_ref{maximum, 4});
  EXPECT_EQ(c.next(), 0x10FFFF);
  EXPECT_TRUE(c.valid());
}

TEST(StringAlgo, Utf8IcmpCaseFold) {
  EXPECT_EQ(utf8_icmp("Hello"_sr, "hello"_sr), 0);   // ASCII fold
  EXPECT_LT(utf8_icmp("abc"_sr, "abd"_sr), 0);
  EXPECT_GT(utf8_icmp("abd"_sr, "abc"_sr), 0);
  EXPECT_EQ(utf8_icmp("abcXX"_sr, "abcYY"_sr, 3), 0);  // limit to 3 cp
  EXPECT_LT(utf8_icmp("ab"_sr, "abc"_sr), 0);          // prefix is "less"
}
