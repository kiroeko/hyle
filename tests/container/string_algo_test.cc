#include "hyle/container/string_algo.h"

#include <gtest/gtest.h>

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
}

TEST(StringAlgo, Utf8IcmpCaseFold) {
  EXPECT_EQ(utf8_icmp("Hello"_sr, "hello"_sr), 0);   // ASCII fold
  EXPECT_LT(utf8_icmp("abc"_sr, "abd"_sr), 0);
  EXPECT_GT(utf8_icmp("abd"_sr, "abc"_sr), 0);
  EXPECT_EQ(utf8_icmp("abcXX"_sr, "abcYY"_sr, 3), 0);  // limit to 3 cp
  EXPECT_LT(utf8_icmp("ab"_sr, "abc"_sr), 0);          // prefix is "less"
}

// ---- container-06c fuzzy path matching -------------------------------------

// A pattern that is an ordered subsequence matches (>0); one that isn't scores 0.
TEST(FuzzyMatch, SubsequenceRequired) {
  EXPECT_GT(fuzzy_match_score("config.h"_sr, "cfg"_sr), 0);   // c..f..g in order
  EXPECT_GT(fuzzy_match_score("config.h"_sr, "cong"_sr), 0);
  EXPECT_EQ(fuzzy_match_score("config.h"_sr, "gfc"_sr), 0);   // wrong order
  EXPECT_EQ(fuzzy_match_score("config.h"_sr, "cfgx"_sr), 0);  // 'x' absent
  EXPECT_EQ(fuzzy_match_score("abc"_sr, "abcd"_sr), 0);       // pattern longer
}

// Empty pattern scores 0 (no discriminating info); empty candidate never matches a
// non-empty pattern.
TEST(FuzzyMatch, EmptyEdges) {
  EXPECT_EQ(fuzzy_match_score("anything"_sr, ""_sr), 0);
  EXPECT_EQ(fuzzy_match_score(""_sr, "x"_sr), 0);
  EXPECT_EQ(fuzzy_match_score(""_sr, ""_sr), 0);
}

// Case-insensitive over ASCII.
TEST(FuzzyMatch, CaseInsensitive) {
  EXPECT_GT(fuzzy_match_score("Config.H"_sr, "cfg"_sr), 0);
  EXPECT_EQ(fuzzy_match_score("Config.H"_sr, "cfg"_sr),
            fuzzy_match_score("config.h"_sr, "CFG"_sr));
}

// Consecutive matches beat scattered ones (both are subsequences of the candidate).
TEST(FuzzyMatch, ConsecutiveBeatsScattered) {
  // "conf" is 4 consecutive chars; "cnfg" is scattered across "config".
  int consec = fuzzy_match_score("config"_sr, "conf"_sr);
  int scattered = fuzzy_match_score("config"_sr, "cnfg"_sr);
  EXPECT_GT(consec, scattered);
}

// A match at a word boundary (after a separator) beats a mid-word match.
TEST(FuzzyMatch, BoundaryBeatsMidWord) {
  // 'm' at the start of the "main" segment (after '/') vs 'm' mid-word in "domain".
  int boundary = fuzzy_match_score("src/main.c"_sr, "m"_sr);
  int midword = fuzzy_match_score("domainx.c"_sr, "m"_sr);
  EXPECT_GT(boundary, midword);
}

// path_match_score weights the filename part: the same pattern scores higher when it
// hits the filename than when it hits a directory component.
TEST(FuzzyMatch, FilenamePartWeighted) {
  // "cfg" fully in the filename vs "cfg" spread in a directory name.
  int in_filename = path_match_score("src/config.h"_sr, "cfg"_sr);
  int in_dir = path_match_score("config/main.cpp"_sr, "cfg"_sr);
  EXPECT_GT(in_filename, in_dir);
}

// find_path_matches returns the best matches best-first, skips non-matches, and
// respects the k cap.
TEST(FuzzyMatch, FindPathMatchesTopK) {
  const string_ref cands[] = {
      "src/config.h"_sr,      // matches: cfg (c,f,g present in order)
      "config/readme.md"_sr,  // matches: cfg in the dir component
      "unrelated.txt"_sr,     // no match (no 'c'...'f'...'g' subsequence)
      "cfg.ini"_sr,           // strongest: cfg consecutive at filename start
  };
  constexpr std::uint32_t n = sizeof(cands) / sizeof(cands[0]);
  std::uint32_t idx[3];
  int score[3];
  std::uint32_t m = find_path_matches(cands, n, "cfg"_sr, idx, score, 3);
  ASSERT_EQ(m, 3u);  // three of the four are subsequence matches
  // Best-first: scores must be non-increasing.
  for (std::uint32_t i = 1; i < m; ++i) EXPECT_GE(score[i - 1], score[i]);
  // The consecutive-at-filename-start "cfg.ini" should rank first.
  EXPECT_EQ(idx[0], 3u);
  // Non-matching "unrelated.txt" (index 2) must never appear.
  for (std::uint32_t i = 0; i < m; ++i) EXPECT_NE(idx[i], 2u);
}

// find_path_matches with more matches than k keeps only the top k.
TEST(FuzzyMatch, FindPathMatchesCap) {
  const string_ref cands[] = {"aXbc"_sr, "abXc"_sr, "abcX"_sr, "Xabc"_sr, "abc"_sr};
  constexpr std::uint32_t n = sizeof(cands) / sizeof(cands[0]);
  std::uint32_t idx[2];
  int score[2];
  std::uint32_t m = find_path_matches(cands, n, "abc"_sr, idx, score, 2, /*path_aware=*/false);
  EXPECT_EQ(m, 2u);  // capped at k=2 though all 5 match
  EXPECT_GE(score[0], score[1]);
  EXPECT_EQ(idx[0], 4u);  // "abc" (pure consecutive from start) is the best
}

// Ties break toward the lower index (stable ordering).
TEST(FuzzyMatch, TieBreakLowerIndex) {
  const string_ref cands[] = {"abc"_sr, "abc"_sr, "abc"_sr};  // identical -> equal score
  std::uint32_t idx[3];
  int score[3];
  std::uint32_t m = find_path_matches(cands, 3, "abc"_sr, idx, score, 3, false);
  ASSERT_EQ(m, 3u);
  EXPECT_EQ(idx[0], 0u);
  EXPECT_EQ(idx[1], 1u);
  EXPECT_EQ(idx[2], 2u);
}
