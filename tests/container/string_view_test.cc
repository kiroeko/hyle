#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "hyle/container/string_view.h"
#include "hyle/container/swiss_table.h"

using hyle::container::equal_ignore_case;
using hyle::container::rsplit;
using hyle::container::slice;
using hyle::container::split;
using hyle::container::StringView;
using hyle::container::SwissTable;
using namespace hyle::container::literals;

TEST(StringView, DefaultIsEmpty) {
  StringView s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);
  EXPECT_EQ(s.data(), nullptr);
}

TEST(StringView, DefaultRangeAndSubstrStayNullSafe) {
  StringView s;
  EXPECT_EQ(s.begin(), nullptr);
  EXPECT_EQ(s.end(), nullptr);
  StringView tail = s.substr(0);
  EXPECT_EQ(tail.data(), nullptr);
  EXPECT_EQ(tail.size(), 0u);
}

TEST(StringView, FromCStringIsNullTerminated) {
  StringView s = "hello";
  EXPECT_EQ(s.size(), 5u);
  EXPECT_TRUE(s.is_null_terminated());
  EXPECT_STREQ(s.c_str(), "hello");  // safe: null_terminated bit set
}

TEST(StringView, PtrSizeCtorBitHonored) {
  const char buf[] = {'a', 'b', 'c'};  // NOT null-terminated
  StringView s(buf, 3, false);
  EXPECT_EQ(s.size(), 3u);
  EXPECT_FALSE(s.is_null_terminated());
  EXPECT_EQ(s[0], 'a');
  EXPECT_EQ(s.back(), 'c');
}

TEST(StringView, Layout16Bytes) {
  // The compact view contract: StringView stays a 16-byte trivial value.
  EXPECT_EQ(sizeof(StringView), 16u);
}

TEST(StringView, FromStdTypesWithoutHeaderLeak) {
  std::string owned = "from std::string";
  StringView a = owned;  // generic-range ctor
  EXPECT_EQ(a.size(), owned.size());
  EXPECT_TRUE(a == StringView{"from std::string"});

  std::string_view sv = "from view";
  StringView b = sv;
  EXPECT_EQ(b.size(), sv.size());
  EXPECT_EQ(b.length(), sv.length());
  EXPECT_EQ(b.as_std(), sv);
  EXPECT_EQ(b.cbegin(), b.begin());
  EXPECT_EQ(b.cend(), b.end());
}

TEST(StringView, PrefixSuffixMutationMatchesStdViewShape) {
  StringView value = "prefix-body-suffix";
  value.remove_prefix(7);
  EXPECT_EQ(value.as_std(), "body-suffix");
  EXPECT_TRUE(value.is_null_terminated());
  value.remove_suffix(7);
  EXPECT_EQ(value.as_std(), "body");
  EXPECT_FALSE(value.is_null_terminated());
}

TEST(StringView, FindChar) {
  StringView s = "abracadabra";
  EXPECT_EQ(s.find('a'), 0u);
  EXPECT_EQ(s.find('a', 1), 3u);
  EXPECT_EQ(s.find('z'), StringView::npos);
  EXPECT_EQ(s.rfind('a'), 10u);
  EXPECT_EQ(s.rfind('b'), 8u);
}

TEST(StringView, FindSubstr) {
  StringView s = "the quick brown fox";
  EXPECT_EQ(s.find("quick"), 4u);
  EXPECT_EQ(s.find("fox"), 16u);
  EXPECT_EQ(s.find("cat"), StringView::npos);
  EXPECT_EQ(s.find(""), 0u);          // empty needle matches at pos
  EXPECT_TRUE(s.contains("brown"));
  EXPECT_FALSE(s.contains("BROWN"));
}

TEST(StringView, FindAgainstStdViewOracle) {
  // Differential vs std::string_view (oracle ONLY in tests -- conventions §1).
  std::string_view o = "mississippi delta mississippi";
  StringView s = "mississippi delta mississippi";
  for (char c : {'m', 'i', 's', 'p', ' ', 'z'}) {
    auto exp = o.find(c);
    auto got = s.find(c);
    EXPECT_EQ(got == StringView::npos ? std::string_view::npos : got, exp)
        << "char " << c;
  }
  for (const char* n : {"miss", "issi", "delta", "xyz", "mississippi"}) {
    auto exp = o.find(n);
    auto got = s.find(n);
    EXPECT_EQ(got == StringView::npos ? std::string_view::npos : got, exp)
        << "needle " << n;
  }
}

// The search family was rewritten with RUNTIME fast paths (find/find(needle) ->
// memchr, find_first_of -> memchr-min, find_first_not_of/last_of/last_not_of ->
// 256-entry membership table, region_eq/starts_with/ends_with -> memcmp). This
// differential fuzz proves every runtime path matches std::string_view across
// many haystacks/sets/positions -- the algorithms changed, so they must be
// re-proven, not assumed.
TEST(StringView, SetFamilyAgainstStdViewOracle) {
  const char* haystacks[] = {
      "",         "a",       "aaaa",    "a.b.c",
      "  hi \t",  "abcABC123 .,;",      "the quick brown fox jumps",
      "....",     "x",       "no_delims_here",
  };
  const char* sets[] = {
      "",    ".",      "abc",   "0123",  " \t\r\n",
      "xyz", "aeiou",  ".,;:",  "z",     "abcdefghijklmnopqrstuvwxyz",
  };
  auto map = [](std::uint32_t v) {
    return v == StringView::npos ? std::string_view::npos
                                 : static_cast<std::size_t>(v);
  };
  for (const char* h : haystacks) {
    std::string_view o{h};
    StringView s{h};
    for (const char* setp : sets) {
      std::string_view oset{setp};
      StringView sset{setp};
      // pos = 0 and a couple interior/past-end positions to exercise bounds.
      for (std::size_t pos : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                              o.size(), o.size() + 5}) {
        std::uint32_t sp = pos > StringView::npos
                               ? StringView::npos
                               : static_cast<std::uint32_t>(pos);
        EXPECT_EQ(map(s.find_first_of(sset, sp)), o.find_first_of(oset, pos))
            << "find_first_of h='" << h << "' set='" << setp << "' pos=" << pos;
        EXPECT_EQ(map(s.find_first_not_of(sset, sp)),
                  o.find_first_not_of(oset, pos))
            << "find_first_not_of h='" << h << "' set='" << setp
            << "' pos=" << pos;
      }
      // last_of/last_not_of default-pos (whole string) -- npos pos = scan from end.
      EXPECT_EQ(map(s.find_last_of(sset)), o.find_last_of(oset))
          << "find_last_of h='" << h << "' set='" << setp << "'";
      EXPECT_EQ(map(s.find_last_not_of(sset)), o.find_last_not_of(oset))
          << "find_last_not_of h='" << h << "' set='" << setp << "'";
    }
  }
}

// The constexpr path is a SEPARATE branch (`if consteval` -> scalar; runtime ->
// memchr/memcmp/table). Evaluate every rewritten op in a constant expression so a
// regression in the scalar branch is a COMPILE error, not a silent runtime-only
// pass. static_assert forces constant evaluation.
TEST(StringView, SearchFamilyConstexpr) {
  constexpr StringView s = "a.b.c.d";
  static_assert(s.find('b') == 2u);
  static_assert(s.find('z') == StringView::npos);
  static_assert(s.find(StringView{"b.c"}) == 2u);
  static_assert(s.find(StringView{"zzz"}) == StringView::npos);
  static_assert(s.find_first_of(StringView{".c"}) == 1u);
  static_assert(s.find_first_not_of(StringView{"a."}) == 2u);
  static_assert(s.find_last_of(StringView{"."}) == 5u);
  static_assert(s.find_last_not_of(StringView{"d."}) == 4u);
  static_assert(s.starts_with(StringView{"a.b"}));
  static_assert(!s.starts_with(StringView{"a.c"}));
  static_assert(s.ends_with(StringView{"c.d"}));
  SUCCEED();  // the real assertions are the static_asserts above (compile-time)
}

TEST(StringView, Substr) {
  StringView s = "hello world";
  EXPECT_TRUE(s.substr(0, 5) == StringView{"hello"});
  EXPECT_TRUE(s.substr(6) == StringView{"world"});
  EXPECT_TRUE(s.substr(6).is_null_terminated());   // tail keeps the terminator
  EXPECT_FALSE(s.substr(0, 5).is_null_terminated());  // prefix does not
  EXPECT_TRUE(s.substr(100).empty());          // clamped past end
  EXPECT_TRUE(s.substr(3, 1000) == StringView{"lo world"});  // len clamped
}

TEST(StringView, StartsEndsWith) {
  StringView s = "filename.txt";
  EXPECT_TRUE(s.starts_with("file"));
  EXPECT_FALSE(s.starts_with("File"));
  EXPECT_TRUE(s.ends_with(".txt"));
  EXPECT_FALSE(s.ends_with(".png"));
  EXPECT_TRUE(s.starts_with(""));
  EXPECT_TRUE(s.ends_with(""));
}

TEST(StringView, Trim) {
  EXPECT_TRUE(StringView{"  hi \t\n"}.trim() == StringView{"hi"});
  EXPECT_TRUE(StringView{"xxhixx"}.trim("x") == StringView{"hi"});
  EXPECT_TRUE(StringView{"   "}.trim().empty());  // all-whitespace -> empty
  EXPECT_TRUE(StringView{"none"}.trim() == StringView{"none"});
}

TEST(StringView, FindFirstLastOf) {
  StringView s = "a.b.c";
  EXPECT_EQ(s.find_first_of("."), 1u);
  EXPECT_EQ(s.find_first_not_of("a."), 2u);
  EXPECT_EQ(s.find_last_not_of("c."), 2u);
  EXPECT_EQ(s.find_last_of("."), 3u);     // rightmost '.'
  EXPECT_EQ(s.find_last_of("xyz"), StringView::npos);
}

// find_*_of runtime path is SSE4.2 PCMPESTRI (set<=16), 16-byte blocks + a <16 tail
// + a backward variant. The 5-char case above is tail-only; drive the vectorized loop
// with a >16B buffer and oracle-cross-check every op against the scalar definition.
TEST(StringView, FindOfSimdCrossesBlockAndTail) {
  // 70 bytes: mostly 'a', with '.'/';'/digits sprinkled at known indices.
  std::string b(70, 'a');
  b[5] = '.';
  b[17] = ';';   // just past the first 16B block
  b[33] = '7';
  b[69] = '.';   // last byte
  StringView s{b.data(), static_cast<std::uint32_t>(b.size())};
  const StringView set = ".;";  // <=16 -> SSE4.2 path

  auto oracle_first_of = [&](bool want) -> std::uint32_t {
    for (std::uint32_t i = 0; i < b.size(); ++i) {
      bool in = (b[i] == '.' || b[i] == ';');
      if (in == want) return i;
    }
    return StringView::npos;
  };
  auto oracle_last_of = [&](bool want) -> std::uint32_t {
    for (std::uint32_t i = static_cast<std::uint32_t>(b.size()); i-- > 0;) {
      bool in = (b[i] == '.' || b[i] == ';');
      if (in == want) return i;
    }
    return StringView::npos;
  };
  EXPECT_EQ(s.find_first_of(set), oracle_first_of(true));       // 5
  EXPECT_EQ(s.find_first_of(set), 5u);
  EXPECT_EQ(s.find_first_not_of(set), oracle_first_of(false));  // 0
  EXPECT_EQ(s.find_last_of(set), oracle_last_of(true));         // 69
  EXPECT_EQ(s.find_last_of(set), 69u);
  EXPECT_EQ(s.find_last_not_of(set), oracle_last_of(false));    // 68
  // absent set -> npos both directions (full scan)
  EXPECT_EQ(s.find_first_of("XYZ"), StringView::npos);
  EXPECT_EQ(s.find_last_of("XYZ"), StringView::npos);
  // all-in-set from the far end
  StringView alldot{".;.;.;.;.;.;.;.;.;.;", 20};  // 20 bytes, crosses block
  EXPECT_EQ(alldot.find_first_not_of(set), StringView::npos);
  EXPECT_EQ(alldot.find_last_not_of(set), StringView::npos);
  EXPECT_EQ(alldot.find_first_of(set), 0u);
  EXPECT_EQ(alldot.find_last_of(set), 19u);

  // set > 16 chars -> forces the scalar-table fallback (still correct).
  const StringView bigset = "0123456789abcdefgh";  // 18 chars
  std::string t(50, 'z');
  t[40] = 'c';  // in bigset
  StringView ts{t.data(), static_cast<std::uint32_t>(t.size())};
  EXPECT_EQ(ts.find_first_of(bigset), 40u);
  EXPECT_EQ(ts.find_last_of(bigset), 40u);
  EXPECT_EQ(ts.find_first_not_of(bigset), 0u);  // 'z' not in bigset
}

TEST(StringView, RfindSubstrAndCount) {
  StringView s = "ab_ab_ab";
  EXPECT_EQ(s.rfind(StringView{"ab"}), 6u);     // rightmost "ab"
  EXPECT_EQ(s.rfind(StringView{"ab"}, 5), 3u);  // bounded before pos 5
  EXPECT_EQ(s.rfind(StringView{"zz"}), StringView::npos);
  EXPECT_EQ(s.count('a'), 3u);
  EXPECT_EQ(s.count('_'), 2u);
  EXPECT_EQ(s.count('z'), 0u);
}

// count()'s runtime path is SSE2 (16-byte blocks + scalar tail); the ab_ab_ab
// case above is 8B => tail-only. Exercise the vectorized main loop AND a
// non-block-aligned tail: 70 chars = 4 full 16B groups + a 6B tail.
TEST(StringView, CountSimdPathCrossesBlockAndTail) {
  std::string buf;
  buf.reserve(70);
  for (int i = 0; i < 70; ++i) buf.push_back((i % 3 == 0) ? 'x' : 'o');
  StringView s{buf.data(), static_cast<std::uint32_t>(buf.size())};
  // 'x' at indices 0,3,6,...,69 => 24 of them; oracle-cross-check with scalar.
  std::uint32_t oracle = 0;
  for (char c : buf) oracle += (c == 'x');
  EXPECT_EQ(s.count('x'), oracle);
  EXPECT_EQ(s.count('x'), 24u);
  EXPECT_EQ(s.count('o'), 46u);
  EXPECT_EQ(s.count('z'), 0u);  // absent => full scan, 0
  // exactly one 16B block, no tail
  StringView b16{buf.data(), 16};
  std::uint32_t o16 = 0;
  for (int i = 0; i < 16; ++i) o16 += (buf[i] == 'x');
  EXPECT_EQ(b16.count('x'), o16);
}

// rfind(char) and rfind(needle) runtime paths are SSE2 backward scans (16B
// blocks + scalar head). The short cases above are tail/head-only; drive the
// vectorized backward loop with a >16B buffer and cross-check against an oracle.
TEST(StringView, RfindSimdBackwardPath) {
  // 100 chars: 'z' seeded at a few known spots, all else 'a'. Verify rightmost.
  std::string buf(100, 'a');
  buf[3] = 'z';
  buf[50] = 'z';
  buf[97] = 'z';  // rightmost, inside the last full 16B block from the high end
  StringView s{buf.data(), static_cast<std::uint32_t>(buf.size())};
  EXPECT_EQ(s.rfind('z'), 97u);
  EXPECT_EQ(s.rfind('z', 96), 50u);  // bounded before 97 -> next is 50
  EXPECT_EQ(s.rfind('z', 49), 3u);
  EXPECT_EQ(s.rfind('q'), StringView::npos);  // absent => full backward scan
  // needle path: place "zz" so the first-char candidate walk must reject a lone
  // 'z' before finding the real pair.
  std::string nbuf(80, 'a');
  nbuf[10] = 'z';            // lone 'z' (candidate, but not "zz")
  nbuf[60] = nbuf[61] = 'z'; // the real "zz"
  StringView ns{nbuf.data(), static_cast<std::uint32_t>(nbuf.size())};
  EXPECT_EQ(ns.rfind(StringView{"zz"}), 60u);
  EXPECT_EQ(ns.rfind(StringView{"zz"}, 59), StringView::npos);  // none <= 59
  EXPECT_EQ(ns.rfind(StringView{"qq"}), StringView::npos);
}

TEST(StringView, Comparison) {
  EXPECT_TRUE(StringView{"abc"} == StringView{"abc"});
  EXPECT_FALSE(StringView{"abc"} == StringView{"abd"});
  EXPECT_TRUE(StringView{"abc"} < StringView{"abd"});
  EXPECT_TRUE(StringView{"abc"} < StringView{"abcd"});  // prefix < longer
  EXPECT_TRUE(StringView{"abd"} > StringView{"abc"});
  EXPECT_TRUE(equal_ignore_case("HeLLo", "hello"));
  EXPECT_FALSE(equal_ignore_case("hello", "help"));
}

TEST(StringView, CopyTo) {
  char buf[8];
  StringView s = "hello world";
  auto n = s.copy_to(buf);
  EXPECT_EQ(n, 7u);                 // truncated to cap-1
  EXPECT_STREQ(buf, "hello w");     // always NUL-terminated
  EXPECT_EQ(StringView{"hi"}.copy_to(buf), 2u);
  EXPECT_STREQ(buf, "hi");
}

TEST(StringView, Literal) {
  auto s = "literal"_sv;
  EXPECT_EQ(s.size(), 7u);
  EXPECT_TRUE(s.is_null_terminated());
  EXPECT_TRUE(s == StringView{"literal"});
}

TEST(StringView, CallerOwnedCopyMakesExplicitCString) {
  const char raw[] = {'h', 'i', '!'};
  StringView nt(raw, 3, false);
  char storage[4];
  EXPECT_EQ(nt.copy_to(storage), 3u);
  EXPECT_NE(storage, raw);
  EXPECT_STREQ(storage, "hi!");

  // A view already known to be terminated remains a zero-copy C string.
  StringView t = "term";
  EXPECT_EQ(t.c_str(), t.data());
}

TEST(StringView, SliceAndSplit) {
  EXPECT_TRUE(slice(StringView{"hello"}, 2) == StringView{"llo"});
  EXPECT_TRUE(slice(StringView{"hi"}, 5).empty());

  const StringView source = "tail";
  const StringView empty_tail = slice(source, source.size());
  EXPECT_EQ(empty_tail.data(), source.data() + source.size());
  EXPECT_TRUE(empty_tail.empty());
  EXPECT_TRUE(empty_tail.is_null_terminated());

  StringView csv = "a,bb,ccc";
  EXPECT_TRUE(split(&csv, ',') == StringView{"a"});
  EXPECT_TRUE(split(&csv, ',') == StringView{"bb"});
  EXPECT_TRUE(split(&csv, ',') == StringView{"ccc"});
  EXPECT_TRUE(csv.empty());  // exhausted

  StringView no_delim = "whole";
  EXPECT_TRUE(split(&no_delim, ',') == StringView{"whole"});
  EXPECT_TRUE(no_delim.empty());
}

TEST(StringView, Rsplit) {
  // rsplit consumes right-to-left: returns the tail after the last delimiter.
  StringView path = "a/b/c";
  EXPECT_TRUE(rsplit(&path, '/') == StringView{"c"});
  EXPECT_TRUE(path == StringView{"a/b"});
  EXPECT_TRUE(rsplit(&path, '/') == StringView{"b"});
  EXPECT_TRUE(path == StringView{"a"});
  EXPECT_TRUE(rsplit(&path, '/') == StringView{"a"});  // no delimiter left
  EXPECT_TRUE(path.empty());
}

// StringView is usable as a SwissTable key via its Hash specialization. Keys
// must outlive the table (the table stores views, not copies).
TEST(StringView, AsSwissTableKey) {
  SwissTable<StringView, int> t;
  const char* keys[] = {"alpha", "beta", "gamma", "delta"};
  for (int i = 0; i < 4; ++i) t.insert(keys[i], i);
  EXPECT_EQ(t.size(), 4u);
  ASSERT_NE(t.find("gamma"), nullptr);
  EXPECT_EQ(*t.find("gamma"), 2);
  EXPECT_EQ(t.find("epsilon"), nullptr);
  // distinct buffers, equal content -> same key
  std::string built = std::string("al") + "pha";
  ASSERT_NE(t.find(built), nullptr);
  EXPECT_EQ(*t.find(built), 0);
}
