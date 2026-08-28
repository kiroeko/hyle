#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"

using hyle::container::equal_ignore_case;
using hyle::container::rsplit;
using hyle::container::slice;
using hyle::container::split;
using hyle::container::string_ref;
using hyle::container::SwissTable;
using namespace hyle::container::literals;

TEST(StringRef, DefaultIsEmpty) {
  string_ref s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size, 0u);
  EXPECT_EQ(s.data, nullptr);
}

TEST(StringRef, DefaultRangeAndSubstrStayNullSafe) {
  string_ref s;
  EXPECT_EQ(s.begin(), nullptr);
  EXPECT_EQ(s.end(), nullptr);
  string_ref tail = s.substr(0);
  EXPECT_EQ(tail.data, nullptr);
  EXPECT_EQ(tail.size, 0u);
}

TEST(StringRef, FromCStringIsNullTerminated) {
  string_ref s = "hello";
  EXPECT_EQ(s.size, 5u);
  EXPECT_TRUE(s.null_terminated);
  EXPECT_STREQ(s.c_str(), "hello");  // safe: null_terminated bit set
}

TEST(StringRef, PtrSizeCtorBitHonored) {
  const char buf[] = {'a', 'b', 'c'};  // NOT null-terminated
  string_ref s(buf, 3, false);
  EXPECT_EQ(s.size, 3u);
  EXPECT_FALSE(s.null_terminated);
  EXPECT_EQ(s[0], 'a');
  EXPECT_EQ(s.back(), 'c');
}

TEST(StringRef, Layout16Bytes) {
  // The compact view contract: string_ref stays a 16-byte POD.
  EXPECT_EQ(sizeof(string_ref), 16u);
}

TEST(StringRef, FromStdTypesWithoutHeaderLeak) {
  std::string owned = "from std::string";
  string_ref a = owned;  // generic-range ctor
  EXPECT_EQ(a.size, owned.size());
  EXPECT_TRUE(a == string_ref{"from std::string"});

  std::string_view sv = "from view";
  string_ref b = sv;
  EXPECT_EQ(b.size, sv.size());
}

TEST(StringRef, FindChar) {
  string_ref s = "abracadabra";
  EXPECT_EQ(s.find('a'), 0u);
  EXPECT_EQ(s.find('a', 1), 3u);
  EXPECT_EQ(s.find('z'), string_ref::npos);
  EXPECT_EQ(s.rfind('a'), 10u);
  EXPECT_EQ(s.rfind('b'), 8u);
}

TEST(StringRef, FindSubstr) {
  string_ref s = "the quick brown fox";
  EXPECT_EQ(s.find("quick"), 4u);
  EXPECT_EQ(s.find("fox"), 16u);
  EXPECT_EQ(s.find("cat"), string_ref::npos);
  EXPECT_EQ(s.find(""), 0u);          // empty needle matches at pos
  EXPECT_TRUE(s.contains("brown"));
  EXPECT_FALSE(s.contains("BROWN"));
}

TEST(StringRef, FindAgainstStdViewOracle) {
  // Differential vs std::string_view (oracle ONLY in tests -- conventions §1).
  std::string_view o = "mississippi delta mississippi";
  string_ref s = "mississippi delta mississippi";
  for (char c : {'m', 'i', 's', 'p', ' ', 'z'}) {
    auto exp = o.find(c);
    auto got = s.find(c);
    EXPECT_EQ(got == string_ref::npos ? std::string_view::npos : got, exp)
        << "char " << c;
  }
  for (const char* n : {"miss", "issi", "delta", "xyz", "mississippi"}) {
    auto exp = o.find(n);
    auto got = s.find(n);
    EXPECT_EQ(got == string_ref::npos ? std::string_view::npos : got, exp)
        << "needle " << n;
  }
}

// The search family was rewritten with RUNTIME fast paths (find/find(needle) ->
// memchr, find_first_of -> memchr-min, find_first_not_of/last_of/last_not_of ->
// 256-entry membership table, region_eq/starts_with/ends_with -> memcmp). This
// differential fuzz proves every runtime path matches std::string_view across
// many haystacks/sets/positions -- the algorithms changed, so they must be
// re-proven, not assumed.
TEST(StringRef, SetFamilyAgainstStdViewOracle) {
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
    return v == string_ref::npos ? std::string_view::npos
                                 : static_cast<std::size_t>(v);
  };
  for (const char* h : haystacks) {
    std::string_view o{h};
    string_ref s{h};
    for (const char* setp : sets) {
      std::string_view oset{setp};
      string_ref sset{setp};
      // pos = 0 and a couple interior/past-end positions to exercise bounds.
      for (std::size_t pos : {std::size_t{0}, std::size_t{1}, std::size_t{3},
                              o.size(), o.size() + 5}) {
        std::uint32_t sp = pos > string_ref::npos
                               ? string_ref::npos
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
TEST(StringRef, SearchFamilyConstexpr) {
  constexpr string_ref s = "a.b.c.d";
  static_assert(s.find('b') == 2u);
  static_assert(s.find('z') == string_ref::npos);
  static_assert(s.find(string_ref{"b.c"}) == 2u);
  static_assert(s.find(string_ref{"zzz"}) == string_ref::npos);
  static_assert(s.find_first_of(string_ref{".c"}) == 1u);
  static_assert(s.find_first_not_of(string_ref{"a."}) == 2u);
  static_assert(s.find_last_of(string_ref{"."}) == 5u);
  static_assert(s.find_last_not_of(string_ref{"d."}) == 4u);
  static_assert(s.starts_with(string_ref{"a.b"}));
  static_assert(!s.starts_with(string_ref{"a.c"}));
  static_assert(s.ends_with(string_ref{"c.d"}));
  SUCCEED();  // the real assertions are the static_asserts above (compile-time)
}

TEST(StringRef, Substr) {
  string_ref s = "hello world";
  EXPECT_TRUE(s.substr(0, 5) == string_ref{"hello"});
  EXPECT_TRUE(s.substr(6) == string_ref{"world"});
  EXPECT_TRUE(s.substr(6).null_terminated);   // tail keeps the terminator
  EXPECT_FALSE(s.substr(0, 5).null_terminated);  // prefix does not
  EXPECT_TRUE(s.substr(100).empty());          // clamped past end
  EXPECT_TRUE(s.substr(3, 1000) == string_ref{"lo world"});  // len clamped
}

TEST(StringRef, StartsEndsWith) {
  string_ref s = "filename.txt";
  EXPECT_TRUE(s.starts_with("file"));
  EXPECT_FALSE(s.starts_with("File"));
  EXPECT_TRUE(s.ends_with(".txt"));
  EXPECT_FALSE(s.ends_with(".png"));
  EXPECT_TRUE(s.starts_with(""));
  EXPECT_TRUE(s.ends_with(""));
}

TEST(StringRef, Trim) {
  EXPECT_TRUE(string_ref{"  hi \t\n"}.trim() == string_ref{"hi"});
  EXPECT_TRUE(string_ref{"xxhixx"}.trim("x") == string_ref{"hi"});
  EXPECT_TRUE(string_ref{"   "}.trim().empty());  // all-whitespace -> empty
  EXPECT_TRUE(string_ref{"none"}.trim() == string_ref{"none"});
}

TEST(StringRef, FindFirstLastOf) {
  string_ref s = "a.b.c";
  EXPECT_EQ(s.find_first_of("."), 1u);
  EXPECT_EQ(s.find_first_not_of("a."), 2u);
  EXPECT_EQ(s.find_last_not_of("c."), 2u);
  EXPECT_EQ(s.find_last_of("."), 3u);     // rightmost '.'
  EXPECT_EQ(s.find_last_of("xyz"), string_ref::npos);
}

// find_*_of runtime path is SSE4.2 PCMPESTRI (set<=16), 16-byte blocks + a <16 tail
// + a backward variant. The 5-char case above is tail-only; drive the vectorized loop
// with a >16B buffer and oracle-cross-check every op against the scalar definition.
TEST(StringRef, FindOfSimdCrossesBlockAndTail) {
  // 70 bytes: mostly 'a', with '.'/';'/digits sprinkled at known indices.
  std::string b(70, 'a');
  b[5] = '.';
  b[17] = ';';   // just past the first 16B block
  b[33] = '7';
  b[69] = '.';   // last byte
  string_ref s{b.data(), static_cast<std::uint32_t>(b.size())};
  const string_ref set = ".;";  // <=16 -> SSE4.2 path

  auto oracle_first_of = [&](bool want) -> std::uint32_t {
    for (std::uint32_t i = 0; i < b.size(); ++i) {
      bool in = (b[i] == '.' || b[i] == ';');
      if (in == want) return i;
    }
    return string_ref::npos;
  };
  auto oracle_last_of = [&](bool want) -> std::uint32_t {
    for (std::uint32_t i = static_cast<std::uint32_t>(b.size()); i-- > 0;) {
      bool in = (b[i] == '.' || b[i] == ';');
      if (in == want) return i;
    }
    return string_ref::npos;
  };
  EXPECT_EQ(s.find_first_of(set), oracle_first_of(true));       // 5
  EXPECT_EQ(s.find_first_of(set), 5u);
  EXPECT_EQ(s.find_first_not_of(set), oracle_first_of(false));  // 0
  EXPECT_EQ(s.find_last_of(set), oracle_last_of(true));         // 69
  EXPECT_EQ(s.find_last_of(set), 69u);
  EXPECT_EQ(s.find_last_not_of(set), oracle_last_of(false));    // 68
  // absent set -> npos both directions (full scan)
  EXPECT_EQ(s.find_first_of("XYZ"), string_ref::npos);
  EXPECT_EQ(s.find_last_of("XYZ"), string_ref::npos);
  // all-in-set from the far end
  string_ref alldot{".;.;.;.;.;.;.;.;.;.;", 20};  // 20 bytes, crosses block
  EXPECT_EQ(alldot.find_first_not_of(set), string_ref::npos);
  EXPECT_EQ(alldot.find_last_not_of(set), string_ref::npos);
  EXPECT_EQ(alldot.find_first_of(set), 0u);
  EXPECT_EQ(alldot.find_last_of(set), 19u);

  // set > 16 chars -> forces the scalar-table fallback (still correct).
  const string_ref bigset = "0123456789abcdefgh";  // 18 chars
  std::string t(50, 'z');
  t[40] = 'c';  // in bigset
  string_ref ts{t.data(), static_cast<std::uint32_t>(t.size())};
  EXPECT_EQ(ts.find_first_of(bigset), 40u);
  EXPECT_EQ(ts.find_last_of(bigset), 40u);
  EXPECT_EQ(ts.find_first_not_of(bigset), 0u);  // 'z' not in bigset
}

TEST(StringRef, RfindSubstrAndCount) {
  string_ref s = "ab_ab_ab";
  EXPECT_EQ(s.rfind(string_ref{"ab"}), 6u);     // rightmost "ab"
  EXPECT_EQ(s.rfind(string_ref{"ab"}, 5), 3u);  // bounded before pos 5
  EXPECT_EQ(s.rfind(string_ref{"zz"}), string_ref::npos);
  EXPECT_EQ(s.count('a'), 3u);
  EXPECT_EQ(s.count('_'), 2u);
  EXPECT_EQ(s.count('z'), 0u);
}

// count()'s runtime path is SSE2 (16-byte blocks + scalar tail); the ab_ab_ab
// case above is 8B => tail-only. Exercise the vectorized main loop AND a
// non-block-aligned tail: 70 chars = 4 full 16B groups + a 6B tail.
TEST(StringRef, CountSimdPathCrossesBlockAndTail) {
  std::string buf;
  buf.reserve(70);
  for (int i = 0; i < 70; ++i) buf.push_back((i % 3 == 0) ? 'x' : 'o');
  string_ref s{buf.data(), static_cast<std::uint32_t>(buf.size())};
  // 'x' at indices 0,3,6,...,69 => 24 of them; oracle-cross-check with scalar.
  std::uint32_t oracle = 0;
  for (char c : buf) oracle += (c == 'x');
  EXPECT_EQ(s.count('x'), oracle);
  EXPECT_EQ(s.count('x'), 24u);
  EXPECT_EQ(s.count('o'), 46u);
  EXPECT_EQ(s.count('z'), 0u);  // absent => full scan, 0
  // exactly one 16B block, no tail
  string_ref b16{buf.data(), 16};
  std::uint32_t o16 = 0;
  for (int i = 0; i < 16; ++i) o16 += (buf[i] == 'x');
  EXPECT_EQ(b16.count('x'), o16);
}

// rfind(char) and rfind(needle) runtime paths are SSE2 backward scans (16B
// blocks + scalar head). The short cases above are tail/head-only; drive the
// vectorized backward loop with a >16B buffer and cross-check against an oracle.
TEST(StringRef, RfindSimdBackwardPath) {
  // 100 chars: 'z' seeded at a few known spots, all else 'a'. Verify rightmost.
  std::string buf(100, 'a');
  buf[3] = 'z';
  buf[50] = 'z';
  buf[97] = 'z';  // rightmost, inside the last full 16B block from the high end
  string_ref s{buf.data(), static_cast<std::uint32_t>(buf.size())};
  EXPECT_EQ(s.rfind('z'), 97u);
  EXPECT_EQ(s.rfind('z', 96), 50u);  // bounded before 97 -> next is 50
  EXPECT_EQ(s.rfind('z', 49), 3u);
  EXPECT_EQ(s.rfind('q'), string_ref::npos);  // absent => full backward scan
  // needle path: place "zz" so the first-char candidate walk must reject a lone
  // 'z' before finding the real pair.
  std::string nbuf(80, 'a');
  nbuf[10] = 'z';            // lone 'z' (candidate, but not "zz")
  nbuf[60] = nbuf[61] = 'z'; // the real "zz"
  string_ref ns{nbuf.data(), static_cast<std::uint32_t>(nbuf.size())};
  EXPECT_EQ(ns.rfind(string_ref{"zz"}), 60u);
  EXPECT_EQ(ns.rfind(string_ref{"zz"}, 59), string_ref::npos);  // none <= 59
  EXPECT_EQ(ns.rfind(string_ref{"qq"}), string_ref::npos);
}

TEST(StringRef, Comparison) {
  EXPECT_TRUE(string_ref{"abc"} == string_ref{"abc"});
  EXPECT_FALSE(string_ref{"abc"} == string_ref{"abd"});
  EXPECT_TRUE(string_ref{"abc"} < string_ref{"abd"});
  EXPECT_TRUE(string_ref{"abc"} < string_ref{"abcd"});  // prefix < longer
  EXPECT_TRUE(string_ref{"abd"} > string_ref{"abc"});
  EXPECT_TRUE(equal_ignore_case("HeLLo", "hello"));
  EXPECT_FALSE(equal_ignore_case("hello", "help"));
}

TEST(StringRef, CopyTo) {
  char buf[8];
  string_ref s = "hello world";
  auto n = s.copy_to(buf);
  EXPECT_EQ(n, 7u);                 // truncated to cap-1
  EXPECT_STREQ(buf, "hello w");     // always NUL-terminated
  EXPECT_EQ(string_ref{"hi"}.copy_to(buf), 2u);
  EXPECT_STREQ(buf, "hi");
}

TEST(StringRef, Literal) {
  auto s = "literal"_sr;
  EXPECT_EQ(s.size, 7u);
  EXPECT_TRUE(s.null_terminated);
  EXPECT_TRUE(s == string_ref{"literal"});
}

TEST(StringRef, CallerOwnedCopyMakesExplicitCString) {
  const char raw[] = {'h', 'i', '!'};
  string_ref nt(raw, 3, false);
  char storage[4];
  EXPECT_EQ(nt.copy_to(storage), 3u);
  EXPECT_NE(storage, raw);
  EXPECT_STREQ(storage, "hi!");

  // A view already known to be terminated remains a zero-copy C string.
  string_ref t = "term";
  EXPECT_EQ(t.c_str(), t.data);
}

TEST(StringRef, SliceAndSplit) {
  EXPECT_TRUE(slice(string_ref{"hello"}, 2) == string_ref{"llo"});
  EXPECT_TRUE(slice(string_ref{"hi"}, 5).empty());

  string_ref csv = "a,bb,ccc";
  EXPECT_TRUE(split(&csv, ',') == string_ref{"a"});
  EXPECT_TRUE(split(&csv, ',') == string_ref{"bb"});
  EXPECT_TRUE(split(&csv, ',') == string_ref{"ccc"});
  EXPECT_TRUE(csv.empty());  // exhausted

  string_ref no_delim = "whole";
  EXPECT_TRUE(split(&no_delim, ',') == string_ref{"whole"});
  EXPECT_TRUE(no_delim.empty());
}

TEST(StringRef, Rsplit) {
  // rsplit consumes right-to-left: returns the tail after the last delimiter.
  string_ref path = "a/b/c";
  EXPECT_TRUE(rsplit(&path, '/') == string_ref{"c"});
  EXPECT_TRUE(path == string_ref{"a/b"});
  EXPECT_TRUE(rsplit(&path, '/') == string_ref{"b"});
  EXPECT_TRUE(path == string_ref{"a"});
  EXPECT_TRUE(rsplit(&path, '/') == string_ref{"a"});  // no delimiter left
  EXPECT_TRUE(path.empty());
}

// string_ref is usable as a SwissTable key via its Hash specialization. Keys
// must outlive the table (the table stores views, not copies).
TEST(StringRef, AsSwissTableKey) {
  SwissTable<string_ref, int> t;
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
