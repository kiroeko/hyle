#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"

using hyle::container::concat;
using hyle::container::dup;
using hyle::container::equal_ignore_case;
using hyle::container::rsplit;
using hyle::container::slice;
using hyle::container::split;
using hyle::container::string_ref;
using hyle::container::SwissTable;
using hyle::mem::system_allocator;
using namespace hyle::container::literals;

TEST(StringRef, DefaultIsEmpty) {
  string_ref s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size, 0u);
  EXPECT_EQ(s.data, nullptr);
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
  // The C-ABI contract: a string_ref must stay a 16-byte POD across the seam.
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

TEST(StringRef, RfindSubstrAndCount) {
  string_ref s = "ab_ab_ab";
  EXPECT_EQ(s.rfind(string_ref{"ab"}), 6u);     // rightmost "ab"
  EXPECT_EQ(s.rfind(string_ref{"ab"}, 5), 3u);  // bounded before pos 5
  EXPECT_EQ(s.rfind(string_ref{"zz"}), string_ref::npos);
  EXPECT_EQ(s.count('a'), 3u);
  EXPECT_EQ(s.count('_'), 2u);
  EXPECT_EQ(s.count('z'), 0u);
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
  auto n = s.copy_to(buf, sizeof(buf));
  EXPECT_EQ(n, 7u);                 // truncated to cap-1
  EXPECT_STREQ(buf, "hello w");     // always NUL-terminated
  EXPECT_EQ(string_ref{"hi"}.copy_to(buf, sizeof(buf)), 2u);
  EXPECT_STREQ(buf, "hi");
}

TEST(StringRef, Literal) {
  auto s = "literal"_sr;
  EXPECT_EQ(s.size, 7u);
  EXPECT_TRUE(s.null_terminated);
  EXPECT_TRUE(s == string_ref{"literal"});
}

TEST(StringRef, AllocatorHelpers) {
  auto a = system_allocator();
  // c_str on a non-terminated view must allocate a copy.
  const char raw[] = {'h', 'i', '!'};
  string_ref nt(raw, 3, false);
  const char* c = hyle::container::c_str(nt, a);
  EXPECT_NE(c, raw);             // a fresh allocation
  EXPECT_STREQ(c, "hi!");
  a.deallocate(const_cast<char*>(c), 4);

  // c_str on a terminated view returns the same pointer (zero copy).
  string_ref t = "term";
  EXPECT_EQ(hyle::container::c_str(t, a), t.data);

  string_ref d = dup("copyme", a);
  EXPECT_TRUE(d == string_ref{"copyme"});
  EXPECT_TRUE(d.null_terminated);
  a.deallocate(const_cast<char*>(d.data), d.size + 1);

  string_ref cat = concat("foo", "bar", a);
  EXPECT_TRUE(cat == string_ref{"foobar"});
  EXPECT_STREQ(cat.c_str(), "foobar");
  a.deallocate(const_cast<char*>(cat.data), cat.size + 1);
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
  SwissTable<string_ref, int> t(system_allocator());
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
