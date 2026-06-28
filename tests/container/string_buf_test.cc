#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "hyle/container/string_buf.h"
#include "hyle/container/string_ref.h"
#include "hyle/mem/allocator.h"

using hyle::container::destroy_string_buf;
using hyle::container::make_string_buf;
using hyle::container::string_ref;
using hyle::container::StringBuf;
using hyle::mem::system_allocator;

TEST(StringBuf, AddAndLookup) {
  StringBuf buf(system_allocator());
  std::uint64_t h = buf.add("hello");
  EXPECT_NE(h, 0u);
  string_ref s = buf.lookup(h);
  EXPECT_TRUE(s == string_ref{"hello"});
  EXPECT_TRUE(s.null_terminated);  // interned strings are always C-usable
  EXPECT_EQ(buf.size(), 1u);
}

TEST(StringBuf, EmptyStringIsImplicitZero) {
  StringBuf buf(system_allocator());
  EXPECT_EQ(buf.add(""), 0u);          // empty maps to 0, never stored
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.lookup(0).empty());  // 0 -> empty string
  EXPECT_TRUE(buf.lookup(0).null_terminated);
}

TEST(StringBuf, MissingHashLookupIsEmpty) {
  StringBuf buf(system_allocator());
  EXPECT_TRUE(buf.lookup(0xDEADBEEFu).empty());
}

TEST(StringBuf, InternReturnsStablePointerForEqualContent) {
  StringBuf buf(system_allocator());
  // Two distinct source buffers with identical content.
  std::string a = "interned_value";
  std::string b = std::string("interned") + "_value";
  ASSERT_NE(a.data(), b.data());

  string_ref ra = buf.intern(a);
  string_ref rb = buf.intern(b);
  EXPECT_EQ(ra.data, rb.data);   // same interned pointer -> pointer identity
  EXPECT_EQ(buf.size(), 1u);     // stored once
  EXPECT_TRUE(ra == string_ref{"interned_value"});
}

TEST(StringBuf, InternedAddressSurvivesMoreInterns) {
  StringBuf buf(system_allocator());
  string_ref first = buf.intern("keep_me");
  const char* addr = first.data;
  for (int i = 0; i < 1000; ++i) {
    buf.intern(std::string("filler_") + std::to_string(i));
  }
  // The reverse index (SwissTable) rehashed many times; the string storage must
  // not have moved (each unique string has its own allocation).
  string_ref again = buf.lookup(buf.add("keep_me"));
  EXPECT_EQ(again.data, addr);
  buf.remove(buf.add("keep_me"));  // undo the extra add above
}

TEST(StringBuf, RefcountReclaimsOnLastRemove) {
  StringBuf buf(system_allocator());
  std::uint64_t h = buf.add("refcounted");  // refs = 1
  buf.retain(h);                            // refs = 2  (== add again)
  EXPECT_EQ(buf.size(), 1u);

  buf.remove(h);                            // refs = 1, still alive
  EXPECT_TRUE(buf.lookup(h) == string_ref{"refcounted"});
  EXPECT_EQ(buf.size(), 1u);

  buf.remove(h);                            // refs = 0 -> reclaimed
  EXPECT_TRUE(buf.lookup(h).empty());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, AddSameContentBumpsRefcountNotSize) {
  StringBuf buf(system_allocator());
  std::uint64_t h1 = buf.add("dup");
  std::uint64_t h2 = buf.add("dup");
  EXPECT_EQ(h1, h2);             // same content -> same handle
  EXPECT_EQ(buf.size(), 1u);     // one allocation
  buf.remove(h1);
  EXPECT_EQ(buf.size(), 1u);     // still one ref left
  buf.remove(h2);
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, RemoveUnknownOrZeroIsNoop) {
  StringBuf buf(system_allocator());
  buf.remove(0);            // no-op (empty string)
  buf.remove(0xABCDu);      // no-op (never added)
  buf.retain(0xABCDu);      // no-op
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, ManyDistinctStrings) {
  StringBuf buf(system_allocator());
  std::vector<std::uint64_t> handles;
  for (int i = 0; i < 5000; ++i)
    handles.push_back(buf.add(std::string("s") + std::to_string(i)));
  EXPECT_EQ(buf.size(), 5000u);
  for (int i = 0; i < 5000; ++i) {
    string_ref s = buf.lookup(handles[i]);
    EXPECT_TRUE(s == string_ref{(std::string("s") + std::to_string(i)).c_str()});
  }
}

TEST(StringBuf, MakeDestroyFactory) {
  StringBuf* buf = make_string_buf(system_allocator());
  ASSERT_NE(buf, nullptr);
  std::uint64_t h = buf->add("factory");
  EXPECT_TRUE(buf->lookup(h) == string_ref{"factory"});
  destroy_string_buf(buf);  // must free the pool + all interned strings
}

TEST(StringBuf, DestructorFreesAllInterned) {
  // Real leak check happens under ASan/Valgrind; here we just exercise teardown
  // with many live strings so for_each-based cleanup runs over a grown table.
  StringBuf buf(system_allocator());
  for (int i = 0; i < 2000; ++i) buf.add(std::string("leak_") + std::to_string(i));
  EXPECT_EQ(buf.size(), 2000u);
  // ~StringBuf walks every slot and frees it.
}

// Concurrent interning: the internal lock must serialize mutators. Under WSL
// TSan this is the data-race gate. Equal content from many threads must collapse
// to a single stored string with a single stable pointer.
TEST(StringBuf, ConcurrentInternSameContent) {
  StringBuf buf(system_allocator());
  constexpr int kThreads = 8;
  std::atomic<const char*> seen{nullptr};
  std::atomic<bool> mismatch{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) {
        string_ref s = buf.intern("shared_content");
        const char* p = s.data;
        const char* prev = seen.exchange(p);
        if (prev != nullptr && prev != p) mismatch = true;
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_FALSE(mismatch.load());
  EXPECT_TRUE(buf.lookup(buf.add("shared_content")) ==
              string_ref{"shared_content"});
}

TEST(StringBuf, ConcurrentInternDistinctContent) {
  StringBuf buf(system_allocator());
  constexpr int kThreads = 8;
  constexpr int kPer = 1000;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kPer; ++i)
        buf.add(std::string("t") + std::to_string(t) + "_" + std::to_string(i));
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(buf.size(), static_cast<std::uint32_t>(kThreads * kPer));
}
