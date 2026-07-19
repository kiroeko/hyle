#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "hyle/container/string_buf.h"
#include "hyle/container/string_ref.h"
#include "hyle/mem/allocator.h"

using hyle::container::destroy_string_buf;
using hyle::container::InternId;
using hyle::container::make_string_buf;
using hyle::container::string_ref;
using hyle::container::StringBuf;
using hyle::mem::system_allocator;

TEST(StringBuf, AddAndLookup) {
  StringBuf buf(system_allocator());
  InternId h = buf.add("hello");
  EXPECT_FALSE(h.empty());
  string_ref s = buf.lookup(h);
  EXPECT_TRUE(s == string_ref{"hello"});
  EXPECT_TRUE(s.null_terminated);  // interned strings are always C-usable
  EXPECT_EQ(buf.size(), 1u);
}

TEST(StringBuf, EmptyStringIsImplicitZero) {
  StringBuf buf(system_allocator());
  EXPECT_TRUE(buf.add("").empty());          // empty maps to empty id, never stored
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.lookup(InternId{0}).empty());  // empty id -> empty string
  EXPECT_TRUE(buf.lookup(InternId{0}).null_terminated);
}

TEST(StringBuf, MissingIdLookupIsEmpty) {
  StringBuf buf(system_allocator());
  EXPECT_TRUE(buf.lookup(InternId{0xDEADBEEF}).empty());
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
  // not have moved (short strings occupy stable slab blocks).
  string_ref again = buf.lookup(buf.add("keep_me"));
  EXPECT_EQ(again.data, addr);
  buf.remove(buf.add("keep_me"));  // undo the extra add above
}

TEST(StringBuf, RefcountReclaimsOnLastRemove) {
  StringBuf buf(system_allocator());
  InternId h = buf.add("refcounted");       // refs = 1
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
  InternId h1 = buf.add("dup");
  InternId h2 = buf.add("dup");
  EXPECT_EQ(h1, h2);             // same content -> same handle
  EXPECT_EQ(buf.size(), 1u);     // one allocation
  buf.remove(h1);
  EXPECT_EQ(buf.size(), 1u);     // still one ref left
  buf.remove(h2);
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, RemoveUnknownOrZeroIsNoop) {
  StringBuf buf(system_allocator());
  EXPECT_TRUE(buf.remove(InternId{0}));        // empty string is always valid
  EXPECT_FALSE(buf.remove(InternId{0xABCD}));  // never added
  EXPECT_FALSE(buf.retain(InternId{0xABCD}));
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, PermanentInternDoesNotAccumulateHiddenReferences) {
  StringBuf buf(system_allocator());
  string_ref first = buf.intern("permanent");
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(buf.intern("permanent").data, first.data);
  InternId id = buf.add("permanent");
  EXPECT_TRUE(buf.remove(id));
  EXPECT_TRUE(buf.lookup(id) == string_ref{"permanent"});
  EXPECT_FALSE(buf.remove(id));  // no releasable reference remains
  EXPECT_EQ(buf.size(), 1u);     // permanent until pool destruction
}

TEST(StringBuf, ReusedSlotRejectsStaleGeneration) {
  StringBuf buf(system_allocator());
  const InternId old = buf.add(string_ref{"first"});
  ASSERT_FALSE(old.empty());
  ASSERT_TRUE(buf.remove(old));

  const InternId current = buf.add(string_ref{"second"});
  ASSERT_FALSE(current.empty());
  EXPECT_NE(old.value, current.value);
  EXPECT_EQ(buf.lookup(old).data, nullptr);
  EXPECT_FALSE(buf.retain(old));
  EXPECT_FALSE(buf.remove(old));
  EXPECT_TRUE(buf.lookup(current) == string_ref{"second"});
}

TEST(StringBuf, InternManyReturnsStableViewsAndCanPrewarm) {
  StringBuf buf(system_allocator());
  const string_ref input[] = {string_ref{"alpha"}, string_ref{"beta"},
                              string_ref{"alpha"}};
  string_ref output[3];
  EXPECT_EQ(buf.intern_many(input, output, 3), 3u);
  EXPECT_EQ(output[0].data, output[2].data);
  EXPECT_TRUE(output[1] == string_ref{"beta"});
  EXPECT_EQ(buf.size(), 2u);
  EXPECT_EQ(buf.intern_many(input, nullptr, 3), 3u);
  EXPECT_EQ(buf.size(), 2u);
}

TEST(StringBuf, BatchLifecyclePreservesOrderAndReclaims) {
  StringBuf buf(system_allocator());
  const string_ref input[] = {string_ref{"alpha"}, string_ref{"beta"},
                              string_ref{"alpha"}, string_ref{""}};
  InternId ids[4];
  EXPECT_EQ(buf.add_many(input, ids, 4), 4u);
  EXPECT_EQ(ids[0], ids[2]);
  EXPECT_TRUE(ids[3].empty());
  EXPECT_EQ(buf.size(), 2u);

  EXPECT_EQ(buf.retain_many(ids, 4), 4u);
  EXPECT_EQ(buf.remove_many(ids, 4), 4u);
  EXPECT_EQ(buf.size(), 2u);
  EXPECT_EQ(buf.remove_many(ids, 3), 3u);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.lookup(ids[0]).empty());
}

TEST(StringBuf, BatchLifecycleStopsAtStaleHandle) {
  StringBuf buf(system_allocator());
  InternId ids[] = {buf.add("one"), buf.add("two")};
  ASSERT_TRUE(buf.remove(ids[0]));
  const InternId sequence[] = {ids[1], ids[0], ids[1]};
  EXPECT_EQ(buf.retain_many(sequence, 3), 1u);
  EXPECT_EQ(buf.remove_many(sequence, 3), 1u);
  EXPECT_TRUE(buf.lookup(ids[1]) == string_ref{"two"});
}

TEST(StringBuf, LongStringsSurviveGrowthAndEraseWithoutIdentityLoss) {
  StringBuf buf(system_allocator());
  std::vector<std::string> storage;
  std::vector<InternId> ids;
  storage.reserve(2048);
  ids.reserve(2048);
  for (int i = 0; i < 2048; ++i) {
    storage.push_back(std::string(1024, static_cast<char>('a' + i % 26)) +
                      std::to_string(i));
    ids.push_back(buf.add(storage.back()));
  }
  EXPECT_EQ(buf.size(), ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    EXPECT_TRUE(buf.lookup(ids[i]) == string_ref{storage[i]});
  EXPECT_EQ(buf.remove_many(ids.data(), ids.size()), ids.size());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringBuf, ManyDistinctStrings) {
  StringBuf buf(system_allocator());
  std::vector<InternId> handles;
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
  InternId h = buf->add("factory");
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
