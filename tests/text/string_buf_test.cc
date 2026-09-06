#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "hyle/container/string_interner.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/container/string_view.h"
#include "hyle/mem/allocator.h"

using hyle::container::destroy_string_interner;
using hyle::container::StringId;
using hyle::container::make_string_interner;
using hyle::container::StringView;
using hyle::container::StringInterner;
using hyle::mem::system_allocator;

TEST(StringInterner, AcquireAndLookup) {
  StringInterner buf(system_allocator());
  StringId h = buf.acquire("hello");
  EXPECT_FALSE(h.empty());
  StringView s = buf.lookup(h);
  EXPECT_TRUE(s == StringView{"hello"});
  EXPECT_TRUE(s.is_null_terminated());  // interned strings are always C-usable
  EXPECT_EQ(buf.size(), 1u);
}
TEST(StringInterner, ConstLookupAndStringIdHashAreNatural) {
  StringInterner buf(system_allocator());
  const StringId id = buf.acquire("const-lookup");
  const StringInterner& read_only = buf;
  EXPECT_EQ(read_only.lookup(id).as_std(), "const-lookup");

  hyle::container::FlatHashMap<StringId, int> by_id;
  ASSERT_TRUE(by_id.insert(id, 42));
  ASSERT_NE(by_id.find(id), nullptr);
  EXPECT_EQ(*by_id.find(id), 42);
}
TEST(StringInterner, EmptyStringIsImplicitZero) {
  StringInterner buf(system_allocator());
  EXPECT_TRUE(buf.acquire("").empty());          // empty maps to empty id, never stored
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.lookup(StringId{0}).empty());  // empty id -> empty string
  EXPECT_TRUE(buf.lookup(StringId{0}).is_null_terminated());
}

TEST(StringInterner, MissingIdLookupIsEmpty) {
  StringInterner buf(system_allocator());
  EXPECT_TRUE(buf.lookup(StringId{0xDEADBEEF}).empty());
}

TEST(StringInterner, InternReturnsStablePointerForEqualContent) {
  StringInterner buf(system_allocator());
  // Two distinct source buffers with identical content.
  std::string a = "interned_value";
  std::string b = std::string("interned") + "_value";
  ASSERT_NE(a.data(), b.data());

  StringView ra = buf.intern(a);
  StringView rb = buf.intern(b);
  EXPECT_EQ(ra.data(), rb.data());   // same interned pointer -> pointer identity
  EXPECT_EQ(buf.size(), 1u);     // stored once
  EXPECT_TRUE(ra == StringView{"interned_value"});
}

TEST(StringInterner, InternedAddressSurvivesMoreInterns) {
  StringInterner buf(system_allocator());
  StringView first = buf.intern("keep_me");
  const char* addr = first.data();
  for (int i = 0; i < 1000; ++i) {
    buf.intern(std::string("filler_") + std::to_string(i));
  }
  // The reverse index (FlatHashMap) rehashed many times; the string storage must
  // not have moved (short strings occupy stable slab blocks).
  StringView again = buf.lookup(buf.acquire("keep_me"));
  EXPECT_EQ(again.data(), addr);
  buf.release(buf.acquire("keep_me"));  // undo the extra add above
}

TEST(StringInterner, RefcountReclaimsOnLastRelease) {
  StringInterner buf(system_allocator());
  StringId h = buf.acquire("refcounted");       // refs = 1
  buf.retain(h);                            // refs = 2  (== add again)
  EXPECT_EQ(buf.size(), 1u);

  buf.release(h);                            // refs = 1, still alive
  EXPECT_TRUE(buf.lookup(h) == StringView{"refcounted"});
  EXPECT_EQ(buf.size(), 1u);

  buf.release(h);                            // refs = 0 -> reclaimed
  EXPECT_TRUE(buf.lookup(h).empty());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringInterner, AcquireSameContentBumpsRefcountNotSize) {
  StringInterner buf(system_allocator());
  StringId h1 = buf.acquire("dup");
  StringId h2 = buf.acquire("dup");
  EXPECT_EQ(h1, h2);             // same content -> same handle
  EXPECT_EQ(buf.size(), 1u);     // one allocation
  buf.release(h1);
  EXPECT_EQ(buf.size(), 1u);     // still one ref left
  buf.release(h2);
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringInterner, ReleaseUnknownOrZeroIsNoop) {
  StringInterner buf(system_allocator());
  EXPECT_TRUE(buf.release(StringId{0}));        // empty string is always valid
  EXPECT_FALSE(buf.release(StringId{0xABCD}));  // never added
  EXPECT_FALSE(buf.retain(StringId{0xABCD}));
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringInterner, PermanentInternDoesNotAccumulateHiddenReferences) {
  StringInterner buf(system_allocator());
  StringView first = buf.intern("permanent");
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(buf.intern("permanent").data(), first.data());
  StringId id = buf.acquire("permanent");
  EXPECT_TRUE(buf.release(id));
  EXPECT_TRUE(buf.lookup(id) == StringView{"permanent"});
  EXPECT_FALSE(buf.release(id));  // no releasable reference remains
  EXPECT_EQ(buf.size(), 1u);     // permanent until pool destruction
}

TEST(StringInterner, ReusedSlotRejectsStaleGeneration) {
  StringInterner buf(system_allocator());
  const StringId old = buf.acquire(StringView{"first"});
  ASSERT_FALSE(old.empty());
  ASSERT_TRUE(buf.release(old));

  const StringId current = buf.acquire(StringView{"second"});
  ASSERT_FALSE(current.empty());
  EXPECT_NE(old.value, current.value);
  EXPECT_EQ(buf.lookup(old).data(), nullptr);
  EXPECT_FALSE(buf.retain(old));
  EXPECT_FALSE(buf.release(old));
  EXPECT_TRUE(buf.lookup(current) == StringView{"second"});
}

TEST(StringInterner, InternManyReturnsStableViewsAndCanPrewarm) {
  StringInterner buf(system_allocator());
  const StringView input[] = {StringView{"alpha"}, StringView{"beta"},
                              StringView{"alpha"}};
  StringView output[3];
  EXPECT_EQ(buf.intern_many(input, output), 3u);
  EXPECT_EQ(output[0].data(), output[2].data());
  EXPECT_TRUE(output[1] == StringView{"beta"});
  EXPECT_EQ(buf.size(), 2u);
  EXPECT_EQ(buf.prewarm(input), 3u);
  EXPECT_EQ(buf.size(), 2u);
}

TEST(StringInterner, BatchLifecyclePreservesOrderAndReclaims) {
  StringInterner buf(system_allocator());
  const StringView input[] = {StringView{"alpha"}, StringView{"beta"},
                              StringView{"alpha"}, StringView{""}};
  StringId ids[4];
  EXPECT_EQ(buf.acquire_many(input, ids), 4u);
  EXPECT_EQ(ids[0], ids[2]);
  EXPECT_TRUE(ids[3].empty());
  EXPECT_EQ(buf.size(), 2u);

  EXPECT_EQ(buf.retain_many(ids), 4u);
  EXPECT_EQ(buf.release_many(ids), 4u);
  EXPECT_EQ(buf.size(), 2u);
  EXPECT_EQ(buf.release_many(std::span{ids}.first(3)), 3u);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.lookup(ids[0]).empty());
}

TEST(StringInterner, BatchLifecycleStopsAtStaleHandle) {
  StringInterner buf(system_allocator());
  StringId ids[] = {buf.acquire("one"), buf.acquire("two")};
  ASSERT_TRUE(buf.release(ids[0]));
  const StringId sequence[] = {ids[1], ids[0], ids[1]};
  EXPECT_EQ(buf.retain_many(sequence), 1u);
  EXPECT_EQ(buf.release_many(sequence), 1u);
  EXPECT_TRUE(buf.lookup(ids[1]) == StringView{"two"});
}

TEST(StringInterner, LongStringsSurviveGrowthAndEraseWithoutIdentityLoss) {
  StringInterner buf(system_allocator());
  std::vector<std::string> storage;
  std::vector<StringId> ids;
  storage.reserve(2048);
  ids.reserve(2048);
  for (int i = 0; i < 2048; ++i) {
    storage.push_back(std::string(1024, static_cast<char>('a' + i % 26)) +
                      std::to_string(i));
    ids.push_back(buf.acquire(storage.back()));
  }
  EXPECT_EQ(buf.size(), ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    EXPECT_TRUE(buf.lookup(ids[i]) == StringView{storage[i]});
  EXPECT_EQ(buf.release_many(ids), ids.size());
  EXPECT_EQ(buf.size(), 0u);
}

TEST(StringInterner, ManyDistinctStrings) {
  StringInterner buf(system_allocator());
  std::vector<StringId> handles;
  for (int i = 0; i < 5000; ++i)
    handles.push_back(buf.acquire(std::string("s") + std::to_string(i)));
  EXPECT_EQ(buf.size(), 5000u);
  for (int i = 0; i < 5000; ++i) {
    StringView s = buf.lookup(handles[i]);
    EXPECT_TRUE(s == StringView{(std::string("s") + std::to_string(i)).c_str()});
  }
}

TEST(StringInterner, MakeDestroyFactory) {
  StringInterner* buf = make_string_interner(system_allocator());
  ASSERT_NE(buf, nullptr);
  StringId h = buf->acquire("factory");
  EXPECT_TRUE(buf->lookup(h) == StringView{"factory"});
  destroy_string_interner(buf);  // must free the pool + all interned strings
}

TEST(StringInterner, DestructorFreesAllInterned) {
  // Real leak check happens under ASan/Valgrind; here we just exercise teardown
  // with many live strings so for_each-based cleanup runs over a grown table.
  StringInterner buf(system_allocator());
  for (int i = 0; i < 2000; ++i) buf.acquire(std::string("leak_") + std::to_string(i));
  EXPECT_EQ(buf.size(), 2000u);
  // ~StringInterner walks every slot and frees it.
}

// Concurrent interning: the internal lock must serialize mutators. Under WSL
// TSan this is the data-race gate. Equal content from many threads must collapse
// to a single stored string with a single stable pointer.
TEST(StringInterner, ConcurrentInternSameContent) {
  StringInterner buf(system_allocator());
  constexpr int kThreads = 8;
  std::atomic<const char*> seen{nullptr};
  std::atomic<bool> mismatch{false};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&] {
      for (int i = 0; i < 2000; ++i) {
        StringView s = buf.intern("shared_content");
        const char* p = s.data();
        const char* prev = seen.exchange(p);
        if (prev != nullptr && prev != p) mismatch = true;
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_FALSE(mismatch.load());
  EXPECT_TRUE(buf.lookup(buf.acquire("shared_content")) ==
              StringView{"shared_content"});
}

TEST(StringInterner, ConcurrentInternDistinctContent) {
  StringInterner buf(system_allocator());
  constexpr int kThreads = 8;
  constexpr int kPer = 1000;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kPer; ++i)
        buf.acquire(std::string("t") + std::to_string(t) + "_" + std::to_string(i));
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(buf.size(), static_cast<std::uint32_t>(kThreads * kPer));
}
