#include "hyle/container/string_buf.h"

#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <thread>

#include "hyle/container/carray.h"
#include "hyle/container/swiss_table.h"

namespace hyle::container {

namespace {

constexpr std::uint32_t kMaxShardBits = 6;
constexpr std::uint32_t kMaxShardCount = 1u << kMaxShardBits;
constexpr std::uint32_t kSlotBits = 32 - kMaxShardBits;
constexpr std::uint32_t kSlotMask = (1u << kSlotBits) - 1;
constexpr std::uint32_t kNoSlot = UINT32_MAX;

struct InternKey {
  string_ref text;
  std::uint64_t hash;
};

struct InternHash {
  HYLE_FORCE_INLINE std::uint64_t operator()(const InternKey& key) const noexcept {
    return key.hash;
  }
};

struct InternEq {
  HYLE_FORCE_INLINE bool operator()(const InternKey& left, const InternKey& right) const noexcept {
    if (left.hash != right.hash || left.text.size != right.text.size) return false;
    if (left.text.data == right.text.data) return true;
    return left.text.size == 0 || std::memcmp(left.text.data, right.text.data, left.text.size) == 0;
  }
};

// One owned, NUL-terminated storage record per unique string + its live ref
// count. Short records live in stable slab blocks; long records are direct
// allocations. The cached table hash makes table growth and erase O(1) in
// resident string bytes.
struct Entry {
  char* data;
  std::uint64_t hash;
  std::uint32_t size;
  std::uint32_t refs;
  InternId id;
  bool permanent;
};

struct IdSlot {
  Entry* entry;
  std::uint32_t generation;
  std::uint32_t next_free;
};

struct FreeBlock {
  FreeBlock* next;
};

struct Slab {
  Slab* next;
};

constexpr std::uint32_t kSlabClassCount = 3;
constexpr std::size_t kSlabSizes[kSlabClassCount] = {64, 128, 256};
constexpr std::size_t kBlocksPerSlab = 16;

HYLE_FORCE_INLINE std::uint32_t slab_class(std::size_t bytes) noexcept {
  if (bytes <= kSlabSizes[0]) return 0;
  if (bytes <= kSlabSizes[1]) return 1;
  if (bytes <= kSlabSizes[2]) return 2;
  return kSlabClassCount;
}

struct alignas(64) Shard {
  explicit Shard(mem::Allocator alloc) noexcept : by_string(alloc), by_id(alloc) {}

  std::mutex mu;
  SwissTable<InternKey, Entry*, InternHash, InternEq> by_string;
  CArray<IdSlot> by_id;
  std::uint32_t free_head = kNoSlot;
  std::uint32_t live_size = 0;
  FreeBlock* free_blocks[kSlabClassCount]{};
  Slab* slabs[kSlabClassCount]{};
};

struct HashedText {
  string_ref text;
  std::uint64_t table_hash;
  std::uint32_t shard;
};

HYLE_FORCE_INLINE std::uint64_t hash_intern_text(string_ref text) noexcept {
  if (text.size <= 16) return hash_bytes(text.data, text.size);
  if (text.size <= 32) {
    // Symbol names commonly sit just above hash_bytes' 16B inline cutoff. An
    // out-of-line XXH3 call is a measurable fixed cost there (especially across
    // an Apple dylib). First+last 16B cover every byte for 17..32B; three
    // folded 128-bit multiplies retain avalanche without a call or over-read.
    std::uint64_t first0, first1, last0, last1;
    std::memcpy(&first0, text.data, 8);
    std::memcpy(&first1, text.data + 8, 8);
    std::memcpy(&last0, text.data + text.size - 16, 8);
    std::memcpy(&last1, text.data + text.size - 8, 8);
    const std::uint64_t left =
        detail::mix64_2(first0 ^ 0x9ddfea08eb382d69ull,
                        first1 ^ (static_cast<std::uint64_t>(text.size) * 0x9e3779b97f4a7c15ull));
    const std::uint64_t right =
        detail::mix64_2(last0 ^ 0xff51afd7ed558ccdull, last1 ^ 0xc4ceb9fe1a85ec53ull);
    return detail::mix64_2(left ^ 0x243f6a8885a308d3ull, right ^ 0x13198a2e03707344ull);
  }
  return hash_bytes(text.data, text.size);
}

HYLE_FORCE_INLINE HashedText prepare(string_ref text, std::uint32_t shard_bits) noexcept {
  const std::uint64_t content_hash = hash_intern_text(text);
  // SwissTable uses the hash's low bits for H1 and top 7 bits for H2. Select a
  // shard from disjoint middle bits, preserving full entropy in both table
  // components and avoiding a second avalanche on every lookup.
  return HashedText{text, content_hash,
                    static_cast<std::uint32_t>(content_hash >> 32) & ((1u << shard_bits) - 1)};
}

HYLE_FORCE_INLINE InternId make_id(std::uint32_t shard, std::uint32_t slot,
                                   std::uint32_t generation) noexcept {
  const std::uint32_t low = (shard << kSlotBits) | (slot + 1);
  return InternId{(static_cast<std::uint64_t>(generation) << 32) | low};
}

HYLE_FORCE_INLINE std::uint32_t id_shard(InternId id) noexcept {
  return static_cast<std::uint32_t>(id.value) >> kSlotBits;
}

Entry* allocate_entry(Shard& shard, mem::Allocator alloc, std::size_t bytes) noexcept {
  const std::uint32_t size_class = slab_class(bytes);
  if (size_class == kSlabClassCount)
    return static_cast<Entry*>(alloc.allocate(bytes, alignof(Entry)));

  if (shard.free_blocks[size_class] == nullptr) {
    constexpr std::size_t kHeaderBytes =
        (sizeof(Slab) + alignof(Entry) - 1) & ~(alignof(Entry) - 1);
    const std::size_t block_size = kSlabSizes[size_class];
    const std::size_t allocation_size = kHeaderBytes + block_size * kBlocksPerSlab;
    void* memory = alloc.allocate(allocation_size, alignof(Entry));
    if (memory == nullptr) return nullptr;
    auto* slab = ::new (memory) Slab{shard.slabs[size_class]};
    shard.slabs[size_class] = slab;
    char* blocks = static_cast<char*>(memory) + kHeaderBytes;
    for (std::size_t i = 0; i < kBlocksPerSlab; ++i) {
      auto* block = reinterpret_cast<FreeBlock*>(blocks + i * block_size);
      block->next = shard.free_blocks[size_class];
      shard.free_blocks[size_class] = block;
    }
  }
  FreeBlock* block = shard.free_blocks[size_class];
  shard.free_blocks[size_class] = block->next;
  return reinterpret_cast<Entry*>(block);
}

void release_entry(Shard& shard, mem::Allocator alloc, Entry* entry, bool pool_small) noexcept {
  const std::size_t bytes = sizeof(Entry) + static_cast<std::size_t>(entry->size) + 1;
  const std::uint32_t size_class = slab_class(bytes);
  entry->~Entry();
  if (size_class == kSlabClassCount) {
    alloc.deallocate(entry, bytes, alignof(Entry));
  } else if (pool_small) {
    auto* block = reinterpret_cast<FreeBlock*>(entry);
    block->next = shard.free_blocks[size_class];
    shard.free_blocks[size_class] = block;
  }
}

void release_slabs(Shard& shard, mem::Allocator alloc) noexcept {
  constexpr std::size_t kHeaderBytes = (sizeof(Slab) + alignof(Entry) - 1) & ~(alignof(Entry) - 1);
  for (std::uint32_t size_class = 0; size_class < kSlabClassCount; ++size_class) {
    const std::size_t bytes = kHeaderBytes + kSlabSizes[size_class] * kBlocksPerSlab;
    Slab* slab = shard.slabs[size_class];
    while (slab != nullptr) {
      Slab* next = slab->next;
      slab->~Slab();
      alloc.deallocate(slab, bytes, alignof(Entry));
      slab = next;
    }
  }
}

template <class Impl> Entry* find_id_locked(Impl& impl, InternId id) noexcept {
  const std::uint32_t low = static_cast<std::uint32_t>(id.value);
  const std::uint32_t encoded_slot = low & kSlotMask;
  const std::uint32_t generation = static_cast<std::uint32_t>(id.value >> 32);
  if (encoded_slot == 0 || generation == 0) return nullptr;
  Shard& shard = *impl.shards[id_shard(id)];
  if (encoded_slot > shard.by_id.size()) return nullptr;
  IdSlot& slot = shard.by_id[encoded_slot - 1];
  return slot.generation == generation ? slot.entry : nullptr;
}

template <class Impl>
bool assign_id_locked(Impl& impl, Shard& shard, std::uint32_t shard_index, Entry* entry) noexcept {
  if (!entry->id.empty()) return true;
  const bool reuse = shard.free_head != kNoSlot;
  if (!reuse && shard.by_id.size() >= kSlotMask) return false;
  if (!reuse && shard.by_id.size() == shard.by_id.capacity()) {
    std::lock_guard<std::mutex> growth_lock(impl.growth_mu);
    if (!shard.by_id.try_reserve(shard.by_id.size() + 1)) return false;
  }

  std::uint32_t slot_index;
  std::uint32_t generation;
  if (reuse) {
    slot_index = shard.free_head;
    IdSlot& slot = shard.by_id[slot_index];
    shard.free_head = slot.next_free;
    generation = slot.generation;
    slot.entry = entry;
    slot.next_free = kNoSlot;
  } else {
    slot_index = static_cast<std::uint32_t>(shard.by_id.size());
    generation = 1;
    HYLE_FATAL_ASSERT(shard.by_id.try_push_back(IdSlot{entry, generation, kNoSlot}));
  }
  entry->id = make_id(shard_index, slot_index, generation);
  return true;
}

template <class Impl>
Entry* find_or_add_locked(Impl& impl, mem::Allocator alloc, const HashedText& input,
                          bool permanent) noexcept {
  Shard& shard = *impl.shards[input.shard];
  const InternKey query{input.text, input.table_hash};
  if (Entry** found = shard.by_string.find(query)) {
    Entry* entry = *found;
    if (permanent) {
      entry->permanent = true;
    } else {
      // Permanent-only intern() entries do not pay for an ID slot. Materialize
      // the reverse index lazily on the first add() that actually exposes one.
      if (!assign_id_locked(impl, shard, input.shard, entry)) return nullptr;
      if (entry->refs == UINT32_MAX) return nullptr;
      ++entry->refs;
    }
    return entry;
  }

  if (shard.by_string.size() == std::numeric_limits<std::size_t>::max()) return nullptr;
  if (shard.by_string.needs_prepare_insert()) {
    // Concurrent allocator payload copies from distinct shards produced a TSan
    // report in the resize path. Growth/tombstone cleanup is logarithmically
    // rare, so give those cold copies an explicit happens-before relation.
    // try_prepare_insert also makes the following insert allocation-free, which
    // preserves StringBuf's recoverable-OOM contract.
    std::lock_guard<std::mutex> growth_lock(impl.growth_mu);
    if (!shard.by_string.try_prepare_insert()) return nullptr;
  }

  const std::size_t bytes = sizeof(Entry) + static_cast<std::size_t>(input.text.size) + 1;
  Entry* entry = allocate_entry(shard, alloc, bytes);
  if (entry == nullptr) return nullptr;
  ::new (entry) Entry{};
  char* data = reinterpret_cast<char*>(entry + 1);
  std::memcpy(data, input.text.data, input.text.size);
  data[input.text.size] = '\0';

  const string_ref owned{data, input.text.size, true};
  *entry =
      Entry{data, input.table_hash, input.text.size, permanent ? 0u : 1u, InternId{0}, permanent};
  if (!permanent && !assign_id_locked(impl, shard, input.shard, entry)) {
    release_entry(shard, alloc, entry, true);
    return nullptr;
  }
  shard.by_string.insert(InternKey{owned, input.table_hash}, entry);
  ++shard.live_size;
  return entry;
}

template <class Impl> bool retain_locked(Impl& impl, InternId id) noexcept {
  Entry* entry = find_id_locked(impl, id);
  if (entry == nullptr || entry->refs == UINT32_MAX) return false;
  ++entry->refs;
  return true;
}

template <class Impl> bool remove_locked(Impl& impl, mem::Allocator alloc, InternId id) noexcept {
  Entry* entry = find_id_locked(impl, id);
  if (entry == nullptr || entry->refs == 0) return false;
  --entry->refs;
  if (entry->refs != 0 || entry->permanent) return true;

  Shard& shard = *impl.shards[id_shard(id)];
  const std::uint32_t slot_index = (static_cast<std::uint32_t>(id.value) & kSlotMask) - 1;
  IdSlot& slot = shard.by_id[slot_index];
  slot.entry = nullptr;
  if (slot.generation != UINT32_MAX) {
    ++slot.generation;
    slot.next_free = shard.free_head;
    shard.free_head = slot_index;
  }

  const InternKey owned{string_ref{entry->data, entry->size, true}, entry->hash};
  HYLE_FATAL_ASSERT(shard.by_string.erase(owned));
  release_entry(shard, alloc, entry, true);
  HYLE_FATAL_ASSERT(shard.live_size != 0);
  --shard.live_size;
  return true;
}

} // namespace

struct StringBuf::Impl {
  explicit Impl(mem::Allocator alloc) noexcept {
    const unsigned workers = std::thread::hardware_concurrency();
    shard_bits = workers > 32 ? 6u : workers > 8 ? 5u : 4u;
    shard_count = 1u << shard_bits;
    for (auto& shard : shards) shard.emplace(alloc);
  }

  std::array<std::optional<Shard>, kMaxShardCount> shards;
  std::uint32_t shard_bits = 4;
  std::uint32_t shard_count = 16;
  std::mutex growth_mu;
};

StringBuf::StringBuf(mem::Allocator alloc) : alloc_(alloc), impl_(alloc.construct<Impl>(alloc)) {}

StringBuf::~StringBuf() {
  if (impl_ == nullptr) return;
  // Destruction requires external quiescence, like every container destructor.
  for (auto& optional_shard : impl_->shards) {
    Shard& shard = *optional_shard;
    shard.by_string.for_each([&](const InternKey&, Entry*& entry) {
      // Small entries belong to slabs; large entries remain direct allocations.
      release_entry(shard, alloc_, entry, false);
    });
    release_slabs(shard, alloc_);
  }
  alloc_.destroy(impl_);
}

namespace {

template <class Impl> auto lock_all(Impl& impl) {
  std::array<std::unique_lock<std::mutex>, kMaxShardCount> locks;
  for (std::uint32_t i = 0; i < impl.shard_count; ++i)
    locks[i] = std::unique_lock<std::mutex>(impl.shards[i]->mu);
  return locks;
}

} // namespace

InternId StringBuf::add(string_ref text) noexcept {
  if (text.size == 0) return InternId{0};
  if (impl_ == nullptr) return InternId{0};
  const HashedText input = prepare(text, impl_->shard_bits);
  std::lock_guard<std::mutex> lock(impl_->shards[input.shard]->mu);
  Entry* entry = find_or_add_locked(*impl_, alloc_, input, false);
  return entry == nullptr ? InternId{0} : entry->id;
}

std::size_t StringBuf::add_many(const string_ref* input, InternId* output,
                                std::size_t count) noexcept {
  if (count == 0) return 0;
  HYLE_FATAL_ASSERT(input != nullptr && output != nullptr);
  if (impl_ == nullptr) return 0;
  auto locks = lock_all(*impl_);
  for (std::size_t i = 0; i < count; ++i) {
    if (input[i].size == 0) {
      output[i] = InternId{0};
      continue;
    }
    Entry* entry = find_or_add_locked(*impl_, alloc_, prepare(input[i], impl_->shard_bits), false);
    if (entry == nullptr) return i;
    output[i] = entry->id;
  }
  return count;
}

bool StringBuf::retain(InternId id) noexcept {
  if (id.value == 0) return true;
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->shards[id_shard(id)]->mu);
  return retain_locked(*impl_, id);
}

std::size_t StringBuf::retain_many(const InternId* ids, std::size_t count) noexcept {
  if (count == 0) return 0;
  HYLE_FATAL_ASSERT(ids != nullptr);
  if (impl_ == nullptr) return 0;
  auto locks = lock_all(*impl_);
  for (std::size_t i = 0; i < count; ++i) {
    if (ids[i].value != 0 && !retain_locked(*impl_, ids[i])) return i;
  }
  return count;
}

bool StringBuf::remove(InternId id) noexcept {
  if (id.value == 0) return true;
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->shards[id_shard(id)]->mu);
  return remove_locked(*impl_, alloc_, id);
}

std::size_t StringBuf::remove_many(const InternId* ids, std::size_t count) noexcept {
  if (count == 0) return 0;
  HYLE_FATAL_ASSERT(ids != nullptr);
  if (impl_ == nullptr) return 0;
  auto locks = lock_all(*impl_);
  for (std::size_t i = 0; i < count; ++i) {
    if (ids[i].value != 0 && !remove_locked(*impl_, alloc_, ids[i])) return i;
  }
  return count;
}

string_ref StringBuf::lookup(InternId id) noexcept {
  if (id.value == 0) return string_ref{"", 0, true};
  if (impl_ == nullptr) return string_ref{};
  std::lock_guard<std::mutex> lock(impl_->shards[id_shard(id)]->mu);
  if (const Entry* entry = find_id_locked(*impl_, id))
    return string_ref{entry->data, entry->size, true};
  return string_ref{};
}

string_ref StringBuf::intern(string_ref text) noexcept {
  if (text.size == 0) return string_ref{"", 0, true};
  if (impl_ == nullptr) return string_ref{};
  const HashedText input = prepare(text, impl_->shard_bits);
  std::lock_guard<std::mutex> lock(impl_->shards[input.shard]->mu);
  Entry* entry = find_or_add_locked(*impl_, alloc_, input, true);
  return entry == nullptr ? string_ref{} : string_ref{entry->data, entry->size, true};
}

std::size_t StringBuf::intern_many(const string_ref* input, string_ref* output,
                                   std::size_t count) noexcept {
  if (count == 0) return 0;
  HYLE_FATAL_ASSERT(input != nullptr);
  if (impl_ == nullptr) return 0;
  auto locks = lock_all(*impl_);
  for (std::size_t i = 0; i < count; ++i) {
    string_ref result;
    if (input[i].size == 0) {
      result = string_ref{"", 0, true};
    } else {
      Entry* entry = find_or_add_locked(*impl_, alloc_, prepare(input[i], impl_->shard_bits), true);
      if (entry == nullptr) return i;
      result = string_ref{entry->data, entry->size, true};
    }
    if (output != nullptr) output[i] = result;
  }
  return count;
}

std::uint32_t StringBuf::size() const noexcept {
  if (impl_ == nullptr) return 0;
  auto locks = lock_all(*impl_);
  std::uint32_t total = 0;
  for (std::uint32_t i = 0; i < impl_->shard_count; ++i) {
    HYLE_FATAL_ASSERT(total <= UINT32_MAX - impl_->shards[i]->live_size);
    total += impl_->shards[i]->live_size;
  }
  return total;
}

StringBuf* make_string_buf(mem::Allocator alloc) {
  StringBuf* buffer = alloc.construct<StringBuf>(alloc);
  if (buffer != nullptr && !buffer->valid()) {
    alloc.destroy(buffer);
    return nullptr;
  }
  return buffer;
}

void destroy_string_buf(StringBuf* buf) noexcept {
  if (buf == nullptr) return;
  mem::Allocator alloc = buf->allocator();
  alloc.destroy(buf);
}

} // namespace hyle::container
