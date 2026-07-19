#include "hyle/container/string_buf.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#include "hyle/container/carray.h"
#include "hyle/container/swiss_table.h"

namespace hyle::container {

// One owned, NUL-terminated allocation per unique string + its live ref count.
// Trivially copyable so it can be a SwissTable value (slots relocate by memcpy).
struct Entry {
  char* data;           // owned: alloc.allocate(size + 1), data[size] == '\0'
  std::uint32_t size;   // length without the terminator
  std::uint32_t refs;   // releasable add()/retain() count
  InternId id;
  bool permanent;       // intern() keeps it until StringBuf destruction
};

struct IdSlot {
  Entry* entry;
  std::uint32_t generation;
  std::uint32_t next_free;
};

struct StringBuf::Impl {
  explicit Impl(mem::Allocator a) : by_string(a), by_id(a) {}

  std::mutex mu;
  // Content equality remains the authoritative identity. A generation-tagged
  // slot makes ID lookup direct while allowing bounded slot reuse. Stale IDs
  // fail the generation check instead of naming a later string.
  SwissTable<string_ref, Entry*> by_string;
  CArray<IdSlot> by_id;
  std::uint32_t free_head = UINT32_MAX;
};

StringBuf::StringBuf(mem::Allocator alloc)
    : alloc_(alloc), impl_(alloc.construct<Impl>(alloc)) {}

StringBuf::~StringBuf() {
  if (impl_ == nullptr) return;
  // Release every interned string through the pool's allocator. SwissTable has
  // no iterator, so for_each (added in container-04) walks the live slots; the
  // table's own storage is freed by ~Impl right after.
  impl_->by_string.for_each([&](string_ref, Entry*& e) {
    const std::size_t bytes =
        sizeof(Entry) + static_cast<std::size_t>(e->size) + 1;
    e->~Entry();
    alloc_.deallocate(e, bytes, alignof(Entry));
  });
  alloc_.destroy(impl_);
}

namespace {
// Find-or-create under a held lock. Returns the entry whose data pointer is
// stable for the string's lifetime. Shared by add() and intern() so neither
// does a second lock/find (intern used to be lookup(add(s)) -- two locks, two
// finds, two hashes; the benchmark showed that losing to a plain std interner).
template <class Impl>
Entry* find_or_add_locked(Impl& impl, mem::Allocator alloc,
                          string_ref s, bool permanent) noexcept {
  if (Entry** found = impl.by_string.find(s)) {
    Entry* e = *found;
    if (permanent) {
      e->permanent = true;
    } else {
      HYLE_FATAL_ASSERT(e->refs != UINT32_MAX);
      ++e->refs;
    }
    return e;
  }
  const bool reuse = impl.free_head != UINT32_MAX;
  if (impl.by_string.size() == std::numeric_limits<std::size_t>::max() ||
      (!reuse && impl.by_id.size() == UINT32_MAX) ||
      !impl.by_string.reserve(impl.by_string.size() + 1) ||
      (!reuse && !impl.by_id.try_reserve(impl.by_id.size() + 1)))
    return nullptr;
  const std::size_t bytes =
      sizeof(Entry) + static_cast<std::size_t>(s.size) + 1;
  void* memory = alloc.allocate(bytes, alignof(Entry));
  if (memory == nullptr) return nullptr;
  Entry* entry = ::new (memory) Entry{};
  char* buf = reinterpret_cast<char*>(entry + 1);
  std::memcpy(buf, s.data, s.size);
  buf[s.size] = '\0';
  std::uint32_t slot_index;
  std::uint32_t generation;
  if (reuse) {
    slot_index = impl.free_head;
    IdSlot& slot = impl.by_id[slot_index];
    impl.free_head = slot.next_free;
    generation = slot.generation;
    slot.entry = entry;
    slot.next_free = UINT32_MAX;
  } else {
    slot_index = static_cast<std::uint32_t>(impl.by_id.size());
    generation = 1;
    // Capacity was reserved above, therefore this cannot fail.
    HYLE_FATAL_ASSERT(
        impl.by_id.try_push_back(IdSlot{entry, generation, UINT32_MAX}));
  }
  const InternId id{(static_cast<std::uint64_t>(generation) << 32) |
                    (static_cast<std::uint64_t>(slot_index) + 1)};
  const string_ref owned{buf, s.size, true};
  *entry = Entry{buf, s.size, permanent ? 0u : 1u, id, permanent};
  impl.by_string.insert(owned, entry);
  return entry;
}

template <class Impl>
Entry* find_id_locked(Impl& impl, InternId id) noexcept {
  const std::uint32_t encoded_index = static_cast<std::uint32_t>(id.value);
  const std::uint32_t generation = static_cast<std::uint32_t>(id.value >> 32);
  if (encoded_index == 0 || generation == 0 ||
      encoded_index > impl.by_id.size())
    return nullptr;
  IdSlot& slot = impl.by_id[encoded_index - 1];
  return slot.generation == generation ? slot.entry : nullptr;
}
}  // namespace

InternId StringBuf::add(string_ref s) noexcept {
  if (s.size == 0) return InternId{0};  // empty string is implicit, never stored
  if (impl_ == nullptr) return InternId{0};
  std::lock_guard<std::mutex> lock(impl_->mu);
  Entry* entry = find_or_add_locked(*impl_, alloc_, s, false);
  return entry == nullptr ? InternId{0} : entry->id;
}

bool StringBuf::retain(InternId id) noexcept {
  if (id.value == 0) return true;
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (Entry* e = find_id_locked(*impl_, id)) {
    HYLE_FATAL_ASSERT(e->refs != UINT32_MAX);
    ++e->refs;
    return true;
  }
  return false;
}

bool StringBuf::remove(InternId id) noexcept {
  if (id.value == 0) return true;
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mu);
  Entry* e = find_id_locked(*impl_, id);
  if (e == nullptr) return false;
  if (e->refs == 0) return false;
  --e->refs;
  if (e->refs == 0 && !e->permanent) {
    const string_ref owned{e->data, e->size, true};
    const std::size_t bytes =
        sizeof(Entry) + static_cast<std::size_t>(e->size) + 1;
    const std::uint32_t slot_index =
        static_cast<std::uint32_t>(id.value) - 1;
    IdSlot& slot = impl_->by_id[slot_index];
    slot.entry = nullptr;
    if (slot.generation != UINT32_MAX) {
      ++slot.generation;
      slot.next_free = impl_->free_head;
      impl_->free_head = slot_index;
    }
    impl_->by_string.erase(owned);
    e->~Entry();
    alloc_.deallocate(e, bytes, alignof(Entry));
  }
  return true;
}

string_ref StringBuf::lookup(InternId id) noexcept {
  if (id.value == 0) return string_ref{"", 0, true};
  if (impl_ == nullptr) return string_ref{};
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (const Entry* e = find_id_locked(*impl_, id)) {
    return string_ref{e->data, e->size, true};
  }
  return string_ref{};
}

string_ref StringBuf::intern(string_ref s) noexcept {
  if (s.size == 0) return string_ref{"", 0, true};
  if (impl_ == nullptr) return string_ref{};
  std::lock_guard<std::mutex> lock(impl_->mu);
  Entry* e = find_or_add_locked(*impl_, alloc_, s, true);
  if (e == nullptr) return string_ref{};
  return string_ref{e->data, e->size, true};  // one lock, one find (hot path)
}

std::size_t StringBuf::intern_many(const string_ref* input, string_ref* output,
                                   std::size_t count) noexcept {
  if (count == 0) return 0;
  HYLE_FATAL_ASSERT(input != nullptr);
  if (impl_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(impl_->mu);
  for (std::size_t i = 0; i < count; ++i) {
    string_ref result;
    if (input[i].size == 0) {
      result = string_ref{"", 0, true};
    } else {
      Entry* entry = find_or_add_locked(*impl_, alloc_, input[i], true);
      if (entry == nullptr) return i;
      result = string_ref{entry->data, entry->size, true};
    }
    if (output != nullptr) output[i] = result;
  }
  return count;
}

std::uint32_t StringBuf::size() const noexcept {
  if (impl_ == nullptr) return 0;
  std::lock_guard<std::mutex> lock(impl_->mu);
  HYLE_FATAL_ASSERT(impl_->by_string.size() <= UINT32_MAX);
  return static_cast<std::uint32_t>(impl_->by_string.size());
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
  mem::Allocator alloc = buf->allocator();  // free via the same allocator
  alloc.destroy(buf);
}

}  // namespace hyle::container
