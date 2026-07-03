#include "hyle/container/string_buf.h"

#include <cstring>
#include <mutex>

#include "hyle/container/swiss_table.h"

namespace hyle::container {

// One owned, NUL-terminated allocation per unique string + its live ref count.
// Trivially copyable so it can be a SwissTable value (slots relocate by memcpy).
struct Entry {
  char* data;           // owned: alloc.allocate(size + 1), data[size] == '\0'
  std::uint32_t size;   // length without the terminator
  std::uint32_t refs;   // live add()/retain() count
};

struct StringBuf::Impl {
  explicit Impl(mem::Allocator a) : table(a) {}

  std::mutex mu;
  SwissTable<std::uint64_t, Entry> table;  // hash -> entry (reverse index)
};

StringBuf::StringBuf(mem::Allocator alloc)
    : alloc_(alloc), impl_(alloc.construct<Impl>(alloc)) {
  HYLE_FATAL_ASSERT(impl_ != nullptr);
}

StringBuf::~StringBuf() {
  if (impl_ == nullptr) return;
  // Release every interned string through the pool's allocator. SwissTable has
  // no iterator, so for_each (added in container-04) walks the live slots; the
  // table's own storage is freed by ~Impl right after.
  impl_->table.for_each([&](std::uint64_t, Entry& e) {
    alloc_.deallocate(e.data, e.size + 1);
  });
  alloc_.destroy(impl_);
}

namespace {
// Find-or-create under a held lock. Returns the entry whose data pointer is
// stable for the string's lifetime. Shared by add() and intern() so neither
// does a second lock/find (intern used to be lookup(add(s)) -- two locks, two
// finds, two hashes; the benchmark showed that losing to a plain std interner).
Entry* find_or_add_locked(SwissTable<std::uint64_t, Entry>& table,
                          mem::Allocator alloc, string_ref s,
                          std::uint64_t h) noexcept {
  if (Entry* e = table.find(h)) {
    // Astronomically rare with XXH3, but a hash collision would silently alias
    // two distinct strings. Catch it as a bug instead of corrupting data.
    HYLE_FATAL_ASSERT(e->size == s.size &&
                      std::memcmp(e->data, s.data, s.size) == 0);
    ++e->refs;
    return e;
  }
  char* buf = static_cast<char*>(alloc.allocate(s.size + 1));
  HYLE_FATAL_ASSERT(buf != nullptr);
  std::memcpy(buf, s.data, s.size);
  buf[s.size] = '\0';
  table.insert(h, Entry{buf, s.size, 1});
  // The Entry is copied into the slot; re-find to hand back the slot's address.
  // Only on the cold (new-string) path -- the hot dup path returns above.
  return table.find(h);
}
}  // namespace

InternId StringBuf::add(string_ref s) noexcept {
  if (s.size == 0) return InternId{0};  // empty string is implicit, never stored
  std::uint64_t h = hash_bytes(s.data, s.size);
  std::lock_guard<std::mutex> lock(impl_->mu);
  find_or_add_locked(impl_->table, alloc_, s, h);
  return InternId{h};
}

void StringBuf::retain(InternId id) noexcept {
  if (id.value == 0) return;
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (Entry* e = impl_->table.find(id.value)) ++e->refs;
}

void StringBuf::remove(InternId id) noexcept {
  if (id.value == 0) return;
  std::lock_guard<std::mutex> lock(impl_->mu);
  Entry* e = impl_->table.find(id.value);
  if (e == nullptr) return;
  if (--e->refs == 0) {
    char* data = e->data;
    std::uint32_t size = e->size;
    impl_->table.erase(id.value);  // invalidates e
    alloc_.deallocate(data, size + 1);
  }
}

string_ref StringBuf::lookup(InternId id) noexcept {
  if (id.value == 0) return string_ref{"", 0, true};
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (Entry* e = impl_->table.find(id.value))
    return string_ref{e->data, e->size, true};
  return string_ref{};
}

string_ref StringBuf::intern(string_ref s) noexcept {
  if (s.size == 0) return string_ref{"", 0, true};
  std::uint64_t h = hash_bytes(s.data, s.size);
  std::lock_guard<std::mutex> lock(impl_->mu);
  Entry* e = find_or_add_locked(impl_->table, alloc_, s, h);
  return string_ref{e->data, e->size, true};  // one lock, one find (hot path)
}

std::uint32_t StringBuf::size() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return static_cast<std::uint32_t>(impl_->table.size());
}

StringBuf* make_string_buf(mem::Allocator alloc) {
  return alloc.construct<StringBuf>(alloc);
}

void destroy_string_buf(StringBuf* buf) noexcept {
  if (buf == nullptr) return;
  mem::Allocator alloc = buf->allocator();  // free via the same allocator
  alloc.destroy(buf);
}

}  // namespace hyle::container
