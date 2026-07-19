#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/container/string_ref.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Reference-counted string interning pool (clay string_buf). Stores each unique
// string ONCE and hands back a stable pointer that lives until the matching
// number of remove()s. Two interns of equal content return the same pointer, so
// downstream code can compare interned strings by pointer identity and a hashmap
// keyed on them needs no per-lookup string hashing.
//
// Interned strings never move (each unique string gets its own allocation), so a
// pointer cached from intern()/lookup() stays valid for the string's lifetime.
// Thread-safe: all four mutators take an internal lock (interning is not a hot
// path; one lock keeps it simple and correct).
//
// The SwissTable reverse index + the lock live in the .cc via a pimpl, so this
// header stays ABI-clean (no std::mutex / SwissTable in the layout) and pulls in
// no <mutex>. Export is per-METHOD (not the whole class): the class holds a
// mem::Allocator (a header-only, non-exported type), so exporting the class
// whole would trip MSVC C4251. Methods that cross the DLL carry the macro; the
// layout is instantiated client-side (conventions §3.5).
// Opaque handle to an interned string, returned by StringBuf::add and consumed by
// retain/remove/lookup. It is a pool-local generation-tagged slot id, not a
// content hash: hash values are not identities and may collide. A removed
// handle never aliases a later string even when its slot is reused. add() is the
// only sanctioned way to mint one. Trivially copyable; zero is the empty string.
struct InternId {
  std::uint64_t value = 0;
  friend bool operator==(InternId, InternId) noexcept = default;
  bool empty() const noexcept { return value == 0; }
};

class StringBuf {
 public:
  HYLE_CONTAINER_API explicit StringBuf(
      mem::Allocator alloc = mem::system_allocator());
  HYLE_CONTAINER_API ~StringBuf();
  HYLE_DISABLE_COPY_AND_MOVE(StringBuf);

  // Adds `s` (bumping its refcount if already present). Returns its InternId
  // handle for lookup()/retain()/remove(). The empty string always maps to the
  // empty InternId (value 0) and is never stored.
  HYLE_CONTAINER_API InternId add(string_ref s) noexcept;

  // Bumps the refcount of an already-interned string by its handle (== add of the
  // same content, without needing the bytes again).
  HYLE_CONTAINER_API bool retain(InternId id) noexcept;

  // Drops one reference. When the count reaches the number of add()s, the
  // string's storage is reclaimed and its pointer becomes invalid.
  HYLE_CONTAINER_API bool remove(InternId id) noexcept;

  // String for `id`, or empty if absent. The result is always null-terminated.
  HYLE_CONTAINER_API string_ref lookup(InternId id) noexcept;

  // Permanently interns `s` until pool destruction and returns its stable,
  // null-terminated view. Repeated calls do not grow a hidden refcount. Use
  // add()/remove() instead when individual lifetime reclamation is required.
  HYLE_CONTAINER_API string_ref intern(string_ref s) noexcept;

  // Permanently intern a batch under one lock. `output` may be null when the
  // caller only wants to prewarm the pool. Returns the processed count; a short
  // result reports allocation failure after preserving the completed prefix.
  HYLE_CONTAINER_API std::size_t intern_many(const string_ref* input,
                                             string_ref* output,
                                             std::size_t count) noexcept;

  // Number of distinct strings currently held (excludes the implicit empty).
  HYLE_CONTAINER_API std::uint32_t size() const noexcept;
  bool valid() const noexcept { return impl_ != nullptr; }

  // The injected allocator (also used to free interned strings and the pool).
  // Inline (no DLL hop), so destroy_string_buf reaches it without a friend.
  mem::Allocator allocator() const noexcept { return alloc_; }

 private:
  struct Impl;
  mem::Allocator alloc_;
  Impl* impl_;
};

// make/destroy factory (conventions §4): allocation responsibility stays inside
// the DLL so a client built against a different CRT can't free DLL-owned memory.
// Equivalent to `alloc.construct<StringBuf>(alloc)` / `alloc.destroy(p)`; offered
// for clay API parity and cross-DLL handoff. The pool remembers its allocator,
// so destroy_string_buf needs only the pointer.
HYLE_CONTAINER_API StringBuf* make_string_buf(
    mem::Allocator alloc = mem::system_allocator());
HYLE_CONTAINER_API void destroy_string_buf(StringBuf* buf) noexcept;

}  // namespace hyle::container
