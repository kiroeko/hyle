#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/container/hash.h"  // HYLE_CONTAINER_API
#include "hyle/container/string_ref.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Thread-safe string interning pool. Stores each unique
// string ONCE and hands back a stable pointer that lives until the matching
// number of release()s. Two interns of equal content return the same pointer, so
// downstream code can compare interned strings by pointer identity and a hashmap
// keyed on them needs no per-lookup string hashing.
//
// Interned strings never move (short entries occupy stable slab blocks; long
// entries use direct allocations), so a pointer cached from intern()/lookup()
// stays valid for the string's lifetime.
// Thread-safe: construction selects 16/32/64 content shards once from the host's
// hardware parallelism, so unrelated strings and IDs do not serialize on one
// process-wide pool lock and the hot path does no repeated topology detection.
//
// The SwissTable reverse index + lock live in the .cc via a pimpl, keeping this
// public header small and avoiding implementation containers in each source
// consumer. Compiled methods carry the component visibility macro for shared
// C++ delivery; HYLE does not promise a stable cross-version C++ ABI.
// Opaque handle to an interned string, returned by StringInterner::acquire and
// consumed by retain/release/lookup. It is a pool-local generation-tagged slot
// id, not a content hash: hash values are not identities and may collide. A
// released handle never aliases a later string even when its slot is reused.
// acquire() is the only sanctioned way to mint one. Trivially copyable; zero is
// the empty string.
struct StringId {
  std::uint64_t value = 0;
  friend bool operator==(StringId, StringId) noexcept = default;
  bool empty() const noexcept { return value == 0; }
};

class StringInterner {
 public:
  HYLE_CONTAINER_API explicit StringInterner(
      mem::Allocator alloc = mem::system_allocator());
  HYLE_CONTAINER_API ~StringInterner();
  StringInterner(const StringInterner&) = delete;
  StringInterner& operator=(const StringInterner&) = delete;
  StringInterner(StringInterner&&) = delete;
  StringInterner& operator=(StringInterner&&) = delete;

  // Acquires one reference to `s` (bumping its refcount if already present).
  // Returns its StringId for lookup()/retain()/release(). The empty string maps
  // to StringId{0}; for non-empty input StringId{0} reports allocation failure.
  HYLE_CONTAINER_API StringId acquire(StringView s) noexcept;

  // Batch acquire. Input and output spans must have equal size. Returns the
  // completed prefix; a short result reports allocation failure. Bounded chunks
  // hash each input once and lock only the shards touched by that chunk.
  HYLE_CONTAINER_API std::size_t acquire_many(
      std::span<const StringView> input, std::span<StringId> output) noexcept;

  // Bumps the refcount of an already-interned string by its handle (== acquire
  // of the same content, without needing the bytes again).
  HYLE_CONTAINER_API bool retain(StringId id) noexcept;

  // Batch retain/release return the completed prefix. A short result means the
  // next handle was stale, absent, or could not be retained.
  HYLE_CONTAINER_API std::size_t retain_many(
      std::span<const StringId> ids) noexcept;

  // Releases one reference. When the count reaches zero, the string's storage is
  // reclaimed and its pointer becomes invalid unless it was permanently
  // interned.
  HYLE_CONTAINER_API bool release(StringId id) noexcept;
  HYLE_CONTAINER_API std::size_t release_many(
      std::span<const StringId> ids) noexcept;

  // String for `id`, or empty if absent. The result is always null-terminated.
  HYLE_CONTAINER_API StringView lookup(StringId id) noexcept;

  // Permanently interns `s` until pool destruction and returns its stable,
  // null-terminated view. Repeated calls do not grow a hidden refcount. Use
  // acquire()/release() instead when individual lifetime reclamation is
  // required.
  HYLE_CONTAINER_API StringView intern(StringView s) noexcept;

  // Permanently intern a batch. Input and output spans must have equal size;
  // returned views borrow stable interner-owned bytes. `prewarm` is the explicit
  // no-output form. Both return the completed prefix, with a short result
  // reporting allocation failure after preserving that prefix.
  HYLE_CONTAINER_API std::size_t intern_many(
      std::span<const StringView> input, std::span<StringView> output) noexcept;
  HYLE_CONTAINER_API std::size_t prewarm(
      std::span<const StringView> input) noexcept;

  // Number of distinct strings currently held (excludes the implicit empty).
  HYLE_CONTAINER_API std::uint32_t size() const noexcept;
  bool valid() const noexcept { return impl_ != nullptr; }

  // The injected allocator (also used to free interned strings and the pool).
  // Inline (no DLL hop), so destroy_string_interner reaches it without a friend.
  mem::Allocator allocator() const noexcept { return alloc_; }

 private:
  std::size_t intern_many_impl(std::span<const StringView> input,
                               std::span<StringView> output) noexcept;
  struct Impl;
  mem::Allocator alloc_;
  Impl* impl_;
};

// make/destroy factory (conventions §4): allocation responsibility stays inside
// the DLL so a client built against a different CRT can't free DLL-owned memory.
// Equivalent to `alloc.construct<StringInterner>(alloc)` / `alloc.destroy(p)`.
// The pool remembers its allocator, so destroy_string_interner needs only the
// pointer.
HYLE_CONTAINER_API StringInterner* make_string_interner(
    mem::Allocator alloc = mem::system_allocator());
HYLE_CONTAINER_API void destroy_string_interner(StringInterner* interner) noexcept;

}  // namespace hyle::container
