#include "hyle/container/swiss_table.h"

#include <cstring>

#include "hyle/container/probe.h"
#include "hyle/macros.h"

// Type-erased cold path for SwissTable (conventions §7). The heavy resize loop
// (allocate, memset, rehash-reinsert, free) is compiled exactly ONCE here rather
// than per <K,V>; the typed shell hands it slot_size/align + a hash-of-slot thunk
// and gets the new storage back. The hot find/insert/erase stay inlined in the
// header -- type-erasing those would force an indirect hash/eq call on every
// probe, which is unacceptable for cheap keys.

namespace hyle::container::detail {
namespace {

// First empty-or-deleted slot along the probe sequence for hash h. Pure control-
// byte logic (no key/value type), so it lives in the core. Mirrors the shell's
// inline copy used by insert().
std::size_t core_find_insert_slot(const ctrl_t* ctrl, std::size_t mask,
                                  std::uint64_t h) noexcept {
  std::size_t pos = h1(h) & mask;
  std::size_t stride = 0;
  for (;;) {
    Group g(ctrl + pos);
    if (BitMask m = g.match_empty_or_deleted()) return (pos + m.lowest()) & mask;
    stride += kGroupWidth;
    pos = (pos + stride) & mask;
  }
}

// ctrl[i] = c, mirroring the first kGroupWidth bytes at the tail so a group load
// near the end wraps consistently.
void core_set_ctrl(ctrl_t* ctrl, std::size_t cap, std::size_t i,
                   ctrl_t c) noexcept {
  ctrl[i] = c;
  if (i < kGroupWidth) ctrl[cap + i] = c;
}

}  // namespace

void swiss_resize(SwissStorage& s, std::size_t new_cap, std::size_t slot_size,
                  std::size_t slot_align, mem::Allocator alloc,
                  std::uint64_t (*hash_slot)(const void*, const void*),
                  const void* hash_state) {
  ctrl_t* old_ctrl = s.ctrl;
  auto* old_slots = static_cast<std::byte*>(s.slots);
  std::size_t old_cap = s.cap;

  auto* new_ctrl = static_cast<ctrl_t*>(
      alloc.allocate(new_cap + kGroupWidth, alignof(ctrl_t)));
  HYLE_FATAL_ASSERT(new_ctrl != nullptr);
  auto* new_slots =
      static_cast<std::byte*>(alloc.allocate(new_cap * slot_size, slot_align));
  HYLE_FATAL_ASSERT(new_slots != nullptr);
  std::memset(new_ctrl, static_cast<int>(kEmpty), new_cap + kGroupWidth);

  std::size_t new_mask = new_cap - 1;
  std::size_t new_size = 0;
  for (std::size_t i = 0; i < old_cap; ++i) {
    if (ctrl_is_full(old_ctrl[i])) {
      const void* slot = old_slots + i * slot_size;
      std::uint64_t h = hash_slot(slot, hash_state);
      std::size_t idx = core_find_insert_slot(new_ctrl, new_mask, h);
      core_set_ctrl(new_ctrl, new_cap, idx, h2(h));
      std::memcpy(new_slots + idx * slot_size, slot, slot_size);
      ++new_size;
    }
  }

  if (old_ctrl != nullptr) {
    alloc.deallocate(old_ctrl, old_cap + kGroupWidth, alignof(ctrl_t));
    alloc.deallocate(old_slots, old_cap * slot_size, slot_align);
  }

  s.ctrl = new_ctrl;
  s.slots = new_slots;
  s.cap = new_cap;
  s.mask = new_mask;
  s.size = new_size;
  s.growth_left = new_cap * 7 / 8 - new_size;
}

void swiss_free(SwissStorage& s, std::size_t slot_size, std::size_t slot_align,
                mem::Allocator alloc) {
  if (s.ctrl != nullptr) {
    alloc.deallocate(s.ctrl, s.cap + kGroupWidth, alignof(ctrl_t));
    alloc.deallocate(s.slots, s.cap * slot_size, slot_align);
  }
}

}  // namespace hyle::container::detail
