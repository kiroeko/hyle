#include "hyle/mem/range_allocator.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"

namespace hyle::mem {
namespace {

using NodeIndex = std::uint32_t;
inline constexpr NodeIndex kNoNode =
    (std::numeric_limits<NodeIndex>::max)();
inline constexpr NodeIndex kAllocatedNode = kNoNode - 1;
inline constexpr unsigned kSecondLevelBits = 3;
inline constexpr unsigned kSecondLevelCount = 1u << kSecondLevelBits;
inline constexpr unsigned kFirstLevelCount =
    std::numeric_limits<std::uint64_t>::digits - kSecondLevelBits + 1;
inline constexpr std::uint64_t kSmallRange =
    std::uint64_t{1} << kSecondLevelBits;

struct RangeNode {
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  NodeIndex previous_physical = kNoNode;
  NodeIndex next_physical = kNoNode;
  NodeIndex previous_free = kNoNode;
  NodeIndex next_free = kNoNode;
};

static_assert(sizeof(RangeNode) == 32,
              "range metadata must remain one half cache line");
static_assert(std::is_trivially_destructible_v<RangeNode>);

bool is_free(const RangeNode& node) noexcept {
  return node.previous_free != kAllocatedNode;
}

struct RangeAllocatorImpl {
  RangeAllocatorImpl(AllocatorRef selected_allocator,
                     std::uint64_t selected_capacity,
                     std::uint32_t selected_max_allocations) noexcept
      : metadata_allocator(selected_allocator),
        capacity(selected_capacity),
        max_allocations(selected_max_allocations) {}

  AllocatorRef metadata_allocator;
  RangeNode* nodes = nullptr;
  NodeIndex* free_nodes = nullptr;
  std::size_t node_count = 0;
  std::size_t node_bytes = 0;
  std::size_t free_node_bytes = 0;
  std::size_t free_node_count = 0;
  std::uint64_t capacity = 0;
  std::uint64_t allocated = 0;
  std::uint64_t peak_allocated = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t generation = 1;
  std::uint64_t references = 1;  // the RangeAllocator owner plus leases
  std::uint32_t live_allocations = 0;
  std::uint32_t max_allocations = 0;
  NodeIndex physical_head = kNoNode;
  NodeIndex bins[kFirstLevelCount][kSecondLevelCount]{};
  std::uint8_t second_level_nonempty[kFirstLevelCount]{};
  std::uint64_t first_level_nonempty = 0;
  bool owner_alive = true;
};

RangeAllocatorImpl* as_impl(void* pointer) noexcept {
  return static_cast<RangeAllocatorImpl*>(pointer);
}

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t* output) noexcept {
  if (lhs != 0 && rhs > (std::numeric_limits<std::size_t>::max)() / lhs) {
    return false;
  }
  *output = lhs * rhs;
  return true;
}

void clear_index(RangeAllocatorImpl& impl) noexcept {
  for (auto& row : impl.bins) {
    for (NodeIndex& head : row) head = kNoNode;
  }
  for (std::uint8_t& map : impl.second_level_nonempty) map = 0;
  impl.first_level_nonempty = 0;
}

void mapping_insert(std::uint64_t size, unsigned* first,
                    unsigned* second) noexcept {
  HYLE_ASSUME(size != 0 && first != nullptr && second != nullptr);
  if (size < kSmallRange) {
    *first = 0;
    *second = static_cast<unsigned>(size);
    return;
  }
  const unsigned msb = static_cast<unsigned>(std::bit_width(size) - 1);
  *first = msb - (kSecondLevelBits - 1);
  *second = static_cast<unsigned>(size >> (msb - kSecondLevelBits)) ^
            kSecondLevelCount;
}

bool mapping_search(std::uint64_t size, unsigned* first,
                    unsigned* second) noexcept {
  if (size >= kSmallRange) {
    const unsigned msb = static_cast<unsigned>(std::bit_width(size) - 1);
    const std::uint64_t rounding =
        (std::uint64_t{1} << (msb - kSecondLevelBits)) - 1;
    if (size > (std::numeric_limits<std::uint64_t>::max)() - rounding) {
      return false;
    }
    size += rounding;
  }
  mapping_insert(size, first, second);
  return *first < kFirstLevelCount;
}

void insert_free(RangeAllocatorImpl& impl, NodeIndex index) noexcept {
  RangeNode& node = impl.nodes[index];
  HYLE_ASSUME(is_free(node) && node.size != 0);
  unsigned first = 0;
  unsigned second = 0;
  mapping_insert(node.size, &first, &second);
  HYLE_ASSUME(first < kFirstLevelCount && second < kSecondLevelCount);
  node.previous_free = kNoNode;
  node.next_free = impl.bins[first][second];
  if (node.next_free != kNoNode) {
    impl.nodes[node.next_free].previous_free = index;
  }
  impl.bins[first][second] = index;
  impl.first_level_nonempty |= std::uint64_t{1} << first;
  impl.second_level_nonempty[first] |=
      static_cast<std::uint8_t>(1u << second);
}

void remove_free(RangeAllocatorImpl& impl, NodeIndex index) noexcept {
  RangeNode& node = impl.nodes[index];
  HYLE_ASSUME(is_free(node) && node.size != 0);
  unsigned first = 0;
  unsigned second = 0;
  mapping_insert(node.size, &first, &second);
  if (node.previous_free != kNoNode) {
    impl.nodes[node.previous_free].next_free = node.next_free;
  } else {
    HYLE_ASSUME(impl.bins[first][second] == index);
    impl.bins[first][second] = node.next_free;
  }
  if (node.next_free != kNoNode) {
    impl.nodes[node.next_free].previous_free = node.previous_free;
  }
  if (impl.bins[first][second] == kNoNode) {
    impl.second_level_nonempty[first] &=
        static_cast<std::uint8_t>(~(1u << second));
    if (impl.second_level_nonempty[first] == 0) {
      impl.first_level_nonempty &= ~(std::uint64_t{1} << first);
    }
  }
  node.previous_free = kNoNode;
  node.next_free = kNoNode;
}

NodeIndex find_suitable(RangeAllocatorImpl& impl,
                        std::uint64_t required) noexcept {
  unsigned first = 0;
  unsigned second = 0;
  if (!mapping_search(required, &first, &second)) return kNoNode;

  std::uint32_t second_map =
      static_cast<std::uint32_t>(impl.second_level_nonempty[first]) &
      (std::uint32_t{0xff} << second);
  if (second_map == 0) {
    const std::uint64_t first_map =
        impl.first_level_nonempty &
        (~std::uint64_t{0} << static_cast<unsigned>(first + 1));
    if (first_map == 0) return kNoNode;
    first = static_cast<unsigned>(std::countr_zero(first_map));
    second_map = impl.second_level_nonempty[first];
  }
  second = static_cast<unsigned>(std::countr_zero(second_map));
  const NodeIndex result = impl.bins[first][second];
  HYLE_ASSUME(result != kNoNode && impl.nodes[result].size >= required);
  return result;
}

bool aligned_start(const RangeNode& node, std::uint64_t size,
                   std::uint64_t alignment,
                   std::uint64_t* output) noexcept {
  const std::uint64_t mask = alignment - 1;
  if (node.offset > (std::numeric_limits<std::uint64_t>::max)() - mask) {
    return false;
  }
  const std::uint64_t aligned = (node.offset + mask) & ~mask;
  const std::uint64_t prefix = aligned - node.offset;
  if (prefix > node.size || size > node.size - prefix) return false;
  *output = aligned;
  return true;
}

NodeIndex acquire_node(RangeAllocatorImpl& impl) noexcept {
  HYLE_ASSUME(impl.free_node_count != 0);
  return impl.free_nodes[--impl.free_node_count];
}

void release_node(RangeAllocatorImpl& impl, NodeIndex index) noexcept {
  HYLE_ASSUME(index != kNoNode && impl.free_node_count < impl.node_count);
  impl.nodes[index] = {};
  impl.free_nodes[impl.free_node_count++] = index;
}

void initialize_empty(RangeAllocatorImpl& impl) noexcept {
  clear_index(impl);
  impl.free_node_count = 0;
  for (std::size_t index = impl.node_count; index-- > 1;) {
    impl.nodes[index] = {};
    impl.free_nodes[impl.free_node_count++] =
        static_cast<NodeIndex>(index);
  }
  impl.nodes[0] = RangeNode{.offset = 0,
                            .size = impl.capacity,
                            .previous_physical = kNoNode,
                            .next_physical = kNoNode,
                            .previous_free = kNoNode,
                            .next_free = kNoNode};
  impl.physical_head = 0;
  impl.allocated = 0;
  impl.live_allocations = 0;
  insert_free(impl, 0);
}

void destroy_impl(RangeAllocatorImpl* impl) noexcept {
  AllocatorRef allocator = impl->metadata_allocator;
  RangeNode* nodes = impl->nodes;
  NodeIndex* free_nodes = impl->free_nodes;
  const std::size_t node_bytes = impl->node_bytes;
  const std::size_t free_node_bytes = impl->free_node_bytes;
  impl->~RangeAllocatorImpl();
  allocator.deallocate(nodes, node_bytes, alignof(RangeNode));
  allocator.deallocate(free_nodes, free_node_bytes, alignof(NodeIndex));
  allocator.deallocate(impl, sizeof(RangeAllocatorImpl),
                       alignof(RangeAllocatorImpl));
}

void release_reference(RangeAllocatorImpl* impl) noexcept {
  HYLE_ASSUME(impl->references != 0);
  if (--impl->references == 0) destroy_impl(impl);
}

void invalidate_owner(RangeAllocatorImpl* impl) noexcept {
  if (impl == nullptr) return;
  HYLE_ASSUME(impl->owner_alive);
  impl->owner_alive = false;
  release_reference(impl);
}

NodeIndex allocate_node(RangeAllocatorImpl& impl, std::uint64_t size,
                        std::uint64_t alignment,
                        std::uint64_t* allocated_offset) noexcept {
  NodeIndex selected = kNoNode;
  const std::uint64_t mask = alignment - 1;
  if (size <= (std::numeric_limits<std::uint64_t>::max)() - mask) {
    selected = find_suitable(impl, size + mask);
  }

  std::uint64_t aligned = 0;
  if (selected == kNoNode) {
    // Address-sensitive completeness fallback: a range shorter than the
    // conservative size+alignment-1 bound may still already be aligned.
    for (NodeIndex current = impl.physical_head; current != kNoNode;
         current = impl.nodes[current].next_physical) {
      if (is_free(impl.nodes[current]) &&
          aligned_start(impl.nodes[current], size, alignment, &aligned)) {
        selected = current;
        break;
      }
    }
    if (selected == kNoNode) return kNoNode;
  } else {
    const bool fits =
        aligned_start(impl.nodes[selected], size, alignment, &aligned);
    HYLE_ASSUME(fits);
  }

  RangeNode original = impl.nodes[selected];
  remove_free(impl, selected);
  const std::uint64_t prefix_size = aligned - original.offset;
  const std::uint64_t suffix_offset = aligned + size;
  const std::uint64_t suffix_size =
      original.size - prefix_size - size;

  NodeIndex prefix = kNoNode;
  NodeIndex suffix = kNoNode;
  if (prefix_size != 0) prefix = acquire_node(impl);
  if (suffix_size != 0) suffix = acquire_node(impl);

  RangeNode& allocation = impl.nodes[selected];
  allocation = RangeNode{.offset = aligned,
                         .size = size,
                         .previous_physical =
                             prefix != kNoNode ? prefix
                                               : original.previous_physical,
                         .next_physical = suffix != kNoNode
                                              ? suffix
                                              : original.next_physical,
                         .previous_free = kAllocatedNode,
                         .next_free = kNoNode};

  if (prefix != kNoNode) {
    impl.nodes[prefix] =
        RangeNode{.offset = original.offset,
                  .size = prefix_size,
                  .previous_physical = original.previous_physical,
                  .next_physical = selected,
                  .previous_free = kNoNode,
                  .next_free = kNoNode};
    if (original.previous_physical != kNoNode) {
      impl.nodes[original.previous_physical].next_physical = prefix;
    } else {
      impl.physical_head = prefix;
    }
    insert_free(impl, prefix);
  } else if (original.previous_physical != kNoNode) {
    impl.nodes[original.previous_physical].next_physical = selected;
  } else {
    impl.physical_head = selected;
  }

  if (suffix != kNoNode) {
    impl.nodes[suffix] =
        RangeNode{.offset = suffix_offset,
                  .size = suffix_size,
                  .previous_physical = selected,
                  .next_physical = original.next_physical,
                  .previous_free = kNoNode,
                  .next_free = kNoNode};
    if (original.next_physical != kNoNode) {
      impl.nodes[original.next_physical].previous_physical = suffix;
    }
    insert_free(impl, suffix);
  } else if (original.next_physical != kNoNode) {
    impl.nodes[original.next_physical].previous_physical = selected;
  }

  impl.allocated += size;
  impl.peak_allocated = (std::max)(impl.peak_allocated, impl.allocated);
  ++impl.live_allocations;
  ++impl.references;
  *allocated_offset = aligned;
  return selected;
}

void free_node(RangeAllocatorImpl& impl, NodeIndex index) noexcept {
  RangeNode& released = impl.nodes[index];
  HYLE_ASSUME(!is_free(released) && released.size != 0 &&
              impl.live_allocations != 0 && impl.allocated >= released.size);
  impl.allocated -= released.size;
  --impl.live_allocations;
  released.previous_free = kNoNode;

  NodeIndex merged = index;
  if (released.previous_physical != kNoNode &&
      is_free(impl.nodes[released.previous_physical])) {
    const NodeIndex previous = released.previous_physical;
    RangeNode& previous_node = impl.nodes[previous];
    remove_free(impl, previous);
    previous_node.size += released.size;
    previous_node.next_physical = released.next_physical;
    if (released.next_physical != kNoNode) {
      impl.nodes[released.next_physical].previous_physical = previous;
    }
    release_node(impl, index);
    merged = previous;
  }

  RangeNode& merged_node = impl.nodes[merged];
  if (merged_node.next_physical != kNoNode &&
      is_free(impl.nodes[merged_node.next_physical])) {
    const NodeIndex next = merged_node.next_physical;
    RangeNode& next_node = impl.nodes[next];
    remove_free(impl, next);
    merged_node.size += next_node.size;
    merged_node.next_physical = next_node.next_physical;
    if (next_node.next_physical != kNoNode) {
      impl.nodes[next_node.next_physical].previous_physical = merged;
    }
    release_node(impl, next);
  }
  insert_free(impl, merged);
}

}  // namespace

Result<RangeAllocator> RangeAllocator::create(
    std::uint64_t capacity, RangeAllocatorOptions options) noexcept {
  constexpr std::uint32_t kMaximumLiveAllocations =
      (kAllocatedNode - 1u) / 2u;
  if (capacity == 0) {
    return std::unexpected(
        Status::InvalidArgument("range allocator capacity must be non-zero"));
  }
  if (options.max_allocations == 0 ||
      options.max_allocations > kMaximumLiveAllocations) {
    return std::unexpected(Status::InvalidArgument(
        "range allocator max_allocations is zero or not representable"));
  }
  if (!options.metadata_allocator.valid()) {
    return std::unexpected(Status::InvalidArgument(
        "range allocator metadata allocator is invalid"));
  }

  const std::uint64_t node_count64 =
      std::uint64_t{2} * options.max_allocations + 1;
  if (node_count64 > (std::numeric_limits<std::size_t>::max)()) {
    return std::unexpected(Status::ResourceExhausted());
  }
  const std::size_t node_count = static_cast<std::size_t>(node_count64);
  std::size_t node_bytes = 0;
  std::size_t free_node_bytes = 0;
  if (!checked_multiply(node_count, sizeof(RangeNode), &node_bytes) ||
      !checked_multiply(node_count, sizeof(NodeIndex), &free_node_bytes)) {
    return std::unexpected(Status::ResourceExhausted());
  }
  const std::size_t maximum_size =
      (std::numeric_limits<std::size_t>::max)();
  if (node_bytes > maximum_size - sizeof(RangeAllocatorImpl) ||
      free_node_bytes >
          maximum_size - sizeof(RangeAllocatorImpl) - node_bytes) {
    return std::unexpected(Status::ResourceExhausted());
  }

  void* impl_storage = options.metadata_allocator.allocate(
      sizeof(RangeAllocatorImpl), alignof(RangeAllocatorImpl));
  if (impl_storage == nullptr) {
    return std::unexpected(Status::ResourceExhausted());
  }
  auto* impl = ::new (impl_storage) RangeAllocatorImpl(
      options.metadata_allocator, capacity, options.max_allocations);
  impl->node_count = node_count;
  impl->node_bytes = node_bytes;
  impl->free_node_bytes = free_node_bytes;

  impl->nodes = static_cast<RangeNode*>(
      options.metadata_allocator.allocate(node_bytes, alignof(RangeNode)));
  if (impl->nodes == nullptr) {
    impl->~RangeAllocatorImpl();
    options.metadata_allocator.deallocate(
        impl_storage, sizeof(RangeAllocatorImpl), alignof(RangeAllocatorImpl));
    return std::unexpected(Status::ResourceExhausted());
  }
  impl->free_nodes = static_cast<NodeIndex*>(options.metadata_allocator.allocate(
      free_node_bytes, alignof(NodeIndex)));
  if (impl->free_nodes == nullptr) {
    options.metadata_allocator.deallocate(impl->nodes, node_bytes,
                                          alignof(RangeNode));
    impl->~RangeAllocatorImpl();
    options.metadata_allocator.deallocate(
        impl_storage, sizeof(RangeAllocatorImpl), alignof(RangeAllocatorImpl));
    return std::unexpected(Status::ResourceExhausted());
  }
  for (std::size_t index = 0; index < node_count; ++index) {
    ::new (static_cast<void*>(impl->nodes + index)) RangeNode;
  }
  impl->metadata_bytes = sizeof(RangeAllocatorImpl) + node_bytes +
                         free_node_bytes;
  initialize_empty(*impl);
  return RangeAllocator(impl);
}

RangeAllocator::RangeAllocator(RangeAllocator&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

RangeAllocator& RangeAllocator::operator=(RangeAllocator&& other) noexcept {
  if (this != &other) {
    invalidate_owner(as_impl(impl_));
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

RangeAllocator::~RangeAllocator() { invalidate_owner(as_impl(impl_)); }

Result<RangeAllocation> RangeAllocator::allocate(std::uint64_t size,
                                                  std::uint64_t alignment) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("range allocator is invalid"));
  }
  if (size == 0) {
    return std::unexpected(
        Status::InvalidArgument("range allocation size must be non-zero"));
  }
  if (!std::has_single_bit(alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "range allocation alignment must be a non-zero power of two"));
  }
  RangeAllocatorImpl& impl = *as_impl(impl_);
  HYLE_ASSUME(impl.owner_alive);
  if (impl.live_allocations == impl.max_allocations) {
    return std::unexpected(Status::ResourceExhausted());
  }
  if (impl.references == (std::numeric_limits<std::uint64_t>::max)()) {
    return std::unexpected(Status::ResourceExhausted());
  }
  std::uint64_t offset = 0;
  const NodeIndex node = allocate_node(impl, size, alignment, &offset);
  if (node == kNoNode) {
    return std::unexpected(Status::ResourceExhausted());
  }
  return RangeAllocation(&impl, node, impl.generation, offset);
}

void RangeAllocator::reset() noexcept {
  if (!valid()) return;
  RangeAllocatorImpl& impl = *as_impl(impl_);
  HYLE_FATAL_ASSERT(impl.owner_alive &&
                    impl.generation !=
                        (std::numeric_limits<std::uint64_t>::max)());
  ++impl.generation;
  initialize_empty(impl);
}

RangeAllocatorStats RangeAllocator::stats() const noexcept {
  if (!valid()) return {};
  const RangeAllocatorImpl& impl = *as_impl(impl_);
  return RangeAllocatorStats{
      .capacity_bytes = impl.capacity,
      .allocated_bytes = impl.allocated,
      .free_bytes = impl.capacity - impl.allocated,
      .peak_allocated_bytes = impl.peak_allocated,
      .metadata_bytes = impl.metadata_bytes,
      .outstanding_handles = impl.references - 1u,
      .live_allocations = impl.live_allocations,
      .max_allocations = impl.max_allocations,
  };
}

RangeAllocatorDetailedStats RangeAllocator::detailed_stats() const noexcept {
  RangeAllocatorDetailedStats result{.summary = stats()};
  if (!valid()) return result;
  const RangeAllocatorImpl& impl = *as_impl(impl_);
  for (NodeIndex current = impl.physical_head; current != kNoNode;
       current = impl.nodes[current].next_physical) {
    const RangeNode& node = impl.nodes[current];
    if (is_free(node)) {
      ++result.free_range_count;
      result.largest_free_range =
          (std::max)(result.largest_free_range, node.size);
    }
  }
  return result;
}

bool RangeAllocation::valid() const noexcept {
  if (impl_ == nullptr) return false;
  const RangeAllocatorImpl& impl = *as_impl(impl_);
  if (!impl.owner_alive || generation_ != impl.generation ||
      node_ >= impl.node_count) {
    return false;
  }
  const RangeNode& node = impl.nodes[node_];
  return !is_free(node) && node.offset == offset_;
}

std::uint64_t RangeAllocation::size() const noexcept {
  HYLE_ASSUME(impl_ != nullptr);
  return as_impl(impl_)->nodes[node_].size;
}

void RangeAllocation::reset() noexcept {
  if (impl_ == nullptr) return;
  RangeAllocatorImpl* impl = as_impl(impl_);
  if (impl->owner_alive && generation_ == impl->generation) {
    HYLE_ASSUME(node_ < impl->node_count);
    const RangeNode& node = impl->nodes[node_];
    HYLE_ASSUME(!is_free(node) && node.offset == offset_);
    free_node(*impl, node_);
  }
  release_reference(impl);
  impl_ = nullptr;
}

}  // namespace hyle::mem
