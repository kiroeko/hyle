#include "hyle/mem/lifetime_plan.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {
namespace {

constexpr std::uint32_t kInvalidIndex = UINT32_MAX;

template <class T>
class RawArray {
 public:
  static_assert(std::is_nothrow_default_constructible_v<T>);
  static_assert(std::is_nothrow_destructible_v<T>);

  RawArray() noexcept = default;
  RawArray(const RawArray&) = delete;
  RawArray& operator=(const RawArray&) = delete;
  ~RawArray() { reset(); }

  bool allocate(std::size_t count, AllocatorRef allocator) noexcept {
    HYLE_FATAL_ASSERT(data_ == nullptr);
    if (count == 0) return true;
    if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
      return false;
    }
    void* storage = allocator.allocate(count * sizeof(T), alignof(T));
    if (storage == nullptr) return false;
    data_ = static_cast<T*>(storage);
    count_ = count;
    allocator_ = allocator;
    for (std::size_t index = 0; index < count_; ++index) {
      ::new (static_cast<void*>(data_ + index)) T{};
    }
    return true;
  }

  void reset() noexcept {
    if (data_ == nullptr) return;
    for (std::size_t index = count_; index != 0; --index) {
      data_[index - 1].~T();
    }
    allocator_.deallocate(data_, count_ * sizeof(T), alignof(T));
    data_ = nullptr;
    count_ = 0;
  }

  T* release() noexcept {
    T* output = data_;
    data_ = nullptr;
    count_ = 0;
    return output;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  T& operator[](std::size_t index) noexcept { return data_[index]; }
  const T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
  AllocatorRef allocator_{};
};

template <class... Arrays>
bool allocate_all(std::size_t count, AllocatorRef allocator,
                  Arrays&... arrays) noexcept {
  return (arrays.allocate(count, allocator) && ...);
}

struct Group {
  bool present = false;
  bool no_reuse = false;
  std::uint32_t resource = 0;
  std::uint32_t root = 0;
  std::size_t size = 0;
  std::size_t alignment = 1;
  std::uint32_t placement_backing = 0;
  std::size_t placement_offset = 0;
  std::uint64_t begin = (std::numeric_limits<std::uint64_t>::max)();
  std::uint64_t end = 0;
};

struct CapacityKey {
  std::uint32_t resource = 0;
  std::size_t capacity = 0;
};

struct ComboKey {
  std::uint32_t resource = 0;
  std::uint8_t alignment_class = 0;
  std::size_t capacity = 0;
};

struct ResourceWork {
  bool seen = false;
  std::size_t peak = 0;
  std::size_t minimum_alignment = alignof(std::max_align_t);
  std::size_t maximum_allocation = SIZE_MAX;
  std::uint32_t backing_count = 0;
  std::uint32_t current_backing = kInvalidIndex;
  std::size_t capacity_begin = 0;
  std::size_t capacity_end = 0;
};

struct BackingWork {
  std::uint32_t resource = 0;
  std::uint32_t local_index = 0;
  std::size_t size = 0;
  std::size_t alignment = alignof(std::max_align_t);
};

struct Slot {
  std::uint32_t resource = 0;
  std::uint32_t backing = 0;
  std::uint32_t combo = kInvalidIndex;
  std::uint32_t next_free = kInvalidIndex;
  std::size_t offset = 0;
  std::size_t capacity = 0;
  std::size_t alignment = 1;
};

struct ActiveSlot {
  std::uint64_t end = 0;
  std::uint32_t slot = 0;
};

struct ResourceRecord {
  LifetimeResourcePlan plan{};
  std::size_t backing_begin = 0;
};

struct PlanImpl {
  std::size_t request_count = 0;
  std::size_t resource_count = 0;
  std::size_t backing_count = 0;
  MemoryPlacement* placements = nullptr;
  ResourceRecord* resources = nullptr;
  LifetimeBackingPlan* backings = nullptr;
  AllocatorRef allocator{};
};

struct PlannedImpl {
  std::size_t request_count = 0;
  std::size_t resource_count = 0;
  std::size_t backing_count = 0;
  MemoryPlacement* placements = nullptr;
  ResourceRecord* resources = nullptr;
  Buffer* buffers = nullptr;
  AllocatorRef allocator{};
};

static_assert(std::is_trivially_copyable_v<MemoryPlacement>);
static_assert(std::is_trivially_copyable_v<ResourceRecord>);
static_assert(std::is_trivially_copyable_v<LifetimeBackingPlan>);

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

bool valid_flags(LifetimeFlag flags) noexcept {
  constexpr std::uint8_t kKnown =
      static_cast<std::uint8_t>(LifetimeFlag::kNoReuse) |
      static_cast<std::uint8_t>(LifetimeFlag::kPersistent) |
      static_cast<std::uint8_t>(LifetimeFlag::kOutput);
  return (static_cast<std::uint8_t>(flags) & ~kKnown) == 0;
}

bool requires_no_reuse(LifetimeFlag flags) noexcept {
  return has_lifetime_flag(flags, LifetimeFlag::kNoReuse) ||
         has_lifetime_flag(flags, LifetimeFlag::kPersistent) ||
         has_lifetime_flag(flags, LifetimeFlag::kOutput);
}

bool capacity_less(const CapacityKey& left,
                   const CapacityKey& right) noexcept {
  return left.resource != right.resource ? left.resource < right.resource
                                         : left.capacity < right.capacity;
}

bool capacity_equal(const CapacityKey& left,
                    const CapacityKey& right) noexcept {
  return left.resource == right.resource && left.capacity == right.capacity;
}

bool combo_less(const ComboKey& left, const ComboKey& right) noexcept {
  if (left.resource != right.resource) return left.resource < right.resource;
  if (left.capacity != right.capacity) return left.capacity < right.capacity;
  return left.alignment_class < right.alignment_class;
}

bool combo_equal(const ComboKey& left, const ComboKey& right) noexcept {
  return left.resource == right.resource && left.capacity == right.capacity &&
         left.alignment_class == right.alignment_class;
}

std::size_t lower_capacity(const CapacityKey* keys, std::size_t begin,
                           std::size_t end, std::size_t size) noexcept {
  while (begin < end) {
    const std::size_t middle = begin + (end - begin) / 2;
    if (keys[middle].capacity < size) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

std::uint32_t find_combo(const ComboKey* combos, std::size_t count,
                         ComboKey key) noexcept {
  std::size_t begin = 0;
  std::size_t end = count;
  while (begin < end) {
    const std::size_t middle = begin + (end - begin) / 2;
    if (combo_less(combos[middle], key)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == count || !combo_equal(combos[begin], key)) {
    return kInvalidIndex;
  }
  return static_cast<std::uint32_t>(begin);
}

std::size_t find_capacity_leaf(const CapacityKey* keys, std::size_t count,
                               CapacityKey key) noexcept {
  std::size_t begin = 0;
  std::size_t end = count;
  while (begin < end) {
    const std::size_t middle = begin + (end - begin) / 2;
    if (capacity_less(keys[middle], key)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  HYLE_FATAL_ASSERT(begin < count && capacity_equal(keys[begin], key));
  return begin;
}

void update_tree(std::uint64_t* tree, std::size_t base,
                 std::size_t leaf, std::uint64_t mask) noexcept {
  std::size_t node = base + leaf;
  tree[node] = mask;
  while (node > 1) {
    node /= 2;
    tree[node] = tree[node * 2] | tree[node * 2 + 1];
  }
}

std::size_t find_first_tree(const std::uint64_t* tree, std::size_t node,
                            std::size_t node_begin, std::size_t node_end,
                            std::size_t query_begin, std::size_t query_end,
                            std::uint64_t accepted) noexcept {
  if (query_end <= node_begin || node_end <= query_begin ||
      (tree[node] & accepted) == 0) {
    return (std::numeric_limits<std::size_t>::max)();
  }
  if (node_end - node_begin == 1) return node_begin;
  const std::size_t middle = node_begin + (node_end - node_begin) / 2;
  const std::size_t left = find_first_tree(
      tree, node * 2, node_begin, middle, query_begin, query_end, accepted);
  if (left != (std::numeric_limits<std::size_t>::max)()) return left;
  return find_first_tree(tree, node * 2 + 1, middle, node_end, query_begin,
                         query_end, accepted);
}

bool active_less(ActiveSlot left, ActiveSlot right) noexcept {
  return left.end != right.end ? left.end < right.end : left.slot < right.slot;
}

void push_active(ActiveSlot* heap, std::size_t& count,
                 ActiveSlot entry) noexcept {
  std::size_t index = count++;
  heap[index] = entry;
  while (index != 0) {
    const std::size_t parent = (index - 1) / 2;
    if (!active_less(heap[index], heap[parent])) break;
    std::swap(heap[index], heap[parent]);
    index = parent;
  }
}

ActiveSlot pop_active(ActiveSlot* heap, std::size_t& count) noexcept {
  HYLE_FATAL_ASSERT(count != 0);
  ActiveSlot output = heap[0];
  --count;
  if (count == 0) return output;
  heap[0] = heap[count];
  std::size_t index = 0;
  while (true) {
    const std::size_t left = index * 2 + 1;
    if (left >= count) break;
    const std::size_t right = left + 1;
    std::size_t child = left;
    if (right < count && active_less(heap[right], heap[left])) child = right;
    if (!active_less(heap[child], heap[index])) break;
    std::swap(heap[index], heap[child]);
    index = child;
  }
  return output;
}

void destroy_plan_impl(PlanImpl* impl) noexcept {
  if (impl == nullptr) return;
  const auto allocator = impl->allocator;
  if (impl->placements != nullptr) {
    allocator.deallocate(impl->placements,
                         impl->request_count * sizeof(MemoryPlacement),
                         alignof(MemoryPlacement));
  }
  if (impl->resources != nullptr) {
    allocator.deallocate(impl->resources,
                         impl->resource_count * sizeof(ResourceRecord),
                         alignof(ResourceRecord));
  }
  if (impl->backings != nullptr) {
    allocator.deallocate(impl->backings,
                         impl->backing_count * sizeof(LifetimeBackingPlan),
                         alignof(LifetimeBackingPlan));
  }
  impl->~PlanImpl();
  allocator.deallocate(impl, sizeof(PlanImpl), alignof(PlanImpl));
}

void destroy_planned_impl(PlannedImpl* impl) noexcept {
  if (impl == nullptr) return;
  const auto allocator = impl->allocator;
  if (impl->buffers != nullptr) {
    for (std::size_t index = impl->backing_count; index != 0; --index) {
      impl->buffers[index - 1].~Buffer();
    }
    allocator.deallocate(impl->buffers, impl->backing_count * sizeof(Buffer),
                         alignof(Buffer));
  }
  if (impl->resources != nullptr) {
    allocator.deallocate(impl->resources,
                         impl->resource_count * sizeof(ResourceRecord),
                         alignof(ResourceRecord));
  }
  if (impl->placements != nullptr) {
    allocator.deallocate(impl->placements,
                         impl->request_count * sizeof(MemoryPlacement),
                         alignof(MemoryPlacement));
  }
  impl->~PlannedImpl();
  allocator.deallocate(impl, sizeof(PlannedImpl), alignof(PlannedImpl));
}

Result<void*> make_plan_impl(
    std::size_t request_count, std::size_t resource_count,
    std::size_t backing_count, RawArray<MemoryPlacement>& placements,
    RawArray<ResourceRecord>& resources,
    RawArray<LifetimeBackingPlan>& backings, AllocatorRef allocator) noexcept {
  void* storage = allocator.allocate(sizeof(PlanImpl), alignof(PlanImpl));
  if (storage == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "lifetime plan metadata allocation failed"));
  }
  auto* impl = ::new (storage) PlanImpl{
      request_count, resource_count, backing_count, placements.release(),
      resources.release(), backings.release(), allocator};
  return static_cast<void*>(impl);
}

}  // namespace

Result<LifetimePlan> LifetimePlanner::build(
    std::span<const LifetimeRequest> requests,
    std::span<const MemoryResourceRef> resources,
    AllocatorRef allocator) noexcept {
  if (!allocator.valid())
    return std::unexpected(Status::InvalidArgument(
        "lifetime planner metadata allocator is invalid"));
  const std::size_t count = requests.size();
  if (count > kInvalidIndex) {
    return std::unexpected(
        Status::OutOfRange("lifetime plan has too many requests"));
  }

  RawArray<MemoryPlacement> placements;
  RawArray<ResourceRecord> resource_records;
  RawArray<LifetimeBackingPlan> backing_plans;
  if (count == 0) {
    if (!resources.empty()) {
      return std::unexpected(Status::InvalidArgument(
          "an empty lifetime plan cannot name resources"));
    }
    auto impl = make_plan_impl(0, 0, 0, placements, resource_records,
                               backing_plans, allocator);
    if (!impl) return std::unexpected(std::move(impl).error());
    return LifetimePlan(*impl);
  }

  std::uint32_t maximum_resource = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const LifetimeRequest& request = requests[index];
    if (request.resource_slot >= count) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime resource slots must be dense"));
    }
    maximum_resource = (std::max)(maximum_resource, request.resource_slot);
    if (!is_power_of_two(request.alignment)) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime alignment must be a non-zero power of two"));
    }
    if (request.begin >= request.end) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime interval must be non-empty and half-open"));
    }
    if (request.alias_of != kNoLifetimeAlias &&
        request.alias_of >= count) {
      return std::unexpected(
          Status::InvalidArgument("lifetime alias id is out of range"));
    }
    if (request.alias_of == kNoLifetimeAlias && request.alias_offset != 0) {
      return std::unexpected(Status::InvalidArgument(
          "a root lifetime request cannot have an alias offset"));
    }
    if (!valid_flags(request.flags)) {
      return std::unexpected(
          Status::InvalidArgument("lifetime request has unknown flags"));
    }
  }
  const std::size_t resource_count =
      static_cast<std::size_t>(maximum_resource) + 1;
  if (resources.size() != resource_count) {
    return std::unexpected(Status::InvalidArgument(
        "resource count does not match dense lifetime resource slots"));
  }

  RawArray<std::uint32_t> roots;
  RawArray<std::size_t> relative_offsets;
  RawArray<std::uint8_t> states;
  RawArray<std::uint32_t> path;
  RawArray<Group> groups;
  RawArray<std::uint32_t> order;
  RawArray<ResourceWork> resource_work;
  RawArray<BackingWork> backing_work;
  RawArray<Slot> slots;
  RawArray<ActiveSlot> active;
  if (!allocate_all(count, allocator, roots, relative_offsets, states, path,
                    groups, order, slots, active) ||
      !resource_work.allocate(resource_count, allocator) ||
      !backing_work.allocate(count, allocator)) {
    return std::unexpected(Status::ResourceExhausted(
        "lifetime planner scratch allocation failed"));
  }

  for (const LifetimeRequest& request : requests) {
    resource_work[request.resource_slot].seen = true;
  }
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    if (!resource_work[resource].seen) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime resource slots must be dense"));
    }
    if (!resources[resource].valid()) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime plan resource is invalid"));
    }
    const MemoryProperties properties = resources[resource].properties();
    if (!is_power_of_two(properties.minimum_alignment)) {
      return std::unexpected(Status::InvalidArgument(
          "memory resource reports an invalid minimum alignment"));
    }
    resource_work[resource].minimum_alignment =
        properties.minimum_alignment;
    resource_work[resource].maximum_allocation =
        properties.maximum_allocation;
  }

  // Resolve arbitrary acyclic alias chains without recursion.
  for (std::uint32_t start = 0; start < count; ++start) {
    if (states[start] == 2) continue;
    std::size_t path_count = 0;
    std::uint32_t node = start;
    while (states[node] == 0) {
      states[node] = 1;
      path[path_count++] = node;
      const std::uint32_t parent = requests[node].alias_of;
      if (parent == kNoLifetimeAlias) {
        roots[node] = node;
        relative_offsets[node] = 0;
        states[node] = 2;
        --path_count;
        break;
      }
      node = parent;
    }
    if (states[node] == 1) {
      return std::unexpected(
          Status::InvalidArgument("lifetime alias graph contains a cycle"));
    }
    while (path_count != 0) {
      const std::uint32_t child = path[--path_count];
      const std::uint32_t parent = requests[child].alias_of;
      HYLE_FATAL_ASSERT(parent != kNoLifetimeAlias && states[parent] == 2);
      if (requests[child].resource_slot != requests[parent].resource_slot) {
        return std::unexpected(Status::InvalidArgument(
            "lifetime alias crosses resource slots"));
      }
      if (requests[child].alias_offset >
          (std::numeric_limits<std::size_t>::max)() -
              relative_offsets[parent]) {
        return std::unexpected(
            Status::OutOfRange("lifetime alias offset overflows"));
      }
      roots[child] = roots[parent];
      relative_offsets[child] =
          relative_offsets[parent] + requests[child].alias_offset;
      states[child] = 2;
    }
  }

  std::size_t group_count = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const LifetimeRequest& request = requests[index];
    const std::uint32_t root = roots[index];
    if ((relative_offsets[index] & (request.alignment - 1)) != 0) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime alias offset violates requested alignment"));
    }
    if (request.size > (std::numeric_limits<std::size_t>::max)() -
                           relative_offsets[index]) {
      return std::unexpected(
          Status::OutOfRange("lifetime alias extent overflows"));
    }
    const std::size_t extent = relative_offsets[index] + request.size;
    Group& group = groups[root];
    if (!group.present) {
      group.present = true;
      group.root = root;
      group.resource = request.resource_slot;
      ++group_count;
    }
    group.size = (std::max)(group.size, extent);
    group.alignment = (std::max)(group.alignment, request.alignment);
    group.begin = (std::min)(group.begin, request.begin);
    group.end = (std::max)(group.end, request.end);
    group.no_reuse = group.no_reuse || requires_no_reuse(request.flags);
  }

  std::size_t order_count = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    if (groups[index].present) order[order_count++] = index;
  }
  HYLE_FATAL_ASSERT(order_count == group_count);
  std::sort(order.data(), order.data() + order_count,
            [&groups](std::uint32_t left, std::uint32_t right) noexcept {
              const Group& a = groups[left];
              const Group& b = groups[right];
              if (a.begin != b.begin) return a.begin < b.begin;
              if (a.end != b.end) return a.end < b.end;
              return a.root < b.root;
            });

  RawArray<CapacityKey> capacities;
  RawArray<ComboKey> combos;
  if (!capacities.allocate(group_count, allocator) ||
      !combos.allocate(group_count, allocator)) {
    return std::unexpected(Status::ResourceExhausted(
        "lifetime planner index allocation failed"));
  }
  std::size_t capacity_count = 0;
  std::size_t combo_count = 0;
  for (std::size_t ordered = 0; ordered < order_count; ++ordered) {
    const Group& group = groups[order[ordered]];
    ResourceWork& resource = resource_work[group.resource];
    if (group.size > resource.maximum_allocation) {
      return std::unexpected(Status::OutOfRange(
          "lifetime alias group exceeds resource maximum allocation"));
    }
    if (group.size == 0 || group.no_reuse) continue;
    capacities[capacity_count++] = CapacityKey{group.resource, group.size};
    combos[combo_count++] =
        ComboKey{group.resource,
                 static_cast<std::uint8_t>(std::countr_zero(group.alignment)),
                 group.size};
  }
  if (capacity_count > 1) {
    std::sort(capacities.data(), capacities.data() + capacity_count,
              capacity_less);
  }
  if (capacity_count != 0) {
    capacity_count = static_cast<std::size_t>(
        std::unique(capacities.data(), capacities.data() + capacity_count,
                    capacity_equal) -
        capacities.data());
  }
  if (combo_count > 1) {
    std::sort(combos.data(), combos.data() + combo_count, combo_less);
  }
  if (combo_count != 0) {
    combo_count = static_cast<std::size_t>(
        std::unique(combos.data(), combos.data() + combo_count, combo_equal) -
        combos.data());
  }
  if (combo_count > kInvalidIndex) {
    return std::unexpected(
        Status::OutOfRange("lifetime planner has too many slot classes"));
  }

  std::size_t capacity_cursor = 0;
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    resource_work[resource].capacity_begin = capacity_cursor;
    while (capacity_cursor < capacity_count &&
           capacities[capacity_cursor].resource == resource) {
      ++capacity_cursor;
    }
    resource_work[resource].capacity_end = capacity_cursor;
  }

  std::size_t tree_base = 1;
  while (tree_base < capacity_count) {
    if (tree_base > (std::numeric_limits<std::size_t>::max)() / 2) {
      return std::unexpected(
          Status::OutOfRange("lifetime planner index overflows"));
    }
    tree_base *= 2;
  }
  if (tree_base > (std::numeric_limits<std::size_t>::max)() / 2) {
    return std::unexpected(
        Status::OutOfRange("lifetime planner index overflows"));
  }
  RawArray<std::uint64_t> tree;
  RawArray<std::uint32_t> free_heads;
  if (!tree.allocate(tree_base * 2, allocator) ||
      !free_heads.allocate(combo_count, allocator)) {
    return std::unexpected(Status::ResourceExhausted(
        "lifetime planner free-slot index allocation failed"));
  }
  for (std::size_t index = 0; index < combo_count; ++index) {
    free_heads[index] = kInvalidIndex;
  }

  auto update_combo_mask = [&](std::uint32_t combo) noexcept {
    const ComboKey& key = combos[combo];
    const std::size_t leaf = find_capacity_leaf(
        capacities.data(), capacity_count,
        CapacityKey{key.resource, key.capacity});
    std::uint64_t mask = tree[tree_base + leaf];
    const std::uint64_t bit = std::uint64_t{1} << key.alignment_class;
    if (free_heads[combo] == kInvalidIndex) {
      mask &= ~bit;
    } else {
      mask |= bit;
    }
    update_tree(tree.data(), tree_base, leaf, mask);
  };

  auto release_slot = [&](std::uint32_t slot_index) noexcept {
    Slot& slot = slots[slot_index];
    HYLE_FATAL_ASSERT(slot.combo != kInvalidIndex);
    slot.next_free = free_heads[slot.combo];
    free_heads[slot.combo] = slot_index;
    update_combo_mask(slot.combo);
  };

  auto take_slot = [&](const Group& group) noexcept -> std::uint32_t {
    const ResourceWork& resource = resource_work[group.resource];
    const std::size_t query_begin = lower_capacity(
        capacities.data(), resource.capacity_begin, resource.capacity_end,
        group.size);
    if (query_begin == resource.capacity_end) return kInvalidIndex;
    const std::uint8_t alignment_class =
        static_cast<std::uint8_t>(std::countr_zero(group.alignment));
    const std::uint64_t accepted =
        (std::numeric_limits<std::uint64_t>::max)() << alignment_class;
    const std::size_t leaf = find_first_tree(
        tree.data(), 1, 0, tree_base, query_begin, resource.capacity_end,
        accepted);
    if (leaf == (std::numeric_limits<std::size_t>::max)()) {
      return kInvalidIndex;
    }
    const std::uint64_t available = tree[tree_base + leaf] & accepted;
    HYLE_FATAL_ASSERT(available != 0);
    const std::uint8_t chosen_alignment =
        static_cast<std::uint8_t>(std::countr_zero(available));
    const CapacityKey& capacity = capacities[leaf];
    const std::uint32_t combo = find_combo(
        combos.data(), combo_count,
        ComboKey{capacity.resource, chosen_alignment, capacity.capacity});
    HYLE_FATAL_ASSERT(combo != kInvalidIndex &&
                      free_heads[combo] != kInvalidIndex);
    const std::uint32_t output = free_heads[combo];
    free_heads[combo] = slots[output].next_free;
    slots[output].next_free = kInvalidIndex;
    update_combo_mask(combo);
    return output;
  };

  std::size_t active_count = 0;
  std::size_t slot_count = 0;
  std::size_t backing_count = 0;
  for (std::size_t ordered = 0; ordered < order_count; ++ordered) {
    Group& group = groups[order[ordered]];
    while (active_count != 0 && active[0].end <= group.begin) {
      release_slot(pop_active(active.data(), active_count).slot);
    }

    if (group.size == 0) {
      group.placement_backing = 0;
      group.placement_offset = 0;
      continue;
    }

    // no_reuse prevents this placement from becoming available afterward; it
    // does not forbid safely inheriting storage whose earlier lifetime ended.
    std::uint32_t selected = take_slot(group);
    if (selected == kInvalidIndex) {
      ResourceWork& resource = resource_work[group.resource];
      const std::size_t mask = group.alignment - 1;
      std::size_t offset = 0;
      BackingWork* backing = nullptr;
      if (resource.current_backing != kInvalidIndex) {
        BackingWork& current = backing_work[resource.current_backing];
        if (current.size <=
            (std::numeric_limits<std::size_t>::max)() - mask) {
          const std::size_t candidate = (current.size + mask) & ~mask;
          if (candidate <= resource.maximum_allocation &&
              group.size <= resource.maximum_allocation - candidate) {
            offset = candidate;
            backing = &current;
          }
        }
      }
      if (backing == nullptr) {
        HYLE_FATAL_ASSERT(backing_count < count &&
                          resource.backing_count < kInvalidIndex);
        const std::uint32_t global_index =
            static_cast<std::uint32_t>(backing_count++);
        resource.current_backing = global_index;
        BackingWork& created = backing_work[global_index];
        created.resource = group.resource;
        created.local_index = resource.backing_count++;
        created.alignment = resource.minimum_alignment;
        backing = &created;
      }
      HYLE_FATAL_ASSERT(backing->resource == group.resource &&
                        offset <= resource.maximum_allocation &&
                        group.size <= resource.maximum_allocation - offset);
      const std::size_t old_backing_size = backing->size;
      const std::size_t new_backing_size = offset + group.size;
      backing->size = new_backing_size;
      backing->alignment = (std::max)(backing->alignment, group.alignment);
      const std::size_t growth = new_backing_size - old_backing_size;
      if (growth > (std::numeric_limits<std::size_t>::max)() -
                       resource.peak) {
        return std::unexpected(
            Status::OutOfRange("lifetime placement size overflows"));
      }
      resource.peak += growth;
      selected = static_cast<std::uint32_t>(slot_count++);
      Slot& slot = slots[selected];
      slot.resource = group.resource;
      slot.backing = backing->local_index;
      slot.offset = offset;
      slot.capacity = group.size;
      slot.alignment = group.alignment;
      if (!group.no_reuse) {
        slot.combo = find_combo(
            combos.data(), combo_count,
            ComboKey{group.resource,
                     static_cast<std::uint8_t>(
                         std::countr_zero(group.alignment)),
                     group.size});
        HYLE_FATAL_ASSERT(slot.combo != kInvalidIndex);
      }
    }
    Slot& slot = slots[selected];
    HYLE_FATAL_ASSERT(slot.resource == group.resource &&
                      slot.capacity >= group.size &&
                      slot.alignment >= group.alignment);
    group.placement_backing = slot.backing;
    group.placement_offset = slot.offset;
    if (!group.no_reuse) {
      push_active(active.data(), active_count,
                  ActiveSlot{group.end, selected});
    }
  }

  // A zero-only dense resource still owns one zero-sized backing so its views
  // preserve the explicitly selected provider identity.
  for (std::size_t resource_index = 0; resource_index < resource_count;
       ++resource_index) {
    ResourceWork& resource = resource_work[resource_index];
    if (resource.backing_count != 0) continue;
    HYLE_FATAL_ASSERT(backing_count < count);
    const std::uint32_t global_index =
        static_cast<std::uint32_t>(backing_count++);
    resource.current_backing = global_index;
    resource.backing_count = 1;
    backing_work[global_index] = BackingWork{
        static_cast<std::uint32_t>(resource_index), 0, 0,
        resource.minimum_alignment};
  }

  if (!placements.allocate(count, allocator) ||
      !resource_records.allocate(resource_count, allocator) ||
      !backing_plans.allocate(backing_count, allocator)) {
    return std::unexpected(Status::ResourceExhausted(
        "lifetime plan result allocation failed"));
  }
  for (std::size_t index = 0; index < count; ++index) {
    const LifetimeRequest& request = requests[index];
    const Group& group = groups[roots[index]];
    HYLE_FATAL_ASSERT(relative_offsets[index] <= group.size &&
                      group.placement_offset <=
                          (std::numeric_limits<std::size_t>::max)() -
                              relative_offsets[index]);
    placements[index] = MemoryPlacement{
        request.resource_slot,
        group.placement_backing,
        group.placement_offset + relative_offsets[index],
        request.size,
        request.alignment,
    };
  }
  std::size_t backing_begin = 0;
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    resource_records[resource] = ResourceRecord{
        LifetimeResourcePlan{resource_work[resource].peak,
                             resource_work[resource].backing_count},
        backing_begin};
    backing_begin += resource_work[resource].backing_count;
  }
  HYLE_FATAL_ASSERT(backing_begin == backing_count);
  for (std::size_t index = 0; index < backing_count; ++index) {
    const BackingWork& backing = backing_work[index];
    const std::size_t flat_index =
        resource_records[backing.resource].backing_begin +
        backing.local_index;
    HYLE_FATAL_ASSERT(flat_index < backing_count);
    backing_plans[flat_index] =
        LifetimeBackingPlan{backing.size, backing.alignment};
  }
  auto impl = make_plan_impl(count, resource_count, backing_count, placements,
                             resource_records, backing_plans, allocator);
  if (!impl) return std::unexpected(std::move(impl).error());
  return LifetimePlan(*impl);
}

LifetimePlan::LifetimePlan(LifetimePlan&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

LifetimePlan& LifetimePlan::operator=(LifetimePlan&& other) noexcept {
  if (this != &other) {
    destroy_plan_impl(static_cast<PlanImpl*>(impl_));
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

LifetimePlan::~LifetimePlan() {
  destroy_plan_impl(static_cast<PlanImpl*>(impl_));
}

std::size_t LifetimePlan::request_count() const noexcept {
  return valid() ? static_cast<PlanImpl*>(impl_)->request_count : 0;
}

std::size_t LifetimePlan::resource_count() const noexcept {
  return valid() ? static_cast<PlanImpl*>(impl_)->resource_count : 0;
}

Result<MemoryPlacement> LifetimePlan::placement(
    std::size_t request_id) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("lifetime plan is invalid"));
  }
  const PlanImpl& plan = *static_cast<PlanImpl*>(impl_);
  if (request_id >= plan.request_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime request id is out of range"));
  }
  return plan.placements[request_id];
}

Result<LifetimeResourcePlan> LifetimePlan::resource_plan(
    std::size_t resource_slot) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("lifetime plan is invalid"));
  }
  const PlanImpl& plan = *static_cast<PlanImpl*>(impl_);
  if (resource_slot >= plan.resource_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime resource slot is out of range"));
  }
  return plan.resources[resource_slot].plan;
}

Result<LifetimeBackingPlan> LifetimePlan::backing_plan(
    std::size_t resource_slot, std::size_t backing_index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("lifetime plan is invalid"));
  }
  const PlanImpl& plan = *static_cast<PlanImpl*>(impl_);
  if (resource_slot >= plan.resource_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime resource slot is out of range"));
  }
  const ResourceRecord& resource = plan.resources[resource_slot];
  if (backing_index >= resource.plan.backing_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime backing index is out of range"));
  }
  return plan.backings[resource.backing_begin + backing_index];
}

Result<PlannedBuffers> LifetimePlan::instantiate(
    std::span<const MemoryResourceRef> resources) const noexcept {
  if (!valid())
    return std::unexpected(
        Status::FailedPrecondition("lifetime plan is invalid"));
  return instantiate(resources,
                     static_cast<const PlanImpl*>(impl_)->allocator);
}

Result<PlannedBuffers> LifetimePlan::instantiate(
    std::span<const MemoryResourceRef> resources,
    AllocatorRef allocator) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("lifetime plan is invalid"));
  }
  if (!allocator.valid())
    return std::unexpected(Status::InvalidArgument(
        "planned-buffer metadata allocator is invalid"));
  const PlanImpl& plan = *static_cast<PlanImpl*>(impl_);
  if (resources.size() != plan.resource_count) {
    return std::unexpected(Status::InvalidArgument(
        "resource count does not match lifetime plan"));
  }
  for (std::size_t resource_index = 0;
       resource_index < plan.resource_count; ++resource_index) {
    const MemoryResourceRef resource = resources[resource_index];
    if (!resource.valid()) {
      return std::unexpected(Status::InvalidArgument(
          "lifetime plan resource is invalid"));
    }
    const MemoryProperties properties = resource.properties();
    if (!is_power_of_two(properties.minimum_alignment)) {
      return std::unexpected(Status::InvalidArgument(
          "memory resource reports an invalid minimum alignment"));
    }
    const ResourceRecord& record = plan.resources[resource_index];
    for (std::size_t backing = 0; backing < record.plan.backing_count;
         ++backing) {
      if (plan.backings[record.backing_begin + backing].size >
          properties.maximum_allocation) {
        return std::unexpected(Status::OutOfRange(
            "lifetime backing exceeds resource maximum allocation"));
      }
    }
  }

  void* storage = allocator.allocate(sizeof(PlannedImpl), alignof(PlannedImpl));
  if (storage == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "planned-buffer metadata allocation failed"));
  }
  auto* output = ::new (storage) PlannedImpl{
      plan.request_count, plan.resource_count, plan.backing_count, nullptr,
      nullptr, nullptr, allocator};

  RawArray<MemoryPlacement> copied_placements;
  RawArray<ResourceRecord> copied_resources;
  RawArray<Buffer> buffers;
  if (!copied_placements.allocate(plan.request_count, allocator) ||
      !copied_resources.allocate(plan.resource_count, allocator) ||
      !buffers.allocate(plan.backing_count, allocator)) {
    output->~PlannedImpl();
    allocator.deallocate(output, sizeof(PlannedImpl), alignof(PlannedImpl));
    return std::unexpected(Status::ResourceExhausted(
        "planned-buffer arrays allocation failed"));
  }
  if (plan.request_count != 0) {
    std::memcpy(copied_placements.data(), plan.placements,
                plan.request_count * sizeof(MemoryPlacement));
  }
  if (plan.resource_count != 0) {
    std::memcpy(copied_resources.data(), plan.resources,
                plan.resource_count * sizeof(ResourceRecord));
  }
  for (std::size_t resource = 0; resource < plan.resource_count; ++resource) {
    const ResourceRecord& record = plan.resources[resource];
    for (std::size_t backing = 0; backing < record.plan.backing_count;
         ++backing) {
      const std::size_t flat_index = record.backing_begin + backing;
      const LifetimeBackingPlan& backing_plan = plan.backings[flat_index];
      auto created = Buffer::create(resources[resource], backing_plan.size,
                                    backing_plan.alignment);
      if (!created) {
        output->~PlannedImpl();
        allocator.deallocate(output, sizeof(PlannedImpl), alignof(PlannedImpl));
        return std::unexpected(std::move(created).error());
      }
      buffers[flat_index] = std::move(*created);
    }
  }
  output->placements = copied_placements.release();
  output->resources = copied_resources.release();
  output->buffers = buffers.release();
  return PlannedBuffers(output);
}

PlannedBuffers::PlannedBuffers(PlannedBuffers&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

PlannedBuffers& PlannedBuffers::operator=(PlannedBuffers&& other) noexcept {
  if (this != &other) {
    reset();
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

PlannedBuffers::~PlannedBuffers() { reset(); }

std::size_t PlannedBuffers::request_count() const noexcept {
  return valid() ? static_cast<PlannedImpl*>(impl_)->request_count : 0;
}

std::size_t PlannedBuffers::resource_count() const noexcept {
  return valid() ? static_cast<PlannedImpl*>(impl_)->resource_count : 0;
}

Result<std::size_t> PlannedBuffers::backing_count(
    std::size_t resource_slot) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("planned buffers are invalid"));
  }
  const PlannedImpl& buffers = *static_cast<PlannedImpl*>(impl_);
  if (resource_slot >= buffers.resource_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime resource slot is out of range"));
  }
  return buffers.resources[resource_slot].plan.backing_count;
}

Result<BufferView> PlannedBuffers::view(std::size_t request_id) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("planned buffers are invalid"));
  }
  PlannedImpl& buffers = *static_cast<PlannedImpl*>(impl_);
  if (request_id >= buffers.request_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime request id is out of range"));
  }
  const MemoryPlacement& placement = buffers.placements[request_id];
  const ResourceRecord& resource = buffers.resources[placement.resource_slot];
  return buffers.buffers[resource.backing_begin + placement.backing_index]
      .view()
      .subview(
      placement.offset, placement.size);
}

Result<ConstBufferView> PlannedBuffers::view(
    std::size_t request_id) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("planned buffers are invalid"));
  }
  const PlannedImpl& buffers = *static_cast<PlannedImpl*>(impl_);
  if (request_id >= buffers.request_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime request id is out of range"));
  }
  const MemoryPlacement& placement = buffers.placements[request_id];
  const ResourceRecord& resource = buffers.resources[placement.resource_slot];
  const Buffer& backing =
      buffers.buffers[resource.backing_begin + placement.backing_index];
  return backing.view().subview(placement.offset, placement.size);
}

Result<BufferView> PlannedBuffers::backing(
    std::size_t resource_slot, std::size_t backing_index) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("planned buffers are invalid"));
  }
  PlannedImpl& buffers = *static_cast<PlannedImpl*>(impl_);
  if (resource_slot >= buffers.resource_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime resource slot is out of range"));
  }
  const ResourceRecord& resource = buffers.resources[resource_slot];
  if (backing_index >= resource.plan.backing_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime backing index is out of range"));
  }
  return buffers.buffers[resource.backing_begin + backing_index].view();
}

Result<ConstBufferView> PlannedBuffers::backing(
    std::size_t resource_slot, std::size_t backing_index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("planned buffers are invalid"));
  }
  const PlannedImpl& buffers = *static_cast<PlannedImpl*>(impl_);
  if (resource_slot >= buffers.resource_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime resource slot is out of range"));
  }
  const ResourceRecord& resource = buffers.resources[resource_slot];
  if (backing_index >= resource.plan.backing_count) {
    return std::unexpected(
        Status::OutOfRange("lifetime backing index is out of range"));
  }
  const Buffer& backing =
      buffers.buffers[resource.backing_begin + backing_index];
  return backing.view();
}

void PlannedBuffers::reset() noexcept {
  destroy_planned_impl(static_cast<PlannedImpl*>(impl_));
  impl_ = nullptr;
}

}  // namespace hyle::mem
