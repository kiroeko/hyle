#include "hyle/c/container.h"

#include "c_options.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "hyle/mem/allocator.h"
#include "../error/status_ffi_source_helpers.h"

namespace {

namespace hc = hyle::container;

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

bool valid_backing(const hyle_allocator &backing) noexcept {
  return backing.alloc != nullptr && backing.free != nullptr &&
         backing.realloc != nullptr;
}

bool checked_align_up(std::size_t value, std::size_t alignment,
                      std::size_t &out) noexcept {
  if (!is_power_of_two(alignment) || value > SIZE_MAX - (alignment - 1))
    return false;
  out = (value + alignment - 1) & ~(alignment - 1);
  return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t &out) noexcept {
  if (right > SIZE_MAX - left)
    return false;
  out = left + right;
  return true;
}

bool valid_builtin_scalar_compare_size(
    const hyle_container_btree_options &options) noexcept {
  if (options.compare == hyle_container_compare_i32 ||
      options.compare == hyle_container_compare_u32)
    return options.key_size == sizeof(std::uint32_t);
  if (options.compare == hyle_container_compare_i64 ||
      options.compare == hyle_container_compare_u64)
    return options.key_size == sizeof(std::uint64_t);
  return true;
}

bool valid_options(const hyle_container_btree_options &options,
                   bool require_value) noexcept {
  if (options.reserved != 0 || options.key_size == 0 ||
      !is_power_of_two(options.key_alignment) ||
      !is_power_of_two(options.value_alignment) || options.compare == nullptr ||
      !valid_builtin_scalar_compare_size(options) ||
      !valid_backing(options.backing) ||
      (require_value && options.value_size == 0) ||
      (!require_value && options.value_size != 0))
    return false;

  const std::size_t entry_alignment =
      std::max(options.key_alignment, options.value_alignment);
  std::size_t value_offset = options.key_size;
  std::size_t end = options.key_size;
  std::size_t stride = 0;
  if (options.value_size != 0 &&
      (!checked_align_up(options.key_size, options.value_alignment,
                         value_offset) ||
       !checked_add(value_offset, options.value_size, end)))
    return false;
  if (!checked_align_up(end, entry_alignment, stride) || stride == 0)
    return false;
  // The implementation guarantees at least five entries per node. Reject a
  // layout whose minimum node would overflow even before child pointers.
  return stride <= (SIZE_MAX - 4096) / 6;
}

bool decode_btree_options(const hyle_container_btree_options *input,
                          hyle_container_btree_options &output) noexcept {
  output =
      hyle_container_btree_options HYLE_CONTAINER_BTREE_OPTIONS_INIT;
  std::uint32_t declared_size = 0;
  if (!hc::c_detail::read_required_options_prefix(
          input, HYLE_CONTAINER_BTREE_OPTIONS_REQUIRED_SIZE, output,
          declared_size))
    return false;
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_btree_options, context), output.context);
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_btree_options, backing), output.backing);
  return true;
}

void *c_allocate(void *state, std::size_t size,
                 std::size_t alignment) noexcept {
  const auto *backing = static_cast<const hyle_allocator *>(state);
  return backing->alloc(backing->state, size, alignment);
}

void c_deallocate(void *state, void *pointer, std::size_t size,
                  std::size_t alignment) noexcept {
  const auto *backing = static_cast<const hyle_allocator *>(state);
  backing->free(backing->state, pointer, size, alignment);
}

void *c_reallocate(void *state, void *pointer, std::size_t old_size,
                   std::size_t new_size, std::size_t alignment) noexcept {
  const auto *backing = static_cast<const hyle_allocator *>(state);
  return backing->realloc(backing->state, pointer, old_size, new_size,
                          alignment);
}

hyle::mem::Allocator to_cpp(hyle_allocator *backing) noexcept {
  return hyle::mem::Allocator(c_allocate, c_deallocate, c_reallocate, backing);
}

class RawBTree {
public:
  struct Node {
    Node *parent;
    std::uint16_t count;
    std::uint16_t position;
    bool leaf;
  };

  struct Location {
    Node *node;
    std::uint16_t index;
  };

  enum class InsertResult : std::uint8_t {
    kInserted,
    kAlreadyPresent,
    kAssigned,
    kOutOfMemory,
  };

  RawBTree(const hyle_container_btree_options &input,
           hyle::mem::Allocator allocator) noexcept
      : options_(input), allocator_(allocator) {
    entry_alignment_ =
        std::max(options_.key_alignment, options_.value_alignment);
    value_offset_ = options_.key_size;
    std::size_t end = options_.key_size;
    if (options_.value_size != 0) {
      value_offset_ = align_up(options_.key_size, options_.value_alignment);
      end = value_offset_ + options_.value_size;
    }
    entry_stride_ = align_up(end, entry_alignment_);
    entries_offset_ = align_up(sizeof(Node), entry_alignment_);
    const std::size_t raw_capacity =
        kTargetLeafBytes > entries_offset_
            ? (kTargetLeafBytes - entries_offset_) / entry_stride_
            : 0;
    const std::size_t at_least_five = std::max<std::size_t>(5, raw_capacity);
    capacity_ = static_cast<std::uint16_t>(
        (at_least_five & 1) != 0 ? at_least_five : at_least_five - 1);
    min_keys_ = static_cast<std::uint16_t>(capacity_ / 2);
    min_degree_ = static_cast<std::uint16_t>(min_keys_ + 1);
    children_offset_ =
        align_up(entries_offset_ + capacity_ * entry_stride_, alignof(Node *));
    leaf_bytes_ = entries_offset_ + capacity_ * entry_stride_;
    internal_bytes_ =
        children_offset_ +
        (static_cast<std::size_t>(capacity_) + 1) * sizeof(Node *);
    node_alignment_ =
        std::max({alignof(Node), entry_alignment_, alignof(Node *)});
  }

  ~RawBTree() { clear(); }

  RawBTree(const RawBTree &) = delete;
  RawBTree &operator=(const RawBTree &) = delete;

  std::size_t size() const noexcept { return size_; }
  const hyle_container_btree_options &options() const noexcept {
    return options_;
  }

  Location begin() const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0};
    while (!node->leaf)
      node = child(node, 0);
    return {node, 0};
  }

  Location last() const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0};
    while (!node->leaf)
      node = child(node, node->count);
    return {node, static_cast<std::uint16_t>(node->count - 1)};
  }

  Location find(const void *search_key) const noexcept {
    Node *node = root_;
    while (node != nullptr) {
      const std::uint16_t index = node_lower_bound(node, search_key);
      if (index < node->count && compare(search_key, key(node, index)) == 0)
        return {node, index};
      if (node->leaf)
        break;
      node = child(node, index);
    }
    return {nullptr, 0};
  }

  Location lower_bound(const void *search_key) const noexcept {
    return bound(search_key, false);
  }

  Location upper_bound(const void *search_key) const noexcept {
    return bound(search_key, true);
  }

  bool advance(Location &location) const noexcept {
    Node *node = location.node;
    std::uint16_t index = location.index;
    if (node == nullptr)
      return false;
    if (node->leaf && index + 1 < node->count) {
      location.index = static_cast<std::uint16_t>(index + 1);
      return true;
    }
    if (!node->leaf) {
      node = child(node, index + 1);
      while (!node->leaf)
        node = child(node, 0);
      location = {node, 0};
      return true;
    }
    while (node->parent != nullptr) {
      const std::uint16_t position = node->position;
      node = node->parent;
      if (position < node->count) {
        location = {node, position};
        return true;
      }
    }
    location = {nullptr, 0};
    return false;
  }

  bool retreat(Location &location) const noexcept {
    Node *node = location.node;
    std::uint16_t index = location.index;
    if (node == nullptr) {
      location = last();
      return location.node != nullptr;
    }
    if (node->leaf && index != 0) {
      location.index = static_cast<std::uint16_t>(index - 1);
      return true;
    }
    if (!node->leaf) {
      node = child(node, index);
      while (!node->leaf)
        node = child(node, node->count);
      location = {node, static_cast<std::uint16_t>(node->count - 1)};
      return true;
    }
    while (node->parent != nullptr) {
      const std::uint16_t position = node->position;
      node = node->parent;
      if (position != 0) {
        location = {node, static_cast<std::uint16_t>(position - 1)};
        return true;
      }
    }
    return false;
  }

  bool valid(Location location) const noexcept {
    return location.node != nullptr && location.index < location.node->count;
  }

  void *key(Location location) noexcept {
    return key(location.node, location.index);
  }
  const void *key(Location location) const noexcept {
    return key(location.node, location.index);
  }
  void *value(Location location) noexcept {
    return entry(location.node, location.index) + value_offset_;
  }
  const void *value(Location location) const noexcept {
    return entry(location.node, location.index) + value_offset_;
  }

  InsertResult insert(const void *input_key, const void *input_value,
                      bool assign_existing) noexcept {
    InsertLocation location = locate_for_insert(input_key);
    if (location.found) {
      if (assign_existing && options_.value_size != 0) {
        std::memmove(value({location.node, location.index}), input_value,
                     options_.value_size);
        return InsertResult::kAssigned;
      }
      return InsertResult::kAlreadyPresent;
    }

    EntryScratch scratch(*this);
    if (scratch.data == nullptr)
      return InsertResult::kOutOfMemory;
    std::memset(scratch.data, 0, entry_stride_);
    std::memmove(scratch.data, input_key, options_.key_size);
    if (options_.value_size != 0)
      std::memmove(scratch.data + value_offset_, input_value,
                   options_.value_size);

    if (location.node == nullptr) {
      Node *root = allocate_node(true, nullptr);
      if (root == nullptr)
        return InsertResult::kOutOfMemory;
      copy_entry(root, 0, scratch.data);
      root->count = 1;
      root_ = root;
      size_ = 1;
      return InsertResult::kInserted;
    }
    if (!make_room(location.node, location.index))
      return InsertResult::kOutOfMemory;
    insert_entry(location.node, location.index, scratch.data);
    ++size_;
    return InsertResult::kInserted;
  }

  bool erase(const void *search_key) noexcept {
    const Location location = find(search_key);
    if (location.node == nullptr)
      return false;
    erase_location(location.node, location.index);
    return true;
  }

  std::size_t erase_range(const void *lower, const void *upper) noexcept {
    if (lower != nullptr && upper != nullptr && compare(lower, upper) >= 0)
      return 0;
    Location first = lower == nullptr ? begin() : lower_bound(lower);
    const Location last =
        upper == nullptr ? Location{nullptr, 0} : lower_bound(upper);
    if (first.node == last.node && first.index == last.index)
      return 0;

    std::size_t count = 0;
    for (Location cursor = first;
         cursor.node != last.node || cursor.index != last.index;) {
      if (!cursor.node->leaf) {
        ++count;
        advance(cursor);
        continue;
      }
      if (cursor.node == last.node) {
        count += last.index - cursor.index;
        break;
      }
      count += cursor.node->count - cursor.index;
      cursor.index = static_cast<std::uint16_t>(cursor.node->count - 1);
      advance(cursor);
    }
    if (count == size_) {
      clear();
      return count;
    }

    const std::size_t target_size = size_ - count;
    while (size_ != target_size) {
      if (!first.node->leaf) {
        first = erase_and_get_next(first.node, first.index);
        continue;
      }
      const std::uint16_t available =
          static_cast<std::uint16_t>(first.node->count - first.index);
      const std::uint16_t to_erase = static_cast<std::uint16_t>(
          std::min<std::size_t>(size_ - target_size, available));
      Node *node = first.node;
      remove_entries(node, first.index, to_erase);
      size_ -= to_erase;
      if (node->count == 0 && node != root_ && size_ != target_size &&
          node->position < node->parent->count) {
        Node *parent = node->parent;
        const std::uint16_t position = node->position;
        remove_entry(parent, position);
        --size_;
        for (std::size_t child_index = position; child_index <= parent->count;
             ++child_index)
          child(parent, child_index) = child(parent, child_index + 1);
        update_child_positions(parent, position);
        first = {child(parent, position), 0};
        deallocate_empty_node(node);
        rebalance_after_delete(parent, &first);
      } else {
        rebalance_after_delete(node, &first);
      }
      normalize(first);
    }
    return count;
  }

  void clear() noexcept {
    destroy_subtree(root_);
    root_ = nullptr;
    size_ = 0;
  }

  int compare(const void *left, const void *right) const noexcept {
    return options_.compare(left, right, options_.key_size, options_.context);
  }

private:
  struct InsertLocation {
    Node *node;
    std::uint16_t index;
    bool found;
  };

  struct EntryScratch {
    explicit EntryScratch(RawBTree &owner) noexcept : tree(owner) {
      if (tree.entry_stride_ <= sizeof(local) && tree.entry_alignment_ <= 64) {
        data = local;
      } else {
        data = static_cast<std::byte *>(tree.allocator_.allocate(
            tree.entry_stride_, tree.entry_alignment_));
        heap = data != nullptr;
      }
    }
    ~EntryScratch() {
      if (heap)
        tree.allocator_.deallocate(data, tree.entry_stride_,
                                   tree.entry_alignment_);
    }
    RawBTree &tree;
    alignas(64) std::byte local[1024];
    std::byte *data = nullptr;
    bool heap = false;
  };

  static constexpr std::size_t kTargetLeafBytes = 256;

  static std::size_t align_up(std::size_t value,
                              std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  std::byte *entry(Node *node, std::size_t index) const noexcept {
    return reinterpret_cast<std::byte *>(node) + entries_offset_ +
           index * entry_stride_;
  }

  const std::byte *entry(const Node *node, std::size_t index) const noexcept {
    return reinterpret_cast<const std::byte *>(node) + entries_offset_ +
           index * entry_stride_;
  }

  void *key(Node *node, std::size_t index) const noexcept {
    return entry(node, index);
  }

  const void *key(const Node *node, std::size_t index) const noexcept {
    return entry(node, index);
  }

  Node **children(Node *node) const noexcept {
    return reinterpret_cast<Node **>(reinterpret_cast<std::byte *>(node) +
                                     children_offset_);
  }

  Node *const *children(const Node *node) const noexcept {
    return reinterpret_cast<Node *const *>(
        reinterpret_cast<const std::byte *>(node) + children_offset_);
  }

  Node *&child(Node *node, std::size_t index) const noexcept {
    return children(node)[index];
  }

  Node *child(const Node *node, std::size_t index) const noexcept {
    return children(node)[index];
  }

  Node *allocate_node(bool leaf, Node *parent) noexcept {
    const std::size_t bytes = leaf ? leaf_bytes_ : internal_bytes_;
    void *storage = allocator_.allocate(bytes, node_alignment_);
    if (storage == nullptr)
      return nullptr;
    return ::new (storage) Node{parent, 0, 0, leaf};
  }

  void deallocate_empty_node(Node *node) noexcept {
    if (node == nullptr)
      return;
    const std::size_t bytes = node->leaf ? leaf_bytes_ : internal_bytes_;
    node->~Node();
    allocator_.deallocate(node, bytes, node_alignment_);
  }

  void copy_entry(Node *node, std::size_t index,
                  const void *source) const noexcept {
    std::memcpy(entry(node, index), source, entry_stride_);
  }

  void insert_entry(Node *node, std::uint16_t index,
                    const void *source) const noexcept {
    std::memmove(entry(node, index + 1), entry(node, index),
                 (node->count - index) * entry_stride_);
    copy_entry(node, index, source);
    ++node->count;
  }

  void remove_entry(Node *node, std::uint16_t index) const noexcept {
    std::memmove(entry(node, index), entry(node, index + 1),
                 (node->count - index - 1) * entry_stride_);
    --node->count;
  }

  void remove_entries(Node *node, std::uint16_t index,
                      std::uint16_t count) const noexcept {
    const std::uint16_t old_count = node->count;
    std::memmove(entry(node, index), entry(node, index + count),
                 static_cast<std::size_t>(old_count - index - count) *
                     entry_stride_);
    node->count = static_cast<std::uint16_t>(old_count - count);
  }

  std::uint16_t node_lower_bound(const Node *node,
                                 const void *search_key) const noexcept {
    std::uint16_t first = 0;
    std::uint16_t last_index = node->count;
    while (first != last_index) {
      const std::uint16_t middle =
          static_cast<std::uint16_t>((first + last_index) / 2);
      if (compare(key(node, middle), search_key) < 0)
        first = static_cast<std::uint16_t>(middle + 1);
      else
        last_index = middle;
    }
    return first;
  }

  std::uint16_t node_upper_bound(const Node *node,
                                 const void *search_key) const noexcept {
    std::uint16_t first = 0;
    std::uint16_t last_index = node->count;
    while (first != last_index) {
      const std::uint16_t middle =
          static_cast<std::uint16_t>((first + last_index) / 2);
      if (compare(search_key, key(node, middle)) >= 0)
        first = static_cast<std::uint16_t>(middle + 1);
      else
        last_index = middle;
    }
    return first;
  }

  InsertLocation locate_for_insert(const void *search_key) const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0, false};
    for (;;) {
      const std::uint16_t index = node_lower_bound(node, search_key);
      if (index < node->count && compare(search_key, key(node, index)) == 0)
        return {node, index, true};
      if (node->leaf)
        return {node, index, false};
      node = child(node, index);
    }
  }

  Location bound(const void *search_key, bool upper) const noexcept {
    Node *node = root_;
    Node *candidate = nullptr;
    std::uint16_t candidate_index = 0;
    while (node != nullptr) {
      const std::uint16_t index = upper ? node_upper_bound(node, search_key)
                                        : node_lower_bound(node, search_key);
      if (index < node->count) {
        candidate = node;
        candidate_index = index;
      }
      if (node->leaf)
        break;
      node = child(node, index);
    }
    return {candidate, candidate_index};
  }

  void update_child_positions(Node *node, std::size_t first) const noexcept {
    if (node->leaf)
      return;
    for (std::size_t index = first; index <= node->count; ++index) {
      child(node, index)->parent = node;
      child(node, index)->position = static_cast<std::uint16_t>(index);
    }
  }

  void split_child(Node *parent, std::uint16_t child_index,
                   Node *sibling) const noexcept {
    Node *source = child(parent, child_index);
    const std::uint16_t old_parent_count = parent->count;
    for (std::size_t index = old_parent_count + 1;
         index > static_cast<std::size_t>(child_index) + 1; --index)
      child(parent, index) = child(parent, index - 1);
    child(parent, child_index + 1) = sibling;

    std::memcpy(entry(sibling, 0), entry(source, min_degree_),
                static_cast<std::size_t>(min_keys_) * entry_stride_);
    sibling->count = min_keys_;
    insert_entry(parent, child_index, entry(source, min_keys_));
    source->count = min_keys_;

    if (!source->leaf) {
      for (std::size_t index = 0; index <= min_keys_; ++index)
        child(sibling, index) = child(source, min_degree_ + index);
      update_child_positions(sibling, 0);
    }
    update_child_positions(parent, child_index);
  }

  void borrow_from_left(Node *parent,
                        std::uint16_t child_index) const noexcept {
    Node *left = child(parent, child_index - 1);
    Node *target = child(parent, child_index);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t target_old_count = target->count;
    std::memmove(entry(target, 1), entry(target, 0),
                 static_cast<std::size_t>(target_old_count) * entry_stride_);
    copy_entry(target, 0, entry(parent, child_index - 1));
    copy_entry(parent, child_index - 1, entry(left, left_old_count - 1));
    --left->count;
    ++target->count;
    if (!target->leaf) {
      for (std::size_t index = target_old_count + 1; index != 0; --index)
        child(target, index) = child(target, index - 1);
      child(target, 0) = child(left, left_old_count);
      update_child_positions(target, 0);
    }
  }

  void borrow_from_right(Node *parent,
                         std::uint16_t child_index) const noexcept {
    Node *target = child(parent, child_index);
    Node *right = child(parent, child_index + 1);
    const std::uint16_t target_old_count = target->count;
    const std::uint16_t right_old_count = right->count;
    copy_entry(target, target_old_count, entry(parent, child_index));
    copy_entry(parent, child_index, entry(right, 0));
    std::memmove(entry(right, 0), entry(right, 1),
                 static_cast<std::size_t>(right_old_count - 1) * entry_stride_);
    ++target->count;
    --right->count;
    if (!target->leaf) {
      child(target, target_old_count + 1) = child(right, 0);
      for (std::size_t index = 0; index < right_old_count; ++index)
        child(right, index) = child(right, index + 1);
      update_child_positions(target, target_old_count + 1);
      update_child_positions(right, 0);
    }
  }

  void borrow_from_left_n(Node *parent, std::uint16_t child_index,
                          std::uint16_t moves) const noexcept {
    Node *left = child(parent, child_index - 1);
    Node *target = child(parent, child_index);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t target_old_count = target->count;
    std::memmove(entry(target, moves), entry(target, 0),
                 static_cast<std::size_t>(target_old_count) * entry_stride_);
    std::memcpy(entry(target, 0), entry(left, left_old_count - moves + 1),
                static_cast<std::size_t>(moves - 1) * entry_stride_);
    copy_entry(target, moves - 1, entry(parent, child_index - 1));
    copy_entry(parent, child_index - 1, entry(left, left_old_count - moves));
    left->count = static_cast<std::uint16_t>(left_old_count - moves);
    target->count = static_cast<std::uint16_t>(target_old_count + moves);
    if (!target->leaf) {
      std::memmove(children(target) + moves, children(target),
                   (static_cast<std::size_t>(target_old_count) + 1) *
                       sizeof(Node *));
      std::memcpy(children(target), children(left) + left->count + 1,
                  static_cast<std::size_t>(moves) * sizeof(Node *));
      update_child_positions(target, 0);
    }
  }

  void borrow_from_right_n(Node *parent, std::uint16_t child_index,
                           std::uint16_t moves) const noexcept {
    Node *target = child(parent, child_index);
    Node *right = child(parent, child_index + 1);
    const std::uint16_t target_old_count = target->count;
    const std::uint16_t right_old_count = right->count;
    copy_entry(target, target_old_count, entry(parent, child_index));
    std::memcpy(entry(target, target_old_count + 1), entry(right, 0),
                static_cast<std::size_t>(moves - 1) * entry_stride_);
    copy_entry(parent, child_index, entry(right, moves - 1));
    std::memmove(entry(right, 0), entry(right, moves),
                 static_cast<std::size_t>(right_old_count - moves) *
                     entry_stride_);
    target->count = static_cast<std::uint16_t>(target_old_count + moves);
    right->count = static_cast<std::uint16_t>(right_old_count - moves);
    if (!target->leaf) {
      std::memcpy(children(target) + target_old_count + 1, children(right),
                  static_cast<std::size_t>(moves) * sizeof(Node *));
      std::memmove(children(right), children(right) + moves,
                   (static_cast<std::size_t>(right_old_count - moves) + 1) *
                       sizeof(Node *));
      update_child_positions(target, target_old_count + 1);
      update_child_positions(right, 0);
    }
  }

  Node *merge_children(Node *parent, std::uint16_t index) noexcept {
    Node *left = child(parent, index);
    Node *right = child(parent, index + 1);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t right_count = right->count;
    const std::uint16_t parent_old_count = parent->count;
    copy_entry(left, left_old_count, entry(parent, index));
    std::memcpy(entry(left, left_old_count + 1), entry(right, 0),
                static_cast<std::size_t>(right_count) * entry_stride_);
    left->count = static_cast<std::uint16_t>(left_old_count + 1 + right_count);
    if (!left->leaf) {
      for (std::size_t child_index = 0; child_index <= right_count;
           ++child_index)
        child(left, left_old_count + 1 + child_index) =
            child(right, child_index);
      update_child_positions(left, left_old_count + 1);
    }
    std::memmove(entry(parent, index), entry(parent, index + 1),
                 static_cast<std::size_t>(parent_old_count - index - 1) *
                     entry_stride_);
    --parent->count;
    for (std::size_t child_index = index + 1; child_index < parent_old_count;
         ++child_index)
      child(parent, child_index) = child(parent, child_index + 1);
    update_child_positions(parent, index);
    right->count = 0;
    deallocate_empty_node(right);
    return left;
  }

  bool make_room(Node *&node, std::uint16_t &index) noexcept {
    if (node->count < capacity_)
      return true;
    if (node != root_) {
      Node *parent = node->parent;
      const std::uint16_t position = node->position;
      Node *left = position == 0 ? nullptr : child(parent, position - 1);
      Node *right =
          position == parent->count ? nullptr : child(parent, position + 1);
      const bool can_use_left = left != nullptr && left->count < capacity_ &&
                                (index != 0 || left->count + 1 < capacity_);
      const bool can_use_right =
          right != nullptr && right->count < capacity_ &&
          (index != node->count || right->count + 1 < capacity_);
      const auto rebalance_into_left = [&]() noexcept {
        const std::uint16_t old_left_count = left->count;
        const std::uint16_t moves = static_cast<std::uint16_t>(std::max(
            std::size_t{1}, (capacity_ - left->count) / std::size_t{2}));
        borrow_from_right_n(parent, static_cast<std::uint16_t>(position - 1),
                            moves);
        if (moves > index) {
          node = left;
          index = static_cast<std::uint16_t>(old_left_count + 1 + index);
        } else {
          index = static_cast<std::uint16_t>(index - moves);
        }
      };
      const auto rebalance_into_right = [&]() noexcept {
        const std::uint16_t old_count = node->count;
        const std::uint16_t moves = static_cast<std::uint16_t>(std::max(
            std::size_t{1}, (capacity_ - right->count) / std::size_t{2}));
        borrow_from_left_n(parent, static_cast<std::uint16_t>(position + 1),
                           moves);
        const std::uint16_t remaining =
            static_cast<std::uint16_t>(old_count - moves);
        if (index > remaining) {
          node = right;
          index = static_cast<std::uint16_t>(index - remaining - 1);
        }
      };
      if (can_use_left && (index <= min_keys_ || !can_use_right)) {
        rebalance_into_left();
        return true;
      }
      if (can_use_right) {
        rebalance_into_right();
        return true;
      }
      if (can_use_left) {
        rebalance_into_left();
        return true;
      }
    }

    if (node == root_) {
      Node *new_root = allocate_node(false, nullptr);
      Node *sibling = allocate_node(node->leaf, new_root);
      if (new_root == nullptr || sibling == nullptr) {
        deallocate_empty_node(new_root);
        deallocate_empty_node(sibling);
        return false;
      }
      child(new_root, 0) = node;
      node->parent = new_root;
      node->position = 0;
      split_child(new_root, 0, sibling);
      root_ = new_root;
      if (index > min_keys_) {
        node = sibling;
        index = static_cast<std::uint16_t>(index - min_degree_);
      }
      return true;
    }

    // Reserve this level's sibling before a recursive parent split so OOM can
    // never leave the tree structurally changed by only half an insertion.
    Node *sibling = allocate_node(node->leaf, node->parent);
    if (sibling == nullptr)
      return false;
    Node *parent = node->parent;
    std::uint16_t parent_index = node->position;
    if (!make_room(parent, parent_index)) {
      deallocate_empty_node(sibling);
      return false;
    }
    parent = node->parent;
    split_child(parent, node->position, sibling);
    if (index > min_keys_) {
      node = sibling;
      index = static_cast<std::uint16_t>(index - min_degree_);
    }
    return true;
  }

  void rebalance_after_delete(Node *node, Location *cursor = nullptr) noexcept {
    while (node != root_ && node->count < min_keys_) {
      Node *parent = node->parent;
      const std::uint16_t position = node->position;
      if (position != 0 && child(parent, position - 1)->count > min_keys_) {
        Node *left = child(parent, position - 1);
        const std::uint16_t moves = static_cast<std::uint16_t>(
            std::min(static_cast<std::size_t>(min_keys_ - node->count),
                     static_cast<std::size_t>(left->count - min_keys_)));
        borrow_from_left_n(parent, position, moves);
        if (cursor != nullptr && cursor->node == node)
          cursor->index = static_cast<std::uint16_t>(cursor->index + moves);
        if (node->count >= min_keys_)
          break;
      }
      if (position < parent->count &&
          child(parent, position + 1)->count > min_keys_) {
        Node *right = child(parent, position + 1);
        const std::uint16_t moves = static_cast<std::uint16_t>(
            std::min(static_cast<std::size_t>(min_keys_ - node->count),
                     static_cast<std::size_t>(right->count - min_keys_)));
        borrow_from_right_n(parent, position, moves);
        if (node->count >= min_keys_)
          break;
      }
      if (position < parent->count) {
        merge_children(parent, position);
      } else {
        Node *left = child(parent, position - 1);
        const std::uint16_t left_old_count = left->count;
        if (cursor != nullptr && cursor->node == node) {
          cursor->node = left;
          cursor->index =
              static_cast<std::uint16_t>(left_old_count + 1 + cursor->index);
        }
        merge_children(parent, static_cast<std::uint16_t>(position - 1));
      }
      node = parent;
    }
    collapse_empty_root(cursor);
  }

  void normalize(Location &location) const noexcept {
    if (location.node == nullptr || location.index != location.node->count)
      return;
    if (location.node->count == 0) {
      location = {nullptr, 0};
      return;
    }
    location.index = static_cast<std::uint16_t>(location.node->count - 1);
    advance(location);
  }

  Location erase_and_get_next(Node *node, std::uint16_t index) noexcept {
    const bool internal_delete = !node->leaf;
    if (internal_delete) {
      Node *predecessor = child(node, index);
      while (!predecessor->leaf)
        predecessor = child(predecessor, predecessor->count);
      const std::uint16_t predecessor_index =
          static_cast<std::uint16_t>(predecessor->count - 1);
      copy_entry(node, index, entry(predecessor, predecessor_index));
      --predecessor->count;
      node = predecessor;
      index = predecessor_index;
    } else {
      remove_entry(node, index);
    }
    --size_;
    Location next{node, index};
    rebalance_after_delete(node, &next);
    normalize(next);
    if (internal_delete && next.node != nullptr)
      advance(next);
    return next;
  }

  void erase_location(Node *node, std::uint16_t index) noexcept {
    (void)erase_and_get_next(node, index);
  }

  void collapse_empty_root(Location *cursor = nullptr) noexcept {
    if (root_ == nullptr || root_->count != 0)
      return;
    Node *old_root = root_;
    if (old_root->leaf) {
      root_ = nullptr;
    } else {
      root_ = child(old_root, 0);
      root_->parent = nullptr;
      root_->position = 0;
    }
    if (cursor != nullptr && cursor->node == old_root) {
      cursor->node = root_;
      cursor->index = 0;
    }
    deallocate_empty_node(old_root);
  }

  void destroy_subtree(Node *node) noexcept {
    if (node == nullptr)
      return;
    if (!node->leaf)
      for (std::size_t index = 0; index <= node->count; ++index)
        destroy_subtree(child(node, index));
    node->count = 0;
    deallocate_empty_node(node);
  }

  hyle_container_btree_options options_;
  hyle::mem::Allocator allocator_;
  Node *root_ = nullptr;
  std::size_t size_ = 0;
  std::size_t entry_alignment_ = 0;
  std::size_t value_offset_ = 0;
  std::size_t entry_stride_ = 0;
  std::size_t entries_offset_ = 0;
  std::size_t children_offset_ = 0;
  std::size_t leaf_bytes_ = 0;
  std::size_t internal_bytes_ = 0;
  std::size_t node_alignment_ = 0;
  std::uint16_t capacity_ = 0;
  std::uint16_t min_keys_ = 0;
  std::uint16_t min_degree_ = 0;
};

template <class T, class... Args>
T *allocate_handle(hyle_allocator backing, Args &&...args) noexcept {
  void *storage = backing.alloc(backing.state, sizeof(T), alignof(T));
  if (storage == nullptr)
    return nullptr;
  return ::new (storage) T(std::forward<Args>(args)...);
}

template <class T> void free_handle(T *handle) noexcept {
  if (handle == nullptr)
    return;
  const hyle_allocator backing = handle->backing;
  handle->~T();
  backing.free(backing.state, handle, sizeof(T), alignof(T));
}

RawBTree::Location
cursor_location(const hyle_container_btree_cursor &cursor) noexcept {
  return {const_cast<RawBTree::Node *>(
              static_cast<const RawBTree::Node *>(cursor._node)),
          static_cast<std::uint16_t>(cursor._index)};
}

void set_cursor(hyle_container_btree_cursor &cursor, const void *owner,
                RawBTree::Location location) noexcept {
  cursor._owner = owner;
  cursor._node = location.node;
  cursor._index = location.index;
}

hyle_container_insert_result to_c(RawBTree::InsertResult result) noexcept {
  switch (result) {
  case RawBTree::InsertResult::kInserted:
    return HYLE_CONTAINER_INSERTED;
  case RawBTree::InsertResult::kAssigned:
    return HYLE_CONTAINER_ASSIGNED;
  default:
    return HYLE_CONTAINER_ALREADY_PRESENT;
  }
}

template <class T>
int compare_scalar(const void *left, const void *right) noexcept {
  T left_value;
  T right_value;
  std::memcpy(&left_value, left, sizeof(T));
  std::memcpy(&right_value, right, sizeof(T));
  return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
}

} // namespace

struct hyle_container_btree_map_o {
  hyle_allocator backing;
  explicit hyle_container_btree_map_o(
      const hyle_container_btree_options &options)
      : backing(options.backing), value(options, to_cpp(&backing)) {}
  RawBTree value;
};

struct hyle_container_btree_set_o {
  hyle_allocator backing;
  explicit hyle_container_btree_set_o(
      const hyle_container_btree_options &options)
      : backing(options.backing), value(options, to_cpp(&backing)) {}
  RawBTree value;
};

extern "C" {

int hyle_container_compare_i32(const void *left, const void *right, size_t,
                               void *) {
  return compare_scalar<std::int32_t>(left, right);
}
int hyle_container_compare_u32(const void *left, const void *right, size_t,
                               void *) {
  return compare_scalar<std::uint32_t>(left, right);
}
int hyle_container_compare_i64(const void *left, const void *right, size_t,
                               void *) {
  return compare_scalar<std::int64_t>(left, right);
}
int hyle_container_compare_u64(const void *left, const void *right, size_t,
                               void *) {
  return compare_scalar<std::uint64_t>(left, right);
}
int hyle_container_compare_bytes(const void *left, const void *right,
                                 size_t key_size, void *) {
  const int result = std::memcmp(left, right, key_size);
  return result < 0 ? -1 : result > 0 ? 1 : 0;
}

hyle_status
hyle_container_btree_map_create(hyle_container_btree_map **out,
                                const hyle_container_btree_options *options) {
  if (out == nullptr)
    return hyle_status_invalid_argument("B-tree map output is NULL");
  *out = nullptr;
  hyle_container_btree_options decoded = HYLE_CONTAINER_BTREE_OPTIONS_INIT;
  if (!decode_btree_options(options, decoded) ||
      !valid_options(decoded, true))
    return hyle_status_invalid_argument("invalid B-tree map options");
  auto *map = allocate_handle<hyle_container_btree_map>(decoded.backing,
                                                         decoded);
  if (map == nullptr)
    return hyle_status_resource_exhausted("B-tree map allocation failed");
  *out = map;
  return hyle_status_ok();
}

void hyle_container_btree_map_destroy(hyle_container_btree_map *map) {
  free_handle(map);
}

size_t hyle_container_btree_map_size(const hyle_container_btree_map *map) {
  return map == nullptr ? 0 : map->value.size();
}

void hyle_container_btree_map_clear(hyle_container_btree_map *map) {
  if (map != nullptr)
    map->value.clear();
}

static hyle_status
btree_map_insert_impl(hyle_container_insert_result *out_result,
                      hyle_container_btree_map *map, const void *key,
                      const void *value, bool assign) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || map == nullptr || key == nullptr ||
      value == nullptr)
    return hyle_status_invalid_argument(
        "result, B-tree map, key, and value are required");
  const RawBTree::InsertResult result = map->value.insert(key, value, assign);
  if (result == RawBTree::InsertResult::kOutOfMemory)
    return hyle_status_resource_exhausted("B-tree node allocation failed");
  *out_result = to_c(result);
  return hyle_status_ok();
}

hyle_status
hyle_container_btree_map_insert(hyle_container_insert_result *out_result,
                                hyle_container_btree_map *map, const void *key,
                                const void *value) {
  return btree_map_insert_impl(out_result, map, key, value, false);
}

hyle_status hyle_container_btree_map_insert_or_assign(
    hyle_container_insert_result *out_result, hyle_container_btree_map *map,
    const void *key, const void *value) {
  return btree_map_insert_impl(out_result, map, key, value, true);
}

void *hyle_container_btree_map_find(hyle_container_btree_map *map,
                                    const void *key) {
  if (map == nullptr || key == nullptr)
    return nullptr;
  const RawBTree::Location location = map->value.find(key);
  return location.node == nullptr ? nullptr : map->value.value(location);
}

const void *
hyle_container_btree_map_find_const(const hyle_container_btree_map *map,
                                    const void *key) {
  if (map == nullptr || key == nullptr)
    return nullptr;
  const RawBTree::Location location = map->value.find(key);
  return location.node == nullptr ? nullptr : map->value.value(location);
}

int hyle_container_btree_map_contains(const hyle_container_btree_map *map,
                                      const void *key) {
  return map != nullptr && key != nullptr &&
         map->value.find(key).node != nullptr;
}

int hyle_container_btree_map_erase(hyle_container_btree_map *map,
                                   const void *key) {
  return map != nullptr && key != nullptr && map->value.erase(key);
}

size_t hyle_container_btree_map_erase_range(hyle_container_btree_map *map,
                                            const void *lower,
                                            const void *upper) {
  return map == nullptr ? 0 : map->value.erase_range(lower, upper);
}

static int map_set_cursor(const hyle_container_btree_map *map,
                          hyle_container_btree_cursor *out,
                          RawBTree::Location location) {
  if (map == nullptr || out == nullptr)
    return 0;
  set_cursor(*out, map, location);
  return location.node != nullptr;
}

int hyle_container_btree_map_begin(const hyle_container_btree_map *map,
                                   hyle_container_btree_cursor *out) {
  return map_set_cursor(map, out,
                        map == nullptr ? RawBTree::Location{nullptr, 0}
                                       : map->value.begin());
}

int hyle_container_btree_map_lower_bound(const hyle_container_btree_map *map,
                                         const void *key,
                                         hyle_container_btree_cursor *out) {
  if (key == nullptr)
    return 0;
  return map_set_cursor(map, out,
                        map == nullptr ? RawBTree::Location{nullptr, 0}
                                       : map->value.lower_bound(key));
}

int hyle_container_btree_map_upper_bound(const hyle_container_btree_map *map,
                                         const void *key,
                                         hyle_container_btree_cursor *out) {
  if (key == nullptr)
    return 0;
  return map_set_cursor(map, out,
                        map == nullptr ? RawBTree::Location{nullptr, 0}
                                       : map->value.upper_bound(key));
}

int hyle_container_btree_map_cursor_valid(
    const hyle_container_btree_map *map,
    const hyle_container_btree_cursor *cursor) {
  return map != nullptr && cursor != nullptr && cursor->_owner == map &&
         map->value.valid(cursor_location(*cursor));
}

const void *
hyle_container_btree_map_cursor_key(const hyle_container_btree_map *map,
                                    const hyle_container_btree_cursor *cursor) {
  return hyle_container_btree_map_cursor_valid(map, cursor)
             ? map->value.key(cursor_location(*cursor))
             : nullptr;
}

void *hyle_container_btree_map_cursor_value(
    hyle_container_btree_map *map, const hyle_container_btree_cursor *cursor) {
  return hyle_container_btree_map_cursor_valid(map, cursor)
             ? map->value.value(cursor_location(*cursor))
             : nullptr;
}

int hyle_container_btree_map_cursor_next(const hyle_container_btree_map *map,
                                         hyle_container_btree_cursor *cursor) {
  if (!hyle_container_btree_map_cursor_valid(map, cursor))
    return 0;
  RawBTree::Location location = cursor_location(*cursor);
  const bool valid = map->value.advance(location);
  set_cursor(*cursor, map, location);
  return valid;
}

int hyle_container_btree_map_cursor_previous(
    const hyle_container_btree_map *map, hyle_container_btree_cursor *cursor) {
  if (map == nullptr || cursor == nullptr || cursor->_owner != map)
    return 0;
  RawBTree::Location location = cursor_location(*cursor);
  if (location.node != nullptr && !map->value.valid(location))
    return 0;
  if (!map->value.retreat(location))
    return 0;
  set_cursor(*cursor, map, location);
  return true;
}

int hyle_container_btree_map_visit_range(
    hyle_container_btree_map *map, const void *lower, const void *upper,
    hyle_container_btree_map_visit_fn visitor, void *context) {
  if (map == nullptr || visitor == nullptr)
    return 0;
  RawBTree::Location location =
      lower == nullptr ? map->value.begin() : map->value.lower_bound(lower);
  while (location.node != nullptr) {
    if (upper != nullptr &&
        map->value.compare(map->value.key(location), upper) >= 0)
      break;
    if (!visitor(map->value.key(location), map->value.value(location), context))
      return 0;
    map->value.advance(location);
  }
  return 1;
}

hyle_status
hyle_container_btree_set_create(hyle_container_btree_set **out,
                                const hyle_container_btree_options *options) {
  if (out == nullptr)
    return hyle_status_invalid_argument("B-tree set output is NULL");
  *out = nullptr;
  hyle_container_btree_options decoded = HYLE_CONTAINER_BTREE_OPTIONS_INIT;
  if (!decode_btree_options(options, decoded) ||
      !valid_options(decoded, false))
    return hyle_status_invalid_argument("invalid B-tree set options");
  auto *set = allocate_handle<hyle_container_btree_set>(decoded.backing,
                                                         decoded);
  if (set == nullptr)
    return hyle_status_resource_exhausted("B-tree set allocation failed");
  *out = set;
  return hyle_status_ok();
}

void hyle_container_btree_set_destroy(hyle_container_btree_set *set) {
  free_handle(set);
}

size_t hyle_container_btree_set_size(const hyle_container_btree_set *set) {
  return set == nullptr ? 0 : set->value.size();
}

void hyle_container_btree_set_clear(hyle_container_btree_set *set) {
  if (set != nullptr)
    set->value.clear();
}

hyle_status
hyle_container_btree_set_insert(hyle_container_insert_result *out_result,
                                hyle_container_btree_set *set,
                                const void *key) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || set == nullptr || key == nullptr)
    return hyle_status_invalid_argument(
        "result, B-tree set, and key are required");
  const RawBTree::InsertResult result = set->value.insert(key, nullptr, false);
  if (result == RawBTree::InsertResult::kOutOfMemory)
    return hyle_status_resource_exhausted("B-tree node allocation failed");
  *out_result = to_c(result);
  return hyle_status_ok();
}

const void *hyle_container_btree_set_find(const hyle_container_btree_set *set,
                                          const void *key) {
  if (set == nullptr || key == nullptr)
    return nullptr;
  const RawBTree::Location location = set->value.find(key);
  return location.node == nullptr ? nullptr : set->value.key(location);
}

int hyle_container_btree_set_contains(const hyle_container_btree_set *set,
                                      const void *key) {
  return set != nullptr && key != nullptr &&
         set->value.find(key).node != nullptr;
}

int hyle_container_btree_set_erase(hyle_container_btree_set *set,
                                   const void *key) {
  return set != nullptr && key != nullptr && set->value.erase(key);
}

size_t hyle_container_btree_set_erase_range(hyle_container_btree_set *set,
                                            const void *lower,
                                            const void *upper) {
  return set == nullptr ? 0 : set->value.erase_range(lower, upper);
}

static int set_set_cursor(const hyle_container_btree_set *set,
                          hyle_container_btree_cursor *out,
                          RawBTree::Location location) {
  if (set == nullptr || out == nullptr)
    return 0;
  set_cursor(*out, set, location);
  return location.node != nullptr;
}

int hyle_container_btree_set_begin(const hyle_container_btree_set *set,
                                   hyle_container_btree_cursor *out) {
  return set_set_cursor(set, out,
                        set == nullptr ? RawBTree::Location{nullptr, 0}
                                       : set->value.begin());
}

int hyle_container_btree_set_lower_bound(const hyle_container_btree_set *set,
                                         const void *key,
                                         hyle_container_btree_cursor *out) {
  if (key == nullptr)
    return 0;
  return set_set_cursor(set, out,
                        set == nullptr ? RawBTree::Location{nullptr, 0}
                                       : set->value.lower_bound(key));
}

int hyle_container_btree_set_upper_bound(const hyle_container_btree_set *set,
                                         const void *key,
                                         hyle_container_btree_cursor *out) {
  if (key == nullptr)
    return 0;
  return set_set_cursor(set, out,
                        set == nullptr ? RawBTree::Location{nullptr, 0}
                                       : set->value.upper_bound(key));
}

int hyle_container_btree_set_cursor_valid(
    const hyle_container_btree_set *set,
    const hyle_container_btree_cursor *cursor) {
  return set != nullptr && cursor != nullptr && cursor->_owner == set &&
         set->value.valid(cursor_location(*cursor));
}

const void *
hyle_container_btree_set_cursor_key(const hyle_container_btree_set *set,
                                    const hyle_container_btree_cursor *cursor) {
  return hyle_container_btree_set_cursor_valid(set, cursor)
             ? set->value.key(cursor_location(*cursor))
             : nullptr;
}

int hyle_container_btree_set_cursor_next(const hyle_container_btree_set *set,
                                         hyle_container_btree_cursor *cursor) {
  if (!hyle_container_btree_set_cursor_valid(set, cursor))
    return 0;
  RawBTree::Location location = cursor_location(*cursor);
  const bool valid = set->value.advance(location);
  set_cursor(*cursor, set, location);
  return valid;
}

int hyle_container_btree_set_cursor_previous(
    const hyle_container_btree_set *set, hyle_container_btree_cursor *cursor) {
  if (set == nullptr || cursor == nullptr || cursor->_owner != set)
    return 0;
  RawBTree::Location location = cursor_location(*cursor);
  if (location.node != nullptr && !set->value.valid(location))
    return 0;
  if (!set->value.retreat(location))
    return 0;
  set_cursor(*cursor, set, location);
  return 1;
}

int hyle_container_btree_set_visit_range(
    const hyle_container_btree_set *set, const void *lower, const void *upper,
    hyle_container_btree_set_visit_fn visitor, void *context) {
  if (set == nullptr || visitor == nullptr)
    return 0;
  RawBTree::Location location =
      lower == nullptr ? set->value.begin() : set->value.lower_bound(lower);
  while (location.node != nullptr) {
    if (upper != nullptr &&
        set->value.compare(set->value.key(location), upper) >= 0)
      break;
    if (!visitor(set->value.key(location), context))
      return 0;
    set->value.advance(location);
  }
  return 1;
}

} // extern "C"
