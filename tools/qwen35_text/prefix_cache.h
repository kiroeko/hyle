// Process-local consumer policy. Snapshot owns the original state permissions;
// this bounded directory never manufactures a state or starts native work.
#pragma once

#include "schedule.h"
#include "session_state.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <span>

namespace qwen35::cache {
struct TokenPlan final {
  std::vector<std::int32_t> ids;
  std::vector<std::uint64_t> hashes;
  std::vector<std::size_t> checkpoints;
};
using TokenPlanRef = std::shared_ptr<const TokenPlan>;
inline TokenPlanRef token_plan(std::span<const benchmark::Phase> phases, std::size_t capacity) {
  auto result = std::make_shared<TokenPlan>();
  std::size_t total = 0;
  for (const auto &phase : phases) total = add(total, phase.ids.size());
  if (!total || total > capacity) throw std::runtime_error("prefix token plan exceeds context capacity");
  result->ids.reserve(total); result->hashes.reserve(add(total, 1));
  result->checkpoints.reserve(phases.size()); result->hashes.push_back(1469598103934665603ULL);
  for (const auto &phase : phases) {
    for (const auto id : phase.ids) {
      result->ids.push_back(id);
      result->hashes.push_back((result->hashes.back() ^ static_cast<std::uint32_t>(id)) * 1099511628211ULL);
    }
    result->checkpoints.push_back(result->ids.size()); // Prompt and complete turn boundaries only.
  }
  return result;
}
inline bool same_prefix(const TokenPlan &a, const TokenPlan &b, std::size_t count) noexcept {
  return count <= a.ids.size() && count <= b.ids.size() && a.hashes[count] == b.hashes[count] &&
      std::equal(a.ids.begin(), a.ids.begin() + static_cast<std::ptrdiff_t>(count), b.ids.begin());
}

// Rebuild recipe, not a cache key. A hit may import a differently partitioned
// legal state; rebuilding must replay the actual source chunks, not a later
// logical prompt that happened to contain those same tokens.
struct ReplayChunk final {
  TokenPlanRef input;
  std::size_t first = 0, count = 0;
  std::int32_t next_id = -1;
};
struct ReplayHistory final {
  std::vector<ReplayChunk> chunks;
  std::size_t length = 0, capacity = 0;
  explicit ReplayHistory(std::size_t limit) : capacity(limit) { chunks.reserve(limit); }
  void append(TokenPlanRef input, std::size_t first, std::size_t count, std::int32_t next) {
    if (!input || !count || length > capacity || first != length || first > input->ids.size() || count > input->ids.size() - first ||
        count > capacity - length || chunks.size() == chunks.capacity() || next < 0)
      throw std::runtime_error("invalid actual prefix rebuild chunk");
    chunks.push_back({std::move(input), first, count, next}); length += count;
  }
};
struct ReplayPrefix final {
  std::shared_ptr<const ReplayHistory> history;
  std::size_t chunks = 0, length = 0;
};
inline std::shared_ptr<ReplayHistory> copy_history(const ReplayPrefix &prefix, std::size_t capacity) {
  if (!prefix.history || prefix.chunks > prefix.history->chunks.size())
    throw std::runtime_error("prefix is missing its actual rebuild history");
  auto result = std::make_shared<ReplayHistory>(capacity);
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    result->append(part.input, part.first, part.count, part.next_id);
  }
  if (result->length != prefix.length) throw std::runtime_error("prefix rebuild length differs from its state");
  return result;
}

// Externally serialized at consumer phase boundaries. The cache outlives its
// Hits; a Hit only pins directory eviction, never grants state write permission.
class PrefixCache final {
  static constexpr auto none = (std::numeric_limits<std::size_t>::max)();
  // Each domain has an empty root. Every other node is a payload boundary or
  // a branching point; payload-free unary nodes are merged after eviction.
  // Thus P payloads occupy at most 2P nodes, including all domain roots.
  struct Node final {
    TokenPlanRef input;
    std::size_t first = 0, count = 0, depth = 0;
    std::size_t parent = none, child = none, previous = none, next = none, entry = none;
  };
  struct Domain final { const void *model; const void *math; std::size_t root; };
  struct Entry final {
    SnapshotRef snapshot;
    ReplayPrefix replay;
    std::size_t node = none, pins = 0, previous = none, next = none;
  };
  std::size_t limit_ = 0, size_ = 0, oldest_ = none, newest_ = none;
  std::vector<Entry> entries_;
  std::vector<Node> nodes_;
  std::vector<std::size_t> free_entries_, free_nodes_;
  std::vector<Domain> domains_;
  static bool domain_less(const Domain &a, const Domain &b) noexcept {
    const std::less<const void *> less;
    return less(a.model, b.model) || (a.model == b.model && less(a.math, b.math));
  }
  std::size_t domain_position(const void *model, const void *math) const noexcept {
    return static_cast<std::size_t>(std::lower_bound(domains_.begin(), domains_.end(),
        Domain{model, math, none}, domain_less) - domains_.begin());
  }
  bool same_domain(std::size_t position, const void *model, const void *math) const noexcept {
    return position != domains_.size() && domains_[position].model == model && domains_[position].math == math;
  }
  std::int32_t first_token(std::size_t node) const noexcept {
    return nodes_[node].input->ids[nodes_[node].first];
  }
  std::size_t child_at(std::size_t parent, std::int32_t token) const noexcept {
    auto child = nodes_[parent].child;
    while (child != none && first_token(child) < token) child = nodes_[child].next;
    return child;
  }
  std::size_t allocate_node() noexcept {
    assert(!free_nodes_.empty());
    const auto node = free_nodes_.back(); free_nodes_.pop_back(); return node;
  }
  void free_node(std::size_t node) noexcept {
    nodes_[node] = {}; free_nodes_.push_back(node);
  }
  void replace_child(std::size_t parent, std::size_t old, std::size_t next) noexcept {
    auto &old_node = nodes_[old]; auto &next_node = nodes_[next];
    next_node.parent = parent; next_node.previous = old_node.previous; next_node.next = old_node.next;
    if (old_node.previous == none) nodes_[parent].child = next;
    else nodes_[old_node.previous].next = next;
    if (old_node.next != none) nodes_[old_node.next].previous = next;
  }
  void insert_child(std::size_t parent, std::size_t node) noexcept {
    const auto next = child_at(parent, first_token(node));
    auto &value = nodes_[node]; value.parent = parent; value.next = next;
    if (next == none) {
      auto previous = nodes_[parent].child;
      if (previous != none) while (nodes_[previous].next != none) previous = nodes_[previous].next;
      value.previous = previous;
    } else value.previous = nodes_[next].previous;
    if (value.previous == none) nodes_[parent].child = node;
    else nodes_[value.previous].next = node;
    if (next != none) nodes_[next].previous = node;
  }
  std::size_t path_node(const TokenPlan &input, std::size_t length, std::size_t root) const noexcept {
    auto current = root; std::size_t position = 0;
    while (position < length) {
      const auto child = child_at(current, input.ids[position]);
      if (child == none || first_token(child) != input.ids[position]) return none;
      const auto &edge = nodes_[child];
      if (edge.count > length - position || !std::equal(edge.input->ids.begin() + static_cast<std::ptrdiff_t>(edge.first),
          edge.input->ids.begin() + static_cast<std::ptrdiff_t>(edge.first + edge.count),
          input.ids.begin() + static_cast<std::ptrdiff_t>(position))) return none;
      position += edge.count; current = child;
    }
    return current;
  }
  std::size_t insert_path(const TokenPlanRef &input, std::size_t length, std::size_t root) noexcept {
    auto current = root; std::size_t position = 0;
    while (position < length) {
      const auto child = child_at(current, input->ids[position]);
      if (child == none || first_token(child) != input->ids[position]) {
        const auto leaf = allocate_node(); auto &value = nodes_[leaf];
        value.input = input; value.first = position; value.count = length - position; value.depth = length;
        insert_child(current, leaf); return leaf;
      }
      auto &edge = nodes_[child]; std::size_t common = 0;
      while (common < edge.count && common < length - position &&
          edge.input->ids[edge.first + common] == input->ids[position + common]) ++common;
      if (common == edge.count) { position += common; current = child; continue; }
      const auto branch = allocate_node(); auto &value = nodes_[branch];
      value.input = edge.input; value.first = edge.first; value.count = common; value.depth = position + common;
      replace_child(current, child, branch);
      edge.first += common; edge.count -= common; edge.parent = branch; edge.previous = edge.next = none;
      value.child = child;
      position += common;
      if (position == length) return branch; // A real newly published boundary, not a fabricated split state.
      const auto leaf = allocate_node(); auto &suffix = nodes_[leaf];
      suffix.input = input; suffix.first = position; suffix.count = length - position; suffix.depth = length;
      insert_child(branch, leaf); return leaf;
    }
    return current;
  }
  void unlink_lru(std::size_t slot) noexcept {
    const auto &entry = entries_[slot];
    if (entry.previous == none) oldest_ = entry.next; else entries_[entry.previous].next = entry.next;
    if (entry.next == none) newest_ = entry.previous; else entries_[entry.next].previous = entry.previous;
  }
  void touch(std::size_t slot) noexcept {
    if (newest_ == slot) return;
    auto &entry = entries_[slot];
    if (entry.previous != none || entry.next != none || oldest_ == slot) unlink_lru(slot);
    entry.previous = newest_; entry.next = none;
    if (newest_ == none) oldest_ = slot; else entries_[newest_].next = slot;
    newest_ = slot;
  }
  std::size_t evictable() const noexcept {
    auto slot = oldest_;
    while (slot != none && entries_[slot].pins) slot = entries_[slot].next;
    return slot;
  }
  void prune(std::size_t node) noexcept {
    while (nodes_[node].parent != none) {
      auto &value = nodes_[node]; const auto child = value.child;
      if (value.entry != none || (child != none && nodes_[child].next != none)) return;
      const auto parent = value.parent;
      if (child == none) {
        if (value.previous == none) nodes_[parent].child = value.next;
        else nodes_[value.previous].next = value.next;
        if (value.next != none) nodes_[value.next].previous = value.previous;
      } else {
        replace_child(parent, node, child);
        // A child's TokenPlan contains this whole path, including the merged edge.
        nodes_[child].first = value.first; nodes_[child].count += value.count;
      }
      free_node(node); node = parent;
    }
    if (nodes_[node].child != none) return;
    const auto domain = std::find_if(domains_.begin(), domains_.end(),
        [node](const auto &value) { return value.root == node; });
    assert(domain != domains_.end()); domains_.erase(domain); free_node(node);
  }
  void erase_entry(std::size_t slot) noexcept {
    auto &entry = entries_[slot]; assert(!entry.pins);
    const auto node = entry.node; unlink_lru(slot); nodes_[node].entry = none;
    entry = {}; free_entries_.push_back(slot); --size_; prune(node);
  }
public:
  class Hit final {
    friend class PrefixCache;
    PrefixCache *owner_ = nullptr;
    std::size_t slot_ = 0;
    Hit(PrefixCache &owner, std::size_t slot) : owner_(&owner), slot_(slot),
        snapshot(owner.entries_[slot].snapshot), replay(owner.entries_[slot].replay) {
      if (owner.entries_[slot].pins == none) throw std::runtime_error("prefix hit pin count overflows");
      ++owner.entries_[slot].pins; owner.touch(slot);
    }
  public:
    SnapshotRef snapshot;
    ReplayPrefix replay;
    Hit() noexcept = default;
    Hit(const Hit &) = delete;
    Hit &operator=(const Hit &) = delete;
    Hit(Hit &&other) noexcept : owner_(std::exchange(other.owner_, nullptr)), slot_(other.slot_),
        snapshot(std::move(other.snapshot)), replay(std::move(other.replay)) {}
    Hit &operator=(Hit &&other) noexcept {
      if (this != &other) { reset(); owner_ = std::exchange(other.owner_, nullptr); slot_ = other.slot_;
        snapshot = std::move(other.snapshot); replay = std::move(other.replay); }
      return *this;
    }
    ~Hit() { reset(); }
    explicit operator bool() const noexcept { return bool(snapshot); }
    void reset() noexcept {
      snapshot.reset(); replay = {};
      if (auto *owner = std::exchange(owner_, nullptr)) { assert(owner->entries_[slot_].pins); --owner->entries_[slot_].pins; }
    }
  };
  std::size_t evictions = 0;
  explicit PrefixCache(std::size_t limit) : limit_(limit) {
    // One transient payload and two nodes let publish install the new key
    // before pruning its victim, even when that victim is an internal boundary.
    if (!limit || limit >= entries_.max_size() || limit >= free_entries_.max_size() ||
        limit > (nodes_.max_size() - 2) / 2 || limit > (free_nodes_.max_size() - 2) / 2 ||
        limit >= domains_.max_size())
      throw std::runtime_error("prefix directory requires a finite positive entry limit");
    entries_.resize(limit + 1); nodes_.resize(limit * 2 + 2);
    free_entries_.reserve(entries_.size()); free_nodes_.reserve(nodes_.size()); domains_.reserve(limit + 1);
    for (auto i = entries_.size(); i; --i) free_entries_.push_back(i - 1);
    for (auto i = nodes_.size(); i; --i) free_nodes_.push_back(i - 1);
  }
  PrefixCache(const PrefixCache &) = delete;
  PrefixCache &operator=(const PrefixCache &) = delete;
  PrefixCache(PrefixCache &&) = delete;
  PrefixCache &operator=(PrefixCache &&) = delete;
  bool retains(const Snapshot *snapshot) const noexcept {
    for (const auto &entry : entries_) if (entry.snapshot.get() == snapshot) return true;
    return false;
  }
  std::size_t size() const noexcept { return size_; }
  bool evict_one() noexcept {
    const auto victim = evictable(); if (victim == none) return false;
    erase_entry(victim); ++evictions; return true;
  }
  Hit lookup(const TokenPlan &input, std::size_t target, std::size_t minimum,
             const void *model, const void *math) {
    const auto domain = domain_position(model, math);
    if (!same_domain(domain, model, math)) return {};
    const auto limit = (std::min)(target, input.ids.size());
    auto current = domains_[domain].root; std::size_t position = 0, best = none;
    while (position < limit) {
      const auto child = child_at(current, input.ids[position]);
      if (child == none || first_token(child) != input.ids[position]) break;
      const auto &edge = nodes_[child];
      if (edge.count > limit - position || !std::equal(edge.input->ids.begin() + static_cast<std::ptrdiff_t>(edge.first),
          edge.input->ids.begin() + static_cast<std::ptrdiff_t>(edge.first + edge.count),
          input.ids.begin() + static_cast<std::ptrdiff_t>(position))) break;
      position += edge.count; current = child;
      if (edge.entry != none) {
        const auto &head = entries_[edge.entry].snapshot;
        if (head->model == model && head->math == math && head->next_id >= 0 &&
            head->length == position && position >= minimum) best = edge.entry;
      }
    }
    return best == none ? Hit{} : Hit(*this, best);
  }
  bool publish(TokenPlanRef input, const SnapshotRef &snapshot, const ReplayPrefix &replay) {
    if (!input || !snapshot || !snapshot->length || snapshot->next_id < 0 ||
        snapshot->length > input->ids.size() || !replay.history || replay.length != snapshot->length ||
        replay.chunks == 0 || replay.chunks > replay.history->chunks.size() ||
        replay.history->chunks[replay.chunks - 1].next_id != snapshot->next_id)
      throw std::runtime_error("only a complete actual prefix boundary may enter the directory");
    std::size_t covered = 0;
    for (std::size_t i = 0; i < replay.chunks; ++i) {
      const auto &part = replay.history->chunks[i];
      if (!part.input || !part.count || part.first != covered || part.first > input->ids.size() ||
          part.count > input->ids.size() - part.first || part.first > part.input->ids.size() ||
          part.count > part.input->ids.size() - part.first ||
          !std::equal(part.input->ids.begin() + static_cast<std::ptrdiff_t>(part.first),
              part.input->ids.begin() + static_cast<std::ptrdiff_t>(part.first + part.count),
              input->ids.begin() + static_cast<std::ptrdiff_t>(part.first)))
        throw std::runtime_error("prefix rebuild history differs from its exact token key");
      covered += part.count;
    }
    if (covered != snapshot->length) throw std::runtime_error("prefix rebuild history does not cover its snapshot");
    const auto position = domain_position(snapshot->model, snapshot->math);
    const auto found = same_domain(position, snapshot->model, snapshot->math);
    const auto old_node = found ? path_node(*input, snapshot->length, domains_[position].root) : none;
    if (old_node != none && nodes_[old_node].entry != none) {
      const auto slot = nodes_[old_node].entry; auto &entry = entries_[slot];
      if (entry.pins) return false;
      entry.snapshot = snapshot; entry.replay = replay; touch(slot); return true;
    }
    const auto victim = size_ == limit_ ? evictable() : none;
    if (size_ == limit_ && victim == none) return false;
    // Everything fallible is above. Fixed cold pools cover the transient E+1
    // payload tree; no allocation/eviction can fail after this commit starts.
    auto root = found ? domains_[position].root : allocate_node();
    if (!found) {
      domains_.insert(domains_.begin() + static_cast<std::ptrdiff_t>(position),
          Domain{snapshot->model, snapshot->math, root});
    }
    const auto node = insert_path(input, snapshot->length, root);
    assert(!free_entries_.empty()); const auto slot = free_entries_.back(); free_entries_.pop_back();
    entries_[slot].snapshot = snapshot; entries_[slot].replay = replay; entries_[slot].node = node;
    nodes_[node].entry = slot; ++size_; touch(slot);
    if (victim != none) { erase_entry(victim); ++evictions; }
    return true;
  }
};
} // namespace qwen35::cache
