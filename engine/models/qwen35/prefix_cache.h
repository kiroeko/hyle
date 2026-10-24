// Process-local consumer policy. Snapshot owns the original state permissions;
// the typed shared directory never manufactures a state or starts native work.
#pragma once

#include "session_state.h"
#include "../../inference/prefix_directory.h"
#include <algorithm>
#include <array>
#include <memory>
#include <span>

namespace qwen35::cache {
using hyle::inference::TokenPlan;
using hyle::inference::TokenPlanRef;
using hyle::inference::token_plan;
using hyle::inference::same_prefix;

// Committed-token rebuild recipe, not a cache key or strict MTP episode log.
// It preserves consumed chunks, but has no verify candidates, acceptance
// decisions, original batch geometry or RNG transcript. A hit may import a
// differently partitioned legal state; rebuilding must use these source
// chunks, not a later logical prompt containing the same tokens.
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

// Borrowed only while assembling one completed turn/checkpoint. In particular
// emitted tokens that have not been consumed by the model are not input here.
struct ConsumedChunk final {
  std::span<const std::int32_t> ids;
  std::int32_t next_id = -1;
};
struct Checkpoint final {
  TokenPlanRef input;
  ReplayPrefix replay;
};
inline Checkpoint consumed_checkpoint(std::span<const ConsumedChunk> chunks,
    const Snapshot &head, std::size_t capacity) {
  if (chunks.empty() || !head.length || head.length > capacity || head.next_id < 0)
    throw std::runtime_error("checkpoint requires a complete nonempty consumed boundary within context");
  std::size_t total = 0;
  for (const auto &chunk : chunks) {
    if (chunk.ids.empty() || chunk.next_id < 0 || chunk.ids.size() > capacity - total ||
        std::ranges::any_of(chunk.ids, [](auto id) { return id < 0; }))
      throw std::runtime_error("checkpoint contains an invalid or unbounded consumed chunk");
    total += chunk.ids.size();
  }
  if (total != head.length || chunks.back().next_id != head.next_id)
    throw std::runtime_error("checkpoint consumed length or pending token differs from its committed snapshot");
  auto input = std::make_shared<TokenPlan>();
  input->ids.reserve(total); input->hashes.reserve(add(total, 1));
  input->hashes.push_back(1469598103934665603ULL);
  for (const auto &chunk : chunks) for (const auto id : chunk.ids) {
    input->ids.push_back(id);
    input->hashes.push_back((input->hashes.back() ^ static_cast<std::uint32_t>(id)) * 1099511628211ULL);
  }
  // Only this final Snapshot is available. Historical chunk ends are rebuild
  // boundaries, not extra cached states manufactured by a radix split.
  input->checkpoints.push_back(total);
  auto history = std::make_shared<ReplayHistory>(capacity);
  std::size_t first = 0;
  for (const auto &chunk : chunks) {
    history->append(input, first, chunk.ids.size(), chunk.next_id);
    first += chunk.ids.size();
  }
  return {std::move(input), {std::move(history), chunks.size(), total}};
}

// Model completeness/identity and rebuild policy stay here; radix/LRU/pins
// belong to the one shared typed directory. It never manufactures a Snapshot.
struct PrefixPayload final {
  SnapshotRef snapshot;
  ReplayPrefix replay;
};
class PrefixCache final {
  using Directory = hyle::inference::PrefixDirectory<PrefixPayload>;
  Directory directory_;
public:
  using Hit = Directory::Hit;
  explicit PrefixCache(std::size_t limit) : directory_(limit) {}
  PrefixCache(const PrefixCache &) = delete;
  PrefixCache &operator=(const PrefixCache &) = delete;
  PrefixCache(PrefixCache &&) = delete;
  PrefixCache &operator=(PrefixCache &&) = delete;
  bool retains(const Snapshot *snapshot) const noexcept {
    if (!snapshot) return false;
    return directory_.retains([snapshot](const PrefixPayload &payload) {
      return payload.snapshot && payload.snapshot->state_identity() == snapshot->state_identity();
    });
  }
  std::size_t size() const noexcept { return directory_.size(); }
  std::size_t evictions() const noexcept { return directory_.evictions; }
  bool evict_one() noexcept { return directory_.evict_one(); }
  Hit lookup(const TokenPlan &input, std::size_t target, std::size_t minimum,
             const void *model, const void *math) {
    auto hit = directory_.lookup({model, math}, input, target, minimum);
    if (hit) {
      const auto &head = hit.payload.snapshot;
      // Published payloads are immutable. Recheck the original model contract
      // once and fail closed if a caller violates that prerequisite.
      if (head && head->model == model && head->math == math && head->next_id >= 0 &&
          head->length == hit.complete_length()) return hit;
    }
    return {};
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
    return directory_.publish({snapshot->model, snapshot->math}, std::move(input), snapshot->length,
        PrefixPayload{snapshot, replay});
  }
};
} // namespace qwen35::cache
