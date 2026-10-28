#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hyle::inference {

enum class ColdDemandKind { kCounted, kPossible };

// Local cold-planning metadata, not measured native calls or an execution
// cache. The opaque key and row extent retain their caller's model semantics.
// A use is the original schedule hint/conservative bound. Counted adds one;
// Possible raises the existing bound to one. Their original order matters.
template<class Key>
class ColdDemandInventory final {
 public:
  struct Entry final {
    std::size_t uses = 0;
    std::vector<std::size_t> members;

    void note(std::size_t member, ColdDemandKind kind) {
      const auto next = ColdDemandInventory::next_uses(uses, kind);
      // Allocate support before publishing its use count. A failed note
      // cannot leave the two facts out of sync.
      if (std::find(members.begin(), members.end(), member) == members.end())
        members.push_back(member);
      uses = next;
    }
    template<class Matches>
    std::size_t eligible_count(Matches &&matches) const {
      return static_cast<std::size_t>(std::count_if(members.begin(), members.end(), matches));
    }
    template<class Matches>
    bool supports(Matches &&matches) const {
      return std::any_of(members.begin(), members.end(), matches);
    }
  };
  using Rows = std::map<std::size_t, Entry>;
  using Entries = std::map<Key, Rows>;

  const Entries &entries() const noexcept { return entries_; }
  const Rows *find(const Key &key) const {
    const auto found = entries_.find(key);
    return found == entries_.end() ? nullptr : &found->second;
  }
  std::size_t uses(const Key &key, std::size_t rows) const {
    const auto *group = find(key);
    if (!group) return 0;
    const auto found = group->find(rows);
    return found == group->end() ? 0 : found->second.uses;
  }

  // An inventory-only reconstruction shape is valid with zero hinted uses
  // and no request member. It must not become a possible or counted demand.
  void inventory(const Key &key, std::size_t rows) {
    const auto *group = find(key);
    if (group && group->contains(rows)) return;
    insert(key, rows, {});
  }
  void note(const Key &key, std::size_t rows, std::size_t member,
            ColdDemandKind kind = ColdDemandKind::kCounted) {
    const auto group = entries_.find(key);
    if (group != entries_.end()) {
      const auto found = group->second.find(rows);
      if (found != group->second.end()) {
        found->second.note(member, kind);
        return;
      }
    }
    Entry entry;
    entry.note(member, kind);
    insert(key, rows, std::move(entry));
  }

  // Visit ascending row extents with the cumulative unique matching member
  // count. Main still decides width eligibility and physical/role limits.
  // Both functions run only while preparing the original cold inventory;
  // neither callable nor the temporary union is stored on an execution path.
  template<class Matches, class Visit>
  static std::size_t visit_support(const Rows &rows, Matches &&matches, Visit &&visit) {
    std::set<std::size_t> eligible;
    for (const auto &[extent, entry] : rows) {
      for (const auto member : entry.members) if (matches(member)) eligible.insert(member);
      visit(extent, entry, eligible.size());
    }
    return eligible.size();
  }
  template<class Matches>
  static std::size_t eligible_members(const Rows &rows, Matches &&matches) {
    return visit_support(rows, matches, [](std::size_t, const Entry &, std::size_t) {});
  }

 private:
  static std::size_t next_uses(std::size_t count, ColdDemandKind kind) {
    if (kind == ColdDemandKind::kPossible) return (std::max)(count, std::size_t{1});
    if (count == (std::numeric_limits<std::size_t>::max)())
      throw std::overflow_error("cold demand use count overflows");
    return count + 1;
  }
  void insert(const Key &key, std::size_t rows, Entry entry) {
    const auto group = entries_.find(key);
    if (group != entries_.end()) group->second.emplace(rows, std::move(entry));
    else {
      Rows staged;
      staged.emplace(rows, std::move(entry));
      entries_.emplace(key, std::move(staged));
    }
  }
  Entries entries_;
};
} // namespace hyle::inference
