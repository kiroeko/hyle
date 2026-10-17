#include "state_internal.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <new>

namespace hyle::compute {
namespace d = detail;
namespace detail {
void StateSchemaData::destroy() noexcept {
  const auto a = allocator; this->~StateSchemaData(); a.deallocate(this, sizeof(*this), alignof(StateSchemaData));
}
StateStoreData::StateStoreData(StateSchema s, StateStoreOptions o, mem::AllocatorRef a)
    : schema(std::move(s)), options(std::move(o)), allocator(a),
      records(mem::StdAllocator<StateRecord>(a)), backing(mem::StdAllocator<StateBacking>(a)),
      routes(mem::StdAllocator<StateCopyRoute>(a)) {
  const auto count = std::size_t(options.max_versions) + options.max_drafts;
  records.reserve(count);
  for (std::size_t i = 0; i < count; ++i) records.emplace_back(a, schema.size());
  backing.resize(options.max_retained_blocks);
  routes.reserve(options.max_clone_slots);
}
void StateStoreData::destroy() noexcept {
  const auto a = allocator; this->~StateStoreData(); a.deallocate(this, sizeof(*this), alignof(StateStoreData));
}
void StateHeadData::destroy() noexcept {
  const auto a = allocator; this->~StateHeadData(); a.deallocate(this, sizeof(*this), alignof(StateHeadData));
}
Result<std::uint32_t> StateAccess::entry(const StateStoreData &s, const StateEntryId &id) noexcept {
  if (schema(id.schema_) != schema(s.schema) || id.index_ >= s.schema.size())
    return std::unexpected(Status::InvalidArgument("state entry belongs to a different schema"));
  return id.index_;
}
void StateAccess::drop_record(StateStoreData *s, std::uint32_t slot) noexcept {
  auto &r = s->records[slot];
  if (r.refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  std::lock_guard guard(s->lock);
  for (auto &member : r.members) { member.lease.reset(); member.pending = false; member.route = member.copy = UINT32_MAX; }
  r.live = false; r.finished = false; r.failure = Status::Ok();
}
} // namespace detail

namespace {
// Store monitor owns admission and record reuse; refs protect immutable
// records. Existing handles keep store/schema alive, preventing pointer ABA.
Result<std::uint32_t> vacant(d::StateStoreData &s, bool draft) noexcept {
  if (s.closed) return std::unexpected(Status::FailedPrecondition("state store is closed"));
  if (s.next_generation == UINT64_MAX)
    return std::unexpected(Status::OutOfRange("state version generation exhausted"));
  const auto begin = draft ? s.options.max_versions : 0u;
  const auto end = draft ? std::uint32_t(s.records.size()) : s.options.max_versions;
  for (auto i = begin; i < end; ++i) if (!s.records[i].live) return i;
  return std::unexpected(Status::Unavailable("state directory capacity is retained"));
}
void activate(d::StateStoreData &s, std::uint32_t slot) noexcept {
  auto &r = s.records[slot]; r.generation = s.next_generation++; r.live = true;
  r.finished = false; r.failure = Status::Ok(); r.refs.store(1, std::memory_order_release);
}
bool clone_root(const d::StateStoreData &s, const void *root) noexcept {
  for (const auto &r : s.routes) if (r.root == root) return true;
  return false;
}
StateStoreStats stats_locked(d::StateStoreData &s) noexcept {
  StateStoreStats stats; stats.closed = s.closed;
  for (std::size_t i = 0; i < s.records.size(); ++i) if (s.records[i].live) {
    if (i < s.options.max_versions) ++stats.versions; else ++stats.drafts;
  }
  for (const auto &r : s.routes) {
    stats.registered_clone_slots += r.driver.slots;
    stats.active_clones += r.driver.active(r.driver.object);
    stats.clone_capacity_bytes += r.capacity;
  }
  stats.backing_bytes = stats.clone_capacity_bytes;
  // Ledger positions are private, unlike version and clone-route ordinals.
  // Detect root order during accounting so shared pools need only one pass.
  const auto root_order = [](const d::StateBacking &a, const d::StateBacking &b) noexcept {
    if (a.anchor.valid() != b.anchor.valid()) return a.anchor.valid();
    return a.anchor.valid() && std::less<const void *>{}(a.backing.identity, b.backing.identity);
  };
  const auto charge_root = [&](const d::StateBacking &b) noexcept {
    if (!clone_root(s, b.backing.identity)) {
      ++stats.backing_roots;
      stats.backing_bytes += b.backing.bytes;
    }
  };
  const void *previous_root = nullptr;
  bool has_previous_root = false;
  bool ordered_roots = true;
  for (const auto &b : s.backing) {
    if (!b.anchor.valid()) continue;
    ++stats.retained_blocks;
    if (ordered_roots && (!has_previous_root || previous_root != b.backing.identity)) {
      ordered_roots = !has_previous_root ||
          !std::less<const void *>{}(b.backing.identity, previous_root);
      if (ordered_roots) charge_root(b);
      previous_root = b.backing.identity;
      has_previous_root = true;
    }
    if (b.metadata) stats.host_snapshot_bytes += b.anchor.size();
  }
  if (!ordered_roots) {
    // Only a changed, unordered ledger needs sorting. Payload accounting above
    // is independent of order; recompute just the distinct-root totals.
    std::sort(s.backing.begin(), s.backing.end(), root_order);
    stats.backing_roots = 0;
    stats.backing_bytes = stats.clone_capacity_bytes;
    has_previous_root = false;
    for (const auto &b : s.backing) {
      if (!b.anchor.valid()) break;
      if (!has_previous_root || previous_root != b.backing.identity) {
        charge_root(b);
        previous_root = b.backing.identity;
        has_previous_root = true;
      }
    }
  }
  stats.backing_roots += std::uint32_t(s.routes.size());
  return stats;
}
void collect_backing(d::StateStoreData &s) noexcept {
  for (auto &b : s.backing) if (b.anchor.valid() && d::LeaseAccess::unique_read_owner(b.anchor)) {
    b.anchor.reset(); b.backing = {}; b.metadata = false;
  }
}
bool same_block(const ReadLease &a, const ReadLease &b) noexcept {
  return d::LeaseAccess::identity(a) == d::LeaseAccess::identity(b);
}
// Validate every new root/block before changing the ledger. Earlier input
// handles deduplicate both block records and whole-pool capacity.
Status preflight_backing(d::StateStoreData &s, std::span<const ReadLease> incoming,
                        std::span<const unsigned char> metadata,
                        StateStoreStats *projected = nullptr) noexcept {
  collect_backing(s);
  auto stats = stats_locked(s);
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    if (!incoming[i].valid() && metadata[i]) continue; // forecast owned snapshot before allocation
    HYLE_ASSIGN_OR_RETURN(auto b, d::LeaseAccess::backing(incoming[i]));
    bool block = false, root = clone_root(s, b.identity);
    for (const auto &old : s.backing) if (old.anchor.valid()) {
      block |= same_block(old.anchor, incoming[i]); root |= old.backing.identity == b.identity;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (!incoming[j].valid()) continue;
      block |= same_block(incoming[j], incoming[i]);
      HYLE_ASSIGN_OR_RETURN(auto earlier, d::LeaseAccess::backing(incoming[j])); root |= earlier.identity == b.identity;
    }
    if (!block) {
      if (stats.retained_blocks == s.options.max_retained_blocks)
        return Status::Unavailable("state retained block capacity");
      ++stats.retained_blocks;
      if (metadata[i]) {
        if (incoming[i].size() > s.options.max_host_snapshot_bytes - stats.host_snapshot_bytes)
          return Status::Unavailable("state host snapshot byte capacity");
        stats.host_snapshot_bytes += incoming[i].size();
      }
    }
    if (!root) {
      if (b.bytes > s.options.max_backing_bytes - stats.backing_bytes)
        return Status::Unavailable("state backing byte capacity");
      stats.backing_bytes += b.bytes;
    }
  }
  if (projected) *projected = stats;
  return Status::Ok();
}
void retain_backing(d::StateStoreData &s, const ReadLease &incoming, bool metadata) noexcept {
  for (const auto &b : s.backing) if (b.anchor.valid() && same_block(b.anchor, incoming)) return;
  for (auto &b : s.backing) if (!b.anchor.valid()) {
    b.backing = *d::LeaseAccess::backing(incoming); b.anchor = incoming; b.metadata = metadata; return;
  }
  HYLE_FATAL_ASSERT(false); // preflight reserved enough fixed ledger records
}
Status check_resource(const StateEntrySpec &spec, const ReadLease &value) noexcept {
  if (spec.kind != StateEntryKind::kResource || !value.valid())
    return Status::InvalidArgument("state resource member requires a readonly lease");
  HYLE_ASSIGN_OR_RETURN(auto bytes, spec.description.required_bytes());
  if (value.size() != bytes) return Status::InvalidArgument("state resource byte extent disagrees with schema");
  HYLE_ASSIGN_OR_RETURN(auto tensor, value.tensor(spec.description));
  (void)tensor;
  HYLE_ASSIGN_OR_RETURN(auto b, d::LeaseAccess::backing(value));
  if (spec.resource.valid() && !(spec.resource == b.resource))
    return Status::InvalidArgument("state member resource disagrees with schema");
  return Status::Ok();
}
Result<ReadLease> metadata_copy(d::StateStoreData &s, std::span<const std::byte> source) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto backing, mem::Buffer::create(mem::system_resource(), source.size(), 64));
  HYLE_RETURN_IF_ERROR(backing.view().write(source));
  return ReadLease::adopt(std::move(backing), {.allocator=s.allocator, .resource_budget=s.options.resource_budget});
}
Status open_draft(const d::StateStoreData &s, const d::StateRecord &r) noexcept {
  if (s.closed || r.finished) return Status::FailedPrecondition("state draft is closed or finalized");
  return r.failure;
}
} // namespace

StateSchema::StateSchema(const StateSchema &s) noexcept : data_(s.data_) { if (data_) data_->incref(); }
StateSchema &StateSchema::operator=(const StateSchema &s) noexcept {
  if (this != &s) { StateSchema copy(s); std::swap(data_, copy.data_); } return *this;
}
StateSchema::StateSchema(StateSchema &&s) noexcept : data_(std::exchange(s.data_, nullptr)) {}
StateSchema &StateSchema::operator=(StateSchema &&s) noexcept {
  if (this != &s) { StateSchema copy(std::move(s)); std::swap(data_, copy.data_); } return *this;
}
StateSchema::~StateSchema() { if (data_) data_->decref(); }
Result<StateSchema> StateSchema::create(std::span<const StateEntrySpec> specs, mem::AllocatorRef a) noexcept {
  if (!a.valid() || specs.size() >= UINT32_MAX)
    return std::unexpected(Status::InvalidArgument("invalid state schema allocator or size"));
  for (const auto &s : specs) {
    if (s.kind == StateEntryKind::kResource) {
      if (!s.description.valid())
        return std::unexpected(Status::InvalidArgument("state resource requires a description"));
      HYLE_ASSIGN_OR_RETURN(auto bytes, s.description.required_bytes());
      if (bytes == 0 || s.metadata_bytes != 0)
        return std::unexpected(Status::InvalidArgument("state resource has empty or conflicting extent"));
    } else if (s.kind != StateEntryKind::kHostMetadata || s.metadata_bytes == 0 ||
               s.description.valid() || s.resource.valid())
      return std::unexpected(Status::InvalidArgument("state host metadata requires only a nonempty byte extent"));
  }
  try {
    auto *p = a.construct<d::StateSchemaData>(a);
    if (!p) return std::unexpected(Status::ResourceExhausted("state schema metadata"));
    StateSchema result(p); p->specs.assign(specs.begin(), specs.end()); return result;
  } catch (const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("state schema metadata")); }
}
std::size_t StateSchema::size() const noexcept { return data_ ? data_->specs.size() : 0; }
Result<StateEntryId> StateSchema::entry(std::size_t index) const noexcept {
  if (!data_ || index >= size()) return std::unexpected(Status::OutOfRange("state schema entry index"));
  return StateEntryId(*this, std::uint32_t(index));
}
Result<StateEntrySpec> StateSchema::specification(const StateEntryId &id) const noexcept {
  if (!data_ || id.schema_.data_ != data_ || id.index_ >= size())
    return std::unexpected(Status::InvalidArgument("foreign state entry ID"));
  return data_->specs[id.index_];
}
bool StateEntryId::operator==(const StateEntryId &id) const noexcept {
  return d::StateAccess::schema(schema_) == d::StateAccess::schema(id.schema_) && index_ == id.index_;
}

StateStore::StateStore(const StateStore &s) noexcept : data_(s.data_) { if (data_) data_->incref(); }
StateStore &StateStore::operator=(const StateStore &s) noexcept {
  if (this != &s) { StateStore copy(s); std::swap(data_, copy.data_); } return *this;
}
StateStore::StateStore(StateStore &&s) noexcept : data_(std::exchange(s.data_, nullptr)) {}
StateStore &StateStore::operator=(StateStore &&s) noexcept {
  if (this != &s) { StateStore copy(std::move(s)); std::swap(data_, copy.data_); } return *this;
}
StateStore::~StateStore() { if (data_) data_->decref(); }
Result<StateStore> StateStore::create(const StateSchema &schema, StateStoreOptions o) noexcept {
  if (!schema.valid() || !o.allocator.valid() || !o.max_versions || !o.max_drafts ||
      o.max_versions >= UINT32_MAX-o.max_drafts || !o.max_retained_blocks ||
      !o.max_clone_slots || o.max_clone_slots == UINT32_MAX || o.max_retained_blocks == UINT32_MAX ||
      !o.max_backing_bytes || !o.max_host_snapshot_bytes)
    return std::unexpected(Status::InvalidArgument("invalid state store capacities"));
  const auto count = std::size_t(o.max_versions) + o.max_drafts;
  if (schema.size() && count > SIZE_MAX / schema.size() / sizeof(d::StateCell))
    return std::unexpected(Status::OutOfRange("state directory extent overflow"));
  auto a = o.allocator;
  if (o.resource_budget.valid()) { HYLE_ASSIGN_OR_RETURN(a, o.resource_budget.metadata_allocator()); }
  try {
    auto *p = a.construct<d::StateStoreData>(schema, std::move(o), a);
    if (!p) return std::unexpected(Status::ResourceExhausted("state store metadata"));
    return StateStore(p);
  } catch (const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("state store metadata")); }
}
Result<StateVersion> StateStore::import(std::span<const StateMember> members) noexcept {
  if (!data_) return std::unexpected(Status::FailedPrecondition("state store is empty"));
  try {
    std::lock_guard guard(data_->lock);
    HYLE_ASSIGN_OR_RETURN(auto slot, vacant(*data_, false));
    if (members.size() != data_->schema.size())
      return std::unexpected(Status::InvalidArgument("state import must supply the complete schema"));
    d::StateArray<ReadLease> incoming{mem::StdAllocator<ReadLease>(data_->allocator)};
    d::StateArray<unsigned char> seen{mem::StdAllocator<unsigned char>(data_->allocator)};
    incoming.resize(members.size()); seen.resize(members.size());
    const auto &specs = d::StateAccess::schema(data_->schema)->specs;
    for (const auto &m : members) {
      HYLE_ASSIGN_OR_RETURN(auto index, d::StateAccess::entry(*data_, m.entry));
      if (seen[index]) return std::unexpected(Status::InvalidArgument("duplicate state member"));
      seen[index] = 1;
      if (specs[index].kind == StateEntryKind::kResource) {
        if (!m.metadata.empty()) return std::unexpected(Status::InvalidArgument("resource member has metadata bytes"));
        HYLE_RETURN_IF_ERROR(check_resource(specs[index], m.resource));
      } else if (m.resource.valid() || m.metadata.size() != specs[index].metadata_bytes)
        return std::unexpected(Status::InvalidArgument("state metadata byte extent disagrees with schema"));
    }
    std::fill(seen.begin(), seen.end(), static_cast<unsigned char>(0));
    for (const auto &m : members) {
      const auto index = *d::StateAccess::entry(*data_, m.entry);
      if (specs[index].kind == StateEntryKind::kResource) incoming[index] = m.resource;
      else seen[index] = 1;
    }
    StateStoreStats projected;
    HYLE_RETURN_IF_ERROR(preflight_backing(*data_, incoming, seen, &projected));
    // Each owned snapshot creates one fresh block/root. Forecast all bytes and
    // records before allocating or adopting any host payload.
    for (const auto &m : members) if (!m.metadata.empty()) {
      if (projected.retained_blocks == data_->options.max_retained_blocks ||
          m.metadata.size() > data_->options.max_host_snapshot_bytes - projected.host_snapshot_bytes ||
          m.metadata.size() > data_->options.max_backing_bytes - projected.backing_bytes)
        return std::unexpected(Status::Unavailable("state snapshot capacity before allocation"));
      ++projected.retained_blocks; projected.host_snapshot_bytes += m.metadata.size();
      projected.backing_bytes += m.metadata.size();
    }
    for (const auto &m : members) {
      const auto index = *d::StateAccess::entry(*data_, m.entry);
      if (seen[index]) { HYLE_ASSIGN_OR_RETURN(incoming[index], metadata_copy(*data_, m.metadata)); }
    }
    HYLE_RETURN_IF_ERROR(preflight_backing(*data_, incoming, seen));
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      retain_backing(*data_, incoming[i], seen[i] != 0);
      data_->records[slot].members[i].lease = std::move(incoming[i]);
    }
    activate(*data_, slot); return d::StateAccess::version(data_, slot);
  } catch (const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("state import metadata")); }
}
Result<StateDraft> StateStore::begin(const StateVersion &base) noexcept {
  if (!data_ || d::StateAccess::store(base) != data_)
    return std::unexpected(Status::InvalidArgument("state base belongs to a different store"));
  std::lock_guard guard(data_->lock);
  HYLE_ASSIGN_OR_RETURN(auto slot, vacant(*data_, true));
  const auto &source = d::StateAccess::record(base).members;
  for (std::size_t i = 0; i < source.size(); ++i) data_->records[slot].members[i].lease = source[i].lease;
  activate(*data_, slot); return d::StateAccess::draft(data_, slot);
}
StateStoreStats StateStore::stats() const noexcept {
  if (!data_) return {};
  std::lock_guard guard(data_->lock); return stats_locked(*data_);
}
Status StateStore::collect() noexcept {
  if (!data_) return Status::Ok();
  std::lock_guard guard(data_->lock);
  Status first;
  for (auto &r : data_->routes) { const auto status = r.driver.collect(r.driver.object); if (first.ok()) first = status; }
  collect_backing(*data_); return first;
}
Status StateStore::close() noexcept {
  if (!data_) return Status::Ok();
  std::lock_guard guard(data_->lock);
  if (data_->closed) return stats_locked(*data_).active_clones ? Status::Unavailable("state clones still retire") : Status::Ok();
  data_->closed = true;
  Status first;
  for (auto &r : data_->routes) { const auto status = r.driver.close(r.driver.object); if (first.ok()) first = status; }
  return first;
}
Status StateStore::drain() noexcept {
  close().IgnoreError(); // pending work makes close unavailable; drain must still wait
  if (!data_) return Status::Ok();
  // closed prevents route registration. Handles retain this store; no monitor
  // is held across original native host waits.
  {
    std::lock_guard guard(data_->lock);
    for (auto &r : data_->routes) r.driver.observe(r.driver.object, UINT32_MAX, true);
  }
  Status first;
  for (auto &r : data_->routes) { const auto status = r.driver.drain(r.driver.object); if (first.ok()) first = status; }
  {
    std::lock_guard guard(data_->lock);
    for (auto &r : data_->routes) r.driver.observe(r.driver.object, UINT32_MAX, false);
  }
  const auto collected = collect(); return first.ok() ? collected : first;
}
StateVersion::StateVersion(const StateVersion &v) noexcept : store_(v.store_), slot_(v.slot_), generation_(v.generation_) {
  if (valid()) HYLE_FATAL_ASSERT(d::StateAccess::record(*this).refs.fetch_add(1, std::memory_order_relaxed) != UINT32_MAX);
}
StateVersion &StateVersion::operator=(const StateVersion &v) noexcept {
  if (this != &v) { StateVersion copy(v); std::swap(store_, copy.store_); std::swap(slot_, copy.slot_); std::swap(generation_, copy.generation_); }
  return *this;
}
StateVersion::StateVersion(StateVersion &&v) noexcept
    : store_(std::move(v.store_)), slot_(v.slot_), generation_(v.generation_) {}
StateVersion &StateVersion::operator=(StateVersion &&v) noexcept {
  if (this != &v) { reset(); store_ = std::move(v.store_); slot_ = v.slot_; generation_ = v.generation_; } return *this;
}
StateVersion::~StateVersion() { reset(); }
void StateVersion::reset() noexcept {
  if (valid()) { d::StateAccess::drop_record(d::StateAccess::store(*this), slot_); store_ = {}; }
}
Result<ReadLease> StateVersion::member(const StateEntryId &id) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("state version is empty"));
  HYLE_ASSIGN_OR_RETURN(auto entry, d::StateAccess::entry(*d::StateAccess::store(*this), id));
  return d::StateAccess::record(*this).members[entry].lease;
}
StateIdentity StateVersion::identity() const noexcept { return StateIdentity(*this); }
bool StateIdentity::operator==(const StateIdentity &id) const noexcept {
  return d::StateAccess::store(version_) == d::StateAccess::store(id.version_) &&
      version_.generation_ == id.version_.generation_;
}
StateDraft::StateDraft(StateDraft &&v) noexcept : store_(std::move(v.store_)), slot_(v.slot_), generation_(v.generation_) {}
StateDraft &StateDraft::operator=(StateDraft &&v) noexcept {
  if (this != &v) { StateDraft old(std::move(*this)); store_ = std::move(v.store_); slot_ = v.slot_; generation_ = v.generation_; }
  return *this;
}
StateDraft::~StateDraft() {
  if (valid()) { cancel().IgnoreError(); d::StateAccess::drop_record(d::StateAccess::store(*this), slot_); }
}
Status StateDraft::cancel() noexcept {
  if (!valid()) return Status::Ok();
  auto &s = *d::StateAccess::store(*this); std::lock_guard guard(s.lock);
  auto &r = d::StateAccess::record(*this);
  if (!r.finished && r.failure.ok()) r.failure = Status::Cancelled("state draft abandoned");
  for (auto &m : r.members) if (m.pending && m.route != UINT32_MAX && m.copy != UINT32_MAX) {
    auto &driver=s.routes[m.route].driver; driver.cancel(driver.object,m.copy).IgnoreError();
  }
  return Status::Ok();
}
Status StateDraft::replace(const StateEntryId &id, const ReadLease &value) noexcept {
  if (!valid()) return Status::FailedPrecondition("state draft is empty");
  auto &s = *d::StateAccess::store(*this); std::lock_guard guard(s.lock);
  auto &r = d::StateAccess::record(*this); HYLE_RETURN_IF_ERROR(open_draft(s, r));
  HYLE_ASSIGN_OR_RETURN(auto entry, d::StateAccess::entry(s, id));
  if (r.members[entry].pending) return Status::Unavailable("state member has a pending writer");
  HYLE_RETURN_IF_ERROR(check_resource(d::StateAccess::schema(s.schema)->specs[entry], value));
  const unsigned char metadata = 0;
  HYLE_RETURN_IF_ERROR(preflight_backing(s, std::span(&value, 1), std::span(&metadata, 1)));
  retain_backing(s, value, false); r.members[entry].lease = value; return Status::Ok();
}
Status StateDraft::replace_metadata(const StateEntryId &id, std::span<const std::byte> value) noexcept {
  if (!valid()) return Status::FailedPrecondition("state draft is empty");
  auto &s = *d::StateAccess::store(*this); std::lock_guard guard(s.lock);
  auto &r = d::StateAccess::record(*this); HYLE_RETURN_IF_ERROR(open_draft(s, r));
  HYLE_ASSIGN_OR_RETURN(auto entry, d::StateAccess::entry(s, id));
  const auto &spec = d::StateAccess::schema(s.schema)->specs[entry];
  if (spec.kind != StateEntryKind::kHostMetadata || value.size() != spec.metadata_bytes)
    return Status::InvalidArgument("state metadata extent disagrees with schema");
  collect_backing(s);const auto stats=stats_locked(s);
  if(stats.retained_blocks==s.options.max_retained_blocks ||
      value.size()>s.options.max_host_snapshot_bytes-stats.host_snapshot_bytes ||
      value.size()>s.options.max_backing_bytes-stats.backing_bytes)
    return Status::Unavailable("state snapshot capacity before allocation");
  HYLE_ASSIGN_OR_RETURN(auto ready, metadata_copy(s, value));
  const unsigned char metadata = 1;
  HYLE_RETURN_IF_ERROR(preflight_backing(s, std::span(&ready, 1), std::span(&metadata, 1)));
  retain_backing(s, ready, true); r.members[entry].lease = std::move(ready); return Status::Ok();
}
Result<StateVersion> StateDraft::finalize() noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("state draft is empty"));
  auto &s = *d::StateAccess::store(*this); std::lock_guard guard(s.lock);
  auto &r = d::StateAccess::record(*this); HYLE_RETURN_IF_ERROR(open_draft(s, r));
  for (const auto &m : r.members) if (m.pending || !m.lease.valid())
    return std::unexpected(Status::Unavailable("state candidate has an unfinished member"));
  HYLE_ASSIGN_OR_RETURN(auto slot, vacant(s, false));
  for (std::size_t i = 0; i < r.members.size(); ++i) s.records[slot].members[i].lease = r.members[i].lease;
  activate(s, slot); r.finished = true; return d::StateAccess::version(&s, slot);
}

StateHead::StateHead(const StateHead &h) noexcept : data_(h.data_) { if (data_) data_->incref(); }
StateHead &StateHead::operator=(const StateHead &h) noexcept {
  if (this != &h) { StateHead copy(h); std::swap(data_, copy.data_); } return *this;
}
StateHead::StateHead(StateHead &&h) noexcept : data_(std::exchange(h.data_, nullptr)) {}
StateHead &StateHead::operator=(StateHead &&h) noexcept {
  if (this != &h) { StateHead copy(std::move(h)); std::swap(data_, copy.data_); } return *this;
}
StateHead::~StateHead() { if (data_) data_->decref(); }
Result<StateHead> StateHead::create(const StateStore &store, const StateVersion &initial) noexcept {
  auto *s = d::StateAccess::store(store);
  if (!s || d::StateAccess::store(initial) != s)
    return std::unexpected(Status::InvalidArgument("head initial version belongs to a different store"));
  std::lock_guard guard(s->lock);
  if (s->closed) return std::unexpected(Status::FailedPrecondition("state store is closed"));
  auto *p = s->allocator.construct<d::StateHeadData>(store, initial, s->allocator);
  if (!p) return std::unexpected(Status::ResourceExhausted("state head metadata"));
  return StateHead(p);
}
Result<StateVersion> StateHead::current() const noexcept {
  if (!data_) return std::unexpected(Status::FailedPrecondition("state head is empty"));
  auto *s = d::StateAccess::store(data_->store); std::lock_guard guard(s->lock); return data_->version;
}
Status StateHead::publish(const StateIdentity &expected, const StateVersion &candidate) noexcept {
  if (!data_) return Status::FailedPrecondition("state head is empty");
  auto *s = d::StateAccess::store(data_->store);
  if (d::StateAccess::store(candidate) != s || d::StateAccess::store(expected.version_) != s)
    return Status::InvalidArgument("head publication requires versions from this store");
  StateVersion previous;
  {
    std::lock_guard guard(s->lock);
    if (s->closed) return Status::FailedPrecondition("state store is closed");
    if (!(data_->version.identity() == expected)) return Status::Unavailable("state head expected-base conflict");
    previous = std::move(data_->version); data_->version = candidate;
  }
  return Status::Ok(); // previous releases outside the store monitor
}

Result<mem::AllocatorRef> d::StateCloneAccess::allocator(const StateStore &store) noexcept {
  auto *s=d::StateAccess::store(store);
  if (!s) return std::unexpected(Status::FailedPrecondition("state store is empty"));
  std::lock_guard guard(s->lock);
  if(s->closed) return std::unexpected(Status::FailedPrecondition("state store is closed"));
  return s->allocator;
}
Result<StateCloneRoute> d::StateCloneAccess::register_driver(const StateStore &store,
    StateCopyDriver &driver, mem::MemoryResourceRef resource, std::size_t capacity,
    std::size_t alignment) noexcept {
  auto *s=d::StateAccess::store(store);
  if (!s || !driver.object || !driver.slots || !resource.valid() || !capacity)
    return std::unexpected(Status::InvalidArgument("invalid state copy registration"));
  std::lock_guard guard(s->lock);
  if(s->closed) return std::unexpected(Status::FailedPrecondition("state store is closed"));
  collect_backing(*s);const auto stats=stats_locked(*s);
  if(driver.slots>s->options.max_clone_slots-stats.registered_clone_slots ||
     capacity>s->options.max_backing_bytes-stats.backing_bytes)
    return std::unexpected(Status::Unavailable("state clone registration capacity"));
  HYLE_ASSIGN_OR_RETURN(auto pool,LeasePool::create(resource,capacity,
      {.max_blocks=s->options.max_retained_blocks,.backing_alignment=alignment,
       .allocator=s->allocator,.resource_budget=s->options.resource_budget}));
  StateCopyRoute route;route.pool=std::move(pool);route.resource=resource;
  route.capacity=capacity;route.alignment=alignment;
  const auto index=std::uint32_t(s->routes.size());
  s->routes.push_back(std::move(route)); // capacity reserved at Store creation
  s->routes.back().driver=std::exchange(driver,{});
  return StateCloneRoute(store,index);
}
Status d::StateCloneAccess::preflight(const StateStore &store,std::uint32_t slots,std::size_t capacity) noexcept {
  auto *s=d::StateAccess::store(store);
  if(!s) return Status::FailedPrecondition("state store is empty");
  std::lock_guard guard(s->lock);
  if(s->closed) return Status::FailedPrecondition("state store is closed");
  collect_backing(*s);const auto stats=stats_locked(*s);
  if(slots>s->options.max_clone_slots-stats.registered_clone_slots ||
      capacity>s->options.max_backing_bytes-stats.backing_bytes)
    return Status::Unavailable("state clone capacity before cold factory");
  return Status::Ok();
}
Status StateCloneRoute::rearm() noexcept {
  if(!valid()) return Status::FailedPrecondition("state clone route is empty");
  auto *s=d::StateAccess::store(store_);std::lock_guard guard(s->lock);
  if(s->closed) return Status::FailedPrecondition("state store is closed");
  auto &driver=s->routes[route_].driver;return driver.rearm(driver.object);
}
Result<CopyBatchReport> StateCloneRoute::report() const noexcept {
  if(!valid()) return std::unexpected(Status::FailedPrecondition("state clone route is empty"));
  auto *s=d::StateAccess::store(store_);std::lock_guard guard(s->lock);
  auto &driver=s->routes[route_].driver;return driver.report(driver.object);
}
Result<CloneTicket> StateDraft::clone_for_write(const StateEntryId &id, const StateCloneRoute &route) noexcept {
  if(!valid() || d::StateAccess::store(d::StateAccess::owner(route)) != d::StateAccess::store(*this))
    return std::unexpected(Status::InvalidArgument("state clone requires a route from this store"));
  auto *s=d::StateAccess::store(*this);std::lock_guard guard(s->lock);
  auto &record=d::StateAccess::record(*this);HYLE_RETURN_IF_ERROR(open_draft(*s,record));
  HYLE_ASSIGN_OR_RETURN(auto entry,d::StateAccess::entry(*s,id));
  const auto &spec=d::StateAccess::schema(s->schema)->specs[entry];
  if(spec.kind!=StateEntryKind::kResource || spec.description.storage_format().kind()==StorageFormatKind::kOpaque)
    return std::unexpected(Status::Unimplemented("state clone requires qualified byte-addressable resource encoding"));
  auto &member=record.members[entry];
  if(member.pending) return std::unexpected(Status::Unavailable("state member already has a pending writer"));
  auto &target=s->routes[d::StateAccess::route(route)];
  if(spec.resource.valid() && !(spec.resource==target.resource))
    return std::unexpected(Status::InvalidArgument("state clone destination violates schema resource"));
  HYLE_ASSIGN_OR_RETURN(auto destination,target.pool.allocate(member.lease.size(),target.alignment));
  HYLE_ASSIGN_OR_RETURN(auto root,d::LeaseAccess::backing(destination));target.root=root.identity;
  HYLE_ASSIGN_OR_RETURN(auto copy,target.driver.start(target.driver.object,member.lease,std::move(destination),member.lease.size()));
  member.pending=true;member.route=d::StateAccess::route(route);member.copy=copy;
  HYLE_FATAL_ASSERT(record.refs.fetch_add(1,std::memory_order_relaxed)!=UINT32_MAX);
  return CloneTicket(store_,slot_,entry,member.route,copy);
}
CloneTicket::CloneTicket(CloneTicket &&t) noexcept : store_(std::move(t.store_)),draft_(t.draft_),
    entry_(t.entry_),route_(t.route_),slot_(t.slot_) {}
CloneTicket &CloneTicket::operator=(CloneTicket &&t) noexcept {
  if(this!=&t) { CloneTicket old(std::move(*this));store_=std::move(t.store_);draft_=t.draft_;
    entry_=t.entry_;route_=t.route_;slot_=t.slot_; }return *this;
}
CloneTicket::~CloneTicket() {
  if(valid()) { request_cancel().IgnoreError();d::StateAccess::drop_record(d::StateAccess::store(store_),draft_); }
}
Status CloneTicket::request_cancel() noexcept {
  if(!valid()) return Status::Ok();
  auto *s=d::StateAccess::store(store_);std::lock_guard guard(s->lock);
  auto &record=s->records[draft_];
  if(record.failure.ok()) record.failure=Status::Cancelled("state clone candidate cancelled");
  auto &driver=s->routes[route_].driver;return driver.cancel(driver.object,slot_);
}
Result<StateWriter> CloneTicket::poll() noexcept { return take(false); }
Result<StateWriter> CloneTicket::wait() noexcept { return take(true); }
Result<StateWriter> CloneTicket::take(bool wait) noexcept {
  if(!valid()) return std::unexpected(Status::FailedPrecondition("state clone ticket is empty"));
  auto *s=d::StateAccess::store(store_);auto &driver=s->routes[route_].driver;
  if(wait) {
    { std::lock_guard guard(s->lock);driver.observe(driver.object,slot_,true); }
    const auto terminal=driver.wait(driver.object,slot_);
    { std::lock_guard guard(s->lock);driver.observe(driver.object,slot_,false);
      if(!terminal.ok() && !terminal.IsUnavailable() && s->records[draft_].failure.ok())
        s->records[draft_].failure=terminal; }
    if(!terminal.ok()) return std::unexpected(terminal);
  }
  std::lock_guard guard(s->lock);auto &record=s->records[draft_];
  HYLE_RETURN_IF_ERROR(open_draft(*s,record));
  auto taken=driver.take(driver.object,slot_,false);
  if(!taken) {
    if(!taken.error().IsUnavailable()) record.failure=taken.error();
    return std::unexpected(taken.error());
  }
  auto &member=record.members[entry_];member.copy=UINT32_MAX;
  // Transfer this ticket's draft reference; no extra root/permission is made.
  return StateWriter(std::move(store_),draft_,entry_,std::move(*taken));
}
StateWriter::StateWriter(StateStore store,std::uint32_t draft,std::uint32_t entry,WriteLease writer) noexcept
    : store_(std::move(store)),draft_(draft),entry_(entry),writer_(std::move(writer)),
      block_(d::LeaseAccess::identity(writer_)),generation_(writer_.generation()) {}
StateWriter::StateWriter(StateWriter &&w) noexcept : store_(std::move(w.store_)),draft_(w.draft_),entry_(w.entry_),
    writer_(std::move(w.writer_)),block_(w.block_),generation_(w.generation_),complete_(w.complete_) {}
StateWriter &StateWriter::operator=(StateWriter &&w) noexcept {
  if(this!=&w) { StateWriter old(std::move(*this));store_=std::move(w.store_);draft_=w.draft_;entry_=w.entry_;
    writer_=std::move(w.writer_);block_=w.block_;generation_=w.generation_;complete_=w.complete_; }return *this;
}
StateWriter::~StateWriter() {
  if(valid()) {
    auto *s=d::StateAccess::store(store_);
    if(!complete_) { std::lock_guard guard(s->lock);auto &r=s->records[draft_];
      if(r.failure.ok()) r.failure=Status::Cancelled("state writer abandoned without publication"); }
    writer_.reset();d::StateAccess::drop_record(s,draft_);
  }
}
Status StateWriter::publish() noexcept {
  if(!valid() || complete_) return Status::FailedPrecondition("state writer is empty or complete");
  auto *s=d::StateAccess::store(store_);std::lock_guard guard(s->lock);
  auto &r=s->records[draft_];HYLE_RETURN_IF_ERROR(open_draft(*s,r));
  if(!writer_.valid()) return Status::FailedPrecondition("native consumer owns the writer; complete its ready reader");
  if(d::LeaseAccess::identity(writer_)!=block_ || writer_.generation()!=generation_)
    return Status::InvalidArgument("state writer mutable handle was replaced by a foreign block");
  // Reserve a tracking record before publish consumes the mutable handle.
  collect_backing(*s);
  if(stats_locked(*s).retained_blocks==s->options.max_retained_blocks)
    return Status::Unavailable("state retained block capacity");
  auto published=writer_.publish();
  if(!published) {
    if(!published.error().IsUnavailable() && r.failure.ok()) r.failure=published.error();
    return published.error();
  }
  auto ready=std::move(*published);
  retain_backing(*s,ready,false);r.members[entry_].lease=std::move(ready);
  r.members[entry_].pending=false;r.members[entry_].route=r.members[entry_].copy=UINT32_MAX;
  complete_=true;return Status::Ok();
}
Status StateWriter::complete(const ReadLease &ready) noexcept {
  if(!valid() || complete_) return Status::FailedPrecondition("state writer is empty or complete");
  if(writer_.valid() || d::LeaseAccess::identity(ready)!=block_ || ready.generation()!=generation_)
    return Status::InvalidArgument("state writer completion must return its original ready block");
  auto *s=d::StateAccess::store(store_);std::lock_guard guard(s->lock);
  auto &r=s->records[draft_];HYLE_RETURN_IF_ERROR(open_draft(*s,r));
  HYLE_RETURN_IF_ERROR(check_resource(d::StateAccess::schema(s->schema)->specs[entry_],ready));
  const unsigned char metadata=0;
  HYLE_RETURN_IF_ERROR(preflight_backing(*s,std::span(&ready,1),std::span(&metadata,1)));
  retain_backing(*s,ready,false);r.members[entry_].lease=ready;
  r.members[entry_].pending=false;r.members[entry_].route=r.members[entry_].copy=UINT32_MAX;
  complete_=true;return Status::Ok();
}
} // namespace hyle::compute
