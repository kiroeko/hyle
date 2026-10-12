#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/compute/batch_transfer.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_STATE_API
#elif defined(HYLE_COMPUTE_STATE_BUILD)
#define HYLE_COMPUTE_STATE_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_STATE_API HYLE_IMPORT
#endif

namespace hyle::compute {
namespace detail { struct StateSchemaData; struct StateStoreData; struct StateAccess; }
class StateEntryId;
class StateVersion;
class StateDraft;
class StateHead;
class StateCloneRoute;
class CloneTicket;
class StateWriter;
class StateIdentity;

enum class StateEntryKind : unsigned char { kResource, kHostMetadata };
struct StateEntrySpec final {
  StateEntryKind kind = StateEntryKind::kResource;
  TensorDesc description{};
  // Resource members use description.required_bytes(); metadata has this
  // exact byte length and is copied, not interpreted or serialized.
  std::size_t metadata_bytes = 0;
  mem::MemoryResourceRef resource{}; // empty accepts any qualified F1 resource
};

// Immutable schema. Injected allocator remains borrowed through all IDs and
// versions. A schema ID retains its owner; ordinals alone are not identities.
class StateSchema final {
public:
  StateSchema() noexcept = default;
  HYLE_COMPUTE_STATE_API StateSchema(const StateSchema &) noexcept;
  HYLE_COMPUTE_STATE_API StateSchema &operator=(const StateSchema &) noexcept;
  HYLE_COMPUTE_STATE_API StateSchema(StateSchema &&) noexcept;
  HYLE_COMPUTE_STATE_API StateSchema &operator=(StateSchema &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateSchema();
  HYLE_COMPUTE_STATE_API static Result<StateSchema> create(std::span<const StateEntrySpec>,
      mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_STATE_API std::size_t size() const noexcept;
  HYLE_COMPUTE_STATE_API Result<StateEntryId> entry(std::size_t) const noexcept;
  HYLE_COMPUTE_STATE_API Result<StateEntrySpec> specification(const StateEntryId &) const noexcept;
private:
  friend struct detail::StateAccess;
  explicit StateSchema(detail::StateSchemaData *data) noexcept : data_(data) {}
  detail::StateSchemaData *data_ = nullptr;
};

class StateEntryId final {
public:
  StateEntryId() noexcept = default;
  bool valid() const noexcept { return schema_.valid(); }
  bool operator==(const StateEntryId &) const noexcept;
private:
  friend class StateSchema;
  friend struct detail::StateAccess;
  StateEntryId(StateSchema schema, std::uint32_t index) noexcept
      : schema_(std::move(schema)), index_(index) {}
  StateSchema schema_;
  std::uint32_t index_ = 0;
};

struct StateMember final {
  StateEntryId entry;
  ReadLease resource{};
  std::span<const std::byte> metadata{};
};
struct StateStoreOptions final {
  std::uint32_t max_versions = 16, max_drafts = 8;
  std::uint32_t max_clone_slots = 4, max_retained_blocks = 128;
  std::size_t max_backing_bytes = 128u * 1024u * 1024u;
  std::size_t max_host_snapshot_bytes = 1024u * 1024u;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct StateStoreStats final {
  std::uint32_t versions = 0, drafts = 0, retained_blocks = 0, backing_roots = 0;
  std::uint32_t registered_clone_slots = 0, active_clones = 0;
  std::size_t backing_bytes = 0, host_snapshot_bytes = 0, clone_capacity_bytes = 0;
  bool closed = false;
};

// Opt-in bounded collection owner. Sharing never copies payload. Native
// resources/Contexts and allocators remain borrowed through every final use.
// Store/head operations are thread-safe; each Draft/Writer/Ticket is compatible
// with one externally synchronized owner. Copies/drops of Versions may cross
// threads. collect is nonblocking; drain waits only this store's clone work.
class StateStore final {
public:
  StateStore() noexcept = default;
  HYLE_COMPUTE_STATE_API StateStore(const StateStore &) noexcept;
  HYLE_COMPUTE_STATE_API StateStore &operator=(const StateStore &) noexcept;
  HYLE_COMPUTE_STATE_API StateStore(StateStore &&) noexcept;
  HYLE_COMPUTE_STATE_API StateStore &operator=(StateStore &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateStore();
  HYLE_COMPUTE_STATE_API static Result<StateStore> create(const StateSchema &,
      StateStoreOptions options = {}) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_STATE_API Result<StateVersion> import(std::span<const StateMember>) noexcept;
  HYLE_COMPUTE_STATE_API Result<StateDraft> begin(const StateVersion &) noexcept;
  HYLE_COMPUTE_STATE_API StateStoreStats stats() const noexcept;
  HYLE_COMPUTE_STATE_API Status collect() noexcept;
  HYLE_COMPUTE_STATE_API Status close() noexcept;
  HYLE_COMPUTE_STATE_API Status drain() noexcept;
private:
  friend struct detail::StateAccess;
  explicit StateStore(detail::StateStoreData *data) noexcept : data_(data) {}
  detail::StateStoreData *data_ = nullptr;
};

// All members are successfully ready readonly leases. This handle owns the
// collection, not just an ordinal. member exports a shared F1 reader; it may
// outlive the directory and remains in the store's backing capacity ledger.
class StateVersion final {
public:
  StateVersion() noexcept = default;
  HYLE_COMPUTE_STATE_API StateVersion(const StateVersion &) noexcept;
  HYLE_COMPUTE_STATE_API StateVersion &operator=(const StateVersion &) noexcept;
  HYLE_COMPUTE_STATE_API StateVersion(StateVersion &&) noexcept;
  HYLE_COMPUTE_STATE_API StateVersion &operator=(StateVersion &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateVersion();
  bool valid() const noexcept { return store_.valid(); }
  HYLE_COMPUTE_STATE_API Result<ReadLease> member(const StateEntryId &) const noexcept;
  HYLE_COMPUTE_STATE_API StateIdentity identity() const noexcept;
  HYLE_COMPUTE_STATE_API void reset() noexcept;
private:
  friend struct detail::StateAccess;
  friend class StateIdentity;
  StateVersion(StateStore store, std::uint32_t slot, std::uint64_t generation) noexcept
      : store_(std::move(store)), slot_(slot), generation_(generation) {}
  StateStore store_;
  std::uint32_t slot_ = 0;
  std::uint64_t generation_ = 0;
};
class StateIdentity final {
public:
  StateIdentity() noexcept = default;
  bool valid() const noexcept { return version_.valid(); }
  HYLE_COMPUTE_STATE_API bool operator==(const StateIdentity &) const noexcept;
private:
  friend class StateVersion;
  friend class StateHead;
  explicit StateIdentity(StateVersion version) noexcept : version_(std::move(version)) {}
  StateVersion version_;
};

class StateDraft final {
public:
  StateDraft() noexcept = default;
  StateDraft(const StateDraft &) = delete;
  StateDraft &operator=(const StateDraft &) = delete;
  HYLE_COMPUTE_STATE_API StateDraft(StateDraft &&) noexcept;
  HYLE_COMPUTE_STATE_API StateDraft &operator=(StateDraft &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateDraft();
  bool valid() const noexcept { return store_.valid(); }
  HYLE_COMPUTE_STATE_API Status replace(const StateEntryId &, const ReadLease &) noexcept;
  HYLE_COMPUTE_STATE_API Status replace_metadata(const StateEntryId &,
                                                std::span<const std::byte>) noexcept;
  HYLE_COMPUTE_STATE_API Result<CloneTicket> clone_for_write(const StateEntryId &,
                                                           const StateCloneRoute &) noexcept;
  HYLE_COMPUTE_STATE_API Result<StateVersion> finalize() noexcept;
  HYLE_COMPUTE_STATE_API Status cancel() noexcept;
private:
  friend struct detail::StateAccess;
  StateDraft(StateStore store, std::uint32_t slot, std::uint64_t generation) noexcept
      : store_(std::move(store)), slot_(slot), generation_(generation) {}
  StateStore store_;
  std::uint32_t slot_ = 0;
  std::uint64_t generation_ = 0;
};

namespace detail { struct StateHeadData; }
class StateHead final {
public:
  StateHead() noexcept = default;
  HYLE_COMPUTE_STATE_API StateHead(const StateHead &) noexcept;
  HYLE_COMPUTE_STATE_API StateHead &operator=(const StateHead &) noexcept;
  HYLE_COMPUTE_STATE_API StateHead(StateHead &&) noexcept;
  HYLE_COMPUTE_STATE_API StateHead &operator=(StateHead &&) noexcept;
  HYLE_COMPUTE_STATE_API ~StateHead();
  HYLE_COMPUTE_STATE_API static Result<StateHead> create(const StateStore &,
                                                       const StateVersion &) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_STATE_API Result<StateVersion> current() const noexcept;
  HYLE_COMPUTE_STATE_API Status publish(const StateIdentity &, const StateVersion &) noexcept;
private:
  explicit StateHead(detail::StateHeadData *data) noexcept : data_(data) {}
  detail::StateHeadData *data_ = nullptr;
};
} // namespace hyle::compute
