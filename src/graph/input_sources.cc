#include "input_sources.h"

#include <algorithm>
#include <functional>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

#include "hyle/graph/detail/input_source.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/scope_guard.h"
#include "plan_state.h"
#include "submission_internal.h"

namespace hyle::graph::detail {
namespace {
template <class T> using Vector = std::vector<T, mem::StdAllocator<T>>;

struct Binding final { ValueId value{}; InputSource source{}; };
struct Order final {
  std::size_t source = 0;
  std::uint32_t group = 0;
  InputDependencyRoute route = InputDependencyRoute::kHostWait;
  bool ordered = false;
  bool used = false;
};
struct Use final { std::size_t segment = 0; std::size_t order = 0; };
struct Range final { std::size_t begin = 0; std::size_t count = 0; };
struct RootSource final { std::uint32_t root; std::size_t source; };
struct Consumer final { std::size_t segment; std::size_t source; std::uint32_t group; };
struct Prepared final {
  explicit Prepared(mem::AllocatorRef allocator) noexcept
      : sources(mem::StdAllocator<InputSource>(allocator)),
        orders(mem::StdAllocator<Order>(allocator)),
        uses(mem::StdAllocator<Use>(allocator)),
        ranges(mem::StdAllocator<Range>(allocator)),
        results(mem::StdAllocator<std::size_t>(allocator)),
        ready(mem::StdAllocator<unsigned char>(allocator)) {}
  void clear() noexcept {
    uses.clear(); ranges.clear(); orders.clear(); results.clear(); ready.clear(); sources.clear();
    report = {}; valid = false;
  }
  void swap(Prepared &other) noexcept {
    sources.swap(other.sources); orders.swap(other.orders); uses.swap(other.uses);
    ranges.swap(other.ranges); results.swap(other.results); ready.swap(other.ready);
    std::swap(report, other.report); std::swap(valid, other.valid);
  }
  Vector<InputSource> sources;
  Vector<Order> orders;
  Vector<Use> uses;
  Vector<Range> ranges;
  Vector<std::size_t> results;
  Vector<unsigned char> ready;
  InputSourceReport report{};
  bool valid = false;
};

bool source_less(const InputSource &a, const InputSource &b) noexcept {
  if (a.vtable() != b.vtable())
    return std::less<const InputSourceVTable *>{}(a.vtable(), b.vtable());
  return std::less<const void *>{}(a.identity(), b.identity());
}
std::uint32_t root_instance(const PlanState &plan, std::uint32_t instance) noexcept {
  while (plan.alias_sources[instance] != kNoInPlaceSource)
    instance = plan.alias_sources[instance];
  return instance;
}
} // namespace

struct InputSourcesState final {
  explicit InputSourcesState(mem::AllocatorRef owner) noexcept
      : allocator(owner), bindings(mem::StdAllocator<Binding>(owner)),
        active(owner), candidate(owner),
        roots(mem::StdAllocator<RootSource>(owner)),
        consumers(mem::StdAllocator<Consumer>(owner)) {}
  mem::AllocatorRef allocator;
  Vector<Binding> bindings;
  Prepared active;
  Prepared candidate;
  // Cold scratch belongs to this original slot, outside the replay transaction.
  // Entries are integer indices only; clearing preserves capacity, not owners.
  Vector<RootSource> roots;
  Vector<Consumer> consumers;
};

namespace {
Binding *find_binding(InputSourcesState &state, ValueId value) noexcept {
  const auto found = std::lower_bound(state.bindings.begin(), state.bindings.end(), value,
      [](const Binding &binding, ValueId wanted) {
        return binding.value.ordinal() < wanted.ordinal();
      });
  return found != state.bindings.end() && found->value == value ? &*found : nullptr;
}
void invalidate(InputSourcesState &state) noexcept {
  state.active.clear(); state.candidate.clear();
}
Status require_owner(InvocationState &state) noexcept {
  if (state.input_sources) return Status::Ok();
  InputSourcesOwner candidate;
  try {
    auto *created = state.allocator.construct<InputSourcesState>(state.allocator);
    if (!created)
      return Status::ResourceExhausted("allocating graph immutable input owners");
    candidate = InputSourcesOwner(created);
    created->bindings.reserve(state.plan->input_ids.size());
    for (const auto value : state.plan->input_ids)
      created->bindings.push_back({value, {}});
    std::sort(created->bindings.begin(), created->bindings.end(),
        [](const Binding &a, const Binding &b) { return a.value.ordinal() < b.value.ordinal(); });
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("allocating graph immutable input bindings");
  }
  state.input_sources = std::move(candidate);
  return Status::Ok();
}
Result<Prepared *> prepared_for(InvocationState &state, std::size_t segment) noexcept {
  if (!state.input_sources) return nullptr;
  auto &active = state.input_sources.get()->active;
  if (!active.valid || segment >= active.ranges.size())
    return std::unexpected(Status::FailedPrecondition("graph input sources are not prepared"));
  return &active;
}
} // namespace

Result<InputSource> InputSource::acquire(
    void *state, const InputSourceVTable *vtable, const void *identity) noexcept {
  if (!state || !identity || !vtable || !vtable->retain || !vtable->release || !vtable->route ||
      !vtable->poll || !vtable->wait || !vtable->enqueue || !vtable->consumer_retired)
    return std::unexpected(Status::InvalidArgument("immutable input source contract is incomplete"));
  vtable->retain(state);
  return InputSource(state, vtable, identity);
}

InputSourcesOwner::InputSourcesOwner(InputSourcesOwner &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
InputSourcesOwner &InputSourcesOwner::operator=(InputSourcesOwner &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
InputSourcesOwner::~InputSourcesOwner() { reset(); }
void InputSourcesOwner::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    const auto allocator = state->allocator;
    allocator.destroy(state);
  }
}

Status InputAccess::bind(Invocation &invocation, ValueId value,
    compute::ConstTensorView tensor, InputSource source) noexcept {
  const InputSourceBinding binding{value, tensor, std::move(source)};
  return bind(invocation, std::span<const InputSourceBinding>(&binding, 1));
}
Status InputAccess::bind(Invocation &invocation,
    std::span<const InputSourceBinding> incoming) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (incoming.empty() || incoming.size() > state->plan->input_ids.size())
    return Status::InvalidArgument("immutable input bundle has an invalid size");
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    const auto &item = incoming[i];
    HYLE_RETURN_IF_ERROR(check_binding_replacement(*state, item.value));
    if (!item.source.valid())
      return Status::InvalidArgument("immutable input source is empty");
    if (std::find(state->plan->input_ids.begin(), state->plan->input_ids.end(), item.value) ==
        state->plan->input_ids.end())
      return Status::InvalidArgument("immutable input source requires a graph input");
    for (std::size_t j = 0; j < i; ++j)
      if (incoming[j].value == item.value)
        return Status::InvalidArgument("immutable input bundle repeats a graph input");
  }
  const bool had_sources = static_cast<bool>(state->input_sources);
  HYLE_RETURN_IF_ERROR(require_owner(*state));
  // Attaching even an empty, unprepared source state changes ordinary replay.
  // On failure withdraw only this newly created attachment, after all local
  // rollback owners have been restored/destroyed; keep any preexisting state.
  util::ScopeGuard restore_new_owner{[&]() noexcept {
    if (!had_sources) state->input_sources.reset();
  }};
  HYLE_RETURN_IF_ERROR(prepare_submission_lifetime(*state, state->allocator));
  auto &sources = *state->input_sources.get();
  struct Previous final { Binding *binding = nullptr; InputSource source{}; };
  container::SmallVector<Previous, 4, mem::AllocatorRef> previous(state->allocator);
  if (!previous.try_resize(incoming.size()))
    return Status::ResourceExhausted("allocating immutable input bundle rollback owners");
  // All metadata allocation precedes the temporary immutable markers. Keep
  // every previous owner until all original recording invalidations finish.
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    auto *binding = find_binding(sources, incoming[i].value);
    HYLE_FATAL_ASSERT(binding != nullptr);
    previous[i] = {binding, std::move(binding->source)};
    binding->source = incoming[i].source;
  }
  for (const auto &item : incoming) {
    auto status = validate_read_binding(*state, item.value, item.tensor, incoming);
    if (!status.ok()) {
      for (auto &old : previous) old.binding->source = std::move(old.source);
      return status;
    }
  }
  // No fallible operation remains. The original raw hook clears each marker
  // while invalidating old native imports; reinstall its already-retained
  // incoming source immediately, without allocation or producer observation.
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    const auto &item = incoming[i];
    commit_read_binding(*state, item.value, item.tensor);
    previous[i].binding->source = item.source;
  }
  invalidate(sources);
  restore_new_owner.dismiss();
  return Status::Ok();
}

void note_borrowed_input_source(InvocationState &state, ValueId value) noexcept {
  if (!state.input_sources) return;
  auto &sources = *state.input_sources.get();
  if (auto *binding = find_binding(sources, value)) binding->source.reset();
  invalidate(sources);
}
void reset_input_source_preparation(InvocationState &state) noexcept {
  if (!state.input_sources) return;
  auto &sources = *state.input_sources.get();
  invalidate(sources);
  sources.roots.clear(); sources.consumers.clear();
}
void reset_input_source_bindings(InvocationState &state) noexcept {
  reset_input_source_preparation(state);
  if (!state.input_sources) return;
  for (auto &binding : state.input_sources.get()->bindings) binding.source.reset();
}
bool has_input_source(const InvocationState &state, ValueId value) noexcept {
  if (!state.input_sources) return false;
  const auto *binding = find_binding(*state.input_sources.get(), value);
  return binding && binding->source.valid();
}
bool input_source_matches(const InvocationState &state, ValueId value,
    const InputSource &source) noexcept {
  if (!state.input_sources) return false;
  const auto *binding = find_binding(*state.input_sources.get(), value);
  return binding && binding->source.valid() && binding->source.same_owner(source);
}

Status prepare_input_sources(InvocationState &state,
    std::span<const QueueBinding> queues, HostWaitPolicy policy) noexcept {
  // Queue validation and retained candidate identity belong to the original
  // replay transaction. Never re-admit a Queue while describing its source.
  (void)queues;
  if (!state.input_sources) return Status::Ok();
  auto &sources = *state.input_sources.get();
  auto &candidate = sources.candidate;
  auto &roots = sources.roots;
  auto &consumers = sources.consumers;
  candidate.clear();
  roots.clear();
  consumers.clear();
  struct ClearScratch final {
    Vector<RootSource> &roots;
    Vector<Consumer> &consumers;
    ~ClearScratch() { roots.clear(); consumers.clear(); }
  } clear_scratch{roots, consumers};
  const auto &plan = *state.plan;
  try {
    for (const auto &binding : state.input_sources.get()->bindings)
      if (binding.source.valid()) candidate.sources.push_back(binding.source);
    std::sort(candidate.sources.begin(), candidate.sources.end(), source_less);
    candidate.sources.erase(std::unique(candidate.sources.begin(), candidate.sources.end(),
        [](const InputSource &a, const InputSource &b) { return a.same_owner(b); }),
        candidate.sources.end());
    candidate.ready.resize(candidate.sources.size(), 0);
    candidate.ranges.resize(plan.segments.size());
    roots.reserve(state.input_sources.get()->bindings.size());
    for (const auto &binding : state.input_sources.get()->bindings) {
      if (!binding.source.valid()) continue;
      const auto source = static_cast<std::size_t>(std::lower_bound(
          candidate.sources.begin(), candidate.sources.end(), binding.source, source_less) -
          candidate.sources.begin());
      roots.push_back({root_instance(plan, plan.external_instances[binding.value.ordinal()]), source});
    }
    std::sort(roots.begin(), roots.end(), [](const RootSource &a, const RootSource &b) {
      return std::tie(a.root, a.source) < std::tie(b.root, b.source);
    });
    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
      const auto &step = plan.steps[index];
      if (!step.emits_command()) continue;
      const auto segment = plan.step_segments[index];
      if (segment >= plan.segments.size())
        return Status::Internal("immutable input reader has no graph segment");
      const auto lane = plan.segments[segment].lane;
      auto group = lane.ordinal();
      if (shared_lane(plan, lane)) {
        const auto &queue = state.candidate_queues[lane.ordinal()];
        if (!queue.valid()) return Status::Internal("immutable input reader has no Queue");
        for (std::uint32_t earlier = 0; earlier < lane.ordinal(); ++earlier)
          if (queue.same_owner(state.candidate_queues[earlier])) { group = earlier; break; }
      }
      for (std::size_t port = 0; port < step.input_count; ++port) {
        const auto root = root_instance(plan, plan.node_inputs[step.input_begin + port]);
        auto found = std::lower_bound(roots.begin(), roots.end(), root,
            [](const RootSource &entry, std::uint32_t wanted) { return entry.root < wanted; });
        for (; found != roots.end() && found->root == root; ++found)
          consumers.push_back({segment, found->source, group});
      }
    }
    // A published input or required metadata alias is observed directly by
    // the caller. Its producer must be terminal before the original Graph
    // publishes success even when no mathematical segment reads this source.
    // Optional arithmetic aliases retain their ordinary execution dependency.
    for (const auto output : plan.output_ids) {
      if ((plan.binding_roles[output.ordinal()] &
           (kExternalInput | kSourceAliasedOutput)) == 0) continue;
      const auto root = root_instance(plan, plan.external_instances[output.ordinal()]);
      auto found = std::lower_bound(roots.begin(), roots.end(), root,
          [](const RootSource &entry, std::uint32_t wanted) { return entry.root < wanted; });
      for (; found != roots.end() && found->root == root; ++found)
        candidate.results.push_back(found->source);
    }
    std::sort(candidate.results.begin(), candidate.results.end());
    candidate.results.erase(std::unique(candidate.results.begin(), candidate.results.end()),
                            candidate.results.end());
    if (!candidate.results.empty() && policy == HostWaitPolicy::kReject)
      return Status::Unimplemented("published immutable input requires a forbidden host wait");
    candidate.report.host_wait_count += candidate.results.size();
    std::sort(consumers.begin(), consumers.end(), [](const Consumer &a, const Consumer &b) {
      return std::tie(a.source, a.group, a.segment) < std::tie(b.source, b.group, b.segment);
    });
    consumers.erase(std::unique(consumers.begin(), consumers.end(),
        [](const Consumer &a, const Consumer &b) {
          return a.source == b.source && a.group == b.group && a.segment == b.segment;
        }), consumers.end());
    for (const auto &consumer : consumers) {
      if (candidate.orders.empty() || candidate.orders.back().source != consumer.source ||
          candidate.orders.back().group != consumer.group) {
        const auto lane = plan.segments[consumer.segment].lane;
        const auto *queue = shared_lane(plan, lane)
            ? &state.candidate_queues[lane.ordinal()] : nullptr;
        const auto &source = candidate.sources[consumer.source];
        HYLE_ASSIGN_OR_RETURN(auto route, source.vtable()->route(source.state(), queue));
        if (route != InputDependencyRoute::kHostWait &&
            route != InputDependencyRoute::kDeviceWait)
          return Status::Internal("immutable input source returned an invalid route");
        if (!queue && route != InputDependencyRoute::kHostWait)
          return Status::Unimplemented("immutable input source has no delegated import adapter");
        if (route == InputDependencyRoute::kHostWait && policy == HostWaitPolicy::kReject)
          return Status::Unimplemented("immutable input source requires a forbidden host wait");
        candidate.orders.push_back({consumer.source, consumer.group, route, false, false});
        if (route == InputDependencyRoute::kDeviceWait) ++candidate.report.device_wait_count;
        else ++candidate.report.host_wait_count;
      }
      candidate.uses.push_back({consumer.segment, candidate.orders.size() - 1});
    }
    std::sort(candidate.uses.begin(), candidate.uses.end(), [](const Use &a, const Use &b) {
      return std::tie(a.segment, a.order) < std::tie(b.segment, b.order);
    });
    for (std::size_t index = 0; index < candidate.uses.size(); ++index) {
      auto &range = candidate.ranges[candidate.uses[index].segment];
      if (range.count == 0) range.begin = index;
      ++range.count;
    }
    candidate.report.source_count = candidate.sources.size();
    candidate.valid = true;
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    candidate.clear();
    return Status::ResourceExhausted("preparing graph immutable input dependencies");
  }
}

void commit_input_sources(InvocationState &state) noexcept {
  if (!state.input_sources) return;
  auto &sources = *state.input_sources.get();
  HYLE_FATAL_ASSERT(sources.candidate.valid);
  sources.active.swap(sources.candidate);
  sources.candidate.clear();
}
void begin_input_sources(InvocationState &state) noexcept {
  if (!state.input_sources) return;
  auto &active = state.input_sources.get()->active;
  std::fill(active.ready.begin(), active.ready.end(), static_cast<unsigned char>(0));
  for (auto &order : active.orders) { order.ordered = false; order.used = false; }
}
Result<bool> input_sources_ready(InvocationState &state, std::size_t segment) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *active, prepared_for(state, segment));
  if (!active) return true;
  const auto range = active->ranges[segment];
  for (std::size_t index = range.begin; index < range.begin + range.count; ++index) {
    const auto &order = active->orders[active->uses[index].order];
    if (order.route != InputDependencyRoute::kHostWait || active->ready[order.source]) continue;
    const auto &source = active->sources[order.source];
    HYLE_ASSIGN_OR_RETURN(auto ready, source.vtable()->poll(source.state()));
    if (ready == compute::CompletionPoll::kPending) return false;
    active->ready[order.source] = 1;
  }
  return true;
}
Status wait_input_sources(InvocationState &state, std::size_t segment) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *active, prepared_for(state, segment));
  if (!active) return Status::Ok();
  const auto range = active->ranges[segment];
  for (std::size_t index = range.begin; index < range.begin + range.count; ++index) {
    const auto &order = active->orders[active->uses[index].order];
    if (order.route != InputDependencyRoute::kHostWait || active->ready[order.source]) continue;
    const auto &source = active->sources[order.source];
    HYLE_RETURN_IF_ERROR(source.vtable()->wait(source.state()));
    active->ready[order.source] = 1;
  }
  return Status::Ok();
}
Status enqueue_input_sources(InvocationState &state, std::size_t segment,
    compute::detail::QueueSequence &sequence) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *active, prepared_for(state, segment));
  if (!active) return Status::Ok();
  const auto range = active->ranges[segment];
  for (std::size_t index = range.begin; index < range.begin + range.count; ++index) {
    auto &order = active->orders[active->uses[index].order];
    if (order.route != InputDependencyRoute::kDeviceWait || order.ordered) continue;
    const auto &source = active->sources[order.source];
    auto result = source.vtable()->enqueue(source.state(), sequence);
    const auto status = result.status;
    const bool accepted = result.accepted;
    // Accepted errors still own native work. Never lose their actual prefix
    // or retry that work via a different dependency route.
    sequence.record(std::move(result));
    if (!status.ok()) return status;
    order.ordered = true;
    order.used = accepted;
  }
  return Status::Ok();
}
Result<bool> input_results_ready(InvocationState &state) noexcept {
  if (!state.input_sources) return true;
  auto &active = state.input_sources.get()->active;
  if (!active.valid)
    return std::unexpected(Status::FailedPrecondition("graph input sources are not prepared"));
  for (const auto index : active.results) {
    if (active.ready[index]) continue;
    const auto &source = active.sources[index];
    HYLE_ASSIGN_OR_RETURN(auto ready, source.vtable()->poll(source.state()));
    if (ready == compute::CompletionPoll::kPending) return false;
    active.ready[index] = 1;
  }
  return true;
}
Status wait_input_results(InvocationState &state) noexcept {
  if (!state.input_sources) return Status::Ok();
  auto &active = state.input_sources.get()->active;
  if (!active.valid)
    return Status::FailedPrecondition("graph input sources are not prepared");
  for (const auto index : active.results) {
    if (active.ready[index]) continue;
    const auto &source = active.sources[index];
    HYLE_RETURN_IF_ERROR(source.vtable()->wait(source.state()));
    active.ready[index] = 1;
  }
  return Status::Ok();
}
Status reconcile_input_sources(InvocationState &state, const Status &terminal) noexcept {
  if (!terminal.ok() || !state.input_sources) return Status::Ok();
  auto &active = state.input_sources.get()->active;
  // All accepted guards have retired and the progress owner has returned its
  // sentinel. No worker can still mutate these original per-run facts.
  for (const auto &order : active.orders)
    if (order.used) active.ready[order.source] |= 2;
  Status result{};
  for (std::size_t index = 0; index < active.sources.size(); ++index) {
    if ((active.ready[index] & 2) == 0) continue;
    const auto &source = active.sources[index];
    const auto checked = source.vtable()->consumer_retired(source.state());
    if (result.ok() && !checked.ok()) result = checked;
  }
  return result;
}
} // namespace hyle::graph::detail

namespace hyle::graph {
Result<InputSourceReport> Invocation::input_source_report() const noexcept {
  const auto *state = detail::InvocationAccess::state(*this);
  if (!state || !state->replay_prepared)
    return std::unexpected(Status::FailedPrecondition("graph replay is not prepared"));
  if (!state->input_sources) return InputSourceReport{};
  const auto &active = state->input_sources.get()->active;
  if (!active.valid)
    return std::unexpected(Status::FailedPrecondition("graph input sources are not prepared"));
  return active.report;
}
} // namespace hyle::graph
