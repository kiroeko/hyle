#pragma once

// Qwen3.5 model implementation for the Hyle inference executable. All model
// mathematics uses Compute/Graph; native execution remains backend-owned.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "session_state.h"
#include "host_image.h"
#include "../../inference/host_state_storage.h"
#include "../../inference/state_transfer.h"
#include "mtp_graph.h"
#include "block_graph.h"
#include "../../inference/mtp_acceptance.h"
#include "../../inference/mtp_acceptance_graph.h"
#include "../../inference/sampling_graph.h"
#include "../../inference/artifact_digest.h"
#include "../../inference/logits_inputs.h"
#include "../../inference/execution_resources.h"
#include "../../inference/program_carrier.h"
#include "../../inference/program_slots.h"
#include "../../inference/execution_episode.h"
#include "../../inference/program_io.h"
#include "../../inference/output_pool_capacity.h"
#include "../../inference/graph_bindings.h"
#include "../../inference/prepared_program_family.h"
#include "../../inference/cold_shape_plan.h"
#include "../../inference/cold_demand.h"
#include "../../inference/slot_transactions.h"
#include "../../inference/append_transactions.h"
#include "../../inference/paged_append_capacity.h"
#include "../../inference/retirement_observation.h"
#include "../../inference/evaluation_failure.h"
#include "../../inference/request_execution_failure.h"
#include "../../inference/prepared_output.h"
#include "../../inference/commit_observation.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/request_window.h"
#include "../../inference/packed_rows.h"
#include "../../inference/paged_append.h"
#include "../../inference/execution_observation.h"
#include "generation_tokens.h"
#include "state_boundary.h"
#include "publication_facts.h"
#include "prepared_roles.h"
#include "ffn_placement.h"
#include "projection_binding.h"
#include "weights.h"
#include "weight_inventory.h"
#include "mtp_capabilities.h"
#include "gguf_capabilities.h"
#include "graph_builder.h"
#include "target_graph.h"
#include "family_factory.h"
#include "phase.h"
#include "prefill_inputs.h"
#include "auxiliary_programs.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/detail/batch_device_lengths.h"
#include "hyle/compute/detail/device_lengths_paged_append_backend.h"
#include "hyle/core/assert.h"
#include "precision_profile.h"
#include "cache_precision.h"
#include "hyle/container/small_vector.h"

#include "../../inference/backend.h"
#include "../../inference/readback.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/sequence_ops.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/slots.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/source.h"
#include "hyle/graph/paged_append_source.h"
#include "hyle/mem/buffer.h"
#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/sched/join.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"
#include "hyle/util/scope_guard.h"

namespace qwen35 {
namespace sampling = hyle::inference::sampling;
namespace inference = hyle::inference;
using inference::Backend;
using inference::BackendKind;
using inference::BackendSelection;
using inference::PreparedReadback;
using inference::PreparedOutputFacts;
enum class ReplayPolicy { kAuto, kCommands, kPreferCapture };
constexpr g::ReplayMode select_replay_mode(ReplayPolicy policy,
    std::size_t known_uses) noexcept {
  return policy == ReplayPolicy::kCommands
      || (policy == ReplayPolicy::kAuto && known_uses <= 1)
      ? g::ReplayMode::kCommands : g::ReplayMode::kPreferCapture;
}
inline constexpr std::array<const char *, 9> layer0_probe_names{
    "01-gdn-output", "02-out-proj-input", "03-out-proj-output",
    "04-down-proj-input", "05-layer-output", "06-post-attn-residual",
    "07-post-attn-norm", "08-gate-proj-output", "09-up-proj-output"};

inline std::vector<StateSchema> state_schema_for(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision = {}) {
  geometry.validate();
  if (capacity == 0 || capacity > static_cast<std::size_t>(geometry.context_limit)) throw std::runtime_error("capacity outside model context limit");
  std::vector<StateSchema> result(geometry.layers.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto &layer = result[i];
    if (geometry.is_gdn(i)) {
      layer.history = dense(precision.activation_type(), {1, geometry.conv_kernel - 1, geometry.gdn_qkv_width()});
      layer.recurrent = dense(Type::kFloat32, {1, geometry.gdn_value_heads, geometry.gdn_value_depth, geometry.gdn_key_depth});
    }
  }
  return result;
}
inline std::size_t recurrent_bytes(std::span<const StateSchema> schema, const ModelGeometry &geometry) {
  std::size_t bytes = 0;
  const auto add = [&](const c::TensorDesc &description, std::size_t copies) {
    const auto size = take(description.required_bytes());
    if (size > ((std::numeric_limits<std::size_t>::max)() - bytes) / copies)
      throw std::runtime_error("session cache byte count overflows");
    bytes += size * copies;
  };
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (geometry.is_gdn(i)) { add(schema[i].history, 1); add(schema[i].recurrent, 1); }
  }
  return bytes;
}
inline void validate_mtp_source(const ModelGeometry &geometry, bool enabled) {
  if (enabled && geometry.mtp_layers != 1)
    throw std::runtime_error("MTP requires one NextN layer in the prepared source geometry");
}
inline BackendSelection select_execution(BackendSelection selection, std::size_t cpu_layers) {
  if (selection.primary == BackendKind::kDefault)
    selection.primary = take(Backend::default_selection()).primary;
  if (std::ranges::find(Backend::compiled_backends(), selection.primary) == Backend::compiled_backends().end())
    throw std::runtime_error("selected backend is not compiled into this image");
  selection.cpu_assist = cpu_layers != 0;
  if (selection.cpu_assist && (selection.primary == BackendKind::kCpu ||
      std::ranges::find(Backend::compiled_backends(), BackendKind::kCpu) == Backend::compiled_backends().end()))
    throw std::runtime_error("--cpu-ffn-layers requires a GPU primary and the compiled CPU route");
  return selection;
}
inline Backend create_execution(const ModelGeometry &geometry, const cache::Plan &plan,
    std::size_t cpu_layers, BackendSelection selection,
    const c::cpu::DefaultContextOptions *cpu_options = nullptr) {
  geometry.validate();
  validate_mtp_source(geometry, plan.draft_cache);
  FfnPlacement::validate(cpu_layers, geometry.layers.size());
  selection = select_execution(selection, cpu_layers);
  if (selection.primary != BackendKind::kCpu && !cpu_layers) {
    if (cpu_options) return take(Backend::create(selection, {}, *cpu_options));
    return take(Backend::create(selection));
  }
  // Conservative finite admission for this graph's known workspace only.
  // Sum model-sized activation envelopes instead of assuming workspace reuse;
  // unknown native-private storage remains explicitly outside this bound.
  const auto rows = static_cast<std::size_t>(plan.max_query_rows);
  const auto activation_elements = cache::add(cache::multiply(geometry.layers.size(),
      cache::add(cache::multiply(16, geometry.intermediate), cache::multiply(32, geometry.hidden))), cache::multiply(2, geometry.vocabulary));
  const auto compute = cache::add(64u << 20, cache::multiply(cache::multiply(rows, activation_elements), sizeof(float)));
  if (selection.primary == BackendKind::kCpu) {
    std::cerr << "cpu_workspace_admission host=" << compute
        << " scope=known_per_invocation_workspace native_private_unknown=1 physical_memory_cap=0\n";
    // CPU staging aliases its host allocation domain. Give that domain one
    // finite limit; do not count the same memory as a second staging budget.
    if (cpu_options) return take(Backend::create(selection, {.host = compute}, *cpu_options));
    return take(Backend::create(selection, {.host = compute}));
  }
  const auto staging = cache::add(64u << 20,
      cache::multiply(cache::multiply(rows, cache::multiply(static_cast<std::size_t>(geometry.hidden), 8)), cpu_layers));
  std::cerr << "hybrid_workspace_admission device=" << compute << " host=" << compute
      << " staging=" << staging << " scope=known_per_invocation_workspace native_private_unknown=1 physical_vram_cap=0\n";
  if (cpu_options) return take(Backend::create(selection, {compute, compute, staging}, *cpu_options));
  return take(Backend::create(selection, {compute, compute, staging}));
}
inline c::BatchCapacity attention_capacity(const ModelGeometry &geometry, PrecisionProfile precision, const cache::Plan &plan) {
  const auto storage = take(c::CacheStorage::plain(precision.cache_type()));
  return {.storage = c::BatchStorage::kPaged, .query_type = precision.cache_type(),
      .output_type = precision.cache_type(), .key_storage = storage, .value_storage = storage,
      .max_requests = plan.max_requests, .max_query_rows = plan.max_query_rows,
      .max_kv_rows = plan.max_kv_rows, .max_pages = plan.max_pages, .max_page_refs = plan.max_page_refs,
      .query_heads = static_cast<std::uint32_t>(geometry.query_heads),
      .kv_heads = static_cast<std::uint32_t>(geometry.kv_heads),
      .qk_head_dim = static_cast<std::uint32_t>(geometry.key_depth),
      .value_head_dim = static_cast<std::uint32_t>(geometry.value_depth), .page_size = cache::Plan::page_rows};
}
inline bool supports_mtp_attention(Backend &execution, c::BatchCapacity capacity,
    std::span<const CacheStoragePair> target_storage, const c::ResourceBudget &budget = {}) {
  if (!take(execution.supports_device_lengths_batch(capacity, budget))) return false;
  for (const auto &storage : target_storage) {
    auto target = capacity;
    target.key_storage = storage[0]; target.value_storage = storage[1];
    if (!take(execution.supports_device_lengths_batch(target, budget))) return false;
  }
  return true;
}
// Typed ready inputs shared by an episode or prepared for each conditional.
// Bias/mask are immutable [1,V] snapshots; counts is C0 and mathematical
// prefix updates create later exact I32 counts. Copies retain
// the original read permission. Use a lease carrying its existing StorageOwner,
// or keep its borrowed allocator/resource/Context alive through every retained
// copy and native use. Owners may supply different leases next episode; no
// modifier payload enters a Snapshot or KV cache; the ledger holds only its
// own budgeted derived host counts, never external bias/mask payloads.
using inference::LogitsInput;
using inference::LogitsInputs;
struct PreparedSampling;
struct ColdResources;
inline void validate_sampling_configuration(const SamplingConfiguration &value, std::int64_t vocabulary) {
  if (vocabulary <= 0 || vocabulary > INT32_MAX || !std::isfinite(value.policy.temperature) || value.policy.temperature <= 0 ||
      !std::isfinite(value.policy.top_p) || value.policy.top_p <= 0 || value.policy.top_p > 1 ||
      value.policy.top_k > static_cast<std::size_t>(vocabulary) ||
      (!value.enabled && (value.policy.temperature != 1 || value.policy.top_k || value.policy.top_p != 1)) ||
      !sampling::valid_history_policy(value.modifiers.history))
    throw std::runtime_error("invalid finite sampling configuration for vocabulary");
}
inline bool same_sampling_configuration(const SamplingConfiguration &a, const SamplingConfiguration &b) noexcept {
  return a.enabled == b.enabled && a.policy.temperature == b.policy.temperature &&
      a.policy.top_k == b.policy.top_k && a.policy.top_p == b.policy.top_p && a.modifiers == b.modifiers;
}

// Only deterministic port/descriptor facts: no payload reads, lease admission
// or numeric distribution checks. Those remain with their original owners.
inline hyle::Status logits_inputs_metadata_status(const SamplingConfiguration &configuration,
    const LogitsInputs &inputs, std::int64_t vocabulary) noexcept {
  const auto require = [&](const LogitsInput &input, bool enabled, Type type) -> hyle::Status {
    if (input.lease.valid() != enabled || input.description.valid() != enabled)
      return hyle::Status::InvalidArgument("logits input does not match its cold port configuration");
    if (enabled) {
      const std::array<std::int64_t, 2> shape{1, vocabulary};
      HYLE_ASSIGN_OR_RETURN(const auto expected, c::TensorDesc::dense(type, shape));
      if (input.description != expected)
        return hyle::Status::InvalidArgument("logits input requires the exact cold dense [1,V] descriptor");
    }
    return hyle::Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(require(inputs.bias, configuration.modifiers.bias, Type::kFloat32));
  HYLE_RETURN_IF_ERROR(require(inputs.allowed, configuration.modifiers.allowed, Type::kBool));
  return require(inputs.counts, !configuration.modifiers.history.empty(), Type::kInt32);
}
inline void validate_logits_input_leases(const SamplingConfiguration &configuration, const LogitsInputs &inputs) {
  if (configuration.modifiers.bias) (void)take(inputs.bias.lease.tensor(inputs.bias.description));
  if (configuration.modifiers.allowed) (void)take(inputs.allowed.lease.tensor(inputs.allowed.description));
  if (!configuration.modifiers.history.empty()) (void)take(inputs.counts.lease.tensor(inputs.counts.description));
}
inline void validate_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
                            std::int64_t vocabulary) {
  check(logits_inputs_metadata_status(configuration, inputs, vocabulary));
  validate_logits_input_leases(configuration, inputs);
}
// The base retains the one C0 history. Conditional rows supply only immutable
// bias/mask snapshots. Descriptor errors can retain the original request key;
// lease permission and native checks remain with their original mechanisms.
inline hyle::Status prefix_logits_inputs_metadata_status(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional, std::size_t rows, std::int64_t vocabulary) noexcept {
  if (conditional.empty()) return logits_inputs_metadata_status(configuration, inputs, vocabulary);
  if (conditional.size() != rows || (!configuration.modifiers.bias && !configuration.modifiers.allowed))
    return hyle::Status::InvalidArgument("conditional logits require one configured bias/mask row per actual target including bonus");
  auto base = configuration, per_row = configuration;
  base.modifiers.bias = false; base.modifiers.allowed = false;
  per_row.modifiers.history = {};
  HYLE_RETURN_IF_ERROR(logits_inputs_metadata_status(base, inputs, vocabulary));
  for (const auto &row : conditional) HYLE_RETURN_IF_ERROR(logits_inputs_metadata_status(per_row, row, vocabulary));
  return hyle::Status::Ok();
}
inline void validate_prefix_logits_input_leases(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional) {
  if (conditional.empty()) { validate_logits_input_leases(configuration, inputs); return; }
  auto base = configuration, per_row = configuration;
  base.modifiers.bias = false; base.modifiers.allowed = false; per_row.modifiers.history = {};
  validate_logits_input_leases(base, inputs);
  for (const auto &row : conditional) validate_logits_input_leases(per_row, row);
}
inline void validate_prefix_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::span<const LogitsInputs> conditional, std::size_t rows, std::int64_t vocabulary) {
  check(prefix_logits_inputs_metadata_status(configuration, inputs, conditional, rows, vocabulary));
  validate_prefix_logits_input_leases(configuration, inputs, conditional);
}
// A registered request has its own complete key before any bucket or native
// episode exists. Do not manufacture EvaluationFailure.participant from it.
inline void check_request_logits_metadata(hyle::Status status, inference::RequestKey key) {
  if (status.IsInvalidArgument() && !status.has_native_error() && key.generation &&
      key.slot != (std::numeric_limits<std::size_t>::max)())
    throw inference::RequestExecutionFailure::from_preflight(std::move(status), key);
  check(std::move(status)); // Unregistered/raw consumers retain their original error path.
}
inline void validate_request_logits_inputs(const SamplingConfiguration &configuration, const LogitsInputs &inputs,
    std::int64_t vocabulary, inference::RequestKey key) {
  check_request_logits_metadata(logits_inputs_metadata_status(configuration, inputs, vocabulary), key);
  validate_logits_input_leases(configuration, inputs);
}
inline GenerationBoundary generation_boundary(const cache::State &head) noexcept {
  return {head.length, head.draft ? head.draft->frontier : 0,
      head.draft ? head.draft->lag : 0, head.next_id,
      head.draft ? head.draft->catchup_token : -1, head.sampling};
}
struct FamilyKey final {
  std::size_t requests, verification_rows;
  GraphRole role;
  TextCompletion completion;
  std::size_t sampling_index;
  RowPacking packing = RowPacking::kRectangular;
  FamilyKey(std::size_t count, GraphRole selected = GraphRole::kText,
            std::size_t rows = 0, TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0, RowPacking packing = RowPacking::kRectangular) noexcept
      : requests(count), verification_rows(selected == GraphRole::kVerify ? rows : 0), role(selected), completion(mode), sampling_index(sampling), packing(packing) {}
  static FamilyKey exact(std::size_t count, GraphRole selected, std::size_t rows,
                         TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0) noexcept {
    FamilyKey key(count, selected, 0, mode, sampling); key.verification_rows = rows; return key;
  }
  auto operator<=>(const FamilyKey &) const = default;
};
struct ExecutionShape final {
  std::size_t requests, tokens;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t sampling_index = 0;
  bool imported_sources = false;
  RowPacking packing = RowPacking::kRectangular;
  auto operator<=>(const ExecutionShape &) const = default;
};
struct FamilyChild final {
  Family &family;
  const inference::PreparedProgram &program;
};
// One original Graph episode owns all dirty pages and candidate state. Until
// finish replaces the Session head, neither a ready layer nor an error changes
// the committed snapshot. Failure recovery remains attached to that Session.
// One host-only upload workspace for the original serial Model carrier.
// Its integer ranges never escape synchronous Tensor::write; retained model
// state, page permissions and submitted resources keep their existing owners.
struct InputPreparationScratch final {
  std::vector<std::int32_t> packed_ids, positions, draft_positions, catchups;
  std::vector<inference::PackedRowInput> row_inputs;
  std::vector<std::int32_t> query_indptr, last_rows;
  std::vector<std::uint32_t> sampling_identities;
  // Shared preparation reuses cold-bounded integers and temporary footprints.
  // Its source pins never survive a call; Graph retains its original leases.
  inference::PagedAppendScratch paged_appends;
  inference::PagedAppendCapacityScratch paged_append_capacity;
  void prepare(const Phase &phase) {
    const auto &io = phase.slots[0].data; // All slots share the same cold descriptions.
    const auto reserve = [](auto &values, const c::TensorDesc &description) {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      values.reserve(take(description.required_bytes()) / sizeof(Element));
    };
    row_inputs.reserve(phase.family->requests);
    query_indptr.reserve(phase.family->requests + 1);
    last_rows.reserve(phase.family->requests);
    if (phase.family->requests > 1 || packs_target(phase.family->packing))
      reserve(packed_ids, io.ordinary_io.packet() ? io.input_packet.tokens().description() : io.tokens.description);
    reserve(positions, io.ordinary_io.packet() ? io.input_packet.positions().description() : io.positions.description);
    if (io.ordinary_io.packet_sampling_identities)
      reserve(sampling_identities, io.input_packet.sampling_identities().description());
    else if (io.sampling_identities.caller_owned()) reserve(sampling_identities, io.sampling_identities.description);
    if (io.draft_positions.caller_owned()) reserve(draft_positions, io.draft_positions.description);
    if (io.catchup_token.caller_owned()) reserve(catchups, io.catchup_token.description);
  }
};


class Model;
// Admission borrows the live Model. The cold resource owner and an explicit
// original retirement continuation survive host Model teardown when needed.
class Session final : public std::enable_shared_from_this<Session> {
  friend class Model;
  Model *model_;
  std::shared_ptr<ColdResources> resources_;
  explicit Session(Model &model, std::shared_ptr<ColdResources> resources, const PreparedSampling &sampling) noexcept
      : model_(&model), resources_(std::move(resources)), sampling_(sampling) {}
  const PreparedSampling &sampling_;
public:
  cache::WorkingState head;
  hyle::container::SmallVector<c::PagedPageAppender, 32> tails;
  std::array<c::PagedPageAppender, 2> draft_tails;
  std::uint64_t tail_epoch = 0, draft_tail_epoch = 0;
  // Monotonic accepted-Graph active input rows; never part of a Snapshot or RNG.
  std::size_t submitted_draft_rows = 0;
  std::size_t proposal_graph_calls = 0;
  StepState *pending = nullptr; // Borrowed from Model; episode pins this Session.
  bool failed = false, stopping = false;
  hyle::Status retirement_status;
  std::size_t length() const noexcept { return head ? head->length : 0; }
  cache::SnapshotRef checkpoint() {
    if (pending) throw std::runtime_error("checkpoint requires an idle committed model boundary");
    return head.checkpoint();
  }
  const PreparedSampling &sampling_configuration() const noexcept { return sampling_; }
  std::weak_ptr<ColdResources> sampling_owner() const noexcept { return resources_; }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  // After multi-request submission, stop withdraws only this member's
  // publication while a survivor remains, including MTP. Single/all-member and
  // pre-submit stops abort the whole episode. Cancellation is not retirement.
  hyle::Status stop() noexcept;
  inference::RetirementObservation try_retire() noexcept;
  inference::RetirementObservation drain() noexcept;
  ~Session();
};
inline bool StepState::withdrawn(const StepMember &member) const noexcept {
  return discard_stopped_members && member.session->stopping;
}
struct EvaluationRequest final {
  std::shared_ptr<Session> session;
  std::span<const std::int32_t> ids;
  const inference::RequestStopPolicy *stop_policy = nullptr;
  LogitsInputs logits{};
  inference::ConditionalLogitsInputs conditional_logits{};
  inference::RequestKey request_key{}; // Diagnostic identity only; not admission authority.
  PrefillInputPackets *prefill_packets = nullptr;
  std::span<const std::int32_t> prefill_prompt{};
  std::size_t prefill_offset = 0, prefill_turn = 0;
  // Original proposal submission, exporting only the complete token packet.
  // No target state or page publication is imported from the private draft.
  std::shared_ptr<StepSources> verification_tokens;
};

// Metadata only for the original cold Invocation. Graph enforces that these
// borrowed coordinates cannot be submitted or exported as an execution Source.
struct RecordingStepInputs final {
  Phase *phase = nullptr;
  std::size_t slot = 0;
  c::ConstTensorView token;
  c::ConstTensorView catchup;
  c::ConstTensorView prefix;
  std::span<const g::detail::RecordingStoreInputs> states;
};
struct DecodeRecording final {
  std::size_t preparations = 0, captures = 0;
  double seconds = 0, capture_preparation_seconds = 0;
};
enum class DecodeRecordingStage { kStartup, kRecovery };

// CPU topology only, never a candidate/head or permission. A weak identity and
// the next committed epoch qualify consumption after the original predecessor.
struct DecodeAheadMetadata final {
  template<class T, std::size_t N>
  using Values = hyle::container::SmallVector<T, N, mem::AllocatorRef>;
  struct Member final {
    std::weak_ptr<Session> session;
    std::uint64_t epoch = 0;
    std::size_t base_length = 0, maximum_length = 0, sampling_index = 0;
    bool ready = false, selected = false;
  };
  c::ResourceBudget budget; // Outlives its borrowed metadata allocators.
  Values<Member, 2> members;
  // One fixed stripe of original target-page IDs per member. Selecting a
  // prefix must not overwrite a still-unconsumed sibling's integers.
  Values<std::int32_t, 2> candidate_ids;
  Values<std::size_t, 2> selected;
  Values<std::int32_t, 2> positions, old_ids, old_indptr, old_last, old_query;
  std::size_t page_stride = 0, maximum_rows = 1;
  bool ready = false;
  DecodeAheadMetadata(c::ResourceBudget owner, mem::AllocatorRef allocator)
      : budget(std::move(owner)), members(allocator), candidate_ids(allocator), selected(allocator),
        positions(allocator), old_ids(allocator), old_indptr(allocator), old_last(allocator), old_query(allocator) {}
  void clear_output() noexcept {
    for (const auto index : selected) members[index].selected = false;
    selected.clear(); positions.clear(); old_ids.clear(); old_indptr.clear(); old_last.clear(); old_query.clear();
  }
  void refresh() noexcept {
    ready = std::ranges::any_of(members, [](const auto &member) { return member.ready; });
  }
  void clear() noexcept {
    ready = false;
    for (auto &member : members) member = {};
    clear_output();
  }
};
enum class EvaluationDisposition : std::uint8_t { kNotPublished, kPublished, kCancelled };
struct EvaluationOutput final {
  std::shared_ptr<Session> session;
  bool capture_tensors = false, capture_probes = false;
  double *complete_seconds = nullptr;
  Clock::time_point *host_endpoint = nullptr;
  inference::StepObservation *observation = nullptr;
  inference::PreparationObservation *preparation = nullptr;
  // Optional actual publication fact. A stopped member's raw next_ids element
  // is not a delivered result; kCancelled has no observer/head/ledger commit.
  EvaluationDisposition *disposition = nullptr;
};
// Borrowed only during the synchronous post-commit observer. Original output
// leases and the carrier admission guard remain in finish's stack scope.
struct CompletedEvaluationView final {
  inference::StepObservation &observation;
  const cache::State &head;
  std::size_t member_index = 0, requests = 1;
  // Packed Verify exposes compact input rows; other outputs stay request-major.
  std::size_t packed_input_begin = 0, packed_input_rows = 0;
  c::ConstTensorView logits{}, hidden{};
  std::span<const c::ConstTensorView> probes{};
};
// Entry-owned naming and IO policy. The model receives only capture demands.
struct EvaluationIO final {
  std::string_view name;
  const std::filesystem::path *trace = nullptr;
  bool trace_states = true;
  std::string_view probe_subdirectory{};
};
struct MtpRoundResult final {
  std::int32_t pending = -1, draft = -1, target_first = -1, target_second = -1, next_id = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  std::size_t processed_target = 0, processed_draft = 0;
  std::size_t proposal_count = 0, consumed_drafts = 0, verified_rows = 0;
  std::size_t auxiliary_graph_calls = 0;
  std::vector<std::int32_t> emitted;
  void reset() noexcept {
    auto storage = std::move(emitted);
    *this = MtpRoundResult{};
    emitted = std::move(storage); emitted.clear();
  }
};
struct MtpGenerationSelection final {
  std::shared_ptr<Session> session;
  std::int32_t expected_pending = -1;
  std::size_t output_quota = 0;
  std::span<const c::ReadLease> draft_log_probabilities{};
  std::size_t proposal_limit = 0;
};

// Finite NextN-only programs share the original model weights and execution
// endpoint. Their speculative pages never become a Session head. Target
// verification repairs the retained draft prefix with actual target hidden
// rows before the existing Snapshot transaction can publish it.

struct DraftWork final {
  cache::DraftState state;
  std::array<c::PagedPageAppender, 2> tails;
  // Optional attribution to the original Session pinned by this synchronous
  // proposal/repair call. Standalone private work still counts on the Model.
  Session *session = nullptr;
};
struct DraftProposal final {
  std::int32_t id = -1;
  c::ReadLease hidden, log_probabilities, next_counts;
};
// Call-local linear proposal resolution. Row zero was resolved before request
// admission; later rows are retained once and frozen for the original verifier.
struct ConditionalLogitsProposal final {
  const inference::LogitsInputProvider &provider;
  std::span<const std::int32_t> committed_outputs;
  std::vector<LogitsInputs> rows;
};


struct PrefixSamplingResult final {
  std::vector<std::int32_t> final_ids;
  inference::mtp::AcceptedPrefix prefix;
};





// Actual cold storage and policy addresses borrowed by submitted Graphs. The
// application Model and Session links are absent: a failed native prefix must
// never retain or later dereference a destroyed host application object.
struct ColdResources final {
  // The Model hands off its original endpoint on teardown, including normal
  // retirement. Destroy payloads before this resource controller's final owner.
  std::optional<Backend> retirement_backend;
  std::vector<c::LeasePool> weight_pools;
  std::map<std::string, LoadedWeight> weights;
  ProjectionBindings projections;
  std::optional<inference::DraftVocabulary> draft_vocabulary;
  std::string draft_vocabulary_identity;
  std::optional<PreparedDraftHead> draft_head;
  double draft_head_prepare_seconds=0;
  WeightMeanings weight_meanings;
  std::string source_identity;
  std::vector<StateSchema> state_schema;
  const prepared::CachePolicy cache_policy;
  const std::vector<CacheStoragePair> attention_storage;
  c::ResourceBudget cache_budget, execution_budget, allocation_budget;
  cache::VersionCount snapshot_versions;
  std::size_t carrier_borrowed = 0;
  hyle::inference::PagedPageInventory page_inventory, draft_inventory;
  std::vector<cache::StatePools> state_pools;
  c::LeasePool hidden_pool, history_counts_pool;
  c::LeaseSlotLayout hidden_layout;
  std::vector<std::unique_ptr<PreparedSampling>> sampling_configurations;
  const bool retain_sampling_frontier;
  std::map<FamilyKey, std::unique_ptr<Family>> families, probe_families;
  std::map<ExecutionShape, std::unique_ptr<Phase>> phases, probe_phases;
  std::map<DraftProgramKey, std::unique_ptr<DraftProgram>> draft_programs;
  std::map<PrefixSamplingKey, std::unique_ptr<PrefixSamplingProgram>> prefix_sampling_programs;

  explicit ColdResources(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision,
      SamplingConfiguration first, std::span<const SamplingConfiguration> additional,
      prepared::CachePolicy policy, std::vector<CacheStoragePair> storage)
      : state_schema(state_schema_for(geometry, capacity, precision)),
        cache_policy(policy), attention_storage(std::move(storage)), state_pools(geometry.layers.size()),
        retain_sampling_frontier(first.enabled || !first.modifiers.empty() || std::ranges::any_of(additional,
            [](const auto &value) { return value.enabled || !value.modifiers.empty(); })) {
    sampling_configurations.reserve(cache::add(additional.size(), 1));
    const auto add = [&](SamplingConfiguration value) {
      validate_sampling_configuration(value, geometry.vocabulary);
      for (const auto &old : sampling_configurations)
        if (same_sampling_configuration(old->configuration, value))
          throw std::runtime_error("duplicate cold sampling configuration");
      sampling_configurations.push_back(std::make_unique<PreparedSampling>(sampling_configurations.size(), value));
    };
    add(first); for (const auto &value : additional) add(value);
  }

  const PreparedSampling &sampling_configuration(std::size_t index) const {
    if (index >= sampling_configurations.size()) throw std::runtime_error("sampling configuration was not cold prepared");
    return *sampling_configurations[index];
  }
  std::size_t sampling_index(const PreparedSampling *value) const {
    if (!value) return 0;
    for (const auto &entry : sampling_configurations) if (entry.get() == value) return entry->index;
    throw std::runtime_error("sampling configuration belongs to another model");
  }
  bool pending_readbacks() const noexcept {
    for (const auto *cache : {&phases, &probe_phases})
      for (const auto &[shape, phase] : *cache) {
        (void)shape;
        for (const auto &slot : phase->slots.slots())
          if (slot.data.small_results.in_flight()) return true;
      }
    return false;
  }
  bool pending() const noexcept {
    if (pending_readbacks()) return true;
    for (const auto *cache : {&phases, &probe_phases})
      for (const auto &[shape, phase] : *cache) {
        (void)shape;
        if (phase->slots.pending()) return true;
      }
    for (const auto &[shape, program] : draft_programs) {
      (void)shape; if (program->carrier.pending()) return true;
    }
    for (const auto &[width, program] : prefix_sampling_programs) {
      (void)width; if (program->carrier.pending()) return true;
    }
    return std::ranges::any_of(sampling_configurations, [](const auto &entry) {
      return entry->frontier && entry->frontier->carrier.pending();
    });
  }
  void release_invocation_roots() noexcept {
    // Releasing even an idle Graph can destroy CUDA workspace/capture and
    // synchronize its Context. Preserve every root while an independent
    // result copy still reads that Context; its original retention owns us.
    if (pending_readbacks()) return;
    // The accepted attachment is the only temporary scope->Graph->scope
    // cycle. Drop these application roots at handoff; the original native
    // Completion still retains its Graph until actual retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape; phase->slots.release_invocation_roots();
      }
    for (auto &[shape, program] : draft_programs) {
      (void)shape;
      program->guard.execution.completion = {}; program->guard.execution.run = nullptr; program->guard.execution.carrier = nullptr;
      program->carrier.release_invocation_root();
    }
    for (auto &[width, program] : prefix_sampling_programs) { (void)width; program->carrier.release_invocation_root(); }
    for (auto &entry : sampling_configurations) if (entry->frontier)
      entry->frontier->carrier.release_invocation_root();
  }
  inference::RetirementObservation retire(bool wait) noexcept {
    hyle::Status status;
    if (retirement_backend) {
      g::Invocation empty;
      status = wait ? retirement_backend->drain_retired(empty) : retirement_backend->collect_retired();
    }
    // A copy may retain completed Graph outputs after the host Model has
    // handed off its endpoint. Recover that original copy first; neither an
    // idle Graph nor a failed copy observation authorizes early retirement.
    bool copies_retired = true;
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape;
        for (auto &slot : phase->slots.slots()) {
          const auto observed = slot.data.small_results.recover(wait);
          copies_retired = copies_retired && observed.retired;
          inference::record_retirement_status(status, observed.status);
        }
      }
    if (retirement_backend && copies_retired && !pending_readbacks()) release_invocation_roots();
    // A pending original guard is not an observation failure. Preserve actual
    // backend errors independently; neither fact substitutes for the other.
    return {copies_retired && !pending(), status};
  }
};

class Model final {
public:
  struct Sampling final {
    bool enabled = false;
    sampling::Policy policy;
    std::uint64_t seed = 0;
    sampling::LogitsPolicy modifiers;
  };
  struct DraftPolicy final {
    std::size_t max_drafts = 1;
  };
private:
  DraftPolicy draft_policy_;
  bool preparation_started_ = false;
  bool greedy_chain_available_ = false;
  bool draft_packet_available_ = false;
  // Immutable per-sampler route eligibility, resolved once after backend setup.
  struct RelayPlan final { bool decode = false, prefill = false; };
  std::vector<RelayPlan> relay_by_sampling_;
  struct ColdFamilyOutcome final {
    std::optional<FamilyBuildResult> result;
    std::exception_ptr error;
  };
  struct PendingColdFamily final {
    FamilyKey key;
    const PreparedSampling *sampling;
    hyle::sched::JoinHandle<ColdFamilyOutcome> result;
  };
  struct PendingColdUse final {
    std::size_t tokens, requests, count;
    GraphRole role;
    TextCompletion completion;
    std::size_t sampling_index;
    RowPacking packing;
  };
  std::unique_ptr<hyle::sched::BlockingPool> cold_family_pool_;
  std::vector<PendingColdFamily> cold_families_;
  std::vector<PendingColdUse> cold_uses_;
public:
  const std::uint64_t sampling_seed;
  std::size_t max_drafts() const noexcept { return draft_policy_.max_drafts; }
  void configure_draft_policy(DraftPolicy policy) {
    if (preparation_started_)
      throw std::runtime_error("configure draft policy before the first model preparation");
    if (policy.max_drafts >= capacity ||
        (cache_plan.draft_cache && policy.max_drafts >= cache_plan.candidate_boundaries))
      throw std::runtime_error("MTP maximum drafts exceed the original context or candidate capacity");
    if ((resources_->draft_vocabulary || resources_->draft_head) && !policy.max_drafts)
      throw std::runtime_error("prepared draft projection requires active drafting");
    draft_policy_ = policy;
    bool supported_projection = false;
    if (policy.max_drafts >= 2 && policy.max_drafts <= kMaximumDeviceMtpDrafts) {
      const auto found = projections.find(geometry.head_role());
      const auto *stored = found == projections.end() ? nullptr : std::get_if<StoredProjection>(&found->second);
      if (stored && stored->weight) {
        const auto &description = stored->weight->description;
        if (description.is_plain()) supported_projection = true;
        else {
          const auto codec = c::gguf_quantized_type(description.storage_format());
          const auto *capability = codec ? gguf_capabilities(*codec) : nullptr;
          supported_projection = capability && capability->stored_draft_head;
        }
      }
    }
    const GreedyChainCapabilities chain{policy.max_drafts,
        DeviceRelayCapabilities{execution.kind(), execution.has_cpu_assist(), bool(scheduler), false, false}.eligible(),
        cache_plan.active_sequences, cache_plan.auxiliary_hidden_rows, cache_plan.private_draft_stages,
        replay_policy != ReplayPolicy::kCommands, supported_projection};
    draft_packet_available_ = chain.packet_available();
    greedy_chain_available_ = chain.available();
  }
  bool recursive_mtp() const noexcept { return max_drafts() > 1; }
  bool device_mtp_successor_enabled() const noexcept { return device_mtp_successor_; }
  bool bounded_device_mtp_policy()const noexcept{
    return max_drafts()>=1 && max_drafts()<=kMaximumDeviceMtpDrafts &&
        (max_drafts()==1 || (cache_plan.in_flight_batches>=3 && cache_plan.private_draft_stages>=max_drafts()));
  }
  bool device_mtp_continuous_enabled() const noexcept {
    return device_mtp_successor_ && cache_plan.in_flight_batches >= 3;
  }
  bool device_mtp_model_contract() const noexcept {
    constexpr auto exact_token_domain = std::int64_t{1} << std::numeric_limits<float>::digits;
    return cache_plan.draft_cache && cache_plan.active_sequences == 1 &&
        cache_plan.resident_sessions == 1 && cache_plan.in_flight_batches >= 2 &&
        !execution.has_cpu_assist() && !scheduler && bounded_device_mtp_policy() &&
        target_kv_layers && vocabulary <= exact_token_domain;
  }
  bool supports_device_mtp() {
    if (!device_mtp_model_contract()) return false;
    if (max_drafts() > 1 && !greedy_draft_chain_enabled(sampling_configuration(0).configuration)) return false;
    if (!backend_device_mtp_support_) {
      backend_device_mtp_support_ = supports_mtp_attention(execution, attention_capacity(),
          resources_->attention_storage, allocation_budget);
    }
    return *backend_device_mtp_support_;
  }
  void configure_device_mtp_successor(bool enabled) {
    if (preparation_started_)
      throw std::runtime_error("configure device MTP continuation before model preparation");
    if (enabled && !supports_device_mtp())
      throw std::runtime_error("device MTP continuation requires bounded CUDA greedy R1 N1");
    device_mtp_successor_ = enabled;
  }

  void prepare_default_device_mtp_successor(bool requested) {
    if (requested && !preparation_started_ && supports_device_mtp()) device_mtp_successor_ = true;
  }

  bool prepare_draft_projection(const SamplingConfiguration &sampling) {
    if (!resources_->draft_vocabulary && !resources_->draft_head) return false;
    if (sampling.enabled || !sampling.modifiers.empty())
      throw std::runtime_error("draft vocabulary requires unmodified greedy sampling");
    if (resources_->draft_head && (!resources_->draft_vocabulary ||
        resources_->draft_head->original_ids.description.valid())) return true;
    if (active_step || successor_step || carrier_borrowed)
      throw std::logic_error("draft vocabulary preparation requires the original idle model");
    const auto &mapping = *resources_->draft_vocabulary;
    const auto begin = Clock::now();
    if (resources_->draft_head) {
      auto selected = subset_prepared_draft_head(execution, *resources_->draft_head,
          mapping, cache_plan.active_sequences, allocation_budget);
      resources_->draft_head.emplace(std::move(selected));
    } else {
      const auto &projection = projections.at(std::string(geometry.head_role()));
      const auto *stored = std::get_if<StoredProjection>(&projection);
      if (!stored || !stored->weight)
        throw std::runtime_error("draft vocabulary requires the original stored head");
      const auto type = stored->weight->description.data_type();
      if (type != Type::kBFloat16 && type != Type::kFloat16 && type != Type::kFloat32)
        throw std::runtime_error("draft vocabulary requires a BF16/F16/F32 head");
      auto &original = weights.at(std::string(geometry.head_role()));
      resources_->draft_head.emplace(prepare_draft_head(execution, original, mapping,
          cache_plan.active_sequences, allocation_budget));
    }
    resources_->draft_head_prepare_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
    return true;
  }

  DraftProjectionView prepared_draft_projection(bool subset) {
    if (!subset) return {};
    return {{&resources_->draft_head->projection_weight()},
        resources_->draft_head->original_ids.description.valid()?&resources_->draft_head->original_ids:nullptr,
        resources_->draft_head->indexed_weight != nullptr};
  }
  V draft_scores(Recipe &recipe, V selected_hidden, bool subset) {
    return qwen35::draft_scores(recipe, selected_hidden, prepared_draft_projection(subset));
  }
  V original_draft_id(Recipe &recipe, V compact_id, bool subset) {
    return qwen35::original_draft_id(recipe, compact_id, prepared_draft_projection(subset));
  }
  const FfnPlacement ffn_placement;
  std::size_t host_weight_bytes = 0, device_weight_bytes = 0;
  std::size_t physical_text_roles = 0, physical_mtp_roles = 0, canonical_weight_bytes = 0;
  std::size_t ct_channel_groups = 0, ct_local_groups = 0, ct_block_groups = 0, omitted_cache_roles = 0;
  std::size_t fp8_weight_groups = 0, fp8_import_peak_bytes = 0;
  bool float8_weight_policy = false;
  std::array<std::size_t, 3> known_workspace_peak{}, known_workspace_limits{};
  const PrecisionProfile precision;
  const ModelGeometry geometry;
  const std::int64_t hidden, intermediate, vocabulary;
  const std::size_t target_kv_layers, draft_bank;
  const std::vector<std::size_t> attention_layer_ids;
  Backend execution;
  std::shared_ptr<ColdResources> resources_;
  const SamplingConfiguration &default_sampler; // Aggregate diagnostics describe this immutable default.
  // These original runtimes outlive every phase/slot and its native retirement.
  std::unique_ptr<hyle::sched::Scheduler> scheduler;
  std::unique_ptr<hyle::sched::BlockingPool> workers;
  std::vector<c::LeasePool> &weight_pools;
  std::map<std::string, LoadedWeight> &weights;
  ProjectionBindings &projections;
  WeightMeanings &weight_meanings;
  std::string &source_identity;
  std::vector<StateSchema> &state_schema;
  c::ResourceBudget &cache_budget, &execution_budget, &allocation_budget;
  std::size_t known_cache_backing_bytes = 0;
  bool state_storage_prepared = false;
  bool device_mtp_successor_ = false;
  std::optional<bool> backend_device_mtp_support_;
  const cache::Plan cache_plan;
  // The original plain storage/layout root also serves draft attention.
  // Calibrated target storage views share this metadata domain, not its dtype.
  c::BatchSchema attention_schema;
  std::vector<c::BatchSchema> attention_schemas;
  const c::BatchSchema &target_attention_schema(std::size_t layer) const {
    if (layer >= target_kv_layers) throw std::runtime_error("invalid target attention cache slot");
    return attention_schemas.empty() ? attention_schema : attention_schemas.at(layer);
  }
  c::BatchCapacity attention_capacity() const {
    return qwen35::attention_capacity(geometry, precision, cache_plan);
  }
  cache::VersionCount &snapshot_versions;
  std::size_t &carrier_borrowed;
  static constexpr std::size_t slot_index = 0;
  hyle::inference::PagedPageInventory &page_inventory, &draft_inventory;
  std::vector<cache::StatePools> &state_pools;
  c::LeasePool &hidden_pool;
  c::LeaseSlotLayout &hidden_layout;
  // Family/child owners are shared by sessions; no session addresses enter compile.
  std::map<FamilyKey, std::unique_ptr<Family>> &families, &probe_families;
  inference::ColdShapePlan<ExecutionShape, FamilyKey, execution_slot_count> shape_plan;
  // One carrier per actually prepared finite shape. Serial admission borrows
  // its one slot and shared IO through native retirement and all result reads.
  std::map<ExecutionShape, std::unique_ptr<Phase>> &phases, &probe_phases;
  // Configured bounded-prefill policy only. Original Graph Invocations retain
  // their shared scratch; IO, state pools and decode carriers stay independent.
  std::size_t bounded_prefill_rows = 0;
  g::InvocationWorkspace execution_workspace;
  std::map<DraftProgramKey, std::unique_ptr<DraftProgram>> &draft_programs;
  std::map<PrefixSamplingKey, std::unique_ptr<PrefixSamplingProgram>> &prefix_sampling_programs;
  std::size_t draft_program_graph_calls = 0, prefix_decision_graph_calls = 0;
  std::size_t draft_chain_graph_calls = 0, draft_chain_attention_rows = 0;
  std::size_t device_verification_handoffs = 0;
  // Same admission scope as target work: active inputs of original accepted
  // Graph submissions, including discarded/failed work and private NextN.
  // Padding, provider-internal preparation and native kernel FLOPs are separate.
  std::size_t submitted_draft_rows = 0;
  const PreparedSampling &sampling_configuration(std::size_t index = 0) const { return resources_->sampling_configuration(index); }
  std::size_t sampling_configuration_count() const noexcept { return resources_->sampling_configurations.size(); }
  bool supports_device_relay(std::size_t sampling_index) const {
    return relay_by_sampling_.at(sampling_index).decode;
  }
  bool supports_prefill_relay(std::size_t sampling_index) const {
    return relay_by_sampling_.at(sampling_index).prefill;
  }
  std::size_t sampling_index(const PreparedSampling *value) const { return resources_->sampling_index(value); }
  bool retains_sampling_frontier() const noexcept {
    return resources_->retain_sampling_frontier;
  }
  std::vector<std::weak_ptr<Session>> session_links;
  std::size_t frontier_sampling_graph_calls = 0;
  double frontier_sampling_seconds = 0;
  // Destroy episode pins/borrowed links before carriers and the raw policy
  // owners referenced by Snapshot/PageRef, including exceptional shutdown.
  InputPreparationScratch input_scratch;
  std::shared_ptr<StepState> active_step;
  std::shared_ptr<StepState> successor_step;
  std::optional<DecodeAheadMetadata> decode_ahead;
  std::size_t cpu_ahead_prepared = 0, cpu_ahead_hits = 0, cpu_ahead_discarded = 0;
  std::size_t packed_prefill_calls = 0, packed_prefill_active_rows = 0, packed_prefill_physical_rows = 0;
  double cpu_ahead_prepare_seconds = 0;
  const std::size_t capacity;
  const ReplayPolicy replay_policy;
  const std::filesystem::path layer0_probe;
  std::filesystem::path trace_routes;
  explicit Model(const std::filesystem::path &manifest, const ModelGeometry &model_geometry, std::size_t selected_capacity,
                 cache::Plan selected_cache,
                 std::filesystem::path probe = {}, bool progress = false,
                  ReplayPolicy replay = ReplayPolicy::kAuto,
                  std::size_t max_cache_bytes = (std::numeric_limits<std::size_t>::max)(),
                  Sampling sampling_configuration = {false, {}, 0, {}},
                  std::size_t cpu_ffn_layer_count = 0,
                  BackendSelection backend_selection = {}, std::span<const SamplingConfiguration> additional_sampling = {})
       : Model(prepare_weight_source(manifest, selected_cache.draft_cache), model_geometry, selected_capacity,
           selected_cache, std::move(probe), progress, replay, max_cache_bytes,
           sampling_configuration, cpu_ffn_layer_count, backend_selection, additional_sampling) {}

  Model(PreparedWeightSource source, const ModelGeometry &model_geometry, std::size_t selected_capacity,
      cache::Plan selected_cache, std::filesystem::path probe, bool progress, ReplayPolicy replay,
      std::size_t max_cache_bytes, Sampling sampling_configuration, std::size_t cpu_ffn_layer_count,
      BackendSelection backend_selection, std::span<const SamplingConfiguration> additional_sampling = {})
       : Model(create_execution(model_geometry, selected_cache, cpu_ffn_layer_count, backend_selection), {},
           std::move(source), model_geometry, selected_capacity, selected_cache, std::move(probe), progress,
           replay, max_cache_bytes, sampling_configuration, cpu_ffn_layer_count, additional_sampling) {}

  // Explicit cold seam: endpoints exist before the caller constructs one
  // requested-byte scope from their actual resource identities. No new owner
  // or endpoint is created here. The default constructor delegates with no
  // allocation budget, retaining its original allocation path.
  Model(Backend &&prepared_execution, c::ResourceBudget shared_budget,
      PreparedWeightSource source, const ModelGeometry &model_geometry, std::size_t selected_capacity,
      cache::Plan selected_cache, std::filesystem::path probe, bool progress, ReplayPolicy replay,
      std::size_t max_cache_bytes, Sampling sampling_configuration, std::size_t cpu_ffn_layer_count,
      std::span<const SamplingConfiguration> additional_sampling = {})
       : sampling_seed(sampling_configuration.seed), ffn_placement(source.roles, cpu_ffn_layer_count),
         precision(PrecisionProfile::from_roles(source.roles)),
         geometry(model_geometry), hidden(geometry.hidden), intermediate(geometry.intermediate), vocabulary(geometry.vocabulary),
         target_kv_layers(geometry.attention_layers()), draft_bank(target_kv_layers * 2),
         attention_layer_ids([&] {
           std::vector<std::size_t> indices; indices.reserve(target_kv_layers);
           for (std::size_t i = 0; i < geometry.layers.size(); ++i) if (!geometry.is_gdn(i)) indices.push_back(i);
           return indices;
         }()),
         execution(std::move(prepared_execution)),
         resources_(std::make_shared<ColdResources>(geometry, selected_capacity, precision,
              SamplingConfiguration{sampling_configuration.enabled, sampling_configuration.policy, sampling_configuration.modifiers}, additional_sampling,
              source.roles.cache_policy, std::move(source.cache_storage))),
          default_sampler(resources_->sampling_configurations[0]->configuration),
         weight_pools(resources_->weight_pools), weights(resources_->weights), projections(resources_->projections),
         weight_meanings(resources_->weight_meanings), source_identity(resources_->source_identity),
         state_schema(resources_->state_schema), cache_budget(resources_->cache_budget), execution_budget(resources_->execution_budget),
         allocation_budget(resources_->allocation_budget),
         cache_plan(selected_cache), snapshot_versions(resources_->snapshot_versions), carrier_borrowed(resources_->carrier_borrowed),
         page_inventory(resources_->page_inventory), draft_inventory(resources_->draft_inventory),
         state_pools(resources_->state_pools), hidden_pool(resources_->hidden_pool), hidden_layout(resources_->hidden_layout),
         families(resources_->families), probe_families(resources_->probe_families),
         phases(resources_->phases), probe_phases(resources_->probe_phases),
         draft_programs(resources_->draft_programs), prefix_sampling_programs(resources_->prefix_sampling_programs),
         capacity(selected_capacity),
         replay_policy(replay), layer0_probe(std::move(probe)) {
    geometry.validate();
    validate_mtp_source(geometry, cache_plan.draft_cache);
    FfnPlacement::validate(ffn_placement.cpu_layers(), geometry.layers.size());
    if (execution.kind() == BackendKind::kDefault || execution.has_cpu_assist() != (ffn_placement.cpu_layers() != 0))
      throw std::runtime_error("prepared Backend differs from the model CPU placement");
    allocation_budget = std::move(shared_budget);
    if (allocation_budget) {
      // Validate actual rows before any managed source allocation. Resources
      // are explicit; absent rows never become inferred/unattributed charges.
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        (void)take(allocation_budget.usage(resource));
      if (execution.has_cpu_assist()) (void)take(allocation_budget.usage(execution.host_device()));
    }
    const auto &source_roles = source.roles;
    if (source_roles.geometry != geometry) throw std::runtime_error("source geometry changed before weight loading");
    if (source.draft_vocabulary || source.draft_head) {
      if (!cache_plan.draft_cache || (source.draft_vocabulary && source.draft_vocabulary->full_size() != vocabulary))
        throw std::runtime_error("draft vocabulary differs from the original complete model");
      if (std::ranges::any_of(resources_->sampling_configurations, [](const auto &prepared) {
            return prepared->configuration.enabled || !prepared->configuration.modifiers.empty();
          }))
        throw std::runtime_error("draft vocabulary requires unmodified greedy sampling for every request");
      if (source.draft_vocabulary && source.draft_vocabulary_identity != inference::artifact::sha256(std::as_bytes(source.draft_vocabulary->ids())))
        throw std::runtime_error("draft vocabulary identity changed before weight loading");
    }
    const bool calibrated = resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration;
    if ((calibrated && (!target_kv_layers || resources_->attention_storage.size() != target_kv_layers)) ||
        (!calibrated && !resources_->attention_storage.empty()))
      throw std::runtime_error("prepared E4 cache storage requires its complete original target calibration pairs");
    const auto plain = take(c::CacheStorage::plain(precision.cache_type()));
    const auto plain_rows = cache_storage_row_bytes(geometry, {plain, plain});
    const auto target_rows = calibrated ? cache_storage_row_bytes(geometry, resources_->attention_storage.front()) : plain_rows;
    const cache::KvRowBytes actual_target{target_rows[0], target_rows[1]}, actual_draft{plain_rows[0], plain_rows[1]};
    if (source.cache_row_bytes.target != actual_target ||
        ((source.cache_row_bytes.draft.key || source.cache_row_bytes.draft.value || cache_plan.draft_cache) &&
            source.cache_row_bytes.draft != actual_draft) ||
        std::ranges::any_of(resources_->attention_storage, [&](const auto &pair) {
          return cache_storage_row_bytes(geometry, pair) != target_rows;
        }))
      throw std::runtime_error("prepared target/draft row extents differ from their actual typed source carriers");
    const auto bank_rows = cache::multiply(cache_plan.max_pages, cache::Plan::page_rows);
    if (cache_plan.row_bytes != source.cache_row_bytes ||
        cache_plan.key_bank_bytes != cache::add(cache::multiply(bank_rows, source.cache_row_bytes.target.key), 1) ||
        cache_plan.value_bank_bytes != cache::add(cache::multiply(bank_rows, source.cache_row_bytes.target.value), 1) ||
        cache_plan.draft_key_bank_bytes != (cache_plan.draft_cache
            ? cache::add(cache::multiply(bank_rows, source.cache_row_bytes.draft.key), 1) : 0) ||
        cache_plan.draft_value_bank_bytes != (cache_plan.draft_cache
            ? cache::add(cache::multiply(bank_rows, source.cache_row_bytes.draft.value), 1) : 0))
      throw std::runtime_error("cache backing plan differs from independent target/draft physical storage");
    if (cache_plan.hidden_row_bytes != cache::multiply(static_cast<std::size_t>(hidden), precision.activation_bytes()) || cache_plan.attention_layers != target_kv_layers ||
        cache_plan.recurrent_layers != geometry.recurrent_layers())
      throw std::runtime_error("cache layout differs from source model geometry");
    if (retains_sampling_frontier() && (!cache_plan.hidden_bytes || cache_plan.hidden_slots < cache_plan.snapshot_slots))
      throw std::runtime_error("sampling requires a cache plan with retained frontier storage");
    const std::array limits{c::ResourceBudgetLimit{execution.device(), max_cache_bytes},
        c::ResourceBudgetLimit{mem::system_resource(), (std::numeric_limits<std::size_t>::max)()}};
    // CPU payload and host metadata can use the same real resource. Keep the
    // caller's finite cache limit when they coincide; never replace it with
    // the otherwise independent unlimited metadata-domain entry.
    const auto limit_count = execution.device() == mem::system_resource() ? 1u : 2u;
    // This allocator couples only actual CPU metadata. The explicit second
    // source-factory constraint below separately gates original cache payloads.
    const auto cache_allocator = allocation_budget ? take(allocation_budget.metadata_allocator()) : mem::system_allocator();
    cache_budget = take(c::ResourceBudget::create({.limits = std::span(limits).first(limit_count),
        .allocator = cache_allocator}));
    if (allocation_budget) execution_budget = allocation_budget;
    else {
      // Execution controls and native outputs have their own actual resource
      // scope on every backend. State/KV/hidden pools remain cache storage;
      // private slot indices and their pinned staging do not consume its plan.
      std::vector<c::ResourceBudgetLimit> execution_limits;
      execution_limits.reserve(4);
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        if (std::ranges::none_of(execution_limits, [resource](const auto &limit) {
              return limit.resource == resource;
            }))
          execution_limits.push_back({resource});
      if (ffn_placement.cpu_layers()) {
        const auto host = weight_resource(execution, true);
        if (std::ranges::none_of(execution_limits, [host](const auto &limit) { return limit.resource == host; }))
          execution_limits.push_back({host});
      }
      execution_budget = take(c::ResourceBudget::create({.limits = execution_limits, .allow_unattributed = true}));
    }
    if (std::ranges::any_of(resources_->sampling_configurations,
        [](const auto &configuration) { return !configuration->configuration.modifiers.history.empty(); })) {
      const auto blocks = cache_plan.active_sequences;
      if (!blocks || blocks >= UINT32_MAX) throw std::overflow_error("history snapshot block capacity exceeds lease indexing");
      const auto bytes = cache::multiply(static_cast<std::size_t>(vocabulary), sizeof(std::int32_t));
      const auto pool_capacity = take(inference::checked_output_pool_capacity(
          std::array{bytes}, blocks, cache::state_pool_alignment));
      resources_->history_counts_pool = take(c::LeasePool::create(execution.device(),
          cache::add(pool_capacity.backing_bytes, 1),
          {.max_blocks = pool_capacity.max_blocks, .resource_budget = execution_budget}));
    }
    if (progress) {
      auto status = hyle::Status::Ok();
      scheduler = std::make_unique<hyle::sched::Scheduler>(1, mem::system_allocator(), 64, &status);
      check(status);
      workers = std::make_unique<hyle::sched::BlockingPool>(mem::system_allocator(), 1,
                                                          std::chrono::seconds(10), &status);
      check(status);
    }
    relay_by_sampling_.reserve(resources_->sampling_configurations.size());
    for (const auto &prepared : resources_->sampling_configurations) {
      const auto &sampling = prepared->configuration;
      relay_by_sampling_.push_back({
          DeviceRelayCapabilities{execution.kind(), execution.has_cpu_assist(), bool(scheduler),
              sampling.enabled && cache_plan.draft_cache, !sampling.modifiers.empty()}.eligible(),
          PrefillRelayFacts{execution.kind(), execution.has_cpu_assist(), bool(scheduler),
              cache_plan.draft_cache, cache_plan.in_flight_batches,
              !sampling.enabled && sampling.modifiers.empty()}.eligible()});
    }
    const auto weight_load_start = Clock::now();
    auto loading = load_selected_weight_roles(source_roles, execution, weights, weight_pools, ffn_placement, allocation_budget);
    source_identity = loading.identity;
    weight_meanings = std::move(loading.meanings);
    projections = prepare_projection_bindings(weights, cache_plan.draft_cache, geometry);
    if(source.draft_head){
      const auto begin=Clock::now();
      resources_->draft_head.emplace(load_prepared_draft_head(execution,source_roles,*source.draft_head,allocation_budget,
          source.draft_vocabulary ? &*source.draft_vocabulary : nullptr));
      resources_->draft_head_prepare_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    }
    if (source.draft_vocabulary) {
      resources_->draft_vocabulary = std::move(source.draft_vocabulary);
      resources_->draft_vocabulary_identity = std::move(source.draft_vocabulary_identity);
    }
    if(resources_->draft_vocabulary || resources_->draft_head)(void)prepare_draft_projection(default_sampler);
    host_weight_bytes = loading.host_bytes; device_weight_bytes = loading.device_bytes;
    physical_text_roles = loading.physical_text_roles; physical_mtp_roles = loading.physical_mtp_roles;
    canonical_weight_bytes = loading.canonical_bytes; fp8_weight_groups = loading.fp8_groups;
    fp8_import_peak_bytes = loading.import_peak_bytes; float8_weight_policy = loading.float8_policy;
    ct_channel_groups = loading.ct_channel_groups; ct_local_groups = loading.ct_local_groups;
    ct_block_groups = loading.ct_block_groups;
    omitted_cache_roles = loading.omitted_cache_roles;
    const auto weight_load_seconds = source.metadata_prepare_seconds +
        std::chrono::duration<double>(Clock::now() - weight_load_start).count();
    std::cerr << "resident text weights: " << weights.size() << " tensors, " << loading.bytes << " bytes\n";
    if (float8_weight_policy) std::cerr << "fp8_weight_import original_physical_roles="
        << physical_text_roles + physical_mtp_roles << " original_bytes=" << loading.bytes
        << " canonical_tensors=" << weights.size() << " canonical_bytes=" << canonical_weight_bytes
        << " groups=" << fp8_weight_groups << " temporary_group_peak_bytes=" << fp8_import_peak_bytes << '\n';
    if (loading.nf4_groups) std::cerr << "nf4_source_operands groups=" << loading.nf4_groups
        << " original_group_bytes=" << loading.nf4_original_bytes << " canonical_bank_bytes=" << canonical_weight_bytes
        << " input_copy_bytes=" << loading.nf4_input_copy_bytes
        << " known_group_peak_bytes=" << loading.nf4_known_group_peak_bytes
        << " source_operands_resident=true decode_precision=F16 selected_plan_reports_packed_or_decomposed=true\n";
    std::cerr << "weights_load_seconds=" << weight_load_seconds << '\n';
    if (ffn_placement.cpu_layers()) std::cerr << "weight_placement cpu_ffn_layers=" << ffn_placement.cpu_layers()
        << " original_host_weight_bytes=" << host_weight_bytes
        << " original_device_weight_bytes=" << device_weight_bytes
        << " duplicated_original_weight_bytes=0\n";
    std::cerr << "weight_loader backend=" << execution.name() << " staging_bytes=" << loading.stream.staging_bytes
              << " io_calls=" << loading.stream.io_calls
              << " native_calls=" << loading.stream.native_calls
              << " native_waits=" << loading.stream.native_waits
              << " host_waits=" << loading.stream.host_waits
              << " phase_completions=" << loading.stream.phase_completions
              << " copied_physical_bytes=" << loading.stream.copied_physical_bytes
              << " peak_backing_bytes=" << loading.stream.peak_backing_bytes << '\n';
  }

  ~Model() {
    cancel_cold_families();
    // Stop the host driver before joining its original scheduler/pool. Native
    // cancellation remains advisory and never serves as retirement proof.
    for (auto &weak : session_links) if (auto session = weak.lock()) session->stop().IgnoreError();
    (void)abort_episode();
    // Attempt actual native retirement before detaching application links.
    // Scheduler idleness alone does not prove native/completion retirement.
    if (!resources_->pending_readbacks())
      for (auto *cache : {&phases, &probe_phases})
        for (auto &[tokens, phase] : *cache) {
          (void)tokens;
          (void)phase->slots.retire_all(execution);
        }
    if (workers) workers->shutdown();
    if (scheduler) scheduler->drain();
    (void)abort_episode();
    // A persistent unknown observation may outlive this application Model.
    // Preserve the original native endpoint as an explicit Session retirement
    // continuation, while the accepted guard pins the complete cold scope.
    resources_->retirement_backend.emplace(std::move(execution));
    for (auto &weak : session_links) if (auto session = weak.lock()) {
      session->model_ = nullptr;
      session->stopping = true;
      session->pending = nullptr;
      if (active_step) inference::record_retirement_status(session->retirement_status, active_step->execution.terminal);
    }
    // Move the original copy observations to their native reaper at handoff.
    // Their cold retention nodes remain queryable and retain the source scope
    // until actual retirement, even without an escaped Session to poll it.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape;
        for (auto &slot : phase->slots.slots()) slot.data.small_results.abandon_observation();
      }
    resources_->release_invocation_roots();
    // Teardown never promotes an unpublished successor back into escaped
    // Sessions whose Model link has already been withdrawn above.
    if (successor_step) successor_step->execution.candidate_admission = nullptr;
    if (active_step) active_step->execution.candidate_admission = nullptr;
    successor_step.reset(); active_step.reset();
  }

  std::string_view score_precision_policy() const {
    return score_policy(projections.at(std::string(geometry.head_role())), precision);
  }
  WeightInventoryFacts weight_inventory_facts() const noexcept {
    const bool cache_float8 = resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration;
    return {.text_tensors = physical_text_roles, .mtp_tensors = physical_mtp_roles,
        .float8_weight_policy = float8_weight_policy, .canonical_tensors = weights.size(),
        .canonical_bytes = canonical_weight_bytes, .fp8_groups = fp8_weight_groups,
        .fp8_import_peak_bytes = fp8_import_peak_bytes, .fp8_reader = prepared::fp8_reader,
        .ct_channel_groups = ct_channel_groups, .ct_local_groups = ct_local_groups,
        .ct_block_groups = ct_block_groups, .omitted_cache_roles = omitted_cache_roles,
        .cache_float8 = cache_float8,
        .cache_calibration_roles = cache_float8 ? resources_->attention_storage.size() * 2 : 0,
        .draft_cache = cache_plan.draft_cache, .plain_cache_f16 = precision.is_f16()};
  }
  Tensor caller_tensor(mem::MemoryResourceRef resource, c::TensorDesc description, bool caller = true) const {
    return Tensor(resource, description, caller, allocation_budget);
  }

  c::ResourceBudgetUsage cache_usage() const {
    const auto usage = take(cache_budget.usage(execution.device()));
    const auto metadata = usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)];
    const auto payload = usage.live_bytes - metadata;
    if (payload != known_cache_backing_bytes)
      throw std::runtime_error("actual cache backing differs: payload=" + std::to_string(payload) +
          " known=" + std::to_string(known_cache_backing_bytes) + " metadata=" + std::to_string(metadata));
    return usage;
  }

  void merge_execution_plan(const ExecutionPlanFacts &facts) {
    for (std::size_t i = 0; i < facts.domain_count; ++i) {
      known_workspace_peak[i] = (std::max)(known_workspace_peak[i], facts.workspace_peak[i]);
      known_workspace_limits[i] = facts.workspace_limits[i];
    }
  }
  void record_execution_plan(const g::CompileReport &report, std::span<const g::ConstantBinding> constants) {
    merge_execution_plan(inspect_execution_plan(execution, ffn_placement.cpu_layers(), report, constants));
  }

  void prepare_cache() {
    preparation_started_ = true;
    if (attention_schema.valid()) {
      return;
    }
    auto prepared_schema = take(c::BatchSchema::create(attention_capacity(),
        {.resource_budget = cache_budget}));
    std::vector<c::BatchSchema> prepared_schemas;
    prepared_schemas.reserve(resources_->attention_storage.size());
    for (const auto &pair : resources_->attention_storage)
      prepared_schemas.push_back(take(prepared_schema.with_cache_storage(pair[0], pair[1])));
    hyle::inference::PagedPageInventory prepared_inventory, prepared_draft;
    check(prepared_inventory.prepare(cache_plan.max_pages, target_kv_layers * 2));
    for (std::size_t i = 0; i < prepared_inventory.banks.size() / 2; ++i) {
      const auto &schema = prepared_schemas.empty() ? prepared_schema : prepared_schemas[i];
      prepared_inventory.banks[2 * i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_inventory.banks[2 * i + 1] = take(c::PagedArena::create(schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    if (cache_plan.draft_cache) {
      check(prepared_draft.prepare(cache_plan.max_pages, 2));
      prepared_draft.banks[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.draft_key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_draft.banks[1] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.draft_value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    const auto prepared_bytes = cache::add(cache::multiply(
        cache::add(cache_plan.key_bank_bytes, cache_plan.value_bank_bytes), target_kv_layers),
        cache::add(cache_plan.draft_key_bank_bytes, cache_plan.draft_value_bank_bytes));
    const auto layout_capacity = prepared_schema.capacity();
    input_scratch.paged_appends.prepare(layout_capacity.max_requests, layout_capacity.max_page_refs);
    input_scratch.paged_append_capacity.prepare(layout_capacity.max_page_refs, draft_bank);
    // No PageRef exists until this complete cold source tuple is published.
    // Leave the schema sentinel invalid on failure so released quota can retry.
    static_assert(std::is_nothrow_move_assignable_v<hyle::inference::PagedPageInventory>);
    static_assert(std::is_nothrow_move_assignable_v<c::BatchSchema>);
    page_inventory = std::move(prepared_inventory);
    draft_inventory = std::move(prepared_draft);
    known_cache_backing_bytes = prepared_bytes;
    snapshot_versions.capacity = cache_plan.snapshot_slots;
    attention_schemas = std::move(prepared_schemas);
    attention_schema = std::move(prepared_schema);
    prepare_decode_ahead_storage();
  }

  void prepare_state_storage() {
    preparation_started_ = true;
    if (state_storage_prepared) return;
    prepare_sampling_frontier();
    // Every canonical state row uses its original versioned pool. Logical
    // Store outputs do not depend on a provider's ordinary output ownership.
    // Publish together so a failed allocation releases its temporary backing
    // and leaves this Model ready for a later admission attempt.
    std::vector<cache::StatePools> prepared_pools(geometry.layers.size());
    c::LeasePool prepared_hidden;
    c::LeaseSlotLayout prepared_hidden_layout;
    auto prepared_bytes = known_cache_backing_bytes;
    for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
      const auto &description = state_schema[i]; auto &pools = prepared_pools[i];
      const auto history_slots = cache_plan.state_slots;
      const auto recurrent_slots = cache_plan.state_slots;
      const auto history_bytes = cache::state_pool_backing_bytes(take(description.history.required_bytes()), history_slots);
      const auto recurrent_bytes = cache::state_pool_backing_bytes(take(description.recurrent.required_bytes()), recurrent_slots);
      pools.history = take(c::LeasePool::create(execution.device(), history_bytes,
          {.max_blocks = history_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      pools.recurrent = take(c::LeasePool::create(execution.device(), recurrent_bytes,
          {.max_blocks = recurrent_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      const auto history_payload = take(description.history.required_bytes());
      const auto recurrent_payload = take(description.recurrent.required_bytes());
      pools.history_layout = take(c::LeaseSlotLayout::create(pools.history, description.history,
          history_slots, cache::add(history_payload, cache::state_pool_padding(history_payload))));
      pools.recurrent_layout = take(c::LeaseSlotLayout::create(pools.recurrent, description.recurrent,
          recurrent_slots, cache::add(recurrent_payload, cache::state_pool_padding(recurrent_payload))));
      check(pools.history_reuse.prepare(cache_plan.active_sequences, history_payload));
      check(pools.recurrent_reuse.prepare(cache_plan.active_sequences, recurrent_payload));
      prepared_bytes = cache::add(prepared_bytes, cache::add(history_bytes, recurrent_bytes));
    }
    if (cache_plan.hidden_bytes) {
      const auto slots = cache_plan.hidden_slots;
      const auto bytes = cache::state_pool_backing_bytes(cache_plan.hidden_row_bytes, slots);
      prepared_hidden = take(c::LeasePool::create(execution.device(), bytes,
          {.max_blocks = slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      prepared_hidden_layout = take(c::LeaseSlotLayout::create(prepared_hidden,
          dense(precision.activation_type(), {1, hidden}), slots,
          cache::add(cache_plan.hidden_row_bytes, cache::state_pool_padding(cache_plan.hidden_row_bytes))));
      prepared_bytes = cache::add(prepared_bytes, bytes);
    }
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes - usage.live_by_category[
            static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] != prepared_bytes)
      throw std::runtime_error("shared cache backing inventory differs from the actual reserved bytes");
    state_pools = std::move(prepared_pools); hidden_pool = std::move(prepared_hidden);
    hidden_layout = std::move(prepared_hidden_layout);
    known_cache_backing_bytes = prepared_bytes;
    state_storage_prepared = true;
    for (auto *children : {&phases, &probe_phases})
      for (auto &[key, phase] : *children) prepare_transaction_layouts(*phase);
    for (auto &[key, program] : draft_programs) if (program->head)
      check(program->hidden_transactions.set_layout(0, hidden_layout));
  }

  // Graph/counts may be prepared before admission allocates canonical storage.
  // The pool's immutable layout is attached only at that original cold boundary.
  void prepare_transaction_layouts(Phase &phase) {
    for (std::size_t bank = 0; bank < phase.state_banks.size(); ++bank) {
      const auto &facts = phase.state_banks[bank];
      const auto &layout = facts.kind == StateBankKind::kHidden ? hidden_layout
          : facts.kind == StateBankKind::kRecurrent ? state_pools[facts.layer].recurrent_layout
              : state_pools[facts.layer].history_layout;
      for (std::size_t slot = 0; slot < phase.slots.size(); ++slot)
        check(phase.slots[slot].transactions.set_layout(bank, layout));
    }
  }

  // Compile the default single-session CUDA families concurrently only when
  // every shared weight is already an immutable read. Supported original
  // GGUF bindings are borrowed without rewriting their source bytes; their
  // native recipes and packing caches belong to each compiled Family.
  // Other packed codecs and caller budgets keep serial admission.
  bool begin_cold_families() {
    if (preparation_started_ || state_storage_prepared || allocation_budget ||
        !families.empty() || !probe_families.empty() || cold_family_pool_ ||
        std::ranges::any_of(weights, [](const auto &item) {
          const auto codec = item.second.codec;
          const auto &description = item.second.tensor.description;
          const auto quantized = c::gguf_quantized_type(description.storage_format());
          const auto *capability = quantized ? gguf_capabilities(*quantized) : nullptr;
          const bool original_gguf = capability && capability->concurrent_cold_family;
          return bool(item.second.tensor.writer) || !item.second.tensor.ready ||
              (!description.is_plain() && !original_gguf) ||
              (codec != prepared::ProjectionCodec::kUnspecified && codec != prepared::ProjectionCodec::kDense);
        })) return false;
    // The optional row transform finishes with the original cold Model owner.
    // Family workers only borrow its immutable projection and ID-map leases.
    if (resources_->draft_vocabulary && !resources_->draft_head) return false;
    if (resources_->draft_head && (resources_->draft_head->projection_weight().writer ||
        !resources_->draft_head->projection_weight().ready ||
        (resources_->draft_head->original_ids.description.valid() && (resources_->draft_head->original_ids.writer ||
            !resources_->draft_head->original_ids.ready)))) return false;
    const auto available = std::thread::hardware_concurrency();
    if (available < 2) return false;
    const auto threads = (std::min)(available, 8u);
    auto status = hyle::Status::Ok();
    std::unique_ptr<hyle::sched::BlockingPool> pool;
    try {
      pool = std::make_unique<hyle::sched::BlockingPool>(mem::system_allocator(), threads,
          std::chrono::seconds(10), &status);
    } catch (const std::bad_alloc &) {
      return false;
    }
    if (!status.ok()) return false;
    prepare_cache();
    cold_family_pool_ = std::move(pool);
    return true;
  }

  void cancel_cold_families() noexcept {
    // Jobs borrow Model, its backend and the immutable weight bank. A
    // cancelled startup must join them before any of those owners retire.
    if (cold_family_pool_) cold_family_pool_->shutdown();
    cold_family_pool_.reset();
    cold_families_.clear();
    cold_uses_.clear();
  }

  void finish_cold_families() {
    if (!cold_family_pool_) return;
    std::vector<FamilyBuildResult> completed;
    completed.reserve(cold_families_.size());
    for (auto &pending : cold_families_) {
      auto outcome = take(pending.result.join());
      if (outcome.error) std::rethrow_exception(outcome.error);
      if (!outcome.result || !outcome.result->family)
        throw inference::EvaluationFailure(hyle::Status::Unimplemented(
            "configured bounded prefill has no original packed implementation"));
      completed.push_back(std::move(*outcome.result));
    }
    cold_family_pool_->shutdown();
    cold_family_pool_.reset();
    for (std::size_t i = 0; i < completed.size(); ++i) {
      auto &built = completed[i];
      auto &pending = cold_families_[i];
      built.family->sampling = pending.sampling;
      merge_execution_plan(built.plan);
      if (!families.emplace(pending.key, std::move(built.family)).second)
        throw std::runtime_error("text finite family was already prepared");
    }
    cold_families_.clear();
    auto uses = std::move(cold_uses_);
    cold_uses_.clear();
    for (const auto &use : uses)
      plan_replay_uses(use.tokens, use.requests, use.count, use.role,
                       use.completion, use.sampling_index, use.packing);
  }

  FamilyBuildResult build_prepared_family(std::span<const std::size_t> chunks,
      std::size_t requests, std::size_t prefix_rows, GraphRole role, TextCompletion completion,
      RowPacking packing, bool diagnostic, bool admitted_exact, bool subset,
      const SamplingConfiguration &selected_sampler) {
    return build_family(
        {chunks, requests, prefix_rows, role, completion, packing, diagnostic, admitted_exact},
        {geometry, precision, weights, projections, weight_meanings, ffn_placement.cpu_layers()},
        {attention_schema, attention_schemas, state_schema, cache_plan.state_slots, cache_plan.hidden_slots,
         retains_sampling_frontier(), cache_plan.in_flight_batches >= 2 && !scheduler &&
             (role != GraphRole::kMtpContinuation || device_mtp_continuous_enabled()),
         prepared_draft_projection(subset), device_mtp_successor_ && requests == 1 &&
             drafts_token(role) && (role != GraphRole::kMtpContinuation || device_mtp_continuous_enabled()) &&
             !selected_sampler.enabled && selected_sampler.modifiers.empty(), device_mtp_continuous_enabled(),
         role == GraphRole::kVerify && prefix_rows >= 2 && requests == 1 && packing == RowPacking::kRectangular &&
             greedy_draft_chain_enabled(selected_sampler) && cache_plan.in_flight_batches >= 3,
         (role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound || role==GraphRole::kMtpContinuation) && requests == 1 &&
             packing == RowPacking::kRectangular && prefix_rows == max_drafts() + 1 &&
             greedy_draft_chain_enabled(selected_sampler) && cache_plan.in_flight_batches >= 3 ? max_drafts() : 0},
        selected_sampler, {execution, allocation_budget, execution_budget});
  }

  bool prepare_family(std::span<const std::size_t> planned_chunks, bool diagnostic = false,
                      std::size_t requests = 1, GraphRole role = GraphRole::kText,
                      bool admitted_exact = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                      RowPacking packing = RowPacking::kRectangular) {
    const bool packed_prefill = packs_prefill(packing);
    const bool compact_draft = packs_draft(packing);
    const bool packed_verify = packs_verify(packing);
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role == GraphRole::kMtpContinuation && (!device_mtp_successor_ || requests != 1 ||
        selected_sampler.enabled || !selected_sampler.modifiers.empty() || packing != RowPacking::kRectangular))
      throw std::runtime_error("device MTP family requires its original configured greedy R1 continuation");
    if (!supports_completion(role, completion))
      throw std::runtime_error("state-only completion requires a Text or teacher-forced MTP input family");
    if (packed_prefill && (diagnostic || admitted_exact ||
        (role != GraphRole::kText && !is_mtp_prefill(role))))
      throw std::runtime_error("packed prefill requires an original cold Text or MTP prefill family");
    if (compact_draft && (diagnostic || admitted_exact || requests < 2 ||
        role != GraphRole::kMtpCatchupRound || completion != TextCompletion::kEmitNext))
      throw std::runtime_error("compact draft requires a multi-member K1 round capacity");
    if (packed_verify && (diagnostic || admitted_exact || role != GraphRole::kVerify ||
        completion != TextCompletion::kEmitNext || planned_chunks.size() != 1 || !target_kv_layers))
      throw std::runtime_error("packed verification requires one original cold prefix capacity and attention layout");
    preparation_started_ = true;
    if (state_storage_prepared && !admitted_exact)
      throw std::runtime_error("compile finite families before preparing their selected state storage");
    if (admitted_exact && (active_step || carrier_borrowed || diagnostic || planned_chunks.size() != 1 || verifies_prefix(role)))
      throw std::runtime_error("exact suffix admission requires an idle single cold shape within the original state plan");
    if (planned_chunks.empty()) throw std::runtime_error("text finite family requires known nonempty chunks");
    // Snapshot count is a fixed operation descriptor. Each verifier width
    // therefore owns its exact cold family, rather than changing it at submit.
    if (role == GraphRole::kVerify && planned_chunks.size() != 1) {
      std::set<std::size_t> widths(planned_chunks.begin(), planned_chunks.end());
      for (const auto width : widths) {
        const std::array chunk{width};
        prepare_family(chunk, diagnostic, requests, role, admitted_exact, completion, sampling_index, packing);
      }
      return true;
    }
    const std::size_t prefix_rows = role == GraphRole::kVerify ? planned_chunks.front()
        : drafts_token(role) ? ((role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound || role==GraphRole::kMtpContinuation) &&
            requests == 1 && packing == RowPacking::kRectangular &&
            greedy_draft_chain_enabled(sampling_configuration(sampling_index).configuration) &&
            cache_plan.in_flight_batches >= 3 ? max_drafts() + 1 : 2) : 1;
    if (!prefix_rows || prefix_rows > cache_plan.candidate_boundaries)
      throw std::runtime_error("verification width exceeds the positive prepared prefix capacity");
    const auto ports_per_request = cache::add(76, cache::multiply(38, prefix_rows - 1));
    if (!requests || requests > cache_plan.active_sequences ||
        requests > (std::numeric_limits<std::size_t>::max)() / ports_per_request)
      throw std::runtime_error("text finite family request count exceeds its configured capacity");
    if (verifies_prefix(role) && (diagnostic || !layer0_probe.empty() ||
        std::ranges::any_of(planned_chunks, [&](auto t) { return t != prefix_rows; })))
      throw std::runtime_error("verification requires one exact width and independently budgeted prefix boundaries");
    if (has_mtp(role) && (!cache_plan.draft_cache || diagnostic || !layer0_probe.empty()))
      throw std::runtime_error("MTP requires its complete cache plan and a non-diagnostic family");
    if (has_mtp(role) && !drafts_token(role) && !is_mtp_prefill(role) &&
        std::ranges::any_of(planned_chunks, [](auto t) { return t != 1; }))
      throw std::runtime_error("MTP single-token families require T1");
    if (is_mtp_prefill(role) && std::ranges::any_of(planned_chunks, [](auto t) { return t < 2; }))
      throw std::runtime_error("MTP multi-token prefill requires at least two rows; use its single-token family for T1");
    const auto family_key = admitted_exact ? FamilyKey::exact(requests, role, planned_chunks.front(), completion, sampling_index) : FamilyKey(requests, role, prefix_rows, completion, sampling_index, packing);
    auto &table = diagnostic ? probe_families : families;
    if (cold_family_pool_ && (diagnostic || admitted_exact || compact_draft || packed_verify))
      finish_cold_families();
    if (table.contains(family_key) || (!diagnostic && std::ranges::any_of(cold_families_,
            [&](const auto &pending) { return pending.key == family_key; })))
      throw std::runtime_error("text finite family was already prepared");
    // Validate admission before materializing the original cache/head tuple.
    // A later factory failure keeps that successfully prepared tuple alive.
    for (const auto tokens : planned_chunks)
      if (tokens == 0 || tokens > capacity)
        throw std::runtime_error("finite text chunk outside nonempty context capacity");
    const auto max_query_rows = *std::max_element(planned_chunks.begin(), planned_chunks.end());
    if (requests > cache_plan.active_sequences || cache::multiply(requests, max_query_rows) > cache_plan.max_query_rows)
      throw std::runtime_error("finite family exceeds the configured batch capacity");
    prepare_cache();
    const bool subset = drafts_token(role) && prepare_draft_projection(selected_sampler);
    if (cold_family_pool_) {
      auto task = hyle::sched::spawn_blocking(*cold_family_pool_, mem::system_allocator(),
          [this, chunks = std::vector<std::size_t>(planned_chunks.begin(), planned_chunks.end()),
           requests, prefix_rows, role, completion, packing, diagnostic, admitted_exact, subset,
           selected = &selected_sampler]() -> ColdFamilyOutcome {
        try {
          const auto planned = std::span<const std::size_t>(chunks);
          auto built = build_prepared_family(planned, requests, prefix_rows, role, completion, packing,
                                             diagnostic, admitted_exact, subset, *selected);
          return {.result = std::move(built)};
        } catch (...) {
          return {.error = std::current_exception()};
        }
      });
      cold_families_.push_back({family_key, &prepared_sampling, std::move(task)});
      return true;
    }
    auto built = build_prepared_family(planned_chunks, requests, prefix_rows, role, completion, packing,
                                       diagnostic, admitted_exact, subset, selected_sampler);
    if (!built.family) return false;
    built.family->sampling = &prepared_sampling;
    merge_execution_plan(built.plan);
    // Publish only the complete stable candidate after all children and checks.
    table.emplace(family_key, std::move(built.family));
    return true;
  }

  // These capacities are physical carriers. Actual prefill consumption remains
  // the caller's true input slice and never becomes a padded Session boundary.
  bool prepare_prefill_family(std::span<const std::size_t> capacities,
      TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    // This adapter reuses the original attention mirror's true query control.
    // A graph without that carrier, or a diagnostic graph with different
    // published boundaries, has no implementation of this packed signature.
    if (!target_kv_layers || !layer0_probe.empty()) return false;
    return prepare_family(capacities, false, requests, role, false, completion, sampling_index, RowPacking::kPrefill);
  }

  std::optional<FamilyChild> find_child(std::size_t requests, std::size_t tokens, GraphRole role,
                      bool diagnostic = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                      RowPacking packing = RowPacking::kRectangular) {
    auto &table = diagnostic ? probe_families : families;
    const auto select = [&](const FamilyKey &key) -> std::optional<FamilyChild> {
      const auto found = table.find(key);
      if (found == table.end()) return {};
      auto &family = *found->second;
      const auto point = std::ranges::find(family.chunks, tokens);
      if (point == family.chunks.end()) return {};
      return FamilyChild{family, family.programs.child(static_cast<std::size_t>(point - family.chunks.begin()))};
    };
    if (auto primary = select({requests, role, tokens, completion, sampling_index, packing})) return primary;
    if (packing != RowPacking::kRectangular) return {};
    return select(FamilyKey::exact(requests, role, tokens, completion, sampling_index));
  }

  std::size_t prefill_capacity(std::size_t rows, TextCompletion completion, std::size_t sampling_index,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) const {
    if (!rows || !layer0_probe.empty()) return 0;
    const auto found = families.find({requests, role, 0, completion, sampling_index, RowPacking::kPrefill});
    if (found == families.end() || !found->second) return 0;
    std::size_t selected = 0;
    for (const auto physical : found->second->chunks)
      if (physical >= rows && (!selected || physical < selected)) selected = physical;
    return selected;
  }

  std::size_t verification_capacity(std::size_t rows, std::size_t requests = 1,
                                    std::size_t sampling_index = 0) const {
    std::size_t selected = 0;
    for (const auto &[key, family] : families)
      if (family && key.role == GraphRole::kVerify && key.requests == requests &&
          key.completion == TextCompletion::kEmitNext && key.sampling_index == sampling_index && packs_verify(key.packing))
        for (const auto extent : family->chunks)
          if (rows <= extent && (!selected || extent < selected)) selected = extent;
    return selected;
  }

  // Session admission proves a prepared cohort can execute this member. The
  // actual batch still selects its exact width/capacity before taking a loan.
  bool planned_prefill_member(std::size_t rows, TextCompletion completion, std::size_t sampling_index,
      GraphRole role = GraphRole::kText) const {
    for (const auto &[shape, uses] : shape_plan.entries()) {
      if (!packs_prefill(shape.packing) || shape.role != role || shape.completion != completion ||
          shape.sampling_index != sampling_index || shape.tokens < rows ||
          std::ranges::none_of(uses, [](auto count) { return count != 0; })) continue;
      if (prefill_capacity(rows, completion, sampling_index, shape.requests, role)) return true;
    }
    return false;
  }

  // A generated-token checkpoint can expose a suffix whose length was unknown
  // at model setup. Specialize only that actual cold request. The same original
  // Family/Phase owners remain bounded by the frozen cold shape plan; decoded
  // hot-loop shapes stay in their original immutable finite families.
  void freeze_shape_plan() {
    if (shape_plan.freeze())
      std::cerr << "cold_shape_plan planned_shapes=" << shape_plan.size()
                << " extra_capacity=" << shape_plan.capacity() << '\n';
  }

  void admit_exact_shape(std::size_t tokens, std::size_t requests, GraphRole role,
                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (find_child(requests, tokens, role, false, completion, sampling_index)) return;
    const auto admission_start = Clock::now();
    if (!state_storage_prepared || active_step || carrier_borrowed || !tokens || tokens > capacity ||
        !requests || requests > cache_plan.active_sequences ||
        cache::multiply(tokens, requests) > cache_plan.max_query_rows || verifies_prefix(role))
      throw std::runtime_error("actual suffix admission exceeds the idle model's original query/state bounds");
    // Direct component consumers may not call the explicit startup boundary.
    // Freeze before the first extra shape can mutate the original inventory.
    freeze_shape_plan();
    const auto key = FamilyKey::exact(requests, role, tokens, completion, sampling_index);
    const ExecutionShape shape{requests, tokens, role, completion, sampling_index};
    shape_plan.admit(shape, key,
        [&](const FamilyKey &victim, const ExecutionShape &old_shape) {
          auto old = families.find(victim);
          if (old == families.end() || !old->second->admitted_exact || old->second->chunks.size() != 1 ||
              old_shape != ExecutionShape{victim.requests, old->second->chunks[0], victim.role,
                  victim.completion, victim.sampling_index})
            throw std::runtime_error("exact family inventory is inconsistent");
          for (const bool imported : {false, true}) {
            auto physical = old_shape; physical.imported_sources = imported;
            if (auto phase = phases.find(physical); phase != phases.end()) {
              const auto retired = phase->second->slots.retire_all(execution);
              check(retired.status);
              if (!retired.retired) throw std::runtime_error("exact family carrier has not actually retired");
              phases.erase(phase);
            }
          }
          families.erase(old);
        },
        [&] {
          const std::array chunk{tokens};
          prepare_family(chunk, false, requests, role, true, completion, sampling_index);
        },
        [&]() noexcept { families.erase(key); });
    std::cerr << "cold_exact_admission requests=" << requests << " T=" << tokens
              << " role=" << static_cast<int>(role) << " retained=" << shape_plan.retained()
              << " limit=" << shape_plan.capacity()
              << " seconds=" << std::chrono::duration<double>(Clock::now() - admission_start).count() << '\n';
  }

  void prepare_draft_program(std::size_t rows, bool head, std::size_t sampling_index = 0,
      std::size_t planned_uses = 1, std::size_t steps = 1) {
    preparation_started_ = true;
    if (state_storage_prepared || !cache_plan.draft_cache || !rows || rows > capacity)
      throw std::runtime_error("prepare a positive NextN width before selected cache storage");
    const auto selected_index = head ? sampling_index : 0;
    const auto &selected_sampler = sampling_configuration(selected_index).configuration;
    const DraftProgramKey key{rows, head, selected_index, steps};
    if (draft_programs.contains(key)) return;
    prepare_cache();
    // Optional source transform remains with its original cold Model owner.
    const bool subset = head && prepare_draft_projection(selected_sampler);
    // Repeated proposal heads share this original carrier. Their cold usage
    // bound can select capture; writer-only repair retains its ready-leaf path.
    const auto replay_mode = !selected_sampler.modifiers.history.empty() && head ? g::ReplayMode::kCommands
        : select_replay_mode(replay_policy, head ? planned_uses : 1);
    const auto packet_rows = head && draft_packet_enabled(selected_sampler) ? max_drafts() + 1 : 0;
    auto built = build_draft_program({rows, selected_index, head, replay_mode, packet_rows, steps},
        {geometry, precision, weights, projections, weight_meanings, ffn_placement.cpu_layers()},
        {attention_schema, cache_plan.hidden_slots, cache_plan.active_sequences, cache_plan.candidate_boundaries,
         prepared_draft_projection(subset)}, selected_sampler,
        {execution, allocation_budget, execution_budget, cache_budget});
    auto prepared = std::move(built.program); auto &program = *prepared;
    merge_execution_plan(built.plan);
    input_scratch.draft_positions.reserve(rows + steps - 1);
    program.guard.execution.configure(draft_bank + 2 * steps);
    const auto dirty_bound = cache::add(rows + steps - 1, cache::Plan::page_rows - 2) / cache::Plan::page_rows + 1;
    for (std::size_t bank = draft_bank; bank < draft_bank + 2 * steps; ++bank)
      program.guard.execution.dirty[bank].reserve(dirty_bound);
    if (steps == 1) {
      program.guard.execution.nodes[draft_bank] = program.append.key;
      program.guard.execution.nodes[draft_bank + 1] = program.append.value;
    } else for (std::size_t i = 0; i < steps; ++i) {
      program.guard.execution.nodes[draft_bank + 2 * i] = program.chain[i].append_key;
      program.guard.execution.nodes[draft_bank + 2 * i + 1] = program.chain[i].append_value;
    }
    initialize_draft_program(program);
    draft_programs.emplace(key, std::move(prepared));
  }

  void initialize_draft_program(DraftProgram &program) {
    const std::array sources{g::slots::SourceBinding{program.hidden_state.store, program.hidden_state.gather}};
    inference::ProgramInitialization initialization{
        .io = program.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}};
    if (program.head) initialization.slots.emplace(inference::ProgramInitialization::Slots{
        program.programs.child(0).resource_graph(), sources,
        {.host_workspace = execution.staging(), .resource_budget = execution_budget ? execution_budget : cache_budget}});
    check(program.carrier.initialize(initialization));
  }

  // The shared carrier observes its original native owner. The original Step
  // alone can take back this NextN append/Store cohort before carrier reset.
  hyle::Result<bool> recover_draft_program(DraftProgram &program) noexcept {
    auto &guard = program.guard;
    (void)guard.execution.retire(execution);
    // A completed outgoing packet still owns its original observation. Let
    // the same guard detach those exports before checking carrier retirement;
    // recover() retains all writers whenever native work is actually pending.
    auto recovered = guard.try_abort();
    if (recovered && *recovered)
      for (auto &pages : program.chain_pages) pages = {};
    return recovered;
  }

  // Append to an unpublished draft only. The first append uses COW; later
  // recursive rows may extend that private tail. Published Session tails are
  // never borrowed, so aborting proposal/verification leaves the old head valid.
  bool draft_packet_enabled(const SamplingConfiguration &sampling) const noexcept {
    return draft_packet_available_ && !sampling.enabled && sampling.modifiers.empty();
  }
  bool greedy_draft_chain_enabled(const SamplingConfiguration &sampling) const noexcept {
    return greedy_chain_available_ && !sampling.enabled && sampling.modifiers.empty();
  }

  // Each finite stage has independent COW writers from the same real base.
  // Cumulative KV and the selected hidden/ID remain values in this one Graph;
  // no intermediate page is promoted or lent as a fabricated ready source.
  template<class PrepareVerifier = std::nullptr_t>
  std::vector<std::int32_t> execute_draft_chain(Session &session, std::size_t steps,
      PrepareVerifier &&prepare_verifier = nullptr) {
    const auto &base = *session.head->draft;
    const auto rows = std::size_t{1} + base.lag;
    auto found = draft_programs.find({rows, true, session.sampling_configuration().index, steps});
    if (found == draft_programs.end()) throw std::logic_error("draft chain was not cold prepared");
    auto &program = *found->second; auto &episode = program.guard.execution;
    if (episode.cleanup_pending) {
      if (!take(recover_draft_program(program))) throw std::runtime_error("draft chain is still retiring");
      episode.cleanup_pending = false; episode.terminal = {};
    }
    if (!program.carrier.initialized()) initialize_draft_program(program);
    auto &run = *take(program.carrier.borrow());
    episode.run = &run; episode.carrier = &program.carrier;
    episode.state_transactions = &program.hidden_transactions;
    std::vector<std::int32_t> packet(steps + 1);
    try {
      std::array<g::batch::NodeLayout, 3> layouts;
      std::array<V, 6> cache_inputs;
      std::size_t attention_rows = 0;
      for (std::size_t i = 0; i < steps; ++i) {
        auto &pages = program.chain_pages[i];
        if (!pages.pages.empty()) throw std::logic_error("draft chain retained a preceding private COW candidate");
        if (!pages.frames.try_resize(2)) throw std::bad_alloc{};
        auto &layout = layouts[i].layout;
        const auto width = program.chain[i].rows;
        const auto member = [&](std::size_t) noexcept -> inference::PagedAppendMember {
          return {base.paged, &pages, layout, base.frontier, width};
        };
        const auto bank = draft_bank + 2 * i;
        layout = take(inference::prepare_paged_append(attention_schema, 1,
            inference::PagedAppendBanks{draft_inventory, std::span(episode.dirty.data() + bank, 2),
                std::span(episode.tails.data() + bank, 2), std::span(episode.appends.data() + bank, 2)},
            input_scratch.paged_appends, member,
            {.capacity_rows = capacity, .append_options = {.resource_budget = cache_budget}}));
        layouts[i].node = program.chain[i].attention;
        cache_inputs[2 * i] = program.chain[i].key; cache_inputs[2 * i + 1] = program.chain[i].value;
        attention_rows = cache::add(attention_rows, width);
      }
      std::array<std::int32_t, 2> ids{base.catchup_token, session.head->next_id};
      program.token_values.write<std::int32_t>(std::span(ids).subspan(base.lag ? 0 : 1));
      auto &positions = input_scratch.draft_positions; positions.clear();
      for (std::size_t i = 0; i < rows + steps - 1; ++i)
        positions.push_back(static_cast<std::int32_t>(base.frontier + i));
      program.position_values.write<std::int32_t>(positions);
      auto &transactions = program.hidden_transactions;
      auto sources = transactions.sources(0);
      sources[0] = base.lag ? session.head->prior_hidden : session.head->carry_hidden;
      if (base.lag) sources[1] = session.head->carry_hidden;
      transactions.writers(0)[0] = take(hidden_pool.allocate(transactions.writer_bytes(0), cache::state_pool_alignment));
      check(transactions.stage(0));
      check(episode.prepare_appends(program.graph, std::span(layouts).first(steps), draft_bank, 2 * steps,
          std::span(cache_inputs).first(2 * steps), {.host_workspace = execution.staging()}));
      check(episode.bind_consumers(program.programs.child(0).resource_graph()));
      check(episode.prepare_replay({.mode = program.replay_mode, .host_wait = g::HostWaitPolicy::kReject}));
      // These are newly projected MTP input rows. Cumulative attention work
      // is separately reported; it must not be mistaken for new proposals.
      const auto model_rows = cache::add(submitted_draft_rows, rows + steps - 1);
      const auto session_rows = cache::add(session.submitted_draft_rows, rows + steps - 1);
      const auto actual_attention = cache::add(draft_chain_attention_rows, attention_rows);
      check(episode.submit(resources_));
      submitted_draft_rows = model_rows; session.submitted_draft_rows = session_rows;
      draft_chain_attention_rows = actual_attention;
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<PrepareVerifier>>) {
        episode.sources = std::make_shared<StepSources>();
        const auto *readback = take(program.io.readback(program.verification_value));
        const std::array<inference::SubmissionViews::OutputCapture, 1> outputs{{
            {program.verification_packet, execution.storage_owner(), readback->pool()}}};
        const auto captured = take(episode.sources->capture_outputs(run, episode.completion,
            outputs, execution.storage_owner(), true, false));
        if (captured) {
          // Only shape and the original pending token are host inputs here.
          // Every proposal is supplied by the original complete device packet.
          std::vector<std::int32_t> shape(steps, 0); shape[0] = session.head->next_id;
          prepare_verifier(std::span<const std::int32_t>(shape), episode.sources, false);
          check(episode.wait()); check(episode.sources->detach_outputs());
          prepare_verifier(std::span<const std::int32_t>(shape), episode.sources, true);
        } else episode.sources.reset();
      }
      check(episode.finish_success(draft_bank, 2 * steps, take(program.io.withdrawal_values()),
          [&]() noexcept { ++draft_program_graph_calls; ++draft_chain_graph_calls; },
          [&]() -> hyle::Status {
            check(program.io.read(run, program.verification_value, std::as_writable_bytes(std::span(packet))));
            if (packet[0] != session.head->next_id || std::ranges::any_of(packet,
                [&](auto id) { return id < 0 || id >= vocabulary; }))
              throw std::runtime_error("draft chain packet differs from its original vocabulary/pending token");
            return hyle::Status::Ok();
          }, [&]() -> hyle::Status {
            // Approximate hidden and every private COW page are intentionally
            // discarded after real success/take; only target repair can commit.
            check(program.hidden_transactions.clear(0));
            return hyle::Status::Ok();
          }));
      for (auto &append : episode.appends) append = {};
      for (auto &bank : episode.dirty) bank.clear();
      for (auto &bank : episode.tails) bank.clear();
      for (auto &pages : program.chain_pages) pages = {};
      episode.sources.reset();
      episode.run = nullptr; episode.carrier = nullptr; episode.state_transactions = nullptr;
      return packet;
    } catch (inference::EvaluationFailure &failure) {
      if (active_step && active_step->execution.incoming && active_step->execution.incoming->owner == episode.sources)
        (void)abort_episode();
      episode.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (!recovered) inference::record_retirement_status(episode.terminal, recovered.error());
      failure.observe_retirement({recovered && *recovered, episode.terminal});
      if (recovered && *recovered) { episode.cleanup_pending = false; episode.terminal = {}; }
      throw;
    } catch (...) {
      if (active_step && active_step->execution.incoming && active_step->execution.incoming->owner == episode.sources)
        (void)abort_episode();
      episode.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (recovered && *recovered) { episode.cleanup_pending = false; episode.terminal = {}; }
      throw;
    }
  }

  template<class PrepareVerifier = std::nullptr_t>
  DraftProposal execute_draft(DraftWork &work, std::span<const std::int32_t> ids,
      std::span<const c::ReadLease> shifted, bool head, const std::array<std::uint32_t, 4> &identity = {},
      std::size_t sampling_index = 0, const LogitsInputs &logits = {},
      std::span<const std::int32_t> verification_prefix = {}, PrepareVerifier &&prepare_verifier = nullptr) {
    if (ids.empty() || ids.size() != shifted.size() || !state_storage_prepared ||
        work.state.frontier > capacity || ids.size() > capacity - work.state.frontier)
      throw std::runtime_error("NextN input rows exceed the prepared context or hidden geometry");
    auto found = draft_programs.find({ids.size(), head, head ? sampling_index : 0, 1});
    if (found == draft_programs.end()) throw std::runtime_error("NextN finite program was not cold prepared");
    auto &program = *found->second;
    const auto &selected_sampler = sampling_configuration(program.sampling_index).configuration;
    if (head) validate_logits_inputs(selected_sampler, logits, vocabulary);
    auto &guard = program.guard;
    if (guard.execution.cleanup_pending) {
      // Keep the same original failed episode and all its append/state facts
      // until real retirement; no new carrier can reuse these permissions.
      if (!take(recover_draft_program(program)))
        throw std::runtime_error("NextN original episode is still retiring");
      guard.execution.cleanup_pending = false; guard.execution.terminal = {};
    }
    if (!program.carrier.initialized()) initialize_draft_program(program);
    auto &run = *take(program.carrier.borrow());
    guard.execution.run = &run; guard.execution.carrier = &program.carrier;
    if (head) guard.execution.state_transactions = &program.hidden_transactions;
    cache::DraftState next;
    DraftProposal output;
    try {
      const auto length = work.state.frontier;
      next.frontier = length + ids.size();
      if (!next.paged.frames.try_resize(work.state.paged.frames.size())) throw std::bad_alloc{};
      std::array<c::PagedArenaCommit, 2> commits;
      inference::PagedSequenceEdit edit(work.state.paged, next.paged, commits);
      c::BatchLayoutVersion layout;
      const auto member_at = [&](std::size_t) noexcept -> inference::PagedAppendMember {
        return {work.state.paged, &next.paged, layout, length, ids.size(), work.tails, true};
      };
      layout = take(inference::prepare_paged_append(attention_schema, 1,
          inference::PagedAppendBanks{draft_inventory,
              std::span(guard.execution.dirty.data() + draft_bank, 2),
              std::span(guard.execution.tails.data() + draft_bank, 2),
              std::span(guard.execution.appends.data() + draft_bank, 2)},
          input_scratch.paged_appends, member_at,
          {.capacity_rows = capacity, .append_options = {.resource_budget = cache_budget}}));
      auto &positions = input_scratch.draft_positions;
      positions.clear();
      for (std::size_t i = 0; i < ids.size(); ++i) positions.push_back(static_cast<std::int32_t>(length + i));
      program.token_values.write<std::int32_t>(ids); program.position_values.write<std::int32_t>(positions);
      if (!verification_prefix.empty()) {
        if (!program.verification_packet || verification_prefix.size() !=
            static_cast<std::size_t>(program.verification_prefix_values.description.dimensions()[0]))
          throw std::runtime_error("device verifier prefix differs from its original cold packet");
        program.verification_prefix_values.write<std::int32_t>(verification_prefix);
      }
      if (head) {
        auto &transactions = program.hidden_transactions;
        std::copy(shifted.begin(), shifted.end(), transactions.sources(0).begin());
        transactions.writers(0)[0] = take(hidden_pool.allocate(transactions.writer_bytes(0), cache::state_pool_alignment));
        check(transactions.stage(0));
      } else for (std::size_t row = 0; row < ids.size(); ++row)
        check(program.io.bind(run, program.hidden_bindings[row], shifted[row]));
      if (program.bias_input) check(program.io.bind(run, program.bias_input, logits.bias.description, logits.bias.lease));
      if (program.allowed_input) check(program.io.bind(run, program.allowed_input, logits.allowed.description, logits.allowed.lease));
      if (program.counts_input) check(program.io.bind(run, program.counts_input, logits.counts.description, logits.counts.lease));
      if (head && selected_sampler.enabled) {
        program.identity_values.write<std::uint32_t>(identity);
        check(program.io.bind_output(run, program.probabilities_output, program.probability_pool));
      }
      if (program.next_counts) check(program.io.bind_output(run, program.next_counts_output, program.next_counts_pool));
      const std::array layouts{g::batch::NodeLayout{program.attention, layout}};
      const std::array cache_inputs{head ? program.key : V{}, head ? program.value : V{}};
      check(guard.execution.prepare_appends(program.graph,
          head ? std::span<const g::batch::NodeLayout>(layouts) : std::span<const g::batch::NodeLayout>{},
          draft_bank, 2, cache_inputs, {.host_workspace = execution.staging()}));
      check(guard.execution.bind_consumers(program.programs.child(0).resource_graph()));
      check(guard.execution.prepare_replay({.mode = program.replay_mode, .host_wait = g::HostWaitPolicy::kAllow}));
      const auto model_rows_after = cache::add(submitted_draft_rows, program.rows);
      const auto session_rows_after = work.session
          ? cache::add(work.session->submitted_draft_rows, program.rows) : 0;
      check(guard.execution.submit(resources_));
      submitted_draft_rows = model_rows_after;
      if (work.session) work.session->submitted_draft_rows = session_rows_after;
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<PrepareVerifier>>) {
        if (!verification_prefix.empty()) {
          guard.execution.sources = std::make_shared<StepSources>();
          const auto *packet = take(program.io.readback(program.verification_value));
          const std::array<inference::SubmissionViews::OutputCapture, 1> outputs{{
              {program.verification_packet, execution.storage_owner(), packet->pool()}}};
          // Export the existing device Source while pending. Final capture
          // preparation follows whole producer success, without forcing the
          // target back to managed Commands.
          const auto captured = take(guard.execution.sources->capture_outputs(run, guard.execution.completion,
              outputs, execution.storage_owner(), true, false));
          if (captured) {
            prepare_verifier(guard.execution.sources, false);
            check(guard.execution.wait());
            check(guard.execution.sources->detach_outputs());
            // Keep exports open until the target binds its ready packet. Only
            // the final source/replay admission waits; batch/state prep overlapped.
            prepare_verifier(guard.execution.sources, true);
          } else guard.execution.sources.reset(); // Typed capability refusal preserves the original submit proxy.
        }
      }
      std::array<c::PagedPageAppender, 2> tails;
      check(guard.execution.finish_success(draft_bank, 2, take(program.io.withdrawal_values()),
          [&]() noexcept { ++draft_program_graph_calls; },
          [&]() -> hyle::Status {
            if (head) {
              if (program.valid) {
                const auto flag = take(program.io.predicate(run, program.valid_value));
                if (flag == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
                    "recursive draft distribution is invalid; no head or RNG was published"));
                if (flag != 1) throw std::runtime_error("recursive draft validity is outside its predicate encoding");
                if (selected_sampler.enabled) output.log_probabilities =
                    take(program.io.retain(run, program.probabilities_output, execution_budget));
              }
              check(program.io.read(run, program.id_value, std::as_writable_bytes(std::span(&output.id, 1))));
              if (output.id < 0 || output.id >= vocabulary) throw std::runtime_error("NextN proposal is outside vocabulary");
              if (program.next_counts) output.next_counts =
                  take(program.io.retain(run, program.next_counts_output, execution_budget));
            }
            return hyle::Status::Ok();
          },
          [&]() -> hyle::Status {
            if (head) {
              auto writers = take(program.hidden_transactions.returned(0));
              output.hidden = take(writers[0].publish());
              check(program.hidden_transactions.clear(0));
            }
            const auto finish_at = [&](std::size_t) noexcept -> inference::PagedAppendFinishMember {
              return {work.state.paged, next.paged, commits, layout, next.paged.pages.back().page(),
                  static_cast<std::uint32_t>(next.frontier % cache::Plan::page_rows), tails};
            };
            check(inference::prepare_paged_append_commits(
                std::span(guard.execution.appends.data() + draft_bank, 2), 1, finish_at));
            check(inference::return_paged_append_tails(
                guard.execution.append_transactions().subspan(draft_bank, 2), 2, 1,
                input_scratch.paged_appends, finish_at));
            return hyle::Status::Ok();
          }));
      auto publication = take(edit.prepare_publication());
      publication.publish(&work.state.paged);
      work.tails = std::move(tails); work.state = std::move(next);
      guard.execution.sources.reset(); // The admitted target retains its original view.
      for (auto &append : guard.execution.appends) append = {};
      for (auto &bank : guard.execution.dirty) bank.clear();
      for (auto &bank : guard.execution.tails) bank.clear();
      guard.execution.run = nullptr; guard.execution.carrier = nullptr; guard.execution.state_transactions = nullptr;
      return output;
    } catch (inference::EvaluationFailure &failure) {
      if (active_step && active_step->execution.incoming &&
          active_step->execution.incoming->owner == guard.execution.sources)
        (void)abort_episode();
      // This validator has no target Step participant. Keep its original
      // private guard's actual retirement observation for a known caller key.
      guard.execution.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (!recovered) inference::record_retirement_status(guard.execution.terminal, recovered.error());
      failure.observe_retirement({recovered && *recovered, guard.execution.terminal});
      if (recovered && *recovered) { guard.execution.cleanup_pending = false; guard.execution.terminal = {}; }
      throw;
    } catch (...) {
      if (active_step && active_step->execution.incoming &&
          active_step->execution.incoming->owner == guard.execution.sources)
        (void)abort_episode();
      // An unsuccessful private append is never a partially published head.
      guard.execution.cleanup_pending = true;
      const auto recovered = recover_draft_program(program);
      if (recovered && *recovered) { guard.execution.cleanup_pending = false; guard.execution.terminal = {}; }
      // Otherwise the original cold guard/run retain the partial cleanup facts;
      // the next use must finish this same retirement before it can initialize.
      throw;
    }
  }

  void prepare_prefix_sampling(std::size_t proposals, std::size_t sampling_index = 0, bool conditional = false) {
    preparation_started_ = true;
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    if ((!selected_sampler.enabled && selected_sampler.modifiers.empty()) || state_storage_prepared || proposals >= cache_plan.candidate_boundaries)
      throw std::runtime_error("prepare bounded speculative-sampling widths before cache storage");
    if (conditional && !selected_sampler.modifiers.bias && !selected_sampler.modifiers.allowed)
      throw std::runtime_error("conditional prefix inputs require configured bias or mask ports");
    const PrefixSamplingKey key{proposals, sampling_index, conditional};
    if (prefix_sampling_programs.contains(key)) return;
    auto built = build_prefix_sampling({proposals, sampling_index, conditional},
        {geometry, precision, weights, projections, weight_meanings, ffn_placement.cpu_layers()}, selected_sampler,
        {execution, allocation_budget, execution_budget, cache_budget});
    auto prepared = std::move(built.program); auto &program = *prepared;
    merge_execution_plan(built.plan);
    if (program.identities)
      input_scratch.sampling_identities.reserve(take(program.identity_values.description.required_bytes()) / sizeof(std::uint32_t));
    initialize_prefix_sampling(program);
    prefix_sampling_programs.emplace(key, std::move(prepared));
  }

  void initialize_prefix_sampling(PrefixSamplingProgram &program) {
    check(program.carrier.initialize({
        .io = program.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}}));
  }

  // Pure private consumers have no append/Store transaction to recover. A
  // previously closed run can be replaced only after its real owner retires.
  void recover_private_carrier(inference::ProgramCarrier<ColdResources> &carrier) {
    if (!carrier.closing()) return;
    const auto observed = carrier.retire(execution);
    if (!observed.retired) {
      check(observed.status);
      throw std::runtime_error("original private program carrier is still retiring");
    }
    check(carrier.reset_retired());
    check(observed.status);
  }

  // Target family widths and replay counts stay with the finite request plan;
  // every recursive consumer shares this model's actual NextN/sampler programs.
  void prepare_recursive_round_programs(std::size_t sampling_index = 0, bool conditional = false,
      std::size_t planned_uses = 1, std::size_t maximum_proposals = SIZE_MAX) {
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    const auto maximum_rows = maximum_proposals == SIZE_MAX ? cache_plan.candidate_boundaries
        : (std::min)(maximum_proposals, max_drafts()) + 1;
    maximum_proposals = (std::min)(maximum_proposals, max_drafts());
    if (!maximum_proposals && max_drafts())
      throw std::logic_error("recursive programs require a positive proposal limit for active drafting");
    for (std::size_t rows = 1; rows <= maximum_rows; ++rows) {
      prepare_draft_program(rows, false);
      if (selected_sampler.enabled || !selected_sampler.modifiers.empty())
        prepare_prefix_sampling(rows - 1, sampling_index, conditional);
    }
    prepare_draft_program(1, true, sampling_index, planned_uses);
    if (capacity > 1) prepare_draft_program(2, true, sampling_index, planned_uses);
    if (!conditional && greedy_draft_chain_enabled(selected_sampler))
      for (std::size_t steps = 2; steps <= maximum_proposals; ++steps)
        for (std::size_t rows = 1; rows <= 2; ++rows)
          prepare_draft_program(rows, true, sampling_index, planned_uses, steps);
  }

  PrefixSamplingResult sample_recursive_prefix(c::ConstTensorView logits,
      std::span<const std::int32_t> proposals, std::span<const c::ReadLease> q,
      cache::SamplingCursor cursor, std::size_t quota, std::span<const std::int32_t> eos,
      const inference::RequestStopPolicy *stop_policy, std::size_t sampling_index = 0, const LogitsInputs &inputs = {},
      std::span<const LogitsInputs> conditional_inputs = {}) {
    const auto &selected_sampler = sampling_configuration(sampling_index).configuration;
    if ((selected_sampler.enabled ? proposals.size() != q.size() : !q.empty() || selected_sampler.modifiers.empty()))
      throw std::runtime_error("speculative sampling requires its actual proposal distributions or greedy modifiers");
    if (!quota) throw std::runtime_error("speculative prefix requires a positive output quota");
    inference::mtp::validate_token_ids(proposals, static_cast<std::int32_t>(vocabulary));
    inference::request_stop_policy(stop_policy).validate(vocabulary);
    validate_prefix_logits_inputs(selected_sampler, inputs, conditional_inputs, cache::add(proposals.size(), 1), vocabulary);
    auto found = prefix_sampling_programs.find({proposals.size(), sampling_index, !conditional_inputs.empty()});
    if (found == prefix_sampling_programs.end()) throw std::runtime_error("speculative-sampling width was not cold prepared");
    auto &program = *found->second;
    recover_private_carrier(program.carrier);
    if (!program.carrier.initialized()) initialize_prefix_sampling(program);
    auto &run = *take(program.carrier.borrow());
    const auto count = proposals.size();
    if (selected_sampler.enabled) {
      (void)cursor.advanced(cache::add(cache::multiply(2, count), 1));
      auto &identities = input_scratch.sampling_identities;
      identities.clear();
      for (std::size_t row = 0; row <= count; ++row) {
        const auto identity = cursor.identity(row == count ? 2 * count : 2 * row + 1);
        identities.insert(identities.end(), identity.begin(), identity.end());
      }
      program.identity_values.write<std::uint32_t>(identities);
    }
    if (program.ids) program.id_values.write<std::int32_t>(proposals);
    if (program.continuation) {
      for (std::size_t row = 0; row < count; ++row)
        program.continuation_host[row] = row < quota - 1 &&
            !inference::request_token_stopped(proposals[row], eos, stop_policy) ? 1.F : 0.F;
      program.continuation_values.write<float>(program.continuation_host);
    }
    try {
      check(program.io.bind(run, program.logits_input, logits));
      if (program.bias_input) check(program.io.bind(run, program.bias_input, inputs.bias.description, inputs.bias.lease));
      if (program.allowed_input) check(program.io.bind(run, program.allowed_input, inputs.allowed.description, inputs.allowed.lease));
      if (program.counts_input) check(program.io.bind(run, program.counts_input, inputs.counts.description, inputs.counts.lease));
      for (std::size_t row = 0; row < program.conditional_inputs.size(); ++row) {
        const auto &ports = program.conditional_inputs[row]; const auto &values = conditional_inputs[row];
        if (ports.bias_input) check(program.io.bind(run, ports.bias_input, values.bias.description, values.bias.lease));
        if (ports.allowed_input) check(program.io.bind(run, ports.allowed_input, values.allowed.description, values.allowed.lease));
      }
      for (std::size_t row = 0; row < q.size(); ++row)
        check(program.io.bind(run, program.q_inputs[row], q[row]));
      const auto queues = program.carrier.queues();
      check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      auto done = take(program.carrier.submit(resources_)); check(done.wait());
      ++prefix_decision_graph_calls;
      check(program.io.read(run, program.packet_value,
          std::as_writable_bytes(std::span(program.packet_host))));
      const auto words = std::span<const std::int32_t>(program.packet_host);
      if (words[0] == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
          "recursive target/q/residual distribution is invalid; no head or RNG was published"));
      if (words[0] != 1) throw std::runtime_error("recursive target validity is outside its predicate encoding");
      PrefixSamplingResult result;
      if (selected_sampler.enabled) {
        const auto consumed = words[1];
        const auto emitted = words.subspan(4);
        inference::mtp::validate_token_ids(emitted, static_cast<std::int32_t>(vocabulary));
        if (consumed <= 0 || static_cast<std::size_t>(consumed) > (std::min)(quota, count + 1) ||
            (words[3] != 0 && words[3] != 1) || words[2] != emitted[static_cast<std::size_t>(consumed) - 1])
          throw std::runtime_error("device prefix result differs from its bounded emitted rows");
        result.prefix = {.consumed = static_cast<std::size_t>(consumed),
            .consumed_drafts = static_cast<std::size_t>(consumed) - 1, .next_id = words[2],
            .first_agrees = words[3] != 0,
            .stopped = inference::request_token_stopped(words[2], eos, stop_policy)};
        result.final_ids.assign(emitted.begin(), emitted.begin() + consumed);
      } else {
        const auto ids = words.subspan(1);
        inference::mtp::validate_token_ids(ids, static_cast<std::int32_t>(vocabulary));
        result.final_ids.assign(ids.begin(), ids.end());
      }
      check(program.io.release_inputs(program.carrier));
      return result;
    } catch (inference::EvaluationFailure &failure) {
      const auto observed = program.carrier.retire(execution);
      failure.observe_retirement(observed);
      if (observed.retired) program.carrier.reset_retired().IgnoreError();
      throw;
    } catch (...) {
      // The outer target episode retains its original member proof and drives
      // actual retirement; this finite private consumer releases only its pins.
      const auto observed = program.carrier.retire(execution);
      if (observed.retired) program.carrier.reset_retired().IgnoreError();
      throw;
    }
  }

  static c::PagedArenaFrame working_frame(c::PagedArenaRead read) {
    return take(c::PagedArenaFrame::fork(take(c::PagedArenaFootprint::from_read(read))));
  }
  static cache::DraftState fork_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.paged = take(source.paged.fork()); next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    return next;
  }
  static cache::DraftState alias_draft(const cache::DraftState &source) {
    cache::DraftState next;
    next.paged = take(source.paged.frozen_alias()); next.frontier = source.frontier;
    next.lag = source.lag; next.catchup_token = source.catchup_token;
    return next;
  }

  std::unique_ptr<cache::State> reserve_state(cache::WorkingState &owner) {
    auto version = take(cache::VersionRef::acquire(snapshot_versions));
    auto result = owner.acquire_candidate();
    result->resource_owner = resources_;
    if (!result->paged.frames.try_resize(target_kv_layers * 2) ||
        !result->state.try_resize(geometry.layers.size())) throw std::bad_alloc{};
    result->version = std::move(version); result->model = this; result->math = math_identity();
    return result;
  }

  // A request's pending token/cursor is not part of the reusable model state.
  // Wrapper origins are canonical so aliases share exactly one finite version;
  // mutable page-tail capabilities stay with the original Session only.
  std::shared_ptr<cache::State> sampling_fork(cache::SnapshotRef base,
      cache::SamplingCursor cursor, std::int32_t pending) const {
    if (!base || base->model != this || base->math != math_identity())
      throw std::runtime_error("sampling fork belongs to another model or compiled policy");
    auto next = std::make_shared<cache::State>();
    next->state_origin = base->state_origin ? base->state_origin : base;
    next->paged = take(base->paged.frozen_alias());
    next->state = base->state; next->carry_hidden = base->carry_hidden;
    next->prior_hidden = base->prior_hidden;
    if (base->draft) next->draft.emplace(alias_draft(*base->draft));
    next->model = base->model; next->math = base->math; next->length = base->length;
    next->next_id = pending; next->sampling = cursor; return next;
  }

  void prepare_sampling_frontier() {
    preparation_started_ = true;
    for (auto &entry : resources_->sampling_configurations) {
      const auto &selected_sampler = entry->configuration;
      if (!selected_sampler.enabled && selected_sampler.modifiers.empty() && sampling_configuration_count() == 1) continue;
      if (!entry->frontier) {
        auto built = build_sampling_frontier(
            {geometry, precision, weights, projections, weight_meanings, ffn_placement.cpu_layers()}, selected_sampler,
            {execution, allocation_budget, execution_budget, cache_budget});
        merge_execution_plan(built.plan);
        auto prepared = std::move(built.program);
        entry->frontier = std::move(prepared);
      }
    }
  }

  cache::SnapshotRef sample_prefix(cache::SnapshotRef base, cache::SamplingCursor cursor, std::size_t sampling_index = 0,
                                   const LogitsInputs &inputs = {}) {
    const auto &selected = sampling_configuration(sampling_index);
    const auto &selected_sampler = selected.configuration;
    validate_logits_inputs(selected_sampler, inputs, vocabulary);
    if (!base || !base->length || !base->carry_hidden.valid() || active_step || carrier_borrowed || !selected.frontier)
      throw std::runtime_error("prefix sampling requires its cold-prepared graph and an idle complete model-state hidden frontier");
    const auto next_cursor = selected_sampler.enabled ? cursor.advanced(1) : cache::SamplingCursor{};
    const auto started = Clock::now();
    auto &frontier = *selected.frontier;
    recover_private_carrier(frontier.carrier);
    // This is the same Graph execution API/runtime as target/draft. Keep the
    // original fresh-run frontier policy, with atomic fixed-IO installation.
    check(frontier.carrier.initialize({
        .io = frontier.io,
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}}));
    auto &run = *take(frontier.carrier.borrow());
    try {
      if (selected_sampler.enabled) { const auto identity = cursor.identity(); frontier.identity.write<std::uint32_t>(identity); }
      check(frontier.io.bind(run, frontier.hidden_input, base->carry_hidden));
      if (frontier.bias_input) check(frontier.io.bind(run, frontier.bias_input, inputs.bias.description, inputs.bias.lease));
      if (frontier.allowed_input) check(frontier.io.bind(run, frontier.allowed_input, inputs.allowed.description, inputs.allowed.lease));
      if (frontier.counts_input) check(frontier.io.bind(run, frontier.counts_input, inputs.counts.description, inputs.counts.lease));
      const auto queues = frontier.carrier.queues();
      check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      auto done = take(frontier.carrier.submit(resources_)); check(done.wait());
      ++frontier_sampling_graph_calls;
      const auto valid = frontier.valid_port ? take(frontier.io.predicate(run, frontier.valid)) : std::uint8_t{1};
      std::int32_t pending = -1;
      check(frontier.io.read(run, frontier.id, std::as_writable_bytes(std::span(&pending, 1))));
      if (valid == 0) throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
          "cached frontier sampling is invalid; no request RNG cursor was published"));
      if (valid != 1 || pending < 0 || pending >= vocabulary)
        throw std::runtime_error("cached frontier sampling is outside its declared result domain");
      auto next = sampling_fork(std::move(base), next_cursor, pending);
      frontier_sampling_seconds += std::chrono::duration<double>(Clock::now() - started).count();
      check(frontier.carrier.reset_retired());
      return next;
    } catch (inference::EvaluationFailure &failure) {
      const auto observed = frontier.carrier.retire(execution);
      failure.observe_retirement(observed);
      if (observed.retired) frontier.carrier.reset_retired().IgnoreError();
      throw;
    } catch (...) {
      const auto observed = frontier.carrier.retire(execution);
      if (observed.retired) frontier.carrier.reset_retired().IgnoreError();
      throw;
    }
  }

  const void *math_identity() const noexcept {
    // Every family uses this Model's source-derived score boundary. These identities also
    // separate probe math; no prefix can cross a Model or its compiled policy.
    return layer0_probe.empty() ? static_cast<const void *>(&families) : static_cast<const void *>(&probe_families);
  }

  // Aggregate ordered call plans before any execution carriers are created.
  // Prefix adoption can skip full-hit calls. Counts describe actual possible
  // uses of the serial slot, not additional execution carriers.
  void plan_execution(std::span<const std::size_t> calls, std::size_t requests = 1, GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    for (const auto tokens : calls) {
      if (!find_child(requests, tokens, role, false, completion, sampling_index))
        throw std::runtime_error("execution plan chunk is outside the shared finite family");
      shape_plan.add_use({requests, tokens, role, completion, sampling_index}, slot_index);
    }
  }

  // Finite wave/suffix plans can count many possible restores without
  // materializing repeated host call tapes or execution carriers.
  void plan_replay_uses(std::size_t tokens, std::size_t requests, std::size_t count,
                        GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                        RowPacking packing = RowPacking::kRectangular) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    if (cold_family_pool_) {
      cold_uses_.push_back({tokens, requests, count, role, completion, sampling_index, packing});
      return;
    }
    if (!find_child(requests, tokens, role, false, completion, sampling_index, packing))
      throw std::runtime_error("execution use shape is outside the shared finite family");
    shape_plan.add_uses({requests, tokens, role, completion, sampling_index, false, packing}, count);
  }

  std::shared_ptr<Session> create_session(std::span<const std::size_t> planned_chunks,
                                        cache::SnapshotRef prefix = {},
                                        GraphRole role = GraphRole::kText,
                                        std::optional<cache::SamplingCursor> cursor = {},
                                        bool resample_frontier = false,
                                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                                         const LogitsInputs &logits = {}) {
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (resample_frontier) validate_logits_inputs(selected_sampler, logits, vocabulary);
    prepare_state_storage();
    const auto initial_rows = planned_chunks.empty() ? 0 : planned_chunks.front();
    const auto prepared_rows = [&](std::size_t rows) {
      return find_child(1, rows, role, false, completion, sampling_index) ||
          ((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(rows, completion, sampling_index, role));
    };
    for (const auto rows : planned_chunks)
      if (!prepared_rows(rows)) admit_exact_shape(rows, 1, role, completion, sampling_index);
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("session must use its complete target/draft cache family");
    const bool prepared_adoption = families.contains({1, role, 0, completion, sampling_index}) ||
        ((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(1, completion, sampling_index, role));
    if ((planned_chunks.empty() ? !prepared_adoption : !prepared_rows(initial_rows)) ||
        (!layer0_probe.empty() && (planned_chunks.empty()
            ? !probe_families.contains({1, role, 0, completion, sampling_index})
            : !find_child(1, initial_rows, role, true, completion, sampling_index))))
      throw std::runtime_error("session requires the shared text families first");
    if (planned_chunks.empty() && !prefix) throw std::runtime_error("session requires an ordered call plan or a complete prefix");
    if (prefix && (prefix->model != this || prefix->math != math_identity() || prefix->length > capacity ||
        (planned_chunks.empty() && (!prefix->length || prefix->next_id < 0))))
      throw std::runtime_error("prefix belongs to another model or context");
    if (prefix) validate_mtp_snapshot(*prefix);
    auto session = std::shared_ptr<Session>(new Session(*this, resources_, prepared_sampling));
    std::erase_if(session_links, [](const auto &weak) { return weak.expired(); });
    session_links.push_back(session);
    if (!session->tails.try_resize(draft_bank)) throw std::bad_alloc{};
    std::size_t total = prefix ? prefix->length : 0;
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity - total)
        throw std::runtime_error("session plan exceeds nonempty context capacity");
      if (!layer0_probe.empty() && !find_child(1, tokens, role, true, completion, sampling_index))
        throw std::runtime_error("session chunk is outside the shared diagnostic family");
      total += tokens;
      if (!shape_plan.contains({1, tokens, role, completion, sampling_index}) &&
          !((role == GraphRole::kText || is_mtp_prefill(role)) && planned_prefill_member(tokens, completion, sampling_index, role)))
        throw std::runtime_error("session chunk has no model execution use plan");
    }
    if (prefix) {
      if (resample_frontier && (selected_sampler.enabled || !selected_sampler.modifiers.empty() || sampling_configuration_count() > 1))
        prefix = sample_prefix(std::move(prefix), cursor.value_or(cache::SamplingCursor{}), sampling_index, logits);
      else if ((selected_sampler.enabled || !selected_sampler.modifiers.empty()) && cursor)
        prefix = sampling_fork(std::move(prefix), selected_sampler.enabled ? *cursor : cache::SamplingCursor{}, -1);
      else if (resample_frontier) {
        // Recovery wrappers can preserve a historical pending token. A new
        // greedy request starts from the immutable evaluated state's result.
        const auto fresh = prefix->state_identity()->next_id;
        if (fresh < 0 || fresh >= vocabulary)
          throw std::runtime_error("cached evaluated frontier has no valid greedy result");
        prefix = sampling_fork(std::move(prefix), {}, fresh);
      } else if (cursor && sampling_configuration_count() > 1) {
        // A new greedy request can adopt consumed state from a random request.
        // The old sampled pending/cursor is not this request's boundary.
        prefix = sampling_fork(std::move(prefix), {}, -1);
      }
      session->head.adopt(std::move(prefix)); // Exact evaluated-state fork; no KV/state payload copy.
    }
    else {
      auto initial = reserve_state(session->head);
      initial->sampling = selected_sampler.enabled ? cursor.value_or(cache::SamplingCursor{sampling_seed, 0, 0})
          : cache::SamplingCursor{};
      const std::array<std::int32_t, 2> empty{0, 0};
      const std::array<std::int32_t, 1> tail{0};
      const auto layout = take(attention_schema.version({empty, empty, {}, tail}));
      for (std::size_t i = 0; i < initial->paged.frames.size() / 2; ++i) {
        initial->paged.frames[2 * i] = working_frame(take(page_inventory.banks[2 * i].read(layout, {})));
        initial->paged.frames[2 * i + 1] = working_frame(take(page_inventory.banks[2 * i + 1].read(layout, {})));
      }
      if (cache_plan.draft_cache) {
        initial->draft.emplace();
        if (!initial->draft->paged.frames.try_resize(2)) throw std::bad_alloc{};
        initial->draft->paged.frames[0] = working_frame(take(draft_inventory.banks[0].read(layout, {})));
        initial->draft->paged.frames[1] = working_frame(take(draft_inventory.banks[1].read(layout, {})));
        auto zero = take(hidden_pool.allocate(cache_plan.hidden_row_bytes, cache::state_pool_alignment));
        check(zero.view()->fill(std::byte{})); initial->carry_hidden = take(zero.publish());
      }
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
        auto history = take(state_pools[i].acquire_history(take(state_schema[i].history.required_bytes())));
        auto recurrent = take(state_pools[i].acquire_recurrent(take(state_schema[i].recurrent.required_bytes())));
        check(history.view()->fill(std::byte{})); check(recurrent.view()->fill(std::byte{}));
        initial->state[i].history = take(history.publish());
        initial->state[i].recurrent = take(recurrent.publish());
      }
      session->head.commit(std::move(initial));
    }
    // State admission does not allocate execution carriers. The first actual
    // explicit prepare creates its needed Phase and slot; idle/full-hit cached
    // Sessions never duplicate IO, workspace or native execution preparation.
    return session;
  }

  void restore_pending_boundary(Session &session, cache::SamplingCursor cursor, std::int32_t pending) const {
    if (session.pending || active_step || carrier_borrowed || pending < -1 || pending >= vocabulary)
      throw std::runtime_error("sampling recovery requires an idle committed boundary and historical pending token");
    session.head.preflight_commit();
    auto replacement = sampling_fork(session.checkpoint(), cursor, pending);
    for (auto &tail : session.tails) tail = {};
    session.draft_tails = {};
    session.tail_epoch = 0; session.draft_tail_epoch = 0;
    session.head.adopt(std::move(replacement));
  }

  // Complete same-model cold parking. No token replay, sampling or native
  // state publication occurs until the original byte transfer succeeds.
  cache::HostImageSchema host_image_schema() const;
  cache::HostImageIdentity host_image_identity(std::size_t sampling_index) const;
  std::shared_ptr<const cache::HostImage> export_session(Session &session,
      const std::shared_ptr<inference::HostStateStorage> &storage);
  std::shared_ptr<Session> restore_session(std::shared_ptr<const cache::HostImage> image);

  void validate_mtp_snapshot(const cache::State &head, const StepMember *candidate = nullptr) const {
    if (cache_plan.draft_cache != head.draft.has_value())
      throw std::runtime_error("snapshot does not contain the configured complete MTP state");
    if (!head.draft) return;
    const auto &draft = *head.draft;
    if (draft.paged.frames.size() != 2)
      throw std::runtime_error("snapshot does not contain its complete draft bank cohort");
    const auto key = candidate && candidate->draft_updates[0].valid()
        ? take(candidate->draft_updates[0].view()) : take(draft.paged.frames[0].view());
    const auto value = candidate && candidate->draft_updates[1].valid()
        ? take(candidate->draft_updates[1].view()) : take(draft.paged.frames[1].view());
    if (draft.lag > 1 || head.length < draft.lag || draft.frontier != head.length - draft.lag ||
        !key.valid() || !value.valid() || key.layout().facts().kv_rows != draft.frontier ||
        value.layout().facts().kv_rows != draft.frontier || !head.carry_hidden.valid() ||
        (draft.lag && (!head.prior_hidden.valid() || draft.catchup_token < 0 || draft.catchup_token >= vocabulary)) ||
        (!draft.lag && draft.catchup_token != -1))
      throw std::runtime_error("inconsistent MTP frontier, catch-up token or owned target hidden state");
  }

  Phase::ExecutionSlots::Initialization slot_initialization(const Phase &phase) const {
    Phase::ExecutionSlots::Initialization initialization{
        .invocation = {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}},
        .progress = {scheduler.get(), workers.get()}};
    if (execution_workspace.valid()) initialization.workspace = &execution_workspace;
    else if (bounded_prefill_rows && packs_prefill(phase.family->packing))
      throw std::logic_error("bounded prefill requires its qualified shared workspace before slot initialization");
    if (!phase.state_banks.empty()) initialization.slots.emplace(inference::ProgramInitialization::Slots{
        phase.state_graph(), phase.state_sources,
        {.host_workspace = execution.staging(), .resource_budget = execution_budget ? execution_budget : cache_budget}});
    return initialization;
  }

  void initialize_slot(Phase &phase, std::size_t slot) {
    check(phase.slots.initialize_with(slot, [&] { return slot_initialization(phase); }));
  }

  Phase &prepare_child(std::size_t tokens, bool diagnostic, std::size_t requests = 1, GraphRole role = GraphRole::kText,
                       TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
                       bool imported_sources = false, RowPacking packing = RowPacking::kRectangular) {
    auto &cache = diagnostic ? probe_phases : phases;
    const auto found = cache.find({requests, tokens, role, completion, sampling_index, imported_sources, packing});
    if (found != cache.end()) return *found->second;
    const auto selected = find_child(requests, tokens, role, diagnostic, completion, sampling_index, packing);
    if (!selected) throw std::runtime_error("text finite family has not been prepared");
    auto &definition = selected->family;
    const auto &selected_sampler = definition.sampling->configuration;
    const auto start = Clock::now();
    std::size_t physical_slots = 1;
    if (imported_sources) {
      const bool prefill = requests == 1 && role == GraphRole::kText && packs_prefill(packing) &&
          !cache_plan.draft_cache && !selected_sampler.enabled && selected_sampler.modifiers.empty();
      const bool ordinary = tokens == 1 && role == GraphRole::kText && !cache_plan.draft_cache;
      const bool mtp = tokens == max_drafts()+1 && requests == 1 &&
          (role == GraphRole::kMtpCatchupRound || (role == GraphRole::kMtpContinuation && device_mtp_successor_)) &&
          cache_plan.draft_cache && (definition.successor_token_port || role == GraphRole::kMtpContinuation) &&
          !selected_sampler.enabled && selected_sampler.modifiers.empty() && packing == RowPacking::kRectangular;
      if ((!ordinary && !mtp && !prefill) || diagnostic || (!prefill && completion != TextCompletion::kEmitNext) ||
          cache_plan.in_flight_batches < 2 || scheduler)
        throw std::runtime_error("imported decode carriers require a complete supported predecessor and two candidate depths");
      physical_slots = role == GraphRole::kMtpContinuation && !device_mtp_continuous_enabled() ? 1 : execution_slot_count;
    }
    const auto planned_slot_uses = shape_plan.uses({requests, tokens, role, completion, sampling_index, false, packing});
    const bool fixed_prefill_capture = packs_prefill(packing) && replay_policy == ReplayPolicy::kAuto &&
        supports_prefill_relay(sampling_index) && role == GraphRole::kText && tokens == bounded_prefill_rows;
    std::array<g::ReplayMode, execution_slot_count> replay_modes{};
    for (std::size_t slot = 0; slot < physical_slots; ++slot)
      // Fixed full chunks retain qualified captures across changing prefixes.
      // Other packed extents keep the original Commands fallback.
      replay_modes[slot] = (!selected_sampler.modifiers.history.empty() ||
          (packs_prefill(packing) && replay_policy == ReplayPolicy::kAuto && !fixed_prefill_capture)) ? g::ReplayMode::kCommands
          : fixed_prefill_capture || imported_sources ? g::ReplayMode::kPreferCapture : select_replay_mode(
          replay_policy, planned_slot_uses[slot]);
    auto result = prepare_phase({definition, selected->program, selected_sampler},
        {tokens, physical_slots, diagnostic, imported_sources, planned_slot_uses, replay_modes},
        {execution, geometry, state_schema, allocation_budget, execution_budget, cache_budget});
    auto &phase = *result;
    if (state_storage_prepared) prepare_transaction_layouts(phase);
    const auto &route_directory = layer0_probe.empty() ? trace_routes : layer0_probe;
    if (!route_directory.empty()) {
      std::filesystem::create_directories(route_directory);
      const auto label = std::string(diagnostic ? "published" : "default") + (has_mtp(role) ? "-role" + std::to_string(static_cast<int>(role)) : "")
          + (sampling_configuration_count() > 1 ? "-sampling" + std::to_string(sampling_index) : "");
      std::ofstream route(route_directory / (std::string(label) + "-R" + std::to_string(requests) + "-T" + std::to_string(tokens) + ".route.tsv"));
      route << "step\tkind\tprovider\timplementation\tnodes\n";
      for (std::size_t step = 0; step < phase.child->compiled.report.step_count(); ++step) {
        const auto detail = take(phase.child->compiled.report.step(step));
        route << step << '\t' << static_cast<unsigned>(detail.kind) << '\t' << detail.provider.name
              << '\t' << detail.implementation.value() << '\t';
        for (std::size_t n = 0; n < detail.node_count; ++n) {
          const auto node = take(phase.child->compiled.report.step_node(step, n));
          const auto kind = take(definition.graph.operation_kind(node));
          route << (n ? "," : "") << node.ordinal() << ':'
                << (kind == c::OperationKind::kView ? c::operation_name(kind)
                    : c::operation_name(take(definition.graph.operation(node))));
        }
        route << '\n';
      }
      std::ofstream boundaries(route_directory / (std::string(label) + "-R" + std::to_string(requests) + "-T" + std::to_string(tokens) + ".boundaries.tsv"));
      boundaries << "point\tvalue\tproducer\tshape\n";
      const auto signature = phase.child->compiled.executable.specialization();
      for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
        const auto value = phase.probe_ports[i];
        boundaries << layer0_probe_names[i] << '\t' << value.ordinal() << '\t'
                   << take(definition.graph.producer(value)).ordinal() << '\t';
        for (const auto dimension : definition.specs.at(value.ordinal()).dimensions()) {
          if (dimension.is_fixed()) { boundaries << dimension.fixed_value() << ','; continue; }
          bool resolved = false;
          for (std::size_t binding = 0; binding < signature.binding_count(); ++binding) {
            const auto entry = take(signature.binding(binding));
            if (entry.dimension != dimension.symbol()) continue;
            boundaries << entry.value << ',';
            resolved = true;
            break;
          }
          if (!resolved) throw std::runtime_error("exact child diagnostic dimension is unresolved");
        }
        boundaries << '\n';
      }
      if (!route || !boundaries) throw std::runtime_error("cannot write layer0 route diagnostics");
    }
    const auto storage = phase.child->compiled.report.compiled_storage();
    std::cerr << "planned_replay T=" << tokens
              << " slot_count=" << phase.slots.size()
              << " slot0_uses=" << phase.slots[0].planned_uses
              << " slot0=" << (phase.slots[0].replay_mode == g::ReplayMode::kCommands ? "commands" : "prefer-capture") << '\n';
    std::cerr << "compiled T=" << tokens << " steps=" << phase.child->compiled.report.step_count()
              << " workspace/slot=" << phase.child->compiled.executable.workspace_bytes()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " child_prepare_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    input_scratch.prepare(phase);
    auto *address = result.get(); cache.emplace(ExecutionShape{requests, tokens, role, completion, sampling_index, imported_sources, packing}, std::move(result)); return *address;
  }

  // An engine with a known ordered request can explicitly prepare its actual
  // carriers before admission/ready, preserving the original cold boundary.
  // Candidate prefix suffixes remain unmaterialized until a hit selects one.
  void prepare_execution(std::span<const std::size_t> calls,
                         std::size_t requests = 1, GraphRole role = GraphRole::kText,
                          TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0, RowPacking packing = RowPacking::kRectangular) {
    const bool packed_prefill = packs_prefill(packing);
    if (carrier_borrowed || active_step)
      throw std::runtime_error("cannot cold-prepare carriers during an active execution loan");
    for (const auto tokens : calls) {
      // Explicit idle warmup can cover a suffix not present in the initial
      // schedule. Use the same bounded admission and original family owners
      // as execution, before borrowing a Session or accepting native work.
      if (!packed_prefill && state_storage_prepared && (role == GraphRole::kText || is_mtp_prefill(role)) &&
          !find_child(requests, tokens, role, false, completion, sampling_index))
        admit_exact_shape(tokens, requests, role, completion, sampling_index);
      if (!layer0_probe.empty()) prepare_child(tokens, false, requests, role, completion, sampling_index);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), requests, role, completion, sampling_index, false, packing);
      initialize_slot(phase, slot_index);
    }
  }

  void prepare_prefill_execution(std::span<const std::size_t> capacities,
      TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0,
      std::size_t requests = 1, GraphRole role = GraphRole::kText) {
    prepare_execution(capacities, requests, role, completion, sampling_index, RowPacking::kPrefill);
  }

  // Qualify every cold-known role before any Invocation borrows the root.
  // Failure is a typed Graph refusal, never permission to allocate private
  // scratch instead. The original per-route slots/IO keep their own owners.
  void prepare_shared_execution_workspace() {
    if (active_step || successor_step || carrier_borrowed || execution_workspace.valid())
      throw std::logic_error("shared execution preparation requires its original idle uninitialized inventory");
    if (!execution_budget)
      throw std::logic_error("shared execution workspace requires the original execution resource budget");
    std::vector<g::Executable> routes;
    std::vector<ExecutionShape> shapes;
    for (const auto &[key, family] : families) {
      if (!family) continue;
      if (key.requests != 1)
        throw std::logic_error("bounded shared execution requires R1 exact routes");
      for (const auto rows : family->chunks) {
        const auto selected = find_child(1, rows, key.role, false, key.completion, key.sampling_index, key.packing);
        if (!selected) throw std::logic_error("shared execution lost its original exact child");
        routes.push_back(selected->program.compiled.executable);
        const ExecutionShape shape{1, rows, key.role, key.completion, key.sampling_index, false, key.packing};
        if (const auto found = phases.find(shape); found != phases.end())
          for (const auto &slot : found->second->slots.slots())
            if (slot.carrier.initialized() || slot.carrier.pending())
              throw std::logic_error("cannot replace scratch on an existing Invocation");
        if (packs_prefill(key.packing)) shapes.push_back(shape);
      }
    }
    if (routes.empty())
      throw inference::EvaluationFailure(hyle::Status::Unimplemented("no cold-known route supports shared scratch"));
    execution_workspace = take(g::InvocationWorkspace::create(routes, {.resource_budget = execution_budget}));
    std::cerr << "execution_shared_workspace routes=" << routes.size() << " bytes=" << execution_workspace.bytes()
              << " resource_budget=execution configured_rows=" << bounded_prefill_rows << '\n';
    for (const auto &shape : shapes) {
      const std::array rows{shape.tokens};
      prepare_prefill_execution(rows, shape.completion, shape.sampling_index, 1, shape.role);
      if (shape.tokens == bounded_prefill_rows && shape.role == GraphRole::kText &&
          supports_prefill_relay(shape.sampling_index)) {
        auto &relay = prepare_child(shape.tokens, false, 1, shape.role, shape.completion,
            shape.sampling_index, true, RowPacking::kPrefill);
        check(relay.slots.initialize_all_with([&] { return slot_initialization(relay); }));
      }
    }
  }

  void prepare_relay_execution(std::size_t requests = 1, std::size_t sampling_index = 0,
      GraphRole role = GraphRole::kText) {
    if (active_step || carrier_borrowed) throw std::runtime_error("relay carrier preparation requires an idle model");
    auto &phase = prepare_child(drafts_token(role) ? max_drafts()+1 : 1, false, requests, role,
        TextCompletion::kEmitNext, sampling_index, true);
    check(phase.slots.initialize_all_with([&] { return slot_initialization(phase); }));
  }

  // Prepare the existing cold plan, not every possible context length. This
  // creates carriers only: no Session, state transition, sampling or submit.
  // Native capture still needs actual bindings and remains a measured cost.
  std::size_t prepare_planned_execution() {
    if (active_step || carrier_borrowed)
      throw std::runtime_error("cannot prepare planned shapes during an active episode");
    std::size_t count = 0;
    for (const auto &[shape, uses] : shape_plan.entries()) {
      if (std::ranges::none_of(uses, [](auto planned_uses) { return planned_uses != 0; })) continue;
      const std::array rows{shape.tokens};
      prepare_execution(rows, shape.requests, shape.role, shape.completion, shape.sampling_index, shape.packing);
      ++count;
    }
    return count;
  }

  DecodeRecording prepare_ordinary_decode_recordings(std::size_t sampling_index,
      std::shared_ptr<Session> *prepared_session = nullptr) {
    if (cache_plan.draft_cache) return {};
    return prepare_decode_recordings(sampling_index, false, prepared_session, DecodeRecordingStage::kStartup);
  }

  DecodeRecording prepare_mtp_decode_recordings(std::size_t sampling_index,
      std::shared_ptr<Session> *prepared_session = nullptr,
      DecodeRecordingStage stage = DecodeRecordingStage::kStartup) {
    if (!cache_plan.draft_cache || ((!device_mtp_successor_) && (max_drafts() != 1 || recursive_mtp()))) return {};
    try { return prepare_decode_recordings(sampling_index, true, prepared_session, stage); }
    catch (const inference::EvaluationFailure &failure) {
      if (!device_mtp_successor_ || !failure.status().IsUnimplemented() ||
          max_drafts()>1 ||
          failure.status().has_native_error() || active_step || successor_step || carrier_borrowed ||
          snapshot_versions.live || (prepared_session && *prepared_session) ||
          (failure.retirement_observed() && (!failure.retirement().retired || !failure.retirement().status.ok()))) throw;
      // An optional native shape was declined before request admission. The
      // real private recording episode has retired; preserve the old route.
      device_mtp_successor_ = false;
      return prepare_decode_recordings(sampling_index, true, prepared_session, stage);
    }
  }

  // A cancelled/failed submitted episode must destroy its original Graph
  // carrier. Rebuild the existing capture inventory at the next idle request
  // admission, after its Session and every native use have actually retired.
  void restore_mtp_decode_recordings(std::size_t sampling_index) {
    if (!device_mtp_successor_ || active_step || successor_step || carrier_borrowed || snapshot_versions.live)
      return;
    if (!supports_device_relay(sampling_index)) return;
    const auto phase_for = [&](GraphRole role, bool imported) -> Phase * {
      const auto found = phases.find({1, max_drafts()+1, role, TextCompletion::kEmitNext, sampling_index, imported,
          RowPacking::kRectangular});
      return found == phases.end() ? nullptr : found->second.get();
    };
    const std::array selected{phase_for(GraphRole::kMtpRound, false),
        phase_for(GraphRole::kMtpCatchupRound, false), phase_for(GraphRole::kMtpContinuation, true)};
    if (std::ranges::any_of(selected, [](const auto *phase) { return !phase || !phase->slots.size(); })) return;
    const bool missing = !selected[0]->slots[0].carrier.initialized() || selected[0]->slots[0].carrier.closing() ||
        !selected[1]->slots[0].carrier.initialized() || selected[1]->slots[0].carrier.closing() ||
        std::ranges::any_of(selected[2]->slots.slots(), [](const auto &slot) {
          return !slot.carrier.initialized() || slot.carrier.closing();
        });
    if (!missing || !private_publications_retired()) return;
    for (auto *phase : selected) for (const auto &slot : phase->slots.slots())
      if (!slot.carrier.retired()) return;
    for (auto *phase : selected) {
      const auto retired = phase->slots.retire_all(execution);
      if (!retired.retired) {
        inference::EvaluationFailure failure(retired.status.ok()
            ? hyle::Status::Unavailable("MTP capture recovery has unretired native work") : retired.status);
        failure.observe_retirement(retired);
        throw failure;
      }
      check(retired.status);
      for (auto &slot : phase->slots.slots()) check(slot.carrier.reset_retired());
    }
    (void)prepare_mtp_decode_recordings(sampling_index, nullptr, DecodeRecordingStage::kRecovery);
  }

  DecodeRecording prepare_decode_recordings(std::size_t sampling_index, bool mtp,
      std::shared_ptr<Session> *prepared_session, DecodeRecordingStage stage,
      std::size_t prefill_rows = 0) {
    DecodeRecording report;
    if (prepared_session && *prepared_session)
      throw std::runtime_error("decode recording requires an empty session destination");
    if (capacity < 2 || cache_plan.in_flight_batches < 2 ||
        !supports_device_relay(sampling_index)) return report;
    const auto role = mtp ? GraphRole::kMtpRound : GraphRole::kText;
    const bool device_pair = mtp && device_mtp_successor_;
    const auto relay_role = mtp ? (device_pair ? GraphRole::kMtpContinuation : GraphRole::kMtpCatchupRound) : GraphRole::kText;
    const bool prefill_recording = prefill_rows != 0;
    if (prefill_recording && (mtp || cache_plan.draft_cache || prefill_rows != bounded_prefill_rows))
      throw std::logic_error("prefill recording requires its original fixed ordinary chunk");
    const auto packing = prefill_recording ? RowPacking::kPrefill : RowPacking::kRectangular;
    const auto rows = prefill_recording ? prefill_rows : mtp ? max_drafts()+1 : std::size_t{1};
    const std::array chunk{rows};
    prepare_execution(chunk, 1, role, TextCompletion::kEmitNext, sampling_index, packing);
    if (mtp) prepare_execution(chunk, 1, GraphRole::kMtpCatchupRound, TextCompletion::kEmitNext, sampling_index);
    if (!prefill_recording) prepare_relay_execution(1, sampling_index, relay_role);
    auto &direct = prepare_child(rows, false, 1, role, TextCompletion::kEmitNext, sampling_index, false, packing);
    auto &imported = prepare_child(rows, false, 1, relay_role, TextCompletion::kEmitNext, sampling_index, true, packing);
    if (direct.slots[0].replay_mode == g::ReplayMode::kCommands) return report;
    if (active_step || successor_step || carrier_borrowed || snapshot_versions.live)
      throw std::runtime_error("decode recording requires an idle model before request admission");
    const auto begin = Clock::now();
    const auto recording_phase = prefill_recording ? "prefill_recording"
        : stage == DecodeRecordingStage::kStartup ? "decode_warmup" : "decode_recovery";
    std::cerr << recording_phase << "_begin\n";
    auto session = create_session(chunk, {}, role, {}, false, TextCompletion::kEmitNext, sampling_index);
    hyle::util::ScopeGuard retire_session{[&]() noexcept {
      (void)abort_episode(); session->stop().IgnoreError(); (void)session->drain();
    }};
    struct Arguments final {
      c::ConstTensorView token;
      c::ConstTensorView catchup;
      c::ConstTensorView prefix;
      std::vector<g::detail::RecordingStoreInputs> states;
    };
    const auto record = [&](Phase &phase, std::size_t slot, const Arguments *previous) {
      auto &carrier = phase.slots[slot].carrier;
      auto *run = take(carrier.borrow());
      check(g::detail::RecordingAccess::begin(*run));
      hyle::util::ScopeGuard withdraw{[&]() noexcept {
        (void)abort_episode();
        if (carrier.initialized() && !carrier.closing() && carrier.retired())
          if (auto original = carrier.borrow(); original)
            g::detail::RecordingAccess::end(**original).IgnoreError();
      }};
      const std::vector<std::int32_t> placeholder(prefill_recording ? rows : 1, 0);
      const std::array requests{EvaluationRequest{session, placeholder}};
      const RecordingStepInputs inputs{&phase, slot, previous ? previous->token : c::ConstTensorView{},
          previous ? previous->catchup : c::ConstTensorView{},
          previous ? previous->prefix : c::ConstTensorView{},
          previous ? std::span<const g::detail::RecordingStoreInputs>(previous->states) : std::span<const g::detail::RecordingStoreInputs>{}};
      prepare_evaluations<true>(requests, nullptr, phase.family->role, phase.family->completion,
          {}, 0, prefill_recording ? rows : 0, packing, &inputs);
      auto &step = *active_step;
      if (step.submitted || step.execution.has_completion() || session->length() || session->head->next_id != -1)
        throw std::runtime_error("cold recording unexpectedly executed or advanced private model state");
      Arguments next;
      const auto output_view = [&](inference::ProgramIo::Readback port) {
        const auto *output = take(phase.slots[slot].io.readback(port));
        return take(c::ConstTensorView::create(output->facts.description,
            take(c::detail::LeaseAccess::backing_span(output->pool()))));
      };
      if (!mtp || (drafts_token(phase.family->role) &&
          (phase.family->role != GraphRole::kMtpContinuation || device_mtp_continuous_enabled()))) {
        if (phase.family->completion == TextCompletion::kEmitNext)
          next.token = output_view(mtp ? (device_pair ? phase.slots[slot].data.successor_pending
              : phase.slots[slot].data.successor_token) : phase.slots[slot].data.next_id);
        if (mtp) next.catchup = output_view(device_pair ? phase.slots[slot].data.successor_selected_catchup
            : phase.slots[slot].data.successor_catchup);
        if (device_pair) next.prefix = output_view(phase.slots[slot].data.successor_prefix_index);
        next.states.reserve(phase.state_sources.size());
        for (const auto &source : phase.state_sources) {
          auto metadata = take(g::detail::RecordingAccess::store_output(*run, source.store));
          if (mtp && !device_pair && next.states.size() < geometry.recurrent_layers() * 2) {
            // A full K1 producer stores both prefixes. Its successor gathers the
            // final target state, but keeps both hidden rows in chronological order.
            metadata.indices = take(c::ConstTensorView::create(dense(Type::kInt32, {1}),
                take(metadata.indices.storage().subview((phase.prefix_rows-1)*sizeof(std::int32_t), sizeof(std::int32_t)))));
          }
          next.states.push_back(metadata);
        }
      }
      ++report.preparations; report.captures += step.capture_recorded;
      if (step.capture_recorded) report.capture_preparation_seconds += step.cold_batch_seconds;
      if (!take(step.try_discard_recording_preparation()))
        throw std::runtime_error("cold recording did not retire its unsubmitted transactions");
      check(g::detail::RecordingAccess::end(*run));
      detach_episode(); withdraw.dismiss();
      return next;
    };
    auto previous = record(direct, 0, nullptr);
    if (prefill_recording) {
      auto &state_direct = prepare_child(rows, false, 1, role, TextCompletion::kStateOnly,
          sampling_index, false, packing);
      auto &state_imported = prepare_child(rows, false, 1, role, TextCompletion::kStateOnly,
          sampling_index, true, packing);
      const auto initial_states = record(state_direct, 0, nullptr);
      previous = initial_states;
      std::array<Arguments, execution_slot_count> states;
      for (const auto slot : {std::size_t{1}, std::size_t{2}, std::size_t{0}, std::size_t{1}}) {
        previous = record(state_imported, slot, &previous);
        states[slot] = previous;
      }
      for (std::size_t slot = 0; slot < execution_slot_count; ++slot)
        (void)record(imported, slot, &states[(slot + execution_slot_count - 1) % execution_slot_count]);
      (void)record(imported, 1, &initial_states);
    }
    if (mtp) {
      if (device_mtp_continuous_enabled()) {
        for (const auto slot : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{0}})
          previous = record(imported, slot, &previous);
      } else if (device_pair) (void)record(imported, 0, &previous);
      auto &catchup = prepare_child(rows, false, 1, GraphRole::kMtpCatchupRound, TextCompletion::kEmitNext, sampling_index);
      auto catchup_arguments = record(catchup, 0, nullptr);
      if (device_pair) (void)record(imported, device_mtp_continuous_enabled() ? 1 : 0, &catchup_arguments);
      // Terminal T1 roles record on first use. Retaining their private native
      // preparation here exceeds the startup-recording memory budget.
    }
    if (!device_pair && !prefill_recording) for (const auto slot : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{0}})
      previous = record(imported, slot, &previous);
    if (active_step || successor_step || carrier_borrowed || session->pending || session->stopping ||
        session->failed || session->length() || session->head->next_id != -1 || snapshot_versions.live != 1)
      throw std::runtime_error("cold recording retained work or changed the empty session");
    if (prepared_session) {
      // The first real request owns this initialized, still empty Session.
      // Admission and its actual input validation remain in RequestLoop.
      *prepared_session = std::move(session);
    } else {
      check(session->stop());
      const auto retired = session->drain(); check(retired.status);
      if (!retired.retired || snapshot_versions.live)
        throw std::runtime_error("cold recording retained private state");
    }
    retire_session.dismiss();
    report.seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    std::cerr << recording_phase << "_end graph_calls=0 target_rows=0 capture_recorded=" << report.captures
        << " capture_associated_cold_prep_seconds=" << report.capture_preparation_seconds
        << " seconds=" << report.seconds << " recorded_preparations=" << report.preparations << '\n';
    return report;
  }

  DecodeRecording prepare_prefill_recordings(std::size_t sampling_index) {
    // A chunk covering the entire context cannot have a later state-only
    // chunk. No such finite family exists or needs capture preparation.
    if (cache_plan.draft_cache || !bounded_prefill_rows || bounded_prefill_rows >= capacity) return {};
    const auto &sampling = sampling_configuration(sampling_index).configuration;
    if (sampling.enabled || !sampling.modifiers.empty()) return {};
    return prepare_decode_recordings(sampling_index, false, nullptr,
        DecodeRecordingStage::kStartup, bounded_prefill_rows);
  }

  hyle::Status resolve_device_mtp_successor(StepState &step, const cache::State &head) noexcept {
    if (!step.device_input_choice || !step.phase || step.members.size() != 1 || !step.submitted ||
        !step.execution.incoming || !step.execution.incoming->owner)
      return hyle::Status::InvalidArgument("device successor has no original submitted episode");
    auto &member = step.members[0];
    const auto &source = *step.execution.incoming->owner;
    if (!source.mtp_device_choice || source.boundaries.size() != 1 || member.base != &head ||
        !head.draft || head.length < member.choice_base_length ||
        head.length - member.choice_base_length >= member.target_input_choices.size() ||
        inference::request_token_stopped(head.next_id, model_eos_tokens, member.stop_policy))
      return hyle::Status::InvalidArgument("device successor cannot bind its producer's actual boundary");
    const auto choice = head.length - member.choice_base_length;
    if(choice>=member.draft_input_choices.size())return hyle::Status::InvalidArgument("device successor lost its draft coordinate");
    const auto draft_layout=member.draft_input_choices[choice];
    const auto rows=std::size_t(draft_layout.facts().query_rows);
    const auto lag=choice?1u:0u;
    const auto passes=step.phase->draft_chain_steps?2u:1u;
    const auto work=rows*passes;
    const auto attention=step.phase->draft_chain_steps;
    if (step.execution.sources && step.execution.sources->boundaries.size() != 1)
      return hyle::Status::InvalidArgument("device successor source lost its original member boundary");
    if (head.draft->frontier!=std::size_t(draft_layout.facts().kv_rows)-rows || head.draft->lag != lag ||
        (!lag && head.draft->catchup_token != -1) || (lag && head.draft->catchup_token < 0))
      return hyle::Status::InvalidArgument("device successor producer has inconsistent draft frontier");
    if (member.selected_input_choice)
      return *member.selected_input_choice == choice ? hyle::Status::Ok()
          : hyle::Status::InvalidArgument("device successor selection changed after binding");
    if (!member.next || !member.next->draft ||
        work > SIZE_MAX - submitted_draft_rows || work > SIZE_MAX - member.session->submitted_draft_rows ||
        (step.draft_rows_deferred && attention>SIZE_MAX-draft_chain_attention_rows))
      return hyle::Status::OutOfRange("device successor row accounting exceeds its original capacity");
    const auto target_pages=member.target_input_choices[choice].host().page_indices.size();
    const auto draft_pages=member.draft_input_choices[choice].host().page_indices.size();
    if(member.next->paged.pages.size()<target_pages || member.next->draft->paged.pages.size()<draft_pages)
      return hyle::Status::InvalidArgument("device successor canonical pages exceed their retained raw cohort");
    member.selected_input_choice = choice;
    member.next->length = head.length + step.phase->prefix_rows;
    member.next->draft->frontier = head.draft->frontier + rows;
    member.layout = member.target_input_choices[choice];
    member.draft_layout = step.draft_layout = member.draft_input_choices[choice];
    // Only clip the canonical view. Original raw cohorts remain owned until
    // the native episode retires, including every unused speculative row.
    member.next->paged.pages.resize(target_pages);
    member.next->draft->paged.pages.resize(draft_pages);
    member.first_input = head.next_id;
    member.draft_input_rows = step.active_draft_rows = rows;
    if (step.execution.sources) {
      auto &boundary = step.execution.sources->boundaries[0];
      boundary.length = member.next->length;
      boundary.draft_frontier = member.next->draft->frontier;
    }
    if (step.draft_rows_deferred) {
      submitted_draft_rows += work;
      member.session->submitted_draft_rows += work;
      draft_chain_attention_rows+=attention;
      step.draft_rows_deferred = false;
    }
    return hyle::Status::Ok();
  }

  void detach_episode() noexcept {
    if (!active_step) return;
    if (successor_step && successor_step->execution.cleanup_pending) {
      const auto cleanup = abort_successor_episode();
      inference::record_retirement_status(active_step->execution.terminal, cleanup.status);
      if (!cleanup.retired) {
        return; // Retain both owners for the original retirement retry.
      }
    }
    // Only a caller-requested, actually retired shared cancellation is local
    // to this episode. An unchanged surviving Session remains runnable; do not
    // poison its later retirement with a peer's normal Graph cancellation.
    // Preserve native and all other failures, and the cancelled owner's fact.
    const bool cancelled_episode = active_step->cancelled &&
        active_step->execution.terminal.IsCancelled() && !active_step->execution.terminal.has_native_error();
    for (auto &member : active_step->members) {
      if (!cancelled_episode || member.session->stopping)
        inference::record_retirement_status(member.session->retirement_status, active_step->execution.terminal);
      member.session->pending = successor_step && std::ranges::any_of(successor_step->members,
          [&](const auto &next) { return next.session == member.session; }) ? successor_step.get() : nullptr;
    }
    active_step = std::move(successor_step);
    if (active_step) for (std::size_t j = 0; j < active_step->members.size(); ++j) {
      auto &member = active_step->members[j];
      const auto *source = active_step->execution.incoming ? active_step->execution.incoming->owner.get() : nullptr;
      const auto *boundary = source ? &source->boundaries[j] : nullptr;
      if (active_step->device_input_choice && member.session->head) {
        const auto selected = resolve_device_mtp_successor(*active_step, *member.session->head);
        if (!selected.ok()) { active_step->cancelled = true; continue; }
      }
      const bool full_boundary = active_step->device_input_choice ? member.selected_input_choice.has_value() :
          !boundary || (member.session->head && member.session->head->length == boundary->length &&
          (!source->mtp_full_candidate || (member.session->head->draft &&
              member.session->head->draft->frontier == boundary->draft_frontier &&
              member.session->head->draft->lag == 1 && !inference::request_token_stopped(
                  member.session->head->next_id, model_eos_tokens, member.stop_policy))));
      if (member.session->head.get() == member.base && member.session->head.epoch() == member.base_epoch && full_boundary) {
        if (!source || !source->prefill) member.first_input = member.session->head->next_id;
      }
      else active_step->cancelled = true;
    }
  }

  inference::RetirementObservation try_retire_episode() noexcept {
    if (!active_step) return {true, {}};
    if (successor_step) {
      inference::record_retirement_status(successor_step->execution.terminal, execution.collect_retired());
      const auto retired = successor_step->try_abort();
      if (!retired) return {false, retired.error()};
      if (!*retired) return {false, successor_step->execution.terminal};
      successor_step.reset();
    }
    // Dropping the public Completion does not remove original native work.
    // Drive its owning environment before checking the unchanged Graph/page
    // retirement gates; an observation error alone never proves retirement.
    inference::record_retirement_status(active_step->execution.terminal, execution.collect_retired());
    const auto retired = active_step->try_abort();
    if (!retired) {
      inference::record_retirement_status(active_step->execution.terminal, retired.error());
      return {false, active_step->execution.terminal};
    }
    const auto terminal = active_step->execution.terminal;
    if (*retired) detach_episode();
    return {*retired, terminal};
  }

  inference::RetirementObservation abort_successor_episode() noexcept {
    if (!successor_step) return {true, {}};
    if (!successor_step->submitted && successor_step->execution.run && active_step &&
        active_step->execution.run && active_step->execution.run->in_flight()) {
      successor_step->cancelled = true;
      successor_step->execution.cleanup_pending = true;
      return {false, {}};
    }
    if (successor_step->execution.run && successor_step->phase &&
        !successor_step->execution.unsubmitted_reuse_declined()) {
      (void)successor_step->execution.retire(execution);
    }
    const auto retired = successor_step->try_abort();
    if (!retired) return {false, retired.error()};
    const auto status = successor_step->execution.terminal;
    if (*retired) successor_step.reset();
    return {*retired, status};
  }

  // A speculative result can be unusable even though its whole original
  // Graph completed successfully. Withdraw that unpublished episode through
  // the same transactions while keeping its open carrier and dormant capture.
  // Any incomplete/failed recovery remains owned for the ordinary abort path.
  inference::RetirementObservation discard_successful_episode() noexcept {
    discard_decode_ahead();
    if (!active_step) return {true, {}};
    auto &step = *active_step;
    if (successor_step || !step.submitted || !step.phase || !step.execution.run ||
        !step.execution.has_completion()) return {false, step.execution.terminal};
    // Session withdrawal can close the original Invocation directly, without
    // setting ProgramCarrier::closing(). Keep that real cancellation on the
    // destructive retirement path even if its native work already succeeded.
    if (std::ranges::any_of(step.members,
        [](const auto &member) { return member.session->stopping; }))
      return {false, step.execution.terminal};
    if (!step.execution.terminal.ok()) return {false, step.execution.terminal};
    const auto completed = step.execution.wait();
    if (!completed.ok()) {
      inference::record_retirement_status(step.execution.terminal, completed);
      return {false, step.execution.terminal};
    }
    const auto discarded = step.try_discard_successful();
    if (!discarded) {
      inference::record_retirement_status(step.execution.terminal, discarded.error());
      return {false, step.execution.terminal};
    }
    const auto status = step.execution.terminal;
    if (*discarded) detach_episode();
    return {*discarded, status};
  }

  inference::RetirementObservation abort_episode() noexcept {
    discard_decode_ahead();
    if (!active_step) return {true, {}};
    if (active_step->phase) {
      const auto copied = active_step->phase->slots[active_step->state_slot].data.small_results.recover();
      inference::record_retirement_status(active_step->execution.terminal, copied.status);
      if (!copied.retired) return {false, active_step->execution.terminal};
    }
    if (successor_step && successor_step->execution.run && successor_step->phase) {
      // Capture destruction for an unsubmitted B can synchronize its native
      // context. Keep the original predecessor gate before even draining B.
      if (!successor_step->submitted && active_step->execution.run && active_step->execution.run->in_flight()) {
        successor_step->cancelled = true;
        successor_step->execution.cleanup_pending = true;
      } else {
        (void)successor_step->execution.retire(execution);
      }
    }
    // Nonblocking collection cannot drain a partially accepted Vulkan prefix.
    // Hand off once to the original blocking owner, then check the unchanged
    // Graph/page gates. Unknown native failure keeps this candidate, Session
    // pins and admission intact for an explicit retry; it must not spin.
    if (active_step->execution.run && active_step->phase) {
      (void)active_step->execution.retire(execution);
    }
    // Episode retirement alone does not release a Session's published head.
    return try_retire_episode();
  }

  // Cohort selection and physical capacity belong to the caller. A batch
  // owns one original Invocation, IO loan, Completion and publication transaction.
  template<bool recording_only = false>
  void prepare_draft_append(StepState &step, GraphRole role) {
    auto &phase = *step.phase;
    auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
    const auto width = step.members.size();
    const bool relay = bool(step.execution.incoming);
    const bool compact_draft = packs_draft(phase.family->packing);
    const auto physical_rows = cache::multiply(width, phase.draft_chain_steps ? phase.draft_chain_steps + draft_lag(role) : is_mtp_prefill(role)
        ? cache::add(phase.query_rows, draft_lag(role)) : draft_lag(role) + 1);
    auto &positions = input_scratch.draft_positions, &catchups = input_scratch.catchups;
    positions.resize(physical_rows); catchups.clear();
    auto &row_inputs = input_scratch.row_inputs;
    auto &offsets = input_scratch.query_indptr;
    auto &last_rows = input_scratch.last_rows;
    row_inputs.clear(); offsets.resize(width + 1); last_rows.resize(width);
    for (const auto &member : step.members) row_inputs.push_back({member.base->draft->frontier,
        phase.draft_chain_steps ? phase.draft_chain_steps + draft_lag(role) : compact_draft ? member.base->draft->lag + 1 : is_mtp_prefill(role)
            ? cache::add(member.input_tokens, draft_lag(role)) : draft_lag(role) + 1});
    (void)inference::pack_sequence_rows(row_inputs, physical_rows, offsets, last_rows, positions);
    if (compact_draft) {
      if (io.draft_hidden_indices.size() != physical_rows || io.draft_token_indices.size() != physical_rows)
        throw std::logic_error("MTP draft selectors differ from their cold physical carrier");
      std::ranges::fill(io.draft_hidden_indices, 0); std::ranges::fill(io.draft_token_indices, 0);
      for (std::size_t j = 0; j < width; ++j) {
        const auto lag = step.members[j].base->draft->lag;
        const auto begin = static_cast<std::size_t>(offsets[j]);
        if (lag > 1 || offsets[j + 1] <= offsets[j] ||
            last_rows[j] != offsets[j + 1] - 1 || row_inputs[j].rows != lag + 1)
          throw std::logic_error("MTP draft selection differs from its actual member rows");
        if (lag) {
          io.draft_hidden_indices[begin] = static_cast<std::int32_t>(width + j);
          io.draft_token_indices[begin] = static_cast<std::int32_t>(j);
        }
        io.draft_hidden_indices[begin + lag] = static_cast<std::int32_t>(j);
        io.draft_token_indices[begin + lag] = static_cast<std::int32_t>(width + j);
      }
    } else if (packs_prefill(phase.family->packing)) {
      const auto target_rows = cache::multiply(width, phase.query_rows);
      const auto prefix_rows = cache::multiply(width, draft_lag(role) + 1);
      const auto hidden_rows = cache::add(prefix_rows, target_rows);
      const auto token_rows = cache::add(width, target_rows);
      if (hidden_rows > static_cast<std::size_t>(INT32_MAX) || token_rows > static_cast<std::size_t>(INT32_MAX) ||
          io.draft_hidden_indices.size() != physical_rows ||
          (draft_lag(role) && io.draft_token_indices.size() != physical_rows))
        throw std::out_of_range("MTP compact selection exceeds its original I32 carrier");
      for (std::size_t j = 0; j < width; ++j) {
        const auto &member = step.members[j];
        if (!member.input_tokens || member.input_row_begin > target_rows ||
            member.input_tokens > target_rows - member.input_row_begin ||
            row_inputs[j].rows != cache::add(member.input_tokens, draft_lag(role)))
          throw std::logic_error("MTP compact selection differs from the target and draft row facts");
      }
      std::ranges::fill(io.draft_hidden_indices, 0);
      std::ranges::fill(io.draft_token_indices, 0);
      for (std::size_t j = 0; j < width; ++j) {
        const auto &member = step.members[j];
        const auto target_begin = member.input_row_begin;
        const auto draft_begin = static_cast<std::size_t>(offsets[j]);
        auto destination = draft_begin;
        if (draft_lag(role)) {
          io.draft_hidden_indices[destination++] = static_cast<std::int32_t>(width + j);
          io.draft_token_indices[draft_begin] = static_cast<std::int32_t>(j);
          for (std::size_t row = 0; row < member.input_tokens; ++row)
            io.draft_token_indices[draft_begin + 1 + row] = static_cast<std::int32_t>(width + target_begin + row);
        }
        io.draft_hidden_indices[destination] = static_cast<std::int32_t>(j);
        for (std::size_t row = 1; row < member.input_tokens; ++row)
          io.draft_hidden_indices[destination + row] = static_cast<std::int32_t>(prefix_rows + target_begin + row - 1);
      }
    }
    const auto begin_member = [&](std::size_t j) {
      auto &member = step.members[j];
      member.next->draft.emplace(); auto &next = *member.next->draft;
      if (!next.paged.frames.try_resize(2)) throw std::bad_alloc{};
      next.frontier = member.base->draft->frontier + row_inputs[j].rows;
    };
    const auto member_at = [&](std::size_t j) noexcept -> inference::PagedAppendMember {
      auto &member = step.members[j]; auto &session = *member.session;
      return {member.base->draft->paged, member.next->draft ? &member.next->draft->paged : nullptr,
          member.draft_layout, member.base->draft->frontier, row_inputs[j].rows,
          relay || phase.draft_chain_steps ? std::span<c::PagedPageAppender>{} : std::span(session.draft_tails),
          !relay && !phase.draft_chain_steps && session.draft_tail_epoch == member.base_epoch};
    };
    const auto invalidate_tail = [&](std::size_t j) noexcept {
      if (!relay) step.members[j].session->draft_tail_epoch = 0;
    };
    step.draft_layout = take(inference::prepare_paged_append(attention_schema, width,
        inference::PagedAppendBanks{draft_inventory,
            std::span(step.execution.dirty.data() + draft_bank, 2),
            std::span(step.execution.tails.data() + draft_bank, 2),
            std::span(step.execution.appends.data() + draft_bank, 2)},
        input_scratch.paged_appends, member_at,
        {.capacity_rows = capacity,
         .incoming = relay ? std::span(step.execution.incoming->owner->pages).subspan(draft_bank, 2)
             : std::span<g::batch::PagedAppendSource>{},
         .pending_predecessor = bool(step.execution.predecessor_episode),
         .append_options = {.resource_budget = cache_budget}},
        invalidate_tail, begin_member));
    // The actual member layouts exclude packed carrier padding and retain
    // each mixed K1 member's own lag. Freeze them before submit/publication.
    step.active_draft_rows = 0;
    for (auto &member : step.members) {
      member.draft_input_rows = member.draft_layout.facts().query_rows;
      step.active_draft_rows = cache::add(step.active_draft_rows, member.draft_input_rows);
    }
    if (step.active_draft_rows != step.draft_layout.facts().query_rows)
      throw std::logic_error("draft submission rows differ from the actual Append cohort");
    const bool recording_imported = recording_only && phase.imported_sources;
    if (draft_lag(role) && !relay && !recording_imported) for (const auto &member : step.members) {
      const auto &old = *member.base->draft;
      catchups.push_back(recording_only ? 0 : compact_draft && !old.lag ? member.first_input : old.catchup_token);
    }
    io.draft_positions.write<std::int32_t>(positions);
    if (draft_lag(role) && !relay && !recording_imported) io.catchup_token.write<std::int32_t>(catchups);
    if (packs_prefill(phase.family->packing) || compact_draft) {
      io.draft_hidden_select.write<std::int32_t>(io.draft_hidden_indices);
      if (draft_lag(role)) io.draft_token_select.write<std::int32_t>(io.draft_token_indices);
    }
    if (compact_draft) io.draft_last_ids.write<std::int32_t>(last_rows);
  }

  void prepare_recursive_repair_append(StepState &step) {
    auto &phase = *step.phase;
    if (!phase.recursive_draft_repair || step.members.size() != 1 || phase.query_rows < 2)
      throw std::logic_error("canonical repair append requires its exact R1 verifier");
    auto &member = step.members[0]; auto &io = phase.slots[step.state_slot].data;
    const auto &base = *member.base->draft;
    member.next->draft.emplace(); auto &next = *member.next->draft;
    if (!next.paged.frames.try_resize(2)) throw std::bad_alloc{};
    next.frontier = base.frontier + phase.query_rows;
    const auto member_at = [&](std::size_t) noexcept -> inference::PagedAppendMember {
      // Never borrow a published tail for speculative target-prefix repair.
      return {base.paged, &next.paged, member.draft_layout, base.frontier, phase.query_rows};
    };
    step.draft_layout = take(inference::prepare_paged_append(attention_schema, 1,
        inference::PagedAppendBanks{draft_inventory,
            std::span(step.execution.dirty.data() + draft_bank, 2),
            std::span(step.execution.tails.data() + draft_bank, 2),
            std::span(step.execution.appends.data() + draft_bank, 2)},
        input_scratch.paged_appends, member_at,
        {.capacity_rows = capacity, .append_options = {.resource_budget = cache_budget}}));
    auto &positions = input_scratch.draft_positions; positions.clear();
    for (std::size_t i = 0; i < phase.query_rows; ++i)
      positions.push_back(static_cast<std::int32_t>(base.frontier + i));
    io.draft_positions.write<std::int32_t>(positions);
    const std::array catchup{base.lag ? base.catchup_token : member.first_input};
    const std::array lag{static_cast<std::int32_t>(base.lag)};
    io.catchup_token.write<std::int32_t>(catchup); io.repair_lag.write<std::int32_t>(lag);
    step.active_draft_rows = member.draft_input_rows = phase.query_rows;
  }

  void prepare_round_draft_chain(StepState &step) {
    auto &phase = *step.phase; auto &member = step.members.front();
    auto &io = phase.slots[step.state_slot].data;
    const auto &base = *member.base->draft;
    member.draft_chain_pages.resize(phase.draft_chain_steps);
    for (std::size_t i = 0; i < phase.draft_chain_steps; ++i) {
      auto &pages = member.draft_chain_pages[i];
      if (!pages.frames.try_resize(2)) throw std::bad_alloc{};
      auto &layout = io.batch_layouts[phase.attention_nodes.size() + i].layout;
      const auto at = [&](std::size_t) noexcept -> inference::PagedAppendMember {
        return {base.paged, &pages, layout, base.frontier, phase.draft_chain[i].rows};
      };
      const auto bank = draft_bank + 2 + 2 * i;
      const bool relay = bool(step.execution.incoming);
      layout = take(inference::prepare_paged_append(attention_schema, 1,
          inference::PagedAppendBanks{draft_inventory, std::span(step.execution.dirty.data() + bank, 2),
              std::span(step.execution.tails.data() + bank, 2), std::span(step.execution.appends.data() + bank, 2)},
          input_scratch.paged_appends, at,
          {.capacity_rows = capacity,
           .incoming = relay ? std::span(step.execution.incoming->owner->pages).subspan(draft_bank, 2)
               : std::span<g::batch::PagedAppendSource>{},
           .pending_predecessor = bool(step.execution.predecessor_episode),
           .append_options = {.resource_budget = cache_budget}}));
    }
    member.verification_inputs.assign(phase.prefix_rows, 0);
    member.verification_inputs[0] = member.first_input;
  }

  void prepare_device_mtp_inputs(StepState &step, bool recording_only);

  void discard_decode_ahead(const Session *session = nullptr) noexcept {
    if (!decode_ahead) return;
    auto &prepared = *decode_ahead;
    if (!session) {
      if (prepared.ready) ++cpu_ahead_discarded;
      prepared.clear(); return;
    }
    bool discarded = false;
    for (auto &member : prepared.members)
      if (member.ready && member.session.lock().get() == session) {
        member = {}; discarded = true;
      }
    if (discarded) ++cpu_ahead_discarded;
    prepared.refresh();
  }

  void prune_decode_ahead() noexcept {
    if (!decode_ahead || !decode_ahead->ready) return;
    bool discarded = false;
    for (auto &member : decode_ahead->members) if (member.ready) {
      const auto owner = member.session.lock();
      if (!owner || owner->failed || owner->stopping || !owner->head ||
          owner->head.epoch() != member.epoch || owner->length() <= member.base_length ||
          owner->length() > member.maximum_length) {
        member = {}; discarded = true;
      }
    }
    if (discarded) ++cpu_ahead_discarded;
    decode_ahead->refresh();
  }

  // Only integer metadata, cold-bounded O(R*pages + R*K). No head, state slot,
  // draft frontier or native permission is retained by this optional cache.
  void prepare_decode_ahead_storage() noexcept {
    if ((!cache_plan.draft_cache && cache_plan.active_sequences < 2) || decode_ahead) return;
    try {
      decode_ahead.emplace(execution_budget, take(execution_budget.metadata_allocator()));
      auto &prepared = *decode_ahead;
      const auto width = cache_plan.active_sequences;
      prepared.maximum_rows = cache_plan.draft_cache ? cache_plan.candidate_boundaries : 1;
      prepared.page_stride = cache::add(capacity, cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      if (!prepared.members.try_resize(width) || !prepared.selected.try_reserve(width) ||
          !prepared.candidate_ids.try_resize(cache::multiply(width, prepared.page_stride)) ||
          !prepared.positions.try_reserve(cache::multiply(width, prepared.maximum_rows)) ||
          !prepared.old_ids.try_reserve(attention_schema.capacity().max_page_refs) ||
          !prepared.old_indptr.try_reserve(width + 1) || !prepared.old_last.try_reserve(width) ||
          !prepared.old_query.try_reserve(width + 1)) decode_ahead.reset();
    } catch (...) { decode_ahead.reset(); } // Optional, before any accepted work.
  }

  // Save each candidate's page stripe and finite accepted-length range, not
  // K^R outcomes. Actual next role/rows, draft repair and permissions remain
  // in the original prepare/finish path.
  hyle::Status prepare_decode_ahead() noexcept {
    if (!decode_ahead) return hyle::Status::Ok();
    if (!active_step || !active_step->submitted || active_step->cancelled || !active_step->phase ||
        active_step->phase->family->completion != TextCompletion::kEmitNext ||
        is_mtp_prefill(active_step->phase->family->role) ||
        (active_step->phase->family->role != GraphRole::kText && !verifies_prefix(active_step->phase->family->role) && !has_mtp(active_step->phase->family->role)))
      return hyle::Status::FailedPrecondition("decode metadata requires an original submitted generation episode");
    const auto begin = Clock::now();
    const auto fail = [&](hyle::Status status) noexcept {
      discard_decode_ahead();
      cpu_ahead_prepare_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
      return status;
    };
    try {
      auto &prepared = *decode_ahead;
      const auto width = active_step->members.size();
      if (!width || width > prepared.members.size() || !active_step->input_tokens ||
          active_step->input_tokens > prepared.maximum_rows)
        return fail(hyle::Status::Unavailable("decode topology exceeds its cold workspace"));
      for (const auto &member : active_step->members)
        if (!member.next || member.session->failed || member.session->stopping ||
            member.base->length >= capacity - 1 || member.base_epoch == UINT64_MAX ||
            member.next->paged.pages.size() > prepared.page_stride)
          return fail(hyle::Status::Unavailable("decode metadata member cannot continue"));
      prune_decode_ahead(); prepared.clear_output();
      bool replaced = false;
      for (const auto &member : active_step->members) {
        auto index = prepared.members.size();
        // Preserve sibling subgroups: prefer this Session, then a vacant slot,
        // otherwise a bounded unused entry. Never replace a slot selected by
        // an earlier member of this same submitted episode.
        for (std::size_t i = 0; i < prepared.members.size(); ++i)
          if (!prepared.members[i].selected && prepared.members[i].ready && prepared.members[i].session.lock() == member.session) {
            index = i; break;
          }
        if (index == prepared.members.size())
          for (std::size_t i = 0; i < prepared.members.size(); ++i)
            if (!prepared.members[i].selected && !prepared.members[i].ready) { index = i; break; }
        if (index == prepared.members.size())
          for (std::size_t i = 0; i < prepared.members.size(); ++i)
            if (!prepared.members[i].selected) { index = i; break; }
        if (index == prepared.members.size())
          return fail(hyle::Status::Unavailable("decode metadata has no cold member slot"));
        auto &saved = prepared.members[index]; replaced = replaced || saved.ready;
        saved = {member.session, member.base_epoch + 1, member.base->length,
            member.next->length, active_step->phase->family->sampling->index, true, true};
        for (std::size_t page = 0; page < member.next->paged.pages.size(); ++page)
          prepared.candidate_ids[index * prepared.page_stride + page] =
              static_cast<std::int32_t>(member.next->paged.pages[page].page());
        prepared.selected.push_back(index);
      }
      if (replaced) ++cpu_ahead_discarded;
      for (const auto index : prepared.selected) prepared.members[index].selected = false;
      prepared.selected.clear(); prepared.refresh(); ++cpu_ahead_prepared;
      cpu_ahead_prepare_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
      return hyle::Status::Ok();
    } catch (const std::bad_alloc &) {
      return fail(hyle::Status::ResourceExhausted("optional decode metadata allocation"));
    } catch (...) {
      return fail(hyle::Status::Unavailable("optional decode metadata preparation"));
    }
  }

  bool take_decode_ahead(std::span<const EvaluationRequest> requests, GraphRole role,
      TextCompletion completion, std::size_t sampling_index, std::size_t tokens) noexcept {
    if (!decode_ahead || !decode_ahead->ready) return false;
    auto &prepared = *decode_ahead;
    prune_decode_ahead(); prepared.clear_output();
    if (requests.empty() || requests.size() > prepared.members.size() || !tokens ||
        tokens > prepared.maximum_rows || completion != TextCompletion::kEmitNext || is_mtp_prefill(role) ||
        (role != GraphRole::kText && !verifies_prefix(role) && !has_mtp(role))) return false;
    const auto miss = [&]() noexcept { prepared.clear_output(); return false; };
    // Standard prepare has already validated actual MTP repair/lag, inputs,
    // role and capacity. Only this exact published target-page prefix may
    // reuse the integers; no predicted draft state or permission is trusted.
    for (const auto &request : requests) {
      auto index = prepared.members.size();
      for (std::size_t i = 0; i < prepared.members.size(); ++i) {
        const auto &saved = prepared.members[i];
        if (!saved.ready || saved.sampling_index != sampling_index) continue;
        auto owner = saved.session.lock();
        if (owner != request.session || owner.owner_before(request.session) || request.session.owner_before(owner)) continue;
        if (owner->head.epoch() != saved.epoch || owner->length() <= saved.base_length ||
            owner->length() > saved.maximum_length || owner->head->paged.pages.size() > prepared.page_stride) break;
        bool same_pages = true;
        for (std::size_t page = 0; page < owner->head->paged.pages.size(); ++page)
          if (owner->head->paged.pages[page].page() != static_cast<std::uint32_t>(
              prepared.candidate_ids[i * prepared.page_stride + page])) { same_pages = false; break; }
        if (same_pages) index = i;
        break;
      }
      if (index == prepared.members.size() || prepared.members[index].selected) return miss();
      prepared.members[index].selected = true; prepared.selected.push_back(index);
    }
    prepared.old_indptr.push_back(0); prepared.old_query.resize(requests.size() + 1, 0);
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto index = prepared.selected[j]; const auto &owner = *requests[j].session;
      for (std::size_t row = 0; row < tokens; ++row)
        prepared.positions.push_back(static_cast<std::int32_t>(owner.length() + row));
      for (std::size_t page = 0; page < owner.head->paged.pages.size(); ++page)
        prepared.old_ids.push_back(prepared.candidate_ids[index * prepared.page_stride + page]);
      prepared.old_indptr.push_back(static_cast<std::int32_t>(prepared.old_ids.size()));
      prepared.old_last.push_back(static_cast<std::int32_t>((owner.length() - 1) % cache::Plan::page_rows + 1));
    }
    for (const auto index : prepared.selected) prepared.members[index] = {};
    prepared.selected.clear(); prepared.refresh(); ++cpu_ahead_hits;
    return true;
  }

  void prepare_prefill_evaluations(std::span<const EvaluationRequest> requests,
      inference::PreparationObservation *preparation = nullptr, GraphRole role = GraphRole::kText,
      TextCompletion completion = TextCompletion::kEmitNext) {
    std::size_t maximum = 0;
    for (const auto &request : requests) maximum = (std::max)(maximum, request.ids.size());
    const auto physical = !requests.empty() && requests[0].session &&
        (role == GraphRole::kText || is_mtp_prefill(role))
        ? prefill_capacity(maximum, completion, requests[0].session->sampling_configuration().index, requests.size(), role) : 0;
    prepare_evaluations(requests, preparation, role, completion, {}, 0, physical,
        physical ? RowPacking::kPrefill : RowPacking::kRectangular);
  }

  void prepare_verification_evaluations(std::span<const EvaluationRequest> requests,
      std::size_t physical_rows, inference::PreparationObservation *preparation = nullptr) {
    prepare_evaluations(requests, preparation, GraphRole::kVerify, TextCompletion::kEmitNext,
        {}, 0, physical_rows, RowPacking::kVerify);
  }

  // Device draft packets prepare the original verifier while their producer is
  // still active. Its Source and retirement remain owned by that same episode.
  void prepare_device_verifier(std::span<const EvaluationRequest> requests) {
    prepare_evaluations(requests, nullptr, GraphRole::kVerify);
    ++device_verification_handoffs;
  }

  void finish_device_verification_preparation() {
    if (!active_step || active_step->submitted || !active_step->prepared ||
        !active_step->phase || active_step->phase->family->role != GraphRole::kVerify ||
        !active_step->execution.incoming || active_step->execution.incoming->bound ||
        active_step->execution.incoming->bindings.size() != 1)
      throw std::logic_error("device verifier requires its original prepared target and token Source");
    auto &step = *active_step;
    const auto begin = Clock::now();
    auto &incoming = *step.execution.incoming;
    check(g::bind_sources(*step.execution.run, incoming.bindings));
    incoming.bound = true; incoming.bindings.clear();
    check(step.execution.prepare_replay({.mode = step.phase->slots[step.state_slot].replay_mode,
        .host_wait = g::HostWaitPolicy::kAllow}));
    step.replay_route = step.execution.run->replay_route();
    const auto replay = take(step.execution.run->replay_report());
    step.capture_recorded = replay.recorded_capture_range_count;
    step.capture_reused = replay.reused_capture_range_count;
    step.cold_batch_seconds += std::chrono::duration<double>(Clock::now() - begin).count();
  }

  // The producer has finished its proposal. Record its actual IDs on the
  // existing unpublished verifier, without transferring ownership to the caller.
  void bind_device_verification_inputs(GraphRole role, std::span<const std::int32_t> ids) {
    if (role != GraphRole::kVerify || !active_step || active_step->submitted ||
        active_step->members.size() != 1 || active_step->input_tokens != ids.size())
      throw std::logic_error("device verifier lost its original prepared target episode");
    active_step->members[0].verification_inputs.assign(ids.begin(), ids.end());
  }

  void start_device_verification_timing() {
    if (!active_step) throw std::logic_error("device verifier lost its original prepared target episode");
    active_step->step_start = Clock::now();
  }

  inference::RetirementObservation try_retire_owned_resources() noexcept {
    return resources_->retire(false);
  }

  template<bool recording_only = false>
  void prepare_evaluations(std::span<const EvaluationRequest> requests,
                           inference::PreparationObservation *preparation = nullptr,
                           GraphRole role = GraphRole::kText,
                           TextCompletion completion = TextCompletion::kEmitNext,
                           std::shared_ptr<StepSources> incoming = {}, std::size_t physical_slot = 0,
                            std::size_t physical_rows = 0, RowPacking packing = RowPacking::kRectangular,
                            const RecordingStepInputs *recording = nullptr) {
    const auto predecessor = incoming && active_step ? active_step : std::shared_ptr<StepState>{};
    const bool relay = bool(incoming);
    const bool recording_imported = recording_only && recording && recording->phase && recording->phase->imported_sources;
    if constexpr (recording_only) {
      const bool fixed_prefill = packs_prefill(packing) && !cache_plan.draft_cache &&
          role == GraphRole::kText && physical_rows == bounded_prefill_rows &&
          requests.size() == 1 && requests[0].ids.size() == physical_rows;
      if (!recording || !recording->phase || recording->slot >= recording->phase->slots.size() || relay ||
          (cache_plan.draft_cache ? (role != GraphRole::kMtpRound && role != GraphRole::kMtpCatchupRound &&
              role != GraphRole::kMtpSingle && role != GraphRole::kMtpCatchupSingle &&
              role != GraphRole::kMtpContinuation) : role != GraphRole::kText) ||
          (!fixed_prefill && (completion != TextCompletion::kEmitNext ||
          requests.size() != 1 || requests[0].ids.size() != 1 || physical_rows || packing != RowPacking::kRectangular)))
        throw std::runtime_error("cold recording requires its original single-request decode carrier");
      if (!g::detail::RecordingAccess::active(*take(recording->phase->slots[recording->slot].carrier.borrow())))
        throw std::runtime_error("cold recording requires Graph's non-executable mode");
    }
    const bool mtp_relay = relay && incoming->mtp_full_candidate;
    const bool prefill_relay = relay && incoming->prefill && packs_prefill(packing) &&
        role == GraphRole::kText && requests.size() == 1 && !cache_plan.draft_cache;
    const bool device_mtp = mtp_relay && incoming->mtp_device_choice;
    const bool device_input_choice = role == GraphRole::kMtpContinuation;
    if (device_input_choice && !recording_only && (!device_mtp || !predecessor))
      throw std::runtime_error("device MTP input selection requires its actual pending predecessor");
    if (!supports_completion(role, completion))
      throw std::runtime_error("state-only evaluation requires Text or teacher-forced MTP input");
    if (requests.empty() || requests.size() > cache_plan.active_sequences)
      throw std::runtime_error("batch width exceeds the configured active sequence capacity");
    if (!relay && (carrier_borrowed || active_step))
      throw std::runtime_error("the original execution carrier has not retired");
    const bool ordinary_relay = relay && !mtp_relay && !cache_plan.draft_cache && role == GraphRole::kText;
    const bool complete_mtp_relay = mtp_relay && cache_plan.draft_cache && requests.size() == 1 &&
        role == (device_mtp ? GraphRole::kMtpContinuation : GraphRole::kMtpCatchupRound) && (!predecessor || (predecessor->phase &&
            drafts_token(predecessor->phase->family->role) && predecessor->input_tokens == max_drafts()+1)) &&
        incoming->extra_outputs.size() == (device_mtp ? 2u : 1u) && incoming->extra_outputs[0].valid() &&
        incoming->successor_states.size() == geometry.recurrent_layers() * 2;
    if (relay && (successor_step || scheduler || cache_plan.in_flight_batches < 2 ||
        (!ordinary_relay && !complete_mtp_relay && !prefill_relay) || (!prefill_relay && completion != TextCompletion::kEmitNext) || physical_slot >= execution_slot_count ||
        incoming->boundaries.size() != requests.size() || !incoming->submission.valid() || (!prefill_relay && !incoming->primary_output.valid()) ||
        incoming->states.size() != geometry.recurrent_layers() * 2 + (mtp_relay ? 1u : 0u) ||
        incoming->pages.size() != draft_bank + (mtp_relay ? 2u : 0u) ||
        (predecessor && (predecessor->execution.sources != incoming || predecessor->cancelled || !predecessor->submitted ||
            predecessor->members.size() != requests.size()))))
      throw std::runtime_error("decode successor requires one exact complete predecessor and independent carriers");
    if (!requests.front().session || requests.front().session->model_ != this)
      throw std::runtime_error("batch session belongs to another model");
    if (!requests.front().session->head)
      throw std::runtime_error("batch session has no live committed head");
    const auto &prepared_sampling = requests.front().session->sampling_configuration();
    const auto sampling_index = prepared_sampling.index;
    const auto &selected_sampler = prepared_sampling.configuration;
    const auto token_source = requests.size() == 1 ? requests.front().verification_tokens : std::shared_ptr<StepSources>{};
    if (token_source && (recording_only || relay || role != GraphRole::kVerify ||
        packing != RowPacking::kRectangular || completion != TextCompletion::kEmitNext ||
        selected_sampler.enabled || !selected_sampler.modifiers.empty() ||
        !token_source->submission.valid() || !token_source->primary_output.valid()))
      throw std::runtime_error("device verifier tokens require one original greedy target episode");
    if (mtp_relay && (selected_sampler.enabled || !selected_sampler.modifiers.empty()))
      throw std::runtime_error("MTP full-candidate successor requires its original greedy sampling configuration");
    const auto supplied_tokens = relay && !prefill_relay ? std::size_t{1} : requests.front().ids.size();
    const bool packed_prefill = packs_prefill(packing), packed_verify = packs_verify(packing);
    const bool packed_target = packs_target(packing);
    if (packed_target != (physical_rows != 0) || (packed_verify && role != GraphRole::kVerify))
      throw std::runtime_error("target row capacity differs from its declared packing");
    // Mixed K1 members share a fixed two-row draft carrier, but keep their
    // actual one/two-row append and original mathematical role in the ledger.
    const auto first_lag = drafts_token(role) && requests.size() > 1 && requests.front().session->head->draft
        ? requests.front().session->head->draft->lag : 0;
    const bool compact_draft = !relay && !packed_prefill && requests.size() > 1 && drafts_token(role) &&
        std::ranges::any_of(requests, [&](const auto &request) {
          return request.session && request.session->head && request.session->head->draft &&
              request.session->head->draft->lag != first_lag;
        });
    if (compact_draft && first_lag != draft_lag(role))
      throw std::runtime_error("MTP role differs from its first member snapshot");
    if (compact_draft) role = GraphRole::kMtpCatchupRound;
    if (compact_draft) packing = RowPacking::kDraft;
    const bool compound_round = requests.size() == 1 &&
        (role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound || role==GraphRole::kMtpContinuation) &&
        greedy_draft_chain_enabled(selected_sampler) && cache_plan.in_flight_batches >= 3;
    auto tokens = drafts_token(role) ? (compound_round ? max_drafts() + 1 : std::size_t{2}) : supplied_tokens;
    if (packed_target) for (const auto &request : requests) tokens = (std::max)(tokens, request.ids.size());
    const auto physical_tokens = packed_target ? physical_rows : tokens;
    if (compact_draft && (completion != TextCompletion::kEmitNext ||
        !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, packing)))
      throw std::runtime_error("mixed MTP draft rows require their original cold compact family");
    if (packed_prefill && ((relay && !prefill_relay) || (role != GraphRole::kText && !is_mtp_prefill(role)) ||
        physical_tokens < tokens || !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, RowPacking::kPrefill)))
      throw std::runtime_error("prefill extent is outside its original cold packed family");
    if (packed_verify && (relay || physical_tokens < tokens || completion != TextCompletion::kEmitNext ||
        !find_child(requests.size(), physical_tokens, role, false, completion, sampling_index, packing)))
      throw std::runtime_error("verification extent is outside its original cold prefix capacity");
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("evaluation must use the configured complete target/draft state");
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      if (!request.session || request.session->model_ != this)
        throw std::runtime_error("batch session belongs to another model");
      const auto &session = *request.session;
      const auto &stop_policy = inference::request_stop_policy(request.stop_policy);
      stop_policy.validate(vocabulary);
      if (&session.sampling_configuration() != &prepared_sampling) throw std::runtime_error("batch requires one cold sampling configuration");
      if (!request.conditional_logits.empty()) {
        const auto rows = request.conditional_logits.rows();
        auto status = role == GraphRole::kVerify && completion == TextCompletion::kEmitNext
            ? prefix_logits_inputs_metadata_status(selected_sampler, request.logits, rows, request.ids.size(), vocabulary)
            : hyle::Status::InvalidArgument("conditional evaluation logits require an emitting Verify episode");
        if (status.ok() && (request.ids.empty() || !prefix_sampling_programs.contains(
            {request.ids.size() - 1, sampling_index, true})))
          status = hyle::Status::InvalidArgument("conditional verification width was not cold prepared");
        check_request_logits_metadata(std::move(status), request.request_key);
        validate_prefix_logits_input_leases(selected_sampler, request.logits, rows);
      } else if (completion == TextCompletion::kEmitNext)
        validate_request_logits_inputs(selected_sampler, request.logits, vocabulary, request.request_key);
      if (session.stopping || session.failed || (session.pending && (!predecessor || session.pending != predecessor.get())))
        throw std::runtime_error("batch session is stopped, failed or already pending");
      session.head.preflight_commit();
      const auto *base = predecessor ? predecessor->members[j].next.get() : session.head.get();
      if (relay) {
        auto owner = incoming->boundaries[j].session.lock();
        const auto epoch = session.head.epoch();
        if (!owner || owner != request.session || !base || incoming->boundaries[j].candidate != base ||
            epoch > UINT64_MAX - (predecessor ? 2u : 1u) ||
            incoming->boundaries[j].epoch != epoch + (predecessor ? 1u : 0u) ||
            (predecessor && predecessor->members[j].session != request.session) ||
            (!prefill_relay && !request.ids.empty() && (predecessor || request.ids.size() != 1 || request.ids[0] != base->next_id)))
          throw std::runtime_error("successor Source differs from its exact Session candidate and epoch");
      }
      const auto member_tokens = packed_target ? request.ids.size() : tokens;
      if (!member_tokens || (!packed_target && !relay && request.ids.size() != supplied_tokens) ||
          member_tokens > capacity - base->length)
        throw std::runtime_error("batch requires valid nonempty chunks within each remaining context");
      validate_mtp_snapshot(*session.head);
      if (has_mtp(role)) {
        const bool lag_matches = mtp_relay ? base->draft && base->draft->frontier == incoming->boundaries[j].draft_frontier &&
            base->length == incoming->boundaries[j].length && base->length - base->draft->frontier == 1 :
            compact_draft ? session.head->draft->lag <= 1 : session.head->draft->lag == draft_lag(role);
        // The real empty Head remains valid. A recording role describes only
        // physical parameters; its unpublished candidate is never evaluated.
        if ((!recording_only && !lag_matches) ||
            (!is_mtp_prefill(role) && supplied_tokens != 1))
          throw std::runtime_error("MTP role, pending token or catch-up lag does not match its snapshot");
      }
      for (std::size_t other = 0; other < j; ++other)
        if (requests[other].session == request.session)
          throw std::runtime_error("batch repeats a mutable session");
      for (auto id : request.ids) if (id < 0 || id >= vocabulary)
        throw std::runtime_error("token ID outside vocabulary");
      // Raw Text callers have no turn quota. Still reject their actual draw
      // before creating an episode, borrowing the carrier or writing inputs.
      if (role == GraphRole::kText && completion == TextCompletion::kEmitNext && selected_sampler.enabled)
        (void)base->sampling.advanced(1);
    }
    // A suffix first seen at a request boundary may compile a new exact family.
    // Charge that real synchronous work to this step, before the carrier loan.
    const auto step_start = Clock::now();
    const ExecutionShape phase_key{requests.size(), physical_tokens, role, completion, sampling_index, relay || recording_imported, packing};
    // Reuse only this call's original complete Phase metadata. Diagnostic
    // execution keeps its existing default/probe preparation sequence.
    const auto resolve_prepared_phase = [&]() -> Phase * {
      if (!layer0_probe.empty()) return nullptr;
      const auto found = phases.find(phase_key);
      return found == phases.end() ? nullptr : found->second.get();
    };
    auto *prepared_phase = resolve_prepared_phase();
    if constexpr (recording_only) {
      if (prepared_phase != recording->phase)
        throw std::runtime_error("cold recording Phase does not belong to this exact model/shape");
    }
    double shape_admission_seconds = 0;
    if (!packed_prefill && (role == GraphRole::kText || is_mtp_prefill(role)) &&
        !prepared_phase && !find_child(requests.size(), tokens, role, false, completion, sampling_index)) {
      admit_exact_shape(tokens, requests.size(), role, completion, sampling_index);
      shape_admission_seconds = std::chrono::duration<double>(Clock::now() - step_start).count();
      // Exact admission can retire/erase Phase and Family inventory entries.
      // Re-resolve from the current map; no pointer is retained across calls.
      prepared_phase = resolve_prepared_phase();
    }
    const bool ahead = !recording_only && !relay && !packed_target && take_decode_ahead(requests, role, completion, sampling_index, tokens);
    auto candidate = std::make_shared<StepState>(); candidate->execution.configure(draft_bank + 2);
    if (!candidate->members.try_reserve(requests.size())) throw std::bad_alloc{};
    for (std::size_t j = 0; j < requests.size(); ++j) {
      const auto &request = requests[j];
      candidate->members.push_back({.session = request.session, .owner_state = &request.session->head,
          .base = predecessor ? predecessor->members[j].next.get() : request.session->head.get(),
          .base_epoch = request.session->head.epoch() + (predecessor ? 1u : 0u),
          .stop_policy = request.stop_policy,
          .logits = completion == TextCompletion::kEmitNext ? request.logits : LogitsInputs{},
          .conditional_logits = request.conditional_logits,
          .input_tokens = packed_target ? request.ids.size() : tokens});
      candidate->members.back().first_input = relay && !prefill_relay ? (predecessor ? -1 : request.session->head->next_id) : request.ids.front();
      if (role == GraphRole::kVerify)
        candidate->members.back().verification_inputs.assign(request.ids.begin(), request.ids.end());
    }
    if (relay) {
      candidate->execution.incoming = std::make_unique<IncomingStepSources>();
      candidate->execution.incoming->owner = std::move(incoming);
    } else if (token_source) {
      candidate->execution.incoming = std::make_unique<IncomingStepSources>();
      candidate->execution.incoming->owner = token_source;
      candidate->execution.incoming->bindings.reserve(1);
    }
    auto &step = *candidate;
    if (predecessor) successor_step = candidate; else active_step = candidate;
    ++carrier_borrowed; step.execution.candidate_admission = &carrier_borrowed;
    if (predecessor) step.execution.predecessor_episode = {predecessor, &predecessor->execution};
    step.step_start = step_start; step.input_tokens = tokens;
    for (auto &member : step.members) {
      member.input_row_begin = step.active_input_rows;
      step.active_input_rows = cache::add(step.active_input_rows, member.input_tokens);
    }
    step.shape_admission_seconds = shape_admission_seconds;
    if (!predecessor) for (auto &member : step.members) member.session->pending = &step;
    try {
      const auto width = requests.size(), slot = recording_only ? recording->slot : relay ? physical_slot : slot_index;
      if (!layer0_probe.empty()) prepare_child(tokens, false, width, role, completion, sampling_index);
      if (predecessor) {
        const auto found = phases.find({width, physical_tokens, role, completion, sampling_index, true, packing});
        if (found == phases.end() || !found->second->slots[slot].carrier.initialized())
          throw std::runtime_error("successor carrier must be cold-prepared before its predecessor submission");
      }
      const auto phase_prepare_start = Clock::now();
      auto &phase = prepared_phase ? *prepared_phase
          : prepare_child(physical_tokens, !layer0_probe.empty(), width, role, completion, sampling_index, relay, packing);
      step.phase_prepare_seconds = prepared_phase ? 0.0
          : std::chrono::duration<double>(Clock::now() - phase_prepare_start).count();
      step.phase_creations = prepared_phase ? 0 : 1;
      auto &carrier = phase.slots[slot].carrier;
      auto &program_slot = phase.slots[slot]; auto &io = program_slot.data;
      // A rejected successor never claims or clears its predecessor's run.
      if (predecessor && predecessor->phase == &phase && predecessor->state_slot == slot)
        throw std::runtime_error("successor cannot reuse its live predecessor carrier");
      const bool initialize = !carrier.initialized();
      const auto slot_initialize_start = Clock::now();
      initialize_slot(phase, slot);
      step.slot_initialize_seconds = initialize
          ? std::chrono::duration<double>(Clock::now() - slot_initialize_start).count() : 0.0;
      step.slot_initializations = initialize ? 1 : 0;
      auto &run = *take(carrier.borrow());
      if constexpr (recording_only) {
        if (!g::detail::RecordingAccess::active(run))
          throw std::runtime_error("cold recording lost Graph's non-executable mode");
      }
      step.phase = &phase; step.execution.run = &run; step.execution.carrier = &carrier; step.state_slot = slot;
      step.execution.state_transactions = &phase.slots[slot].transactions;
      if (phase.draft_chain_steps) step.execution.configure(phase.append_nodes.size());
      step.execution.nodes.assign(phase.append_nodes.begin(), phase.append_nodes.end());
      if constexpr (recording_only) if (recording_imported) {
        if (recording->states.size() != phase.state_sources.size() || (!packed_prefill && !recording->token.valid()))
          throw std::runtime_error("cold recording lacks its original producer coordinates");
        if (!packed_prefill) check(g::detail::RecordingAccess::bind_input(run, phase.token_port, recording->token));
        if (has_mtp(role))
          check(g::detail::RecordingAccess::bind_input(run, phase.catchup_token_port, recording->catchup));
        if (phase.predecessor_prefix_port)
          check(g::detail::RecordingAccess::bind_input(run, phase.predecessor_prefix_port, recording->prefix));
        for (std::size_t bank = 0; bank < recording->states.size(); ++bank)
          check(g::detail::RecordingAccess::bind_store(run, phase.state_sources[bank].store, recording->states[bank]));
      }
      // Batch owns the prior offsets/control episode and withdraws it before
      // refreshing its original metadata writer in batch::prepare below.
      auto &packed_ids = input_scratch.packed_ids, &positions = input_scratch.positions;
      packed_ids.clear(); positions.clear();
      for (std::size_t j = 0; j < width; ++j) {
        const auto &request = requests[j];
        if (!relay && width != 1) packed_ids.insert(packed_ids.end(), request.ids.begin(), request.ids.end());
        if (!ahead && !packed_target) for (std::size_t row = 0; row < step.members[j].input_tokens; ++row)
          positions.push_back(static_cast<std::int32_t>(step.members[j].base->length + row));
      }
      for (std::size_t j = 0; j < width; ++j) {
        if (!phase.logits_bias_ports.empty()) check(run.bind(phase.logits_bias_ports[j], requests[j].logits.bias.description, requests[j].logits.bias.lease));
        if (!phase.logits_allowed_ports.empty()) check(run.bind(phase.logits_allowed_ports[j], requests[j].logits.allowed.description, requests[j].logits.allowed.lease));
        if (!phase.logits_counts_ports.empty()) check(run.bind(phase.logits_counts_ports[j], requests[j].logits.counts.description, requests[j].logits.counts.lease));
      }
      if (packed_target) {
        if (width == 1) packed_ids.assign(requests[0].ids.begin(), requests[0].ids.end());
        const auto carrier_rows = cache::multiply(width, physical_tokens);
        packed_ids.resize(carrier_rows, 0); // Only the inactive carrier suffix is scratch.
        positions.resize(carrier_rows);
        auto &row_inputs = input_scratch.row_inputs;
        auto &offsets = input_scratch.query_indptr;
        row_inputs.clear(); offsets.resize(width + 1);
        for (const auto &member : step.members) row_inputs.push_back({member.base->length, member.input_tokens});
        const auto row_facts = inference::pack_sequence_rows(row_inputs, carrier_rows, offsets, io.last_id_values, positions);
        if (row_facts.active_rows != step.active_input_rows || row_facts.max_member_rows != step.input_tokens)
          throw std::logic_error("packed target rows differ from the admitted episode");
        const auto history_rows = static_cast<std::size_t>(geometry.conv_kernel - 1);
        const auto history_count = cache::multiply(width, history_rows);
        if (packed_verify) {
          (void)inference::pack_sequence_prefixes(offsets, phase.prefix_rows, history_rows,
              io.prefix_id_values, io.history_select_values);
          io.prefix_ids.write<std::int32_t>(io.prefix_id_values);
        } else for (std::size_t j = 0; j < width; ++j) {
          const auto begin = step.members[j].input_row_begin;
          const auto length = step.members[j].input_tokens;
          for (std::size_t row = 0; row < history_rows; ++row) {
            const auto tail = length + row;
            const auto source = tail < history_rows ? j * history_rows + tail
                : history_count + begin + tail - history_rows;
            io.history_select_values[j * history_rows + row] = static_cast<std::int32_t>(source);
          }
        }
        if (packed_prefill && !io.ordinary_io.packet_last_ids)
          io.last_ids.write<std::int32_t>(io.last_id_values);
        if (!io.ordinary_io.packet_history_select)
          io.history_select.write<std::int32_t>(io.history_select_values);
      }
      const auto token_values = packed_target || width != 1
          ? std::span<const std::int32_t>(packed_ids) : requests[0].ids;
      const auto position_values = ahead
          ? std::span<const std::int32_t>(decode_ahead->positions.data(), decode_ahead->positions.size())
          : std::span<const std::int32_t>(positions);
      if (!io.ordinary_io.packet()) {
        if (!relay && !recording_imported && !token_source) io.tokens.write<std::int32_t>(token_values);
        if(!device_input_choice)io.positions.write<std::int32_t>(position_values);
      }
      if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) {
        auto &identities = input_scratch.sampling_identities;
        identities.clear();
        for (const auto &member : step.members) {
          if (role != GraphRole::kText) (void)member.base->sampling.advanced(drafts_token(role) ? 3 : 1);
          for (std::size_t draw = 0; draw < (phase.batched_text_sampling ? 1u : 3u); ++draw) {
            // Batched ordinary Text binds one original slot0 key per row.
            // Other non-round roles retain three slots and consume slot0.
            const auto identity = member.base->sampling.identity(drafts_token(role) ? draw : 0);
            identities.insert(identities.end(), identity.begin(), identity.end());
          }
        }
        if (!io.ordinary_io.packet_sampling_identities)
          io.sampling_identities.write<std::uint32_t>(identities);
      }
      if (io.ordinary_io.packet()) {
        if (relay && !prefill_relay) throw std::logic_error("host input packet cannot replace a device relay token");
        // A multi-chunk current turn can bind frozen request input. All other
        // inputs retain the borrowed original slot's synchronous packet.
        const auto &request = requests[0];
        // Captures name this slot's original packet. Per-request uploaded
        // slices have different addresses/offsets and cannot share that key.
        const bool uploaded = packed_prefill && width == 1 &&
            phase.slots[slot].replay_mode == g::ReplayMode::kCommands &&
            !selected_sampler.enabled && request.prefill_packets &&
            take(request.prefill_packets->prepare(io.input_packet, request.prefill_prompt,
                request.prefill_offset, step.members[0].base->length, physical_tokens,
                static_cast<std::size_t>(geometry.conv_kernel - 1), request.prefill_turn,
                execution.device(), execution, allocation_budget, execution_budget));
        if (packed_prefill && width == 1) {
          const std::array ports{phase.token_port, phase.position_port, phase.last_ids_port, phase.history_select_port};
          const std::array views{io.input_packet.tokens(), io.input_packet.positions(),
              io.input_packet.last_ids(), io.input_packet.history_select()};
          for (std::size_t port = 0; port < ports.size(); ++port) {
            if (uploaded) check(run.bind(ports[port], views[port].description(),
                take(request.prefill_packets->port(io.input_packet, request.prefill_offset, port))));
            else check(run.bind(ports[port], views[port]));
          }
        }
        if (!uploaded) check(io.input_packet.write(token_values, position_values, {
            .last_ids = io.ordinary_io.packet_last_ids
                ? std::span<const std::int32_t>(io.last_id_values) : std::span<const std::int32_t>{},
            .history_select = io.ordinary_io.packet_history_select
                ? std::span<const std::int32_t>(io.history_select_values) : std::span<const std::int32_t>{},
            .sampling_identities = io.ordinary_io.packet_sampling_identities
                ? std::span<const std::uint32_t>(input_scratch.sampling_identities) : std::span<const std::uint32_t>{}}));
      }
      if (ahead) decode_ahead->positions.clear();
      const auto cold_start = Clock::now();
      const auto begin_member = [&](std::size_t j) {
        // The common target cohort already reserved its own tail metadata.
        // Keep the remaining original banks before the first model candidate.
        if (!j) for (auto &bank : std::span(step.execution.tails.data(), step.execution.tails.size()).subspan(draft_bank))
          if (!bank.try_reserve(width)) throw std::bad_alloc{};
        auto &member = step.members[j]; auto &session = *member.session;
        member.next = reserve_state(session.head);
        if (!member.next_tails.try_resize(draft_bank)) throw std::bad_alloc{};
        member.next->length = member.base->length + member.input_tokens;
        member.next->sampling = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify
            ? member.base->sampling.advanced(drafts_token(role) ? 3 : 1) : member.base->sampling;
      };
      const auto member_at = [&](std::size_t j) noexcept -> inference::PagedAppendMember {
        auto &member = step.members[j]; auto &session = *member.session;
        return {member.base->paged, member.next ? &member.next->paged : nullptr, member.layout,
            member.base->length, member.input_tokens,
            relay ? std::span<c::PagedPageAppender>{} : std::span(session.tails.data(), session.tails.size()),
            !relay && session.tail_epoch == member.base_epoch};
      };
      const auto invalidate_tail = [&](std::size_t j) noexcept {
        if (!relay) step.members[j].session->tail_epoch = 0;
      };
      const c::HostBatchLayout previous_layout = ahead ? c::HostBatchLayout{
          {decode_ahead->old_query.data(), decode_ahead->old_query.size()},
          {decode_ahead->old_indptr.data(), decode_ahead->old_indptr.size()},
          {decode_ahead->old_ids.data(), decode_ahead->old_ids.size()},
          {decode_ahead->old_last.data(), decode_ahead->old_last.size()}} : c::HostBatchLayout{};
      if (device_input_choice) {
        prepare_device_mtp_inputs(step, recording_only);
      } else {
      io.attention_layout = take(inference::prepare_paged_append(attention_schema, width,
          inference::PagedAppendBanks{page_inventory,
              std::span(step.execution.dirty.data(), draft_bank),
              std::span(step.execution.tails.data(), draft_bank),
              std::span(step.execution.appends.data(), draft_bank)},
          input_scratch.paged_appends, member_at,
          {.capacity_rows = capacity,
           .incoming = relay ? std::span(step.execution.incoming->owner->pages).first(draft_bank)
               : std::span<g::batch::PagedAppendSource>{},
           .pending_predecessor = bool(predecessor),
           .previous_layout = ahead && width != 1 ? &previous_layout : nullptr,
           .append_options = {.resource_budget = cache_budget}},
          invalidate_tail, begin_member));
      if (ahead) decode_ahead->clear_output(); // Schema copied the original borrowed integers.
      if (has_mtp(role)) prepare_draft_append<recording_only>(step, role);
      else if (phase.recursive_draft_repair) prepare_recursive_repair_append(step);
      if (phase.draft_chain_steps) prepare_round_draft_chain(step);
      }
      auto &transactions = phase.slots[slot].transactions;
      for (std::size_t index = 0; index < phase.state_banks.size(); ++index) {
        const auto &bank = phase.state_banks[index];
        auto state_sources = transactions.sources(index);
        const bool is_hidden = bank.kind == StateBankKind::kHidden;
        const bool recurrent = bank.kind == StateBankKind::kRecurrent;
        const auto bytes = transactions.writer_bytes(index);
        if (!relay && !recording_imported) for (std::size_t j = 0; j < width; ++j) {
          if (is_hidden) {
            if (phase.recursive_draft_repair) {
              state_sources[0] = step.members[0].base->carry_hidden;
              state_sources[1] = step.members[0].base->draft->lag
                  ? step.members[0].base->prior_hidden : step.members[0].base->carry_hidden;
            } else if (has_mtp(role)) {
              const bool chronological = width == 1 && role == GraphRole::kMtpCatchupRound && packing == RowPacking::kRectangular;
              state_sources[chronological ? width + j : j] = step.members[j].base->carry_hidden;
              if (draft_lag(role)) state_sources[chronological ? j : width + j] = compact_draft && !step.members[j].base->draft->lag
                  ? step.members[j].base->carry_hidden : recording_only
                      ? step.members[j].base->carry_hidden : step.members[j].base->prior_hidden;
            }
          } else {
            const auto &state = step.members[j].base->state[bank.layer];
            state_sources[j] = recurrent ? state.recurrent : state.history;
          }
        }
        for (auto &writer : transactions.writers(index))
          writer = is_hidden ? take(hidden_pool.allocate(bytes, cache::state_pool_alignment))
              : take(recurrent ? state_pools[bank.layer].acquire_recurrent(bytes)
                  : state_pools[bank.layer].acquire_history(bytes));
        check(transactions.stage(index));
      }
      if (preparation) { preparation->slot = slot; preparation->phase_slot_prior_uses = phase.slots[slot].preparation_uses; }
      const auto batch_start = Clock::now();
      const auto append_banks = draft_bank + (has_mtp(role) || phase.recursive_draft_repair ? 2u : 0u) + 2 * phase.draft_chain_steps;
      if (device_input_choice) {
        hyle::util::ScopeGuard release_layouts{[&]() noexcept {
          for (auto &entry : io.batch_layouts) entry.layout = {};
        }};
        for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i)
          io.batch_layouts[i].layout = step.members.front().target_capacity_layout;
        if(phase.draft_chain_steps)for(std::size_t i=0;i<phase.draft_chain_steps;++i)
          io.batch_layouts[phase.attention_nodes.size()+i].layout=step.members.front().draft_chain_layouts[i];
        else io.batch_layouts.back().layout = step.members.front().draft_capacity_layout;
        check(step.execution.prepare_length_appends(phase.family->graph,io.batch_layouts,0,append_banks,
            std::span<const V>(phase.append_cache_inputs).first(append_banks), {.host_workspace = execution.staging()}));
      } else {
        hyle::util::ScopeGuard release_layouts{[&]() noexcept {
          for (auto &entry : io.batch_layouts) entry.layout = {};
        }};
        for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i) io.batch_layouts[i].layout = io.attention_layout;
        if (drafts_token(role) && !phase.draft_chain_steps) io.batch_layouts.back().layout = step.draft_layout;
        check(step.execution.prepare_appends(phase.family->graph, io.batch_layouts, 0, append_banks,
            std::span<const V>(phase.append_cache_inputs).first(append_banks), {.host_workspace = execution.staging()}));
      }
      if (relay) {
        auto &input = *step.execution.incoming; auto &previous = *input.owner;
        input.bindings.reserve(1 + previous.states.size() + previous.pages.size() + previous.draft_stage_sources.size() + (device_mtp ? 2u : mtp_relay ? 1u : 0u));
        if (!prefill_relay) input.bindings.push_back(previous.primary_output.reader_binding(phase.token_port));
        if (device_mtp) {
          input.bindings.push_back(previous.extra_outputs[0].reader_binding(phase.predecessor_prefix_port));
          input.bindings.push_back(previous.extra_outputs[1].reader_binding(phase.catchup_token_port));
        } else if (mtp_relay) input.bindings.push_back(previous.extra_outputs[0].reader_binding(phase.catchup_token_port));
        const bool ready_states = !device_mtp && previous.published_states.size() == previous.states.size() * width + (mtp_relay ? phase.prefix_rows-1 : 0u) &&
            std::ranges::all_of(previous.published_states, [](const auto &lease) { return lease.valid(); });
        for (std::size_t bank = 0; bank < previous.states.size(); ++bank) {
          const auto &source = mtp_relay && bank < previous.successor_states.size()
              ? previous.successor_states[bank] : previous.states[bank];
          const auto count = device_mtp || (mtp_relay && bank == previous.successor_states.size()) ? phase.prefix_rows : width;
          input.bindings.push_back(take(source.reader_binding(run, phase.state_graph(),
              phase.state_sources[bank].store, ready_states ? std::span<const c::ReadLease>(previous.published_states).subspan(bank * width, count)
                  : std::span<const c::ReadLease>{})));
        }
        for (std::size_t bank = 0; bank < previous.pages.size(); ++bank)
          input.bindings.push_back(take(g::batch::append_binding(run, phase.append_nodes[bank], previous.pages[bank])));
        for(std::size_t bank=0;bank<previous.draft_stage_sources.size();++bank)
          input.bindings.push_back(take(g::batch::append_binding(run,phase.append_nodes[draft_bank+2+bank],previous.draft_stage_sources[bank])));
      }
      if (token_source)
        step.execution.incoming->bindings.push_back(token_source->primary_output.reader_binding(phase.token_port));
      if (!token_source) check(step.execution.bind_consumers(phase.state_graph()));
      else if (step.execution.state_transactions)
        check(step.execution.state_transactions->bind(run, phase.state_graph()));
      if (packed_target) {
        check(g::batch::bind_query_rows(run, phase.attention_nodes.front(), phase.sequence_offsets_port));
      }
      const auto batch_end = Clock::now();
      if (token_source) { /* Final replay preparation follows this producer's retirement. */ }
      else if (predecessor && !prefill_relay) check(step.execution.prepare_successor_replay());
      else check(step.execution.prepare_replay({.mode = phase.slots[slot].replay_mode, .host_wait = g::HostWaitPolicy::kAllow}));
      const auto replay_end = Clock::now();
      if (preparation) {
        preparation->layout_seconds = std::chrono::duration<double>(batch_start - cold_start).count();
        preparation->batch_prepare_seconds = std::chrono::duration<double>(batch_end - batch_start).count();
        preparation->replay_prepare_seconds = std::chrono::duration<double>(replay_end - batch_end).count();
      }
      if (!token_source) {
        step.replay_route = run.replay_route();
        const auto replay_report = take(run.replay_report());
        step.capture_recorded = replay_report.recorded_capture_range_count;
        step.capture_reused = replay_report.reused_capture_range_count;
      }
      step.cold_batch_seconds = std::chrono::duration<double>(Clock::now() - cold_start).count();
      ++phase.slots[slot].preparation_uses; step.prepared = true;
      if constexpr (recording_only) check(step.execution.mark_recording_prepared());
    } catch (inference::EvaluationFailure &failure) {
      if (step.execution.unsubmitted_reuse_declined()) failure.observe_unsubmitted_reuse_refusal();
      failure.observe_retirement(predecessor ? abort_successor_episode() : abort_episode());
      throw;
    } catch (...) { if (predecessor) (void)abort_successor_episode(); else (void)abort_episode(); throw; }
  }

  void prepare_evaluation(Session &session, std::span<const std::int32_t> ids,
                          inference::PreparationObservation *preparation = nullptr,
                          GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext,
                          const inference::RequestStopPolicy *stop_policy = nullptr, const LogitsInputs &logits = {}) {
    const std::array requests{EvaluationRequest{session.shared_from_this(), ids, stop_policy, logits}};
    prepare_evaluations(requests, preparation, role, completion);
  }

  StepState &prepare_relay_evaluations(std::span<const EvaluationRequest> requests,
      std::shared_ptr<StepSources> previous, std::size_t physical_slot, GraphRole role = GraphRole::kText) {
    prepare_evaluations(requests, nullptr, role, TextCompletion::kEmitNext, std::move(previous), physical_slot);
    return successor_step ? *successor_step : *active_step;
  }

  // Inspect the actual prepared Phase and its original carrier; this only
  // finds a capture candidate and does not grant submission or Source rights.
  bool has_relay_successor_capture(const StepState *producer, const StepSources &source,
      std::size_t requests, std::size_t outputs, GraphRole role, std::size_t sampling_index,
      std::size_t slot, bool device_mtp) {
    Phase *phase = producer ? producer->phase : nullptr;
    if (device_mtp) {
      if (requests != 1 || !source.mtp_device_choice)
        throw std::logic_error("device successor lost its original K1 selection Source");
      const auto found = phases.find({requests, outputs, role, TextCompletion::kEmitNext,
          sampling_index, true, RowPacking::kRectangular});
      if (found == phases.end()) return false;
      phase = found->second.get();
    }
    if (!producer || !phase || !phase->imported_sources || phase->family->role != role ||
        phase->family->completion != TextCompletion::kEmitNext ||
        phase->family->requests != requests || slot >= phase->slots.size())
      throw std::logic_error("successor lost its original imported Phase geometry");
    auto &carrier = phase->slots[slot].carrier;
    if (!carrier.initialized() || carrier.closing() || !carrier.retired()) return false;
    auto *run = take(carrier.borrow());
    return take(g::detail::ReplayAccess::has_capture_candidate(*run,
        source.primary_output.reader_binding(phase->token_port)));
  }

  // A prefill successor can use only a prepared, idle original carrier with
  // a replay shape matching the real tail. The caller retains the Source.
  std::optional<std::size_t> prefill_successor_slot(std::size_t count, std::size_t physical,
      GraphRole role, TextCompletion completion, std::size_t sampling_index) {
    if (active_step->replay_route == g::ReplayRoute::kCaptured && count != physical)
      return std::nullopt;
    const auto found = phases.find({1, physical, role, completion, sampling_index, true, RowPacking::kPrefill});
    if (found == phases.end()) return std::nullopt;
    const auto slot = (active_step->state_slot + 1) % execution_slot_count;
    auto &carrier = found->second->slots[slot].carrier;
    if (!carrier.initialized() || carrier.closing() || !carrier.retired()) return std::nullopt;
    return slot;
  }

  // Reserve typed metadata before accepting native work. The caller can retain
  // the same original views for a successor; no candidate head is exported and
  // the ordinary API remains on its existing direct Completion path.
  std::shared_ptr<StepSources> prepare_submission_sources(StepState &step) {
    const bool prefill = step.phase && packs_prefill(step.phase->family->packing) &&
        step.phase->family->role == GraphRole::kText && !cache_plan.draft_cache &&
        step.phase->family->completion == TextCompletion::kStateOnly;
    const bool mtp = step.phase && drafts_token(step.phase->family->role) && step.members.size() == 1 &&
        step.phase->successor_token_port && step.phase->successor_catchup_port;
    if (!step.prepared || step.submitted || step.cancelled || step.execution.sources || !step.phase ||
        (!mtp && (step.phase->family->role != GraphRole::kText || cache_plan.draft_cache)) ||
        (!prefill && step.phase->family->completion != TextCompletion::kEmitNext) || scheduler)
      throw std::runtime_error("submission views require one complete supported prepared episode");
    auto &phase = *step.phase;
    const auto &storage = execution.storage_owner();
    if (!storage || phase.slots[step.state_slot].carrier.queues().size() != 1 ||
        (!prefill && !take(phase.slots[step.state_slot].io.readback(phase.slots[step.state_slot].data.next_id))->pool().valid()))
      throw inference::EvaluationFailure(hyle::Status::Unimplemented(
          "submission views require the original owned shared Queue and full token writer"));
    const bool device_choice = device_mtp_successor_ && mtp && bool(phase.successor_prefix_index_port);
    auto sources = std::make_shared<StepSources>();
    sources->mtp_full_candidate = mtp;
    sources->mtp_device_choice = device_choice;
    sources->prefill = prefill;
    sources->boundaries.reserve(step.members.size());
    sources->published_states.resize(phase.state_sources.size() * step.members.size() + (mtp ? phase.prefix_rows-1 : 0u));
    for (const auto &member : step.members) {
      if (member.base_epoch == UINT64_MAX) throw std::overflow_error("source candidate epoch exhausted");
      sources->boundaries.push_back({member.session, member.next.get(), member.base_epoch + 1,
          member.next->length, member.next->draft ? member.next->draft->frontier : 0});
    }
    const auto stores = phase.state_sources | std::views::transform([&storage](const auto &source) {
      return inference::SubmissionViews::StoreCapture{source.store, storage};
    });
    const auto appends = std::span<const g::NodeId>(step.execution.nodes.data(), draft_bank + (mtp ? 2u : 0u)) |
        std::views::transform([&storage](g::NodeId append) {
          return inference::SubmissionViews::PageCapture{append, storage};
        });
    check(sources->prepare(*step.execution.run, stores, appends, execution_budget, device_choice ? 2u : mtp ? 1u : 0u));
    if (mtp) {
      sources->successor_states.reserve(geometry.recurrent_layers() * 2);
      for (std::size_t bank = 0; bank < geometry.recurrent_layers() * 2; ++bank)
        sources->successor_states.push_back(take(g::StoreSource::prepare(*step.execution.run,
            phase.state_sources[bank].store, storage,
            device_choice ? g::StoreDestinationRange{0, phase.prefix_rows} : g::StoreDestinationRange{phase.prefix_rows-1, 1},
            {.resource_budget = execution_budget})));
      sources->draft_stage_sources.reserve(2*phase.draft_chain_steps);
      for(std::size_t stage=0;stage<phase.draft_chain_steps;++stage)for(std::size_t bank=0;bank<2;++bank)
        sources->draft_stage_sources.push_back(take(g::batch::PagedAppendSource::prepare(*step.execution.run,
            phase.append_nodes[draft_bank+bank],storage,{.resource_budget=execution_budget})));
    }
    step.execution.sources = sources;
    return sources;
  }

  std::shared_ptr<StepSources> prepare_submission_sources() {
    if (!active_step) throw std::runtime_error("submission views require a prepared episode");
    return prepare_submission_sources(*active_step);
  }

  void submit_evaluations(StepState &step, bool optional_sources = false) {
    if ((active_step.get() != &step && successor_step.get() != &step) || !step.prepared ||
        step.submitted || step.cancelled || !step.execution.run)
      throw std::runtime_error("batch is not prepared or was already submitted/cancelled");
    for (const auto &member : step.members)
      if (member.session->stopping || member.session->failed)
        throw std::runtime_error("batch participant stopped before submission");
    // Validate counter additions before accepting native work. Afterwards only
    // scalar assignment remains, even if Source capture or observation fails.
    const auto draft_passes = step.phase->draft_chain_steps ? 2u : 1u;
    const auto model_rows_after = cache::add(submitted_draft_rows, cache::multiply(draft_passes, step.active_draft_rows));
    std::size_t chain_attention_after = draft_chain_attention_rows;
    if(!step.draft_rows_deferred)for (const auto &stage : step.phase->draft_chain)
      chain_attention_after = cache::add(chain_attention_after, stage.rows);
    for (const auto &member : step.members)
      (void)cache::add(member.session->submitted_draft_rows, cache::multiply(draft_passes, member.draft_input_rows));
    step.start = Clock::now(); step.submitted = true;
    check(step.execution.submit(resources_).WithContext("submitting original evaluation"));
    submitted_draft_rows = model_rows_after;
    draft_chain_attention_rows = chain_attention_after;
    if (step.phase->draft_chain_steps) ++draft_chain_graph_calls;
    for (const auto &member : step.members)
      member.session->submitted_draft_rows += draft_passes * member.draft_input_rows;
    if (step.execution.sources) {
      auto &phase = *step.phase; auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
      auto &sources = *step.execution.sources;
      bool captured = false;
      if (sources.mtp_device_choice) {
        const std::array<inference::SubmissionViews::OutputCapture, 3> outputs{{
            {phase.successor_pending_port, execution.storage_owner(), take(program_slot.io.readback(io.successor_pending))->pool()},
            {phase.successor_prefix_index_port, execution.storage_owner(), take(program_slot.io.readback(io.successor_prefix_index))->pool()},
            {phase.successor_selected_catchup_port, execution.storage_owner(), take(program_slot.io.readback(io.successor_selected_catchup))->pool()}}};
        captured = take(sources.capture_outputs(*step.execution.run, step.execution.completion, outputs,
            execution.storage_owner(), optional_sources));
        if (captured) for (auto &state : sources.successor_states) check(state.capture(sources.submission));
        if (captured) for (auto &source : sources.draft_stage_sources) check(source.capture(sources.submission));
      } else if (sources.mtp_full_candidate) {
        const std::array<inference::SubmissionViews::OutputCapture, 2> outputs{{
            {phase.successor_token_port, execution.storage_owner(), take(program_slot.io.readback(io.successor_token))->pool()},
            {phase.successor_catchup_port, execution.storage_owner(), take(program_slot.io.readback(io.successor_catchup))->pool()}}};
        captured = take(sources.capture_outputs(*step.execution.run, step.execution.completion, outputs,
            execution.storage_owner(), optional_sources));
        if (captured) for (auto &state : sources.successor_states) check(state.capture(sources.submission));
        if (captured) for (auto &source : sources.draft_stage_sources) check(source.capture(sources.submission));
      } else if (sources.prefill) captured = take(sources.capture_outputs(*step.execution.run,
          step.execution.completion, {}, execution.storage_owner(), false,
          step.replay_route != g::ReplayRoute::kCaptured));
      else captured = take(sources.capture(*step.execution.run, step.execution.completion,
          phase.id_port, execution.storage_owner(), take(program_slot.io.readback(io.next_id))->pool(), optional_sources));
      if (!captured) step.execution.sources.reset();
    }
    step.submit_seconds = std::chrono::duration<double>(Clock::now() - step.start).count();
  }

  void submit_evaluations() {
    if (!active_step) throw std::runtime_error("submission requires a prepared episode");
    submit_evaluations(*active_step);
  }

  void submit_evaluation(Session &session) {
    if (session.model_ != this || !active_step || session.pending != active_step.get() || active_step->members.size() != 1)
      throw std::runtime_error("single submission requires its original one-request episode");
    submit_evaluations();
  }

  // Private validators can outlive the target's failure. Read only their
  // original Invocation/retention facts; this failure-only check neither
  // starts another recovery nor treats a cleanup hint as proof of retirement.
  bool private_publications_retired() const noexcept {
    for (const auto &[key, program] : draft_programs) {
      (void)key;
      if (program->guard.execution.run || program->guard.execution.cleanup_pending ||
          !program->carrier.retired()) return false;
    }
    for (const auto &[key, program] : prefix_sampling_programs) {
      (void)key;
      if (!program->carrier.retired()) return false;
    }
    for (const auto &entry : resources_->sampling_configurations)
      if (entry->frontier && !entry->frontier->carrier.retired()) return false;
    return true;
  }

  inference::RetirementObservation retire_failed_publication(bool private_pending = false) noexcept {
    auto retired = abort_episode();
    retired.retired = retired.retired && !private_pending && private_publications_retired();
    return retired;
  }

  template<class Select, class Complete, class Observe, class Publication = NoEvaluationPublication>
    requires std::is_invocable_v<Complete &, StepState &>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Complete complete, Observe observe,
                                  Publication &&publication = {}) {
    if (!active_step || outputs.size() != active_step->members.size() || next_ids.size() != outputs.size() ||
        active_step->cancelled || !active_step->submitted || !active_step->execution.has_completion())
      throw std::runtime_error("finish requires its original complete noncancelled batch");
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      if (outputs[j].session != active_step->members[j].session || outputs[j].session->failed ||
          (outputs[j].session->stopping && !active_step->discard_stopped_members))
        throw std::runtime_error("finish participant does not match the original active batch");
      if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kNotPublished;
    }
    auto &step = *active_step; auto &phase = *step.phase; auto &run = *step.execution.run;
    auto &program_slot = phase.slots[step.state_slot]; auto &io = program_slot.data;
    const auto step_start = step.step_start, start = step.start;
    struct FinishedMember final {
      std::size_t input_rows;
      inference::PreparedRecordCommit *publication = nullptr;
    };
    hyle::container::SmallVector<FinishedMember, 2> finished_members;
    const bool state_only = phase.family->completion == TextCompletion::kStateOnly;
    const auto submit_seconds = step.submit_seconds, cold_batch_seconds = step.cold_batch_seconds;
    const auto shape_admission_seconds = step.shape_admission_seconds;
    const auto phase_prepare_seconds = step.phase_prepare_seconds, slot_initialize_seconds = step.slot_initialize_seconds;
    const auto phase_creations = step.phase_creations, slot_initializations = step.slot_initializations;
    const auto replay_route = step.replay_route;
    const auto capture_recorded = step.capture_recorded, capture_reused = step.capture_reused;
    std::array<c::ConstTensorView, 9> probe_views;
    std::array<c::ReadLease, 9> probe_owners;
    c::ReadLease generated_logits, generated_hidden;
    // An empty local mask means all members publish. Allocate only for an
    // actual withdrawal, before any publication; ordinary R>2 adds no buffer.
    // It stays owned here through detach and all output observers.
    hyle::container::SmallVector<std::uint8_t, 2> published_members;
    struct ReturnReadback final {
      std::size_t *admission = nullptr;
      ~ReturnReadback() { if (admission) { HYLE_FATAL_ASSERT(*admission); --*admission; } }
    } return_readback;
    try {
      if (!finished_members.try_reserve(outputs.size())) throw std::bad_alloc{};
      for (const auto &member : step.members) finished_members.push_back({member.input_tokens, nullptr});
      if (step.discard_stopped_members &&
          !published_members.try_resize(outputs.size(), std::uint8_t{1})) throw std::bad_alloc{};
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        const auto &member = step.members[j];
        const auto member_role = (packs_draft(phase.family->packing) || step.device_input_choice)
            ? (member.base->draft->lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : phase.family->role;
        finished_members[j].publication = publication.prepare(j, EvaluationAttemptFacts{
            member_role, phase.family->completion, generation_boundary(*member.base),
            inference::request_stop_policy(member.stop_policy), member.input_tokens, member.first_input,
            cache_plan.draft_cache == (phase.family->role != GraphRole::kText) &&
                member.base->draft.has_value() == cache_plan.draft_cache,
            step.withdrawn(member)});
      }
      const auto append_banks = draft_bank + (has_mtp(phase.family->role) || phase.recursive_draft_repair ? 2u : 0u) + 2 * phase.draft_chain_steps;
      auto &transactions = phase.slots[step.state_slot].transactions;
      check(step.execution.finish_success(0, append_banks, phase.state_resource_ports,
          []() noexcept {},
          [&]() -> hyle::Status {
            // Trace storage must be pinned before take_append/unbind revoke Graph
            // output observation. Generated arrays keep their actual native owner.
            if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_tensors; })) {
              if (!state_only && !io.logits.caller_owned()) generated_logits = take(inference::retain_output(run, *phase.logits_facts, allocation_budget));
              if (!io.final_hidden.caller_owned()) generated_hidden = take(inference::retain_output(run, *phase.hidden_facts, allocation_budget));
            }
            if (drafts_token(phase.family->role)) {
              check(program_slot.io.read(run, io.draft_id, std::as_writable_bytes(std::span(io.draft_ids))));
              if (phase.draft_chain_steps) {
                auto &member = step.members.front();
                member.verification_inputs[0] = member.first_input;
                std::copy(io.draft_ids.begin(), io.draft_ids.end(), member.verification_inputs.begin() + 1);
                member.draft_token = io.draft_ids.back();
              } else for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].draft_token = io.draft_ids[j];
            }
            const bool batched_small_results = io.ordinary_io.batched_results();
            if (batched_small_results) {
              const bool float_predicates = io.ordinary_io.results == OrdinaryResultMode::kBatchFloat;
              const std::array destinations{
                  float_predicates ? std::as_writable_bytes(std::span(io.sampling_scratch))
                      : std::as_writable_bytes(std::span(io.sampling_valid_values)),
                  std::as_writable_bytes(next_ids)};
              check(io.small_results.read(run, destinations, resources_));
              // Preserve the original predicate's exact Bool/F32 validation;
              // no cast or arithmetic repacks invalid values into success.
              for (std::size_t j = 0; j < io.sampling_valid_values.size(); ++j) {
                if (float_predicates) {
                  const auto value = io.sampling_scratch[j];
                  if (value != 0 && value != 1)
                    throw inference::EvaluationFailure(hyle::Status::InvalidArgument("host predicate is not exact F32 zero or one"));
                  io.sampling_valid_values[j] = static_cast<std::uint8_t>(value == 1);
                } else if (io.sampling_valid_values[j] > 1)
                  throw inference::EvaluationFailure(hyle::Status::InvalidArgument("host predicate is not exact Bool zero or one"));
              }
            }
            if (phase.sampling_valid_port && !state_only && phase.family->role != GraphRole::kVerify) {
              if (!batched_small_results)
                check(take(program_slot.io.readback(io.sampling_valid))->predicates(run, io.sampling_valid_values, io.sampling_scratch));
              auto bad = std::ranges::find_if(io.sampling_valid_values, [](auto value) { return value != 1; });
              if (step.discard_stopped_members)
                while (bad != io.sampling_valid_values.end() &&
                    step.withdrawn(step.members[static_cast<std::size_t>(bad - io.sampling_valid_values.begin())]))
                  bad = std::find_if(bad + 1, io.sampling_valid_values.end(), [](auto value) { return value != 1; });
              if (bad != io.sampling_valid_values.end()) {
                std::optional<std::size_t> participant;
                // Everything before the first bad value is already proven 1.
                // Only the failure path inspects the remainder for a unique 0.
                bool unique = true;
                for (auto it = bad + 1; it != io.sampling_valid_values.end(); ++it)
                  if (*it != 1 && !step.withdrawn(
                      step.members[static_cast<std::size_t>(it - io.sampling_valid_values.begin())])) unique = false;
                if (*bad == 0 && unique)
                  participant = static_cast<std::size_t>(bad - io.sampling_valid_values.begin());
                throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
                    "sampling distribution or proposal is invalid; no session or RNG cursor was published"), participant);
              }
              if (phase.sampling_accept_port) {
                check(take(program_slot.io.readback(io.sampling_accept))->predicates(run, io.sampling_accept_values, io.sampling_scratch));
                for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].sampled_accept = io.sampling_accept_values[j] != 0;
              }
            }
            // Selection observes only retired original Graph results. No head has
            // been published and all state/append guards still belong to this episode.
            if (verifies_prefix(phase.family->role)) {
              check(program_slot.io.read(run, io.next_id, std::as_writable_bytes(std::span(io.prefix_candidates))));
              select(std::span<const std::int32_t>(io.prefix_candidates), step, next_ids);
            } else if (state_only) {
              std::ranges::fill(next_ids, -1);
              select(std::span<const std::int32_t>(next_ids), step, next_ids);
            } else {
              if (!batched_small_results)
                check(program_slot.io.read(run, io.next_id, std::as_writable_bytes(next_ids)));
              select(std::span<const std::int32_t>(next_ids), step, next_ids);
            }
            // Account the actual submitted work even when EOS or cancellation
            // makes B unpublishable. These already retired A results are the
            // exact same greedy predicate that its device selector consumed.
            if (successor_step && successor_step->device_input_choice && successor_step->submitted &&
                successor_step->draft_rows_deferred && step.members.size() == 1 &&
                successor_step->members.size() == 1 && io.prefix_candidates.size() == phase.prefix_rows) {
              std::size_t index=0;
              for(;index+1<phase.prefix_rows;++index){
                const auto draft=phase.draft_chain_steps?step.members[0].verification_inputs[index+1]:step.members[0].draft_token;
                if(io.prefix_candidates[index]!=draft)break;
              }
              const auto lag=index?1u:0u;
              const auto rows=successor_step->phase->prefix_rows-1+lag;
              const auto work=rows*(successor_step->phase->draft_chain_steps?2u:1u);
              auto &next = successor_step->members[0];
              submitted_draft_rows = cache::add(submitted_draft_rows, work);
              next.session->submitted_draft_rows = cache::add(next.session->submitted_draft_rows, work);
              draft_chain_attention_rows=cache::add(draft_chain_attention_rows,successor_step->phase->draft_chain_steps);
              next.draft_input_rows = successor_step->active_draft_rows = rows;
              successor_step->draft_rows_deferred = false;
            }
            for (std::size_t j = 0; j < outputs.size(); ++j) {
              auto &member = step.members[j];
              // Retain every original physical owner for whole-cohort tail return,
              // even when no accepted prefix will be published for this member.
              member.verified_last_page = member.next->paged.pages.back().page();
              if (step.withdrawn(member)) { next_ids[j] = -1; continue; }
              if (!member.accepted_tokens || member.accepted_tokens > member.input_tokens ||
                  (member.accepted_tokens != member.input_tokens && !verifies_prefix(phase.family->role)) ||
                  (!state_only && (next_ids[j] < 0 || next_ids[j] >= vocabulary)))
                throw std::runtime_error("invalid accepted target boundary");
              if (member.accepted_tokens != member.input_tokens) {
                member.next->length = member.base->length + member.accepted_tokens;
                const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
                member.next->paged.pages.resize(page_count);
                auto &layout_rows = input_scratch.paged_appends.next_rows;
                layout_rows.reset();
                check(layout_rows.append(member.accepted_tokens, member.next->length, cache::Plan::page_rows,
                    std::span<const hyle::inference::PagedPageRef>(member.next->paged.pages)));
                member.layout = take(layout_rows.member_version(attention_schema, 0));
              }
            }
            if (has_mtp(phase.family->role) && !phase.draft_chain_steps) for (auto &member : step.members) {
              if (step.withdrawn(member)) continue;
              auto &draft = *member.next->draft;
              draft.lag = static_cast<std::uint32_t>(member.next->length - draft.frontier);
              draft.catchup_token = draft.lag ? member.draft_token : -1;
            }
            if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_probes; }))
              for (std::size_t i = 0; i < probe_views.size(); ++i) {
                const auto &output = *phase.probe_facts[i];
                if (output.caller_owned()) probe_views[i] = take(run.output(output.value));
                else {
                  probe_owners[i] = take(inference::retain_output(run, output, allocation_budget));
                  probe_views[i] = take(probe_owners[i].tensor(output.description));
                }
              }
            return hyle::Status::Ok();
          },
          [&]() -> hyle::Status {
            if (step.device_input_choice) {
              if (step.members.size() != 1 || !step.members[0].selected_input_choice)
                return hyle::Status::InvalidArgument("retired device successor has no resolved producer choice");
              for(std::size_t bank=0;bank<step.execution.appends.size();++bank){
                const auto&member=step.members[0];const auto selected=*member.selected_input_choice;
                const auto&layout=bank<draft_bank?member.target_input_choices[selected]:member.draft_input_choices[selected];
                auto rows=layout.facts().query_rows,after=layout.facts().kv_rows;
                if(bank>=draft_bank+2){const auto stage=(bank-draft_bank-2)/2,steps=phase.draft_chain_steps;
                  rows=static_cast<std::uint32_t>(stage+1+(layout.facts().query_rows-steps));
                  after=static_cast<std::uint32_t>(layout.facts().kv_rows-steps+stage+1);}
                const std::array<std::int32_t,1>q{static_cast<std::int32_t>(rows)},kv{static_cast<std::int32_t>(after)};
                HYLE_RETURN_IF_ERROR(step.execution.appends[bank].select_lengths(q,kv));
              }
            }
            for (std::size_t bank_index = 0; bank_index < phase.state_banks.size(); ++bank_index) {
              const auto &bank = phase.state_banks[bank_index];
              auto writers = take(transactions.returned(bank_index));
              for (std::size_t j = 0; j < outputs.size(); ++j) {
                auto &member = step.members[j];
                if (step.withdrawn(member)) continue;
                if (bank.kind == StateBankKind::kHidden) {
                  const auto count = verifies_prefix(phase.family->role) ? member.accepted_tokens : 1;
                  const bool repair = phase.family->role == GraphRole::kVerify && cache_plan.draft_cache;
                  if (repair) member.carry_prefix.reserve(count);
                  for (std::size_t row = 0; row < count; ++row) {
                    // Exactly one publication of each accepted canonical row. The
                    // final head and draft repair share aliases of that same lease.
                    const auto destination = j * phase.prefix_rows + row;
                    auto ready = step.execution.sources && bank.source_index != SIZE_MAX
                        ? take(step.execution.sources->states[bank.source_index].publish(destination, writers[destination]))
                        : take(writers[destination].publish());
                    if (step.execution.sources && step.execution.sources->mtp_full_candidate)
                      step.execution.sources->published_states[bank.source_index + row] = ready;
                    if (row + 1 == count) member.next->carry_hidden = ready;
                    if (row + 2 == count) member.next->prior_hidden = ready;
                    if (repair) member.carry_prefix.push_back(std::move(ready));
                  }
                  continue;
                }
                const auto prefix = verifies_prefix(phase.family->role) ? step.members[j].accepted_tokens - 1 : 0;
                const auto selected = j * phase.prefix_rows + prefix;
                auto ready = step.execution.sources ? take(step.execution.sources->states[bank.source_index].publish(
                    selected, writers[selected]))
                    : take(writers[selected].publish());
                if (step.execution.sources) step.execution.sources->published_states[bank.source_index * outputs.size() + j] = ready;
                auto &state = step.members[j].next->state[bank.layer];
                (bank.kind == StateBankKind::kRecurrent ? state.recurrent : state.history) = std::move(ready);
              }
              // Release old source guards before the joint head-commit recycle hook.
              // Unaccepted MTP candidates are discarded, never made visible.
              check(transactions.clear(bank_index));
            }
            for (auto &member : step.members)
              if (!step.withdrawn(member) && !member.target_updates.try_resize(page_inventory.banks.size()))
                throw std::bad_alloc{};
            const auto target_finish_at = [&](std::size_t j) noexcept -> inference::PagedAppendFinishMember {
              auto &member = step.members[j];
              return {member.base->paged, member.next->paged,
                  {member.target_updates.data(), member.target_updates.size()}, member.layout,
                  member.verified_last_page, static_cast<std::uint32_t>(member.next->length % cache::Plan::page_rows),
                  {member.next_tails.data(), member.next_tails.size()}, !step.withdrawn(member),
                  member.accepted_tokens == member.input_tokens};
            };
            check(inference::prepare_paged_append_commits(
                std::span(step.execution.appends.data(), draft_bank), outputs.size(), target_finish_at,
                step.execution.sources ? std::span<const g::batch::PagedAppendSource>(step.execution.sources->pages).first(draft_bank)
                    : std::span<const g::batch::PagedAppendSource>{}));
            if (has_mtp(phase.family->role) && !phase.draft_chain_steps) {
              const auto draft_finish_at = [&](std::size_t j) noexcept -> inference::PagedAppendFinishMember {
                auto &member = step.members[j]; auto &draft = *member.next->draft;
                return {member.base->draft->paged, draft.paged, member.draft_updates, member.draft_layout,
                    draft.paged.pages.back().page(), static_cast<std::uint32_t>(draft.frontier % cache::Plan::page_rows),
                    member.next_draft_tails, !step.withdrawn(member)};
              };
              check(inference::prepare_paged_append_commits(
                  std::span(step.execution.appends.data() + draft_bank, 2), outputs.size(), draft_finish_at,
                  step.execution.sources ? std::span<const g::batch::PagedAppendSource>(step.execution.sources->pages).subspan(draft_bank, 2)
                      : std::span<const g::batch::PagedAppendSource>{}));
              check(inference::return_paged_append_tails(
                  step.execution.append_transactions().subspan(draft_bank, 2), 2, outputs.size(),
                  input_scratch.paged_appends, draft_finish_at));
            }
            // Caller-supplied completion is confined to this unpublished episode.
            // In particular, recursive proposals must repair their retained KV from
            // actual target hidden rows here, before ledger checks or any head swap.
            complete(step);
            if (phase.draft_chain_steps) {
              const auto private_appends = step.execution.append_transactions().subspan(draft_bank + 2, 2 * phase.draft_chain_steps);
              if (!take(private_appends.discard())) throw std::logic_error("retired compound draft appends remain borrowed");
              step.members.front().draft_chain_pages.clear();
            }
            if (cache_plan.draft_cache)
              for (const auto &member : step.members)
                if (!step.withdrawn(member)) validate_mtp_snapshot(*member.next, &member);
            for (std::size_t j = 0; j < outputs.size(); ++j) if (finished_members[j].publication) {
              const auto &member = step.members[j];
              auto draft_rows = member.next->draft
                  ? member.next->draft->frontier - member.base->draft->frontier : 0;
              const bool recursive = phase.family->role == GraphRole::kVerify || phase.draft_chain_steps;
              if (recursive) draft_rows = cache::add(draft_rows,
                  member.input_tokens - 1 + (member.input_tokens > 1 ? member.base->draft->lag : 0) +
                  member.discarded_draft_rows);
              auto after = generation_boundary(*member.next); after.next_id = next_ids[j];
              const std::array accepted{member.first_input, member.accepted_tokens == 2 ? member.draft_token : -1};
              const std::array verified{member.first_input, member.draft_token};
              const auto result_role = step.device_input_choice
                  ? (member.base->draft->lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : phase.family->role;
              publication.complete(j, EvaluationResultFacts{
                  result_role, after, outputs.size(), j, member.accepted_tokens, draft_rows, member.input_tokens,
                  recursive ? std::span<const std::int32_t>(member.verification_inputs).first(member.accepted_tokens)
                      : std::span<const std::int32_t>(accepted),
                  recursive ? std::span<const std::int32_t>(member.verification_inputs)
                      : std::span<const std::int32_t>(verified).first(drafts_token(phase.family->role) ? 2u : 1u),
                  next_ids[j], state_only,
                  !state_only && inference::request_token_stopped(next_ids[j], model_eos_tokens, member.stop_policy)});
            }
            // Partial acceptance narrows Frame authority without rewinding the
            // verified physical tail; the next append retains its original COW path.
            check(inference::return_paged_append_tails(
                step.execution.append_transactions().subspan(0, draft_bank), draft_bank, step.members.size(),
                input_scratch.paged_appends, target_finish_at));
            return hyle::Status::Ok();
          }));
      for (const auto &member : step.members)
        if (member.session->head.get() != member.base || member.session->head.epoch() != member.base_epoch ||
            member.session->pending != &step ||
            (member.session->stopping && !step.discard_stopped_members))
          throw std::runtime_error("batch snapshot expected-base conflict");
      for (auto &member : step.members)
        if (!step.withdrawn(member)) member.preflight_frames();
      // Every delta owns its changed guards and every tail has transferred.
      // A successor still retains this original Step, but must not thereby
      // pin replaced input tails from its consumed returned append wrappers.
      for (auto &append : step.execution.appends) append = {};
      // All fallible native/export/metadata work is complete. The head swaps
      // below cannot throw. Stopped members discard their unpublished
      // candidate; every surviving member keeps this original successful work.
      for (std::size_t j = 0; j < step.members.size(); ++j) {
        auto &member = step.members[j];
        if (step.withdrawn(member)) {
          published_members[j] = 0;
          if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kCancelled;
          continue;
        }
        if (outputs[j].disposition) *outputs[j].disposition = EvaluationDisposition::kPublished;
        member.next->next_id = next_ids[j];
        member.session->tails = std::move(member.next_tails);
        member.session->draft_tails = std::move(member.next_draft_tails);
        const auto publish = [&]() noexcept -> hyle::Status {
          member.session->head.commit(std::move(member.next),
            [&](cache::State *previous, cache::State &candidate) noexcept { member.publish_frames(previous, candidate); },
            [&](cache::State &previous) noexcept {
          // Only an exclusively owned successful old working head reaches
          // here, after Graph withdrawal. Frozen checkpoints stay untouched;
          // try_write also rejects escaped readers and original native uses.
          const auto count = (std::min)(previous.state.size(), state_pools.size());
          for (std::size_t i = 0; i < count; ++i) {
            state_pools[i].history_reuse.recycle(previous.state[i].history);
            state_pools[i].recurrent_reuse.recycle(previous.state[i].recurrent);
          }
          });
          return {};
        };
        // Mark each original record immediately after its own no-fail head
        // swap, before any subsequent reporting or observation can fail.
        if (auto *ticket = finished_members[j].publication) {
          const auto status = ticket->publish(publish);
          HYLE_FATAL_ASSERT(status.ok());
        } else {
          (void)publish();
        }
        member.session->tail_epoch = member.session->head.epoch();
        member.session->draft_tail_epoch = member.session->head.epoch();
      }
      if (step.execution.sources) step.execution.sources->release_publication_views();
      if (packs_prefill(phase.family->packing)) {
        ++packed_prefill_calls;
        packed_prefill_active_rows = cache::add(packed_prefill_active_rows, step.active_input_rows);
        packed_prefill_physical_rows = cache::add(packed_prefill_physical_rows,
            cache::multiply(phase.query_rows, phase.family->requests));
      }
      return_readback.admission = std::exchange(step.execution.candidate_admission, nullptr);
      detach_episode(); // Old states/transactions retire inside the host endpoint.
    } catch (inference::EvaluationFailure &failure) {
      // A nested private validator can still own native work independently of
      // this target episode. Its negative observation cannot be overwritten
      // by retiring only the target; retain it until its original owner drains.
      const bool private_pending = failure.retirement_observed() && !failure.retirement().retired;
      const auto retired = retire_failed_publication(private_pending);
      publication.retired(retired); failure.observe_retirement(retired);
      throw;
    } catch (...) {
      publication.retired(retire_failed_publication()); throw;
    }
    const auto finished = Clock::now();
    publication.finished(finished, std::chrono::duration<double>(finished - step_start).count());
    std::size_t compact_input_begin = 0;
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      const auto member_input_begin = compact_input_begin;
      compact_input_begin = cache::add(compact_input_begin, finished_members[j].input_rows);
      if (!published_members.empty() && !published_members[j]) continue;
      const auto &output = outputs[j]; const auto &session = *output.session; const auto next = next_ids[j];
      if (output.complete_seconds) *output.complete_seconds = std::chrono::duration<double>(finished - step_start).count();
      if (output.host_endpoint) *output.host_endpoint = finished;
      if (output.preparation) {
        const auto report_start = Clock::now();
        output.preparation->nodes.resize(phase.attention_nodes.size());
        output.preparation->reports.resize(phase.attention_nodes.size());
        for (std::size_t i = 0; i < output.preparation->reports.size(); ++i) {
          output.preparation->nodes[i] = phase.attention_nodes[i];
          output.preparation->reports[i] = take(g::batch::report(run, phase.attention_nodes[i]));
        }
        output.preparation->report_snapshot_seconds = std::chrono::duration<double>(Clock::now() - report_start).count();
      }
      inference::StepObservation observed;
      auto &observation = output.observation ? *output.observation : observed;
      observation.input_tokens = finished_members[j].input_rows; observation.visible = session.length(); observation.next = next;
      observation.graph_executed = true; observation.graph_input_tokens = finished_members[j].input_rows;
      observation.seconds = std::chrono::duration<double>(finished - start).count();
      observation.submit_seconds = submit_seconds; observation.step_seconds = std::chrono::duration<double>(finished - step_start).count();
      observation.cold_batch_seconds = cold_batch_seconds; observation.replay_route = replay_route;
      observation.shape_admission_seconds = shape_admission_seconds;
      observation.phase_prepare_seconds = phase_prepare_seconds; observation.slot_initialize_seconds = slot_initialize_seconds;
      observation.phase_creations = phase_creations; observation.slot_initializations = slot_initializations;
      observation.capture_recorded = capture_recorded; observation.capture_reused = capture_reused;
      observation.scheduled_progress = static_cast<bool>(scheduler);
      CompletedEvaluationView completed{.observation = observation, .head = *session.head,
          .member_index = j, .requests = outputs.size(),
          .packed_input_begin = packs_verify(phase.family->packing) ? member_input_begin : 0,
          .packed_input_rows = packs_verify(phase.family->packing) ? finished_members[j].input_rows : 0};
      if (output.capture_tensors) {
        if (!state_only) completed.logits = io.logits.caller_owned() ? io.logits.view().as_const()
            : take(generated_logits.tensor(io.logits.description));
        completed.hidden = io.final_hidden.caller_owned() ? io.final_hidden.view().as_const()
            : take(generated_hidden.tensor(io.final_hidden.description));
      }
      if (output.capture_probes) completed.probes = probe_views;
      observe(j, completed);
    }
  }

  template<class Select, class Observe, class Publication = NoEvaluationPublication>
    requires std::is_invocable_v<Observe &, std::size_t, const CompletedEvaluationView &>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Observe observe, Publication &&publication = {}) {
    finish_selected_evaluations(outputs, next_ids, std::move(select), [](StepState &) {}, std::move(observe), publication);
  }

  // Returns the actual verifier input [pending, proposal0, ...]. Approximate
  // hidden is local to this recursion and is deliberately discarded with its
  // private pages; it is never installed in a committed draft cache.
  template<class PrepareVerifier = std::nullptr_t>
  std::vector<std::int32_t> propose_recursive(Session &session, std::size_t maximum,
      std::span<const std::int32_t> eos = {}, std::vector<c::ReadLease> *q = nullptr,
      const inference::RequestStopPolicy *stop_policy = nullptr, const LogitsInputs &logits = {},
      std::span<const LogitsInputs> conditional_inputs = {}, ConditionalLogitsProposal *dynamic = nullptr,
      PrepareVerifier &&prepare_verifier = nullptr) {
    if (session.model_ != this)
      throw std::runtime_error("recursive proposal Session belongs to another model");
    const auto &selected_sampling = session.sampling_configuration();
    const auto &selected_sampler = selected_sampling.configuration;
    validate_prefix_logits_inputs(selected_sampler, logits, conditional_inputs, cache::add(maximum, 1), vocabulary);
    if (dynamic && (!conditional_inputs.empty() || !dynamic->provider || dynamic->rows.size() != 1))
      throw std::runtime_error("dynamic proposal inputs require one previously resolved first row");
    if (dynamic) {
      auto row_policy = selected_sampler; row_policy.modifiers.history = {};
      validate_logits_inputs(row_policy, dynamic->rows.front(), vocabulary);
    }
    if (active_step || carrier_borrowed || session.pending || session.stopping || session.failed ||
        !session.head || !session.head->draft || (selected_sampler.enabled && !q) ||
        maximum >= capacity - session.length() || maximum >= cache_plan.candidate_boundaries)
      throw std::runtime_error("recursive proposals require an idle prepared head, bounded width and every stochastic q row");
    validate_mtp_snapshot(*session.head);
    inference::request_stop_policy(stop_policy).validate(vocabulary);
    const auto ended = [&](auto token) { return inference::request_token_stopped(token, eos, stop_policy); };
    std::vector<std::int32_t> inputs; inputs.reserve(maximum + 1); inputs.push_back(session.head->next_id);
    if (q) { q->clear(); q->reserve(maximum); }
    if (!maximum) { session.proposal_graph_calls = 0; return inputs; }
    if (inputs.front() < 0 || inputs.front() >= vocabulary || ended(inputs.front()))
      throw std::runtime_error("recursive proposals require a valid nonterminal emitted pending token");
    const auto graph_begin = draft_program_graph_calls;
    hyle::util::ScopeGuard proposal_calls{[&]() noexcept {
      session.proposal_graph_calls = draft_program_graph_calls - graph_begin;
    }};
    if (maximum >= 2 && greedy_draft_chain_enabled(selected_sampler) && !dynamic && conditional_inputs.empty()) {
      const auto before = session.head->draft->frontier;
      std::size_t dirty_pages = 0;
      for (std::size_t i = 0; i < maximum; ++i)
        dirty_pages += (before + session.head->draft->lag + 1 + i + cache::Plan::page_rows - 1) /
            cache::Plan::page_rows - before / cache::Plan::page_rows;
      if (cache_plan.in_flight_batches >= 3)
        dirty_pages += (before + maximum + 1 + cache::Plan::page_rows - 1) /
            cache::Plan::page_rows - before / cache::Plan::page_rows;
      // The fixed inventory can refuse extra independent COW tails near the
      // capacity boundary. Preserve its existing bounded sequential route.
      if (dirty_pages <= draft_inventory.free_ids.size()) {
        if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<PrepareVerifier>>) {
          if (maximum == max_drafts())
            return execute_draft_chain(session, maximum, std::forward<PrepareVerifier>(prepare_verifier));
        }
        return execute_draft_chain(session, maximum);
      }
    }
    DraftWork work; work.state = fork_draft(*session.head->draft); work.session = &session;
    c::ReadLease approximate;
    auto current_logits = logits;
    const auto resolve = [&]() -> const LogitsInputs & {
      const auto row = inputs.size() - 1;
      if (row < dynamic->rows.size()) return dynamic->rows[row];
      if (row != dynamic->rows.size()) throw std::logic_error("conditional proposal skipped its linear prefix");
      auto supplied = dynamic->provider(dynamic->committed_outputs, std::span<const std::int32_t>(inputs).subspan(1));
      if (!supplied) throw inference::EvaluationFailure(hyle::Status(supplied.error()));
      auto policy = selected_sampler; policy.modifiers.history = {};
      auto status = logits_inputs_metadata_status(policy, *supplied, vocabulary);
      if (!status.ok()) throw inference::EvaluationFailure(std::move(status));
      validate_logits_input_leases(policy, *supplied);
      dynamic->rows.push_back(std::move(*supplied)); return dynamic->rows.back();
    };
    try {
    for (std::size_t i = 0; i < maximum; ++i) {
      if (dynamic) {
        const auto &row = resolve(); current_logits.bias = row.bias; current_logits.allowed = row.allowed;
      } else if (!conditional_inputs.empty()) {
        current_logits.bias = conditional_inputs[i].bias; current_logits.allowed = conditional_inputs[i].allowed;
      }
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (!i && work.state.lag) { ids.push_back(work.state.catchup_token); shifted.push_back(session.head->prior_hidden); }
      ids.push_back(inputs.back()); shifted.push_back(i ? approximate : session.head->carry_hidden);
      DraftProposal proposal;
      const auto identity = selected_sampler.enabled ? session.head->sampling.identity(2 * i) : std::array<std::uint32_t, 4>{};
      if constexpr (!std::is_null_pointer_v<std::remove_cvref_t<PrepareVerifier>>) {
        if (i + 1 == maximum && maximum == max_drafts())
          proposal = execute_draft(work, ids, shifted, true, identity, selected_sampling.index, current_logits,
              inputs, [&](const std::shared_ptr<StepSources> &source, bool ready) { prepare_verifier(inputs, source, ready); });
        else proposal = execute_draft(work, ids, shifted, true, identity, selected_sampling.index, current_logits);
      } else proposal = execute_draft(work, ids, shifted, true, identity, selected_sampling.index, current_logits);
      inputs.push_back(proposal.id); approximate = std::move(proposal.hidden);
      if (proposal.next_counts.valid()) current_logits.counts.lease = std::move(proposal.next_counts);
      if (q && selected_sampler.enabled) q->push_back(std::move(proposal.log_probabilities));
      if (ended(proposal.id)) break;
    }
    if (dynamic) {
      // An accepted stop proposal ends selection before its bonus; rejection
      // also ends at that proposal. Preserve q and extend the last live policy
      // for this unreachable row, without querying a terminated grammar.
      if (ended(inputs.back()) && dynamic->rows.size() < inputs.size())
        dynamic->rows.push_back(dynamic->rows.back());
      else (void)resolve(); // Resolve the final live policy for the bonus row.
    }
    return inputs;
    } catch (inference::EvaluationFailure &failure) {
      if (dynamic && !failure.retirement_observed()) failure.observe_retirement(resources_->retire(false));
      throw;
    } catch (...) {
      if (dynamic) (void)resources_->retire(false);
      throw;
    }
  }

  void repair_recursive_draft(StepState &step) {
    if ((step.phase->family->role != GraphRole::kVerify && !step.phase->draft_chain_steps) || !cache_plan.draft_cache)
      throw std::runtime_error("draft repair requires its unpublished target verifier");
    if (step.phase->recursive_draft_repair || step.phase->draft_chain_steps) {
      for (auto &member : step.members) {
        if (step.withdrawn(member)) continue;
        auto &next = *member.next->draft;
        const auto physical_last = next.paged.pages.back().page();
        const auto retained = member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1;
        const auto rows = retained + member.base->draft->lag;
        next.frontier = member.base->draft->frontier + rows;
        next.lag = member.accepted_tokens == 1 ? 0u : 1u;
        next.catchup_token = next.lag ? member.verification_inputs[member.accepted_tokens - 1] : -1;
        next.paged.pages.resize((next.frontier + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
        auto &layout_rows = input_scratch.paged_appends.next_rows; layout_rows.reset();
        check(layout_rows.append(rows, next.frontier, cache::Plan::page_rows,
            std::span<const inference::PagedPageRef>(next.paged.pages)));
        member.draft_layout = take(layout_rows.member_version(attention_schema, 0));
        const auto finish_at = [&](std::size_t) noexcept -> inference::PagedAppendFinishMember {
          return {member.base->draft->paged, next.paged, member.draft_updates, member.draft_layout,
              physical_last, static_cast<std::uint32_t>(next.frontier % cache::Plan::page_rows),
              member.next_draft_tails, true, rows == member.draft_input_rows};
        };
        check(inference::prepare_paged_append_commits(
            std::span(step.execution.appends.data() + draft_bank, 2), 1, finish_at,
            step.execution.sources ? std::span<const g::batch::PagedAppendSource>(step.execution.sources->pages).subspan(draft_bank, 2)
                : std::span<const g::batch::PagedAppendSource>{}));
        check(inference::return_paged_append_tails(step.execution.append_transactions().subspan(draft_bank, 2),
            2, 1, input_scratch.paged_appends, finish_at));
      }
      return;
    }
    for (auto &member : step.members) {
      if (step.withdrawn(member)) continue;
      const auto n = member.accepted_tokens;
      if (!n || member.verification_inputs.size() < n || member.carry_prefix.size() != n)
        throw std::runtime_error("draft repair requires every actually consumed target prefix row");
      DraftWork repaired; repaired.state = fork_draft(*member.base->draft); repaired.session = member.session.get();
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (repaired.state.lag) { ids.push_back(repaired.state.catchup_token); shifted.push_back(member.base->prior_hidden); }
      // A one-row target ends at lag0. Longer accepted prefixes retain the
      // canonical lag1 frontier, with the final consumed token left to catch up.
      const auto retained = n == 1 ? 1 : n - 1;
      for (std::size_t i = 0; i < retained; ++i) {
        ids.push_back(member.verification_inputs[i]);
        shifted.push_back(i ? member.carry_prefix[i - 1] : member.base->carry_hidden);
      }
      (void)execute_draft(repaired, ids, shifted, false);
      repaired.state.lag = n == 1 ? 0u : 1u;
      repaired.state.catchup_token = n == 1 ? -1 : member.verification_inputs[n - 1];
      member.next->draft = std::move(repaired.state);
      member.next_draft_tails = std::move(repaired.tails);
      if (member.next->draft->frontier + member.next->draft->lag != member.next->length)
        throw std::runtime_error("draft repair did not reach the selected target prefix");
    }
  }

  template<class Observe, class Publication = NoEvaluationPublication>
  void finish_recursive_mtp(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe, Publication &&publication = {}) {
    if (!active_step || selections.empty() || !active_step->phase ||
        (active_step->phase->family->role != GraphRole::kVerify && !active_step->phase->draft_chain_steps) ||
        !cache_plan.draft_cache || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size())
      throw std::runtime_error("recursive finish requires the complete original verification batch");
    const auto &selected_sampling = *active_step->phase->family->sampling;
    const auto &selected_sampler = selected_sampling.configuration;
    hyle::container::SmallVector<std::int32_t, 2> next;
    if (!next.try_resize(selections.size(), -1)) throw std::bad_alloc{};
    for (std::size_t j = 0; j < selections.size(); ++j) {
      const auto &selection = selections[j];
      if (selection.session != active_step->members[j].session || selection.session != outputs[j].session)
        throw std::runtime_error("recursive selection differs from its original batch member");
      results[j].reset();
      if (active_step->withdrawn(active_step->members[j])) continue;
      if (!selection.output_quota || inference::request_token_stopped(
              selection.expected_pending, eos, active_step->members[j].stop_policy))
        throw std::runtime_error("recursive generation requires its nonterminal pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(active_step->input_tokens);
    }
    finish_selected_evaluations(outputs, std::span(next.data(), next.size()),
        [&](std::span<const std::int32_t> candidates, StepState &step, std::span<std::int32_t> selected) {
      c::ReadLease logits_owner;
      mem::ConstBufferView logits_storage;
      if (selected_sampler.enabled || !selected_sampler.modifiers.empty()) {
        auto &logits = step.phase->slots[step.state_slot].data.logits;
        if (logits.caller_owned()) logits_storage = logits.storage();
        else {
          logits_owner = take(inference::retain_output(*step.execution.run, *step.phase->logits_facts, allocation_budget));
          logits_storage = take(logits_owner.view());
        }
      }
      for (std::size_t j = 0; j < selections.size(); ++j) {
        auto &member = step.members[j]; auto &result = results[j]; const auto &selection = selections[j];
        if (step.withdrawn(member)) { selected[j] = -1; continue; }
        if (member.first_input != selection.expected_pending || selection.expected_pending != member.base->next_id)
          throw std::runtime_error("recursive verifier input differs from the emitted pending token");
         const auto width = member.input_tokens;
         const auto rows = candidates.subspan(member.input_row_begin, width);
        result.pending = selection.expected_pending; result.proposal_count = width - 1; result.verified_rows = width;
        result.auxiliary_graph_calls = step.phase->draft_chain_steps ? 0 : member.session->proposal_graph_calls + !step.phase->recursive_draft_repair +
            (selected_sampler.enabled || !selected_sampler.modifiers.empty() ? 1u : 0u);
        result.draft = width > 1 ? member.verification_inputs[1] : -1;
        result.target_first = rows[0]; result.target_second = width > 1 ? rows[1] : -1;
        hyle::inference::mtp::AcceptedPrefix accepted;
        if (selected_sampler.enabled || !selected_sampler.modifiers.empty()) {
          if (selected_sampler.enabled && selection.draft_log_probabilities.size() != width - 1)
            throw std::runtime_error("stochastic verification requires every actual q row");
          const auto bytes = cache::multiply(cache::multiply(width, vocabulary), sizeof(float));
          const auto logits = take(c::ConstTensorView::create(dense(Type::kFloat32,
               {static_cast<std::int64_t>(width), vocabulary}), take(logits_storage.subview(
                   cache::multiply(cache::multiply(member.input_row_begin, vocabulary), sizeof(float)), bytes))));
          PrefixSamplingResult decision;
          try {
            decision = sample_recursive_prefix(logits,
                std::span<const std::int32_t>(member.verification_inputs).subspan(1),
                selection.draft_log_probabilities, member.base->sampling, selection.output_quota,
                eos, member.stop_policy, selected_sampling.index, member.logits, member.conditional_logits.rows());
          } catch (const inference::EvaluationFailure &failure) {
            // Exact private validation of this original submitted member's
            // target rows; neither a request bucket nor a q-pool index.
            if (!failure.status().IsInvalidArgument() || failure.status().has_native_error() || failure.participant() ||
                !failure.retirement_observed() || !failure.retirement().retired || failure.retirement().status.has_native_error()) throw;
            inference::EvaluationFailure mapped(failure.status(), j);
            if (failure.retirement_observed()) mapped.observe_retirement(failure.retirement());
            throw mapped;
          }
          if (selected_sampler.enabled) {
            accepted = decision.prefix;
            result.emitted.assign(decision.final_ids.begin(), decision.final_ids.end());
            member.next->sampling = member.base->sampling.advanced(cache::add(cache::multiply(2, width - 1), 1));
            result.target_first = result.emitted.front();
            result.target_second = result.emitted.size() > 1 ? result.emitted[1] : -1;
          } else {
            result.target_first = decision.final_ids[0]; result.target_second = width > 1 ? decision.final_ids[1] : -1;
            accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, decision.final_ids,
                selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
          }
        } else {
          accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, rows,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
        }
        member.accepted_tokens = accepted.consumed; selected[j] = result.next_id = accepted.next_id;
        result.agreed = accepted.first_agrees; result.consumed_drafts = accepted.consumed_drafts;
        member.discarded_draft_rows = 0;
        if (step.phase->recursive_draft_repair || step.phase->draft_chain_steps)
          member.discarded_draft_rows += member.draft_input_rows - member.base->draft->lag -
              (member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1);
        result.accepted_draft = result.consumed_drafts != 0;
        result.processed_target = member.accepted_tokens;
        // Zero proposals never ran the draft head or its catch-up.
        result.processed_draft = width - 1 + (width > 1 ? member.base->draft->lag : 0) +
            member.base->draft->lag + (member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1) +
            member.discarded_draft_rows;
        result.stopped = accepted.stopped;
        publication.selected(j, result, selection.output_quota);
      }
    }, [&](StepState &step) { repair_recursive_draft(step); }, std::move(observe), publication);
  }

  template<class Observe, class Publication = NoEvaluationPublication>
  void finish_evaluations(std::span<const EvaluationOutput> outputs,
                         std::span<std::int32_t> next_ids, Observe observe, Publication &&publication = {}) {
    if (active_step && active_step->phase && verifies_prefix(active_step->phase->family->role))
      throw std::runtime_error("verification requires explicit accepted-boundary selection");
    finish_selected_evaluations(outputs, next_ids,
        [](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> next) {
          for (std::size_t j = 0; j < step.members.size(); ++j) {
            if (step.withdrawn(step.members[j])) { next[j] = -1; continue; }
            step.members[j].accepted_tokens = step.members[j].input_tokens; next[j] = ids[j];
          }
        }, std::move(observe), publication);
  }

  // Generation policy observes only the original terminal graph. Pending has
  // already been emitted; an accepted draft and its bonus are new outputs.
  template<class Observe, class Publication = NoEvaluationPublication>
  void finish_mtp_rounds(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe, Publication &&publication = {}) {
    if (active_step && active_step->phase && active_step->phase->draft_chain_steps) {
      finish_recursive_mtp(selections, eos, outputs, results, std::move(observe), std::forward<Publication>(publication));
      return;
    }
    if (!active_step || selections.empty() || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size() || !active_step->phase ||
        !has_mtp(active_step->phase->family->role) || is_mtp_prefill(active_step->phase->family->role))
      throw std::runtime_error("MTP finish requires its original complete generation batch");
    const auto role = active_step->phase->family->role;
    const auto &selected_sampler = active_step->phase->family->sampling->configuration;
    for (std::size_t j = 0; j < selections.size(); ++j) {
      if (selections[j].session != active_step->members[j].session || selections[j].session != outputs[j].session)
        throw std::runtime_error("MTP selection differs from its original batch member");
      results[j].reset();
      if (active_step->withdrawn(active_step->members[j])) continue;
      if (!selections[j].output_quota || selections[j].expected_pending < 0 ||
          selections[j].expected_pending >= vocabulary || inference::request_token_stopped(
              selections[j].expected_pending, eos, active_step->members[j].stop_policy))
        throw std::runtime_error("MTP generation requires its emitted pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(2);
      results[j].pending = selections[j].expected_pending;
    }
    hyle::container::SmallVector<std::int32_t, 2> next;
    if (!next.try_resize(selections.size(), -1)) throw std::bad_alloc{};
    finish_selected_evaluations(outputs, std::span(next.data(), next.size()),
        [&](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> selected) {
          for (std::size_t j = 0; j < selections.size(); ++j) {
            const auto &selection = selections[j]; auto &member = step.members[j]; auto &result = results[j];
            if (step.withdrawn(member)) { selected[j] = -1; continue; }
            if (member.first_input != selection.expected_pending)
              throw std::runtime_error("MTP generation input differs from its emitted pending token");
            result.target_first = ids[drafts_token(role) ? 2 * j : j]; result.draft = member.draft_token;
            result.target_second = -1;
            if (drafts_token(role)) {
              if (result.draft < 0 || result.draft >= vocabulary) throw std::runtime_error("MTP draft ID outside vocabulary");
              result.target_second = ids[2 * j + 1];
            }
            const auto rows = drafts_token(role) ? std::size_t{2} : std::size_t{1};
            const std::array inputs{selection.expected_pending, result.draft};
            const auto verified_inputs = std::span<const std::int32_t>(inputs).first(rows);
            hyle::inference::mtp::AcceptedPrefix accepted;
            if (selected_sampler.enabled) {
              const std::array flags{std::uint8_t(member.sampled_accept)};
              // The original K1 graph's second ID is its residual/bonus final;
              // its first ID is already Select(accept, proposal, final).
              const std::array finals{drafts_token(role) ? result.target_second : result.target_first};
              accepted = hyle::inference::mtp::select_sampled_prefix(verified_inputs,
                  std::span<const std::uint8_t>(flags).first(rows - 1), finals,
                  selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
            } else {
              const std::array targets{result.target_first, result.target_second};
              accepted = hyle::inference::mtp::select_greedy_prefix(verified_inputs,
                  std::span<const std::int32_t>(targets).first(rows), selection.output_quota, eos,
                  static_cast<std::int32_t>(vocabulary), result.emitted, member.stop_policy);
            }
            member.accepted_tokens = accepted.consumed;
            selected[j] = result.next_id = accepted.next_id;
            result.agreed = accepted.first_agrees; result.accepted_draft = accepted.consumed_drafts != 0;
            result.processed_target = accepted.consumed; result.processed_draft = member.base->draft->lag + 1;
            result.proposal_count = drafts_token(role) ? 1 : 0;
            result.consumed_drafts = accepted.consumed_drafts; result.verified_rows = step.input_tokens;
            result.stopped = accepted.stopped;
            publication.selected(j, result, selection.output_quota);
          }
        }, std::move(observe), publication);
  }

};


inline hyle::Status Session::stop() noexcept {
  if (stopping) return retirement_status;
  stopping = true;
  if (pending) {
    if (!pending->cancelled && pending->submitted && pending->execution.has_completion() &&
        pending->phase && model_ && pending->members.size() > 1 &&
        std::ranges::any_of(pending->members, [](const auto &member) { return !member.session->stopping; })) {
      pending->discard_stopped_members = true;
      // This acknowledges the requested withdrawal, rather than reporting a
      // device/retirement error. The completed member reports kCancelled.
      return retirement_status;
    }
    pending->cancelled = true;
    if (pending->execution.has_completion()) {
      const auto cancelled = pending->execution.request_cancel();
      if (!cancelled.IsUnimplemented()) inference::record_retirement_status(retirement_status, cancelled);
    }
    if (pending->execution.run) pending->execution.run->close().IgnoreError();
  }
  return retirement_status;
}
inline inference::RetirementObservation Session::try_retire() noexcept {
  if (!stopping) return {false, hyle::Status::FailedPrecondition("stop the Qwen session before retirement")};
  // The original caller must finish successful work for surviving members.
  // Neither a stopped owner nor its drain may turn this into a peer abort.
  if (pending && pending->discard_stopped_members && !pending->cancelled)
    return {false, retirement_status};
  inference::RetirementObservation observed{true, {}};
  if (!model_) observed = resources_->retire(false);
  else if (pending) observed = model_->try_retire_episode();
  inference::record_retirement_status(retirement_status, observed.status);
  if (observed.retired) {
    tails = {}; draft_tails = {}; tail_epoch = 0; draft_tail_epoch = 0; head.reset();
  }
  return {observed.retired, retirement_status};
}
inline inference::RetirementObservation Session::drain() noexcept {
  if (pending && pending->discard_stopped_members && !pending->cancelled)
    return {false, retirement_status};
  inference::RetirementObservation observed{true, {}};
  if (!model_) observed = resources_->retire(true);
  else if (pending) observed = model_->abort_episode();
  // Driving an unstopped Session preserves its head and continued-use contract.
  // In particular, a peer's ordinary episode cancellation is not made sticky.
  if (!stopping) {
    auto status = retirement_status;
    inference::record_retirement_status(status, observed.status);
    return {false, status};
  }
  // A stopped Session still proves/releases through the original retirement gate.
  inference::record_retirement_status(retirement_status, observed.status);
  return try_retire();
}
inline Session::~Session() { (void)drain(); }


} // namespace qwen35
#include "device_continuation.h"
#include "session_transfer.h"
