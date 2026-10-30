#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/state_versions.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/weight_binding.h"
#if defined(HYLE_TEST_GLM_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_TEST_GLM_STATE_CPU)
#include "hyle/graph/cpu.h"
#endif
#include "d5_glm_remaining_fragments.h"
#include "glm_mtp_target_component.h"
#include "glm_official_index_cache.h"
#include "glm_flash_selection_extent.h"
#include "glm_mtp_target43_component.h"
#include "glm_original_mtp_roles.h"
#include "glm_original_indexer_roles.h"
#include "glm_original_indexer_reference.h"
#include "glm_shared_vocabulary.h"
#include "glm_shared_vocabulary_reference.h"
#include "glm_original_attention_roles.h"
#include "glm_original_attention_reference.h"
#include "glm_original_moe_roles.h"
#include "glm_original_moe_resident_weights.h"
#include "glm_original_moe_fragment.h"
#include "glm_original_moe_reference.h"
#include "glm_original_artifact_source.h"
#include "glm_original_target_source.h"
#include "glm_original_expert_frontier.h"
#if defined(HYLE_TEST_GLM_Q30_NEXTN_ARTIFACTS_HEADER) && defined(HYLE_TEST_GLM_Q30_TARGET43_ARTIFACTS_HEADER) && defined(HYLE_TEST_GLM_Q30_TARGET44_ARTIFACTS_HEADER)
#include HYLE_TEST_GLM_Q30_NEXTN_ARTIFACTS_HEADER
#include HYLE_TEST_GLM_Q30_TARGET43_ARTIFACTS_HEADER
#include HYLE_TEST_GLM_Q30_TARGET44_ARTIFACTS_HEADER
#define HYLE_TEST_GLM_Q30_ACTUAL_PARAMETERS 1
#endif
#if defined(HYLE_TEST_GLM_MLX_NEXTN_ARTIFACTS_HEADER) && defined(HYLE_TEST_GLM_MLX_TARGET_ARTIFACTS_HEADER)
#include HYLE_TEST_GLM_MLX_NEXTN_ARTIFACTS_HEADER
#include HYLE_TEST_GLM_MLX_TARGET_ARTIFACTS_HEADER
#define HYLE_TEST_GLM_MLX_ACTUAL_PARAMETERS 1
#endif
#include "glm_mxfp4_moe_reference.h"
#include "glm_gguf_flash_source.h"
#include "glm_gguf_flash_reference.h"
#if defined(HYLE_TEST_GLM_B12_ARTIFACTS_HEADER)
#include HYLE_TEST_GLM_B12_ARTIFACTS_HEADER
#endif
#if defined(HYLE_TEST_GLM_B13_ARTIFACTS_HEADER)
#include HYLE_TEST_GLM_B13_ARTIFACTS_HEADER
#endif
#include "glm_mlx_runtime_source.h"
#include "glm_mlx_expert_frontier.h"
#include "glm_mlx_original_reference.h"
#include "glm_mlx_target_fragment.h"
#include "glm_mlx_target_reference.h"
#include "model_resident_fragment.h"
#include "model_graph_run.h"
#include "glm_target_verifier_fragment.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "glm_stochastic_prefix_fragment.h"
#include "glm_original_target44_fragment.h"
#include "glm_original_target44_reference.h"
#include "glm_original_target43_fragment.h"
#include "glm_original_target43_roles.h"
#include "glm_original_target43_reference.h"
#include "../../engine/inference/sampling_cursor.h"
#include "sampling_fixture.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace glm = d5_glm_remaining_fragments;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace codec = glm_official_index_cache;
namespace original_prefix = glm_original_mtp_roles;
namespace original_index = glm_original_indexer_roles;
namespace index_reference = glm_original_indexer_reference;
namespace shared_lm = glm_shared_vocabulary;
namespace lm_reference = glm_shared_vocabulary_reference;
namespace original_attention = glm_original_attention_roles;
namespace attention_reference = glm_original_attention_reference;
namespace original_moe = glm_original_moe_roles;
namespace artifact_source = glm_original_artifact_source;
namespace resident_moe = glm_original_moe_resident_weights;
namespace moe_fragment = glm_original_moe_fragment;
namespace moe_reference = glm_original_moe_reference;
namespace resident_fragment = model_resident_fragment;
namespace target_verifier = glm_target_verifier_fragment;
namespace original_target44 = glm_original_target44_roles;
namespace target44_fragment = glm_original_target44_fragment;
namespace target44_reference = glm_original_target44_reference;
namespace original_target43 = glm_original_target43_roles;
namespace target43_fragment = glm_original_target43_fragment;
namespace target43_local = glm_original_target43_reference;
namespace gguf_flash=glm_gguf_flash_fragment;
namespace gguf_source=glm_gguf_flash_source;
namespace gguf_reference=glm_gguf_flash_reference;
namespace packed_reference=qwen24_original_nextn_reference;
namespace packed_base=qwen27_original_nextn_reference;
namespace mlx_model=glm_mlx_mtp_fragment;
namespace mlx_runtime=glm_mlx_runtime_source;
namespace mlx_roles=glm_mlx_original_roles;
namespace mlx_reference=glm_mlx_original_reference;
namespace mlx_target=glm_mlx_target_fragment;
namespace mlx_target_reference=glm_mlx_target_reference;
using Bytes = std::vector<std::byte>;
using Value = sem::ValueId;
using Reference = std::function<std::vector<float>(std::span<const float>)>;
constexpr auto kModel = glm::kCases[4];
constexpr std::int64_t kHidden = 4096, kQueryRank = 1536, kLatent = 512;
constexpr std::int64_t kIndexDepth = 128, kIndexHeads = 32, kRing = 6;
constexpr std::int64_t kCapacity = 2060, kPoolCapacity = kCapacity / 4;
constexpr std::int32_t kBasePosition = 2055;
constexpr std::int64_t kIds = 2051, kSelected = 2048;
constexpr std::int64_t kJournalRows = 2;
static_assert(kIndexDepth == codec::kDepth && kIndexHeads == codec::kHeads);

// Generated GGUF/F32 projections and target boundaries with the fixed06992
// default UE8M0 index-cache/query arithmetic. Both raw key/score rings store
// BF16. Completed K4 pools persist only E4 codes and F32 power-of-two scales;
// pooling, normalized Hadamard and quantization use explicit BF16 boundaries.
// Projection diagnostics remain separate: these generated GGUF/Q8/F32 roles
// do not claim equality to original BF16 checkpoint projection operands.
// Whole NextN EH/MLA/MoE/hidden and all indexer state publish through one Head.
// Same-iteration reuse skips the ENTIRE indexer, retaining both encoded banks,
// BF16 raw rings and indexer cursors while MLA/hidden advance independently.
// The accepted nx journal supports exact catchup; normal draft-extend hands
// its actual last-accepted hidden/full IDs to next decode step0 without a new
// query. The target44 test adds KDA/conv/mHC state to that same atomic draft.
// This logical codec is not vendor page/footer ABI or a full original model.
// The separate original-supplier mode injects the fourteen pinned official HF
// layer45 input/query/indexer roles into the same complete state journey. Its
// declared target-hidden/embedding inputs are explicitly rounded to BF16;
// MLA/MoE/head and the finite initial history remain generated boundaries.
// The additional original-attention mode retains all26 physical suppliers,
// full original MLA geometry, coupled BF16 KV_b, learned kv/post norms and
// explicit BF16 boundaries through the same full-V LM/pending feedback. Its
// target-hidden, initial history and MoE remain declared generated inputs.
template<class T> T take(hyle::Result<T> result, const char *where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, const char *where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> values) {
  auto buffer = take(m::Buffer::create(owner.ref(), values.size(), 256), "initial/parameter backing");
  check(buffer.view().write(values), "explicit initial/parameter bytes");
  return take(owner.adopt_buffer(std::move(buffer)), "qualified readonly initial value");
}
Bytes read(const c::ReadLease &lease) {
  Bytes result(lease.size());
  check(take(lease.view(), "readonly actual state").read(result), "actual state read"); return result;
}
template<class T> std::vector<T> typed(const c::ReadLease &lease) {
  if (lease.size() % sizeof(T)) throw std::runtime_error("state scalar extent");
  std::vector<T> result(lease.size() / sizeof(T));
  check(take(lease.view(), "actual typed state view").read(std::as_writable_bytes(std::span(result))), "typed state read");
  return result;
}
enum Entry : std::size_t {
  kCache, kRawKey, kRawGate, kPool, kPoolScales, kHiddenState, kQueryState, kPending,
  kMlaCursor, kIndexCursor, kPoolCount, kRingOffset, kTailValid, kTailStart,
  kSelection, kJournal, kLagCount, kEpoch, kEntries
};
constexpr std::size_t kResources = kEpoch;
std::vector<Entry> state_resources(bool plain_index=false) {
  std::vector<Entry> r;for(std::size_t i=0;i<kResources;++i)if(!plain_index||i!=kPoolScales)r.push_back(Entry(i));return r;
}
std::array<c::TensorDesc, kResources> descriptions(bool plain_index=false) {
  const auto scalar = sem::dense({1}, c::DataType::kInt32);
  std::array<c::TensorDesc,kResources> result{sem::dense({kCapacity, kLatent}), sem::dense({kRing, kIndexDepth}, c::DataType::kBFloat16),
      sem::dense({kRing, kIndexDepth}, c::DataType::kBFloat16),
      sem::dense({kPoolCapacity, kIndexDepth}, c::DataType::kFloat8E4M3FN), sem::dense({kPoolCapacity, 1}),
      sem::dense({1, kHidden}), sem::dense({1, kQueryRank}), scalar, scalar,
      scalar, scalar, scalar, scalar, scalar, sem::dense({kIds}, c::DataType::kInt32),
      sem::dense({kJournalRows, kHidden}), scalar};
  if(plain_index){result[kRawKey]=sem::dense({kRing,kIndexDepth});result[kRawGate]=result[kRawKey];result[kPool]=sem::dense({kPoolCapacity,kIndexDepth});}
  return result;
}
inline constexpr std::array<Entry, 13> kSparseEntries{kCache, kRawKey, kRawGate, kPool, kPoolScales,
    kQueryState, kMlaCursor, kIndexCursor, kPoolCount, kRingOffset, kTailValid, kTailStart, kSelection};
std::vector<Entry> sparse_entries(bool plain=false) {
  std::vector<Entry> r;for(const auto entry:kSparseEntries)if(!plain||entry!=kPoolScales)r.push_back(entry);return r;
}
struct SparseState13 final {
  std::array<Value, kSparseEntries.size()> values;
  SparseState13() = delete;
  SparseState13(sem::Fragment &f, std::array<Value, kSparseEntries.size()> actual) : values(actual) {
    const auto desc = descriptions();
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (values[i].ordinal() >= f.descriptions.size() || f.description(values[i]) != desc[kSparseEntries[i]])
        throw std::runtime_error("sparse attention requires all13 actual, correctly typed state values");
      for (std::size_t j = 0; j < i; ++j)
        if (values[i].ordinal() == values[j].ordinal()) throw std::runtime_error("independent sparse state slots share a graph identity");
    }
  }
  Value operator[](Entry entry) const {
    const auto found = std::find(kSparseEntries.begin(), kSparseEntries.end(), entry);
    if (found == kSparseEntries.end()) throw std::runtime_error("entry is not a sparse-attention state port");
    return values[std::size_t(found - kSparseEntries.begin())];
  }
};
SparseState13 sparse_state(sem::Fragment &f, const std::array<Value, kResources> &state) {
  return SparseState13(f, {state[kCache], state[kRawKey], state[kRawGate], state[kPool], state[kPoolScales],
      state[kQueryState], state[kMlaCursor], state[kIndexCursor], state[kPoolCount], state[kRingOffset],
      state[kTailValid], state[kTailStart], state[kSelection]});
}
struct SkipJournal final { Value rows, count; };
std::vector<float> embedding_values(std::int32_t token) {
  auto values = d5_gguf_test::activation(kHidden, 11 + 14 * token);
  for (auto &value : values) value = std::abs(value) + 0.5F;
  return values;
}
Value increment(sem::Fragment &f, Value old, std::int32_t delta = 1) {
  const auto amount = f.constant(sem::dense({1}, c::DataType::kInt32), sem::bytes(std::vector{delta}));
  const auto joined = f.one(c::Operation(c::Concat{.axis = 0}), {old, amount});
  const auto prefix = f.one(c::Operation(c::Scan{.axis = 0}), {joined});
  const auto last = f.constant(sem::dense({1}, c::DataType::kInt32), sem::bytes(std::vector<std::int32_t>{1}));
  return f.one(c::Operation(c::Gather{}), {prefix, last});
}
Value advance_ring(sem::Fragment &f, Value old, std::int32_t size) {
  std::vector<std::int32_t> values(std::size_t(size), 0);
  for (std::int32_t i = 0; i < size; ++i) values[std::size_t(i)] = (i + 1) % size;
  const auto table = f.constant(sem::dense({size}, c::DataType::kInt32), sem::bytes(values));
  return f.one(c::Operation(c::Gather{}), {table, old});
}
struct LmObservations final {
  bool full_checked=false;
  std::size_t full_values=0,head_calls=0,pending_checks=0;
};
struct OriginalNextn;
struct OriginalAttention final {
  std::shared_ptr<const original_attention::Data> source;
  original_attention::Roles roles;
  std::array<original_index::ReadonlyRole,2> prepared;
  c::LeasePool prepared_pool;
  bool full_checked=false;
  std::size_t projection_values=0,attention_calls=0,cold_submissions=0;
};
struct OriginalMoe final {
  original_moe::Roles roles;
  glm_fp8_arithmetic::Arithmetic arithmetic;
  std::function<void(std::span<const std::int32_t>)> before_upload;
  std::unique_ptr<resident_moe::CpuSupply> cpu_supply;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  std::unique_ptr<resident_moe::CudaSupply> cuda_supply;
#endif
  bool full_checked=false;
  std::size_t completed_episodes=0,projection_values=0;
  std::vector<original_moe::Selection> selections;
  std::vector<std::size_t> union_sizes;
};
struct OriginalTarget44 final {
  std::shared_ptr<gguf_source::Source> converted;
  std::shared_ptr<mlx_runtime::Source> mlx;
  std::size_t logical_source_bytes=7554137944ULL;
  std::shared_ptr<const OriginalNextn> artifact_owner;
  original_target44::Bound parameters;
  std::shared_ptr<OriginalMoe> moe;
  bool full_checked=false;
  std::size_t checked_calls=0;
};
struct OriginalSparseLayer final {
  const c::ModelArtifactProfileDescription *profile=&glm::core::profiles::glm53_flash_official;
  glm_fp8_arithmetic::Arithmetic arithmetic;
  std::shared_ptr<const original_index::Data> indexer;
  original_index::Roles indexer_roles;
  std::shared_ptr<OriginalAttention> attention;
};
struct OriginalTarget43 final {
  std::shared_ptr<gguf_source::Source> converted;
  std::shared_ptr<mlx_runtime::Source> mlx;
  std::size_t logical_source_bytes=7427838808ULL;
  std::shared_ptr<const OriginalNextn> artifact_owner;
  original_target43::Bound parameters;
  std::shared_ptr<OriginalSparseLayer> sparse;
  std::shared_ptr<OriginalMoe> moe;
  bool full_checked=false;
  std::size_t checked_calls=0;
};
struct OriginalNextn final {
  c::ModelArtifactProfile profile;
  std::shared_ptr<gguf_source::Source> converted;
  std::shared_ptr<mlx_runtime::Source> mlx;
  c::ModelArtifactProfileDescription profile_description=glm::core::profiles::glm53_flash_official;
  std::shared_ptr<const original_prefix::Data> prefix;
  glm_fp8_arithmetic::Arithmetic arithmetic;
  std::array<original_index::ReadonlyRole, static_cast<std::size_t>(original_prefix::Parameter::kCount)> prefix_roles;
  std::shared_ptr<const shared_lm::Vocabulary> vocabulary;
  std::shared_ptr<LmObservations> lm_observations;
  std::shared_ptr<OriginalSparseLayer> sparse;
  std::shared_ptr<OriginalMoe> moe;
};
std::shared_ptr<const OriginalNextn> original_nextn(const std::filesystem::path &input,
    const std::filesystem::path &query, const std::filesystem::path &indexer,
    const c::StorageOwner &owner,std::shared_ptr<const shared_lm::Vocabulary> vocabulary = {},
    std::shared_ptr<OriginalAttention> attention = {},std::shared_ptr<OriginalMoe> moe = {}) {
  auto result = std::make_shared<OriginalNextn>();
  result->vocabulary = std::move(vocabulary);
  result->sparse=std::make_shared<OriginalSparseLayer>();
  result->sparse->attention = std::move(attention);
  result->moe=std::move(moe);
  if(result->moe&&!result->sparse->attention)throw std::runtime_error("original MoE requires the same complete original attention/LM journey");
  if(result->sparse->attention&&!result->vocabulary)throw std::runtime_error("original complete attention mode requires the same original shared LM feedback");
  if(result->vocabulary)result->lm_observations=std::make_shared<LmObservations>();
  result->prefix = std::make_shared<const original_prefix::Data>(original_prefix::Data::load(input, query));
  result->sparse->indexer=std::make_shared<const original_index::Data>(original_index::Data::load(indexer));
  for (std::size_t i = 0; i < result->prefix_roles.size(); ++i) {
    const auto role = static_cast<original_prefix::Parameter>(i);
    result->prefix_roles[i] = {"original GLM NextN input/query", original_prefix::description(role), owner,
        upload(owner, original_prefix::payload(*result->prefix, role))};
  }
  result->sparse->indexer_roles = original_index::bind(*result->sparse->indexer, owner);
  return result;
}
bool full_lm(const std::shared_ptr<const OriginalNextn> &p){return p&&(p->vocabulary||p->mlx||p->converted);}
struct External final { std::size_t constant; original_index::ReadonlyRole role; };
struct CodecPorts final { std::size_t codes = 0, scales = 0, rotated = 0, transformed = 0, input = 0; };
struct AttentionPorts final {
  std::size_t qb_raw=0,qb_stored=0,kva_raw=0,kva_stored=0,kvnorm_raw=0,kvnorm_stored=0;
  std::size_t absorbed_stored=0,attention_stored=0,value_stored=0,output_raw=0,output_stored=0;
  std::size_t postnorm_raw=0,postnorm_stored=0,residual_sum=0,residual_stored=0;
};
struct Recipe final {
  sem::Fragment fragment;
  std::vector<Entry> state_entries=state_resources();
  std::shared_ptr<gguf_source::Source> converted;
  std::optional<gguf_flash::NextnResult> converted_nextn;
  std::optional<gguf_flash::Target43Result> converted43;
  std::optional<target44_fragment::LayerResult<glm_gguf_mtp_fragment::Moe>> converted44;
  std::optional<gguf_flash::Parameters> converted_parameters;
  std::optional<mlx_model::State> pooled_before,pooled_after;
  std::shared_ptr<mlx_runtime::Source> mlx;
  std::optional<mlx_model::Result> mlx_result;
  std::optional<mlx_model::Boundary> mlx_boundary;
  std::optional<mlx_target::Result43<mlx_model::Moe>> mlx43;
  std::optional<target44_fragment::LayerResult<mlx_model::Moe>> mlx44;
  std::optional<mlx_model::Mla> mlx_target_mla;
  std::optional<mlx_model::types::Projection> mlx_target_query;
  std::optional<mlx_model::fp::Norm> mlx_target_query_norm;
  std::optional<mlx_model::History> mlx_history;
  std::optional<mlx_model::Parameters> mlx_parameters;
  std::vector<std::pair<Value,std::size_t>> mlx_ports;
  std::shared_ptr<const void> profile_owner;
  std::shared_ptr<const OriginalNextn> original;
  std::shared_ptr<OriginalSparseLayer> sparse;
  std::size_t sparse_hidden=std::numeric_limits<std::size_t>::max(),sparse_query=std::numeric_limits<std::size_t>::max();
  bool attention_postnorm=false;
  // The same MoE math/supply belongs to its actual layer. A target component
  // must not invent a NextN identity just to use the ordinary two Graph phases.
  std::shared_ptr<OriginalMoe> moe;
  std::shared_ptr<OriginalTarget44> target44_source;
  std::optional<target44_fragment::Result> target44;
  std::vector<std::pair<Value,std::size_t>> target44_ports;
  std::shared_ptr<OriginalTarget43> target43_source;
  std::optional<target43_fragment::Result> original43;
  target43_local::QueryPorts target43_query;
  std::vector<std::pair<Value,std::size_t>> original43_ports;
  std::size_t target43_mhc=std::numeric_limits<std::size_t>::max();
  std::vector<External> external;
  AttentionPorts attention_ports;
  std::optional<Value> sparse_output;
  std::optional<moe_fragment::Result> original_moe;
  std::vector<std::pair<Value,std::size_t>> moe_ports;
  std::size_t moe_input=std::numeric_limits<std::size_t>::max(),moe_output=std::numeric_limits<std::size_t>::max();
  std::array<std::size_t, 7> original_prefix_ports{};
  std::size_t original_hidden = 0, original_embedding = 0, wk_accum = 0, wk_stored = 0;
  std::size_t index_query_accum = 0, index_query_stored = 0;
  std::size_t lm_pending_input = 0,lm_combined = 0,lm_norm_raw = 0,lm_norm_stored = 0,lm_logits = 0,lm_top_values = 0,lm_ids = 0;
  std::array<std::size_t, kResources> state_ports;
  bool fresh = false;
  bool draft_extend = false;
  bool catchup = false, direct_catchup = false, cuda = false;
  std::int32_t position = kBasePosition, selected_tail = 0, journal_slot = 0;
  glm::InputReference input_reference;
  Reference query_reference, absorbed_query_reference, latent_reference, key_reference, gate_reference;
  Reference index_query_reference, score_weight_reference, output_reference;
  glm::moe::MoeReference moe_reference;
  std::vector<float> input_scale, query_scale, ffn_scale, head_scale;
  std::size_t embedding, eh, nx, query, latent, attention, output, residual, ffn_input, ffn_output;
  std::size_t key = 0, raw_key = 0, gate = 0, raw_gate = 0, gate_projection = 0;
  std::size_t compressed = 0, index_query = 0, score_weight = 0, scores = 0;
  CodecPorts pool_codec, query_codec;
  std::array<std::size_t, glm_mtp_target43_component::Ports::count> target43_ports{};
  glm_mtp_target43_component::Reference target43_reference;
};
std::size_t expose(Recipe &r, Value value, bool copy = false) {
  if (copy) value = r.fragment.one(c::Operation(c::Copy{}), {value});
  const auto port = r.fragment.outputs.size(); r.fragment.outputs.push_back(value); return port;
}
Value external(Recipe &r, const original_index::ReadonlyRole &role) {
  const auto slot = r.fragment.constants.size();
  const auto value = r.fragment.constant(role.description, {});
  r.external.push_back({slot, role}); return value;
}
bool has_shared_lm(const Recipe &r) {return bool(r.mlx)||bool(r.converted)||(r.original&&bool(r.original->vocabulary));}
bool has_original_attention(const Recipe &r) {return r.sparse&&bool(r.sparse->attention);}
bool has_original_moe(const Recipe &r) {return bool(r.moe);}
Value external(Recipe &r,const shared_lm::ReadonlyRole &role) {
  // Both test suppliers use ordinary immutable Graph constant permissions.
  return external(r,original_index::ReadonlyRole{role.name,role.description,role.owner,role.lease});
}
moe_fragment::Parameters moe_parameters(Recipe &r) {
  if(!r.moe)throw std::runtime_error("MoE construction requires its actual layer-owned source");
  auto &f=r.fragment;const auto &roles=r.moe->roles;
  moe_fragment::Parameters parameters;
  parameters.router=external(r,roles.router);parameters.bias=external(r,roles.bias);
  for(std::size_t leg=0;leg<3;++leg) {
    parameters.shared[leg]=external(r,roles.shared[leg]);
    parameters.banks[leg]=f.input(original_moe::resident_description(static_cast<original_moe::Leg>(leg),
        roles.source->model(),roles.source->weight_storage()));
  }
  parameters.map=f.input(sem::dense({original_moe::kExpertDomain},c::DataType::kInt32));
  return parameters;
}
Value change(Recipe &r, Entry entry, Value value, bool copy = false) {
  r.state_ports[entry] = expose(r, value, copy); return value;
}
CodecPorts expose_codec(Recipe &r, const codec::Quantized &q) {
  return {expose(r, q.codes), expose(r, q.scale), expose(r, q.rotated),
      expose(r, q.transformed), expose(r, q.input, true)};
}
struct EncodedPool final { Value codes, scales; };
// The same ordinary indexer-history equations serve a fresh query and an
// explicit accepted-feature catchup. A skip recipe never calls this function.
EncodedPool append_history(Recipe &r, const SparseState13 &state,
                     Value nx, Value next_index_cursor, int role_offset = 0) {
  auto &f = r.fragment;
  Value key, raw_key, raw_gate;
  if (r.sparse) {
    // Authenticated layer-local views determine these parameters. role_offset
    // only distinguishes the generated fixtures in the branch below.
    const auto rounded_nx = codec::cast(f, nx, c::DataType::kBFloat16);
    const auto &roles = r.sparse->indexer_roles;
    const auto wk = f.one(c::model_ops::dense_linear_f32(), {rounded_nx, external(r, roles[original_index::kKey])});
    r.wk_accum = expose(r, wk);
    const auto stored_wk = codec::cast(f, wk, c::DataType::kBFloat16); r.wk_stored = expose(r, stored_wk);
    key = f.one(c::Operation(c::LayerNorm{.epsilon = 1.0e-6F}),
        {codec::cast(f, stored_wk, c::DataType::kFloat32), external(r, roles[original_index::kNorm]),
         external(r, roles[original_index::kBias])});
    r.key = expose(r, key);
    raw_key = codec::cast(f, key, c::DataType::kBFloat16); r.raw_key = expose(r, raw_key);
    const auto gate_projection = f.one(c::model_ops::dense_linear_f32(),
        {rounded_nx, external(r, roles[original_index::kGate])});
    r.gate_projection = expose(r, gate_projection);
    raw_gate = codec::cast(f, gate_projection, c::DataType::kBFloat16); r.raw_gate = expose(r, raw_gate);
    r.gate = expose(r, codec::cast(f, raw_gate, c::DataType::kFloat32));
  } else {
  auto wk = glm::projection(f, nx, kIndexDepth, kHidden, 241 + role_offset, false);
  auto scale = sem::oracle::scale(kIndexDepth);
  std::vector<float> bias(kIndexDepth);
  for (std::int64_t d = 0; d < kIndexDepth; ++d) bias[std::size_t(d)] = float(d % 7 - 3) / 32.0F;
  const auto sw = f.constant(sem::dense({kIndexDepth}), sem::bytes(scale));
  const auto bw = f.constant(sem::dense({kIndexDepth}), sem::bytes(bias));
  key = f.one(c::Operation(c::LayerNorm{.epsilon = 1.0e-6F}), {wk.output, sw, bw});
  r.key = expose(r, key);
  raw_key = codec::cast(f, key, c::DataType::kBFloat16); r.raw_key = expose(r, raw_key);
  r.key_reference = [wk = std::move(wk), scale, bias](auto x) { return glm::layer_norm(wk.reference(x), scale, bias); };
  auto wg = glm::linear::projection(f, nx, kIndexDepth, kHidden, glm::linear::Storage::kQ8_0, 257 + role_offset);
  r.gate_projection = expose(r, wg.output);
  raw_gate = codec::cast(f, wg.output, c::DataType::kBFloat16); r.raw_gate = expose(r, raw_gate);
  const auto gate = codec::cast(f, raw_gate, c::DataType::kFloat32); r.gate = expose(r, gate);
  r.gate_reference = [wg = std::move(wg)](auto x) { return wg.reference(x); };
  }
  const auto keys = change(r, kRawKey, f.one(c::Operation(c::Scatter{}), {state[kRawKey], state[kRingOffset], raw_key}));
  const auto gates = change(r, kRawGate, f.one(c::Operation(c::Scatter{}), {state[kRawGate], state[kRingOffset], raw_gate}));
  EncodedPool pool{state[kPool], state[kPoolScales]};
  if (r.position % 4 == 3) {
    std::vector<std::int32_t> ring_rows(4);
    for (std::int32_t t = 0; t < 4; ++t) ring_rows[std::size_t(t)] = (r.position - 3 + t) % std::int32_t(kRing);
    const auto four = f.constant(sem::dense({1, 4}, c::DataType::kInt32), sem::bytes(ring_rows));
    const auto grouped_key = codec::cast(f, f.one(c::Operation(c::Gather{}), {keys, four}), c::DataType::kFloat32);
    const auto grouped_gate = codec::cast(f, f.one(c::Operation(c::Gather{}), {gates, four}), c::DataType::kFloat32);
    std::vector<float> ape(4 * kIndexDepth);
    for (std::size_t i = 0; i < ape.size(); ++i) ape[i] = float(int(i * 7 % 19) - 9) / 16.0F;
    const auto positional = r.sparse ? external(r, r.sparse->indexer_roles[original_index::kApe]) :
        f.constant(sem::dense({1, 4, kIndexDepth}), sem::bytes(ape));
    const auto logits = glm::binary(f, c::BinaryKind::kAdd, grouped_gate, positional);
    const auto rows = f.one(c::Operation(c::Copy{}), {glm::permute(f, logits, {0, 2, 1})});
    auto probability = f.one(c::Operation(c::Softmax{}), {glm::reshape(f, rows, {kIndexDepth, 4})});
    probability = f.one(c::Operation(c::Copy{}), {glm::permute(f, glm::reshape(f, probability, {1, kIndexDepth, 4}), {0, 2, 1})});
    const auto pooled = glm::reduce(f, glm::binary(f, c::BinaryKind::kMultiply, probability, grouped_key), 1);
    r.compressed = expose(r, pooled);
    const auto encoded = codec::quantize(f, codec::bfloat_boundary(f, pooled), 1);
    r.pool_codec = expose_codec(r, encoded);
    pool.codes = change(r, kPool, f.one(c::Operation(c::Scatter{}), {state[kPool], state[kPoolCount], encoded.codes}));
    pool.scales = change(r, kPoolScales, f.one(c::Operation(c::Scatter{}), {state[kPoolScales], state[kPoolCount], encoded.scale}));
    change(r, kPoolCount, increment(f, state[kPoolCount]));
    change(r, kTailStart, increment(f, state[kTailStart], 4));
  }
  change(r, kIndexCursor, next_index_cursor, true);
  change(r, kRingOffset, advance_ring(f, state[kRingOffset], kRing));
  change(r, kTailValid, advance_ring(f, state[kTailValid], 4));
  return pool;
}
// Pure shared Graph construction. nx and all13 state ports belong to this
// layer. Fixture role_offset affects immutable projection roles only; normal
// NextN always uses0. Target43 has no EH/pending/plain residual or skip journal.
Value append_sparse_attention(Recipe &r, const SparseState13 &state, Value nx,
                               int role_offset = 0, const SkipJournal *journal = nullptr,
                               const Value *original_qr = nullptr) {
  if (!r.fresh && !journal) throw std::runtime_error("indexer skip requires its actual NextN journal");
  const auto extent=r.fresh?glm_flash_selection_extent::fresh(r.position):glm_flash_selection_extent::carried(r.selected_tail);
  if(r.fresh&&r.selected_tail!=extent.tail)throw std::runtime_error("Flash fresh selection tail differs from its absolute position");
  auto &f = r.fragment;
  const auto publish = [&](Value value, bool copy = false) { return expose(r, value, copy); };
  const auto changed = [&](Entry entry, Value value, bool copy = false) { return change(r, entry, value, copy); };
  Value qr;
  if (r.sparse) {
    if (!original_qr) throw std::runtime_error("original sparse query requires its same-Graph layer producer");
    qr = changed(kQueryState, codec::cast(f, *original_qr, c::DataType::kFloat32));
  } else {
  auto qa = glm::projection(f, nx, kQueryRank, kHidden, 197 + role_offset, false);
  r.query_scale = sem::oracle::scale(kQueryRank);
  auto query_scale = r.query_scale;
  qr = changed(kQueryState, glm::norm(f, qa.output, query_scale));
  r.query_reference = [qa = std::move(qa), scale = r.query_scale](auto x) {
    return sem::oracle::rms_norm_rows(qa.reference(x), scale, kQueryRank, glm::kNormEpsilon);
  };
  }
  Value query,latent;
  if(has_original_attention(r)) {
    const auto &roles=r.sparse->attention->roles;
    const auto bank=[&](original_attention::Parameter role) {return external(r,roles[static_cast<std::size_t>(role)]);};
    const auto dynamic=[&](Value activation,original_attention::Parameter role) {
      return f.one(c::model_ops::float8_block_linear(r.sparse->arithmetic.policy,r.sparse->arithmetic.absmax_floor),{activation,bank(role)});
    };
    const auto qb=dynamic(qr,original_attention::Parameter::kQueryProjection);
    r.attention_ports.qb_raw=publish(qb,true);
    const auto stored_qb=codec::cast(f,qb,c::DataType::kBFloat16);
    r.attention_ports.qb_stored=publish(stored_qb,true);
    const auto key_rhs=glm::permute(f,external(r,r.sparse->attention->prepared[0]),{0,2,1});
    const auto absorbed=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),
        {glm::reshape(f,stored_qb,{glm::kHeads,1,256}),key_rhs});
    r.attention_ports.absorbed_stored=publish(absorbed,true);
    query=glm::reshape(f,codec::cast(f,absorbed,c::DataType::kFloat32),{1,1,glm::kHeads,kLatent});
    r.query=publish(query,true);
    const auto kva=dynamic(nx,original_attention::Parameter::kKvProjection);
    r.attention_ports.kva_raw=publish(kva,true);
    const auto stored_kva=codec::cast(f,kva,c::DataType::kBFloat16);
    r.attention_ports.kva_stored=publish(stored_kva,true);
    const auto normalized=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-5F}),
        {codec::cast(f,stored_kva,c::DataType::kFloat32),codec::cast(f,bank(original_attention::Parameter::kKvNorm),c::DataType::kFloat32)});
    r.attention_ports.kvnorm_raw=publish(normalized,true);
    const auto stored_latent=codec::cast(f,normalized,c::DataType::kBFloat16);
    r.attention_ports.kvnorm_stored=publish(stored_latent,true);
    // This consumer retains an F32 state carrier, containing only exact BF16
    // widenings. It does not claim the vendor KV-cache physical/page ABI.
    latent=codec::cast(f,stored_latent,c::DataType::kFloat32);r.latent=publish(latent);
  } else {
  auto qb = glm::projection(f, qr, glm::kHeads * 256, kQueryRank, 199 + role_offset, false);
  const auto query_nope = glm::kimi::slice_heads(f, qb.output, glm::kHeads, 256, 0, 256);
  auto kb = glm::head_projection(f, query_nope, kLatent, 256, 223 + role_offset);
  query = glm::reshape(f, kb.output, {1, 1, glm::kHeads, kLatent});
  r.query = publish(query, true);
  r.absorbed_query_reference = [qb = std::move(qb), kb = std::move(kb)](auto x) {
    return kb.reference(qb.reference(x));
  };
  auto kva = glm::projection(f, nx, kLatent, kHidden, 211 + role_offset, false);
  auto latent_scale = sem::oracle::scale(kLatent);
  latent = glm::norm(f, kva.output, latent_scale); r.latent = publish(latent);
  r.latent_reference = [kva = std::move(kva), scale = latent_scale](auto x) {
    return sem::oracle::rms_norm_rows(kva.reference(x), scale, kLatent, glm::kNormEpsilon);
  };
  }
  const auto cache = changed(kCache, f.one(c::Operation(c::Scatter{}), {state[kCache], state[kMlaCursor], latent}));
  const auto cursor = changed(kMlaCursor, increment(f, state[kMlaCursor]));
  Value selected = state[kSelection];
  if (r.fresh) {
    const auto pool = append_history(r, state, nx, cursor, role_offset);
    std::optional<Value> pool_ids;
    if(extent.visible_pools>0) {
    Value iq, weights;
    if (r.sparse) {
      const auto &roles = r.sparse->indexer_roles;
      const auto wq = f.one(c::model_ops::dense_linear_f32(),
          {codec::cast(f, qr, c::DataType::kBFloat16), external(r, roles[original_index::kQuery])});
      r.index_query_accum = publish(wq);
      const auto stored = codec::cast(f, wq, c::DataType::kBFloat16); r.index_query_stored = publish(stored);
      iq = glm::reshape(f, codec::cast(f, stored, c::DataType::kFloat32), {kIndexHeads, kIndexDepth});
      r.index_query = publish(iq, true);
      weights = f.one(c::Operation(c::MatMul{}),
          {nx, glm::permute(f, external(r, roles[original_index::kWeights]), {1, 0})});
      r.score_weight = publish(weights);
    } else {
    auto wq = glm::projection(f, qr, kIndexHeads * kIndexDepth, kQueryRank, 239 + role_offset, false);
    iq = glm::reshape(f, wq.output, {kIndexHeads, kIndexDepth});
    r.index_query = publish(iq, true);
    r.index_query_reference = [wq = std::move(wq)](auto x) { return wq.reference(x); };
    auto ws = glm::linear::projection(f, nx, kIndexHeads, kHidden, glm::linear::Storage::kF32, 251 + role_offset);
    weights = ws.output; r.score_weight = publish(weights);
    r.score_weight_reference = [ws = std::move(ws)](auto x) { return ws.reference(x); };
    }
    const auto encoded_query = codec::quantize(f, codec::bfloat_boundary(f, iq), kIndexHeads);
    r.query_codec = expose_codec(r, encoded_query);
    const auto pool_count = extent.visible_pools;
    const auto scores = codec::score_encoded(f, pool.codes, pool.scales, encoded_query, weights, pool_count);
    r.scores = publish(scores, true);
    const auto top = f.append(c::Operation(c::TopK{.count = std::size_t(extent.selected_pools)}), {scores}, 2);
    std::vector<std::int32_t> expanded(std::size_t(pool_count * 4));
    std::iota(expanded.begin(), expanded.end(), 0);
    const auto table = f.constant(sem::dense({pool_count, 4}, c::DataType::kInt32), sem::bytes(expanded));
    pool_ids = glm::reshape(f, f.one(c::Operation(c::Gather{}), {table, top[1]}), {extent.pool_tokens()});
    }
    Value tail;
    if (r.selected_tail) {
      tail = f.one(c::Operation(c::Copy{}), {state[kTailStart]});
      for (std::int32_t i = 1; i < r.selected_tail; ++i)
        tail = f.one(c::Operation(c::Concat{.axis = 0}), {tail, increment(f, state[kTailStart], i)});
    }
    if (extent.padding()>0) {
      const auto padding = f.constant(sem::dense({extent.padding()}, c::DataType::kInt32),
          sem::bytes(std::vector<std::int32_t>(std::size_t(extent.padding()), -1)));
      tail = r.selected_tail ? f.one(c::Operation(c::Concat{.axis = 0}), {tail, padding}) : padding;
    }
    selected = pool_ids?f.one(c::Operation(c::Concat{.axis = 0}), {*pool_ids, tail}):tail;
    changed(kSelection, selected);
  } else {
    // No wq/wk/wg/pooling/TopK exists in this graph. The actual same-epoch IDs
    // from the StateVersion are the checked Gather's producer permission.
    selected = changed(kSelection, selected, true);
    changed(kJournal, f.one(c::Operation(c::Scatter{}), {journal->rows, journal->count, nx}));
    changed(kLagCount, increment(f, journal->count));
  }
  const auto selected_count = extent.live_tokens();
  const auto valid = glm::reshape(f, glm::core::rows(f, glm::reshape(f, selected, {kIds, 1}), 0, selected_count), {selected_count});
  const auto chosen = f.one(c::Operation(c::Gather{}), {cache, valid});
  const auto attention = f.one(c::Operation(c::ScaledDotProductAttention{.scale = 1.0F / 16.0F,
      .causal = true, .accumulation_type = c::DataType::kFloat32}),
      {query, glm::reshape(f, chosen, {1, selected_count, 1, kLatent}), glm::reshape(f, chosen, {1, selected_count, 1, kLatent})});
  r.attention = publish(attention);
  Value projected;
  if(has_original_attention(r)) {
    const auto stored_attention=codec::cast(f,attention,c::DataType::kBFloat16);
    r.attention_ports.attention_stored=publish(stored_attention,true);
    const auto value_rhs=glm::permute(f,external(r,r.sparse->attention->prepared[1]),{0,2,1});
    const auto value=f.one(c::Operation(c::MatMul{.accumulation_type=c::DataType::kFloat32}),
        {glm::reshape(f,stored_attention,{glm::kHeads,1,kLatent}),value_rhs});
    r.attention_ports.value_stored=publish(value,true);
    const auto raw=f.one(c::model_ops::float8_block_linear(r.sparse->arithmetic.policy,r.sparse->arithmetic.absmax_floor),
        {glm::reshape(f,codec::cast(f,value,c::DataType::kFloat32),{1,glm::kHeads*glm::kValue}),
         external(r,r.sparse->attention->roles[static_cast<std::size_t>(original_attention::Parameter::kOutput)])});
    r.attention_ports.output_raw=publish(raw,true);
    const auto stored=codec::cast(f,raw,c::DataType::kBFloat16);r.attention_ports.output_stored=publish(stored,true);
    r.sparse_output=stored;
    projected=codec::cast(f,stored,c::DataType::kFloat32);r.output=publish(projected);
  } else {
  auto vb = glm::head_projection(f, glm::reshape(f, attention, {1, glm::kHeads, kLatent}), glm::kValue, kLatent, 227 + role_offset);
  auto out = glm::projection(f, glm::reshape(f, vb.output, {1, glm::kHeads * glm::kValue}),
      kHidden, glm::kHeads * glm::kValue, 229 + role_offset, false);
  projected = out.output; r.output = publish(projected);
  r.output_reference = [vb = std::move(vb), out = std::move(out)](auto x) { return out.reference(vb.reference(x)); };
  }
  return projected;
}

namespace target43 = glm_mtp_target43_component;
constexpr int kTarget43RoleOffset = 1021;
Recipe build_gguf_target43(std::int32_t,std::shared_ptr<OriginalTarget43>);
Recipe build_mlx_target43(std::int32_t,std::shared_ptr<OriginalTarget43>);
Recipe build_target43(std::int32_t position,std::shared_ptr<OriginalTarget43> source={}) {
  if(source&&source->converted)return build_gguf_target43(position,std::move(source));
  if(source&&source->mlx)return build_mlx_target43(position,std::move(source));
  Recipe r; r.fresh = true; r.position = position; r.selected_tail = (position + 1) % 4;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f = r.fragment; f.profile = kModel.profile; f.region = 0x474c4d5441523433ULL; f.quantized = true;
  const auto residual = f.input(sem::dense({1, target43::kStreams, kHidden}));
  const auto desc = descriptions();
  std::array<Value, kSparseEntries.size()> actual;
  for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = f.input(desc[kSparseEntries[i]]);
  const SparseState13 state(f, actual);
  if(source) {
    r.target43_source=std::move(source);r.sparse=r.target43_source->sparse;r.moe=r.target43_source->moe;
    f.profile=r.sparse->profile;
    const auto parameter=[&](original_target43::Role role){return external(r,r.target43_source->parameters.at(role));};
    using Role=original_target43::Role;
    const target43_fragment::Parameters parameters{
        .attention={parameter(Role::kHcAttnFn),parameter(Role::kHcAttnScale),parameter(Role::kHcAttnBase),parameter(Role::kInputNorm)},
        .ffn={parameter(Role::kHcFfnFn),parameter(Role::kHcFfnScale),parameter(Role::kHcFfnBase),
            external(r,r.sparse->attention->roles[static_cast<std::size_t>(original_attention::Parameter::kPostNorm)])}};
    const auto prepared=target43_fragment::prepare(f,codec::cast(f,residual,c::DataType::kBFloat16),parameters);
    const auto nx=codec::cast(f,prepared.input,c::DataType::kFloat32);r.nx=expose(r,nx);
    auto &query=r.target43_query;query.input=prepared.input;
    query.raw=f.one(c::model_ops::float8_block_linear(r.sparse->arithmetic.policy,r.sparse->arithmetic.absmax_floor),
        {nx,parameter(Role::kQueryProjection)});
    query.stored=codec::cast(f,query.raw,c::DataType::kBFloat16);
    query.norm_input=codec::cast(f,query.stored,c::DataType::kFloat32);
    query.norm_raw=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-5F}),
        {query.norm_input,codec::cast(f,parameter(Role::kQueryNorm),c::DataType::kFloat32)});
    query.norm_stored=codec::cast(f,query.norm_raw,c::DataType::kBFloat16);
    query.norm_visible=codec::cast(f,query.norm_stored,c::DataType::kFloat32);
    r.sparse_hidden=expose(r,prepared.input,true);r.sparse_query=expose(r,query.norm_stored,true);
    (void)append_sparse_attention(r,state,nx,kTarget43RoleOffset,nullptr,&query.norm_stored);
    if(!r.sparse_output)throw std::runtime_error("original target43 needs the actual MLA O BF16 producer");
    r.original43=target43_fragment::finish(f,prepared,*r.sparse_output,moe_parameters(r),r.moe->arithmetic,r.moe->roles.source->weight_storage());
    r.target43_mhc=expose(r,r.original43->mhc,true);
    const auto observe=[&](Value &value) {
      const auto found=std::find_if(r.original43_ports.begin(),r.original43_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
      if(found==r.original43_ports.end())r.original43_ports.push_back({value,expose(r,value,true)});
    };
    r.original43->ports.visit_values(observe);query.visit_values(observe);
    for(auto value:{r.original43->mhc,r.original43->stored})observe(value);
    const auto input=std::find_if(r.original43_ports.begin(),r.original43_ports.end(),[&](const auto &entry){return entry.first.ordinal()==r.original43->ports.ffn_hc.norm.stored.ordinal();});
    if(input==r.original43_ports.end())throw std::runtime_error("original target43 actual MoE predecessor missing");
    r.moe_input=input->second;r.moe_output=expose(r,codec::cast(f,r.original43->moe.output,c::DataType::kFloat32),true);
    r.original_moe=r.original43->moe;
    r.original_moe->ports.visit_values([&](Value &value){r.moe_ports.push_back({value,expose(r,value,true)});});
    return r;
  }
  auto boundary = target43::prepare(f, residual);
  r.nx = expose(r, boundary.input);
  const auto projected = append_sparse_attention(r, state, boundary.input, kTarget43RoleOffset);
  auto tail = target43::finish(f, std::move(boundary), projected);
  r.target43_reference = std::move(tail.evaluate);
  for (std::size_t i = 0; i < tail.values.size(); ++i) {
    const auto found = std::find_if(f.outputs.begin(), f.outputs.end(),
        [&](Value value) { return value.ordinal() == tail.values[i].ordinal(); });
    r.target43_ports[i] = found == f.outputs.end() ? expose(r, tail.values[i])
        : std::size_t(found - f.outputs.begin());
  }
  r.target43_mhc=r.target43_ports[target43::Ports::mhc];
  return r;
}

mlx_model::State mlx_state(const std::array<Value,kResources> &s) {
  return {s[kCache],s[kRawKey],s[kRawGate],s[kPool],s[kHiddenState],s[kQueryState],s[kPending],s[kMlaCursor],s[kIndexCursor],
      s[kPoolCount],s[kRingOffset],s[kTailValid],s[kTailStart],s[kSelection],s[kJournal],s[kLagCount]};
}
void mlx_changes(Recipe &r,const mlx_model::State &s,bool catchup) {
  const auto store=[&](Entry entry,Value value){change(r,entry,value,true);r.mlx_ports.push_back({value,r.state_ports[entry]});};
  for(const auto &[entry,value]:std::array{std::pair{kRawKey,s.raw_key},std::pair{kRawGate,s.raw_gate},
      std::pair{kIndexCursor,s.index_cursor},std::pair{kRingOffset,s.ring_offset},std::pair{kTailValid,s.tail_valid}})
    if(r.fresh||catchup)store(entry,value);
  if((r.fresh||catchup)&&r.position%4==3){store(kPool,s.pool);store(kPoolCount,s.pool_count);store(kTailStart,s.tail_start);}
  if(!catchup){store(kCache,s.main);store(kMlaCursor,s.mla_cursor);store(kQueryState,s.query);store(kSelection,s.selection);store(kHiddenState,s.hidden);store(kPending,s.pending);}
  if(!r.fresh&&!catchup)store(kJournal,s.journal);
  if(catchup||!r.fresh)store(kLagCount,s.lag);
}
void mlx_expose(Recipe &r,Value &value) {
  const auto found=std::find_if(r.mlx_ports.begin(),r.mlx_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
  if(found==r.mlx_ports.end())r.mlx_ports.push_back({value,expose(r,value,true)});
}
mlx_model::Parameters mlx_parameters(Recipe &r,bool prefix=true) {
  if(!r.mlx)throw std::runtime_error("MLX cold builder requires its actual layer source");
  const auto lookup=[&](std::string_view name) {
    const auto found=std::find_if(r.external.begin(),r.external.end(),[&](const auto &entry){return entry.role.name==name;});
    return found!=r.external.end()?r.fragment.constants.at(found->constant):external(r,r.mlx->parameters.at(name));
  };
  const auto key=external(r,r.mlx->prepared[0]),value=external(r,r.mlx->prepared[1]);
  return prefix?mlx_model::parameters(r.fragment,lookup,key,value):mlx_model::block_parameters(r.fragment,lookup,r.mlx->parameters.source->prefix(),key,value);
}
Recipe build_mlx(bool fresh,std::int32_t position,std::int32_t carried_tail,std::shared_ptr<const OriginalNextn> original) {
  Recipe r;r.original=std::move(original);r.mlx=r.original->mlx;r.fresh=fresh;r.position=position;
  r.selected_tail=fresh?(position+1)%4:carried_tail;r.state_entries=state_resources(true);
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;f.profile=&mlx_runtime::kProfile;
  f.region=fresh?0x514d4c5846524553ULL:0x514d4c58534b4950ULL;f.quantized=true;
  const auto desc=descriptions(true);std::array<Value,kResources> state;
  for(const auto entry:r.state_entries)state[entry]=f.input(desc[entry]);
  r.mlx_parameters=mlx_parameters(r);const auto layout=mlx_runtime::layout();std::vector<Value> banks;
  for(const auto &d:layout.resident)banks.push_back(f.input(d));const auto map=f.input(layout.map_description());
  r.mlx_result=mlx_model::append(f,mlx_state(state),state[kHiddenState],*r.mlx_parameters,banks,map,fresh,position,carried_tail);
  auto &result=*r.mlx_result;r.mlx_boundary=result.moe.boundary;mlx_changes(r,result.state(),false);
  result.visit_values([&](Value &value){mlx_expose(r,value);});
  const auto port=[&](Value value){const auto found=std::find_if(r.mlx_ports.begin(),r.mlx_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==r.mlx_ports.end())throw std::runtime_error("MLX actual diagnostic missing");return found->second;};
  r.nx=port(result.prefix.input_norm.output);r.eh=port(result.prefix.eh.output);r.embedding=port(result.prefix.embedding_norm.input);
  r.query=port(result.prefix.query_norm.output);r.latent=port(result.mla.latent);r.attention=port(result.mla.attention);
  r.output=port(result.mla.output.output);r.residual=port(result.residual);r.ffn_input=port(result.post_norm.output);r.ffn_output=port(result.moe.output);
  r.lm_logits=port(result.logits);r.lm_top_values=port(result.top_values);r.lm_ids=r.state_ports[kPending];
  r.lm_combined=port(result.combined);r.lm_norm_raw=port(result.shared_norm.output);r.lm_pending_input=expose(r,state[kPending],true);
  return r;
}
Recipe build_mlx_catchup(bool direct,std::int32_t position,std::shared_ptr<const OriginalNextn> original,std::int32_t journal_slot=0) {
  Recipe r;r.original=std::move(original);r.mlx=r.original->mlx;r.catchup=true;r.direct_catchup=direct;r.position=position;
  r.journal_slot=journal_slot;r.state_entries=state_resources(true);r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;
  f.profile=&mlx_runtime::kProfile;f.region=0x514d4c5843415443ULL;f.quantized=true;
  const auto desc=descriptions(true);std::array<Value,kResources> values;for(const auto e:r.state_entries)values[e]=f.input(desc[e]);
  auto state=mlx_state(values);r.mlx_parameters=mlx_parameters(r);
  const auto nx=direct?f.input(sem::dense({1,kHidden})):f.one(c::Operation(c::Gather{}),{state.journal,f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{journal_slot}))});
  r.nx=expose(r,nx,true);r.mlx_history=mlx_model::append_history(f,state,nx,*r.mlx_parameters,position);
  state.lag=mlx_model::inc(f,state.lag,-1);mlx_changes(r,state,true);
  r.mlx_history->visit_values([&](Value &v){mlx_expose(r,v);});return r;
}

Recipe build_mlx_target43(std::int32_t position,std::shared_ptr<OriginalTarget43> source) {
  Recipe r;r.target43_source=std::move(source);r.mlx=r.target43_source->mlx;r.position=position;r.fresh=true;r.selected_tail=(position+1)%4;
  r.state_entries=sparse_entries(true);r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;
  f.profile=&mlx_runtime::kProfile;f.region=0x514d4c5854343341ULL;f.quantized=true;
  const auto residual=f.input(sem::dense({1,4,kHidden}));const auto desc=descriptions(true);std::array<Value,kResources> state;
  for(const auto entry:r.state_entries)state[entry]=f.input(desc[entry]);
  const auto lookup=[&](std::string_view name){const auto found=std::find_if(r.external.begin(),r.external.end(),[&](const auto &v){return v.role.name==name;});
    return found!=r.external.end()?f.constants[found->constant]:external(r,r.mlx->parameters.at(name));};
  const auto hc=mlx_target::parameters43(f,lookup,r.mlx->parameters.source->prefix());const auto prepared=mlx_target::prepare43(f,residual,hc);
  r.mlx_parameters=mlx_parameters(r,false);const auto &p=*r.mlx_parameters;
  r.mlx_target_query=p.project.linear(f,prepared.input,p.nextn.block.query_a);
  r.mlx_target_query_norm=mlx_model::fp::norm(f,r.mlx_target_query->output,p.nextn.block.query_norm,1.e-5F);
  r.mlx_target_mla=mlx_model::append_mla(f,mlx_state(state),prepared.input,r.mlx_target_query_norm->output,p,true,position,0);
  const auto layout=mlx_runtime::layout();std::vector<Value> banks;for(const auto &d:layout.resident)banks.push_back(f.input(d));const auto map=f.input(layout.map_description());
  r.mlx43=mlx_target::finish43(f,prepared,r.mlx_target_mla->output.output,[&](sem::Fragment &graph,Value actual){return mlx_model::append_moe(graph,actual,p,banks,map);});
  r.mlx_boundary=r.mlx43->moe.boundary;
  const auto &next=r.mlx_target_mla->state;
  const std::array<Value,kResources> values{next.main,next.raw_key,next.raw_gate,next.pool,Value{},next.hidden,next.query,next.pending,
      next.mla_cursor,next.index_cursor,next.pool_count,next.ring_offset,next.tail_valid,next.tail_start,next.selection,next.journal,next.lag};
  for(const auto entry:r.state_entries)if(values[entry].ordinal()!=state[entry].ordinal()) {change(r,entry,values[entry],true);r.mlx_ports.push_back({values[entry],r.state_ports[entry]});}
  r.nx=expose(r,prepared.input,true);r.mlx_ports.push_back({prepared.input,r.nx});r.target43_mhc=expose(r,r.mlx43->mhc,true);
  r.mlx43->ports.visit_values([&](Value &v){mlx_expose(r,v);});r.mlx43->moe.ports.visit_values([&](Value &v){mlx_expose(r,v);});
  r.mlx_target_query->visit_values([&](Value &v){mlx_expose(r,v);});r.mlx_target_query_norm->visit_values([&](Value &v){mlx_expose(r,v);});
  r.mlx_target_mla->visit_values([&](Value &v){mlx_expose(r,v);});return r;
}

// The representation and state-entry map are the same plain pooled state.
// Only cold source/leaf composition differs; Program and the K1 workflow below
// execute both artifacts without an additional controller or executor.
void converted_boundary(Recipe &r,const glm_gguf_mtp_fragment::Boundary &b) {
  r.mlx_boundary=mlx_model::Boundary{b.input,b.ids,b.weights,{b.banks.begin(),b.banks.end()},b.map};
}
gguf_flash::Parameters converted_parameters(Recipe &r) {
  auto connection=gguf_flash::connect(r.fragment,r.converted->parameters,43,44,45,
      {"token_embd.weight","output.weight","output_norm.weight"},r.converted->artifact->policy);
  for(const auto &entry:connection.constants) {
    const auto found=std::find_if(r.fragment.constants.begin(),r.fragment.constants.end(),[&](Value v){return v.ordinal()==entry.value.ordinal();});
    if(found==r.fragment.constants.end())throw std::runtime_error("converted constant absent from its original graph");
    r.external.push_back({std::size_t(found-r.fragment.constants.begin()),
        {entry.role.name,entry.role.description,entry.role.owner,entry.role.lease}});
  }
  return std::move(connection.parameters);
}
std::pair<std::array<Value,3>,Value> converted_banks(Recipe &r) {
  const auto &layout=r.converted->cpu_supply->layout();std::array<Value,3> banks;
  for(std::size_t i=0;i<3;++i)banks[i]=r.fragment.input(layout.resident(i));
  return {banks,r.fragment.input(layout.map_description())};
}
Recipe build_converted(bool fresh,std::int32_t position,std::int32_t carried_tail,std::shared_ptr<const OriginalNextn> original) {
  Recipe r;r.original=std::move(original);r.converted=r.original->converted;r.profile_owner=r.original;r.fresh=fresh;r.position=position;
  r.selected_tail=fresh?(position+1)%4:carried_tail;r.state_entries=state_resources(true);r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f=r.fragment;f.profile=r.converted->profile;f.region=fresh?0x4747464c46524553ULL:0x4747464c534b4950ULL;f.quantized=true;
  const auto desc=descriptions(true);std::array<Value,kResources> state;for(const auto e:r.state_entries)state[e]=f.input(desc[e]);
  r.pooled_before=mlx_state(state);r.converted_parameters=converted_parameters(r);const auto [banks,map]=converted_banks(r);
  r.converted_nextn=gguf_flash::append_nextn(f,*r.pooled_before,state[kHiddenState],*r.converted_parameters,banks,map,fresh,position,carried_tail);
  auto &v=*r.converted_nextn;r.pooled_after=v.state();converted_boundary(r,v.moe.boundary);mlx_changes(r,v.state(),false);
  v.visit_values([&](Value &value){mlx_expose(r,value);});
  const auto port=[&](Value value){const auto it=std::find_if(r.mlx_ports.begin(),r.mlx_ports.end(),[&](const auto &x){return x.first.ordinal()==value.ordinal();});
    if(it==r.mlx_ports.end())throw std::runtime_error("converted actual diagnostic missing");return it->second;};
  r.nx=port(v.prefix.input_norm.output);r.eh=port(v.prefix.eh.output);r.embedding=port(v.prefix.embedding_norm.input);
  r.query=port(v.prefix.query_norm.output);r.latent=port(v.mla.latent);r.attention=port(v.mla.attention);r.output=port(v.mla.output.output);
  r.residual=port(v.residual);r.ffn_input=port(v.post_norm.output);r.ffn_output=port(v.moe.output);
  r.lm_logits=port(v.logits);r.lm_top_values=port(v.top_values);r.lm_ids=r.state_ports[kPending];r.lm_combined=port(v.combined);
  r.lm_norm_raw=port(v.shared_norm.output);r.lm_pending_input=expose(r,state[kPending],true);return r;
}
Recipe build_converted_catchup(bool direct,std::int32_t position,std::shared_ptr<const OriginalNextn> original,std::int32_t journal_slot=0) {
  Recipe r;r.original=std::move(original);r.converted=r.original->converted;r.profile_owner=r.original;r.catchup=true;r.direct_catchup=direct;r.position=position;
  r.journal_slot=journal_slot;r.state_entries=state_resources(true);r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;
  f.profile=r.converted->profile;f.region=0x4747464c43415443ULL;f.quantized=true;
  const auto desc=descriptions(true);std::array<Value,kResources> values;for(const auto e:r.state_entries)values[e]=f.input(desc[e]);
  auto state=mlx_state(values);r.pooled_before=state;r.converted_parameters=converted_parameters(r);
  const auto nx=direct?f.input(sem::dense({1,kHidden})):f.one(c::Operation(c::Gather{}),{state.journal,f.constant(sem::dense({1},c::DataType::kInt32),sem::bytes(std::vector<std::int32_t>{journal_slot}))});
  r.nx=expose(r,nx,true);r.mlx_history=gguf_flash::append_history(f,state,nx,*r.converted_parameters,position);
  state.lag=mlx_model::inc(f,state.lag,-1);r.pooled_after=state;mlx_changes(r,state,true);r.mlx_history->visit_values([&](Value &v){mlx_expose(r,v);});return r;
}
Recipe build_gguf_target43(std::int32_t position,std::shared_ptr<OriginalTarget43> source) {
  Recipe r;r.target43_source=std::move(source);r.converted=r.target43_source->converted;r.profile_owner=r.target43_source;r.position=position;r.fresh=true;r.selected_tail=(position+1)%4;
  r.state_entries=sparse_entries(true);r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;
  f.profile=r.converted->profile;f.region=0x4747464c54343341ULL;f.quantized=true;
  const auto input=f.input(sem::dense({1,4,kHidden}));const auto desc=descriptions(true);std::array<Value,kResources> state;
  for(const auto e:r.state_entries)state[e]=f.input(desc[e]);r.pooled_before=mlx_state(state);r.converted_parameters=converted_parameters(r);
  const auto [banks,map]=converted_banks(r);r.converted43=gguf_flash::append_target43(f,*r.pooled_before,input,*r.converted_parameters,banks,map,position);
  auto &v=*r.converted43;const auto &next=v.mla.state;r.pooled_after=next;converted_boundary(r,v.moe.boundary);
  const std::array<Value,kResources> values{next.main,next.raw_key,next.raw_gate,next.pool,Value{},next.hidden,next.query,next.pending,
      next.mla_cursor,next.index_cursor,next.pool_count,next.ring_offset,next.tail_valid,next.tail_start,next.selection,next.journal,next.lag};
  for(const auto e:r.state_entries)if(values[e].ordinal()!=state[e].ordinal()){change(r,e,values[e],true);r.mlx_ports.push_back({values[e],r.state_ports[e]});}
  r.nx=expose(r,v.ports.attention_hc.norm.stored,true);r.mlx_ports.push_back({v.ports.attention_hc.norm.stored,r.nx});
  r.target43_mhc=expose(r,v.mhc,true);r.mlx_ports.push_back({v.mhc,r.target43_mhc});v.visit_values([&](Value &value){mlx_expose(r,value);});return r;
}
Recipe build(bool fresh, std::int32_t position = kBasePosition, std::int32_t carried_tail = 0,
             std::shared_ptr<const OriginalNextn> original = {}) {
  if(original&&original->converted)return build_converted(fresh,position,carried_tail,std::move(original));
  if(original&&original->mlx)return build_mlx(fresh,position,carried_tail,std::move(original));
  Recipe r;
  r.fresh = fresh;
  r.original = std::move(original);
  if(r.original)r.sparse=r.original->sparse;
  if(r.original)r.moe=r.original->moe;
  r.position = position;
  r.selected_tail = fresh ? (position + 1) % 4 : carried_tail;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f = r.fragment;
  f.profile = r.original ? &r.original->profile_description : kModel.profile;
  f.region = fresh ? 0x474c4d5354465245ULL : 0x474c4d5354534b50ULL;
  f.quantized = true;
  const auto desc = descriptions();
  std::array<Value, kResources> state;
  for (std::size_t i = 0; i < state.size(); ++i) state[i] = f.input(desc[i]);
  std::optional<Value> next_pending;
  if(!has_shared_lm(r))next_pending=f.input(sem::dense({1}, c::DataType::kInt32));
  const auto publish = [&](Value value, bool copy = false) {
    return expose(r, value, copy);
  };
  const auto changed = [&](Entry entry, Value value, bool copy = false) {
    return change(r, entry, value, copy);
  };
  Value embedding;
  std::optional<Value> supplied_embedding;
  if(has_shared_lm(r)) {
    const auto table=external(r,r.original->vocabulary->at(shared_lm::Role::kEmbedding));
    supplied_embedding=f.one(c::Operation(c::Gather{}),{table,state[kPending]});
    embedding=codec::cast(f,*supplied_embedding,c::DataType::kFloat32);
    r.lm_pending_input=publish(state[kPending],true);
  } else {
    std::vector<float> embeddings;
    for (std::int32_t token = 0; token < 8; ++token) {
      const auto values = embedding_values(token); embeddings.insert(embeddings.end(), values.begin(), values.end());
    }
    const auto token_bank = f.constant(sem::dense({8, kHidden}), sem::bytes(embeddings));
    embedding=f.one(c::Operation(c::Gather{}),{token_bank,state[kPending]});
  }
  r.embedding=publish(embedding);
  Value eh, nx, qr;
  if (r.original) {
    const auto hidden_bf16 = codec::cast(f, state[kHiddenState], c::DataType::kBFloat16);
    const auto embedding_bf16 = supplied_embedding?*supplied_embedding:codec::cast(f, embedding, c::DataType::kBFloat16);
    r.original_hidden = publish(hidden_bf16); r.original_embedding = publish(embedding_bf16,has_shared_lm(r));
    const auto prefix = original_prefix::append_with_constants(f, hidden_bf16, embedding_bf16,
        [&](original_prefix::Parameter parameter) { return external(r, r.original->prefix_roles[static_cast<std::size_t>(parameter)]); },r.original->arithmetic);
    const auto ports = prefix.values();
    for (std::size_t i = 0; i < ports.size(); ++i) r.original_prefix_ports[i] = publish(ports[i], true);
    r.sparse_hidden=r.original_prefix_ports[4];r.sparse_query=r.original_prefix_ports[6];
    eh = codec::cast(f, prefix.eh, c::DataType::kFloat32);
    nx = codec::cast(f, prefix.nx, c::DataType::kFloat32);
    qr = prefix.qr;
  } else {
    const auto input = glm::attach(f, glm::mtp_input(kModel, &r.input_reference, false),
        {state[kHiddenState], embedding});
    eh = input[3];
    r.input_scale = sem::oracle::scale(kHidden);
    auto input_scale = r.input_scale;
    nx = glm::norm(f, eh, input_scale);
  }
  r.eh = publish(eh); r.nx = publish(nx);
  if(next_pending)changed(kPending,*next_pending,true);
  const SkipJournal journal{state[kJournal], state[kLagCount]};
  const auto attention_output = append_sparse_attention(r, sparse_state(f, state), nx, 0, &journal,
      r.original ? &qr : nullptr);
  const auto residual_sum=glm::binary(f,c::BinaryKind::kAdd,eh,attention_output);
  Value residual=residual_sum;
  if(has_original_attention(r)) {
    r.attention_ports.residual_sum=publish(residual_sum,true);
    const auto stored=codec::cast(f,residual_sum,c::DataType::kBFloat16);
    r.attention_ports.residual_stored=publish(stored,true);
    residual=codec::cast(f,stored,c::DataType::kFloat32);
  }
  r.residual=publish(residual);
  Value ffn_input;std::optional<Value> actual_ffn_input;
  if(has_original_attention(r)) {
    const auto raw=f.one(c::Operation(c::RmsNorm{.epsilon=1.0e-5F}),{residual_sum,codec::cast(f,
        external(r,r.sparse->attention->roles[static_cast<std::size_t>(original_attention::Parameter::kPostNorm)]),c::DataType::kFloat32)});
    r.attention_postnorm=true;
    r.attention_ports.postnorm_raw=publish(raw,true);
    const auto stored=codec::cast(f,raw,c::DataType::kBFloat16);r.attention_ports.postnorm_stored=publish(stored,true);actual_ffn_input=stored;
    ffn_input=codec::cast(f,stored,c::DataType::kFloat32);r.ffn_input=publish(ffn_input);
  } else {
  r.ffn_scale = sem::oracle::scale(kHidden); auto ffn_scale = r.ffn_scale;
  ffn_input = glm::norm(f, residual, ffn_scale); r.ffn_input = publish(ffn_input);
  }
  Value ffn_result;
  if(has_original_moe(r)) {
    if(!actual_ffn_input)throw std::runtime_error("original MoE requires its actual original BF16 postnorm output");
    const auto parameters=moe_parameters(r);
    r.original_moe=moe_fragment::append(f,*actual_ffn_input,parameters,original_moe::kFlashGeometry,r.moe->arithmetic,r.moe->roles.source->weight_storage());
    r.original_moe->ports.visit_values([&](Value &value) {r.moe_ports.push_back({value,publish(value,true)});});
    ffn_result=codec::cast(f,r.original_moe->output,c::DataType::kFloat32);r.ffn_output=publish(ffn_result);
    r.moe_input=r.attention_ports.postnorm_stored;r.moe_output=r.ffn_output;
  } else {
    auto moe=glm::moe::moe_chain(glm::moe::cases::kCases[11],false,false,&r.moe_reference,0.5F);
    const auto ff=glm::attach(f,std::move(moe),{ffn_input});ffn_result=glm::reshape(f,ff[5],{1,kHidden});
    r.ffn_output=publish(ff[5]);
  }
  const auto combined=glm::binary(f,c::BinaryKind::kAdd,residual,ffn_result);
  if(has_shared_lm(r)) {
    const auto &vocabulary=*r.original->vocabulary;
    const auto norm_scale=codec::cast(f,external(r,vocabulary.at(shared_lm::Role::kNorm)),c::DataType::kFloat32);
    const auto normalized=f.one(c::Operation(c::RmsNorm{.epsilon=vocabulary.source->geometry.epsilon}),{combined,norm_scale});
    const auto stored=codec::cast(f,normalized,c::DataType::kBFloat16);
    r.lm_combined=publish(combined,true);r.lm_norm_raw=publish(normalized,true);r.lm_norm_stored=publish(stored,true);
    changed(kHiddenState,codec::cast(f,stored,c::DataType::kFloat32));
    const auto logits=f.one(c::model_ops::dense_linear_f32(),{stored,external(r,vocabulary.at(shared_lm::Role::kHead))});
    r.lm_logits=publish(logits,true);
    const auto top=f.append(c::Operation(c::TopK{.count=1}),{logits},2);
    r.lm_top_values=publish(top[0],true);
    r.lm_ids=expose(r,glm::reshape(f,top[1],{1}),true);
    r.state_ports[kPending]=r.lm_ids;
  } else {
    r.head_scale = sem::oracle::scale(kHidden); std::reverse(r.head_scale.begin(), r.head_scale.end());
    auto head_scale = r.head_scale;
    changed(kHiddenState, glm::norm(f, combined, head_scale));
  }
  return r;
}

Recipe build_catchup(bool direct_reference = false, std::int32_t position = 2055,
                     std::shared_ptr<const OriginalNextn> original = {},std::int32_t journal_slot=0) {
  if(journal_slot<0||journal_slot>=kJournalRows)throw std::runtime_error("GLM catchup journal slot outside the finite state schema");
  if(original&&original->converted)return build_converted_catchup(direct_reference,position,std::move(original),journal_slot);
  if(original&&original->mlx)return build_mlx_catchup(direct_reference,position,std::move(original),journal_slot);
  Recipe r;
  r.catchup = true; r.direct_catchup = direct_reference; r.position = position;r.journal_slot=journal_slot;
  r.original = std::move(original);
  if(r.original)r.sparse=r.original->sparse;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f = r.fragment;
  f.profile = r.original ? &r.original->profile_description : kModel.profile;
  f.region = direct_reference ? 0x474c4d4341544449ULL : 0x474c4d4341544a4fULL;
  f.quantized = true;
  const auto desc = descriptions();
  std::array<Value, kResources> state;
  for (std::size_t i = 0; i < state.size(); ++i) state[i] = f.input(desc[i]);
  Value nx;
  if (direct_reference) {
    // A separately executed accepted token supplies this actual producer
    // lease directly, independently of the journal gather in the real path.
    nx = f.input(sem::dense({1, kHidden}));
    r.nx = expose(r, nx, true);
  } else {
    const auto first = f.constant(sem::dense({1}, c::DataType::kInt32), sem::bytes(std::vector<std::int32_t>{journal_slot}));
    nx = f.one(c::Operation(c::Gather{}), {state[kJournal], first});
    r.nx = expose(r, nx);
  }
  (void)append_history(r, sparse_state(f, state), nx, increment(f, state[kIndexCursor]));
  change(r, kLagCount, increment(f, state[kLagCount], -1));
  // Catchup publishes no seed and executes no MLA, NextN, or sampling work.
  // Old I32 IDs retain old epoch provenance until a real fresh query runs.
  for (const auto &node : f.nodes)
    if (node.operation.get_if<c::TopK>() || node.operation.get_if<c::ScaledDotProductAttention>())
      throw std::runtime_error("catchup recipe accidentally executes selection or MLA");
  return r;
}

// The old finite starting boundary contains513 generated completed pools.
// Encode them once through the actual public Graph; do not import host-oracle
// codes/scales as if a real quantizer had produced this state.
Recipe initial_pool_recipe(bool plain=false) {
  if(plain){Recipe r;r.state_ports.fill(std::numeric_limits<std::size_t>::max());r.fragment.profile=&mlx_runtime::kProfile;r.fragment.region=0x514d4c58494e4954ULL;
    const auto value=r.fragment.input(sem::dense({kPoolCapacity,kIndexDepth}));r.fragment.outputs={r.fragment.one(c::Operation(c::Copy{}),{value})};return r;}
  Recipe r; r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f = r.fragment; f.profile = kModel.profile; f.region = 0x474c4d494e495451ULL;
  const auto pools = kBasePosition / 4;
  const auto source = f.input(sem::dense({pools, kIndexDepth}));
  const auto encoded = codec::quantize(f, codec::bfloat_boundary(f, source), pools);
  const auto old_codes = f.constant(sem::dense({kPoolCapacity, kIndexDepth}, c::DataType::kFloat8E4M3FN),
      sem::bytes(std::vector<std::uint8_t>(std::size_t(kPoolCapacity * kIndexDepth), 0x7f)));
  const auto old_scales = f.constant(sem::dense({kPoolCapacity, 1}),
      sem::bytes(std::vector<float>(std::size_t(kPoolCapacity), std::numeric_limits<float>::quiet_NaN())));
  std::vector<std::int32_t> rows(std::size_t(pools), 0); std::iota(rows.begin(), rows.end(), 0);
  const auto indices = f.constant(sem::dense({pools}, c::DataType::kInt32), sem::bytes(rows));
  f.outputs = {f.one(c::Operation(c::Scatter{}), {old_codes, indices, encoded.codes}),
      f.one(c::Operation(c::Scatter{}), {old_scales, indices, encoded.scale})};
  r.pool_codec = expose_codec(r, encoded);
  return r;
}

struct Program;
struct ResidentBoundary final {Value input,ids,weights;std::vector<Value> banks;Value map;};
struct ResidentBatch final {
  std::vector<c::ReadLease> banks;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  std::optional<c::cuda::WeightBatch> cuda;
#endif
  std::function<void()> record;
};
struct ResidentProgram final {
  struct PhaseBindings final {
    std::vector<Value> banks;
    std::optional<model_graph_run::InputPlan> external_inputs;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    std::vector<g::WeightBinding> weights;
#endif
  };
  resident_fragment::ResidentDraft<ResidentBoundary,1> plan;
  std::vector<std::uint64_t> bank_ids;
  std::function<ResidentBatch(std::span<const std::int32_t>,c::Queue&)> upload;
  std::function<void()> completed;
  std::vector<std::unique_ptr<Program>> phases;
  std::vector<PhaseBindings> bindings;
};
// Cold diagnostic provenance follows the actual compiled phase. These small
// located IDs never participate in execution or substitute descriptor equality
// for the original constant/input identity across a cut.
struct ReportValue final {
  const sem::GraphCase *logical=nullptr;const g::CompiledGraph *compiled=nullptr;g::ValueId value;
  friend bool operator==(const ReportValue &a,const ReportValue &b){return a.logical==b.logical&&a.value==b.value;}
};
struct ReportNode final {const sem::GraphCase *logical=nullptr;const g::CompiledGraph *compiled=nullptr;g::NodeId value;};
struct ReportGraph final {
  hyle::Result<ReportNode> producer(ReportValue value) const {
    const auto node=value.logical->graph.producer(value.value);if(!node)return std::unexpected(node.error());
    return ReportNode{value.logical,value.compiled,*node};
  }
  hyle::Result<ReportValue> node_input(ReportNode node,std::size_t index) const {
    const auto value=node.logical->graph.node_input(node.value,index);if(!value)return std::unexpected(value.error());
    return ReportValue{node.logical,node.compiled,*value};
  }
  hyle::Result<ReportValue> result(ReportNode node,std::size_t index) const {
    const auto value=node.logical->graph.result(node.value,index);if(!value)return std::unexpected(value.error());
    return ReportValue{node.logical,node.compiled,*value};
  }
  auto operation(ReportNode node) const {return node.logical->graph.operation(node.value);}
  auto operation_kind(ReportNode node) const {return node.logical->graph.operation_kind(node.value);}
  auto value_description(ReportValue value) const {return value.logical->graph.value_description(value.value);}
};
struct Program final {
  Recipe recipe;
  sem::GraphCase logical;
  std::vector<c::ReadLease> parameters;
  std::vector<std::size_t> parameter_slots;
  std::unique_ptr<model_graph_run::ColdProgram> cold_program;
  g::Invocation invocation;
  c::LeasePool outputs;
  std::shared_ptr<ResidentProgram> resident;
  std::size_t completed_submissions=0;
  ReportValue published(std::size_t port) const {
    if(!resident)return {&logical,&cold_program->compiled,logical.outputs.at(port)};
    const auto location=resident->plan.full_outputs.at(port);const auto &phase=*resident->phases.at(location.phase);
    ReportValue result{&phase.logical,&phase.cold_program->compiled,phase.logical.outputs.at(location.port)};
    const auto original=recipe.fragment.outputs.at(port);
    const auto source=std::find_if(recipe.fragment.nodes.begin(),recipe.fragment.nodes.end(),[&](const auto &node){
      return std::any_of(node.outputs.begin(),node.outputs.end(),[&](Value value){return value.ordinal()==original.ordinal();});});
    // split() materializes an extra Copy only when the semantic output was
    // not already a Copy. Return the same original operation value for reports.
    if(source==recipe.fragment.nodes.end()||!source->operation.get_if<c::Copy>()) {
      const ReportGraph graph;const auto copy=take(graph.producer(result),"resident semantic output materializer");
      if(take(graph.operation_kind(copy),"resident materializer operation")!=c::OperationKind::kCopy)
        throw std::runtime_error("resident non-Copy output must be materialized by its ordinary phase");
      result=take(graph.node_input(copy,0),"actual original semantic output");
    }
    return result;
  }
  ReportValue constant(std::size_t slot,ReportValue consumer) const {
    if(!resident) {
      if(consumer.logical!=&logical)throw std::runtime_error("constant consumer belongs to another Graph");
      return {&logical,&cold_program->compiled,logical.constants.at(slot)};
    }
    const auto original=recipe.fragment.constants.at(slot);
    for(std::size_t index=0;index<resident->phases.size();++index) {
      const auto &phase=*resident->phases[index];if(consumer.logical!=&phase.logical)continue;
      const auto &origins=resident->plan.phases[index].original_constants;
      const auto found=std::find_if(origins.begin(),origins.end(),[&](Value value){return value.ordinal()==original.ordinal();});
      if(found==origins.end())throw std::runtime_error("actual consumer phase lost its original readonly constant");
      return {&phase.logical,&phase.cold_program->compiled,phase.logical.constants.at(std::size_t(found-origins.begin()))};
    }
    throw std::runtime_error("constant consumer is not an actual compiled phase");
  }
  ReportValue original_input(std::size_t slot,ReportValue consumer) const {
    if(!resident)return {&logical,&cold_program->compiled,logical.inputs.at(slot)};
    const auto original=recipe.fragment.inputs.at(slot);
    for(std::size_t index=0;index<resident->phases.size();++index) {
      const auto &phase=*resident->phases[index];if(consumer.logical!=&phase.logical)continue;
      const auto &origins=resident->plan.phases[index].original_inputs;
      const auto found=std::find_if(origins.begin(),origins.end(),[&](Value value){return value.ordinal()==original.ordinal();});
      if(found==origins.end())throw std::runtime_error("actual consumer phase lost its original input");
      return {&phase.logical,&phase.cold_program->compiled,phase.logical.inputs.at(std::size_t(found-origins.begin()))};
    }
    throw std::runtime_error("input consumer is not an actual compiled phase");
  }
  ReportValue phase_input(Value original,ReportValue consumer) const {
    if(!resident)throw std::runtime_error("phase provenance requires actual resident graphs");
    for(std::size_t index=0;index<resident->phases.size();++index) {
      const auto &phase=*resident->phases[index];if(consumer.logical!=&phase.logical)continue;
      const auto &origins=resident->plan.phases[index].original_inputs;
      const auto found=std::find_if(origins.begin(),origins.end(),[&](Value value){return value.ordinal()==original.ordinal();});
      if(found==origins.end())throw std::runtime_error("actual phase lost the direct original producer input");
      return {&phase.logical,&phase.cold_program->compiled,phase.logical.inputs.at(std::size_t(found-origins.begin()))};
    }
    throw std::runtime_error("producer consumer is not an actual compiled phase");
  }
  c::LeasePoolStats output_stats() const {
    if(!resident)return outputs.stats();
    c::LeasePoolStats result;
    for(const auto &phase:resident->phases) {
      const auto stats=phase->output_stats();result.capacity_bytes+=stats.capacity_bytes;result.live_blocks+=stats.live_blocks;result.active_uses+=stats.active_uses;
    }
    return result;
  }
  std::size_t native_submissions() const {
    std::size_t result=completed_submissions;if(resident)for(const auto &phase:resident->phases)result+=phase->native_submissions();return result;
  }
  void report_shared_lm(const c::StorageOwner &owner) const {
    if(!has_shared_lm(recipe)||recipe.catchup||recipe.mlx||recipe.converted)return;
    const ReportGraph graph;const auto &vocabulary=*recipe.original->vocabulary;
    const auto actual=[&](std::size_t port) {
      const auto copy=take(graph.producer(published(port)),"shared LM actual observation producer");
      if(take(graph.operation_kind(copy),"shared LM observation kind")!=c::OperationKind::kCopy)
        throw std::runtime_error("shared LM observations require independent Copy outputs");
      return take(graph.node_input(copy,0),"shared LM observed actual value");
    };
    const auto bank=[&](shared_lm::Role role,ReportValue consumer) {
      const auto &expected=vocabulary.at(role);
      const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const External &entry){return entry.role.name==expected.name;});
      if(found==recipe.external.end()||found->role.description!=expected.description||found->role.owner.ref()!=owner.ref())
        throw std::runtime_error("shared LM must retain its complete original BF16 readonly bank");
      const auto provided=take(found->role.lease.view(),"actual shared LM readonly permission");
      const auto original=take(expected.lease.view(),"one shared LM ready bank");
      if(provided.resource()!=original.resource()||provided.native_handle()!=original.native_handle()||
          provided.offset()!=original.offset()||provided.size()!=original.size()||found->role.lease.generation()!=expected.lease.generation())
        throw std::runtime_error("shared LM Recipe copied/replaced the original ready bank instead of sharing it");
      const auto actual_bank=constant(found->constant,consumer);
      if(take(graph.value_description(actual_bank),"actual shared LM constant geometry")!=shared_lm::description(vocabulary.source->geometry,role))
        throw std::runtime_error("actual shared LM Graph constant must preserve full original BF16 geometry");
      return actual_bank;
    };
    const auto embedded=actual(recipe.original_embedding);
    const auto gather=take(graph.producer(embedded),"full-vocabulary embedding Gather");
    if(!take(graph.operation(gather),"actual shared embedding operation").get_if<c::Gather>()||
        take(graph.node_input(gather,0),"actual shared embedding table")!=bank(shared_lm::Role::kEmbedding,embedded)||
        take(graph.node_input(gather,1),"actual shared embedding IDs")!=original_input(kPending,embedded)||
        take(graph.value_description(embedded),"actual shared embedding type")!=sem::dense({1,kHidden},c::DataType::kBFloat16))
      throw std::runtime_error("original full-V BF16 Gather must consume the actual StateVersion pending input");
    const auto raw=actual(recipe.lm_logits);const auto projection=take(graph.producer(raw),"actual full-V DenseLinearF32");
    const auto stored=actual(recipe.lm_norm_stored);
    if(!take(graph.operation(projection),"actual shared LM operation").custom_as<c::model_ops::DenseLinearF32>()||
        take(graph.node_input(projection,0),"shared LM actual BF16 input")!=stored||
        take(graph.node_input(projection,1),"shared LM original BF16 head")!=bank(shared_lm::Role::kHead,raw)||
        take(graph.value_description(stored),"actual LM input description")!=sem::dense({1,kHidden},c::DataType::kBFloat16)||
        take(graph.value_description(raw),"actual LM full-V result")!=sem::dense({1,vocabulary.source->geometry.vocabulary}))
      throw std::runtime_error("shared LM must use actual stored-BF16 hidden and the same full original [V,H] BF16 bank");
    const auto norm_raw=actual(recipe.lm_norm_raw);const auto norm_node=take(graph.producer(norm_raw),"actual original shared RMS");
    const auto norm_operation=take(graph.operation(norm_node),"actual original shared RMS operation");
    const auto *norm=norm_operation.get_if<c::RmsNorm>();
    const auto gamma=take(graph.node_input(norm_node,1),"actual original shared gamma widen");
    const auto gamma_node=take(graph.producer(gamma),"actual shared gamma conversion");
    const auto gamma_operation=take(graph.operation(gamma_node),"actual shared gamma Cast");
    const auto *gamma_cast=gamma_operation.get_if<c::Cast>();
    const auto store_node=take(graph.producer(stored),"actual shared norm BF16 store");
    const auto store_operation=take(graph.operation(store_node),"actual shared norm storage operation");
    const auto *store_cast=store_operation.get_if<c::Cast>();
    if(!norm||norm->epsilon!=vocabulary.source->geometry.epsilon||take(graph.node_input(norm_node,0),"actual shared norm combined input")!=actual(recipe.lm_combined)||
        !gamma_cast||gamma_cast->output_type!=c::DataType::kFloat32||take(graph.node_input(gamma_node,0),"original shared gamma constant")!=bank(shared_lm::Role::kNorm,norm_raw)||
        !store_cast||store_cast->output_type!=c::DataType::kBFloat16||take(graph.node_input(store_node,0),"actual shared norm store predecessor")!=norm_raw)
      throw std::runtime_error("shared head must normalize actual combined F32 with original BF16 gamma widened exactly, then store BF16");
    const auto pending=actual(recipe.lm_ids);const auto reshape=take(graph.producer(pending),"shared LM pending reshape");
    if(!take(graph.operation(reshape),"shared pending reshape operation").get_if<c::View>())
      throw std::runtime_error("shared LM pending must reshape the actual I32 greedy result");
    const auto greedy=take(graph.node_input(reshape,0),"actual shared greedy I32 IDs");
    const auto top=take(graph.producer(greedy),"actual shared greedy TopK");
    const auto operation=take(graph.operation(top),"actual shared TopK operation");
    const auto *topk=operation.get_if<c::TopK>();
    if(!topk||topk->count!=1||take(graph.node_input(top,0),"shared greedy logits")!=raw||
        take(graph.result(top,1),"actual greedy indices")!=greedy||
        take(graph.result(top,0),"actual greedy scores")!=actual(recipe.lm_top_values)||recipe.state_ports[kPending]!=recipe.lm_ids)
      throw std::runtime_error("shared model pending must publish its actual full-V stable TopK1 result");
    const auto hidden=published(recipe.state_ports[kHiddenState]);
    const auto widen=take(graph.producer(hidden),"shared normalized hidden state producer");
    const auto widened_operation=take(graph.operation(widen),"shared normalized state operation");
    const auto *cast=widened_operation.get_if<c::Cast>();
    if(!cast||cast->output_type!=c::DataType::kFloat32||take(graph.node_input(widen,0),"actual shared normalized state input")!=stored)
      throw std::runtime_error("next shared-LM hidden state must exactly widen its actual BF16 sharednorm return");
    std::size_t implementations=0;std::string provider;
    for(std::size_t step=0;step<projection.compiled->report.step_count();++step) {
      const auto selected=take(projection.compiled->report.step(step),"shared LM actual selected implementation");
      for(std::size_t node=0;node<selected.node_count;++node)
        if(take(projection.compiled->report.step_node(step,node),"shared LM selected Graph node")==projection.value){++implementations;provider=selected.provider.name;}
    }
    if(implementations!=1||(recipe.cuda&&provider!="cublasLt"))
      throw std::runtime_error("full-V BF16 shared head requires one actual implementation; CUDA must select cuBLASLt without a full F32 head shadow");
    ::testing::Test::RecordProperty("shared_lm_head_provider",provider);
    std::printf("GLM original shared LM region=%llu fresh=%d tail=%d provider=%s V=%lld H=%lld; same readonly BF16 bank owner\n",
        static_cast<unsigned long long>(recipe.fragment.region),int(recipe.fresh),recipe.selected_tail,provider.c_str(),
        static_cast<long long>(vocabulary.source->geometry.vocabulary),static_cast<long long>(vocabulary.source->geometry.hidden));
  }
  void report_original_attention(const c::StorageOwner &owner) const {
    if(!has_original_attention(recipe)||recipe.catchup)return;
    const ReportGraph graph;const auto &source=*recipe.sparse->attention;
    const auto actual=[&](std::size_t port) {
      const auto copy=take(graph.producer(published(port)),"original attention observation producer");
      if(take(graph.operation_kind(copy),"original attention observation kind")!=c::OperationKind::kCopy)
        throw std::runtime_error("original attention probes require independent Copy outputs");
      return take(graph.node_input(copy,0),"original attention observed actual value");
    };
    const auto ready=[&](const auto &expected,ReportValue consumer) {
      const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const External &entry){return entry.role.name==expected.name&&entry.role.description==expected.description;});
      if(found==recipe.external.end()||found->role.description!=expected.description||found->role.owner.ref()!=owner.ref())
        throw std::runtime_error("original attention must bind the same ready readonly bank");
      const auto a=take(found->role.lease.view(),"original attention actual readonly bank");
      const auto b=take(expected.lease.view(),"original attention once prepared bank");
      if(a.resource()!=b.resource()||a.native_handle()!=b.native_handle()||a.offset()!=b.offset()||a.size()!=b.size()||
          found->role.lease.generation()!=expected.lease.generation()||a.size()!=take(expected.description.required_bytes(),"attention bank extent"))
        throw std::runtime_error("original attention bank permission/identity changed");
      return constant(found->constant,consumer);
    };
    const auto implementation=[&](auto node,const char *role) {
      std::size_t count=0;std::string provider;
      for(std::size_t step=0;step<node.compiled->report.step_count();++step) {
        const auto entry=take(node.compiled->report.step(step),"original attention selected implementation");
        for(std::size_t i=0;i<entry.node_count;++i)
          if(take(node.compiled->report.step_node(step,i),"original attention selected node")==node.value){++count;provider=entry.provider.name;}
      }
      if(count!=1)throw std::runtime_error("original attention operation requires one actual provider");
      ::testing::Test::RecordProperty(std::string("original_attention_provider_")+role,provider);
    };
    if(recipe.original) {
      const auto qa=actual(recipe.original_prefix_ports[5]);
      const auto node=take(graph.producer(qa),"actual original q_a producer");
      const auto operation=take(graph.operation(node),"actual original q_a operation");
      const auto linear=operation.custom_as<c::model_ops::Float8BlockLinear>();
      const auto &expected=recipe.original->prefix_roles[static_cast<std::size_t>(original_prefix::Parameter::kQueryProjection)];
      if(!linear||linear->scale_policy!=recipe.original->arithmetic.policy||
          linear->absmax_floor!=recipe.original->arithmetic.absmax_floor||
          take(graph.node_input(node,0),"actual original q_a input")!=actual(recipe.original_prefix_ports[4])||
          take(graph.node_input(node,1),"actual original q_a bank")!=ready(expected,qa)||
          take(graph.value_description(qa),"actual original q_a storage").data_type()!=c::DataType::kBFloat16)
        throw std::runtime_error("original q_a must retain its actual BF16 predecessor, bank and FP8 arithmetic");
      implementation(node,"query_a");
    }
    const auto &ports=recipe.attention_ports;
    const std::array raw_ports{ports.qb_raw,ports.kva_raw,ports.output_raw};
    const std::array roles{original_attention::Parameter::kQueryProjection,original_attention::Parameter::kKvProjection,original_attention::Parameter::kOutput};
    const std::array names{"query_b","kv_a","output"};
    for(std::size_t i=0;i<raw_ports.size();++i) {
      const auto raw=actual(raw_ports[i]);const auto node=take(graph.producer(raw),"original FP8 projection producer");
      const auto operation=take(graph.operation(node),"original FP8 projection operation");
      const auto linear=operation.custom_as<c::model_ops::Float8BlockLinear>();
      if(!linear||linear->scale_policy!=recipe.sparse->arithmetic.policy||linear->absmax_floor!=recipe.sparse->arithmetic.absmax_floor||
          take(graph.node_input(node,1),"original FP8 weight operand")!=ready(source.roles[static_cast<std::size_t>(roles[i])],raw)||
          take(graph.value_description(raw),"original FP8 output dtype").data_type()!=c::DataType::kFloat32)
        throw std::runtime_error("original projections must use actual dynamic A128 E4/F32 bank and F32 raw output");
      if(i<2&&take(graph.node_input(node,0),"original FP8 actual input")!=published(i?recipe.nx:recipe.state_ports[kQueryState]))
        throw std::runtime_error("original projection lost its actual rounded prefix operand");
      if(i==2) {
        const auto reshape=take(graph.producer(take(graph.node_input(node,0),"original output activation")),"original output flatten");
        const auto cast=take(graph.producer(take(graph.node_input(reshape,0),"original output widened value")),"original output value widening");
        const auto op=take(graph.operation(cast),"original output exact widening");
        const auto *widen=op.get_if<c::Cast>();
        if(!widen||widen->output_type!=c::DataType::kFloat32||take(graph.node_input(cast,0),"original output BF16 value input")!=actual(ports.value_stored))
          throw std::runtime_error("original output activation must exactly widen its actual BF16 value BMM");
      }
      implementation(node,names[i]);
    }
    const std::array raw_store{std::pair{ports.qb_raw,ports.qb_stored},std::pair{ports.kva_raw,ports.kva_stored},
        std::pair{ports.kvnorm_raw,ports.kvnorm_stored},std::pair{ports.output_raw,ports.output_stored},
        std::pair{ports.postnorm_raw,ports.postnorm_stored},std::pair{ports.residual_sum,ports.residual_stored}};
    for(std::size_t i=0;i<(recipe.attention_postnorm?raw_store.size():4);++i) {
      const auto [raw_port,stored_port]=raw_store[i];
      const auto stored=actual(stored_port);const auto node=take(graph.producer(stored),"original BF16 store producer");
      const auto operation=take(graph.operation(node),"original BF16 store operation");
      const auto *cast=operation.get_if<c::Cast>();
      if(!cast||cast->output_type!=c::DataType::kBFloat16||take(graph.node_input(node,0),"original BF16 actual predecessor")!=actual(raw_port))
        throw std::runtime_error("original attention must store BF16 from its actual F32 predecessor");
    }
    const std::array norm_ports{ports.kvnorm_raw,ports.postnorm_raw};
    const std::array norm_roles{original_attention::Parameter::kKvNorm,original_attention::Parameter::kPostNorm};
    for(std::size_t i=0;i<(recipe.attention_postnorm?norm_ports.size():1);++i) {
      const auto node=take(graph.producer(actual(norm_ports[i])),"original learned norm producer");
      const auto operation=take(graph.operation(node),"original learned norm operation");
      const auto *norm=operation.get_if<c::RmsNorm>();
      const auto coefficient=take(graph.producer(take(graph.node_input(node,1),"original learned norm gamma")),"original norm gamma widening");
      const auto gamma_operation=take(graph.operation(coefficient),"original BF16 gamma cast");
      const auto *gamma=gamma_operation.get_if<c::Cast>();
      if(!norm||norm->epsilon!=1.0e-5F||!gamma||gamma->output_type!=c::DataType::kFloat32||
          take(graph.node_input(coefficient,0),"original BF16 gamma bank")!=ready(source.roles[static_cast<std::size_t>(norm_roles[i])],actual(norm_ports[i])))
        throw std::runtime_error("original learned RMS must retain its original BF16 gamma and epsilon");
      if(i&&take(graph.node_input(node,0),"original post norm actual sum")!=actual(ports.residual_sum))
        throw std::runtime_error("post norm must normalize the actual unrounded EH+O sum");
    }
    for(std::size_t i=0;i<source.prepared.size();++i) {
      const auto stored=actual(i?ports.value_stored:ports.absorbed_stored);
      const auto node=take(graph.producer(stored),"original BF16 BMM producer");
      const auto operation=take(graph.operation(node),"original BF16 BMM operation");
      const auto *matmul=operation.get_if<c::MatMul>();
      const auto rhs=take(graph.node_input(node,1),"original BF16 BMM RHS");
      const auto view=take(graph.producer(rhs),"original BF16 BMM RHS transpose");
      if(!matmul||matmul->accumulation_type!=c::DataType::kFloat32||
          !take(graph.operation(view),"original BF16 RHS view").get_if<c::View>()||
          take(graph.node_input(view,0),"original same prepared KV bank")!=ready(source.prepared[i],stored)||
          take(graph.value_description(stored),"original BMM actual store").data_type()!=c::DataType::kBFloat16)
        throw std::runtime_error("original K/V BMM must retain actual BF16 operands and same cold KV_b bank");
      const auto lhs=take(graph.node_input(node,0),"original BF16 BMM LHS");
      const auto reshape=take(graph.producer(lhs),"original BF16 BMM LHS reshape");
      if(!take(graph.operation(reshape),"original BMM reshape kind").get_if<c::View>()||
          take(graph.node_input(reshape,0),"original BMM actual BF16 activation")!=actual(i?ports.attention_stored:ports.qb_stored))
        throw std::runtime_error("original BMM must use its same actual BF16 predecessor");
      implementation(node,i?"value_bmm":"key_bmm");
    }
  }
  void report_original_moe(const c::StorageOwner &owner) const {
    if(!recipe.original_moe)return;
    if(!resident||resident->phases.size()!=2)throw std::runtime_error("original MoE requires its two actual compiled phases");
    const auto &values=recipe.original_moe->ports;const auto &source=recipe.moe->roles;const ReportGraph graph;
    const auto port=[&](Value value) {
      const auto found=std::find_if(recipe.moe_ports.begin(),recipe.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
      if(found==recipe.moe_ports.end())throw std::runtime_error("actual MoE port missing");return found->second;
    };
    const auto actual=[&](Value value) {
      const auto copy=take(graph.producer(published(port(value))),"actual MoE observer Copy");
      if(take(graph.operation_kind(copy),"actual MoE observer kind")!=c::OperationKind::kCopy)
        throw std::runtime_error("MoE observations require independent phase Copy results");
      return take(graph.node_input(copy,0),"original MoE actual semantic value");
    };
    const auto bank=[&](const shared_lm::ReadonlyRole &expected,ReportValue consumer) {
      const auto found=std::find_if(recipe.external.begin(),recipe.external.end(),[&](const External &entry){return entry.role.name==expected.name;});
      if(found==recipe.external.end()||found->role.description!=expected.description||found->role.owner.ref()!=owner.ref())
        throw std::runtime_error("original MoE readonly parameter identity changed");
      const auto a=take(found->role.lease.view(),"actual MoE bank"),b=take(expected.lease.view(),"one original MoE bank");
      if(a.resource()!=b.resource()||a.native_handle()!=b.native_handle()||a.offset()!=b.offset()||a.size()!=b.size()||
          found->role.lease.generation()!=expected.lease.generation())throw std::runtime_error("original MoE bank was copied/replaced per Recipe");
      return constant(found->constant,consumer);
    };
    const auto provider=[&](ReportNode node,const char *role) {
      std::size_t count=0;std::string selected;
      for(std::size_t step=0;step<node.compiled->report.step_count();++step) {
        const auto entry=take(node.compiled->report.step(step),"actual MoE selected implementation");
        for(std::size_t i=0;i<entry.node_count;++i)if(take(node.compiled->report.step_node(step,i),"actual MoE selected node")==node.value){++count;selected=entry.provider.name;}
      }
      if(count!=1)throw std::runtime_error("MoE projection requires exactly one actual phase implementation");
      ::testing::Test::RecordProperty(std::string("original_moe_provider_")+role,selected);
    };
    const auto logits=actual(values.logits);const auto router=take(graph.producer(logits),"actual original router");
    if(!take(graph.operation(router),"actual router operation").custom_as<c::model_ops::DenseLinearF32>()||
        take(graph.node_input(router,0),"actual BF16 router input")!=actual(values.input)||
        take(graph.node_input(router,1),"original BF16 router bank")!=bank(source.router,logits))
      throw std::runtime_error("original router must use actual BF16 postnorm and original full288 bank with F32 output");
    provider(router,"router");
    const auto selected=actual(values.selection);const auto bias=take(graph.producer(selected),"actual biased routing selection");
    const auto bias_op=take(graph.operation(bias),"actual routing bias operation");
    const auto *add=bias_op.get_if<c::Binary>();
    if(!add||add->kind!=c::BinaryKind::kAdd||take(graph.node_input(bias,0),"actual unbiased score")!=actual(values.scores)||
        take(graph.node_input(bias,1),"original F32 correction bias")!=bank(source.bias,selected))
      throw std::runtime_error("original F32 bias must affect selection only");
    const auto local=actual(values.local_ids);const auto mapping=take(graph.producer(local),"actual original compact ID mapping");
    if(!take(graph.operation(mapping),"actual compact map operation").get_if<c::Gather>()||
        take(graph.node_input(mapping,0),"actual generation map")!=phase_input(recipe.original_moe->boundary.map,local)||
        take(graph.node_input(mapping,1),"actual original global IDs input")!=phase_input(recipe.original_moe->boundary.ids,local))
      throw std::runtime_error("compact IDs must gather the same generation map with the actual router global IDs");
    if(source.source->weight_storage()==original_moe::WeightStorage::kMxfp4Group32) {
      const auto &routed=values.routed;const auto &shared=values.shared;
      if(routed.gate_raw||routed.up_raw||routed.down_raw||shared.gate_raw||shared.up_raw||shared.down_raw||
          !routed.down_stored||!shared.down_stored)
        throw std::runtime_error("MXFP4 reports actual BF16 returns, never invented F32 raw projections");
      const std::array grouped_values{routed.gate_stored,routed.up_stored,*routed.down_stored};
      const std::array shared_values{shared.gate_stored,shared.up_stored,*shared.down_stored};
      const std::array names{"gate","up","down"};
      const auto plane=[&](ReportValue value,ReportValue carrier,std::int64_t first,std::int64_t count) {
        const auto reshape=take(graph.producer(value),"actual MXFP4 original plane reshape");
        const auto reshape_op=take(graph.operation(reshape),"actual MXFP4 reshape operation");
        const auto *view=reshape_op.get_if<c::View>();
        if(!view||view->kind()!=c::ViewKind::kReshape)
          throw std::runtime_error("MXFP4 original plane requires its ordinary reshape");
        const auto sliced=take(graph.producer(take(graph.node_input(reshape,0),"actual MXFP4 plane slice")),"actual MXFP4 plane producer");
        const auto slice_op=take(graph.operation(sliced),"actual MXFP4 slice operation");
        const auto *slice=slice_op.get_if<c::View>();
        if(!slice||slice->kind()!=c::ViewKind::kSlice||slice->axis()!=0||slice->start()!=first||
            slice->count()!=count||slice->step()!=1||take(graph.node_input(sliced,0),"actual MXFP4 original carrier")!=carrier)
          throw std::runtime_error("MXFP4 must view the unchanged code/SF planes of the same immutable carrier");
      };
      for(std::size_t leg=0;leg<3;++leg)for(const bool grouped:{true,false}) {
        const auto output=actual(grouped?grouped_values[leg]:shared_values[leg]);
        const auto node=take(graph.producer(output),"actual MXFP4 projection producer");
        const auto operation=take(graph.operation(node),"actual MXFP4 projection operation");
        const auto dense=operation.custom_as<c::model_ops::Mxfp4Linear>();
        const auto selected_projection=operation.custom_as<c::model_ops::GroupedMxfp4Linear>();
        const auto wanted=c::model_ops::Mxfp4ActivationMode::kFloat4Group32;
        if((grouped?(!selected_projection||selected_projection->activation_mode!=wanted):(!dense||dense->activation_mode!=wanted))||
            take(graph.value_description(output),"actual MXFP4 return type").data_type()!=c::DataType::kBFloat16||
            take(graph.value_description(take(graph.node_input(node,0),"actual MXFP4 A")),"actual MXFP4 A type").data_type()!=c::DataType::kBFloat16)
          throw std::runtime_error("original MXFP4 must retain BF16 A/Y and its W4A4 group32 contract");
        const auto carrier=grouped?phase_input(recipe.original_moe->boundary.banks[leg],output):bank(source.shared[leg],output);
        const auto [n,k]=original_moe::geometry(static_cast<original_moe::Leg>(leg),source.source->model());
        const auto slots=grouped?std::int64_t(source.source->model().slots):1;
        plane(take(graph.node_input(node,1),"actual MXFP4 W"),carrier,0,slots*n*(k/2));
        plane(take(graph.node_input(node,2),"actual MXFP4 SF"),carrier,slots*n*(k/2),slots*n*(k/32));
        if(grouped&&take(graph.node_input(node,3),"actual MXFP4 local IDs")!=local)
          throw std::runtime_error("original MXFP4 must use the same checked generation map IDs");
        provider(node,((grouped?std::string("expert_"):std::string("shared_"))+names[leg]).c_str());
      }
      return;
    }
    if(!values.routed.gate_raw||!values.routed.up_raw||!values.routed.down_raw||
        !values.shared.gate_raw||!values.shared.up_raw||!values.shared.down_raw)
      throw std::runtime_error("BF16-returning original MoE requires its typed projection report");
    const std::array grouped_values{*values.routed.gate_raw,*values.routed.up_raw,*values.routed.down_raw};
    const std::array shared_values{*values.shared.gate_raw,*values.shared.up_raw,*values.shared.down_raw};
    const std::array names{"gate","up","down"};
    for(std::size_t leg=0;leg<grouped_values.size();++leg) {
      const auto grouped_raw=actual(grouped_values[leg]);const auto grouped=take(graph.producer(grouped_raw),"actual grouped original projection");
      const auto grouped_op=take(graph.operation(grouped),"actual grouped FP8 operation");
      const auto grouped_linear=grouped_op.custom_as<c::model_ops::GroupedFloat8Linear>();
      if(!grouped_linear||grouped_linear->scale_policy!=recipe.moe->arithmetic.policy||grouped_linear->absmax_floor!=recipe.moe->arithmetic.absmax_floor||
          take(graph.node_input(grouped,0),"actual compact E4/SF operand")!=phase_input(recipe.original_moe->boundary.banks[leg],grouped_raw)||
          take(graph.node_input(grouped,2),"actual local expert IDs")!=local||
          take(graph.value_description(grouped_raw),"actual grouped output type").data_type()!=c::DataType::kFloat32)
        throw std::runtime_error("grouped projection must use the real compact E4/F32 bank, exact mapped IDs and actual rawF32 result");
      provider(grouped,(std::string("expert_")+names[leg]).c_str());
      const auto shared_raw=actual(shared_values[leg]);const auto node=take(graph.producer(shared_raw),"actual shared original projection");
      const auto operation=take(graph.operation(node),"actual shared FP8 operation");
      const auto linear=operation.custom_as<c::model_ops::Float8BlockLinear>();
      if(!linear||linear->scale_policy!=recipe.moe->arithmetic.policy||linear->absmax_floor!=recipe.moe->arithmetic.absmax_floor||
          take(graph.node_input(node,1),"same original shared bank")!=bank(source.shared[leg],shared_raw)||
          take(graph.value_description(shared_raw),"actual shared projection output").data_type()!=c::DataType::kFloat32)
        throw std::runtime_error("shared FFN must retain its original E4/SF bank and explicit rawF32 projection");
      provider(node,(std::string("shared_")+names[leg]).c_str());
    }
  }
  static std::unique_ptr<Program> create(Recipe recipe, c::Context &context, const c::StorageOwner &owner, bool cuda,
                        std::uint32_t max_generations = 5) {
    if (!max_generations) throw std::runtime_error("output generations must be positive");
    auto result=std::make_unique<Program>();auto &p=*result; p.recipe = std::move(recipe); p.recipe.cuda = cuda; p.logical = sem::make_graph(p.recipe.fragment);
    std::vector<g::ConstantBinding> constants;
    for (std::size_t i = 0; i < p.logical.constants.size(); ++i) {
      const auto supplied = std::find_if(p.recipe.external.begin(), p.recipe.external.end(),
          [&](const External &value) { return value.constant == i; });
      if (supplied != p.recipe.external.end()) {
        if (supplied->role.description != p.recipe.fragment.description(p.recipe.fragment.constants[i]))
          throw std::runtime_error("original GLM constant descriptor changed");
        constants.push_back(take(supplied->role.binding(p.logical.constants[i]), "shared original GLM constant owner"));
      } else {
        p.parameters.push_back(upload(owner, p.recipe.fragment.constant_bytes[i])); p.parameter_slots.push_back(i);
        constants.push_back({p.logical.constants[i], take(p.parameters.back().tensor(
            p.recipe.fragment.description(p.recipe.fragment.constants[i])), "typed NextN constant")});
      }
    }
    const c::WorkloadContext workload{take(c::ModelArtifactProfile::create(*p.recipe.fragment.profile), "actual GLM component profile"),
        p.recipe.draft_extend ? c::WorkloadPhase::kPrefill : c::WorkloadPhase::kDecode,
        {0x48594c454d4f444cULL, p.recipe.fragment.region, 1}};
    const g::CompileOptions options{.constants = constants, .workload = workload};
    const auto compile = [&]() -> hyle::Result<g::CompiledGraph> {
#if defined(HYLE_TEST_GLM_STATE_CPU)
      if (!cuda) return g::cpu::compile(p.logical.graph, context, options);
#endif
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      if (cuda) return g::cuda::compile(p.logical.graph, context, owner, options);
#endif
      throw std::runtime_error("GLM state consumer backend not enabled");
    };
    if(p.recipe.original_moe||p.recipe.mlx_boundary) {
      p.resident=std::make_shared<ResidentProgram>();
      ResidentBoundary boundary;
      if(p.recipe.mlx_boundary){const auto &b=*p.recipe.mlx_boundary;boundary={b.input,b.ids,b.weights,b.banks,b.map};}
      else {const auto &b=p.recipe.original_moe->boundary;boundary={b.input,b.ids,b.weights,{b.banks.begin(),b.banks.end()},b.map};}
      p.resident->plan=resident_fragment::split(p.recipe.fragment,std::array{boundary},0x474c4d4d4f455000ULL);
      if(p.recipe.converted) {
        auto source=p.recipe.converted;p.resident->bank_ids={1,2,3,4};
        p.resident->completed=[source]{++source->episodes;};
        p.resident->upload=[source](std::span<const std::int32_t> ids,c::Queue &queue) {
          (void)queue;auto selection=original_moe::select(ids);source->ensure(selection.globals);ResidentBatch batch;
          const auto ready=take(source->cpu_supply->upload(selection,*source->experts.source),"actual converted Flash packed banks");
          batch.banks.assign(ready.banks.begin(),ready.banks.end());
          batch.record=[source,selection]{source->union_sizes.push_back(selection.globals.size());source->selections.push_back(selection);};return batch;
        };
      } else if(p.recipe.mlx) {
        auto source=p.recipe.mlx;
        for(std::size_t i=0;i<=boundary.banks.size();++i)p.resident->bank_ids.push_back(i+1);
        p.resident->completed=[source]{++source->completed_episodes;};
        p.resident->upload=[source,cuda](std::span<const std::int32_t> ids,c::Queue &queue) {
          auto selection=original_moe::select(ids);source->ensure(selection.globals);ResidentBatch batch;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
          if(cuda){if(!source->cuda_supply)throw std::runtime_error("MLX needs its actual CUDA plane supply");
            batch.cuda.emplace(take(source->cuda_supply->upload(source->completed_episodes%2,selection,*source->parameters.source,queue),"actual MLX original planes"));}
          else
#else
          (void)cuda;(void)queue;
#endif
          {if(!source->cpu_supply)throw std::runtime_error("MLX needs its actual CPU plane supply");
            batch.banks=take(source->cpu_supply->upload(selection,*source->parameters.source),"actual MLX CPU original planes").banks;}
          batch.record=[source,selection]{source->union_sizes.push_back(selection.globals.size());source->selections.push_back(selection);};return batch;
        };
      } else {
        auto source=p.recipe.moe;
        p.resident->bank_ids.assign(resident_moe::kBankIds.begin(),resident_moe::kBankIds.end());
        p.resident->completed=[source]{++source->completed_episodes;};
        p.resident->upload=[source,cuda](std::span<const std::int32_t> ids,c::Queue &queue) {
          auto selection=original_moe::select(ids);if(source->before_upload)source->before_upload(selection.globals);ResidentBatch batch;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
          if(cuda){if(!source->cuda_supply)throw std::runtime_error("original MoE needs its actual CUDA supply");
            batch.cuda.emplace(take(source->cuda_supply->upload(source->completed_episodes%2,selection,*source->roles.source,queue),"actual original GLM planes"));}
          else
#else
          (void)cuda;(void)queue;
#endif
          {if(!source->cpu_supply)throw std::runtime_error("original MoE needs its actual CPU supply");
            const auto ready=take(source->cpu_supply->upload(selection,*source->roles.source),"actual original CPU GLM planes");batch.banks.assign(ready.banks.begin(),ready.banks.end());}
          batch.record=[source,selection]{source->union_sizes.push_back(selection.globals.size());source->selections.push_back(selection);};return batch;
        };
      }
      std::vector<original_index::ReadonlyRole> ready;
      for(std::size_t i=0;i<p.logical.constants.size();++i) {
        const auto supplied=std::find_if(p.recipe.external.begin(),p.recipe.external.end(),[&](const External &value){return value.constant==i;});
        if(supplied!=p.recipe.external.end())ready.push_back(supplied->role);
        else {
          const auto found=std::find(p.parameter_slots.begin(),p.parameter_slots.end(),i);
          if(found==p.parameter_slots.end())throw std::runtime_error("original complete model lost its prepared constant");
          ready.push_back({"shared ready GLM component constant",p.recipe.fragment.description(p.recipe.fragment.constants[i]),owner,
              p.parameters.at(std::size_t(found-p.parameter_slots.begin()))});
        }
      }
      for(const auto &phase:p.resident->plan.phases) {
        Recipe local;local.fragment=phase.fragment;local.draft_extend=p.recipe.draft_extend;
        local.state_ports.fill(std::numeric_limits<std::size_t>::max());
        for(std::size_t slot=0;slot<phase.original_constants.size();++slot) {
          const auto original=phase.original_constants[slot];
          const auto found=std::find_if(p.recipe.fragment.constants.begin(),p.recipe.fragment.constants.end(),[&](Value value){return value.ordinal()==original.ordinal();});
          if(found==p.recipe.fragment.constants.end())throw std::runtime_error("compiled phase constant absent from complete original model");
          local.external.push_back({slot,ready.at(std::size_t(found-p.recipe.fragment.constants.begin()))});
        }
        auto program=Program::create(std::move(local),context,owner,cuda,max_generations);
        ResidentProgram::PhaseBindings bindings;
        if(phase.resident_stage) {
          const auto &router=p.resident->plan.routers.at(*phase.resident_stage);
          bindings.banks=router.banks;bindings.banks.push_back(router.map);
          std::vector<std::size_t> external_ports;
          for(std::size_t bank=0;bank<bindings.banks.size();++bank) {
            const auto found=std::find_if(phase.original_inputs.begin(),phase.original_inputs.end(),[&](Value value){return value.ordinal()==bindings.banks[bank].ordinal();});
            if(found==phase.original_inputs.end())throw std::runtime_error("original plane phase lost a bank or its generation map");
            external_ports.push_back(std::size_t(found-phase.original_inputs.begin()));
#if defined(HYLE_TEST_GLM_STATE_CUDA)
            if(cuda)bindings.weights.push_back({program->logical.inputs.at(external_ports.back()),p.resident->bank_ids.at(bank)});
#endif
          }
          if(cuda)bindings.external_inputs.emplace(program->cold_program->input_plan(external_ports));
        }
        p.resident->phases.push_back(std::move(program));
        p.resident->bindings.push_back(std::move(bindings));
      }
      p.report_shared_lm(owner);p.report_original_attention(owner);p.report_original_moe(owner);
      return result;
    }
    std::vector<model_graph_run::Port> input_ports,output_ports;
    for(std::size_t i=0;i<p.logical.inputs.size();++i)
      input_ports.push_back({p.logical.inputs[i],p.recipe.fragment.description(p.recipe.fragment.inputs[i])});
    for(std::size_t i=0;i<p.logical.outputs.size();++i)
      output_ports.push_back({p.logical.outputs[i],p.recipe.fragment.description(p.recipe.fragment.outputs[i])});
    p.cold_program=model_graph_run::ColdProgram::create(take(compile(), "ordinary connected NextN Graph"),input_ports,output_ports);
    p.report_shared_lm(owner);
    p.report_original_attention(owner);
    p.invocation = take(p.cold_program->compiled.executable.create_invocation({.resources = g::ResourceOptions{}}), "ordinary NextN Invocation");
    std::size_t footprint = 0;
    for (std::size_t i = 0; i < p.cold_program->output_count(); ++i) {
      const auto extent = take(p.cold_program->output(i).description.required_bytes(), "NextN output extent");
      footprint += (extent + 255u) / 256u * 256u;
    }
    p.outputs = take(c::LeasePool::create(owner.ref(), footprint * max_generations,
        {.max_blocks = std::uint32_t(p.logical.outputs.size() * max_generations), .backing_alignment = 256}), "bounded ordinary Graph output generations");
    return result;
  }
  template<model_graph_run::LeaseInputRange Inputs=std::span<const c::ReadLease>>
  std::vector<c::ReadLease> execute_resident(c::Queue &queue,const Inputs &inputs) {
    if(!resident||!resident->upload||!resident->completed||std::ranges::size(inputs)+resident->bank_ids.size()!=recipe.fragment.inputs.size())
      throw std::runtime_error("complete original model ordinary input contract");
    std::vector<c::ReadLease> live(recipe.fragment.descriptions.size());
    for(std::size_t i=0;i<std::ranges::size(inputs);++i)live.at(recipe.fragment.inputs[i].ordinal())=std::ranges::begin(inputs)[i];
    std::array<std::vector<c::ReadLease>,2> completed;
    for(std::size_t index=0;index<resident->phases.size();++index) {
      const auto &phase=resident->plan.phases[index];auto &program=*resident->phases[index];
      const auto &bindings=resident->bindings.at(index);
      std::optional<ResidentBatch> batch;
      const model_graph_run::InputPlan *input_plan=nullptr;std::function<void(g::Invocation&)> bind_external;
      if(phase.resident_stage) {
        const auto &boundary=resident->plan.routers.at(*phase.resident_stage);
        batch.emplace(resident->upload(typed<std::int32_t>(live.at(boundary.ids.ordinal())),queue));
#if defined(HYLE_TEST_GLM_STATE_CUDA)
        if(batch->cuda) {
          input_plan=&bindings.external_inputs.value();
          bind_external=[&](g::Invocation &active){check(g::cuda::bind(active,bindings.weights,*batch->cuda),"atomic original plane/map binding");};
        } else
#endif
        {
          if(batch->banks.size()!=bindings.banks.size())throw std::runtime_error("actual CPU plane bank count differs");
          for(std::size_t bank=0;bank<bindings.banks.size();++bank)live.at(bindings.banks[bank].ordinal())=batch->banks[bank];
        }
      }
      const auto arguments=model_graph_run::project_inputs(phase.original_inputs,live);
      completed[index]=program.execute(queue,arguments,input_plan,bind_external);
      for(std::size_t port=0;port<phase.original_outputs.size();++port)live.at(phase.original_outputs[port].ordinal())=completed[index][port];
      if(batch) {
#if defined(HYLE_TEST_GLM_STATE_CUDA)
        if(batch->cuda){const auto accounting=take(batch->cuda->report(),"actual retired original plane upload accounting");
          EXPECT_EQ(accounting.completed_bytes,accounting.submitted_bytes);EXPECT_FALSE(accounting.copied_bytes_unknown);}
#endif
        batch->record();const auto &boundary=resident->plan.routers.at(*phase.resident_stage);
        for(auto value:boundary.banks)live.at(value.ordinal())={};live.at(boundary.map.ordinal())={};
      }
      // Ordinary execute waited and reset; release input/batch guards before slot reuse.
    }
    std::vector<c::ReadLease> result;for(const auto location:resident->plan.full_outputs)result.push_back(completed.at(location.phase).at(location.port));
    resident->completed();return result;
  }
  template<model_graph_run::LeaseInputRange Inputs=std::span<const c::ReadLease>>
  std::vector<c::ReadLease> execute(c::Queue &queue,const Inputs &inputs,
      const model_graph_run::InputPlan *input_plan=nullptr,const std::function<void(g::Invocation&)> &bind_external={}) {
    if(resident) {
      if(input_plan||bind_external)throw std::runtime_error("resident orchestration owns its actual dynamic bindings");
      return execute_resident(queue,inputs);
    }
    return model_graph_run::execute(*cold_program,invocation,outputs,queue,inputs,
        completed_submissions,input_plan,bind_external);
  }

};


std::shared_ptr<mlx_runtime::Source> prepare_mlx_source(mlx_roles::Bound bound,c::Context &context,
    const c::StorageOwner &owner,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
  auto result=std::make_shared<mlx_runtime::Source>();result->parameters=std::move(bound);
  const auto &data=*result->parameters.source;
  if(data.prefix()!="model.language_model.layers.44.") {
    Recipe cold;cold.state_ports.fill(std::numeric_limits<std::size_t>::max());cold.fragment.profile=&mlx_runtime::kProfile;cold.fragment.region=0x514d4c584b56434fULL;
    const auto original=external(cold,result->parameters.at(data.prefix()+"self_attn.kv_b_proj.weight"));
    const auto prepared=mlx_model::prepare_kv_b(cold.fragment,original);cold.fragment.outputs={prepared[0],prepared[1]};
    auto program_owner=Program::create(std::move(cold),context,owner,cuda,1);
    auto &program=*program_owner;auto queue=take(context.create_queue(),"MLX original KV_b cold queue");
    auto output=program.execute(queue,{});result->cold_submissions=program.native_submissions();
    result->prepared_pool=std::move(program.outputs);
    for(std::size_t i=0;i<2;++i){const auto description=program.recipe.fragment.description(program.recipe.fragment.outputs[i]);
      result->prepared[i]={i?"MLX original BF16 value bank":"MLX original BF16 key bank",description,owner,output[i]};
      const auto actual=typed<std::uint16_t>(output[i]);const auto original_bytes=data.bytes(data.prefix()+"self_attn.kv_b_proj.weight");
      const mlx_reference::BFloat original_weight{original_bytes,32768,512};const std::size_t n=i?256:512,k=i?512:256;
      for(std::size_t h=0;h<64;++h){const mlx_reference::BFloatHead weight{original_weight,h,i!=0};
        for(std::size_t row=0;row<n;++row)for(std::size_t col=0;col<k;++col)EXPECT_EQ(codec::wide(actual[(h*n+row)*k+col]),weight(row,col).value);}
    }
    for(auto &role:result->parameters.roles)if(role.name==data.prefix()+"self_attn.kv_b_proj.weight")role.lease={};
  }
  else {
    Recipe cold;cold.state_ports.fill(std::numeric_limits<std::size_t>::max());cold.fragment.profile=&mlx_runtime::kProfile;cold.fragment.region=0x514d4c58434f4e56ULL;
    constexpr std::array names{"self_attn.q_conv1d.weight","self_attn.k_conv1d.weight","self_attn.v_conv1d.weight"};std::array<Value,3> raw;
    for(std::size_t i=0;i<3;++i)raw[i]=external(cold,result->parameters.at(data.prefix()+names[i]));
    const auto prepared=mlx_target::prepare_convolutions(cold.fragment,raw);cold.fragment.outputs.assign(prepared.outputs.begin(),prepared.outputs.end());
    auto program_owner=Program::create(std::move(cold),context,owner,cuda,1);
    auto &program=*program_owner;auto queue=take(context.create_queue(),"MLX original convolution cold queue");
    auto output=program.execute(queue,{});result->cold_submissions=program.native_submissions();result->prepared_pool=std::move(program.outputs);
    for(std::size_t i=0;i<3;++i){const auto description=program.recipe.fragment.description(program.recipe.fragment.outputs[i]);
      result->convolutions[i]={names[i],description,owner,output[i]};
      EXPECT_EQ(typed<float>(output[i]),mlx_roles::raw::widen_bfloat(data.bytes(data.prefix()+names[i])));}
  }
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda){if(!device)throw std::runtime_error("MLX supply requires the same compiled actual CUDA resource");
    result->cuda_supply=std::make_unique<model_plane_resident_weights::CudaSupply>(take(model_plane_resident_weights::CudaSupply::create(context,owner,*device,mlx_runtime::layout()),"actual MLX CUDA original9-plane supply"));}
  else
#else
  (void)cuda;
#endif
  result->cpu_supply=std::make_unique<model_plane_resident_weights::CpuSupply>(take(model_plane_resident_weights::CpuSupply::create(owner,mlx_runtime::layout()),"actual MLX CPU original9-plane supply"));
  return result;
}
std::shared_ptr<OriginalAttention> prepare_original_attention(std::shared_ptr<const original_attention::Data> source,
    c::Context &context,const c::StorageOwner &owner,bool cuda,
    const c::ModelArtifactProfileDescription *profile=&glm::core::profiles::glm53_flash_official) {
  auto result=std::make_shared<OriginalAttention>();
  if(!source)throw std::runtime_error("original sparse layer requires its authenticated attention source");
  result->source=std::move(source);
  result->roles=original_attention::bind(*result->source,owner);
  Recipe cold;cold.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f=cold.fragment;f.profile=profile;f.region=0x474c4d4b56434f4cULL;
  const auto original=external(cold,result->roles[static_cast<std::size_t>(original_attention::Parameter::kKvB)]);
  const auto heads=glm::reshape(f,original,{glm::kHeads,512,kLatent});
  const auto keys=f.one(c::Operation(take(c::View::slice(1,0,256),"original KV_b key half")),{heads});
  const auto values=f.one(c::Operation(take(c::View::slice(1,256,256),"original KV_b value half")),{heads});
  f.outputs={f.one(c::Operation(c::Copy{}),{glm::permute(f,keys,{0,2,1})}),
      f.one(c::Operation(c::Copy{}),{values})};
  auto program_owner=Program::create(std::move(cold),context,owner,cuda,1);
  auto &program=*program_owner;
  auto queue=take(context.create_queue(),"one original KV_b cold layout queue");
  auto ready=program.execute(queue,{});++result->cold_submissions;
  result->prepared={original_index::ReadonlyRole{"original KV_b key BF16 cold transpose",sem::dense({glm::kHeads,kLatent,256},c::DataType::kBFloat16),owner,ready[0]},
      original_index::ReadonlyRole{"original KV_b value BF16 cold copy",sem::dense({glm::kHeads,256,kLatent},c::DataType::kBFloat16),owner,ready[1]}};
  // The open output pool owns both ready BF16 banks through every compiled
  // reader. The temporary graph/Invocation retires here; the source device
  // bank is no longer needed. Its authenticated mapped original bytes remain
  // available for independent references without another device copy.
  result->prepared_pool=std::move(program.outputs);
  result->roles[static_cast<std::size_t>(original_attention::Parameter::kKvB)].lease={};
  return result;
}
std::shared_ptr<OriginalAttention> prepare_original_attention(const std::filesystem::path &directory,
    c::Context &context,const c::StorageOwner &owner,bool cuda) {
  return prepare_original_attention(std::make_shared<const original_attention::Data>(
      original_attention::Data::load(directory)),context,owner,cuda);
}

// Actual typed source overload. It only binds existing components and uses
// the same Program/phase controller. A hook may explicitly attach authenticated
// missing originals after phase0; Data lookup itself never starts acquisition.
using OriginalExpertHook=std::function<void(std::shared_ptr<original_moe::Data>,std::span<const std::int32_t>)>;
inline std::shared_ptr<const OriginalNextn> original_nextn(std::shared_ptr<const artifact_source::Data> source,
    c::Context &context,const c::StorageOwner &owner,bool cuda,
    const c::ModelArtifactProfileDescription &profile,OriginalExpertHook hook={}
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
  if(!source)throw std::runtime_error("original typed GLM source is missing");
  if(!profile.complete()||profile.artifact_repository!=source->identity.repository||
      profile.artifact_revision!=source->identity.revision)
    throw std::runtime_error("original typed GLM requires its actual immutable artifact profile");
  auto result=std::make_shared<OriginalNextn>();result->arithmetic=source->arithmetic;
  result->profile=take(c::ModelArtifactProfile::create(profile),"own actual GLM source profile");
  result->profile_description=result->profile.description();
  result->prefix=std::shared_ptr<const original_prefix::Data>(source,&source->prefix);
  result->sparse=std::make_shared<OriginalSparseLayer>();result->sparse->arithmetic=source->arithmetic;
  result->sparse->profile=&result->profile_description;
  result->sparse->indexer=std::shared_ptr<const original_index::Data>(source,&source->indexer);
  result->sparse->indexer_roles=original_index::bind(*result->sparse->indexer,owner);
  result->sparse->attention=prepare_original_attention(
      std::shared_ptr<const original_attention::Data>(source,&source->attention),context,owner,cuda,&result->profile_description);
  result->vocabulary=std::make_shared<const shared_lm::Vocabulary>(shared_lm::Vocabulary::bind(
      std::shared_ptr<const shared_lm::Data>(source,&source->vocabulary),owner));
  result->lm_observations=std::make_shared<LmObservations>();
  for(std::size_t i=0;i<result->prefix_roles.size();++i) {
    const auto role=static_cast<original_prefix::Parameter>(i);
    result->prefix_roles[i]={"original GLM NextN input/query",original_prefix::description(role),owner,
        upload(owner,original_prefix::payload(*result->prefix,role))};
  }
  result->moe=std::make_shared<OriginalMoe>();result->moe->arithmetic=source->arithmetic;
  result->moe->roles=original_moe::bind(source->moe,owner);
  result->moe->before_upload=[data=source->moe,hook=std::move(hook)](std::span<const std::int32_t> ids) {
    const auto missing=data->missing(ids);
    if(missing.empty())return;
    if(!hook)throw std::runtime_error("actual original GLM experts are missing; explicit source hook required");
    hook(data,missing);
    if(!data->missing(ids).empty())throw std::runtime_error("explicit GLM source hook did not supply every actual ID");
  };
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("original typed GLM needs the actual compiled device owner");
    result->moe->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(
        resident_moe::CudaSupply::create(context,owner,*device,source->moe->model(),source->moe->weight_storage()),"one actual typed GLM CUDA expert supply"));
  } else
#endif
  result->moe->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(
      resident_moe::CpuSupply::create(owner,source->moe->model(),source->moe->weight_storage()),"one actual typed GLM CPU expert supply"));
  return result;
}

// Shared immutable artifact binding. Only parameters/supply change; the ordinary
// Program and existing K1 controller below execute every selected source kind.
inline std::shared_ptr<OriginalMoe> prepare_artifact_moe(std::shared_ptr<original_moe::Data> data,
    c::Context &context,const c::StorageOwner &owner,bool cuda,glm_fp8_arithmetic::Arithmetic arithmetic,
    OriginalExpertHook hook={}
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
#if !defined(HYLE_TEST_GLM_STATE_CUDA)
  (void)context;
  if(cuda)throw std::runtime_error("artifact MoE CUDA supply was not built");
#endif
  if(!data)throw std::runtime_error("actual artifact MoE Data is absent");
  arithmetic.require();auto result=std::make_shared<OriginalMoe>();result->arithmetic=arithmetic;
  result->roles=original_moe::bind(data,owner);
  result->before_upload=[data,hook=std::move(hook)](std::span<const std::int32_t> ids) {
    const auto missing=data->missing(ids);if(missing.empty())return;
    if(!hook)throw std::runtime_error("actual artifact experts missing; explicit supply hook required");
    hook(data,missing);
    if(!data->missing(ids).empty())throw std::runtime_error("explicit artifact hook left actual original IDs missing");
  };
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("actual artifact supply needs the same typed CUDA device owner");
    result->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(resident_moe::CudaSupply::create(
        context,owner,*device,data->model(),data->weight_storage()),"actual artifact CUDA expert supply"));
  } else
#endif
  result->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(resident_moe::CpuSupply::create(
      owner,data->model(),data->weight_storage()),"actual artifact CPU expert supply"));
  return result;
}
struct ArtifactTargets final {std::shared_ptr<OriginalTarget43> target43;std::shared_ptr<OriginalTarget44> target44;};
inline ArtifactTargets artifact_targets(std::shared_ptr<const OriginalNextn> nextn,
    const std::filesystem::path &directory43,const artifact_source::Manifest &manifest43,
    const std::filesystem::path &directory44,const artifact_source::Manifest &manifest44,
    c::Context &context,const c::StorageOwner &owner,bool cuda,
    OriginalExpertHook hook43={},OriginalExpertHook hook44={}
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
  if(!nextn||!nextn->moe||!nextn->moe->roles.source)
    throw std::runtime_error("actual target binding requires its same immutable NextN/vocabulary source");
  const auto &identity=nextn->moe->roles.source->identity();
  const auto pins43=glm_original_target_source::input43(manifest43,identity);
  const auto pins44=glm_original_target_source::input44(manifest44,identity);
  if(manifest43.arithmetic.policy!=nextn->arithmetic.policy||manifest44.arithmetic.policy!=nextn->arithmetic.policy||
      manifest43.arithmetic.absmax_floor!=nextn->arithmetic.absmax_floor||manifest44.arithmetic.absmax_floor!=nextn->arithmetic.absmax_floor)
    throw std::runtime_error("actual target/NextN FP8 arithmetic contract differs");
  ArtifactTargets result;result.target43=std::make_shared<OriginalTarget43>();result.target44=std::make_shared<OriginalTarget44>();
  auto &target43=*result.target43;target43.artifact_owner=nextn;
  auto parameters43=std::make_shared<const original_target43::Data>(original_target43::Data::load(directory43,pins43.parameters));
  target43.parameters=original_target43::bind(parameters43,owner);
  target43.sparse=std::make_shared<OriginalSparseLayer>();target43.sparse->arithmetic=manifest43.arithmetic;
  target43.sparse->profile=&nextn->profile_description;
  target43.sparse->indexer=std::shared_ptr<const original_index::Data>(parameters43,&parameters43->indexer);
  target43.sparse->indexer_roles=original_index::bind(*target43.sparse->indexer,owner);
  target43.sparse->attention=prepare_original_attention(std::shared_ptr<const original_attention::Data>(
      parameters43,&parameters43->attention),context,owner,cuda,&nextn->profile_description);
  target43.moe=prepare_artifact_moe(std::make_shared<original_moe::Data>(original_moe::Data::load_selected(
      directory43,manifest43.identity,pins43.moe,original_moe::kFlashGeometry,original_moe::WeightStorage::kMxfp4Group32)),
      context,owner,cuda,manifest43.arithmetic,std::move(hook43)
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  auto &target44=*result.target44;target44.artifact_owner=nextn;
  target44.parameters=original_target44::bind(std::make_shared<const original_target44::Data>(
      original_target44::Data::load(directory44,pins44.parameters)),owner);
  target44.moe=prepare_artifact_moe(std::make_shared<original_moe::Data>(original_moe::Data::load_selected(
      directory44,manifest44.identity,pins44.moe,original_moe::kFlashGeometry,original_moe::WeightStorage::kMxfp4Group32)),
      context,owner,cuda,manifest44.arithmetic,std::move(hook44)
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  // Logical complete source sizes describe the authenticated header schema;
  // selected supply still maps only the actual IDs requested by phase0.
  target43.logical_source_bytes=4016880472ULL;target44.logical_source_bytes=4143179608ULL;
  return result;
}

codec::Encoded observe_codec(std::span<const float> source, std::int64_t rows,
                              std::span<const c::ReadLease> output, const CodecPorts &ports) {
  const auto input = typed<float>(output[ports.input]);
  const auto transformed = typed<float>(output[ports.transformed]);
  const auto rounded = typed<float>(output[ports.rotated]);
  if (source.size() != std::size_t(rows * kIndexDepth) || input.size() != source.size() ||
      transformed.size() != source.size() || rounded.size() != source.size())
    throw std::runtime_error("whole NextN codec diagnostic extent");
  constexpr double u = 0x1p-24, gamma129 = 129 * u / (1 - 129 * u);
  const double normalization = 1.0 / std::sqrt(double(kIndexDepth));
  std::size_t bad_input = 0, bad_math = 0, bad_bfloat = 0;
  double max_ratio = 0;
  for (std::int64_t row = 0; row < rows; ++row) {
    double sumabs = 0;
    for (std::int64_t d = 0; d < kIndexDepth; ++d) {
      const auto i = std::size_t(row * kIndexDepth + d);
      const float wanted = codec::wide(codec::bfloat(source[i]));
      if (std::bit_cast<std::uint32_t>(input[i]) != std::bit_cast<std::uint32_t>(wanted)) ++bad_input;
      sumabs += std::abs(double(wanted));
    }
    const double bound = gamma129 * sumabs * normalization;
    for (std::int64_t d = 0; d < kIndexDepth; ++d) {
      double sum = 0;
      for (std::int64_t k = 0; k < kIndexDepth; ++k)
        sum += (std::popcount(unsigned(d & k)) & 1 ? -1.0 : 1.0) *
            codec::wide(codec::bfloat(source[std::size_t(row * kIndexDepth + k)]));
      const auto i = std::size_t(row * kIndexDepth + d);
      const double difference = std::abs(double(transformed[i]) - sum * normalization);
      if (!std::isfinite(transformed[i]) || difference > bound) ++bad_math;
      max_ratio = std::max(max_ratio, bound ? difference / bound : difference);
      if (std::bit_cast<std::uint32_t>(rounded[i]) !=
          std::bit_cast<std::uint32_t>(codec::wide(codec::bfloat(transformed[i])))) ++bad_bfloat;
    }
  }
  EXPECT_EQ(bad_input, 0u) << "exact explicit BF16 boundary before Hadamard";
  EXPECT_EQ(bad_math, 0u) << "independent F64 Sylvester Hadamard, max/gamma129=" << max_ratio;
  EXPECT_EQ(bad_bfloat, 0u) << "BF16 RNE of actual, independently checked transform";
  const auto expected = codec::encode_rotated(rounded, rows);
  EXPECT_EQ(typed<std::uint8_t>(output[ports.codes]), expected.codes);
  const auto scales = typed<float>(output[ports.scales]);
  EXPECT_EQ(scales.size(), expected.scales.size());
  for (std::size_t i = 0; i < std::min(scales.size(), expected.scales.size()); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(scales[i]), std::bit_cast<std::uint32_t>(expected.scales[i]));
  return expected;
}

struct SparseSnapshot final {
  std::array<c::ReadLease,kSparseEntries.size()> values;
  bool plain_index=false;
  const c::ReadLease &operator[](Entry entry) const {
    const auto found = std::find(kSparseEntries.begin(), kSparseEntries.end(), entry);
    if (found == kSparseEntries.end()) throw std::runtime_error("entry is not a sparse snapshot member");
    return values[std::size_t(found - kSparseEntries.begin())];
  }
};
struct SparseAccess final {
  c::ReadLease member(const SparseSnapshot &v, Entry entry) const { return v[entry]; }
  std::int32_t scalar(const SparseSnapshot &v, Entry entry) const { return typed<std::int32_t>(v[entry]).at(0); }
};
SparseSnapshot initial_sparse(const c::StorageOwner &owner, Program &initializer,
                              c::Queue &queue, std::int32_t position,bool plain_index=false) {
  std::array<Bytes,kResources> values;
  std::vector<float> cache(kCapacity * kLatent, std::numeric_limits<float>::quiet_NaN());
  for (std::int64_t t = 0; t < position; ++t) for (std::int64_t d = 0; d < kLatent; ++d)
    cache[std::size_t(t * kLatent + d)] = 0.25F + float(t % 97) / 64.0F + float(d % 11) / 128.0F;
  values[kCache] = sem::bytes(cache);
  std::vector<float> raw(kRing * kIndexDepth), gates(raw.size());
  for (std::int32_t p = position - std::int32_t(kRing); p < position; ++p)
    for (std::int64_t d = 0; d < kIndexDepth; ++d) {
    raw[std::size_t((p % kRing) * kIndexDepth + d)] = float((p * 3 + d * 7) % 97 - 48) / 32.0F;
      gates[std::size_t((p % kRing) * kIndexDepth + d)] = float((p * 7 + d * 11) % 29 - 14) / 8.0F;
    }
  std::vector<std::uint16_t> raw_bfloat(raw.size()), gate_bfloat(gates.size());
  for (std::size_t i = 0; i < raw.size(); ++i) { raw_bfloat[i] = codec::bfloat(raw[i]); gate_bfloat[i] = codec::bfloat(gates[i]); }
  values[kRawKey]=plain_index?sem::bytes(raw):sem::bytes(raw_bfloat);values[kRawGate]=plain_index?sem::bytes(gates):sem::bytes(gate_bfloat);
  if (position / 4 != kBasePosition / 4) throw std::runtime_error("initial encoded pool preparation extent");
  std::vector<float> pools(std::size_t(position / 4 * kIndexDepth));
  for (std::int64_t p = 0; p < position / 4; ++p) for (std::int64_t d = 0; d < kIndexDepth; ++d)
    pools[std::size_t(p * kIndexDepth + d)] = (0.125F + float(p) / 64.0F) * (0.5F + float(d % 11) / 32.0F);
  if(plain_index) {
    pools.resize(std::size_t(kPoolCapacity*kIndexDepth),std::numeric_limits<float>::quiet_NaN());
    values[kPool]=sem::bytes(pools);
    const std::array input{upload(owner,values[kPool])};const auto copied=initializer.execute(queue,input);EXPECT_EQ(read(copied.at(0)),values[kPool]);
    values[kQueryState]=sem::bytes(std::vector<float>(kQueryRank,0));
    const std::array<Entry,6> counters{kMlaCursor,kIndexCursor,kPoolCount,kRingOffset,kTailValid,kTailStart};
    const std::array<std::int32_t,6> scalars{position,position,position/4,position%std::int32_t(kRing),position%4,position-position%4};
    for(std::size_t i=0;i<counters.size();++i)values[counters[i]]=sem::bytes(std::vector{scalars[i]});
    values[kSelection]=sem::bytes(std::vector<std::int32_t>(kIds,-1));SparseSnapshot r;r.plain_index=true;
    for(std::size_t i=0;i<kSparseEntries.size();++i)if(kSparseEntries[i]!=kPoolScales)r.values[i]=kSparseEntries[i]==kPool?copied.at(0):upload(owner,values[kSparseEntries[i]]);
    return r;
  }
  const std::array initial_input{upload(owner,sem::bytes(pools))};
  const auto encoded=initializer.execute(queue,initial_input);
  const auto expected = observe_codec(pools, position / 4, encoded, initializer.recipe.pool_codec);
  const auto actual_codes = typed<std::uint8_t>(encoded[0]);
  const auto actual_scales = typed<float>(encoded[1]);
  EXPECT_TRUE(std::equal(expected.codes.begin(), expected.codes.end(), actual_codes.begin()));
  EXPECT_TRUE(std::equal(expected.scales.begin(), expected.scales.end(), actual_scales.begin()));
  for (std::size_t row = std::size_t(position / 4); row < std::size_t(kPoolCapacity); ++row) {
    EXPECT_TRUE(std::isnan(actual_scales[row]));
    for (std::size_t d = 0; d < std::size_t(kIndexDepth); ++d) EXPECT_EQ(actual_codes[row * std::size_t(kIndexDepth) + d], 0x7f);
  }
  values[kQueryState] = sem::bytes(std::vector<float>(kQueryRank, 0));
  const std::array<Entry, 6> counters{kMlaCursor, kIndexCursor, kPoolCount, kRingOffset, kTailValid, kTailStart};
  const std::array<std::int32_t, 6> scalars{position, position, position / 4,
      position % std::int32_t(kRing), position % 4, position - position % 4};
  for (std::size_t i = 0; i < counters.size(); ++i) values[counters[i]] = sem::bytes(std::vector{scalars[i]});
  values[kSelection] = sem::bytes(std::vector<std::int32_t>(kIds, -1));
  SparseSnapshot result;
  for (std::size_t i = 0; i < kSparseEntries.size(); ++i) {
    const auto entry = kSparseEntries[i];
    result.values[i] = entry == kPool ? encoded[0] : entry == kPoolScales ? encoded[1] : upload(owner, values[entry]);
  }
  return result;
}

struct Epoch final { std::uint64_t epoch, producer_position, reuse_count; };
static_assert(sizeof(Epoch) == 24);
struct State final {
  static constexpr std::uint32_t kMaxVersions=16,kMaxDrafts=2,kMaxRetainedBlocks=256;
  c::StateSchema schema;
  c::StateStore store;
  std::vector<c::StateEntryId> entries;
  std::vector<Entry> resources;
  bool plain_index=false;
  bool present(std::size_t i)const{return !plain_index||i!=kPoolScales;}
  std::uint64_t epoch = 7;
  bool enabled = true;
  std::size_t peak_backing_bytes=0;
  static State create(const c::StorageOwner &owner,
                      std::span<const c::StateEntrySpec> extra_specs = {},
                      std::size_t max_backing_bytes = 128u << 20,bool plain_index=false) {
    const auto desc=descriptions(plain_index);std::vector<c::StateEntrySpec> specs;
    std::vector<std::size_t> logical;
    for(const auto i:state_resources(plain_index)){specs.push_back({.description=desc[i],.resource=owner.ref()});logical.push_back(i);}
    specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Epoch)});logical.push_back(kEpoch);
    for(std::size_t i=0;i<extra_specs.size();++i){specs.push_back(extra_specs[i]);logical.push_back(kEntries+i);}
    State result;result.plain_index=plain_index;result.resources=state_resources(plain_index); result.schema = take(c::StateSchema::create(specs), "complete component schema");
    result.store = take(c::StateStore::create(result.schema, {.max_versions = kMaxVersions, .max_drafts = kMaxDrafts,
        .max_retained_blocks = kMaxRetainedBlocks, .max_backing_bytes = max_backing_bytes}), "bounded complete component state store");
    result.entries.resize(kEntries+extra_specs.size());
    for(std::size_t i=0;i<specs.size();++i)result.entries[logical[i]]=take(result.schema.entry(i),"complete schema entry");
    return result;
  }
  c::ReadLease member(const c::StateVersion &v, std::size_t i) const { if(!present(i))throw std::runtime_error("plain index has no scale resource");return take(v.member(entries[i]), "actual complete state member"); }
  std::int32_t scalar(const c::StateVersion &v, Entry entry) const { return typed<std::int32_t>(member(v, entry)).at(0); }
  Epoch facts(const c::StateVersion &v) const {
    const auto bytes = read(member(v, kEpoch)); Epoch result{}; std::memcpy(&result, bytes.data(), sizeof(result)); return result;
  }
  bool qualifies(const c::StateVersion &v) const { return enabled && v.valid() && facts(v).epoch == epoch; }
  c::StateVersion initial(const c::StorageOwner &owner, Program &initializer, c::Queue &queue,
                         std::int32_t position = kBasePosition,
                         std::span<const c::StateMember> extra_members = {},
                         const c::ReadLease &seed_hidden = {}) {
    if (entries.size() != kEntries + extra_members.size())
      throw std::runtime_error("initial complete component extra member count");
    const auto sparse = initial_sparse(owner, initializer, queue, position,plain_index);
    std::array<Bytes, kResources> values;
    values[kHiddenState] = sem::bytes(embedding_values(0));
    values[kPending] = sem::bytes(std::vector<std::int32_t>{1});
    values[kJournal] = sem::bytes(std::vector<float>(kJournalRows * kHidden, std::numeric_limits<float>::quiet_NaN()));
    values[kLagCount] = sem::bytes(std::vector<std::int32_t>{0});
    std::vector<c::StateMember> members;
    for (const auto i:resources) {
      const auto found = std::find(kSparseEntries.begin(), kSparseEntries.end(), Entry(i));
      members.push_back({entries[i],found!=kSparseEntries.end()?sparse[Entry(i)]:
          i==kHiddenState&&seed_hidden.valid()?seed_hidden:upload(owner,values[i])});
    }
    const Epoch metadata{0,0,0};members.push_back({entries[kEpoch],{},std::as_bytes(std::span(&metadata,1))});
    members.insert(members.end(),extra_members.begin(),extra_members.end());
    auto result=take(store.import(members), "initial declared component checkpoint");
    peak_backing_bytes=std::max(peak_backing_bytes,store.stats().backing_bytes);return result;
  }
  c::StateVersion stage(const c::StateVersion &from, const Recipe &recipe, std::span<const c::ReadLease> outputs,
                        std::span<const c::StateMember> extra_members = {}) {
    auto draft = take(store.begin(from), "whole component draft");
    for (const auto i:resources)
      if (recipe.state_ports[i] != std::numeric_limits<std::size_t>::max())
        check(draft.replace(entries[i], outputs[recipe.state_ports[i]]), "stage actual component output");
    for (const auto &member : extra_members)
      check(draft.replace(member.entry, member.resource), "stage actual target component output in same draft");
    auto metadata = facts(from);
    if (recipe.fresh) metadata = {epoch, std::uint64_t(scalar(from, kMlaCursor)), 0};
    else if (!recipe.catchup) ++metadata.reuse_count;
    check(draft.replace_metadata(entries[kEpoch], std::as_bytes(std::span(&metadata, 1))), "explicit component iteration policy");
    auto result=take(draft.finalize(), "finalize all component state");
    peak_backing_bytes=std::max(peak_backing_bytes,store.stats().backing_bytes);return result;
  }
};
using Image = std::vector<Bytes>;
Image image(const State &s, const c::StateVersion &v) {
  Image result(s.entries.size()); for (std::size_t i = 0; i < s.entries.size(); ++i)if(s.present(i))result[i] = read(s.member(v, i)); return result;
}
void same_storage(const c::ReadLease &a, const c::ReadLease &b) {
  const auto av = take(a.view(), "first actual storage identity"), bv = take(b.view(), "second actual storage identity");
  EXPECT_EQ(av.resource(), bv.resource()); EXPECT_EQ(av.native_handle(), bv.native_handle());
  EXPECT_EQ(av.offset(), bv.offset()); EXPECT_EQ(av.size(), bv.size()); EXPECT_EQ(a.generation(), b.generation());
}
void quantified(const Recipe &recipe, std::span<const float> actual, std::span<const float> expected) {
  sem::expect_output(recipe.fragment, actual, expected);
}
template<class StateView, class Version>
void exact_scatter(const StateView &s, const Version &from, const Version &next,
                   Entry bank, std::size_t row, std::size_t width, const c::ReadLease &update,
                   std::size_t element_bytes = sizeof(float)) {
  auto expected = read(s.member(from, bank)); const auto actual_update = read(update);
  ASSERT_EQ(actual_update.size(), width * element_bytes);
  std::copy(actual_update.begin(), actual_update.end(), expected.begin() + row * width * element_bytes);
  EXPECT_EQ(read(s.member(next, bank)), expected) << "only the actual native-produced row may change";
}

std::size_t mlx_port(const Recipe &r,Value value) {
  const auto found=std::find_if(r.mlx_ports.begin(),r.mlx_ports.end(),[&](const auto &p){return p.first.ordinal()==value.ordinal();});
  if(found==r.mlx_ports.end())throw std::runtime_error("MLX actual predecessor has no original port");return found->second;
}

inline packed_base::ActivationCodec converted_codec(bool cuda) {
  return cuda?packed_base::ActivationCodec::kCudaMmvqQ8_1:packed_base::ActivationCodec::kCpuAvx2;
}
auto converted_original(const Recipe &r) {
  return [&r](Value value)->std::pair<original_named_role_source::Spec,std::span<const std::byte>> {
    const auto it=std::find_if(r.external.begin(),r.external.end(),[&](const External &e){return r.fragment.constants.at(e.constant).ordinal()==value.ordinal();});
    if(it==r.external.end())throw std::runtime_error("converted original observer requires an actual authenticated constant");
    return {r.converted->spec(it->role.name),r.converted->bytes(it->role.name)};
  };
}
auto converted_expert(const Recipe &r) {
  return [&r](std::size_t leg,std::int32_t global) {
    const auto source=r.converted->experts.source;const auto type=glm_gguf_runtime_source::Leg(leg);
    return std::pair{source->native(type),source->expert_bytes(type,global)};
  };
}
template<class StateView,class Version>
auto converted_actual(const Recipe &r,std::span<const c::ReadLease> output,const StateView &s,const Version &before) {
  return [&r,output,&s,&before](Value value)->Bytes {
    const auto found=std::find_if(r.mlx_ports.begin(),r.mlx_ports.end(),[&](const auto &e){return e.first.ordinal()==value.ordinal();});
    if(found!=r.mlx_ports.end())return read(output[found->second]);
    const auto input=std::find_if(r.fragment.inputs.begin(),r.fragment.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
    const auto offset=r.converted43?1U:0U;
    if(input==r.fragment.inputs.end()||std::size_t(input-r.fragment.inputs.begin())<offset||
        std::size_t(input-r.fragment.inputs.begin())-offset>=r.state_entries.size())
      throw std::runtime_error("converted observer predecessor lacks actual diagnostic/input lease");
    return read(s.member(before,r.state_entries[std::size_t(input-r.fragment.inputs.begin())-offset]));
  };
}
template<class StateView,class Version>
void observe_converted_history(const StateView &s,const Version &from,const Version &next,const Recipe &r,std::span<const c::ReadLease> output) {
  auto actual=converted_actual(r,output,s,from);const auto &history=*r.mlx_history;
  EXPECT_TRUE(gguf_reference::observe_history(*r.converted->parameters.source,history,r.fragment,r.converted_parameters->nextn,
      *r.pooled_before,*r.pooled_after,actual,converted_original(r),r.position,!r.converted->full_checked,converted_codec(r.cuda)));
  for(const auto entry:r.state_entries)if(r.state_ports[entry]!=std::numeric_limits<std::size_t>::max())
    same_storage(s.member(next,entry),output[r.state_ports[entry]]);
  if(::testing::Test::HasFailure())throw std::runtime_error("converted Flash accepted-history actual predecessor check failed");
}
void observe_converted(const State &s,const c::StateVersion &from,const c::StateVersion &next,const Recipe &r,
    std::span<const c::ReadLease> output,const c::ReadLease &hidden_input) {
  auto actual=converted_actual(r,output,s,from);auto &source=*r.converted;const auto &result=*r.converted_nextn;const bool full=!source.full_checked;
  EXPECT_EQ(gguf_reference::shared::floats(actual(result.prefix.hidden_norm.input)),typed<float>(hidden_input.valid()?hidden_input:s.member(from,kHiddenState)));
  EXPECT_EQ(typed<std::int32_t>(output[r.lm_pending_input]),typed<std::int32_t>(s.member(from,kPending)));
  EXPECT_TRUE(gguf_reference::observe_nextn(*source.parameters.source,result,r.fragment,*r.converted_parameters,*r.pooled_before,
      actual,converted_original(r),converted_expert(r),s.scalar(from,kMlaCursor),r.selected_tail,full,converted_codec(r.cuda)));
  EXPECT_EQ(typed<float>(s.member(next,kHiddenState)),gguf_reference::shared::floats(actual(result.shared_norm.output)));
  EXPECT_EQ(s.scalar(next,kPending),typed<std::int32_t>(output[r.lm_ids]).at(0));
  if(!r.fresh)exact_scatter(s,from,next,kJournal,std::size_t(s.scalar(from,kLagCount)),kHidden,output[r.nx]);
  if(full){source.full_checked=true;source.projection_values+=154880;}++source.head_calls;++source.pending_checks;
  if(::testing::Test::HasFailure())throw std::runtime_error("converted Flash original local numerical checks failed before StateHead publication");
}
template<class StateView,class Version>
void observe_mlx_history(const StateView &s,const Version &from,const Version &next,const Recipe &r,
    std::span<const c::ReadLease> output,const mlx_model::History &history) {
  const auto read_f32=[&](Value value){return typed<float>(output[mlx_port(r,value)]);};
  auto actual=read_f32;mlx_reference::observe_history(*r.mlx->parameters.source,history,actual,!r.mlx->full_checked);
  exact_scatter(s,from,next,kRawKey,std::size_t(r.position%kRing),kIndexDepth,output[mlx_port(r,history.normalized)]);
  exact_scatter(s,from,next,kRawGate,std::size_t(r.position%kRing),kIndexDepth,output[mlx_port(r,history.gate.output)]);
  EXPECT_EQ(s.scalar(next,kIndexCursor),s.scalar(from,kIndexCursor)+1);
  EXPECT_EQ(s.scalar(next,kRingOffset),(s.scalar(from,kRingOffset)+1)%kRing);
  EXPECT_EQ(s.scalar(next,kTailValid),(s.scalar(from,kTailValid)+1)%4);
  if(history.pooled) {
    const auto keys=typed<float>(s.member(next,kRawKey)),gates=typed<float>(s.member(next,kRawGate));
    const auto ape=r.mlx->parameters.source->coefficients(r.mlx->parameters.source->prefix()+"self_attn.indexer.index_kpool_compress_ape");
    if(!history.logits||!history.probabilities||!history.weights||!history.weighted)throw std::runtime_error("MLX pool must observe each actual F32 predecessor");
    const auto logits=read_f32(*history.logits),probabilities=read_f32(*history.probabilities),weights=read_f32(*history.weights),weighted=read_f32(*history.weighted),pooled=read_f32(*history.pooled);
    for(std::size_t d=0;d<std::size_t(kIndexDepth);++d) {
      double maximum=-std::numeric_limits<double>::infinity(),denominator=0;
      for(std::size_t t=0;t<4;++t){const auto row=std::size_t((r.position-3+std::int32_t(t))%kRing);
        EXPECT_EQ(logits[t*kIndexDepth+d],float(double(gates[row*kIndexDepth+d])+ape[t*kIndexDepth+d]));maximum=std::max(maximum,double(logits[t*kIndexDepth+d]));}
      for(std::size_t t=0;t<4;++t)denominator+=std::exp(double(logits[t*kIndexDepth+d])-maximum);
      for(std::size_t t=0;t<4;++t){EXPECT_TRUE(moe_reference::detail::local(probabilities[d*4+t],float(std::exp(double(logits[t*kIndexDepth+d])-maximum)/denominator),"MLX actual pool Softmax",d*4+t));
        EXPECT_EQ(weights[t*kIndexDepth+d],probabilities[d*4+t]);
        const auto row=std::size_t((r.position-3+std::int32_t(t))%kRing);EXPECT_EQ(weighted[t*kIndexDepth+d],float(double(weights[t*kIndexDepth+d])*keys[row*kIndexDepth+d]));}
      EXPECT_TRUE(moe_reference::detail::reduced(pooled[d],moe_reference::detail::sum(4,[&](std::size_t t){return double(weighted[t*kIndexDepth+d]);}),"MLX actual weighted pool reduction",d));
    }
    exact_scatter(s,from,next,kPool,std::size_t(r.position/4),kIndexDepth,output[mlx_port(r,*history.pooled)]);
    EXPECT_EQ(s.scalar(next,kPoolCount),s.scalar(from,kPoolCount)+1);EXPECT_EQ(s.scalar(next,kTailStart),s.scalar(from,kTailStart)+4);
  } else for(const auto entry:{kPool,kPoolCount,kTailStart})same_storage(s.member(next,entry),s.member(from,entry));
}
template<class StateView,class Version>
void observe_mlx_mla(const StateView &s,const Version &from,const Version &next,const Recipe &r,
    std::span<const c::ReadLease> output,const mlx_model::Mla &p,bool full) {
  auto actual=[&](Value value){return typed<float>(output[mlx_port(r,value)]);};
  const auto data=r.mlx->parameters.source;const auto prefix=data->prefix();
  exact_scatter(s,from,next,kCache,std::size_t(s.scalar(from,kMlaCursor)),kLatent,output[mlx_port(r,p.latent)]);
  const auto cursor=s.scalar(next,kMlaCursor);EXPECT_EQ(cursor,s.scalar(from,kMlaCursor)+1);
  const auto chosen=typed<std::int32_t>(s.member(next,kSelection));ASSERT_EQ(chosen.size(),std::size_t(kIds));
  // Independent from the graph extent helper: read the published pool frontier.
  const auto pool_count=r.fresh?s.scalar(next,kPoolCount):512;ASSERT_GE(pool_count,0);
  const auto pools=std::size_t(pool_count),selected_pools=std::min(pools,std::size_t{512});
  const auto tail=r.fresh?(r.position+1)%4:r.selected_tail;
  if(r.fresh){EXPECT_EQ(cursor,r.position+1);EXPECT_EQ(pool_count,(r.position+1)/4);EXPECT_EQ(s.scalar(next,kTailStart),r.position+1-tail);}
  std::vector<float> selected;for(std::size_t i=0;i<selected_pools*4+std::size_t(tail);++i){
    ASSERT_GE(chosen[i],0);ASSERT_LT(chosen[i],cursor);selected.push_back(float(chosen[i]));}
  const auto query=actual(p.absorbed.output),cache=typed<float>(s.member(next,kCache));
  sem::oracle::expect_close(actual(p.attention),glm::attention_reference(query,cache,selected,std::int64_t(selected.size()),kLatent),8.e-5F,4.e-5F);
  if(r.fresh) {
    if(!p.history||!p.index)throw std::runtime_error("MLX fresh query omitted actual indexer history/selection");
    observe_mlx_history(s,from,next,r,output,*p.history);const auto &index=*p.index;
    ASSERT_EQ(index.scored,pools>0);std::vector<std::int32_t> order;
    if(pools>0) {
      mlx_reference::bfloat_projection(*data,index.query,prefix+"self_attn.indexer.wq_b.weight",4096,1536,actual,full);
      mlx_reference::bfloat_projection(*data,index.weights,prefix+"self_attn.indexer.weights_proj.weight",32,4096,actual,full);
      const auto rawpool=typed<float>(s.member(next,kPool));
      const auto pool=std::span(rawpool).first(pools*kIndexDepth);const auto q=actual(index.query.output),dots=actual(index.dots);
      EXPECT_TRUE(mlx_reference::expect_projection(dots,q,32,pools,128,mlx_reference::FloatRows{pool,pools,128},true));
      const auto projected=actual(index.weights.output),scaled=actual(index.weights_scaled),head=actual(index.head_weights),relu=actual(index.relu),scores=actual(index.scores);
      for(std::size_t h=0;h<32;++h){EXPECT_EQ(scaled[h],float(double(projected[h])*float(1./std::sqrt(32.))));EXPECT_EQ(head[h],float(double(scaled[h])*float(1./std::sqrt(128.))));}
      for(std::size_t row=0;row<pools;++row){for(std::size_t h=0;h<32;++h)EXPECT_EQ(relu[h*pools+row],std::max(dots[h*pools+row],0.F));
        EXPECT_TRUE(moe_reference::detail::reduced(scores[row],moe_reference::detail::sum(32,[&](std::size_t h){return double(float(double(relu[h*pools+row])*head[h]));}),"MLX actual F32 index-head reduction",row));}
      order.resize(pools);std::iota(order.begin(),order.end(),0);
      std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return scores[std::size_t(a)]>scores[std::size_t(b)];});order.resize(selected_pools);std::sort(order.begin(),order.end());
      const auto ids=typed<std::int32_t>(output[mlx_port(r,index.pool_ids)]);EXPECT_EQ(ids,order);
    }
    std::vector<std::int32_t> expected;for(const auto pool_id:order)for(std::int32_t i=0;i<4;++i)expected.push_back(pool_id*4+i);
    for(std::int32_t i=0;i<tail;++i)expected.push_back((r.position+1-tail)+i);
    expected.resize(std::size_t(kIds),-1);EXPECT_EQ(chosen,expected);
  } else {
    for(const auto entry:{kRawKey,kRawGate,kPool,kIndexCursor,kPoolCount,kRingOffset,kTailValid,kTailStart})same_storage(s.member(next,entry),s.member(from,entry));
    EXPECT_EQ(chosen,typed<std::int32_t>(s.member(from,kSelection)));EXPECT_EQ(s.scalar(next,kLagCount),s.scalar(from,kLagCount)+1);
  }
}

void observe_mlx(const State &s,const c::StateVersion &from,const c::StateVersion &next,const Recipe &r,
    std::span<const c::ReadLease> output,const c::ReadLease &hidden_input) {
  if(!r.mlx_result)throw std::runtime_error("MLX full NextN observer requires its actual Fragment ports");
  auto actual=[&](Value v){return typed<float>(output[mlx_port(r,v)]);};auto ids=[&](Value v){return typed<std::int32_t>(output[mlx_port(r,v)]);};
  auto &source=*r.mlx;const auto &result=*r.mlx_result;const bool full=!source.full_checked;
  const auto pending=typed<std::int32_t>(output[r.lm_pending_input]);EXPECT_EQ(pending,typed<std::int32_t>(s.member(from,kPending)));
  const auto embedding=source.parameters.source->bytes("model.language_model.embed_tokens.weight");const auto embedded=actual(result.prefix.embedding_norm.input);
  ASSERT_EQ(pending.size(),1U);ASSERT_GE(pending[0],0);ASSERT_LT(pending[0],154880);
  const mlx_reference::BFloat original_embedding{embedding,154880,4096};
  for(std::size_t i=0;i<4096;++i)EXPECT_EQ(embedded[i],original_embedding(std::size_t(pending[0]),i).value);
  EXPECT_EQ(actual(result.prefix.hidden_norm.input),typed<float>(hidden_input.valid()?hidden_input:s.member(from,kHiddenState)));
  mlx_reference::observe_nextn(*source.parameters.source,result,actual,ids,full);observe_mlx_mla(s,from,next,r,output,result.mla,full);
  EXPECT_EQ(typed<float>(s.member(next,kHiddenState)),actual(result.shared_norm.output));EXPECT_EQ(s.scalar(next,kPending),ids(result.pending).at(0));
  if(!r.fresh)exact_scatter(s,from,next,kJournal,std::size_t(s.scalar(from,kLagCount)),kHidden,output[r.nx]);
  if(full){source.full_checked=true;source.projection_values+=154880;}++source.head_calls;++source.pending_checks;
  if(::testing::Test::HasFailure())throw std::runtime_error("MLX original local numerical check failed before StateHead publication");
}
template<class StateView, class Version>
void observe_history(const StateView &s, const Version &from, const Version &next,
                     const Recipe &r, std::span<const c::ReadLease> output,
                     std::span<const float> nx) {
  if(r.converted){observe_converted_history(s,from,next,r,output);return;}
  if(r.mlx){const auto &p=r.mlx_history?*r.mlx_history:r.mlx_result->mla.history.value();observe_mlx_history(s,from,next,r,output,p);return;}
  const auto fp = [&](std::size_t port) { return typed<float>(output[port]); };
  const auto key = fp(r.key), gate_projection = fp(r.gate_projection), gate = fp(r.gate);
  if (r.sparse) {
    const auto &source = *r.sparse->indexer;
    const auto wk = fp(r.wk_accum);
    EXPECT_TRUE(index_reference::expect_accum(wk, index_reference::linear(nx, source.key, kIndexDepth), "original wk BF16 operands -> F32"));
    const auto rounded_wk = typed<std::uint16_t>(output[r.wk_stored]);
    EXPECT_TRUE(index_reference::expect_bfloat_store(rounded_wk, wk, "original wk BF16 store"));
    sem::oracle::expect_close(key, index_reference::layer_norm(original_index::widen(rounded_wk), source.norm, source.bias), 8e-5F, 4e-5F);
    EXPECT_TRUE(index_reference::expect_accum(gate_projection,
        index_reference::linear(nx, source.gate, kIndexDepth), "original compression gate BF16 operands -> F32"));
  } else {
    quantified(r, key, r.key_reference(nx)); quantified(r, gate_projection, r.gate_reference(nx));
  }
  const auto raw_key = typed<std::uint16_t>(output[r.raw_key]), raw_gate = typed<std::uint16_t>(output[r.raw_gate]);
  ASSERT_EQ(raw_key.size(), key.size()); ASSERT_EQ(raw_gate.size(), gate_projection.size());
  for (std::size_t i = 0; i < key.size(); ++i) {
    EXPECT_EQ(raw_key[i], codec::bfloat(key[i]));
    EXPECT_EQ(raw_gate[i], codec::bfloat(gate_projection[i]));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(gate[i]), std::bit_cast<std::uint32_t>(codec::wide(raw_gate[i])));
  }
  exact_scatter(s, from, next, kRawKey, std::size_t(s.scalar(from, kRingOffset)), kIndexDepth, output[r.raw_key], sizeof(std::uint16_t));
  exact_scatter(s, from, next, kRawGate, std::size_t(s.scalar(from, kRingOffset)), kIndexDepth, output[r.raw_gate], sizeof(std::uint16_t));
  if (r.position % 4 == 3) {
    const auto raw = typed<std::uint16_t>(s.member(next, kRawKey)), gates = typed<std::uint16_t>(s.member(next, kRawGate));
    std::vector<float> pooled(kIndexDepth);
    if (r.sparse) {
      std::array<std::int32_t, 4> ring_rows{};
      for (std::size_t t = 0; t < ring_rows.size(); ++t)
        ring_rows[t] = (r.position - 3 + std::int32_t(t)) % std::int32_t(kRing);
      pooled = index_reference::pool_reference(raw, gates, r.sparse->indexer->ape, ring_rows);
    } else {
    for (std::size_t d = 0; d < kIndexDepth; ++d) {
      std::array<double, 4> logits;
      for (std::size_t t = 0; t < 4; ++t) {
        const auto slot = std::size_t((r.position - 3 + std::int32_t(t)) % std::int32_t(kRing));
        logits[t] = codec::wide(gates[slot * kIndexDepth + d]) + double(int((t * kIndexDepth + d) * 7 % 19) - 9) / 16.0;
      }
      const auto maximum = *std::max_element(logits.begin(), logits.end()); double total = 0, result = 0;
      for (auto &value : logits) total += value = std::exp(value - maximum);
      for (std::size_t t = 0; t < 4; ++t) {
        const auto slot = std::size_t((r.position - 3 + std::int32_t(t)) % std::int32_t(kRing));
        result += logits[t] / total * codec::wide(raw[slot * kIndexDepth + d]);
      }
      pooled[d] = float(result);
    }
    }
    sem::oracle::expect_close(fp(r.compressed), pooled, 8e-5F, 4e-5F);
    (void)observe_codec(fp(r.compressed), 1, output, r.pool_codec);
    exact_scatter(s, from, next, kPool, std::size_t(r.position / 4), kIndexDepth, output[r.pool_codec.codes], sizeof(std::uint8_t));
    exact_scatter(s, from, next, kPoolScales, std::size_t(r.position / 4), 1, output[r.pool_codec.scales]);
  } else {
    for (const auto entry : {kPool, kPoolScales, kPoolCount, kTailStart}) same_storage(s.member(next, entry), s.member(from, entry));
  }
  EXPECT_EQ(s.scalar(next, kIndexCursor), r.position + 1);
  EXPECT_EQ(s.scalar(next, kPoolCount), (r.position + 1) / 4);
  EXPECT_EQ(s.scalar(next, kRingOffset), (r.position + 1) % std::int32_t(kRing));
  EXPECT_EQ(s.scalar(next, kTailValid), (r.position + 1) % 4);
  EXPECT_EQ(s.scalar(next, kTailStart), (r.position + 1) / 4 * 4);
}
template<class StateView, class Version>
void observe_attention(const StateView &s, const Version &from, const Version &next,
             const Recipe &r, std::span<const c::ReadLease> output,
             std::span<const float> nx, std::int32_t producer_position) {
  const auto fp = [&](std::size_t port) { return typed<float>(output[port]); };
  const auto qr = typed<float>(s.member(next, kQueryState));
  if (r.sparse) {
    EXPECT_EQ(qr, original_index::widen(typed<std::uint16_t>(output[r.sparse_query])));
  } else quantified(r, qr, r.query_reference(nx));
  if(has_original_attention(r)) {
    const auto &ports=r.attention_ports;const auto &data=*r.sparse->attention->source;
    const bool full=!r.sparse->attention->full_checked;
    const auto qb=typed<std::uint16_t>(output[ports.qb_stored]);
    const auto kva=typed<std::uint16_t>(output[ports.kva_stored]);
    EXPECT_TRUE(attention_reference::expect_fp8_projection(fp(ports.qb_raw),typed<std::uint16_t>(output[r.sparse_query]),
        data,original_attention::Parameter::kQueryProjection,full,r.sparse->arithmetic));
    EXPECT_TRUE(index_reference::expect_bfloat_store(qb,fp(ports.qb_raw),"original query_b actual BF16 store"));
    EXPECT_TRUE(attention_reference::expect_absorbed_query(typed<std::uint16_t>(output[ports.absorbed_stored]),qb,data,full));
    EXPECT_TRUE(lm_reference::expect_widen(fp(r.query),typed<std::uint16_t>(output[ports.absorbed_stored])));
    EXPECT_TRUE(attention_reference::expect_fp8_projection(fp(ports.kva_raw),typed<std::uint16_t>(output[r.sparse_hidden]),
        data,original_attention::Parameter::kKvProjection,full,r.sparse->arithmetic));
    EXPECT_TRUE(index_reference::expect_bfloat_store(kva,fp(ports.kva_raw),"original kv_a actual BF16 store"));
    sem::oracle::expect_close(fp(ports.kvnorm_raw),attention_reference::norm(original_index::widen(kva),
        data.norm(original_attention::Parameter::kKvNorm),std::size_t(kLatent)),8e-5F,4e-5F);
    EXPECT_TRUE(index_reference::expect_bfloat_store(typed<std::uint16_t>(output[ports.kvnorm_stored]),fp(ports.kvnorm_raw),"original KV latent BF16 norm store"));
    EXPECT_TRUE(lm_reference::expect_widen(fp(r.latent),typed<std::uint16_t>(output[ports.kvnorm_stored])));
  } else quantified(r, fp(r.latent), r.latent_reference(nx));
  exact_scatter(s, from, next, kCache, std::size_t(s.scalar(from, kMlaCursor)), kLatent, output[r.latent]);
  EXPECT_EQ(s.scalar(next, kMlaCursor), s.scalar(from, kMlaCursor) + 1);
  const auto ids = typed<std::int32_t>(s.member(next, kSelection));
  ASSERT_EQ(ids.size(), std::size_t(kIds));
  const auto physical_pools=r.fresh?s.scalar(next,kPoolCount):512;ASSERT_GE(physical_pools,0);
  const auto pool_count=std::size_t(physical_pools),selected_pools=std::min(pool_count,std::size_t{512});
  const auto live=selected_pools*4+std::size_t(r.selected_tail);
  if(r.fresh){EXPECT_EQ(pool_count,(r.position+1)/4);EXPECT_EQ(s.scalar(next,kTailStart),r.position+1-r.selected_tail);}
  for (std::size_t i=selected_pools*4;i<ids.size();++i)
    EXPECT_EQ(ids[i],i<live?producer_position+1-r.selected_tail+std::int32_t(i-selected_pools*4):-1);
  if (r.fresh) {
    observe_history(s, from, next, r, output, nx);
    if(pool_count>0) {
    if (r.sparse) {
      const auto wq = fp(r.index_query_accum);
      EXPECT_TRUE(index_reference::expect_accum(wq,
          index_reference::linear(qr, r.sparse->indexer->query, kIndexHeads * kIndexDepth), "original wq_b BF16 operands -> F32"));
      const auto rounded = typed<std::uint16_t>(output[r.index_query_stored]);
      EXPECT_TRUE(index_reference::expect_bfloat_store(rounded, wq, "original wq_b BF16 store"));
      EXPECT_EQ(fp(r.index_query), original_index::widen(rounded));
      EXPECT_TRUE(index_reference::expect_accum(fp(r.score_weight),
          index_reference::linear(nx, r.sparse->indexer->weights, kIndexHeads), "original widened-F32 weights_proj"));
    } else {
      quantified(r, fp(r.index_query), r.index_query_reference(qr));
      sem::oracle::expect_close(fp(r.score_weight), r.score_weight_reference(nx), 8e-5F, 4e-5F);
    }
    const auto iq = fp(r.index_query), weights = fp(r.score_weight);
    const auto query_encoded = observe_codec(iq, kIndexHeads, output, r.query_codec);
    const auto codes = typed<std::uint8_t>(s.member(next, kPool));
    const auto scales = typed<float>(s.member(next, kPoolScales));
    const auto expected_scores = codec::score_reference(codes, scales, query_encoded, weights, std::int64_t(pool_count));
    const auto scores = fp(r.scores); sem::oracle::expect_close(scores, expected_scores, 8e-5F, 4e-5F);
    std::vector<bool> picked(pool_count, false); float smallest = std::numeric_limits<float>::infinity();
    for (std::size_t t = 0; t < selected_pools; ++t) {
      const auto pool_id = ids[t * 4] / 4;
      ASSERT_GE(pool_id, 0); ASSERT_LT(std::size_t(pool_id), pool_count); EXPECT_FALSE(picked[std::size_t(pool_id)]);
      picked[std::size_t(pool_id)] = true; smallest = std::min(smallest, scores[std::size_t(pool_id)]);
      for (std::size_t j = 0; j < 4; ++j) EXPECT_EQ(ids[t * 4 + j], pool_id * 4 + std::int32_t(j));
    }
    for (std::size_t p = 0; p < picked.size(); ++p) if (!picked[p]) EXPECT_GE(smallest, scores[p]);
    }
  } else {
    for (const auto entry : {kRawKey, kRawGate, kPool, kPoolScales, kIndexCursor, kPoolCount, kRingOffset, kTailValid, kTailStart})
      same_storage(s.member(next, entry), s.member(from, entry));
    EXPECT_EQ(read(s.member(next, kSelection)), read(s.member(from, kSelection)));
    exact_scatter(s, from, next, kJournal, std::size_t(s.scalar(from, kLagCount)), kHidden, output[r.nx]);
    EXPECT_EQ(s.scalar(next, kLagCount), s.scalar(from, kLagCount) + 1);
  }
  const auto query = fp(r.query), cache = typed<float>(s.member(next, kCache));
  if(!has_original_attention(r))quantified(r, query, r.absorbed_query_reference(qr));
  std::vector<float> selected(live);
  for (std::size_t i = 0; i < selected.size(); ++i) selected[i] = float(ids[i]);
  const auto attention = fp(r.attention);
  if(has_original_attention(r)) {
    const auto &ports=r.attention_ports;auto &observed=*r.sparse->attention;const auto &data=*observed.source;
    std::vector<float> chosen(live*std::size_t(kLatent));
    for(std::size_t i=0;i<chosen.size()/std::size_t(kLatent);++i) {
      if(ids[i]<0||ids[i]>=s.scalar(next,kMlaCursor))throw std::runtime_error("original MLA selected IDs require valid prior/current rows");
      std::copy_n(cache.begin()+std::size_t(ids[i])*std::size_t(kLatent),std::size_t(kLatent),chosen.begin()+i*std::size_t(kLatent));
    }
    std::size_t non_bfloat=0;
    for(const float value:std::span(cache).first(std::size_t(s.scalar(next,kMlaCursor))*std::size_t(kLatent)))
      if(!std::isfinite(value)||(std::bit_cast<std::uint32_t>(value)&0xffffu)!=0u)++non_bfloat;
    EXPECT_EQ(non_bfloat,0u)<<"F32 cache carrier stores finite actual BF16 widenings only";
    const auto absorbed=typed<std::uint16_t>(output[ports.absorbed_stored]);
    sem::oracle::expect_close(attention,attention_reference::attention(absorbed,chosen,live),8e-5F,4e-5F);
    const auto stored_attention=typed<std::uint16_t>(output[ports.attention_stored]);
    EXPECT_TRUE(index_reference::expect_bfloat_store(stored_attention,attention,"original compact attention BF16 output"));
    const auto value=typed<std::uint16_t>(output[ports.value_stored]);const bool full=!observed.full_checked;
    EXPECT_TRUE(attention_reference::expect_value_projection(value,stored_attention,data,full));
    EXPECT_TRUE(attention_reference::expect_fp8_projection(fp(ports.output_raw),value,data,original_attention::Parameter::kOutput,full,r.sparse->arithmetic));
    const auto stored_output=typed<std::uint16_t>(output[ports.output_stored]);
    EXPECT_TRUE(index_reference::expect_bfloat_store(stored_output,fp(ports.output_raw),"original O projection actual BF16 store"));
    EXPECT_TRUE(lm_reference::expect_widen(fp(r.output),stored_output));
    if(full){observed.full_checked=true;observed.projection_values+=16384+512+4096+32768+16384;}
    ++observed.attention_calls;
  } else {
    sem::oracle::expect_close(attention, glm::attention_reference(query, cache, selected, std::int64_t(live), kLatent), 8e-5F, 4e-5F);
    const auto projected = fp(r.output); quantified(r, projected, r.output_reference(attention));
  }
}

void observe_original_moe(const Recipe &r,std::span<const c::ReadLease> outputs) {
  if(!r.original_moe||!has_original_moe(r))throw std::runtime_error("original complete MoE observations missing");
  const auto &ports=r.original_moe->ports;auto &observed=*r.moe;const auto &data=*observed.roles.source;
  const auto port=[&](Value value) {
    const auto found=std::find_if(r.moe_ports.begin(),r.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==r.moe_ports.end())throw std::runtime_error("actual original MoE diagnostic not published");return found->second;
  };
  const auto fp=[&](Value value){return typed<float>(outputs[port(value)]);};
  const auto bf=[&](Value value){return typed<std::uint16_t>(outputs[port(value)]);};
  const auto integers=[&](Value value){return typed<std::int32_t>(outputs[port(value)]);};
  const auto input=bf(ports.input);const auto global=integers(ports.ids),local=integers(ports.local_ids),map=integers(ports.map);
  if(r.moe_input>=outputs.size()||r.moe_output>=outputs.size())
    throw std::runtime_error("MoE component actual input/output observations missing");
  EXPECT_EQ(input,typed<std::uint16_t>(outputs[r.moe_input]));
  const auto logits=fp(ports.logits),scores=fp(ports.scores),selection=fp(ports.selection),selected=fp(ports.selected_scores);
  const auto sum=fp(ports.sum),denominator=fp(ports.denominator),normalized=fp(ports.normalized),weights=fp(ports.weights);
  EXPECT_TRUE(moe_reference::expect_router(logits,input,data.router_bytes()));
  EXPECT_TRUE(moe_reference::expect_routing({logits,scores,selection,selected,sum,denominator,normalized,weights,global},data.bias_bytes()));
  if(observed.selections.empty())throw std::runtime_error("actual completed expert generation identity missing");
  const auto &resident=observed.selections.back();
  EXPECT_EQ(map,std::vector<std::int32_t>(resident.map.begin(),resident.map.end()));
  EXPECT_TRUE(moe_reference::expect_mapping(global,map,local,resident.globals));
  const bool full=!observed.full_checked;
  const bool mxfp4=data.weight_storage()==original_moe::WeightStorage::kMxfp4Group32;
  const auto matrix=[&](original_moe::Leg leg,std::optional<std::int32_t> expert={}) {
    const auto [n,k]=original_moe::geometry(leg);
    return moe_reference::Fp8Matrix{expert?data.expert_bytes(leg,*expert,false):data.shared_bytes(leg,false),
        expert?data.expert_bytes(leg,*expert,true):data.shared_bytes(leg,true),std::size_t(n),std::size_t(k)};
  };
  const auto branch=[&](const moe_fragment::ProjectionPorts &values,bool routed) {
    const auto stored_gate=bf(values.gate_stored),stored_up=bf(values.up_stored),activated=bf(values.activation_stored);
    if(mxfp4) {
      if(values.gate_raw||values.up_raw||values.down_raw||!values.down_stored)
        throw std::runtime_error("original MXFP4 must expose actual BF16 stores without invented rawF32");
      const auto down=bf(*values.down_stored);
      const auto original=[&](original_moe::Leg leg,std::optional<std::int32_t> expert={}) {
        const auto [n,k]=original_moe::geometry(leg);
        return glm_mxfp4_moe_reference::Matrix{expert?data.expert_bytes(leg,*expert,false):data.shared_bytes(leg,false),
            expert?data.expert_bytes(leg,*expert,true):data.shared_bytes(leg,true),std::size_t(n),std::size_t(k)};
      };
      for(const auto leg:{original_moe::Leg::kGate,original_moe::Leg::kUp,original_moe::Leg::kDown}) {
        const auto &stored=leg==original_moe::Leg::kGate?stored_gate:leg==original_moe::Leg::kUp?stored_up:down;
        const auto &activation=leg==original_moe::Leg::kDown?activated:input;
        if(routed) {
          EXPECT_TRUE(glm_mxfp4_moe_reference::expect_selected_projection(stored,activation,global,
              [&](std::int32_t id){return original(leg,id);},"original selected MXFP4 BF16 return",full));
        } else EXPECT_TRUE(glm_mxfp4_moe_reference::expect_projection(stored,activation,original(leg),
            "original shared MXFP4 BF16 return",full));
      }
      if(full)observed.projection_values+=stored_gate.size()+stored_up.size()+down.size();
    } else {
      if(!values.gate_raw||!values.up_raw||!values.down_raw||values.down_stored)
        throw std::runtime_error("original E4 requires its actual F32 projection predecessors");
      const auto gate=fp(*values.gate_raw),up=fp(*values.up_raw),down=fp(*values.down_raw);
      if(routed) {
      for(const auto leg:{original_moe::Leg::kGate,original_moe::Leg::kUp,original_moe::Leg::kDown}) {
        const auto &raw=leg==original_moe::Leg::kGate?gate:leg==original_moe::Leg::kUp?up:down;
        const auto &activation=leg==original_moe::Leg::kDown?activated:input;
        EXPECT_TRUE(moe_reference::expect_selected_projection(raw,activation,global,[&](std::int32_t id){return matrix(leg,id);},
            leg==original_moe::Leg::kGate?"original expert gate":leg==original_moe::Leg::kUp?"original expert up":"original expert down",full,original_moe::kFlashGeometry,observed.arithmetic));
      }
      } else {
      EXPECT_TRUE(moe_reference::expect_fp8_projection(gate,input,matrix(original_moe::Leg::kGate),"original shared gate",full,observed.arithmetic));
      EXPECT_TRUE(moe_reference::expect_fp8_projection(up,input,matrix(original_moe::Leg::kUp),"original shared up",full,observed.arithmetic));
      EXPECT_TRUE(moe_reference::expect_fp8_projection(down,activated,matrix(original_moe::Leg::kDown),"original shared down",full,observed.arithmetic));
      }
      EXPECT_TRUE(index_reference::expect_bfloat_store(stored_gate,gate,"original gate BF16 store"));
      EXPECT_TRUE(index_reference::expect_bfloat_store(stored_up,up,"original up BF16 store"));
      if(full)observed.projection_values+=gate.size()+up.size()+down.size();
    }
    const auto raw_activation=fp(values.activation_raw);
    EXPECT_TRUE(moe_reference::expect_activation(fp(values.gate_clipped),fp(values.up_clipped),raw_activation,stored_gate,stored_up));
    EXPECT_TRUE(index_reference::expect_bfloat_store(activated,raw_activation,"original clipped activation BF16 store"));
  };
  branch(ports.routed,true);branch(ports.shared,false);
  const auto weighted=fp(ports.weighted_raw);const auto weighted_store=bf(ports.weighted_stored);
  const auto down_predecessor=mxfp4?original_index::widen(bf(ports.routed.down_stored.value())):fp(ports.routed.down_raw.value());
  EXPECT_TRUE(moe_reference::expect_weighted_down(weighted,down_predecessor,weights));
  EXPECT_TRUE(index_reference::expect_bfloat_store(weighted_store,weighted,"original post-down weighted BF16 store"));
  const auto raw_sum=fp(ports.routed_sum_raw);const auto routed_store=bf(ports.routed_sum_stored);
  EXPECT_TRUE(moe_reference::expect_routed_sum(raw_sum,weighted_store));
  EXPECT_TRUE(index_reference::expect_bfloat_store(routed_store,raw_sum,"original routed sum BF16 return"));
  const auto shared_store=bf(ports.shared_stored);const auto merged=fp(ports.merge_raw);const auto stored_merge=bf(ports.merge_stored);
  if(mxfp4)EXPECT_EQ(shared_store,bf(ports.shared.down_stored.value()));
  else EXPECT_TRUE(index_reference::expect_bfloat_store(shared_store,fp(ports.shared.down_raw.value()),"original shared down BF16 return"));
  EXPECT_TRUE(lm_reference::expect_sum(merged,original_index::widen(routed_store),original_index::widen(shared_store)));
  EXPECT_TRUE(index_reference::expect_bfloat_store(stored_merge,merged,"original shared/routed BF16 merge"));
  EXPECT_TRUE(lm_reference::expect_widen(typed<float>(outputs[r.moe_output]),stored_merge));
  if(full){observed.projection_values+=logits.size();observed.full_checked=true;}
}

void observe(const State &s, const c::StateVersion &from, const c::StateVersion &next,
             const Recipe &r, std::span<const c::ReadLease> output,
             const c::ReadLease &hidden_input = {}) {
  if(r.converted){observe_converted(s,from,next,r,output,hidden_input);return;}
  if(r.mlx){observe_mlx(s,from,next,r,output,hidden_input);return;}
  const auto fp = [&](std::size_t port) { return typed<float>(output[port]); };
  const auto hidden = typed<float>(hidden_input.valid() ? hidden_input : s.member(from, kHiddenState));
  const auto embedding = fp(r.embedding), eh = fp(r.eh), nx = fp(r.nx);
  if(has_shared_lm(r)) {
    const auto pending=typed<std::int32_t>(output[r.lm_pending_input]);
    EXPECT_TRUE(lm_reference::expect_feedback(pending,typed<std::int32_t>(s.member(from,kPending))));
    EXPECT_TRUE(lm_reference::expect_embedding(typed<std::uint16_t>(output[r.original_embedding]),
        r.original->vocabulary->source->bytes(shared_lm::Role::kEmbedding),pending,
        std::size_t(r.original->vocabulary->source->geometry.vocabulary),std::size_t(kHidden)));
    EXPECT_TRUE(lm_reference::expect_widen(embedding,typed<std::uint16_t>(output[r.original_embedding])));
  } else EXPECT_EQ(embedding, embedding_values(s.scalar(from, kPending)));
  if (r.original) {
    const auto actual_hidden = typed<std::uint16_t>(output[r.original_hidden]);
    const auto actual_embedding = typed<std::uint16_t>(output[r.original_embedding]);
    EXPECT_TRUE(index_reference::expect_bfloat_store(actual_hidden, hidden, "declared original-prefix target-hidden boundary"));
    EXPECT_TRUE(index_reference::expect_bfloat_store(actual_embedding, embedding, "declared original-prefix embedding boundary"));
    std::array<std::vector<std::uint16_t>, 7> actual;
    for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = typed<std::uint16_t>(output[r.original_prefix_ports[i]]);
    EXPECT_TRUE(original_prefix::observe_prefix(*r.original->prefix, actual_hidden, actual_embedding, actual,
        r.cuda,r.original->arithmetic));
    EXPECT_EQ(eh, original_index::widen(actual[3])); EXPECT_EQ(nx, original_index::widen(actual[4]));
  } else {
    quantified(r, eh, r.input_reference(hidden, embedding)[3]);
    sem::oracle::expect_close(nx, sem::oracle::rms_norm_rows(eh, r.input_scale, kHidden, glm::kNormEpsilon), 8e-5F, 4e-5F);
  }
  const auto producer_position = r.fresh ? r.position : std::int32_t(s.facts(from).producer_position);
  observe_attention(s, from, next, r, output, nx, producer_position);
  const auto projected = fp(r.output);
  if (r.fresh) {
    same_storage(s.member(next, kJournal), s.member(from, kJournal));
    same_storage(s.member(next, kLagCount), s.member(from, kLagCount));
  }
  auto residual = eh; for (std::size_t i = 0; i < residual.size(); ++i) residual[i] += projected[i];
  const auto ffn_input = fp(r.ffn_input), ffn_output = fp(r.ffn_output);
  if(has_original_attention(r)) {
    const auto &ports=r.attention_ports;
    const auto actual_sum=fp(ports.residual_sum);const auto stored_residual=typed<std::uint16_t>(output[ports.residual_stored]);
    EXPECT_TRUE(lm_reference::expect_sum(actual_sum,eh,projected));
    EXPECT_TRUE(index_reference::expect_bfloat_store(stored_residual,actual_sum,"original RMS actual BF16 residual return"));
    EXPECT_TRUE(lm_reference::expect_widen(fp(r.residual),stored_residual));
    sem::oracle::expect_close(fp(ports.postnorm_raw),attention_reference::norm(actual_sum,
        r.sparse->attention->source->norm(original_attention::Parameter::kPostNorm),std::size_t(kHidden)),8e-5F,4e-5F);
    const auto stored_norm=typed<std::uint16_t>(output[ports.postnorm_stored]);
    EXPECT_TRUE(index_reference::expect_bfloat_store(stored_norm,fp(ports.postnorm_raw),"original postattention norm BF16 store"));
    EXPECT_TRUE(lm_reference::expect_widen(ffn_input,stored_norm));
  } else {
    sem::oracle::expect_close(fp(r.residual), residual, 8e-5F, 4e-5F);
    sem::oracle::expect_close(ffn_input, sem::oracle::rms_norm_rows(fp(r.residual), r.ffn_scale, kHidden, glm::kNormEpsilon), 8e-5F, 4e-5F);
  }
  if(has_original_moe(r))observe_original_moe(r,output);
  else quantified(r, ffn_output, r.moe_reference(ffn_input)[5]);
  const auto actual_residual = fp(r.residual);
  for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = actual_residual[i] + ffn_output[i];
  if(has_shared_lm(r)) {
    const auto &source=*r.original->vocabulary->source;
    const auto actual_combined=fp(r.lm_combined),actual_norm=fp(r.lm_norm_raw);
    const auto stored=typed<std::uint16_t>(output[r.lm_norm_stored]);
    const auto logits=fp(r.lm_logits);const auto predicted=typed<std::int32_t>(output[r.lm_ids]);
    EXPECT_TRUE(lm_reference::expect_sum(actual_combined,actual_residual,ffn_output));
    EXPECT_TRUE(lm_reference::expect_norm(actual_norm,stored,actual_combined,source.norm(),std::size_t(kHidden),source.geometry.epsilon));
    EXPECT_TRUE(lm_reference::expect_widen(typed<float>(s.member(next,kHiddenState)),stored));
    EXPECT_TRUE(lm_reference::expect_greedy(logits,predicted,std::size_t(source.geometry.vocabulary),fp(r.lm_top_values)));
    auto &observed=*r.original->lm_observations;
    const bool full=!observed.full_checked;
    EXPECT_TRUE(lm_reference::expect_logits(logits,stored,source.bytes(shared_lm::Role::kHead),
        std::size_t(source.geometry.vocabulary),std::size_t(kHidden),full,predicted));
    if(full){observed.full_checked=true;observed.full_values+=logits.size();}
    EXPECT_TRUE(lm_reference::expect_feedback(typed<std::int32_t>(s.member(next,kPending)),predicted));
    ++observed.head_calls;++observed.pending_checks;
    std::printf("GLM original fullV pending=%d -> actual greedy=%d logits=%zu; next input reads StateVersion pending lease\n",
        s.scalar(from,kPending),predicted.at(0),logits.size());
  } else sem::oracle::expect_close(typed<float>(s.member(next, kHiddenState)),
      sem::oracle::rms_norm_rows(residual, r.head_scale, kHidden, glm::kNormEpsilon), 8e-5F, 4e-5F);
}

c::StateVersion execute(State &s, Program &p, c::Queue &queue, const c::StorageOwner &owner,
                        const c::StateVersion &from, std::int32_t pending,
                        c::ReadLease *retained_nx = nullptr,
                        std::span<const c::StateMember> extra_members = {},
                        const c::ReadLease &hidden_input = {},
                        c::ReadLease *retained_logits = nullptr) {
  if (p.recipe.catchup) throw std::runtime_error("catchup requires its explicit between-iteration consumer");
  if (!s.enabled || (!p.recipe.fresh && !s.qualifies(from)))
    throw std::runtime_error("consumer refused index carry outside its iteration before prepare");
  const auto position = p.recipe.position;
  if (p.recipe.fresh && (s.scalar(from, kMlaCursor) != position ||
      s.scalar(from, kIndexCursor) != position || s.scalar(from, kPoolCount) != position / 4 ||
      s.scalar(from, kRingOffset) != position % std::int32_t(kRing) || s.scalar(from, kTailValid) != position % 4 ||
      s.scalar(from, kTailStart) != position - position % 4 || s.scalar(from, kLagCount) != 0))
    throw std::runtime_error("fresh finite preparation does not match actual pool/cursor state");
  if (!p.recipe.fresh && ((s.facts(from).producer_position + 1) % 4 != std::uint64_t(p.recipe.selected_tail) ||
      s.scalar(from, kLagCount) < 0 || s.scalar(from, kLagCount) >= kJournalRows))
    throw std::runtime_error("skip carry shape or accepted-feature journal exceeds its prepared domain");
  std::vector<c::ReadLease> arguments;
  for(const auto i:p.recipe.state_entries)arguments.push_back(s.member(from,i));
  if(hidden_input.valid()){const auto hidden=std::find(p.recipe.state_entries.begin(),p.recipe.state_entries.end(),kHiddenState);
    if(hidden==p.recipe.state_entries.end())throw std::runtime_error("hidden input has no actual state entry");arguments[std::size_t(hidden-p.recipe.state_entries.begin())]=hidden_input;}
  if(!has_shared_lm(p.recipe))arguments.push_back(upload(owner, sem::bytes(std::vector{pending})));
  const auto before = image(s, from);
  auto outputs = p.execute(queue, arguments);
  if (retained_nx) *retained_nx = outputs[p.recipe.nx];
  if (retained_logits) {
    if (!has_shared_lm(p.recipe)) throw std::runtime_error("random proposal requires the actual full NextN logits");
    *retained_logits = outputs[p.recipe.lm_logits];
  }
  auto next = s.stage(from, p.recipe, outputs, extra_members);
  observe(s, from, next, p.recipe, outputs, hidden_input);
  EXPECT_EQ(image(s, from), before);
  if(has_shared_lm(p.recipe))EXPECT_EQ(s.scalar(next,kPending),typed<std::int32_t>(outputs[p.recipe.lm_ids]).at(0));
  else EXPECT_EQ(s.scalar(next, kPending), pending);
  for(const auto i:s.resources)
    if(p.recipe.state_ports[i] != std::numeric_limits<std::size_t>::max())
      same_storage(s.member(next, i), outputs[p.recipe.state_ports[i]]);
  return next;
}

c::StateVersion catch_up(State &s, Program &p, c::Queue &queue,
                         const c::StateHead &accepted_head, const c::ReadLease &direct_nx = {}) {
  // Acceptance is the consumer's explicit head publication, not an arbitrary
  // readable speculative version nominated just because its lag count fits.
  const auto from = take(accepted_head.current(), "nominated accepted catchup head");
  if (!p.recipe.catchup || s.enabled || s.facts(from).epoch != s.epoch)
    throw std::runtime_error("catchup requires the sealed accepted epoch before a new iteration");
  const auto position = p.recipe.position;
  const auto lag=s.scalar(from,kLagCount);
  if (lag<1 || lag>kJournalRows || s.facts(from).reuse_count!=std::uint64_t(lag+p.recipe.journal_slot) || s.scalar(from, kMlaCursor) != position + lag ||
      s.scalar(from, kIndexCursor) != position || s.scalar(from, kPoolCount) != position / 4 ||
      s.scalar(from, kRingOffset) != position % std::int32_t(kRing) || s.scalar(from, kTailValid) != position % 4 ||
      s.scalar(from, kTailStart) != position - position % 4)
    throw std::runtime_error("accepted catchup journal does not match the finite producer domain");
  if (p.recipe.direct_catchup != direct_nx.valid())
    throw std::runtime_error("independent direct catchup requires its actual producer lease");
  std::vector<c::ReadLease> arguments;
  for(const auto i:p.recipe.state_entries)arguments.push_back(s.member(from,i));
  if (p.recipe.direct_catchup) arguments.push_back(direct_nx);
  const auto before = image(s, from);
  auto outputs = p.execute(queue, arguments);
  auto next = s.stage(from, p.recipe, outputs);
  const auto actual_nx = typed<float>(outputs[p.recipe.nx]);
  const auto journal = typed<float>(s.member(from, kJournal));
  EXPECT_TRUE(std::equal(actual_nx.begin(), actual_nx.end(), journal.begin()+std::ptrdiff_t(p.recipe.journal_slot*kHidden)));
  if (direct_nx.valid()) EXPECT_EQ(actual_nx, typed<float>(direct_nx));
  observe_history(s, from, next, p.recipe, outputs, actual_nx);
  EXPECT_EQ(s.scalar(next, kLagCount), lag-1);
  EXPECT_EQ(read(s.member(next, kEpoch)), read(s.member(from, kEpoch))) << "catchup does not publish a new seed";
  for(const auto i:s.resources) {
    if(p.recipe.state_ports[i] == std::numeric_limits<std::size_t>::max())
      same_storage(s.member(next, i), s.member(from, i));
    else same_storage(s.member(next, i), outputs[p.recipe.state_ports[i]]);
  }
  EXPECT_EQ(image(s, from), before);
  return next;
}

// Engine policy can correct a rejected draft's next token. Even this small
// publication comes from an ordinary actual Graph output; no observed result
// is converted into a host-created replacement lease.
sem::Fragment pending_policy() {
  sem::Fragment f{.profile = kModel.profile, .region = 0x474c4d50454e44ULL};
  const auto input = f.input(sem::dense({1}, c::DataType::kInt32));
  f.outputs = {f.one(c::Operation(c::Copy{}), {input})}; return f;
}

// The actual draft-extend producer already wrote its epoch, position, full
// I32 selection and normalized hidden in one StateVersion. Selection of the
// last accepted row transfers those SAME leases to the decode loop; it does
// not manufacture a new seed by relabelling an old selection or hidden.
c::StateVersion activate_extend_seed(State &s, const c::StateHead &accepted_head,
                                    const c::StateVersion &producer,
                                    std::int32_t selected_position) {
  const auto accepted = take(accepted_head.current(), "last accepted draft-extend seed");
  const auto facts = s.facts(accepted), source = s.facts(producer);
  if (s.enabled || selected_position < 0 || facts.epoch != s.epoch || source.epoch != s.epoch ||
      facts.producer_position != std::uint64_t(selected_position) ||
      source.producer_position != facts.producer_position ||
      facts.reuse_count != 0 || source.reuse_count != 0 ||
      s.scalar(accepted, kMlaCursor) != selected_position + 1 ||
      s.scalar(accepted, kIndexCursor) != selected_position + 1 ||
      s.scalar(accepted, kLagCount) != 0)
    throw std::runtime_error("draft-extend seed is not the actual fresh last-accepted frontier of this epoch");
  for(const auto entry:s.resources) {
    if(entry==kPending) continue; // A target-policy bonus token may differ.
    const auto a = s.member(accepted, entry), b = s.member(producer, entry);
    const auto av = take(a.view(), "accepted seed member"), bv = take(b.view(), "actual seed producer member");
    if (av.resource() != bv.resource() || av.native_handle() != bv.native_handle() ||
        av.offset() != bv.offset() || av.size() != bv.size() || a.generation() != b.generation())
      throw std::runtime_error("accepted draft-extend seed spliced a different producer resource");
  }
  s.enabled = true;
  return accepted;
}

void run_seed_handoff(c::Context &context, const c::StorageOwner &owner, bool cuda,
                      std::shared_ptr<const OriginalNextn> original = {}) {
  // Fixed06992 eagle_worker_v2:293-302,936-990,1094-1103,1152-1153:
  // the seed producer is NextN draft-extend at the last ACCEPTED row, not
  // target43 and not draft-decode step0. These are bounded sequential M1
  // draft-extend rows, not a claim of a batched vendor extend kernel.
  const auto extend_recipe = [original](std::int32_t position) {
    auto r = build(true, position, 0, original); r.draft_extend = true;
    r.fragment.region = 0x474c4d455854454eULL;
    return r;
  };
  auto row_a_owner=Program::create(extend_recipe(2054), context, owner, cuda);
  auto &row_a=*row_a_owner;
  auto row_b_owner=Program::create(extend_recipe(2055), context, owner, cuda);
  auto &row_b=*row_b_owner;
  auto partial_next_owner=Program::create(extend_recipe(2056), context, owner, cuda);
  auto &partial_next=*partial_next_owner;
  auto full_next_owner=Program::create(extend_recipe(2057), context, owner, cuda);
  auto &full_next=*full_next_owner;
  auto tail3_owner=Program::create(build(false, 0, 3, original), context, owner, cuda);
  auto &tail3=*tail3_owner;
  auto tail0_owner=Program::create(build(false, 0, 0, original), context, owner, cuda);
  auto &tail0=*tail0_owner;
  auto tail1_owner=Program::create(build(false, 0, 1, original), context, owner, cuda);
  auto &tail1=*tail1_owner;
  auto tail2_owner=Program::create(build(false, 0, 2, original), context, owner, cuda);
  auto &tail2=*tail2_owner;
  auto partial_catchup_owner=Program::create(build_catchup(false, 2055, original), context, owner, cuda);
  auto &partial_catchup=*partial_catchup_owner;
  auto full_catchup_owner=Program::create(build_catchup(false, 2056, original), context, owner, cuda);
  auto &full_catchup=*full_catchup_owner;
  auto partial_direct_owner=Program::create(build_catchup(true, 2055, original), context, owner, cuda);
  auto &partial_direct=*partial_direct_owner;
  auto full_direct_owner=Program::create(build_catchup(true, 2056, original), context, owner, cuda);
  auto &full_direct=*full_direct_owner;
  Recipe policy_recipe; policy_recipe.fragment = pending_policy();
  auto policy_owner=Program::create(std::move(policy_recipe), context, owner, cuda);
  auto &policy=*policy_owner;
  auto queue = take(context.create_queue(), "draft-extend seed handoff queue");
  auto initializer_owner=Program::create(initial_pool_recipe(), context, owner, cuda, 2);
  auto &initializer=*initializer_owner;
  const std::array programs{&initializer,&row_a,&row_b,&partial_next,&full_next,&tail3,&tail0,&tail1,&tail2,
      &partial_catchup,&full_catchup,&partial_direct,&full_direct,&policy};
  std::size_t state_budget=128u<<20,pool_backing_bound=0,initial_backing_bound=0,metadata_backing_bound=0,peak_backing=0;
  if(original&&original->vocabulary) {
    const auto add=[](std::size_t &total,std::size_t amount) {
      if(amount>std::numeric_limits<std::size_t>::max()-total)throw std::runtime_error("shared LM state backing bound overflow");
      total+=amount;
    };
    // State leases retain their complete output-pool root, including diagnostic
    // logits and every preallocated generation. Count each actual Program pool
    // once, not its individual ports or a second per-version multiplier.
    for(const auto *program:programs)add(pool_backing_bound,program->output_stats().capacity_bytes);
    const auto rounded=[](std::size_t bytes,std::size_t alignment) {
      if(bytes>std::numeric_limits<std::size_t>::max()-(alignment-1))throw std::runtime_error("state backing alignment overflow");
      return (bytes+alignment-1)/alignment*alignment;
    };
    const auto initial_descriptions=descriptions();
    for(std::size_t entry=0;entry<kResources;++entry)
      if(entry!=kPool&&entry!=kPoolScales)add(initial_backing_bound,rounded(
          take(initial_descriptions[entry].required_bytes(),"initial independent state backing extent"),256));
    // Initial pool/scales already borrow the initializer's one counted root.
    // Independent uploads request alignment256; include possible tail padding.
    // Every metadata payload is Epoch with backing_alignment64. Reserve at
    // most the existing retained-record capacity, without changing its limit.
    metadata_backing_bound=std::size_t(State::kMaxRetainedBlocks)*rounded(sizeof(Epoch),64);
    state_budget=0;add(state_budget,pool_backing_bound);add(state_budget,initial_backing_bound);add(state_budget,metadata_backing_bound);
    std::printf("GLM shared LM state backing bound pools=%zu initialIndependent=%zu metadata=%zu total=%zu; capacity includes diagnostic outputs and configured generations\n",
        pool_backing_bound,initial_backing_bound,metadata_backing_bound,state_budget);
  }
  // Draft-extend consumes target hidden for EACH input position. Keep that
  // generated target-boundary input explicit instead of substituting the
  // preceding row's NextN output: h_(2053+row) pairs with x_(2054+row).
  // The actual target-tap chain is covered by
  // the separate TargetPrefix case; no full target model is claimed here.
  std::array<c::ReadLease, 4> target_hidden;
  for (std::size_t row = 0; row < target_hidden.size(); ++row)
    target_hidden[row] = upload(owner, sem::bytes(d5_gguf_test::activation(kHidden, 317 + int(row) * 14)));
  std::size_t submissions = 0;
  for (const int accepted_rows : {1, 2}) {
    // Count accepted extend rows, including the target-policy bonus row as
    // num_accept_tokens does; select_index=count-1, not correct-draft count.
    SCOPED_TRACE(accepted_rows);
    auto state = State::create(owner,{},state_budget);
    {
      const auto base = state.initial(owner, initializer, queue, 2054); ++submissions;
      const auto base_bytes = image(state, base);
      auto head = take(c::StateHead::create(state.store, base), "single draft-extend/decode accepted head");
      const auto a = execute(state, row_a, queue, owner, base, 2, nullptr, {}, target_hidden[0]); ++submissions;
      const auto b = execute(state, row_b, queue, owner, a, 3, nullptr, {}, target_hidden[1]); ++submissions;
      const auto a_bytes = image(state, a), b_bytes = image(state, b);
      if(original&&original->vocabulary&&accepted_rows==1) {
        // Explicit acceptance0: neither speculative row replaces the base.
        // Continue independently from that same retained base and its pending
        // ID, after both A/B have already executed with the real shared LM.
        const auto versions_before=state.store.stats().versions;
        {
          auto zero_head=take(c::StateHead::create(state.store,base),"zero-accepted shared-LM head");
          const auto zero_base=take(zero_head.current(),"reject both original shared-LM proposals");
          EXPECT_EQ(image(state,zero_base),base_bytes);
          const auto retry=execute(state,row_a,queue,owner,zero_base,2,nullptr,{},target_hidden[0]);++submissions;
          EXPECT_EQ(image(state,retry),a_bytes);
          EXPECT_EQ(image(state,take(zero_head.current(),"zero acceptance keeps original pending/history")),base_bytes);
        }
        check(state.store.collect(),"retire zero-accepted retry before the accepted-prefix journey");
        EXPECT_EQ(state.store.stats().versions,versions_before);
      }
      const auto &producer = accepted_rows == 1 ? a : b;
      const auto seed_position = 2053 + accepted_rows;
      c::StateVersion selected = producer;
      if (accepted_rows == 1) {
        // Row B was outside the accepted prefix. Target policy supplies its
        // bonus token3, preserving A's actual hidden, IDs and all history.
        const std::array inputs{upload(owner, sem::bytes(std::vector<std::int32_t>{3}))};
        auto output = policy.execute(queue, inputs); ++submissions;
        auto draft = take(state.store.begin(a), "last accepted extend row plus target bonus");
        check(draft.replace(state.entries[kPending], output[0]), "actual bonus-token policy output");
        selected = take(draft.finalize(), "one accepted extend row and its actual seed");
      }
      check(head.publish(base.identity(), selected), "select only the last accepted extend row as one version");
      EXPECT_FALSE(head.publish(base.identity(), b).ok());
      state.enabled = false;
      EXPECT_THROW((void)activate_extend_seed(state, head, producer, seed_position + 1), std::runtime_error);
      EXPECT_FALSE(state.enabled);
      const auto selected_bytes = image(state, selected);
      const auto seed = activate_extend_seed(state, head, producer, seed_position);
      EXPECT_EQ(image(state, seed), selected_bytes); // handoff changed no epoch or resource
      same_storage(state.member(seed, kSelection), state.member(producer, kSelection));
      same_storage(state.member(seed, kHiddenState), state.member(producer, kHiddenState));

      // Decode step0 consumes the extend seed immediately: no fresh indexer
      // submission, key/gate/pool update or dynamic extra tail is permitted.
      auto &decode = accepted_rows == 1 ? tail3 : tail0;
      const auto step0 = execute(state, decode, queue, owner, seed, 4); ++submissions;
      EXPECT_EQ(state.facts(step0).producer_position, std::uint64_t(seed_position));
      EXPECT_EQ(state.facts(step0).reuse_count, 1u);
      check(head.publish(selected.identity(), step0), "accept actual seed-reusing decode prefix");
      const auto step0_bytes = image(state, step0);
      state.enabled = false;
      EXPECT_THROW((void)activate_extend_seed(state, head, producer, seed_position), std::runtime_error);
      auto &journal = accepted_rows == 1 ? partial_catchup : full_catchup;
      const auto caught = catch_up(state, journal, queue, head); ++submissions;
      check(head.publish(step0.identity(), caught), "publish only accepted skip-feature history");
      const auto caught_bytes = image(state, caught);
      // Skip publishes an actual Copy of the seed IDs. Catchup retains that
      // accepted step0 lease; it must not revert to the earlier seed backing.
      same_storage(state.member(caught, kSelection), state.member(step0, kSelection));
      EXPECT_EQ(read(state.member(caught, kSelection)), read(state.member(seed, kSelection)));
      same_storage(state.member(caught, kHiddenState), state.member(step0, kHiddenState));

      // A new epoch needs an ACTUAL new draft-extend execution. No copying
      // epoch7 metadata onto an old selection can qualify it for epoch8.
      state.epoch = 8;
      EXPECT_THROW((void)activate_extend_seed(state, head, producer, seed_position), std::runtime_error);
      EXPECT_THROW((void)execute(state, decode, queue, owner, seed, 5), std::runtime_error);
      state.enabled = true;
      auto &next_extend = accepted_rows == 1 ? partial_next : full_next;
      const auto next_producer = execute(state, next_extend, queue, owner, caught, 5, nullptr, {},
          target_hidden[std::size_t(accepted_rows + 1)]); ++submissions;
      check(head.publish(caught.identity(), next_producer), "publish actual next-epoch extend seed");
      state.enabled = false;
      const auto next_seed = activate_extend_seed(state, head, next_producer, seed_position + 2);
      EXPECT_EQ(state.facts(next_seed).epoch, 8u);
      EXPECT_NE(read(state.member(next_seed, kSelection)), read(state.member(seed, kSelection)));
      auto &next_decode = accepted_rows == 1 ? tail1 : tail2;
      const auto next_step0 = execute(state, next_decode, queue, owner, next_seed, 6); ++submissions;
      check(head.publish(next_producer.identity(), next_step0), "publish next epoch's seed-reusing step0");
      const auto final_bytes = image(state, next_step0);

      // Independent accepted-only execution starts from BASE. In prefix1 it
      // never executes B; its actual skip nx feeds the direct history writer
      // instead of the journal Gather. No speculative bytes are rebound.
      state.epoch = 7; state.enabled = true;
      auto reference_a = execute(state, row_a, queue, owner, base,
          accepted_rows == 1 ? 3 : 2, nullptr, {}, target_hidden[0]); ++submissions;
      if(original&&original->vocabulary&&accepted_rows==1) {
        // Independently replay the SAME explicit target bonus policy. Normal
        // model execution owns pending via its actual greedy Graph output;
        // the old numeric fixture argument is not a replacement LM input.
        const std::array inputs{upload(owner,sem::bytes(std::vector<std::int32_t>{3}))};
        const auto corrected=policy.execute(queue,inputs);++submissions;
        auto candidate=take(state.store.begin(reference_a),"independent accepted row plus target bonus");
        check(candidate.replace(state.entries[kPending],corrected[0]),"actual independent target-bonus output");
        reference_a=take(candidate.finalize(),"independent accepted model row and explicit corrected pending");
      }
      c::StateVersion reference_producer = reference_a;
      if (accepted_rows == 2) {
        reference_producer = execute(state, row_b, queue, owner, reference_a, 3, nullptr, {}, target_hidden[1]); ++submissions;
      }
      EXPECT_EQ(image(state, reference_producer), selected_bytes);
      check(head.publish(next_step0.identity(), reference_producer), "nominate independently executed accepted extend prefix");
      state.enabled = false;
      const auto reference_seed = activate_extend_seed(state, head, reference_producer, seed_position);
      c::ReadLease independent_nx;
      const auto reference_step0 = execute(state, decode, queue, owner, reference_seed, 4, &independent_nx); ++submissions;
      check(head.publish(reference_producer.identity(), reference_step0), "publish independent accepted skip");
      EXPECT_EQ(image(state, reference_step0), step0_bytes);
      state.enabled = false;
      auto &direct = accepted_rows == 1 ? partial_direct : full_direct;
      const auto reference_caught = catch_up(state, direct, queue, head, independent_nx); ++submissions;
      check(head.publish(reference_step0.identity(), reference_caught), "publish independent direct accepted history");
      EXPECT_EQ(image(state, reference_caught), caught_bytes);
      state.epoch = 8; state.enabled = true;
      const auto reference_next = execute(state, next_extend, queue, owner, reference_caught, 5, nullptr, {},
          target_hidden[std::size_t(accepted_rows + 1)]); ++submissions;
      check(head.publish(reference_caught.identity(), reference_next), "publish independently produced new seed");
      state.enabled = false;
      const auto reference_next_seed = activate_extend_seed(state, head, reference_next, seed_position + 2);
      const auto reference_final = execute(state, next_decode, queue, owner, reference_next_seed, 6); ++submissions;
      check(head.publish(reference_next.identity(), reference_final), "publish independently continued seed step0");
      EXPECT_EQ(image(state, reference_final), final_bytes);
      EXPECT_EQ(image(state, base), base_bytes); EXPECT_EQ(image(state, a), a_bytes); EXPECT_EQ(image(state, b), b_bytes);
      EXPECT_EQ(image(state, selected), selected_bytes); EXPECT_EQ(image(state, step0), step0_bytes);
      EXPECT_EQ(image(state, caught), caught_bytes); EXPECT_FALSE(state.member(producer, kHiddenState).try_write());
      state.enabled = false;
    }
    peak_backing=std::max(peak_backing,state.peak_backing_bytes);
    EXPECT_LE(state.peak_backing_bytes,state_budget);
    check(state.store.collect(), "retire bounded seed producer and decode branches");
    EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  }
  for (const auto *program : programs) {
    EXPECT_EQ(program->output_stats().live_blocks, 0u); EXPECT_EQ(program->output_stats().active_uses, 0u);
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_EQ(submissions, original&&original->vocabulary?28u:26u);
  std::size_t actual_graphs=0;
  for(const auto *program:programs)actual_graphs+=program->native_submissions();
  EXPECT_EQ(actual_graphs,submissions+(original&&original->moe?original->moe->completed_episodes:0));
  ::testing::Test::RecordProperty("initialization_submissions",original&&original->sparse->attention?3:2);
  ::testing::Test::RecordProperty("logical_index_state_bytes", std::to_string(kPoolCapacity * (kIndexDepth + 4) + kRing * kIndexDepth * 4));
  ::testing::Test::RecordProperty("actual_submissions", int(actual_graphs+(original&&original->sparse->attention?original->sparse->attention->cold_submissions:0)));
  ::testing::Test::RecordProperty("draft_extend_submissions", original&&original->vocabulary?12:11);
  ::testing::Test::RecordProperty("skip_submissions", 8);
  ::testing::Test::RecordProperty("catchup_submissions", 4);
  ::testing::Test::RecordProperty("policy_submissions", original&&original->vocabulary?2:1);
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("journey", "draft-extend A2054/B2055->last accepted row1/2 actual hidden+full2051IDs->step0 immediately skips entire indexer->accepted nx catchup->actual epoch8 extend->step0 skip; independent accepted-only base execution/direct nx reference");
  if (original) {
    for (std::size_t i = 0; i < original->prefix_roles.size(); ++i) {
      auto reader = original->prefix_roles[i].lease; EXPECT_FALSE(reader.try_write());
      const auto expected = original_prefix::payload(*original->prefix, static_cast<original_prefix::Parameter>(i));
      EXPECT_EQ(read(reader), Bytes(expected.begin(), expected.end()));
    }
    for (std::size_t i = 0; i < original->sparse->indexer_roles.size(); ++i) {
      auto reader = original->sparse->indexer_roles[i].lease; EXPECT_FALSE(reader.try_write());
      EXPECT_EQ(read(reader), original_index::payload(*original->sparse->indexer, static_cast<original_index::Parameter>(i)));
    }
    if(original->vocabulary) {
      const auto &vocabulary=*original->vocabulary;
      for(const auto role:{shared_lm::Role::kEmbedding,shared_lm::Role::kHead,shared_lm::Role::kNorm}) {
        const auto &bank=vocabulary.at(role);auto lease=bank.lease;
        EXPECT_FALSE(lease.try_write());EXPECT_EQ(bank.owner.ref(),owner.ref());
        EXPECT_EQ(lease.size(),take(bank.description.required_bytes(),"actual shared LM bank capacity"));
        EXPECT_EQ(bank.description,shared_lm::description(vocabulary.source->geometry,role));
      }
      const auto &observed=*original->lm_observations;
      EXPECT_TRUE(observed.full_checked);EXPECT_EQ(observed.full_values,154880u);
      ::testing::Test::RecordProperty("original_roles",original->moe?1762:original->sparse->attention?26:17);
      ::testing::Test::RecordProperty("original_role_bytes",original->moe?"10030953088":original->sparse->attention?"2753893888":"2625935360");
      ::testing::Test::RecordProperty("original_runtime_bank_bytes",original->moe?"2781689984":original->sparse->attention?"2754157568":"2626199040");
      ::testing::Test::RecordProperty("shared_lm_bank_capacity_bytes","2537562112");
      ::testing::Test::RecordProperty("state_output_pool_backing_bound",std::to_string(pool_backing_bound));
      ::testing::Test::RecordProperty("state_initial_independent_backing_bound",std::to_string(initial_backing_bound));
      ::testing::Test::RecordProperty("state_metadata_backing_bound",std::to_string(metadata_backing_bound));
      ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(state_budget));
      ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
      ::testing::Test::RecordProperty("state_max_versions",int(State::kMaxVersions));
      ::testing::Test::RecordProperty("shared_lm_full_f64_scores_checked",std::to_string(observed.full_values));
      ::testing::Test::RecordProperty("shared_lm_greedy_scores_observed",std::to_string(observed.head_calls*154880));
      ::testing::Test::RecordProperty("shared_lm_prediction_to_pending_checks",std::to_string(observed.head_calls));
      ::testing::Test::RecordProperty("shared_lm_pending_embedding_checks",std::to_string(observed.pending_checks));
      if(!original->sparse->attention)::testing::Test::RecordProperty("scope","official GLM-5.3-Flash eb9eb208 original14 prefix/indexer plus untied fullV154880 BF16 embedding/head and original BF16 sharednorm; actual generated residualF32+FFN_F32 add -> originalgamma exactwiden RMS eps1e-5 -> BF16store -> fullV DenseLinearF32 -> stableTop1 actualI32pending -> next fullV embedding Gather. Shared three banks uploaded once per Context/owner, capacity not measured peak. Firsthead independent fullV F64 bound, subsequent actualTop4/selectedrow dots and allV strictgreedy everyhead, exact BF16stores/embedding/feedback. Acceptance0/1/2, explicit targetbonus, originalseed/skip/journal/directcatchup/epoch8 and atomicStateHead/oldleases remain. Targethidden,remaining MLA/MoE stay generated F32 boundaries; not original BF16 decoder/fullmodel,quality,naturalacceptance or throughput.");
      if(original->sparse->attention) {
        const auto &attention_source=*original->sparse->attention;
        EXPECT_TRUE(attention_source.full_checked);EXPECT_EQ(attention_source.projection_values,70144u);
        EXPECT_EQ(attention_source.attention_calls,observed.head_calls);EXPECT_EQ(attention_source.cold_submissions,1u);
        EXPECT_EQ(attention_source.prepared_pool.stats().capacity_bytes,33554432u);
        EXPECT_EQ(attention_source.prepared_pool.stats().live_blocks,2u);
        EXPECT_EQ(attention_source.prepared_pool.stats().active_uses,0u);
        for(std::size_t i=0;i<attention_source.roles.size();++i) {
          const auto &bank=attention_source.roles[i];
          if(i==static_cast<std::size_t>(original_attention::Parameter::kKvB)) {EXPECT_FALSE(bank.lease.valid());continue;}
          auto lease=bank.lease;EXPECT_FALSE(lease.try_write());EXPECT_EQ(bank.owner.ref(),owner.ref());
          EXPECT_EQ(lease.size(),take(bank.description.required_bytes(),"actual original attention bank capacity"));
        }
        for(const auto &bank:attention_source.prepared) {
          auto lease=bank.lease;EXPECT_FALSE(lease.try_write());EXPECT_EQ(bank.owner.ref(),owner.ref());
          EXPECT_EQ(lease.size(),take(bank.description.required_bytes(),"actual prepared original KV bank capacity"));
        }
        ::testing::Test::RecordProperty("attention_original_roles",9);
        ::testing::Test::RecordProperty("attention_original_bytes","127958528");
        ::testing::Test::RecordProperty("attention_cold_layout_submissions",int(attention_source.cold_submissions));
        ::testing::Test::RecordProperty("attention_prepared_kv_bf16_capacity_bytes",std::to_string(attention_source.prepared_pool.stats().capacity_bytes));
        ::testing::Test::RecordProperty("attention_raw_kv_device_bank_retained",0);
        ::testing::Test::RecordProperty("main_journey_submissions",int(actual_graphs));
        ::testing::Test::RecordProperty("logical_journey_calls",int(submissions));
        ::testing::Test::RecordProperty("attention_full_local_projection_values_checked",std::to_string(attention_source.projection_values));
        ::testing::Test::RecordProperty("attention_calls_checked",std::to_string(attention_source.attention_calls));
        if(!original->moe)::testing::Test::RecordProperty("scope","official GLM-5.3-Flash eb9eb208 original26 physical roles: original14 prefix/indexer, untied original3 fullV shared LM, original9 complete MLA attention/postnorm. Dynamic E4 A128 Divide floor1e-10 projections have actual BF16 stores; original single BF16 KV_b cold-split once into shared BF16 K/V ready banks, original raw device bank retired; ordinary BF16 BMM with F32 accumulation, compact selected same-CKV MQA scale1/16 with BF16 output; original kvnorm/postnorm. F32 cache carrier contains only BF16-widened latent values, not vendor page ABI. Postnorm consumes actual F32 EH+O sum and separately returns actual BF16 residual; generated F32 MoE output then enters existing original sharednorm/fullVhead/actualpending feedback. First complete local projection checks, subsequent actual boundaries plus bounded local projection samples; accept0/1/2, explicit targetbonus, journal/directcatchup/epoch8 and one StateHead preserved. Targethidden, initial maincache/index history and MoE remain declared generated boundaries; no full original target/model, natural acceptance, quality or TPS claim. Capacities are readybank bytes, not measured peak.");
        if(original->moe) {
          const auto &model=*original->moe;
          EXPECT_TRUE(model.full_checked);EXPECT_EQ(model.projection_values,74016u);
          EXPECT_EQ(model.completed_episodes,observed.head_calls);EXPECT_EQ(model.selections.size(),model.completed_episodes);
          EXPECT_EQ(model.union_sizes.size(),model.completed_episodes);
          ::testing::Test::RecordProperty("moe_original_roles",1736);
          ::testing::Test::RecordProperty("moe_original_bytes","7277059200");
          ::testing::Test::RecordProperty("moe_completed_episodes",std::to_string(model.completed_episodes));
          ::testing::Test::RecordProperty("moe_full_local_projection_values_checked",std::to_string(model.projection_values));
          ::testing::Test::RecordProperty("moe_resident_expert_capacity",int(original_moe::kResidentCapacity));
          ::testing::Test::RecordProperty("moe_global_expert_domain",int(original_moe::kExpertDomain));
          ::testing::Test::RecordProperty("moe_resident_slot_logical_bytes",std::to_string(resident_moe::kLogicalBytes));
          ::testing::Test::RecordProperty("moe_resident_slot_capacity_bytes",std::to_string(resident_moe::kSlotBytes));
          ::testing::Test::RecordProperty("moe_resident_slot_count",2);
          std::string unions;
          for(const auto size:model.union_sizes){if(!unions.empty())unions+=",";unions+=std::to_string(size);}
          ::testing::Test::RecordProperty("moe_actual_union_sizes",unions);
          for(const auto *bank:{&model.roles.router,&model.roles.bias,&model.roles.shared[0],&model.roles.shared[1],&model.roles.shared[2]}) {
            auto lease=bank->lease;EXPECT_FALSE(lease.try_write());EXPECT_EQ(bank->owner.ref(),owner.ref());
            EXPECT_EQ(lease.size(),take(bank->description.required_bytes(),"original static MoE ready bank capacity"));
          }
          ::testing::Test::RecordProperty("scope","official GLM-5.3-Flash eb9eb208 complete original layer45 NextN1760 roles plus untied fullV embedding/head2=1762 original roles,10030953088 compressed source bytes. Actual original input/query/indexer/MLA/postnorm/router/sharedFFN/full288 expert domain with actual selected original experts/sharednorm/fullVhead execute in the same seed/skip/catchup/atomicStateHead journey. Two ordinary Graph phases cut after actual original288 sigmoid+bias Top8 and unbiased normalized weights(sum+1e-20,scale2.5once); only8 real globalI32IDs read to host for bounded E8 original E4/SF supply, map and3banks bound atomically on the same compiled resource, activations/state directleases. All actual BF16 projection/clip/activation/post-down-weighted/sum8/sharedmerge boundaries checked. First complete selected MoE projections, later fixed7rows/slot plus every actual boundary/router/LM/feedback. F32router BF16operand contract follows fixedCUDA path, not a vendorCPU bitwise claim. Original KV coldtranspose and F32cache exactBF16widen carrier retained. Targethidden and finite initial history remain explicitly generated external boundaries; no full target model, vendorphysicalcache ABI, naturalacceptance,quality or TPS claim. Readybank/slot capacities and upload API counters are not measured peakVRAM or bus traffic.");
        }
      }
    } else {
      ::testing::Test::RecordProperty("original_roles", 14);
      ::testing::Test::RecordProperty("original_role_bytes", 88373248);
      ::testing::Test::RecordProperty("original_runtime_bank_bytes", 88636928);
      ::testing::Test::RecordProperty("scope", "official HF GLM-5.3-Flash eb9eb208 layer45 original7 input/query+original7 indexer in the existing complete sequential M1 NextN seed/catchup journey; explicit generated target-hidden/embedding and MLA/MoE/head remainder; no primaryGGUF tensor equality,full target,original shared LM,natural acceptance,vendorpage ABI or throughput claim");
    }
  } else ::testing::Test::RecordProperty("scope", "generated selected Flash primary GGUF/F32 sequential M1 draft-extend seed protocol; generated per-position target-hidden boundary; default UE8M0 logical cache/query and BF16 raw rings; no target43 MLA/indexer composition,batched vendor extend kernel,full target,original weights,vendorpage ABI,natural acceptance or throughput claim");
}

void run(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  auto fresh_owner=Program::create(build(true), context, owner, cuda);
  auto &fresh=*fresh_owner;
  auto skip_owner=Program::create(build(false), context, owner, cuda);
  auto &skip=*skip_owner;
  Recipe policy_recipe; policy_recipe.fragment = pending_policy();
  auto policy_owner=Program::create(std::move(policy_recipe), context, owner, cuda);
  auto &policy=*policy_owner;
  auto queue = take(context.create_queue(), "complete component queue");
  auto initializer_owner=Program::create(initial_pool_recipe(), context, owner, cuda, 2);
  auto &initializer=*initializer_owner;
  auto state = State::create(owner);
  std::size_t native_submissions = 0;
  {
    const auto base = state.initial(owner, initializer, queue); ++native_submissions;
    const auto base_bytes = image(state, base);
    auto head = take(c::StateHead::create(state.store, base), "whole component shared head");
    const auto first = execute(state, fresh, queue, owner, base, 2); ++native_submissions;
    const auto first_bytes = image(state, first);
    const auto rejected_tail = execute(state, skip, queue, owner, first, 4); ++native_submissions;
    const auto rejected_bytes = image(state, rejected_tail);
    EXPECT_EQ(state.scalar(rejected_tail, kMlaCursor), kBasePosition + 2);
    EXPECT_EQ(state.scalar(rejected_tail, kIndexCursor), kBasePosition + 1);
    EXPECT_EQ(state.facts(rejected_tail).reuse_count, 1u);
    // An explicit target policy chooses token3 instead of the draft's token2.
    const std::array correction_input{upload(owner, sem::bytes(std::vector<std::int32_t>{3}))};
    auto correction = policy.execute(queue, correction_input); ++native_submissions;
    auto accepted_draft = take(state.store.begin(first), "accepted prefix plus explicit policy token");
    check(accepted_draft.replace(state.entries[kPending], correction[0]), "actual corrected pending output");
    const auto accepted = take(accepted_draft.finalize(), "complete accepted prefix1");
    check(head.publish(base.identity(), accepted), "atomically publish complete prefix1");
    EXPECT_FALSE(head.publish(base.identity(), rejected_tail).ok()) << "stale prefix2 must not replace the accepted head";
    EXPECT_EQ(image(state, take(head.current(), "actual accepted head")), image(state, accepted));
    // No part of token2's MLA append enters the accepted/draft continuation.
    EXPECT_EQ(image(state, first), first_bytes);
    const auto accepted_cache = read(state.member(accepted, kCache));
    const auto rejected_cache = read(state.member(rejected_tail, kCache));
    EXPECT_FALSE(std::equal(accepted_cache.begin() + (kBasePosition + 1) * kLatent * 4,
        accepted_cache.begin() + (kBasePosition + 2) * kLatent * 4,
        rejected_cache.begin() + (kBasePosition + 1) * kLatent * 4));
    const auto continued = execute(state, skip, queue, owner, take(head.current(), "continue actual accepted head"), 5); ++native_submissions;
    check(head.publish(accepted.identity(), continued), "publish prefix1 continuation as a whole");
    // Independent recomputation starts from the retained BASE leases, with
    // target-correct token3 already chosen. No first-run result is rebound.
    const auto independent_first = execute(state, fresh, queue, owner, base, 3); ++native_submissions;
    EXPECT_EQ(image(state, accepted), image(state, independent_first));
    const auto independent_next = execute(state, skip, queue, owner, independent_first, 5); ++native_submissions;
    EXPECT_EQ(image(state, continued), image(state, independent_next));
    EXPECT_NE(read(state.member(continued, kCache)), read(state.member(rejected_tail, kCache)));
    // Prefix2 is valid only when explicitly accepted on its own base branch.
    auto all = take(c::StateHead::create(state.store, base), "independent full-prefix branch");
    check(all.publish(base.identity(), rejected_tail), "explicitly accept prefix2 on its own branch");
    const auto branch = execute(state, skip, queue, owner, take(all.current(), "retained prefix2 branch"), 6); ++native_submissions;
    check(all.publish(rejected_tail.identity(), branch), "publish old branch continuation");
    EXPECT_EQ(state.scalar(branch, kMlaCursor), kBasePosition + 3);
    EXPECT_EQ(state.scalar(branch, kIndexCursor), kBasePosition + 1);
    EXPECT_EQ(image(state, rejected_tail), rejected_bytes);
    EXPECT_EQ(image(state, base), base_bytes);
    // A cancelled multi-member transaction after staging one actual resource
    // cannot expose it or the invalid pending extent through the shared head.
    const auto before_cancel = image(state, take(head.current(), "head before failed candidate"));
    auto cancelled = take(state.store.begin(continued), "failed whole-state candidate");
    check(cancelled.replace(state.entries[kHiddenState], state.member(first, kHiddenState)), "stage candidate hidden only");
    EXPECT_FALSE(cancelled.replace(state.entries[kPending], state.member(first, kHiddenState)).ok());
    check(cancelled.cancel(), "discard incomplete candidate");
    EXPECT_EQ(image(state, take(head.current(), "head after cancellation")), before_cancel);
    state.enabled = false;
    EXPECT_FALSE(state.qualifies(continued)); EXPECT_FALSE(state.qualifies(rejected_tail));
    EXPECT_THROW((void)execute(state, skip, queue, owner, continued, 6), std::runtime_error);
    EXPECT_EQ(image(state, first), first_bytes); EXPECT_EQ(image(state, base), base_bytes);
    EXPECT_FALSE(state.member(first, kCache).try_write());
  }
  check(state.store.collect(), "retire all complete component versions");
  EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  for (const auto *program : {&initializer, &fresh, &skip, &policy}) {
    EXPECT_EQ(program->output_stats().live_blocks, 0u); EXPECT_EQ(program->output_stats().active_uses, 0u);
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_EQ(native_submissions, 8u);
  ::testing::Test::RecordProperty("initialization_submissions", 1);
  ::testing::Test::RecordProperty("logical_index_state_bytes", std::to_string(kPoolCapacity * (kIndexDepth + 4) + kRing * kIndexDepth * 4));
  ::testing::Test::RecordProperty("actual_submissions", int(native_submissions));
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("component_schema", "17 native resources plus explicit iteration policy; BF16 raw key/score rings,E4 pools/F32 scales,complete NextN/indexer state and bounded nx journal/lag count");
  ::testing::Test::RecordProperty("scope", "generated primary GGUF/F32 single complete NextN component; fresh K4 closepool->actual I32 sparse MLA/MoE; same-epoch entire-indexer skip; default UE8M0 logical cache/query and BF16 raw rings; no original checkpoint/full target/KDA/mHC/vendorpage ABI/natural acceptance");
  ::testing::Test::RecordProperty("prefixes", "0 retained;1 corrected-pending atomic publish and independent base recomputation;2 independent branch; rejected tail/cancel/stale publish/old readonly ownership");
  ::testing::Test::RecordProperty("uncovered", "this case is single-epoch; cross-epoch catchup is covered by the separate AcceptedJournal cases; target KDA/conv/mHC,original projection weights,vendorpage ABI and vocabulary/natural acceptance remain separate");
}

void run_catchup(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  // One two-input proposal crosses a real K4 boundary: A at2054 produces a
  // three-token unpooled tail, B at2055 skips the entire indexer. Prefix1
  // rejects B; prefix2 must catch up exactly its accepted nx row before the
  // next iteration. These are two required branches, not a shape matrix.
  auto first_plan_owner=Program::create(build(true, 2054), context, owner, cuda);
  auto &first_plan=*first_plan_owner;
  auto partial_plan_owner=Program::create(build(true, 2055), context, owner, cuda);
  auto &partial_plan=*partial_plan_owner;
  auto full_plan_owner=Program::create(build(true, 2056), context, owner, cuda);
  auto &full_plan=*full_plan_owner;
  auto skip_owner=Program::create(build(false, 0, 3), context, owner, cuda);
  auto &skip=*skip_owner;
  auto journal_owner=Program::create(build_catchup(), context, owner, cuda);
  auto &journal=*journal_owner;
  auto direct_owner=Program::create(build_catchup(true), context, owner, cuda);
  auto &direct=*direct_owner;
  Recipe policy_recipe; policy_recipe.fragment = pending_policy();
  auto policy_owner=Program::create(std::move(policy_recipe), context, owner, cuda);
  auto &policy=*policy_owner;
  auto queue = take(context.create_queue(), "accepted journal consumer queue");
  auto initializer_owner=Program::create(initial_pool_recipe(), context, owner, cuda, 2);
  auto &initializer=*initializer_owner;
  std::size_t submissions = 0;
  for (const int accepted_rows : {1, 2}) {
    SCOPED_TRACE(accepted_rows);
    auto state = State::create(owner);
    {
      const auto base = state.initial(owner, initializer, queue, 2054); ++submissions;
      const auto base_bytes = image(state, base);
      auto head = take(c::StateHead::create(state.store, base), "accepted-journal whole-state head");
      const auto first = execute(state, first_plan, queue, owner, base, 2); ++submissions;
      const auto first_bytes = image(state, first);
      const auto second = execute(state, skip, queue, owner, first, 4); ++submissions;
      const auto second_bytes = image(state, second);
      EXPECT_EQ(state.scalar(first, kMlaCursor), 2055); EXPECT_EQ(state.scalar(first, kIndexCursor), 2055);
      EXPECT_EQ(state.scalar(first, kTailValid), 3); EXPECT_EQ(state.scalar(first, kLagCount), 0);
      EXPECT_EQ(state.scalar(second, kMlaCursor), 2056); EXPECT_EQ(state.scalar(second, kIndexCursor), 2055);
      EXPECT_EQ(state.scalar(second, kLagCount), 1);
      c::StateVersion accepted = second;
      if (accepted_rows == 1) {
        // B's input token2 is rejected; a real ordinary policy Copy chooses
        // token3. No B feature/journal/cursor is spliced into prefix1.
        const std::array policy_input{upload(owner, sem::bytes(std::vector<std::int32_t>{3}))};
        auto output = policy.execute(queue, policy_input); ++submissions;
        auto candidate = take(state.store.begin(first), "prefix1 pending correction");
        check(candidate.replace(state.entries[kPending], output[0]), "actual target-policy token permission");
        accepted = take(candidate.finalize(), "complete prefix1 with target policy");
        EXPECT_EQ(state.scalar(accepted, kLagCount), 0);
        same_storage(state.member(accepted, kJournal), state.member(base, kJournal));
      }
      check(head.publish(base.identity(), accepted), "publish only the chosen complete accepted prefix");
      EXPECT_FALSE(head.publish(base.identity(), second).ok());
      const auto accepted_bytes = image(state, accepted);
      auto cancelled = take(state.store.begin(accepted), "failed journal splice");
      check(cancelled.replace(state.entries[kJournal], state.member(second, kJournal)), "stage a speculative journal permission");
      EXPECT_FALSE(cancelled.replace(state.entries[kLagCount], state.member(second, kJournal)).ok());
      check(cancelled.cancel(), "cancel incomplete journal transaction");
      EXPECT_EQ(image(state, take(head.current(), "head after rejected journal splice")), accepted_bytes);

      state.enabled = false; // source finally: old IDs cease to qualify.
      EXPECT_FALSE(state.qualifies(first)); EXPECT_FALSE(state.qualifies(second));
      EXPECT_THROW((void)execute(state, skip, queue, owner, accepted, 5), std::runtime_error);
      c::StateVersion caught = accepted;
      if (accepted_rows == 2) {
        caught = catch_up(state, journal, queue, head); ++submissions;
        check(head.publish(accepted.identity(), caught), "publish index catchup with its actual counters");
        EXPECT_EQ(state.scalar(caught, kIndexCursor), 2056);
        EXPECT_EQ(state.scalar(caught, kMlaCursor), 2056);
        EXPECT_EQ(state.scalar(caught, kLagCount), 0);
        EXPECT_EQ(state.scalar(caught, kPoolCount), 514); EXPECT_EQ(state.scalar(caught, kTailValid), 0);
        same_storage(state.member(caught, kCache), state.member(accepted, kCache));
        same_storage(state.member(caught, kHiddenState), state.member(accepted, kHiddenState));
        same_storage(state.member(caught, kSelection), state.member(accepted, kSelection));
      } else {
        // Rejected B cannot make a valid catchup input. This refuses before
        // any preparation or native submission, even though B remains readable.
        EXPECT_THROW((void)catch_up(state, journal, queue, head), std::runtime_error);
      }
      const auto caught_bytes = image(state, caught);
      state.epoch = 8; state.enabled = true;
      auto &next_plan = accepted_rows == 1 ? partial_plan : full_plan;
      const auto next = execute(state, next_plan, queue, owner, take(head.current(), "caught-up next-iteration head"), 5); ++submissions;
      check(head.publish(caught.identity(), next), "publish actual epoch8 fresh seed and NextN state");
      EXPECT_EQ(state.facts(next).epoch, 8u);
      EXPECT_EQ(state.facts(next).producer_position, std::uint64_t(2054 + accepted_rows));
      EXPECT_TRUE(state.qualifies(next)); EXPECT_FALSE(state.qualifies(first)); EXPECT_FALSE(state.qualifies(accepted));
      EXPECT_NE(read(state.member(next, kSelection)), read(state.member(first, kSelection))) << "fresh generated IDs must replace the old tail seed";
      EXPECT_THROW((void)execute(state, skip, queue, owner, accepted, 5), std::runtime_error);

      // Independent accepted-sequence execution starts from BASE. The full
      // branch retains its actual independent B nx result and feeds that
      // lease directly into the ordinary history writer, avoiding the real
      // journal Gather path. Math is separately checked at every boundary.
      state.epoch = 7; state.enabled = true;
      const auto reference_first = execute(state, first_plan, queue, owner, base, accepted_rows == 1 ? 3 : 2); ++submissions;
      c::StateVersion reference_caught = reference_first;
      c::ReadLease independent_nx;
      if (accepted_rows == 2) {
        const auto reference_second = execute(state, skip, queue, owner, reference_first, 4, &independent_nx); ++submissions;
        state.enabled = false;
        auto reference_head = take(c::StateHead::create(state.store, reference_second), "independent accepted prefix2 head");
        reference_caught = catch_up(state, direct, queue, reference_head, independent_nx); ++submissions;
      }
      EXPECT_EQ(image(state, reference_caught), caught_bytes);
      state.epoch = 8; state.enabled = true;
      const auto reference_next = execute(state, next_plan, queue, owner, reference_caught, 5); ++submissions;
      EXPECT_EQ(image(state, next), image(state, reference_next));
      EXPECT_EQ(image(state, base), base_bytes); EXPECT_EQ(image(state, first), first_bytes);
      EXPECT_EQ(image(state, second), second_bytes); EXPECT_EQ(image(state, accepted), accepted_bytes);
      EXPECT_EQ(image(state, caught), caught_bytes);
      EXPECT_FALSE(state.member(second, kJournal).try_write());
      state.enabled = false;
      EXPECT_FALSE(state.qualifies(next)); EXPECT_FALSE(state.qualifies(first));
    }
    check(state.store.collect(), "retire accepted journal branch states");
    EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  }
  for (const auto *program : {&initializer, &first_plan, &partial_plan, &full_plan, &skip, &journal, &direct, &policy}) {
    EXPECT_EQ(program->output_stats().live_blocks, 0u); EXPECT_EQ(program->output_stats().active_uses, 0u);
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_EQ(submissions, 16u);
  ::testing::Test::RecordProperty("initialization_submissions", 2);
  ::testing::Test::RecordProperty("logical_index_state_bytes", std::to_string(kPoolCapacity * (kIndexDepth + 4) + kRing * kIndexDepth * 4));
  ::testing::Test::RecordProperty("actual_submissions", int(submissions));
  ::testing::Test::RecordProperty("fresh_submissions", 8);
  ::testing::Test::RecordProperty("skip_submissions", 3);
  ::testing::Test::RecordProperty("catchup_submissions", 2);
  ::testing::Test::RecordProperty("policy_submissions", 1);
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("journal", "actual nx[2,4096] with I32 valid lag count; selected StateVersion determines accepted input; no eager indexer during skip");
  ::testing::Test::RecordProperty("journey", "base2054 tail2; accept1 rejects B and fresh2055 closes pool; accept2 catches actual accepted B2055 then fresh2056 appends tail1; independent actual nx direct-input reference");
  ::testing::Test::RecordProperty("scope", "generated selected GGUF/F32 NextN cross-epoch accepted-feature catchup and fresh seed; default UE8M0 logical cache/query and BF16 raw rings; no target KDA/conv/mHC,original projection weights,vendorpage ABI or natural acceptance claim");
}

namespace target = glm_mtp_target_component;
struct Target44Recipe final {
  Recipe recipe;
  target::Ports ports;
  target::Reference generated_reference;
  std::size_t cursor_port=0;
};

Target44Recipe mlx_target44_recipe(std::shared_ptr<OriginalTarget44> source) {
  Target44Recipe result;auto &r=result.recipe;r.target44_source=std::move(source);r.mlx=r.target44_source->mlx;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;f.profile=&mlx_runtime::kProfile;f.region=0x514d4c5854343441ULL;f.quantized=true;
  const auto residual=f.input(sem::dense({1,4,kHidden})),history=f.input(sem::dense({3,target::kChannels}));
  const auto matrix=f.input(sem::dense({1,target::kHeads,target::kDepth,target::kDepth})),cursor=f.input(sem::dense({1},c::DataType::kInt32));
  const auto lookup=[&](std::string_view name){const auto found=std::find_if(r.external.begin(),r.external.end(),[&](const auto &v){return v.role.name==name;});
    return found!=r.external.end()?f.constants[found->constant]:external(r,r.mlx->parameters.at(name));};
  std::array<Value,3> convolutions;for(std::size_t i=0;i<3;++i)convolutions[i]=external(r,r.mlx->convolutions[i]);
  const auto parameters=mlx_target::parameters44(f,lookup,convolutions);
  r.mlx_parameters=mlx_model::moe_parameters(f,lookup,r.mlx->parameters.source->prefix());
  const auto layout=mlx_runtime::layout();std::vector<Value> banks;for(const auto &d:layout.resident)banks.push_back(f.input(d));const auto map=f.input(layout.map_description());
  const auto unscaled=[](sem::Fragment &graph,Value q,Value k,Value v,Value decay,Value beta,Value state) {
    const auto outputs=graph.append(c::Operation(c::GatedDeltaNet{.output_scale=1.0F}),{q,k,v,decay,beta,state},2);return std::array{outputs[0],outputs[1]};
  };
  r.mlx44=mlx_target::append44(f,residual,history,matrix,parameters,unscaled,[&](sem::Fragment &graph,Value input){return mlx_model::append_moe(graph,input,*r.mlx_parameters,banks,map);});
  const auto &actual=*r.mlx44;r.mlx_boundary=actual.moe.boundary;
  result.ports.history=expose(r,actual.history,true);result.ports.state=expose(r,actual.state,true);
  result.ports.mhc=expose(r,actual.mhc,true);result.ports.tap=expose(r,actual.tap,true);result.cursor_port=expose(r,increment(f,cursor),true);
  r.mlx44->ports.visit_values([&](Value &value){mlx_expose(r,value);});r.mlx44->moe.ports.visit_values([&](Value &value){mlx_expose(r,value);});
  for(auto value:{actual.history,actual.state,actual.mhc,actual.tap})mlx_expose(r,value);return result;
}

Target44Recipe converted_target44_recipe(std::shared_ptr<OriginalTarget44> source) {
  Target44Recipe result;auto &r=result.recipe;r.target44_source=std::move(source);r.converted=r.target44_source->converted;r.profile_owner=r.target44_source;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());auto &f=r.fragment;f.profile=r.converted->profile;f.region=0x4747464c54343441ULL;f.quantized=true;
  const auto residual=f.input(sem::dense({1,4,kHidden})),history=f.input(sem::dense({3,target::kChannels}));
  const auto matrix=f.input(sem::dense({1,target::kHeads,target::kDepth,target::kDepth})),cursor=f.input(sem::dense({1},c::DataType::kInt32));
  r.converted_parameters=converted_parameters(r);const auto [banks,map]=converted_banks(r);
  r.converted44=gguf_flash::append_target44(f,residual,history,matrix,*r.converted_parameters,banks,map);auto &v=*r.converted44;converted_boundary(r,v.moe.boundary);
  result.ports.history=expose(r,v.history,true);result.ports.state=expose(r,v.state,true);result.ports.mhc=expose(r,v.mhc,true);result.ports.tap=expose(r,v.tap,true);
  result.cursor_port=expose(r,increment(f,cursor),true);v.ports.visit_values([&](Value &value){mlx_expose(r,value);});v.moe.visit_values([&](Value &value){mlx_expose(r,value);});
  for(auto value:{v.history,v.state,v.mhc,v.tap})mlx_expose(r,value);return result;
}
Target44Recipe target44_recipe(std::shared_ptr<OriginalTarget44> source={}) {
  if(source&&source->converted)return converted_target44_recipe(std::move(source));
  if(source&&source->mlx)return mlx_target44_recipe(std::move(source));
  Target44Recipe result;auto &r=result.recipe;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  if(!source) {
    auto component=target::build();result.ports=component.ports;
    result.generated_reference=std::move(component.evaluate);r.fragment=std::move(component.fragment);
    const auto cursor=r.fragment.input(sem::dense({1},c::DataType::kInt32));
    result.cursor_port=expose(r,increment(r.fragment,cursor));return result;
  }
  r.target44_source=std::move(source);r.moe=r.target44_source->moe;
  auto &f=r.fragment;f.profile=r.target44_source->artifact_owner?
      &r.target44_source->artifact_owner->profile_description:&glm::core::profiles::glm53_flash_official;
  f.region=0x474c4d34344f5247ULL;f.quantized=true;
  // One common state ABI: residual/tap/history carriers exactly widen the
  // real BF16 stores. The internal KDA matrix remains F32 at every boundary.
  const auto residual=f.input(sem::dense({1,target::kStreams,target::kHidden}));
  const auto history=f.input(sem::dense({3,target::kChannels}));
  const auto matrix=f.input(sem::dense({1,target::kHeads,target::kDepth,target::kDepth}));
  const auto cursor=f.input(sem::dense({1},c::DataType::kInt32));
  target44_fragment::ParameterValues parameters;
  for(std::size_t slot=0;slot<parameters.size();++slot)
    parameters[slot]=external(r,r.target44_source->parameters.at(static_cast<original_target44::Role>(slot)));
  // Dynamic MoE bindings follow the four ordinary component inputs.
  // Program::execute_resident owns their same-generation bounded supply.
  const auto moe=moe_parameters(r);
  r.target44=target44_fragment::append(f,codec::cast(f,residual,c::DataType::kBFloat16),history,matrix,parameters,moe,
      r.moe->arithmetic,r.moe->roles.source->weight_storage());
  const auto &actual=*r.target44;
  result.ports.history=expose(r,actual.history,true);result.ports.state=expose(r,actual.state,true);
  result.ports.mhc=expose(r,actual.mhc,true);result.ports.tap=expose(r,actual.tap,true);
  result.cursor_port=expose(r,increment(f,cursor));
  const auto observe=[&](Value &value) {
    const auto found=std::find_if(r.target44_ports.begin(),r.target44_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found==r.target44_ports.end())r.target44_ports.push_back({value,expose(r,value,true)});
  };
  r.target44->ports.visit_values(observe);
  for(auto value:{actual.history,actual.state,actual.mhc,actual.tap})observe(value);
  const auto input_port=std::find_if(r.target44_ports.begin(),r.target44_ports.end(),[&](const auto &entry){return entry.first.ordinal()==actual.ports.ffn_hc.norm.stored.ordinal();});
  if(input_port==r.target44_ports.end())throw std::runtime_error("original target44 actual MoE predecessor missing");
  r.moe_input=input_port->second;
  r.moe_output=expose(r,codec::cast(f,actual.moe.output,c::DataType::kFloat32),true);
  r.original_moe=actual.moe;
  r.original_moe->ports.visit_values([&](Value &value){r.moe_ports.push_back({value,expose(r,value,true)});});
  return result;
}
void observe_target44(const Recipe &recipe,std::span<const c::ReadLease> input,std::span<const c::ReadLease> output) {
  if(recipe.converted44) {
    const auto actual=[&](Value value)->Bytes {
      const auto found=std::find_if(recipe.mlx_ports.begin(),recipe.mlx_ports.end(),[&](const auto &p){return p.first.ordinal()==value.ordinal();});
      if(found!=recipe.mlx_ports.end())return read(output[found->second]);
      const auto it=std::find_if(recipe.fragment.inputs.begin(),recipe.fragment.inputs.end(),[&](Value v){return v.ordinal()==value.ordinal();});
      if(it==recipe.fragment.inputs.end()||std::size_t(it-recipe.fragment.inputs.begin())>=input.size())throw std::runtime_error("converted KDA actual input missing");
      return read(input[std::size_t(it-recipe.fragment.inputs.begin())]);
    };
    EXPECT_TRUE(gguf_reference::observe_target44(*recipe.converted->parameters.source,*recipe.converted44,recipe.fragment,*recipe.converted_parameters,
        actual,converted_original(recipe),converted_expert(recipe),!recipe.converted->full_checked,converted_codec(recipe.cuda)));
    recipe.converted->full_checked=true;++recipe.target44_source->checked_calls;
    if(::testing::Test::HasFailure())throw std::runtime_error("converted target44 original numerical check failed before publication");return;
  }
  if(recipe.mlx44) {
    auto actual=[&](Value value){return typed<float>(output[mlx_port(recipe,value)]);};auto ids=[&](Value value){return typed<std::int32_t>(output[mlx_port(recipe,value)]);};
    const auto &data=*recipe.mlx->parameters.source;const bool full=!recipe.mlx->full_checked;
    EXPECT_TRUE(mlx_target_reference::observe44(data,*recipe.mlx44,actual,full));
    mlx_reference::observe_moe(data,recipe.mlx44->moe,actual,ids,full);recipe.mlx->full_checked=true;++recipe.target44_source->checked_calls;
    if(::testing::Test::HasFailure())throw std::runtime_error("MLX target44 original local reference failed before publication");return;
  }
  if(!recipe.target44||!recipe.target44_source)throw std::runtime_error("original target44 numerical source missing");
  const auto actual=[&](Value value) {
    const auto found=std::find_if(recipe.target44_ports.begin(),recipe.target44_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(found!=recipe.target44_ports.end())return read(output[found->second]);
    const auto moe=std::find_if(recipe.moe_ports.begin(),recipe.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
    if(moe==recipe.moe_ports.end())throw std::runtime_error("original target44 actual observation missing");
    return read(output[moe->second]);
  };
  const auto residual=recipe.target44->ports.attention_hc.residual;
  const auto stored=actual(residual);std::vector<std::uint16_t> bits(stored.size()/sizeof(std::uint16_t));
  std::memcpy(bits.data(),stored.data(),stored.size());
  EXPECT_TRUE(index_reference::expect_bfloat_store(bits,typed<float>(input[0]),"actual target43 output -> explicit original target44 BF16 input"));
  EXPECT_TRUE(target44_reference::observe(*recipe.target44_source->parameters.source,*recipe.target44,actual,!recipe.target44_source->full_checked));
  observe_original_moe(recipe,output);
  recipe.target44_source->full_checked=true;++recipe.target44_source->checked_calls;
}
enum TargetEntry : std::size_t {
  kTargetHistory = kEntries, kTargetState, kTargetTap, kTargetCursor
};
constexpr std::size_t kTarget43Begin = kTargetCursor + 1;
struct Target43Result final {
  std::vector<c::ReadLease> outputs;
  SparseSnapshot state;
  c::ReadLease mhc;
};
SparseSnapshot target43_snapshot(const State &state,const c::StateVersion &version) {
  SparseSnapshot result;result.plain_index=state.plain_index;const auto active=sparse_entries(state.plain_index);
  for(std::size_t i=0;i<active.size();++i){const auto found=std::find(kSparseEntries.begin(),kSparseEntries.end(),active[i]);
    result.values[std::size_t(found-kSparseEntries.begin())]=state.member(version,kTarget43Begin+i);}
  return result;
}
Target43Result execute_target43(Program &program, c::Queue &queue,
                               const SparseSnapshot &before, const c::ReadLease &residual) {
  const SparseAccess access;
  const auto position = program.recipe.position;
  if (!program.recipe.fresh || access.scalar(before, kMlaCursor) != position ||
      access.scalar(before, kIndexCursor) != position || access.scalar(before, kPoolCount) != position / 4 ||
      access.scalar(before, kRingOffset) != position % std::int32_t(kRing) ||
      access.scalar(before, kTailValid) != position % 4 || access.scalar(before, kTailStart) != position - position % 4)
    throw std::runtime_error("target43 actual cache frontier does not match its prepared attention domain");
  std::vector<c::ReadLease> input{residual};
  for(const auto entry:sparse_entries(before.plain_index))input.push_back(before[entry]);
  Target43Result result; result.outputs = program.execute(queue, input); result.state = before;
  for (std::size_t i = 0; i < kSparseEntries.size(); ++i) {
    const auto port = program.recipe.state_ports[kSparseEntries[i]];
    if (port != std::numeric_limits<std::size_t>::max()) result.state.values[i] = result.outputs[port];
  }
  const auto nx = typed<float>(result.outputs[program.recipe.nx]);
  const auto &recipe=program.recipe;
  if(recipe.converted43) {
    auto actual=converted_actual(recipe,result.outputs,access,before);
    EXPECT_TRUE(gguf_reference::observe_target43(*recipe.converted->parameters.source,*recipe.converted43,recipe.fragment,*recipe.converted_parameters,
        *recipe.pooled_before,actual,converted_original(recipe),converted_expert(recipe),position,!recipe.converted->full_checked,converted_codec(recipe.cuda)));
    recipe.converted->full_checked=true;++recipe.target43_source->checked_calls;
    if(::testing::Test::HasFailure())throw std::runtime_error("converted target43 original numerical check failed before publication");
    result.mhc=result.outputs[recipe.target43_mhc];return result;
  }
  if(recipe.mlx43) {
    auto actual=[&](Value v){return typed<float>(result.outputs[mlx_port(recipe,v)]);};auto ids=[&](Value v){return typed<std::int32_t>(result.outputs[mlx_port(recipe,v)]);};
    const auto &data=*recipe.mlx->parameters.source;const bool full=!recipe.mlx->full_checked;
    EXPECT_TRUE(mlx_target_reference::observe43(data,*recipe.mlx43,actual,data.prefix()));
    mlx_reference::affine_projection(data,*recipe.mlx_target_query,data.prefix()+"self_attn.q_a_proj.",1536,4096,4,actual,full);
    mlx_reference::observe_norm(data,*recipe.mlx_target_query_norm,data.prefix()+"self_attn.q_a_layernorm.weight",actual,1536);
    const auto &mla=*recipe.mlx_target_mla;
    mlx_reference::affine_projection(data,mla.query,data.prefix()+"self_attn.q_b_proj.",16384,1536,4,actual,full);
    mlx_reference::per_head(data,mla.absorbed,false,actual,full);mlx_reference::affine_projection(data,mla.kv,data.prefix()+"self_attn.kv_a_proj_with_mqa.",512,4096,4,actual,full);
    mlx_reference::observe_norm(data,mla.latent_norm,data.prefix()+"self_attn.kv_a_layernorm.weight",actual,512);
    mlx_reference::per_head(data,mla.value,true,actual,full);mlx_reference::affine_projection(data,mla.output,data.prefix()+"self_attn.o_proj.",4096,16384,4,actual,full);
    observe_mlx_mla(access,before,result.state,recipe,result.outputs,mla,full);mlx_reference::observe_moe(data,recipe.mlx43->moe,actual,ids,full);
    recipe.mlx->full_checked=true;++recipe.target43_source->checked_calls;
    if(::testing::Test::HasFailure())throw std::runtime_error("MLX target43 original local reference failed before publication");
    result.mhc=result.outputs[recipe.target43_mhc];return result;
  }
  observe_attention(access,before,result.state,recipe,result.outputs,nx,position);
  if(recipe.original43) {
    if(!recipe.target43_source)throw std::runtime_error("original target43 numerical source missing");
    const auto actual=[&](Value value) {
      const auto found=std::find_if(recipe.original43_ports.begin(),recipe.original43_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
      if(found!=recipe.original43_ports.end())return read(result.outputs[found->second]);
      const auto moe=std::find_if(recipe.moe_ports.begin(),recipe.moe_ports.end(),[&](const auto &entry){return entry.first.ordinal()==value.ordinal();});
      if(moe==recipe.moe_ports.end())throw std::runtime_error("original target43 actual observation missing");
      return read(result.outputs[moe->second]);
    };
    const auto bytes=actual(recipe.original43->ports.attention_hc.residual);
    std::vector<std::uint16_t> bits(bytes.size()/sizeof(std::uint16_t));std::memcpy(bits.data(),bytes.data(),bytes.size());
    EXPECT_TRUE(index_reference::expect_bfloat_store(bits,typed<float>(residual),"explicit upstream42 fixture -> original target43 BF16 input"));
    EXPECT_TRUE(target43_local::observe(*recipe.target43_source->parameters.source,*recipe.original43,
        recipe.target43_query,actual,!recipe.target43_source->full_checked,recipe.sparse->arithmetic));
    observe_original_moe(recipe,result.outputs);
    recipe.target43_source->full_checked=true;++recipe.target43_source->checked_calls;
  } else {
    // The actual MLA projection was independently checked above. Use its actual
    // bounded result to check the next mHC/FFN boundary, never to create input.
    const auto expected = recipe.target43_reference(typed<float>(residual), typed<float>(result.outputs[recipe.output]));
    for (std::size_t i = 0; i < target43::Ports::count; ++i) {
      SCOPED_TRACE(target43::kPortNames[i]);
      sem::expect_output(recipe.fragment, typed<float>(result.outputs[recipe.target43_ports[i]]), expected[i]);
    }
  }
  result.mhc = result.outputs[recipe.target43_mhc];
  return result;
}

std::vector<float> target_residual_values(std::int32_t token) {
  // Generated preceding-layer input for THIS token. Each stream is distinct;
  // no previous token's mHC output or observed result is recreated here.
  std::vector<float> values;
  for (std::int32_t stream = 0; stream < target::kStreams; ++stream) {
    auto row = d5_gguf_test::activation(kHidden, 73 + 14 * token + 19 * stream);
    for (auto &value : row) value += float(stream) / 32.0F;
    values.insert(values.end(), row.begin(), row.end());
  }
  return values;
}

void run_target(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  auto component = target::build();
  const auto ports = component.ports;
  auto target_reference = std::move(component.evaluate);
  Recipe target_recipe; target_recipe.state_ports.fill(std::numeric_limits<std::size_t>::max());
  target_recipe.fragment = std::move(component.fragment); // one native constant bank
  const auto cursor = target_recipe.fragment.input(sem::dense({1}, c::DataType::kInt32));
  const auto cursor_port = expose(target_recipe, increment(target_recipe.fragment, cursor));
  // Ten bounded result generations retain the actual base/partial/full and
  // independent branches. The default remains five for existing consumers.
  auto target_plan_owner=Program::create(std::move(target_recipe), context, owner, cuda, 10);
  auto &target_plan=*target_plan_owner;
  auto first_plan_owner=Program::create(build(true, 2054), context, owner, cuda);
  auto &first_plan=*first_plan_owner;
  auto partial_plan_owner=Program::create(build(true, 2055), context, owner, cuda);
  auto &partial_plan=*partial_plan_owner;
  auto full_plan_owner=Program::create(build(true, 2056), context, owner, cuda);
  auto &full_plan=*full_plan_owner;
  auto skip_owner=Program::create(build(false, 0, 3), context, owner, cuda);
  auto &skip=*skip_owner;
  auto journal_owner=Program::create(build_catchup(), context, owner, cuda);
  auto &journal=*journal_owner;
  auto direct_owner=Program::create(build_catchup(true), context, owner, cuda);
  auto &direct=*direct_owner;
  Recipe policy_recipe; policy_recipe.fragment = pending_policy();
  auto policy_owner=Program::create(std::move(policy_recipe), context, owner, cuda);
  auto &policy=*policy_owner;
  auto queue = take(context.create_queue(), "combined target/NextN queue");
  auto initializer_owner=Program::create(initial_pool_recipe(), context, owner, cuda, 2);
  auto &initializer=*initializer_owner;
  std::size_t submissions = 0, target_submissions = 0, peak_backing = 0;
  const auto target_run = [&](std::span<const c::ReadLease> inputs) {
    const auto expected = target_reference(typed<float>(inputs[0]),
        typed<float>(inputs[1]), typed<float>(inputs[2]));
    auto output = target_plan.execute(queue, inputs); ++submissions; ++target_submissions;
    for (std::size_t port = 0; port < target::Ports::count; ++port) {
      SCOPED_TRACE(target::kPortNames[port]);
      sem::expect_output(target_plan.recipe.fragment, typed<float>(output[port]), expected[port]);
    }
    EXPECT_EQ(typed<std::int32_t>(output[cursor_port]).at(0), typed<std::int32_t>(inputs[3]).at(0) + 1);
    return output;
  };
  std::vector<c::ReadLease> warmup;
  {
    std::vector<float> history(3 * target::kChannels), matrix(target::kHeads * target::kDepth * target::kDepth);
    for (std::size_t i = 0; i < history.size(); ++i) history[i] = float(std::sin(double(i) * 0.013) * 0.01);
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = float(std::sin(double(i) * 0.007) * 0.0005);
    const std::array inputs{upload(owner, sem::bytes(target_residual_values(0))),
        upload(owner, sem::bytes(history)), upload(owner, sem::bytes(matrix)),
        upload(owner, sem::bytes(std::vector<std::int32_t>{2053}))};
    warmup = target_run(inputs); // actual previous-token target tap, not a host seed
  }
  const std::array extra_specs{
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.history]), .resource = owner.ref()},
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.state]), .resource = owner.ref()},
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.tap]), .resource = owner.ref()},
      c::StateEntrySpec{.description = sem::dense({1}, c::DataType::kInt32), .resource = owner.ref()}};
  constexpr std::size_t state_budget = 512u << 20;
  for (const int accepted_rows : {1, 2}) {
    SCOPED_TRACE(accepted_rows);
    // The larger fixed budget accounts actual root backing capacities retained
    // by both KDA/history and NextN branches. It does not disable accounting.
    auto state = State::create(owner, extra_specs, state_budget);
    {
      const auto target_members = [&](std::span<const c::ReadLease> output) {
        return std::array{c::StateMember{state.entries[kTargetHistory], output[ports.history]},
            c::StateMember{state.entries[kTargetState], output[ports.state]},
            c::StateMember{state.entries[kTargetTap], output[ports.tap]},
            c::StateMember{state.entries[kTargetCursor], output[cursor_port]}};
      };
      const auto initial_target = target_members(warmup);
      const auto base = state.initial(owner, initializer, queue, 2054, initial_target, warmup[ports.tap]); ++submissions;
      same_storage(state.member(base, kHiddenState), state.member(base, kTargetTap));
      const auto base_bytes = image(state, base);
      auto head = take(c::StateHead::create(state.store, base), "single combined target/NextN head");
      const auto combined = [&](Program &plan, const c::StateVersion &from,
                                std::int32_t pending, bool target_seed,
                                c::ReadLease *retained_nx = nullptr) {
        if (!state.enabled || (!plan.recipe.fresh && !state.qualifies(from)))
          throw std::runtime_error("combined consumer refused stale epoch before target preparation");
        if (typed<std::int32_t>(state.member(from, kTargetCursor)).at(0) != state.scalar(from, kMlaCursor))
          throw std::runtime_error("target/NextN accepted token frontiers differ");
        if (target_seed && !plan.recipe.fresh)
          throw std::runtime_error("target seed is only bound at a fresh NextN iteration");
        const auto before = image(state, from);
        const std::array target_inputs{upload(owner, sem::bytes(target_residual_values(state.scalar(from, kPending)))),
            state.member(from, kTargetHistory), state.member(from, kTargetState), state.member(from, kTargetCursor)};
        auto produced = target_run(target_inputs);
        const auto members = target_members(produced);
        const auto seed = target_seed ? state.member(from, kTargetTap) : c::ReadLease{};
        // Target verifies the current pending token, while NextN receives the
        // PREVIOUS target tap (fresh) or actual draft hidden (same-epoch skip).
        // Both sets of results enter exactly one StateDraft/finalize below.
        auto next = execute(state, plan, queue, owner, from, pending, retained_nx, members, seed); ++submissions;
        for (const auto &member : members)
          same_storage(take(next.member(member.entry), "combined actual target output"), member.resource);
        EXPECT_EQ(typed<std::int32_t>(state.member(next, kTargetCursor)).at(0), state.scalar(next, kMlaCursor));
        EXPECT_EQ(image(state, from), before);
        peak_backing = std::max(peak_backing, state.store.stats().backing_bytes);
        EXPECT_LE(state.store.stats().backing_bytes, state_budget);
        return next;
      };
      const auto first = combined(first_plan, base, 2, true);
      const auto first_bytes = image(state, first);
      const auto second = combined(skip, first, 4, false);
      const auto second_bytes = image(state, second);
      EXPECT_NE(read(state.member(first, kTargetState)), read(state.member(second, kTargetState)));
      c::StateVersion accepted = second;
      if (accepted_rows == 1) {
        const std::array input{upload(owner, sem::bytes(std::vector<std::int32_t>{3}))};
        auto output = policy.execute(queue, input); ++submissions;
        auto draft = take(state.store.begin(first), "combined prefix1 target-policy correction");
        check(draft.replace(state.entries[kPending], output[0]), "correct only actual pending token");
        accepted = take(draft.finalize(), "one complete target/draft accepted prefix1");
        for (std::size_t entry = kTargetHistory; entry <= kTargetCursor; ++entry)
          same_storage(state.member(accepted, entry), state.member(first, entry));
      }
      check(head.publish(base.identity(), accepted), "atomically accept target/draft same prefix");
      EXPECT_FALSE(head.publish(base.identity(), second).ok());
      const auto accepted_bytes = image(state, accepted);
      auto cancelled = take(state.store.begin(accepted), "incomplete combined target splice");
      check(cancelled.replace(state.entries[kTargetTap], state.member(second, kTargetTap)), "stage target-only speculative member");
      EXPECT_FALSE(cancelled.replace(state.entries[kTargetHistory], state.member(first, kPending)).ok());
      check(cancelled.cancel(), "cancel incomplete combined transaction");
      EXPECT_EQ(image(state, take(head.current(), "combined head after cancellation")), accepted_bytes);
      state.enabled = false;
      EXPECT_THROW((void)execute(state, skip, queue, owner, accepted, 5), std::runtime_error);
      c::StateVersion caught = accepted;
      if (accepted_rows == 2) {
        caught = catch_up(state, journal, queue, head); ++submissions;
        check(head.publish(accepted.identity(), caught), "publish accepted catchup within combined state");
      } else EXPECT_THROW((void)catch_up(state, journal, queue, head), std::runtime_error);
      for (std::size_t entry = kTargetHistory; entry <= kTargetCursor; ++entry)
        same_storage(state.member(caught, entry), state.member(accepted, entry));
      const auto caught_bytes = image(state, caught);
      EXPECT_NE(read(state.member(caught, kHiddenState)), read(state.member(caught, kTargetTap)))
          << "fresh reseed must bind the target producer rather than retained draft hidden";
      state.epoch = 8; state.enabled = true;
      auto &next_plan = accepted_rows == 1 ? partial_plan : full_plan;
      const auto next = combined(next_plan, take(head.current(), "accepted target tap for epoch8"), 5, true);
      check(head.publish(caught.identity(), next), "single publication of epoch8 target and fresh NextN seed");
      EXPECT_EQ(state.facts(next).epoch, 8u);
      EXPECT_NE(read(state.member(next, kSelection)), read(state.member(first, kSelection)));
      EXPECT_FALSE(state.qualifies(accepted));

      // Recompute from BASE without rebinding any candidate result. Reuse the
      // SAME head for nominating this independent accepted checkpoint; no
      // separate target head/store or target-only publication exists.
      state.epoch = 7; state.enabled = true;
      const auto reference_first = combined(first_plan, base, accepted_rows == 1 ? 3 : 2, true);
      c::StateVersion reference_caught = reference_first;
      if (accepted_rows == 2) {
        c::ReadLease independent_nx;
        const auto reference_second = combined(skip, reference_first, 4, false, &independent_nx);
        check(head.publish(next.identity(), reference_second), "nominate independently recomputed accepted prefix2");
        state.enabled = false;
        reference_caught = catch_up(state, direct, queue, head, independent_nx); ++submissions;
        check(head.publish(reference_second.identity(), reference_caught), "publish independent direct catchup");
      } else check(head.publish(next.identity(), reference_caught), "nominate independently recomputed accepted prefix1");
      EXPECT_EQ(image(state, reference_caught), caught_bytes);
      state.epoch = 8; state.enabled = true;
      const auto reference_next = combined(next_plan, take(head.current(), "independent accepted target tap"), 5, true);
      check(head.publish(reference_caught.identity(), reference_next), "publish independently continued whole state");
      EXPECT_EQ(image(state, next), image(state, reference_next));
      EXPECT_EQ(image(state, base), base_bytes); EXPECT_EQ(image(state, first), first_bytes);
      EXPECT_EQ(image(state, second), second_bytes); EXPECT_EQ(image(state, accepted), accepted_bytes);
      EXPECT_EQ(image(state, caught), caught_bytes);
      EXPECT_FALSE(state.member(second, kTargetTap).try_write());
      EXPECT_FALSE(state.member(second, kTargetState).try_write());
      state.enabled = false;
      EXPECT_FALSE(state.qualifies(next)); EXPECT_FALSE(state.qualifies(first));
    }
    check(state.store.collect(), "retire combined target/draft prefix versions");
    EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  }
  warmup.clear();
  for (const auto *program : {&initializer, &target_plan, &first_plan, &partial_plan, &full_plan, &skip, &journal, &direct, &policy}) {
    EXPECT_EQ(program->output_stats().live_blocks, 0u); EXPECT_EQ(program->output_stats().active_uses, 0u);
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_EQ(submissions, 28u); EXPECT_EQ(target_submissions, 12u);
  ::testing::Test::RecordProperty("initialization_submissions", 2);
  ::testing::Test::RecordProperty("logical_index_state_bytes", std::to_string(kPoolCapacity * (kIndexDepth + 4) + kRing * kIndexDepth * 4));
  ::testing::Test::RecordProperty("actual_submissions", int(submissions));
  ::testing::Test::RecordProperty("target_submissions", int(target_submissions));
  ::testing::Test::RecordProperty("state_budget_bytes", std::to_string(state_budget));
  ::testing::Test::RecordProperty("peak_state_backing_bytes", std::to_string(peak_backing));
  ::testing::Test::RecordProperty("target_output_generations", 10);
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("component_schema", "17 NextN/indexer resources plus epoch metadata and4 target resources: raw QKV history,KDA matrix,finalnorm tap,I32 cursor; one StateStore/Head/Draft");
  ::testing::Test::RecordProperty("journey", "actual target warmup->A/B verify pending->accept1/2->accepted journal catchup->epoch8 real target-tap reseed; independent base recomputation/direct nx catchup; rejected tail,stale CAS,cancellation,old readonly leases");
  ::testing::Test::RecordProperty("scope", "generated primary GGUF/F32 layer44 KDA/conv/two mHC branches/sparse FFN/finalnorm target tap composed with complete NextN; upper-layer4H remains generated; default UE8M0 logical cache/query codec; no vendorpage ABI,full45-layer original model,natural acceptance or throughput claim");
}
void run_target43(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  auto component = target::build();
  const auto ports = component.ports;
  auto target_reference = std::move(component.evaluate);
  Recipe target_recipe; target_recipe.state_ports.fill(std::numeric_limits<std::size_t>::max());
  target_recipe.fragment = std::move(component.fragment); // one native constant bank
  const auto cursor = target_recipe.fragment.input(sem::dense({1}, c::DataType::kInt32));
  const auto cursor_port = expose(target_recipe, increment(target_recipe.fragment, cursor));
  // Ten bounded result generations retain the actual base/partial/full and
  // independent branches. The default remains five for existing consumers.
  auto target_plan_owner=Program::create(std::move(target_recipe), context, owner, cuda, 10);
  auto &target_plan=*target_plan_owner;
  auto first_plan_owner=Program::create(build(true, 2054), context, owner, cuda);
  auto &first_plan=*first_plan_owner;
  auto partial_plan_owner=Program::create(build(true, 2055), context, owner, cuda);
  auto &partial_plan=*partial_plan_owner;
  auto full_plan_owner=Program::create(build(true, 2056), context, owner, cuda);
  auto &full_plan=*full_plan_owner;
  auto skip_owner=Program::create(build(false, 0, 3), context, owner, cuda);
  auto &skip=*skip_owner;
  auto journal_owner=Program::create(build_catchup(), context, owner, cuda);
  auto &journal=*journal_owner;
  auto direct_owner=Program::create(build_catchup(true), context, owner, cuda);
  auto &direct=*direct_owner;
  Recipe policy_recipe; policy_recipe.fragment = pending_policy();
  auto policy_owner=Program::create(std::move(policy_recipe), context, owner, cuda);
  auto &policy=*policy_owner;
  auto queue = take(context.create_queue(), "joint target43/target44/NextN queue");
  auto warm43_plan_owner=Program::create(build_target43(2053), context, owner, cuda, 1);
  auto &warm43_plan=*warm43_plan_owner;
  auto token43_a_owner=Program::create(build_target43(2054), context, owner, cuda);
  auto &token43_a=*token43_a_owner;
  auto token43_b_owner=Program::create(build_target43(2055), context, owner, cuda);
  auto &token43_b=*token43_b_owner;
  auto token43_c_owner=Program::create(build_target43(2056), context, owner, cuda);
  auto &token43_c=*token43_c_owner;
  const auto choose43 = [&](std::int32_t position) -> Program & {
    if (position == 2054) return token43_a;
    if (position == 2055) return token43_b;
    if (position == 2056) return token43_c;
    throw std::runtime_error("target43 token frontier is outside the finite prepared domain");
  };
  auto initializer_owner=Program::create(initial_pool_recipe(), context, owner, cuda, 3);
  auto &initializer=*initializer_owner;
  std::size_t submissions = 0, target_submissions = 0, target43_submissions = 0, peak_backing = 0;
  const auto target_run = [&](std::span<const c::ReadLease> inputs) {
    const auto expected = target_reference(typed<float>(inputs[0]),
        typed<float>(inputs[1]), typed<float>(inputs[2]));
    auto output = target_plan.execute(queue, inputs); ++submissions; ++target_submissions;
    for (std::size_t port = 0; port < target::Ports::count; ++port) {
      SCOPED_TRACE(target::kPortNames[port]);
      sem::expect_output(target_plan.recipe.fragment, typed<float>(output[port]), expected[port]);
    }
    EXPECT_EQ(typed<std::int32_t>(output[cursor_port]).at(0), typed<std::int32_t>(inputs[3]).at(0) + 1);
    return output;
  };
  auto initial43 = initial_sparse(owner, initializer, queue, 2053); ++submissions;
  auto warm43 = execute_target43(warm43_plan, queue, initial43,
      upload(owner, sem::bytes(target_residual_values(0)))); ++submissions; ++target43_submissions;
  std::vector<c::ReadLease> warmup;
  {
    std::vector<float> history(3 * target::kChannels), matrix(target::kHeads * target::kDepth * target::kDepth);
    for (std::size_t i = 0; i < history.size(); ++i) history[i] = float(std::sin(double(i) * 0.013) * 0.01);
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = float(std::sin(double(i) * 0.007) * 0.0005);
    const std::array inputs{warm43.mhc,
        upload(owner, sem::bytes(history)), upload(owner, sem::bytes(matrix)),
        upload(owner, sem::bytes(std::vector<std::int32_t>{2053}))};
    warmup = target_run(inputs); // actual previous-token target tap, not a host seed
  }
  std::vector<c::StateEntrySpec> extra_specs{
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.history]), .resource = owner.ref()},
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.state]), .resource = owner.ref()},
      c::StateEntrySpec{.description = target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[ports.tap]), .resource = owner.ref()},
      c::StateEntrySpec{.description = sem::dense({1}, c::DataType::kInt32), .resource = owner.ref()}};
  const auto sparse_desc = descriptions();
  for (const auto entry : kSparseEntries) extra_specs.push_back({.description = sparse_desc[entry], .resource = owner.ref()});
  constexpr std::size_t state_budget = 512u << 20;
  for (const int accepted_rows : {1, 2}) {
    SCOPED_TRACE(accepted_rows);
    // The larger fixed budget accounts actual root backing capacities retained
    // by both KDA/history and NextN branches. It does not disable accounting.
    auto state = State::create(owner, extra_specs, state_budget);
    {
      const auto target_members = [&](std::span<const c::ReadLease> output, const SparseSnapshot &sparse) {
        std::vector<c::StateMember> members{{state.entries[kTargetHistory], output[ports.history]},
            {state.entries[kTargetState], output[ports.state]}, {state.entries[kTargetTap], output[ports.tap]},
            {state.entries[kTargetCursor], output[cursor_port]}};
        for (std::size_t i = 0; i < kSparseEntries.size(); ++i)
          members.push_back({state.entries[kTarget43Begin + i], sparse.values[i]});
        return members;
      };
      const auto initial_target = target_members(warmup, warm43.state);
      const auto base = state.initial(owner, initializer, queue, 2054, initial_target, warmup[ports.tap]); ++submissions;
      same_storage(state.member(base, kHiddenState), state.member(base, kTargetTap));
      const auto base_bytes = image(state, base);
      auto head = take(c::StateHead::create(state.store, base), "single combined target/NextN head");
      const auto combined = [&](Program &plan, const c::StateVersion &from,
                                std::int32_t pending, bool target_seed,
                                c::ReadLease *retained_nx = nullptr) {
        if (!state.enabled || (!plan.recipe.fresh && !state.qualifies(from)))
          throw std::runtime_error("combined consumer refused stale epoch before target preparation");
        if (typed<std::int32_t>(state.member(from, kTargetCursor)).at(0) != state.scalar(from, kMlaCursor))
          throw std::runtime_error("target/NextN accepted token frontiers differ");
        if (target_seed && !plan.recipe.fresh)
          throw std::runtime_error("target seed is only bound at a fresh NextN iteration");
        const auto before = image(state, from);
        auto produced43 = execute_target43(choose43(state.scalar(from, kMlaCursor)), queue,
            target43_snapshot(state, from), upload(owner, sem::bytes(target_residual_values(state.scalar(from, kPending)))));
        ++submissions; ++target43_submissions;
        const std::array target_inputs{produced43.mhc, state.member(from, kTargetHistory),
            state.member(from, kTargetState), state.member(from, kTargetCursor)};
        auto produced = target_run(target_inputs);
        const auto members = target_members(produced, produced43.state);
        const auto seed = target_seed ? state.member(from, kTargetTap) : c::ReadLease{};
        // Target verifies the current pending token, while NextN receives the
        // PREVIOUS target tap (fresh) or actual draft hidden (same-epoch skip).
        // Both sets of results enter exactly one StateDraft/finalize below.
        auto next = execute(state, plan, queue, owner, from, pending, retained_nx, members, seed); ++submissions;
        for (const auto &member : members)
          same_storage(take(next.member(member.entry), "combined actual target output"), member.resource);
        EXPECT_EQ(typed<std::int32_t>(state.member(next, kTargetCursor)).at(0), state.scalar(next, kMlaCursor));
        const auto target43_state = target43_snapshot(state, next);
        EXPECT_EQ(SparseAccess{}.scalar(target43_state, kMlaCursor), state.scalar(next, kMlaCursor));
        EXPECT_EQ(SparseAccess{}.scalar(target43_state, kIndexCursor), state.scalar(next, kMlaCursor));
        for (std::size_t i = 0; i < kSparseEntries.size(); ++i) {
          same_storage(target43_state.values[i], produced43.state.values[i]);
          const auto a = take(target43_state.values[i].view(), "actual target43 member");
          const auto b = take(state.member(next, kSparseEntries[i]).view(), "actual NextN member");
          EXPECT_FALSE(a.resource() == b.resource() && a.native_handle() == b.native_handle() && a.offset() == b.offset());
        }
        EXPECT_EQ(image(state, from), before);
        peak_backing = std::max(peak_backing, state.store.stats().backing_bytes);
        EXPECT_LE(state.store.stats().backing_bytes, state_budget);
        return next;
      };
      const auto first = combined(first_plan, base, 2, true);
      const auto first_bytes = image(state, first);
      const auto second = combined(skip, first, 4, false);
      const auto second_bytes = image(state, second);
      const auto second43 = target43_snapshot(state, second);
      EXPECT_NE(read(second43[kSelection]), read(state.member(second, kSelection)))
          << "target43 closes pool independently; NextN skip keeps its own full prior seed";
      EXPECT_EQ(SparseAccess{}.scalar(second43, kPoolCount), 514);
      EXPECT_EQ(state.scalar(second, kPoolCount), 513);
      EXPECT_NE(read(state.member(first, kTargetState)), read(state.member(second, kTargetState)));
      c::StateVersion accepted = second;
      if (accepted_rows == 1) {
        const std::array input{upload(owner, sem::bytes(std::vector<std::int32_t>{3}))};
        auto output = policy.execute(queue, input); ++submissions;
        auto draft = take(state.store.begin(first), "combined prefix1 target-policy correction");
        check(draft.replace(state.entries[kPending], output[0]), "correct only actual pending token");
        accepted = take(draft.finalize(), "one complete target/draft accepted prefix1");
        for (std::size_t entry = kTargetHistory; entry < kTarget43Begin + kSparseEntries.size(); ++entry)
          same_storage(state.member(accepted, entry), state.member(first, entry));
      }
      check(head.publish(base.identity(), accepted), "atomically accept target/draft same prefix");
      EXPECT_FALSE(head.publish(base.identity(), second).ok());
      const auto accepted_bytes = image(state, accepted);
      auto cancelled = take(state.store.begin(accepted), "incomplete combined target splice");
      check(cancelled.replace(state.entries[kTargetTap], state.member(second, kTargetTap)), "stage target-only speculative member");
      EXPECT_FALSE(cancelled.replace(state.entries[kTargetHistory], state.member(first, kPending)).ok());
      check(cancelled.cancel(), "cancel incomplete combined transaction");
      EXPECT_EQ(image(state, take(head.current(), "combined head after cancellation")), accepted_bytes);
      state.enabled = false;
      EXPECT_THROW((void)execute(state, skip, queue, owner, accepted, 5), std::runtime_error);
      c::StateVersion caught = accepted;
      if (accepted_rows == 2) {
        caught = catch_up(state, journal, queue, head); ++submissions;
        check(head.publish(accepted.identity(), caught), "publish accepted catchup within combined state");
      } else EXPECT_THROW((void)catch_up(state, journal, queue, head), std::runtime_error);
      for (std::size_t entry = kTargetHistory; entry < kTarget43Begin + kSparseEntries.size(); ++entry)
        same_storage(state.member(caught, entry), state.member(accepted, entry));
      const auto caught_bytes = image(state, caught);
      EXPECT_NE(read(state.member(caught, kHiddenState)), read(state.member(caught, kTargetTap)))
          << "fresh reseed must bind the target producer rather than retained draft hidden";
      state.epoch = 8; state.enabled = true;
      auto &next_plan = accepted_rows == 1 ? partial_plan : full_plan;
      const auto next = combined(next_plan, take(head.current(), "accepted target tap for epoch8"), 5, true);
      check(head.publish(caught.identity(), next), "single publication of epoch8 target and fresh NextN seed");
      EXPECT_EQ(state.facts(next).epoch, 8u);
      EXPECT_NE(read(state.member(next, kSelection)), read(state.member(first, kSelection)));
      EXPECT_FALSE(state.qualifies(accepted));

      // Recompute from BASE without rebinding any candidate result. Reuse the
      // SAME head for nominating this independent accepted checkpoint; no
      // separate target head/store or target-only publication exists.
      state.epoch = 7; state.enabled = true;
      const auto reference_first = combined(first_plan, base, accepted_rows == 1 ? 3 : 2, true);
      c::StateVersion reference_caught = reference_first;
      if (accepted_rows == 2) {
        c::ReadLease independent_nx;
        const auto reference_second = combined(skip, reference_first, 4, false, &independent_nx);
        check(head.publish(next.identity(), reference_second), "nominate independently recomputed accepted prefix2");
        state.enabled = false;
        reference_caught = catch_up(state, direct, queue, head, independent_nx); ++submissions;
        check(head.publish(reference_second.identity(), reference_caught), "publish independent direct catchup");
      } else check(head.publish(next.identity(), reference_caught), "nominate independently recomputed accepted prefix1");
      EXPECT_EQ(image(state, reference_caught), caught_bytes);
      state.epoch = 8; state.enabled = true;
      const auto reference_next = combined(next_plan, take(head.current(), "independent accepted target tap"), 5, true);
      check(head.publish(reference_caught.identity(), reference_next), "publish independently continued whole state");
      EXPECT_EQ(image(state, next), image(state, reference_next));
      EXPECT_EQ(image(state, base), base_bytes); EXPECT_EQ(image(state, first), first_bytes);
      EXPECT_EQ(image(state, second), second_bytes); EXPECT_EQ(image(state, accepted), accepted_bytes);
      EXPECT_EQ(image(state, caught), caught_bytes);
      EXPECT_FALSE(state.member(second, kTargetTap).try_write());
      EXPECT_FALSE(state.member(second, kTargetState).try_write());
      state.enabled = false;
      EXPECT_FALSE(state.qualifies(next)); EXPECT_FALSE(state.qualifies(first));
    }
    check(state.store.collect(), "retire combined target/draft prefix versions");
    EXPECT_EQ(state.store.stats().versions, 0u); EXPECT_EQ(state.store.stats().drafts, 0u);
  }
  warmup.clear(); warm43 = {}; initial43 = {};
  for (const auto *program : {&warm43_plan, &token43_a, &token43_b, &token43_c, &initializer, &target_plan, &first_plan, &partial_plan, &full_plan, &skip, &journal, &direct, &policy}) {
    EXPECT_EQ(program->output_stats().live_blocks, 0u); EXPECT_EQ(program->output_stats().active_uses, 0u);
    for (std::size_t i = 0; i < program->parameters.size(); ++i)
      EXPECT_EQ(read(program->parameters[i]), program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_EQ(submissions, 41u); EXPECT_EQ(target_submissions, 12u); EXPECT_EQ(target43_submissions, 12u);
  ::testing::Test::RecordProperty("initialization_submissions", 3);
  ::testing::Test::RecordProperty("logical_index_state_bytes", std::to_string(kPoolCapacity * (kIndexDepth + 4) + kRing * kIndexDepth * 4));
  ::testing::Test::RecordProperty("actual_submissions", int(submissions));
  ::testing::Test::RecordProperty("target_submissions", int(target_submissions));
  ::testing::Test::RecordProperty("target43_submissions", int(target43_submissions));
  ::testing::Test::RecordProperty("state_budget_bytes", std::to_string(state_budget));
  ::testing::Test::RecordProperty("peak_state_backing_bytes", std::to_string(peak_backing));
  ::testing::Test::RecordProperty("target_output_generations", 10);
  ::testing::Test::RecordProperty("backend", cuda ? "ordinary CUDA Graph" : "ordinary CPU Graph");
  ::testing::Test::RecordProperty("component_schema", "34 resources plus epoch metadata:17 NextN,13 independent target43 MLA/indexer,4 target44 KDA/conv/tap/cursor; one StateStore/Head/Draft");
  ::testing::Test::RecordProperty("journey", "actual target43 UE8M0 MLA/indexer->43 mHC/FFN->actual43mhc lease->44 KDA/conv/mHC/finalnorm->NextN; accept1/2 crosses pool close; NextN accepted journal catches up; epoch8 direct target-tap reseed; independent base replay; rejected/stale/cancel/old leases");
  ::testing::Test::RecordProperty("scope", "generated primary GGUF/F32 layer42 four-stream input; actual13-state target43 attention at distinct immutable projection roles,34-resource joint43/44/NextN continuation; BF16 rings and default UE8M0 logical codec; no original checkpoint/full45-layer target,vendorpage ABI,natural acceptance or throughput claim");
}

struct TargetVerifierPorts final {
  std::vector<std::size_t> taps=std::vector<std::size_t>(2),row_ids=std::vector<std::size_t>(2);
  std::size_t stacked=0,logits=0,scores=0,ids=0;
};
struct TargetVerifierRecipe final { Recipe recipe;TargetVerifierPorts ports; };
TargetVerifierRecipe build_target_verifier(const std::shared_ptr<const OriginalNextn> &original,bool single_row=false,std::size_t requested_rows=2) {
  const auto rows=single_row?1u:requested_rows;
  if(rows<1||rows>3)throw std::runtime_error("GLM finite verifier supports one to three actual target taps");
  const auto join=[&](sem::Fragment &f,std::span<const Value> values) {
    auto result=values[0];for(std::size_t i=1;i<values.size();++i)result=f.one(c::Operation(c::Concat{.axis=0}),{result,values[i]});return result;
  };
  if(original&&original->converted) {
    TargetVerifierRecipe result;auto &r=result.recipe;r.original=original;r.profile_owner=original;r.converted=original->converted;r.state_ports.fill(std::numeric_limits<std::size_t>::max());
    result.ports.taps.resize(rows);result.ports.row_ids.resize(rows);
    auto &f=r.fragment;f.profile=r.converted->profile;f.region=0x4747464c56523248ULL;f.quantized=true;std::vector<Value> taps(rows);
    for(std::size_t i=0;i<rows;++i){taps[i]=f.input(sem::dense({1,kHidden}));result.ports.taps[i]=expose(r,taps[i],true);}
    const auto joined=join(f,taps);result.ports.stacked=expose(r,joined,true);
    const auto &bank=r.converted->parameters.at("output.weight");const auto head=external(r,original_index::ReadonlyRole{bank.name,bank.description,bank.owner,bank.lease});
    const auto logits=glm_gguf_projection_fragment::linear(f,joined,head).output;
    const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);result.ports.logits=expose(r,logits,true);result.ports.scores=expose(r,top[0],true);result.ports.ids=expose(r,top[1],true);
    for(std::size_t row=0;row<rows;++row)result.ports.row_ids[row]=expose(r,mlx_model::fp::reshape(f,mlx_model::fp::slice(f,top[1],0,std::int64_t(row),1),{1}),true);return result;
  }
  if(original&&original->mlx) {
    TargetVerifierRecipe result;auto &r=result.recipe;r.profile_owner=original;r.mlx=original->mlx;r.state_ports.fill(std::numeric_limits<std::size_t>::max());
    result.ports.taps.resize(rows);result.ports.row_ids.resize(rows);
    auto &f=r.fragment;f.profile=&mlx_runtime::kProfile;f.region=0x514d4c5856523248ULL;std::vector<Value> taps(rows);
    for(std::size_t i=0;i<rows;++i){taps[i]=f.input(sem::dense({1,kHidden}));result.ports.taps[i]=expose(r,taps[i],true);}
    const auto joined=join(f,taps);result.ports.stacked=expose(r,joined,true);
    const auto head=external(r,original->mlx->parameters.at("lm_head.weight"));const auto logits=f.one(c::model_ops::dense_linear_f32(),{joined,head});
    const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);result.ports.logits=expose(r,logits,true);result.ports.scores=expose(r,top[0],true);result.ports.ids=expose(r,top[1],true);
    for(std::size_t row=0;row<rows;++row)result.ports.row_ids[row]=expose(r,mlx_model::fp::reshape(f,mlx_model::fp::slice(f,top[1],0,std::int64_t(row),1),{1}),true);return result;
  }
  if(!original||!original->vocabulary)throw std::runtime_error("verifier requires its actual same-source vocabulary");
  const auto &vocabulary=*original->vocabulary;
  TargetVerifierRecipe result;auto &r=result.recipe;auto &f=r.fragment;
  result.ports.taps.resize(rows);result.ports.row_ids.resize(rows);
  r.profile_owner=original;
  r.state_ports.fill(std::numeric_limits<std::size_t>::max());
  f.profile=&original->profile_description;f.region=0x474c4d5645524946ULL;
  std::vector<Value> stored(rows);
  for(std::size_t row=0;row<rows;++row) {
    const auto actual=f.input(sem::dense({1,kHidden}));
    stored[row]=codec::cast(f,actual,c::DataType::kBFloat16);
    result.ports.taps[row]=expose(r,stored[row],true);
  }
  const auto joined=join(f,stored);
  result.ports.stacked=expose(r,joined,true);
  const auto head=external(r,vocabulary.at(shared_lm::Role::kHead));
  if(rows!=2) {
    const auto logits=f.one(c::model_ops::dense_linear_f32(),{joined,head});
    const auto top=f.append(c::Operation(c::TopK{.count=1,.stable=true}),{logits},2);
    result.ports.logits=expose(r,logits,true);result.ports.scores=expose(r,top[0],true);result.ports.ids=expose(r,top[1],true);
    for(std::size_t row=0;row<rows;++row)result.ports.row_ids[row]=expose(r,
        mlx_model::fp::reshape(f,mlx_model::fp::slice(f,top[1],0,std::int64_t(row),1),{1}),true);
  } else {
    const auto verifier=target_verifier::append(f,joined,head);
    result.ports.logits=expose(r,verifier.logits,true);
    result.ports.scores=expose(r,verifier.scores,true);
    result.ports.ids=expose(r,verifier.ids,true);
    for(std::size_t row=0;row<verifier.row_ids.size();++row)
      result.ports.row_ids[row]=expose(r,verifier.row_ids[row]);
  }
  return result;
}
void report_target_verifier(const Program &program,const TargetVerifierPorts &ports,
                            const shared_lm::Vocabulary &vocabulary,const c::StorageOwner &owner,bool cuda) {
  const auto &graph=program.logical.graph;
  const auto actual=[&](std::size_t port) {
    const auto copy=take(graph.producer(program.logical.outputs.at(port)),"verifier observation Copy");
    if(take(graph.operation_kind(copy),"verifier observation kind")!=c::OperationKind::kCopy)
      throw std::runtime_error("verifier observation must have its own actual Copy output");
    return take(graph.node_input(copy,0),"actual verifier observation value");
  };
  const auto raw=actual(ports.logits),input=actual(ports.stacked);
  const auto projection=take(graph.producer(raw),"actual target full-V head");
  const auto operation=take(graph.operation(projection),"actual target head operation");
  const auto &bank=vocabulary.at(shared_lm::Role::kHead);
  if(program.recipe.external.size()!=1||program.recipe.external[0].role.description!=bank.description||
      program.recipe.external[0].role.owner.ref()!=owner.ref())
    throw std::runtime_error("target verifier must share the original NextN vocabulary head bank");
  same_storage(program.recipe.external[0].role.lease,bank.lease);
  if(!operation.custom_as<c::model_ops::DenseLinearF32>()||
      take(graph.node_input(projection,0),"target head actual normalized BF16 taps")!=input||
      take(graph.node_input(projection,1),"target head same readonly original constant")!=program.logical.constants.at(program.recipe.external[0].constant)||
      take(graph.value_description(input),"target normalized tap shape")!=sem::dense({2,kHidden},c::DataType::kBFloat16)||
      take(graph.value_description(raw),"target full-V logits shape")!=sem::dense({2,target_verifier::kVocabulary}))
    throw std::runtime_error("target verifier must project both actual BF16 taps through the same original full-V head");
  const auto greedy=actual(ports.ids);const auto top=take(graph.producer(greedy),"actual target stable TopK1");
  const auto top_operation=take(graph.operation(top),"target TopK operation");
  const auto *topk=top_operation.get_if<c::TopK>();
  if(!topk||topk->count!=1||!topk->stable||take(graph.node_input(top,0),"target greedy full logits")!=raw||
      take(graph.result(top,1),"actual target I32 IDs")!=greedy||take(graph.result(top,0),"target actual scores")!=actual(ports.scores))
    throw std::runtime_error("target verifier must choose actual stable greedy IDs over every vocabulary row");
  for(const auto port:ports.row_ids)
    if(take(graph.operation_kind(take(graph.producer(program.logical.outputs[port]),"independent target row ID")),"target row ID output kind")!=c::OperationKind::kCopy||
        take(graph.value_description(program.logical.outputs[port]),"target row ID shape")!=sem::dense({1},c::DataType::kInt32))
      throw std::runtime_error("selected target pending must be an independently owned I32[1] output");
  std::size_t implementations=0;std::string provider;
  for(std::size_t step=0;step<program.cold_program->compiled.report.step_count();++step) {
    const auto selected=take(program.cold_program->compiled.report.step(step),"actual target verifier implementation");
    for(std::size_t node=0;node<selected.node_count;++node)
      if(take(program.cold_program->compiled.report.step_node(step,node),"target verifier selected node")==projection) {
        ++implementations;provider=selected.provider.name;
      }
  }
  if(implementations!=1||(cuda&&provider!="cublasLt"))
    throw std::runtime_error("target full-V BF16 head needs one actual implementation; CUDA must retain cuBLASLt without a full F32 head shadow");
  ::testing::Test::RecordProperty("target_verifier_head_provider",provider);
  std::printf("GLM target verifier actual R2 BF16[2,4096] x original BF16[154880,4096] provider=%s; same readonly NextN head owner\n",provider.c_str());
}

template<class Bank>
void report_plain_target_verifier(const Program &program,const TargetVerifierPorts &ports,
                            const Bank &bank,const c::StorageOwner &owner,bool cuda,bool packed) {
  (void)cuda;
  const auto &graph=program.logical.graph;
  const auto actual=[&](std::size_t port) {
    const auto copy=take(graph.producer(program.logical.outputs.at(port)),"verifier observation Copy");
    if(take(graph.operation_kind(copy),"verifier observation kind")!=c::OperationKind::kCopy)
      throw std::runtime_error("verifier observation must have its own actual Copy output");
    return take(graph.node_input(copy,0),"actual verifier observation value");
  };
  const auto raw=actual(ports.logits),input=actual(ports.stacked);
  const auto projection=take(graph.producer(raw),"actual target full-V head");
  const auto operation=take(graph.operation(projection),"actual target head operation");
  if(program.recipe.external.size()!=1||program.recipe.external[0].role.description!=bank.description||
      program.recipe.external[0].role.owner.ref()!=owner.ref())
    throw std::runtime_error("target verifier must share the original NextN vocabulary head bank");
  same_storage(program.recipe.external[0].role.lease,bank.lease);
  if((packed?!operation.get_if<c::MatMul>():!operation.custom_as<c::model_ops::DenseLinearF32>())||
      take(graph.node_input(projection,0),"target head actual normalized F32 taps")!=input||
      take(graph.node_input(projection,1),"target head same readonly original constant")!=program.logical.constants.at(program.recipe.external[0].constant)||
      take(graph.value_description(input),"target normalized tap shape")!=sem::dense({2,kHidden})||
      take(graph.value_description(raw),"target full-V logits shape")!=sem::dense({2,target_verifier::kVocabulary}))
    throw std::runtime_error("target verifier must project both actual F32 taps through the same original full-V head");
  const auto greedy=actual(ports.ids);const auto top=take(graph.producer(greedy),"actual target stable TopK1");
  const auto top_operation=take(graph.operation(top),"target TopK operation");
  const auto *topk=top_operation.get_if<c::TopK>();
  if(!topk||topk->count!=1||!topk->stable||take(graph.node_input(top,0),"target greedy full logits")!=raw||
      take(graph.result(top,1),"actual target I32 IDs")!=greedy||take(graph.result(top,0),"target actual scores")!=actual(ports.scores))
    throw std::runtime_error("target verifier must choose actual stable greedy IDs over every vocabulary row");
  for(const auto port:ports.row_ids)
    if(take(graph.operation_kind(take(graph.producer(program.logical.outputs[port]),"independent target row ID")),"target row ID output kind")!=c::OperationKind::kCopy||
        take(graph.value_description(program.logical.outputs[port]),"target row ID shape")!=sem::dense({1},c::DataType::kInt32))
      throw std::runtime_error("selected target pending must be an independently owned I32[1] output");
  std::size_t implementations=0;std::string provider;
  for(std::size_t step=0;step<program.cold_program->compiled.report.step_count();++step) {
    const auto selected=take(program.cold_program->compiled.report.step(step),"actual target verifier implementation");
    for(std::size_t node=0;node<selected.node_count;++node)
      if(take(program.cold_program->compiled.report.step_node(step,node),"target verifier selected node")==projection) {
        ++implementations;provider=selected.provider.name;
      }
  }
  if(implementations!=1||provider.empty())
    throw std::runtime_error("target full-V head needs one actual implementation with its original readonly storage");
  ::testing::Test::RecordProperty("target_verifier_head_provider",provider);
  std::printf("GLM target verifier actual R2 F32[2,4096] original head packed=%d provider=%s; same readonly NextN head owner\n",int(packed),provider.c_str());
}

void report_target_verifier(const Program &p,const TargetVerifierPorts &ports,const mlx_runtime::Source &source,const c::StorageOwner &owner,bool cuda) {
  report_plain_target_verifier(p,ports,source.parameters.at("lm_head.weight"),owner,cuda,false);
}
void report_target_verifier(const Program &p,const TargetVerifierPorts &ports,const gguf_source::Source &source,const c::StorageOwner &owner,bool cuda) {
  report_plain_target_verifier(p,ports,source.parameters.at("output.weight"),owner,cuda,true);
}

// K1 integration over computed target43/44 taps and the entire original45
// NextN. Layer42 inputs and initial histories remain generated; original43/44
// each bind their own authenticated parameters and bounded expert supply.
// Mandatory current t0 is present in BOTH checkpoints; zero accepted drafts
// selects first, never the state before consuming t0.
Recipe random_proposal_recipe(const c::ModelArtifactProfileDescription *profile) {
  Recipe result;result.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f=result.fragment;f.profile=profile;f.region=0x474c4d52414e4451ULL;
  const auto logits=f.input(sem::dense({1,target_verifier::kVocabulary}));
  const auto identity=f.input(sem::dense({1,4},c::DataType::kUInt32));
  const auto ports=model_sampling_fragment::append_proposal(f,logits,identity);
  f.outputs={ports.log_probabilities,ports.proposal,ports.valid};return result;
}
Recipe random_decision_recipe(const c::ModelArtifactProfileDescription *profile) {
  Recipe result;result.state_ports.fill(std::numeric_limits<std::size_t>::max());
  auto &f=result.fragment;f.profile=profile;f.region=0x474c4d52414e4450ULL;
  const auto target=f.input(sem::dense({2,target_verifier::kVocabulary}));
  const auto q=f.input(sem::dense({1,target_verifier::kVocabulary}));
  const auto proposal=f.input(sem::dense({1},c::DataType::kInt32));
  const auto accept=f.input(sem::dense({1,4},c::DataType::kUInt32));
  const auto final=f.input(sem::dense({1,4},c::DataType::kUInt32));
  const auto ports=model_sampling_fragment::append_decision(f,target,q,proposal,accept,final);
  f.outputs={ports.accepted,ports.pending,ports.valid,ports.target_log,ports.bonus_log};return result;
}
bool random_flag(const c::ReadLease &flag) {
  const auto value=typed<std::uint8_t>(flag);
  if(value.size()!=1||value[0]>1)throw std::runtime_error("actual sampling Bool scalar is invalid");
  return value[0]!=0;
}
// Independent KAT-qualified Philox and scalar F64 draw equations. This only
// chooses finite test addresses; no value is supplied to the native decision.
bool reference_random_accept(double log_alpha,std::array<std::uint32_t,4> identity,bool cuda) {
  const auto bits=sampling_test::philox({0,0,identity[2],identity[3]},{identity[0],identity[1]});
  if(!cuda) {
    const auto u=(double(bits[0])*0x1p20+double(bits[1]>>12)+.5)*0x1p-52;
    return u<std::exp(log_alpha);
  }
  const auto gumbel=[&](std::uint32_t hi,std::uint32_t lo) {
    const bool upper=(hi&0x80000000u)!=0;if(upper){hi=~hi;lo=~lo;}
    const auto lower_tail=std::ldexp(double(hi),-32)+std::ldexp(double(lo)+.5,-64);
    return -std::log(upper?-std::log1p(-lower_tail):-std::log(lower_tail));
  };
  const auto log_reject=std::log(-std::expm1(log_alpha));
  return log_alpha+gumbel(bits[0],bits[1])>log_reject+gumbel(bits[2],bits[3]);
}

void run_target_verifier(c::Context &context,const c::StorageOwner &owner,bool cuda,
                         std::shared_ptr<const OriginalNextn> original,std::shared_ptr<OriginalTarget44> target44={},
                         std::shared_ptr<OriginalTarget43> target43={},bool stochastic=false,bool controlled_random_branches=true,
                         hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1) {
  if(maximum_drafts<1||maximum_drafts>2||(!stochastic&&maximum_drafts!=1))
    throw std::runtime_error("finite GLM random component supports one or two configured drafts");
  if(stochastic)limits.validate(target_verifier::kVocabulary);
  const bool converted=original&&bool(original->converted);
  const bool plain=original&&(bool(original->mlx)||converted);
  if(target43&&!target44)throw std::runtime_error("original target43 verifier requires its actual original target44 successor");
  if(!original||(plain?(!target43||!target44||(converted?(!target43->converted||!target44->converted):(!target43->mlx||!target44->mlx))):(!original->vocabulary||!original->sparse->attention||!original->moe)))
    throw std::runtime_error("full-V target verification requires the complete original1762-role NextN and its same vocabulary");
  auto component=target44_recipe(target44);const auto target_ports=component.ports;
  auto target_reference=std::move(component.generated_reference);const auto cursor_port=component.cursor_port;
  // Cold programs have one heap owner. Vector growth moves only owning
  // handles; every named Program reference still points to the same object.
  std::vector<std::unique_ptr<Program>> prepared_programs;
  auto &target_plan=*prepared_programs.emplace_back(Program::create(std::move(component.recipe),context,owner,cuda,6));
  const auto extend_recipe=[&](std::int32_t position) {
    auto r=build(true,position,0,original);r.draft_extend=true;
    r.fragment.region=0x474c4d455854454eULL;return r;
  };
  auto &first_plan=*prepared_programs.emplace_back(Program::create(extend_recipe(2054),context,owner,cuda,3));
  auto &next_after_zero=*prepared_programs.emplace_back(Program::create(extend_recipe(2055),context,owner,cuda,3));
  auto &next_after_one=*prepared_programs.emplace_back(Program::create(extend_recipe(2056),context,owner,cuda,3));
  auto &skip=*prepared_programs.emplace_back(Program::create(build(false,0,3,original),context,owner,cuda,3));
  auto &journal=*prepared_programs.emplace_back(Program::create(build_catchup(false,2055,original),context,owner,cuda,2));
  auto &warm43_plan=*prepared_programs.emplace_back(Program::create(build_target43(2053,target43),context,owner,cuda,1));
  auto &token43_a=*prepared_programs.emplace_back(Program::create(build_target43(2054,target43),context,owner,cuda,3));
  auto &token43_b=*prepared_programs.emplace_back(Program::create(build_target43(2055,target43),context,owner,cuda,3));
  auto &token43_c=*prepared_programs.emplace_back(Program::create(build_target43(2056,target43),context,owner,cuda,3));
  auto &token43_d=*prepared_programs.emplace_back(Program::create(build_target43(2057,target43),context,owner,cuda,2));
  auto &handoff_zero=*prepared_programs.emplace_back(Program::create(build(false,0,0,original),context,owner,cuda,2));
  auto &handoff_one=*prepared_programs.emplace_back(Program::create(build(false,0,1,original),context,owner,cuda,2));
  const auto choose43=[&](std::int32_t position)->Program& {
    if(position==2054)return token43_a;if(position==2055)return token43_b;if(position==2056)return token43_c;if(position==2057)return token43_d;
    throw std::runtime_error("target verifier requires its actual finite prepared target43 frontier");
  };
  auto initial_recipe=initial_pool_recipe(plain);if(converted)initial_recipe.fragment.profile=original->converted->profile;
  auto &initializer=*prepared_programs.emplace_back(Program::create(std::move(initial_recipe),context,owner,cuda,4));
  auto verifier_recipe=build_target_verifier(original);const auto verifier_ports=verifier_recipe.ports;
  auto &verifier=*prepared_programs.emplace_back(Program::create(std::move(verifier_recipe.recipe),context,owner,cuda,3));
  if(converted)report_target_verifier(verifier,verifier_ports,*original->converted,owner,cuda);
  else if(plain)report_target_verifier(verifier,verifier_ports,*original->mlx,owner,cuda);else report_target_verifier(verifier,verifier_ports,*original->vocabulary,owner,cuda);
  Recipe policy_recipe;policy_recipe.fragment=pending_policy();if(plain)policy_recipe.fragment.profile=converted?original->converted->profile:&mlx_runtime::kProfile;
  auto &policy=*prepared_programs.emplace_back(Program::create(std::move(policy_recipe),context,owner,cuda,1));
  Program *random_q=nullptr,*random_decision=nullptr,*single_verifier=nullptr;
  TargetVerifierPorts single_ports;
  if(stochastic) {
    random_q=prepared_programs.emplace_back(Program::create(random_proposal_recipe(verifier.recipe.fragment.profile),context,owner,cuda,3)).get();
    random_decision=prepared_programs.emplace_back(Program::create(random_decision_recipe(verifier.recipe.fragment.profile),context,owner,cuda,3)).get();
    auto single=build_target_verifier(original,true);single_ports=single.ports;
    single_verifier=prepared_programs.emplace_back(Program::create(std::move(single.recipe),context,owner,cuda,1)).get();
  }
  Program *recursive_decision=nullptr,*recursive_verifier=nullptr,*recursive_repair=nullptr,*recursive_catchup=nullptr,*recursive_fresh=nullptr;
  TargetVerifierPorts recursive_ports;
  if(maximum_drafts==2) {
    auto recipe=build_target_verifier(original,false,3);recursive_ports=recipe.ports;
    recursive_verifier=prepared_programs.emplace_back(Program::create(std::move(recipe.recipe),context,owner,cuda,1)).get();
    Recipe decision;decision.state_ports.fill(std::numeric_limits<std::size_t>::max());decision.fragment.profile=verifier.recipe.fragment.profile;
    decision.fragment.region=0x474c4d52414e4432ULL;glm_stochastic_prefix_fragment::append(decision.fragment,target_verifier::kVocabulary);
    recursive_decision=prepared_programs.emplace_back(Program::create(std::move(decision),context,owner,cuda,2)).get();
    recursive_repair=prepared_programs.emplace_back(Program::create(build(false,0,3,original),context,owner,cuda,2)).get();
    recursive_catchup=prepared_programs.emplace_back(Program::create(build_catchup(false,2056,original,1),context,owner,cuda,2)).get();
    recursive_fresh=prepared_programs.emplace_back(Program::create(extend_recipe(2057),context,owner,cuda,2)).get();
  }
  auto queue=take(context.create_queue(),"joint target full-V verification queue");
  std::vector<Program*> programs{&target_plan,&first_plan,&next_after_zero,&next_after_one,&skip,&journal,&warm43_plan,
      &token43_a,&token43_b,&token43_c,&token43_d,&handoff_zero,&handoff_one,&initializer,&verifier,&policy};
  if(stochastic){programs.push_back(random_q);programs.push_back(random_decision);programs.push_back(single_verifier);}
  if(maximum_drafts==2)for(auto *program:{recursive_decision,recursive_verifier,recursive_repair,recursive_catchup,recursive_fresh})programs.push_back(program);
  const auto add=[](std::size_t &total,std::size_t amount) {
    if(amount>std::numeric_limits<std::size_t>::max()-total)throw std::runtime_error("target verifier state backing bound overflow");total+=amount;
  };
  const auto aligned=[](std::size_t bytes) {
    if(bytes>std::numeric_limits<std::size_t>::max()-255)throw std::runtime_error("target verifier initial alignment overflow");return (bytes+255)/256*256;
  };
  std::size_t pool_bound=0,initial_bound=0;
  for(const auto *program:programs)add(pool_bound,program->output_stats().capacity_bytes);
  const auto sparse_desc=descriptions(plain);const auto target_entries=sparse_entries(plain);
  for(const auto entry:state_resources(plain))
    if(entry!=kPool&&entry!=kPoolScales)add(initial_bound,aligned(take(sparse_desc[entry].required_bytes(),"NextN initial authority bytes")));
  for(const auto entry:target_entries)
    if(entry!=kPool&&entry!=kPoolScales)add(initial_bound,aligned(take(sparse_desc[entry].required_bytes(),"target43 initial authority bytes")));
  const auto metadata_bound=std::size_t(State::kMaxRetainedBlocks)*64;
  std::size_t state_budget=0;add(state_budget,pool_bound);add(state_budget,initial_bound);add(state_budget,metadata_bound);
  // Each actual output root is charged once, including all diagnostics and
  // preallocated generations. Both independent initial sparse histories are
  // bounded above; their pooled codes/scales already belong to initializer.
  // Original parameters/preparedKV are readonly weights, not State members.
  std::printf("GLM verifier state bound pools=%zu initial=%zu metadata=%zu total=%zu\n",pool_bound,initial_bound,metadata_bound,state_budget);
  std::size_t target_calls=0,target43_calls=0,verifier_calls=0,full_target_values=0,feedback_checks=0,peak_backing=0,accepted_repairs=0;
  bool full_target_checked=false;
  const auto target_run=[&](std::span<const c::ReadLease> inputs) {
    target::Outputs expected;
    if(!target44)expected=target_reference(typed<float>(inputs[0]),typed<float>(inputs[1]),typed<float>(inputs[2]));
    auto output=target_plan.execute(queue,inputs);++target_calls;
    if(target44)observe_target44(target_plan.recipe,inputs,output);
    else for(std::size_t port=0;port<target::Ports::count;++port) {
      SCOPED_TRACE(target::kPortNames[port]);sem::expect_output(target_plan.recipe.fragment,typed<float>(output[port]),expected[port]);
    }
    EXPECT_EQ(typed<std::int32_t>(output[cursor_port]).at(0),typed<std::int32_t>(inputs[3]).at(0)+1);return output;
  };
  auto initial43=initial_sparse(owner,initializer,queue,2053,plain);
  auto warm43=execute_target43(warm43_plan,queue,initial43,upload(owner,sem::bytes(target_residual_values(0))));++target43_calls;
  std::vector<c::ReadLease> warmup;
  {
    std::vector<float> history(3*target::kChannels),matrix(target::kHeads*target::kDepth*target::kDepth);
    for(std::size_t i=0;i<history.size();++i)history[i]=float(std::sin(double(i)*0.013)*0.01);
    if(target44&&!target44->mlx&&!target44->converted)for(auto &value:history)value=codec::wide(codec::bfloat(value));
    for(std::size_t i=0;i<matrix.size();++i)matrix[i]=float(std::sin(double(i)*0.007)*0.0005);
    const std::array input{warm43.mhc,upload(owner,sem::bytes(history)),upload(owner,sem::bytes(matrix)),upload(owner,sem::bytes(std::vector<std::int32_t>{2053}))};
    warmup=target_run(input);
  }
  std::vector<c::StateEntrySpec> extra_specs{
      {.description=target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[target_ports.history]),.resource=owner.ref()},
      {.description=target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[target_ports.state]),.resource=owner.ref()},
      {.description=target_plan.recipe.fragment.description(target_plan.recipe.fragment.outputs[target_ports.tap]),.resource=owner.ref()},
      {.description=sem::dense({1},c::DataType::kInt32),.resource=owner.ref()}};
  for(const auto entry:target_entries)extra_specs.push_back({.description=sparse_desc[entry],.resource=owner.ref()});
  struct Verified final {std::vector<c::ReadLease> output;std::vector<std::int32_t> ids;};
  const auto verify=[&](const c::ReadLease &tap0,const c::ReadLease &tap1) {
    const std::array input{tap0,tap1};Verified result;result.output=verifier.execute(queue,input);++verifier_calls;
    std::vector<std::uint16_t> joined;std::vector<float> joined_f32;
    for(std::size_t row=0;row<input.size();++row) {
      if(plain){const auto actual=typed<float>(result.output[verifier_ports.taps[row]]);EXPECT_EQ(actual,typed<float>(input[row]));joined_f32.insert(joined_f32.end(),actual.begin(),actual.end());}
      else {const auto stored=typed<std::uint16_t>(result.output[verifier_ports.taps[row]]);
        EXPECT_TRUE(index_reference::expect_bfloat_store(stored,typed<float>(input[row]),"actual finalnorm F32 -> explicit original verifier BF16"));joined.insert(joined.end(),stored.begin(),stored.end());}
    }
    const auto logits=typed<float>(result.output[verifier_ports.logits]);result.ids=typed<std::int32_t>(result.output[verifier_ports.ids]);
    if(plain){EXPECT_EQ(typed<float>(result.output[verifier_ports.stacked]),joined_f32);
      if(converted) {
        const auto &source=*original->converted;const auto &head=source.spec("output.weight");
        if(result.ids.size()!=2||std::ranges::any_of(result.ids,[](auto id){return id<0||id>=154880;}))throw std::runtime_error("converted actual verifier IDs outside vocabulary");
        for(std::size_t row=0;row<2;++row) {
          const std::array required{std::size_t(result.ids[row])};
          EXPECT_TRUE(packed_reference::expect_projection(std::span(logits).subspan(row*154880,154880),source.bytes("output.weight"),
              154880,4096,head.native,std::span(joined_f32).subspan(row*4096,4096),converted_codec(cuda),"converted actual M2 original full-V head",!full_target_checked,required));
        }
      } else EXPECT_TRUE(mlx_reference::expect_projection(logits,joined_f32,2,154880,4096,mlx_reference::BFloat{original->mlx->parameters.source->bytes("lm_head.weight"),154880,4096},!full_target_checked,result.ids));}
    else {EXPECT_EQ(typed<std::uint16_t>(result.output[verifier_ports.stacked]),joined);
      EXPECT_TRUE(lm_reference::expect_logits(logits,joined,original->vocabulary->source->bytes(shared_lm::Role::kHead),154880,4096,!full_target_checked,result.ids));}
    if(!full_target_checked){full_target_values=logits.size();full_target_checked=true;}
    EXPECT_TRUE(lm_reference::expect_greedy(logits,result.ids,std::size_t(target_verifier::kVocabulary),typed<float>(result.output[verifier_ports.scores])));
    for(std::size_t row=0;row<verifier_ports.row_ids.size();++row)
      EXPECT_EQ(typed<std::int32_t>(result.output[verifier_ports.row_ids[row]]),std::vector{result.ids.at(row)});
    if(plain&&::testing::Test::HasFailure())throw std::runtime_error("plain fullV target reference failed before accepted publication");
    return result;
  };
  const auto target_members=[&](State &state,std::span<const c::ReadLease> output,const SparseSnapshot &sparse) {
        std::vector<c::StateMember> members{{state.entries[kTargetHistory],output[target_ports.history]},
            {state.entries[kTargetState],output[target_ports.state]},{state.entries[kTargetTap],output[target_ports.tap]},
            {state.entries[kTargetCursor],output[cursor_port]}};
        for(std::size_t i=0;i<target_entries.size();++i)members.push_back({state.entries[kTarget43Begin+i],sparse[target_entries[i]]});return members;
      };
  const auto combined=[&](State &state,Program &plan,const c::StateVersion &from,bool target_seed,c::ReadLease *sample_logits=nullptr) {
        if(!state.enabled||(!plan.recipe.fresh&&!state.qualifies(from)))throw std::runtime_error("target K1 refused stale draft epoch before prepare");
        const auto position=state.scalar(from,kMlaCursor);
        if(typed<std::int32_t>(state.member(from,kTargetCursor)).at(0)!=position||target_seed!=plan.recipe.fresh)
          throw std::runtime_error("target K1 actual target/draft frontier or fresh-seed boundary differs");
        const auto before=image(state,from);const auto consumed=state.scalar(from,kPending);
        if(consumed<0||consumed>=target_verifier::kVocabulary)throw std::runtime_error("actual pending token outside the shared full vocabulary");
        auto produced43=execute_target43(choose43(position),queue,target43_snapshot(state,from),upload(owner,sem::bytes(target_residual_values(consumed))));++target43_calls;
        const std::array target_inputs{produced43.mhc,state.member(from,kTargetHistory),state.member(from,kTargetState),state.member(from,kTargetCursor)};
        auto produced=target_run(target_inputs);const auto members=target_members(state,produced,produced43.state);
        const auto seed=target_seed?state.member(from,kTargetTap):c::ReadLease{};
        auto next=execute(state,plan,queue,owner,from,0,nullptr,members,seed,sample_logits);
        for(const auto &member:members)same_storage(take(next.member(member.entry),"same full K1 target producer member"),member.resource);
        EXPECT_EQ(state.scalar(next,kMlaCursor),position+1);
        EXPECT_EQ(typed<std::int32_t>(state.member(next,kTargetCursor)).at(0),position+1);
        const auto target43_state=target43_snapshot(state,next);
        EXPECT_EQ(SparseAccess{}.scalar(target43_state,kMlaCursor),position+1);
        EXPECT_EQ(SparseAccess{}.scalar(target43_state,kIndexCursor),position+1);
        EXPECT_EQ(image(state,from),before);++feedback_checks;peak_backing=std::max(peak_backing,state.store.stats().backing_bytes);return next;
      };

  if(stochastic) {
    using Cursor=hyle::inference::SamplingCursor;
    const auto sampling_entry=kEntries+extra_specs.size();
    const auto target_end=sampling_entry;
    extra_specs.push_back({.kind=c::StateEntryKind::kHostMetadata,.metadata_bytes=sizeof(Cursor)});
    auto state=State::create(owner,extra_specs,state_budget,plain);
    const Cursor initial_cursor{0x474c4d35ULL,13,0};
    const auto cursor=[&](const c::StateVersion &version) {
      const auto bytes=read(state.member(version,sampling_entry));
      if(bytes.size()!=sizeof(Cursor))throw std::runtime_error("GLM sampling cursor metadata extent");
      Cursor result;std::memcpy(&result,bytes.data(),sizeof(result));return result;
    };
    const auto identity=[&](Cursor at,std::uint64_t offset) {
      const auto words=at.identity(offset);
      return upload(owner,sem::bytes(std::vector<std::uint32_t>(words.begin(),words.end())));
    };
    const auto proposal=[&](const c::ReadLease &logits,Cursor at) {
      const std::array input{logits,identity(at,0)};
      auto output=random_q->execute(queue,input);
      if(!random_flag(output[2]))throw std::runtime_error("actual original NextN q distribution is invalid");
      const auto id=typed<std::int32_t>(output[1]);
      if(id.size()!=1||id[0]<0||id[0]>=target_verifier::kVocabulary)
        throw std::runtime_error("actual sampled GLM proposal is outside its original vocabulary");
      return output;
    };
    const auto decide=[&](const Verified &target,const std::vector<c::ReadLease> &draft,
                          Cursor at,std::uint64_t accept_offset) {
      const std::array input{target.output[verifier_ports.logits],draft[0],draft[1],
          identity(at,accept_offset),identity(at,accept_offset+1)};
      auto output=random_decision->execute(queue,input);
      if(!random_flag(output[2]))throw std::runtime_error("actual GLM p/q/bonus decision is invalid");
      const auto next=typed<std::int32_t>(output[1]);
      if(next.size()!=1||next[0]<0||next[0]>=target_verifier::kVocabulary)
        throw std::runtime_error("actual GLM residual/bonus pending is outside its vocabulary");
      return output;
    };
    auto initial=target_members(state,warmup,warm43.state);
    initial.push_back({state.entries[sampling_entry],{},std::as_bytes(std::span(&initial_cursor,1))});
    const auto base=state.initial(owner,initializer,queue,2054,initial,warmup[target_ports.tap]);
    c::ReadLease logits;
    const auto current_id=state.scalar(base,kPending);
    if(std::ranges::find(limits.eos,current_id)!=limits.eos.end())
      throw std::runtime_error("GLM cannot consume an EOS pending token");
    const auto first=combined(state,first_plan,base,true,&logits);
    const auto first_image=image(state,first);
    // Target-only emission still keeps the mandatory-current NextN state in
    // sync. No q draw, proposed token or speculative second row is submitted.
    const auto single=[&](hyle::inference::mtp::SamplingLimits boundary,bool terminal_control) {
      boundary.validate(target_verifier::kVocabulary);
      const auto before_target=target_calls,before_nextn=first_plan.native_submissions(),before_second=skip.native_submissions();
      const std::array input{state.member(first,kTargetTap)};
      const auto target=single_verifier->execute(queue,input);
      const auto sampled=proposal(target[single_ports.logits],initial_cursor);
      const auto id=typed<std::int32_t>(sampled[1]).at(0);
      const std::array terminal{id};if(terminal_control)boundary.eos=terminal;
      const std::array inputs{current_id},finals{id};std::vector<std::int32_t> emitted;
      const auto selection=hyle::inference::mtp::select_sampled_prefix(inputs,{},finals,
          boundary.quota,boundary.eos,std::int32_t(target_verifier::kVocabulary),emitted);
      auto head=take(c::StateHead::create(state.store,base),"GLM single-row bounded Head");
      auto draft=take(state.store.begin(first),"same complete single-row target/NextN prefix");
      check(draft.replace(state.entries[kPending],sampled[1]),"single target p0 pending lease");
      const auto rng=initial_cursor.advanced(1);
      check(draft.replace_metadata(state.entries[sampling_entry],std::as_bytes(std::span(&rng,1))),"single target draw commits with state");
      const auto selected=take(draft.finalize(),"single target complete checkpoint");
      EXPECT_FALSE(head.publish(first.identity(),selected).ok());EXPECT_EQ(cursor(take(head.current(),"single stale Head")),initial_cursor);
      check(head.publish(base.identity(),selected),"single bounded result publishes once");
      EXPECT_EQ(selection.consumed,1u);EXPECT_EQ(selection.consumed_drafts,0u);EXPECT_EQ(selection.stopped,terminal_control||std::ranges::find(boundary.eos,id)!=boundary.eos.end());
      same_storage(state.member(selected,kPending),sampled[1]);EXPECT_EQ(cursor(selected),rng);
      EXPECT_EQ(state.scalar(selected,kMlaCursor),2055);EXPECT_EQ(typed<std::int32_t>(state.member(selected,kTargetCursor)).at(0),2055);
      for(std::size_t entry=0;entry<state.entries.size();++entry)
        if(state.present(entry)&&entry!=kPending&&entry!=sampling_entry)same_storage(state.member(selected,entry),state.member(first,entry));
      EXPECT_EQ(target_calls,before_target);EXPECT_EQ(first_plan.native_submissions(),before_nextn);EXPECT_EQ(skip.native_submissions(),before_second);
      EXPECT_EQ(image(state,first),first_image);
      std::printf("GLM RANDOM TAIL context=%zu quota=%zu emitted=%d stopped=%d consumed=1 draw=1 second_submissions=0\n",boundary.remaining_context,boundary.quota,id,int(selection.stopped));
    };
    if(limits.single_row()) {
      single(limits,false);
      ::testing::Test::RecordProperty("random_tail_target_rows",1);::testing::Test::RecordProperty("random_tail_draws",1);
      ::testing::Test::RecordProperty("scope","Finite one-row target-only emission with mandatory-current NextN maintenance; no q sample or proposed second-row execution. Complete same-prefix resources/pending/RNG publish once. Not a request scheduler.");
      return;
    }
    // Independent boundary Heads reuse the same authentic first-prefix
    // resources; these do not change the ordinary branch's RNG cursor.
    if(controlled_random_branches){single({1,1,{}},false);check(state.store.collect(),"retire independent single target Head");single({1,1,{}},true);check(state.store.collect(),"retire independent terminal target Head");}
    const auto q=proposal(logits,initial_cursor);
    const auto with_proposal=[&](const c::StateVersion &from,const c::ReadLease &id) {
      auto draft=take(state.store.begin(from),"retain actual sampled original NextN proposal");
      check(draft.replace(state.entries[kPending],id),"sampled I32 proposal enters target row1");
      return take(draft.finalize(),"all original draft and target resources retain sampled proposal");
    };
    const auto proposed=with_proposal(first,q[1]);
    c::ReadLease second_logits;
    const auto second=combined(state,skip,proposed,false,maximum_drafts==2?&second_logits:nullptr);
    const auto first_proposal_id=typed<std::int32_t>(q[1]).at(0);
    const bool can_recurse=maximum_drafts==2&&limits.quota>2&&limits.remaining_context>2&&
        std::ranges::find(limits.eos,first_proposal_id)==limits.eos.end();
    if(can_recurse) {
      const auto q1=proposal(second_logits,initial_cursor.advanced(1));
      const auto second_proposal_id=typed<std::int32_t>(q1[1]).at(0);
      const auto next_proposed=with_proposal(second,q1[1]);
      const auto third=combined(state,skip,next_proposed,false);
      const std::array taps{state.member(first,kTargetTap),state.member(second,kTargetTap),state.member(third,kTargetTap)};
      const auto projected=recursive_verifier->execute(queue,taps);
      const auto target_logits=typed<float>(projected[recursive_ports.logits]);
      EXPECT_EQ(target_logits.size(),3u*std::size_t(target_verifier::kVocabulary));
      std::vector<std::uint16_t> stored;std::vector<float> wide;
      for(std::size_t row=0;row<3;++row) {
        if(plain){const auto values=typed<float>(projected[recursive_ports.taps[row]]);wide.insert(wide.end(),values.begin(),values.end());}
        else {const auto values=typed<std::uint16_t>(projected[recursive_ports.taps[row]]);stored.insert(stored.end(),values.begin(),values.end());}
      }
      const auto actual_ids=typed<std::int32_t>(projected[recursive_ports.ids]);
      if(!plain)EXPECT_TRUE(lm_reference::expect_logits(target_logits,stored,original->vocabulary->source->bytes(shared_lm::Role::kHead),154880,4096,false,actual_ids));
      else if(!converted)EXPECT_TRUE(mlx_reference::expect_projection(target_logits,wide,3,154880,4096,
          mlx_reference::BFloat{original->mlx->parameters.source->bytes("lm_head.weight"),154880,4096},false,actual_ids));
      const std::array proposal_ids{first_proposal_id,second_proposal_id};
      const auto proposal_input=upload(owner,sem::bytes(std::vector(proposal_ids.begin(),proposal_ids.end())));
      const auto decide_prefix=[&](std::array<std::uint64_t,3> offsets) {
        std::vector<std::uint32_t> words;for(const auto offset:offsets){const auto key=initial_cursor.identity(offset);words.insert(words.end(),key.begin(),key.end());}
        const std::array input{projected[recursive_ports.logits],upload(owner,sem::bytes(words)),proposal_input,q[0],q1[0]};
        const auto output=recursive_decision->execute(queue,input);glm_stochastic_prefix_fragment::Decision result;
        for(std::size_t row=0;row<2;++row) {
          if(!random_flag(output[row*3+2]))throw std::runtime_error("actual GLM K2 p/q decision invalid");
          result.accepts[row]=std::uint8_t(random_flag(output[row*3]));result.final_leases[row]=output[row*3+1];
          result.final_ids[row]=typed<std::int32_t>(result.final_leases[row]).at(0);
        }return result;
      };
      std::array<c::StateVersion,3> repaired{first,{},{}};
      const auto repair=[&](std::size_t count) {
        state.epoch=7;state.enabled=true;
        for(std::size_t row=1;row<=count;++row)if(!repaired[row].valid()) {
          const auto input=with_proposal(repaired[row-1],row==1?q[1]:q1[1]);
          const auto &target=row==1?second:third;std::vector<c::StateMember> members;
          for(std::size_t entry=kTargetHistory;entry<target_end;++entry)members.push_back({state.entries[entry],state.member(target,entry)});
          repaired[row]=execute(state,*recursive_repair,queue,owner,input,0,nullptr,members,state.member(row==1?first:second,kTargetTap));
          ++accepted_repairs;
        }return repaired[count];
      };
      const auto observe=[&](const c::StateVersion &selected,const c::StateVersion &checkpoint,
          const hyle::inference::mtp::AcceptedPrefix &selection,Cursor rng) {
        EXPECT_EQ(cursor(selected),rng);EXPECT_EQ(state.scalar(selected,kPending),selection.next_id);
        EXPECT_EQ(state.scalar(selected,kMlaCursor),2054+std::int32_t(selection.consumed));
        EXPECT_EQ(typed<std::int32_t>(state.member(selected,kTargetCursor)).at(0),state.scalar(selected,kMlaCursor));
        EXPECT_EQ(state.scalar(selected,kLagCount),std::int32_t(selection.consumed_drafts));
        for(std::size_t entry=0;entry<state.entries.size();++entry)if(state.present(entry)&&entry!=kPending&&entry!=sampling_entry)
          same_storage(state.member(selected,entry),state.member(checkpoint,entry));
      };
      const auto continued=[&](c::StateHead &head,const c::StateVersion &selected,std::size_t consumed,Cursor rng) {
        state.epoch=7;state.enabled=false;auto caught=selected;
        for(std::size_t slot=0;slot<consumed;++slot) {
          auto next=catch_up(state,slot==0?journal:*recursive_catchup,queue,head);
          check(head.publish(caught.identity(),next),"GLM K2 accepted journal slot publication");caught=std::move(next);
        }
        EXPECT_EQ(state.scalar(caught,kLagCount),0);EXPECT_EQ(state.scalar(caught,kMlaCursor),state.scalar(caught,kIndexCursor));
        state.epoch=8;state.enabled=true;auto &fresh=consumed==0?next_after_zero:consumed==1?next_after_one:*recursive_fresh;
        const auto next=combined(state,fresh,caught,true);
        const std::array input{state.member(next,kTargetTap)};const auto p=single_verifier->execute(queue,input);
        const auto draw=proposal(p[single_ports.logits],rng);auto draft=take(state.store.begin(next),"GLM K2 real pending continuation");
        check(draft.replace(state.entries[kPending],draw[1]),"GLM K2 next target categorical lease");const auto advanced=rng.advanced(1);
        check(draft.replace_metadata(state.entries[sampling_entry],std::as_bytes(std::span(&advanced,1))),"GLM K2 continuation draw same complete Head");
        const auto version=take(draft.finalize(),"GLM K2 complete actual continuation");check(head.publish(caught.identity(),version),"GLM K2 actual pending is consumed next epoch");
        EXPECT_EQ(cursor(version),advanced);EXPECT_EQ(state.scalar(version,kMlaCursor),2056+std::int32_t(consumed));
        std::printf("GLM K2 CONTINUED consumed_drafts=%zu consumed_pending=%d next_pending=%d cursor=%llu\n",consumed,state.scalar(caught,kPending),state.scalar(version,kPending),static_cast<unsigned long long>(advanced.draw));
      };
      const auto log_q0=typed<float>(q[0]),log_q1=typed<float>(q1[0]);const auto vocabulary=std::size_t(target_verifier::kVocabulary);
      const std::array alpha{glm_stochastic_prefix_fragment::log_alpha(std::span(target_logits).first(vocabulary),log_q0,first_proposal_id),
          glm_stochastic_prefix_fragment::log_alpha(std::span(target_logits).subspan(vocabulary,vocabulary),log_q1,second_proposal_id)};
      glm_stochastic_prefix_fragment::exercise(state.store,base,state.entries[kPending],state.entries[sampling_entry],current_id,proposal_ids,
          std::array{q[1],q1[1]},state.member(first,kTargetTap),initial_cursor,std::int32_t(target_verifier::kVocabulary),cuda,alpha,limits,
          decide_prefix,repair,observe,continued);
      EXPECT_EQ(image(state,first),first_image);return;
    }
    const auto checked=verify(state.member(first,kTargetTap),state.member(second,kTargetTap));
    const auto natural=decide(checked,q,initial_cursor,1);
    const auto proposal_id=typed<std::int32_t>(q[1]).at(0);
    const auto log_p=typed<float>(natural[3]),log_q=typed<float>(q[0]);
    const auto log_alpha=std::min(0.,double(log_p.at(proposal_id))-double(log_q.at(proposal_id)));
    const bool natural_accept=random_flag(natural[0]);
    EXPECT_EQ(reference_random_accept(log_alpha,initial_cursor.identity(1),cuda),natural_accept);
    std::vector<std::uint64_t> offsets{1};
    // Explicit branch fixture only. Default consumers use offsets0/1/2 and
    // advance3. Never change p, q, logits or the actual proposal to force a
    // branch; search only a bounded independent Philox address interval.
    const std::uint64_t address_bound=controlled_random_branches?262144:0;
    for(std::uint64_t offset=2;offset<address_bound;++offset)
      if(reference_random_accept(log_alpha,initial_cursor.identity(offset),cuda)!=natural_accept) {
        offsets.push_back(offset);break;
      }
    std::array<std::size_t,2> random_branches{};
    std::size_t published_rounds=0;
    for(const auto offset:offsets) {
      state.epoch=7;state.enabled=true;
      const auto decision=offset==1?natural:decide(checked,q,initial_cursor,offset);
      const auto sampled_accept=random_flag(decision[0]);
      const std::array inputs{current_id,proposal_id};
      const std::array flags{std::uint8_t(sampled_accept)};
      const std::array finals{typed<std::int32_t>(decision[1]).at(0)};std::vector<std::int32_t> emitted;
      const auto selection=hyle::inference::mtp::select_sampled_prefix(inputs,flags,finals,
          limits.quota,limits.eos,std::int32_t(target_verifier::kVocabulary),emitted);
      const bool accepted=selection.consumed_drafts!=0;++random_branches[sampled_accept?1:0];
      const auto pending=sampled_accept&&selection.consumed==1?q[1]:decision[1];
      EXPECT_EQ(typed<std::int32_t>(pending).at(0),selection.next_id);
      EXPECT_EQ(reference_random_accept(log_alpha,initial_cursor.identity(offset),cuda),sampled_accept);
      auto head=take(c::StateHead::create(state.store,base),"random GLM expected-base Head");
      const auto base_image=image(state,base);
      const auto next_cursor=initial_cursor.advanced(offset+2);
      c::StateVersion repaired;
      if(accepted) {
        std::vector<c::StateMember> members;
        for(std::size_t entry=kTargetHistory;entry<target_end;++entry)
          members.push_back({state.entries[entry],state.member(second,entry)});
        repaired=execute(state,skip,queue,owner,proposed,0,nullptr,members,state.member(first,kTargetTap));
        ++accepted_repairs;
      }
      const auto checkpoint=accepted?repaired:first;
      {
        auto abandoned=take(state.store.begin(checkpoint),"cancel sampled pending and cursor together");
        check(abandoned.replace(state.entries[kPending],pending),"stage actual random result before injected failure");
        check(abandoned.replace_metadata(state.entries[sampling_entry],std::as_bytes(std::span(&next_cursor,1))),"stage uncommitted random cursor");
        EXPECT_FALSE(abandoned.replace(state.entries[kPending],state.member(first,kTargetTap)).ok());
        check(abandoned.cancel(),"failure cancels pending/cursor publication");
      }
      EXPECT_EQ(image(state,take(head.current(),"unchanged random Head after cancel")),base_image);
      EXPECT_EQ(cursor(take(head.current(),"unadvanced random cursor after cancel")),initial_cursor);
      auto publish=[&](const c::StateVersion &from,const c::ReadLease &pending,Cursor rng) {
        auto draft=take(state.store.begin(from),"one all-resource random K1 transaction");
        check(draft.replace(state.entries[kPending],pending),"real residual/bonus I32 lease");
        check(draft.replace_metadata(state.entries[sampling_entry],std::as_bytes(std::span(&rng,1))),"same transaction RNG cursor");
        return take(draft.finalize(),"complete random target/NextN prefix and RNG");
      };
      const auto selected=publish(checkpoint,pending,next_cursor);
      EXPECT_FALSE(head.publish(second.identity(),selected).ok());
      EXPECT_EQ(cursor(take(head.current(),"stale publication cannot advance RNG")),initial_cursor);
      check(head.publish(base.identity(),selected),"retry the same original random decision without resampling");++published_rounds;
      EXPECT_EQ(cursor(take(head.current(),"published random cursor")),next_cursor);
      same_storage(state.member(selected,kPending),pending);
      EXPECT_EQ(state.scalar(selected,kMlaCursor),2055+std::int32_t(accepted));
      EXPECT_EQ(typed<std::int32_t>(state.member(selected,kTargetCursor)).at(0),state.scalar(selected,kMlaCursor));
      if(controlled_random_branches&&limits.quota==std::numeric_limits<std::size_t>::max()&&limits.remaining_context==std::numeric_limits<std::size_t>::max()&&limits.eos.empty()) {
        const auto boundary=[&](std::size_t quota,std::span<const std::int32_t> eos,const char *name) {
          std::vector<std::int32_t> tokens;
          const auto policy=hyle::inference::mtp::select_sampled_prefix(inputs,flags,finals,quota,eos,
              std::int32_t(target_verifier::kVocabulary),tokens);
          const auto lease=sampled_accept&&policy.consumed==1?q[1]:decision[1];
          const auto from=policy.consumed_drafts?repaired:first;
          auto branch=take(c::StateHead::create(state.store,base),"independent GLM bounded-policy Head");
          const auto bounded=publish(from,lease,next_cursor);
          check(branch.publish(base.identity(),bounded),"bounded actual lease/state/RNG publication");
          same_storage(state.member(bounded,kPending),lease);EXPECT_EQ(state.scalar(bounded,kPending),policy.next_id);
          EXPECT_EQ(state.scalar(bounded,kMlaCursor),2054+std::int32_t(policy.consumed));
          EXPECT_EQ(typed<std::int32_t>(state.member(bounded,kTargetCursor)).at(0),2054+std::int32_t(policy.consumed));EXPECT_EQ(cursor(bounded),next_cursor);
          for(std::size_t entry=0;entry<state.entries.size();++entry)
            if(state.present(entry)&&entry!=kPending&&entry!=sampling_entry)same_storage(state.member(bounded,entry),state.member(from,entry));
          std::printf("GLM RANDOM BOUNDARY name=%s accept=%d emitted=%zu consumed=%zu pending=%d stopped=%d draw=%llu\n",name,int(sampled_accept),tokens.size(),policy.consumed,policy.next_id,int(policy.stopped),static_cast<unsigned long long>(next_cursor.draw));
        };
        boundary(1,{},"quota1");check(state.store.collect(),"retire independent quota Head");
        if(sampled_accept&&proposal_id!=current_id){const std::array eos{proposal_id};boundary(2,eos,"accepted_proposal_EOS");check(state.store.collect(),"retire independent accepted EOS Head");}
        if(finals[0]!=current_id&&(!sampled_accept||finals[0]!=proposal_id)){const std::array eos{finals[0]};boundary(2,eos,sampled_accept?"bonus_EOS":"residual_EOS");check(state.store.collect(),"retire independent final EOS Head");}
      }
      // Explicit finite limits describe one verifier episode. They do not
      // turn this component fixture into a complete request scheduler.
      if(limits.quota!=std::numeric_limits<std::size_t>::max()||limits.remaining_context!=std::numeric_limits<std::size_t>::max()||!limits.eos.empty())continue;
      state.enabled=false;
      EXPECT_THROW((void)combined(state,skip,selected,false),std::runtime_error);
      EXPECT_EQ(cursor(take(head.current(),"pre-submit failure keeps RNG")),next_cursor);
      auto caught=selected;
      if(accepted) {
        caught=catch_up(state,journal,queue,head);
        check(head.publish(selected.identity(),caught),"random accepted prefix journal catchup");
      } else (void)activate_extend_seed(state,head,first,2054);
      EXPECT_EQ(cursor(take(head.current(),"catchup does not draw randomness")),next_cursor);
      state.epoch=8;state.enabled=true;
      auto &fresh=accepted?next_after_one:next_after_zero;
      c::ReadLease next_logits;
      const auto next_first=combined(state,fresh,take(head.current(),"real random pending for next round"),true,&next_logits);
      const auto next_q=proposal(next_logits,next_cursor);
      const auto next_proposed=with_proposal(next_first,next_q[1]);
      auto &step=accepted?handoff_one:handoff_zero;
      const auto next_second=combined(state,step,next_proposed,false);
      const auto next_checked=verify(state.member(next_first,kTargetTap),state.member(next_second,kTargetTap));
      const auto next_decision=decide(next_checked,next_q,next_cursor,1);
      const auto next_accept=random_flag(next_decision[0]);
      c::StateVersion next_repaired;
      if(next_accept) {
        std::vector<c::StateMember> members;
        for(std::size_t entry=kTargetHistory;entry<target_end;++entry)
          members.push_back({state.entries[entry],state.member(next_second,entry)});
        next_repaired=execute(state,step,queue,owner,next_proposed,0,nullptr,members,state.member(next_first,kTargetTap));
        ++accepted_repairs;
      }
      const auto next_selected=publish(next_accept?next_repaired:next_first,next_decision[1],next_cursor.advanced(3));
      check(head.publish(caught.identity(),next_selected),"next ordinary three-draw K1 round atomically continues");++published_rounds;
      EXPECT_EQ(cursor(take(head.current(),"second committed random cursor")),next_cursor.advanced(3));
      same_storage(state.member(next_selected,kPending),next_decision[1]);
      EXPECT_EQ(state.scalar(next_selected,kMlaCursor),2056+std::int32_t(accepted)+std::int32_t(next_accept));
      EXPECT_EQ(typed<std::int32_t>(state.member(next_selected,kTargetCursor)).at(0),state.scalar(next_selected,kMlaCursor));
      EXPECT_EQ(state.scalar(next_selected,kLagCount),std::int32_t(next_accept));
      std::printf("GLM RANDOM K1 proposal=%d alpha=%.17g accept_offset=%llu reserved=%llu accept=%d pending=%d next_round_accept=%d next_pending=%d cursor=%llu\n",
          proposal_id,std::exp(log_alpha),static_cast<unsigned long long>(offset),static_cast<unsigned long long>(offset+2),
          int(accepted),typed<std::int32_t>(decision[1]).at(0),int(next_accept),state.scalar(next_selected,kPending),
          static_cast<unsigned long long>(cursor(next_selected).draw));
    }
    ::testing::Test::RecordProperty("random_default_schedule","proposal0/accept1/final2; success advances3; cancellation and stale publish advance0");
    ::testing::Test::RecordProperty("random_controlled_address_bound",std::to_string(address_bound));
    ::testing::Test::RecordProperty("random_actual_proposal",proposal_id);
    ::testing::Test::RecordProperty("random_alpha",std::to_string(std::exp(log_alpha)));
    ::testing::Test::RecordProperty("random_accepted",int(random_branches[1]));
    ::testing::Test::RecordProperty("random_rejected",int(random_branches[0]));
    ::testing::Test::RecordProperty("random_published_rounds",int(published_rounds));
    ::testing::Test::RecordProperty("random_opposite_branch_found",offsets.size()==2?1:0);
    ::testing::Test::RecordProperty("scope","Actual original NextN q/proposal and real two-row verifier p0/p1, shared common stochastic decision, same complete target/NextN StateHead plus original RNG metadata. Default three addresses and bounded independent controlled-address branch fixture are distinguished; no probability/logit changes or TopK substitution. Each selected branch consumes its actual residual/bonus and completes a second ordinary K1 round. Source component upstream/initial-history fixtures remain; not full-model task quality or natural acceptance-rate evidence.");
    return;
  }
  c::ReadLease natural_target0;std::int32_t natural_proposal=-1;std::uint32_t natural_accepted=0;
  std::array<std::size_t,2> branch_counts{};
  std::vector<std::int32_t> emitted;emitted.reserve(2);
  for(const int scenario:{0,1,2}) {
    SCOPED_TRACE(scenario); // 0 natural; 1 controlled match; 2 controlled reject.
    auto state=State::create(owner,extra_specs,state_budget,plain);
    {
      const auto initial_members=target_members(state,warmup,warm43.state);
      const auto base=state.initial(owner,initializer,queue,2054,initial_members,warmup[target_ports.tap]);
      const auto base_image=image(state,base);same_storage(state.member(base,kHiddenState),state.member(base,kTargetTap));
      auto head=take(c::StateHead::create(state.store,base),"one whole target/NextN K1 accepted Head");
      const auto first=combined(state,first_plan,base,true);const auto first_image=image(state,first);
      EXPECT_EQ(state.facts(first).producer_position,2054u);EXPECT_EQ(state.facts(first).reuse_count,0u);
      EXPECT_EQ(state.scalar(first,kIndexCursor),2055);EXPECT_EQ(state.scalar(first,kLagCount),0);
      c::StateVersion proposal_source=first;
      if(scenario!=0) {
        auto proposal_draft=take(state.store.begin(first),"explicit controlled verifier proposal policy");
        c::ReadLease proposal_lease=natural_target0;
        if(scenario==2) {
          // An explicit deterministic rejection fixture, never natural draft
          // acceptance. The real target correction is still the verifier's
          // row output; this only supplies a distinct consumed test proposal.
          const auto target_id=typed<std::int32_t>(natural_target0).at(0);
          const auto distinct=std::int32_t((std::int64_t(target_id)+1)%target_verifier::kVocabulary);
          const std::array policy_input{upload(owner,sem::bytes(std::vector{distinct}))};
          proposal_lease=policy.execute(queue,policy_input).at(0);
        }
        check(proposal_draft.replace(state.entries[kPending],proposal_lease),"actual controlled proposal Graph result lease");
        proposal_source=take(proposal_draft.finalize(),"controlled proposal preserves all34 producer resources except pending");
        for(std::size_t entry=0;entry<state.entries.size();++entry)if(state.present(entry)&&entry!=kPending)same_storage(state.member(proposal_source,entry),state.member(first,entry));
      }
      const auto proposal=state.scalar(proposal_source,kPending);
      const auto second=combined(state,skip,proposal_source,false);const auto second_image=image(state,second);
      EXPECT_EQ(state.facts(second).producer_position,2054u);EXPECT_EQ(state.facts(second).reuse_count,1u);
      EXPECT_EQ(state.scalar(second,kIndexCursor),2055);EXPECT_EQ(state.scalar(second,kLagCount),1);
      const auto checked=verify(state.member(first,kTargetTap),state.member(second,kTargetTap));
      const std::array inputs{state.scalar(base,kPending),proposal};
      // The original greedy K1 fixture has two outputs and no EOS policy.
      const auto decision=hyle::inference::mtp::select_greedy_prefix(inputs,checked.ids,2,{},
          std::int32_t(target_verifier::kVocabulary),emitted);
      if(scenario==0) {
        natural_target0=checked.output[verifier_ports.row_ids[0]];natural_proposal=proposal;natural_accepted=std::uint32_t(decision.consumed_drafts);
      } else {
        EXPECT_EQ(typed<std::int32_t>(natural_target0).at(0),checked.ids.at(0));
        EXPECT_EQ(decision.consumed_drafts,scenario==1?1u:0u);
      }
      ++branch_counts.at(decision.consumed_drafts);
      c::StateVersion repaired;
      if(plain&&decision.consumed_drafts) {
        std::vector<c::StateMember> members;for(std::size_t entry=kTargetHistory;entry<state.entries.size();++entry)members.push_back({state.entries[entry],state.member(second,entry)});
        repaired=execute(state,skip,queue,owner,proposal_source,0,nullptr,members,state.member(first,kTargetTap));
        ++accepted_repairs;
        for(std::size_t entry=kTargetHistory;entry<state.entries.size();++entry)same_storage(state.member(repaired,entry),state.member(second,entry));
      }
      const auto &checkpoint=decision.consumed_drafts?(repaired.valid()?repaired:second):first;
      EXPECT_EQ(decision.consumed,1+decision.consumed_drafts);
      auto accepted_draft=take(state.store.begin(checkpoint),"one actual verified K1 prefix and target pending");
      const auto pending=checked.output[verifier_ports.row_ids.at(decision.consumed_drafts)];
      check(accepted_draft.replace(state.entries[kPending],pending),"bind the actual selected full-V target I32[1] output lease");
      const auto accepted=take(accepted_draft.finalize(),"atomic34-resource verified target/NextN checkpoint");
      for(std::size_t entry=0;entry<state.entries.size();++entry)if(state.present(entry)&&entry!=kPending)same_storage(state.member(accepted,entry),state.member(checkpoint,entry));
      same_storage(state.member(accepted,kPending),pending);
      EXPECT_EQ(state.scalar(accepted,kPending),decision.next_id);
      EXPECT_EQ(state.scalar(accepted,kMlaCursor),2055+std::int32_t(decision.consumed_drafts));
      EXPECT_EQ(typed<std::int32_t>(state.member(accepted,kTargetCursor)).at(0),state.scalar(accepted,kMlaCursor));
      check(head.publish(base.identity(),accepted),"publish verified target and draft under the same expected-base Head");
      EXPECT_FALSE(head.publish(base.identity(),second).ok());
      const auto accepted_image=image(state,accepted);
      {
        auto cancelled=take(state.store.begin(accepted),"incomplete target-only splice must not publish");
        check(cancelled.replace(state.entries[kTargetTap],state.member(second,kTargetTap)),"stage incomplete speculative target tap");
        EXPECT_FALSE(cancelled.replace(state.entries[kPending],state.member(first,kTargetTap)).ok());
        check(cancelled.cancel(),"cancel incomplete target verifier draft");
      }
      EXPECT_EQ(image(state,take(head.current(),"Head after cancellation")),accepted_image);
      state.enabled=false;
      EXPECT_THROW((void)execute(state,skip,queue,owner,accepted,0),std::runtime_error);
      EXPECT_EQ(image(state,take(head.current(),"Head after pre-submit failure")),accepted_image);
      c::StateVersion caught=accepted;
      if(decision.consumed_drafts) {
        caught=catch_up(state,journal,queue,head);
        check(head.publish(accepted.identity(),caught),"publish only accepted nx journal catchup");
      } else {
        EXPECT_THROW((void)catch_up(state,journal,queue,head),std::runtime_error);
        // This is the last accepted actual fresh draft-extend seed. Pending
        // correction is allowed, but its hidden/selection/epoch must remain
        // those exact producer leases, never a reconstructed seed.
        const auto active=activate_extend_seed(state,head,first,2054);
        same_storage(state.member(active,kHiddenState),state.member(first,kHiddenState));
        same_storage(state.member(active,kSelection),state.member(first,kSelection));state.enabled=false;
      }
      for(std::size_t entry=kTargetHistory;entry<state.entries.size();++entry)same_storage(state.member(caught,entry),state.member(accepted,entry));
      same_storage(state.member(caught,kPending),pending);
      EXPECT_EQ(state.scalar(caught,kMlaCursor),state.scalar(caught,kIndexCursor));
      EXPECT_EQ(state.scalar(caught,kLagCount),0);
      state.epoch=8;state.enabled=true;
      auto &fresh=decision.consumed_drafts?next_after_one:next_after_zero;
      const auto continued=combined(state,fresh,take(head.current(),"actual accepted target tap for fresh epoch8"),true);
      check(head.publish(caught.identity(),continued),"publish epoch8 with actual previous accepted target tap and target bonus consumed");
      EXPECT_EQ(state.facts(continued).epoch,8u);EXPECT_EQ(state.facts(continued).reuse_count,0u);
      EXPECT_EQ(state.facts(continued).producer_position,std::uint64_t(state.scalar(caught,kMlaCursor)));
      EXPECT_EQ(state.scalar(continued,kMlaCursor),2056+std::int32_t(decision.consumed_drafts));
      const auto continued_image=image(state,continued);
      state.enabled=false;
      const auto accepted_seed=activate_extend_seed(state,head,continued,std::int32_t(state.facts(continued).producer_position));
      for(std::size_t entry=0;entry<state.entries.size();++entry)if(state.present(entry))same_storage(state.member(accepted_seed,entry),state.member(continued,entry));
      auto &step_zero=decision.consumed_drafts?handoff_one:handoff_zero;
      const auto skipped=combined(state,step_zero,accepted_seed,false);
      EXPECT_EQ(state.facts(skipped).epoch,8u);EXPECT_EQ(state.facts(skipped).reuse_count,1u);
      EXPECT_EQ(state.facts(skipped).producer_position,state.facts(continued).producer_position);
      // Skip materializes the original I32 IDs through its own real Copy;
      // values stay equal, while execute() already checked its actual writer lease.
      EXPECT_EQ(read(state.member(skipped,kSelection)),read(state.member(accepted_seed,kSelection)));
      same_storage(state.member(skipped,kPool),state.member(accepted_seed,kPool));
      EXPECT_EQ(state.scalar(skipped,kLagCount),1);
      // This next proposal is merely computed, not target-verified/accepted.
      // Its joint target/draft candidate cannot replace the accepted seed Head.
      EXPECT_EQ(image(state,take(head.current(),"unverified skip0 does not publish")),continued_image);
      EXPECT_EQ(image(state,continued),continued_image);
      EXPECT_EQ(image(state,base),base_image);EXPECT_EQ(image(state,first),first_image);
      EXPECT_EQ(image(state,second),second_image);EXPECT_EQ(image(state,accepted),accepted_image);
      EXPECT_FALSE(state.member(accepted,kPending).try_write());EXPECT_FALSE(state.member(second,kTargetTap).try_write());
      EXPECT_LE(state.peak_backing_bytes,state_budget);peak_backing=std::max(peak_backing,state.peak_backing_bytes);
      std::printf("GLM full-V verifier scenario=%d proposal=%d target0=%d target1=%d accepted_drafts=%u retained_cursor=%d next_pending=%d; mandatory current already consumed\n",
          scenario,proposal,checked.ids[0],checked.ids[1],std::uint32_t(decision.consumed_drafts),state.scalar(accepted,kMlaCursor),decision.next_id);
    }
    check(state.store.collect(),"retire complete verified scenario versions");
    EXPECT_EQ(state.store.stats().versions,0u);EXPECT_EQ(state.store.stats().drafts,0u);
  }
  natural_target0={};warmup.clear();warm43={};initial43={};
  std::size_t submissions=converted?original->converted->cold_submissions:plain?original->mlx->cold_submissions:original->sparse->attention->cold_submissions;
  if(target43)add(submissions,converted?target43->converted->cold_submissions:plain?target43->mlx->cold_submissions:target43->sparse->attention->cold_submissions);
  if(plain)add(submissions,converted?target44->converted->cold_submissions:target44->mlx->cold_submissions);
  for(const auto *program:programs) {
    add(submissions,program->native_submissions());
    EXPECT_EQ(program->output_stats().live_blocks,0u);EXPECT_EQ(program->output_stats().active_uses,0u);
    for(std::size_t i=0;i<program->parameters.size();++i)EXPECT_EQ(read(program->parameters[i]),program->recipe.fragment.constant_bytes[program->parameter_slots[i]]);
  }
  EXPECT_TRUE(full_target_checked);EXPECT_EQ(full_target_values,2u*std::size_t(target_verifier::kVocabulary));
  EXPECT_EQ(verifier_calls,3u);EXPECT_GT(branch_counts[0],0u);EXPECT_GT(branch_counts[1],0u);
  if(plain) {
    ::testing::Test::RecordProperty("actual_submissions",int(submissions));
    ::testing::Test::RecordProperty("artifact_repository",std::string(converted?original->converted->parameters.source->identity().repository:mlx_roles::kRepository));::testing::Test::RecordProperty("artifact_revision",std::string(converted?original->converted->parameters.source->identity().revision:mlx_roles::kRevision));
    if(converted)::testing::Test::RecordProperty("artifact_variant",std::string(original->converted->parameters.source->identity().variant));
    ::testing::Test::RecordProperty("component_resources",32);::testing::Test::RecordProperty("nextn_resources",16);::testing::Test::RecordProperty("target43_resources",12);
    ::testing::Test::RecordProperty("target43_calls",int(target43_calls));::testing::Test::RecordProperty("target44_calls",int(target_calls));::testing::Test::RecordProperty("verifier_calls",int(verifier_calls));
    ::testing::Test::RecordProperty("target_full_f64_scores",std::to_string(full_target_values));::testing::Test::RecordProperty("accepted_target_conditioned_repairs",int(accepted_repairs));
    ::testing::Test::RecordProperty("natural_trials",1);::testing::Test::RecordProperty("natural_accepted_drafts",int(natural_accepted));
    ::testing::Test::RecordProperty("controlled_match_trials",1);::testing::Test::RecordProperty("controlled_reject_trials",1);
    ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(state_budget));::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
    ::testing::Test::RecordProperty("pool_encoding","plain F32 keys, no scale resource");
    ::testing::Test::RecordProperty("scope",converted?"Actual own converted GGUF target43/44 and NextN45, original packed fullV154880 M2 head and distinct finalnorm. F32 activation/hidden/journal/main/index state, model pooled index algorithm retained. Same StateHead/K1 actual current/proposal/pending, target-conditioned accepted1 draft-KV repair/catchup and epoch continuation. Upstream layer42 and initial histories are finite fixtures; not full-model quality/TPS or peer bit identity.":"Actual Q32 original MLX affine/F16 metadata target43/44 and NextN45, distinct globalnorm and original BF16 fullV154880 M2 head. F32 activations, raw/pool/hidden/journal state, explicit rank3 F32A/BF16W KV_b; no Quark/BF16 intermediate substitution. Same StateHead/K1 current/proposal, actual pending, accept1 target-conditioned draft-KV repair, accepted nx catchup and epoch continuation. Upstream layer42 fourstream and initial histories are finite fixtures, not full-model quality/TPS or producer-kernel bit identity.");
    return;
  }
  ::testing::Test::RecordProperty("actual_submissions",int(submissions));
  ::testing::Test::RecordProperty("target_submissions",int(target_calls));
  ::testing::Test::RecordProperty("target43_submissions",int(target43_calls));
  ::testing::Test::RecordProperty("target_verifier_submissions",int(verifier_calls));
  ::testing::Test::RecordProperty("target_full_vocabulary",int(target_verifier::kVocabulary));
  ::testing::Test::RecordProperty("target_first_full_reference_values",std::to_string(full_target_values));
  ::testing::Test::RecordProperty("natural_k1_trials",1);
  ::testing::Test::RecordProperty("natural_k1_proposal",natural_proposal);
  ::testing::Test::RecordProperty("natural_k1_accepted_drafts",int(natural_accepted));
  ::testing::Test::RecordProperty("controlled_matching_trials",1);
  ::testing::Test::RecordProperty("controlled_rejection_trials",1);
  ::testing::Test::RecordProperty("accepted_zero_draft_checkpoints",int(branch_counts[0]));
  ::testing::Test::RecordProperty("accepted_one_draft_checkpoints",int(branch_counts[1]));
  ::testing::Test::RecordProperty("full_original_nextn_roles",1762);
  ::testing::Test::RecordProperty("full_original_nextn_source_bytes","10030953088");
  std::size_t vocabulary_capacity=0;for(const auto &role:original->vocabulary->roles)add(vocabulary_capacity,role.lease.size());
  ::testing::Test::RecordProperty("shared_vocabulary_bank_capacity_bytes",std::to_string(vocabulary_capacity));
  ::testing::Test::RecordProperty("actual_pending_embedding_feedback_checks",int(feedback_checks));
  ::testing::Test::RecordProperty("state_budget_bytes",std::to_string(state_budget));
  ::testing::Test::RecordProperty("state_pool_backing_bound",std::to_string(pool_bound));
  ::testing::Test::RecordProperty("state_initial_backing_bound",std::to_string(initial_bound));
  ::testing::Test::RecordProperty("peak_state_backing_bytes",std::to_string(peak_backing));
  ::testing::Test::RecordProperty("component_schema",target43?
      "34 resources plus Epoch metadata:17 original NextN/indexer,13 independent original target43 MLA/indexer,4 original target44 KDA/conv/finalnormtap/cursor; same StateDraft/expected-base StateHead":target44?
      "34 resources plus Epoch metadata:17 original NextN/indexer,13 independent generated target43 MLA/indexer,4 original target44 KDA/conv/finalnormtap/cursor; same StateDraft/expected-base StateHead":
      "34 resources plus Epoch metadata:17 original NextN/indexer,13 independent generated target43 MLA/indexer,4 generated target44 KDA/conv/finalnormtap/cursor; same StateDraft/expected-base StateHead");
  ::testing::Test::RecordProperty("pending_policy","NaturalK1 compares actual original NextN proposal with target_id0 from full-V R2 head. accept0 keeps after-current first; accept1 keeps after-proposal second; nextpending directly retains target_ids[accepted_drafts] output lease. Matching/rejection controls are separate explicit policy fixtures,not natural acceptance.");
  if(target44) {
    EXPECT_TRUE(target44->full_checked);EXPECT_EQ(target44->checked_calls,target_calls);
    EXPECT_EQ(target44->moe->completed_episodes,target_calls);
    ::testing::Test::RecordProperty("original_target44_roles",1760);
    ::testing::Test::RecordProperty("original_target44_source_bytes",std::to_string(target44->logical_source_bytes));
    ::testing::Test::RecordProperty("original_target44_observed_calls",int(target44->checked_calls));
    ::testing::Test::RecordProperty("original_target44_expert_domain",int(original_moe::kExpertDomain));
    ::testing::Test::RecordProperty("original_target44_resident_experts",int(original_moe::kResidentCapacity));
  }
  if(target43) {
    EXPECT_TRUE(target43->full_checked);EXPECT_EQ(target43->checked_calls,target43_calls);
    EXPECT_EQ(target43->moe->completed_episodes,target43_calls);
    ::testing::Test::RecordProperty("original_target43_roles",1762);
    ::testing::Test::RecordProperty("original_target43_source_bytes",std::to_string(target43->logical_source_bytes));
    ::testing::Test::RecordProperty("original_target43_observed_calls",int(target43->checked_calls));
    ::testing::Test::RecordProperty("original_target43_expert_domain",int(original_moe::kExpertDomain));
    ::testing::Test::RecordProperty("original_target43_resident_experts",int(original_moe::kResidentCapacity));
    ::testing::Test::RecordProperty("scope","Distinct official original1762-role target43 HC/MLA/indexer/288-domain MoE, original1760-role target44 HC/KDA/MoE/finalnorm, and original1762-role layer45 NextN with shared full154880 vocabulary. Each sparse layer owns its actual parameter views and one cold KV preparation; common math, bounded expert supply, Program/Invocation and one34-resource plus Epoch StateHead are reused. Explicit source BF16 boundaries, torch/unfused HC and actual full-V K1 pending/accept/catchup/fresh-seed/skip remain checked. Layer42 inputs and finite initial histories remain generated fixtures,not a complete trained target or task-quality/natural-acceptance benchmark. Natural single trial is separate from matching/rejection controls; no TPS or physical-memory claim.");
  } else if(target44) {
    ::testing::Test::RecordProperty("scope","Distinct official1760-role target44 including its own HC/KDA/norms/288-domain MoE and distinct target model.norm, plus original1762-role layer45 NextN and shared full154880 vocabulary. Target44 uses explicit source BF16 boundaries, torch/unfused HC, F32 KDA[V,K] state, generic decode lowerbound-5, same bounded original-expert supply and full-V K1 verifier; no new executor. Target43 parameters,layer42 inputs and finite initial histories remain generated numerical fixtures,not a complete trained target or task-quality/natural-acceptance benchmark. Natural single trial separate from matching/rejection controls; no TPS or physical-memory claim.");
  } else ::testing::Test::RecordProperty("scope","Original1762-role layer45 NextN and shared full154880 vocabulary; actual generated target43->44 math yields finalnormalized F32 taps,explicit BF16 verifier boundary and same original head. Generated layer42 inputs/target43/44 weights and finite initial history remain; not a trained complete target model,task quality,natural acceptance benchmark or TPS claim. Fresh draft-extend uses PREVIOUS accepted target tap; skip uses actual same-epoch NextN hidden/selection,accepted journal catches up before epoch8 fresh; its published actual last-accepted fresh seed is activated and drives real skip0 without publishing the unverified next candidate.");
  if(target43&&target43->artifact_owner) {
    ::testing::Test::RecordProperty("scope","Actual immutable converted-artifact target43 HC/MLA/indexer, target44 HC/KDA/independent final norm, NextN45 and same original full154880 vocabulary use one34-resource plus Epoch StateHead and the existing K1 controller. Original E4 projections use the artifact MultiplyReciprocal/floor contract; target MoE uses original E2M1/UE8M0 group32 with explicit public MXFP4 A-QDQ/BF16-Y boundaries. Experts are authenticated only for actual selected global IDs. Logical role counts/source bytes describe the complete header schema,not whole-bank acquisition. Layer42 inputs and finite initial histories remain fixtures; controlled prefixes are policy checks,not natural model quality or TPS.");
    ::testing::Test::RecordProperty("target_moe_storage","MXFP4 E2M1+UE8M0 G32; actual A group32 QDQ; BF16 down before routeweight");
  }

}

std::string original_directory(const char *name) {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  if (_dupenv_s(&value, &size, name)) throw std::runtime_error("cannot read original GLM role directory");
  const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
  return value ? value : "";
#else
  const auto *value = std::getenv(name); return value ? value : "";
#endif
}
void run_original_indexer(c::Context &context, const c::StorageOwner &owner, bool cuda) {
  const auto input = original_directory("HYLE_GLM_MTP_INPUT_DIR");
  const auto query = original_directory("HYLE_GLM_MTP_QUERY_DIR");
  const auto indexer = original_directory("HYLE_GLM_MTP_INDEXER_DIR");
  if (input.empty() || query.empty() || indexer.empty()) {
    GTEST_SKIP() << "Set HYLE_GLM_MTP_INPUT_DIR, HYLE_GLM_MTP_QUERY_DIR and HYLE_GLM_MTP_INDEXER_DIR to authenticated layer45 roles";
  }
  run_seed_handoff(context, owner, cuda, original_nextn(input, query, indexer, owner));
}

void run_original_shared_lm(c::Context &context,const c::StorageOwner &owner,bool cuda) {
  const auto input=original_directory("HYLE_GLM_MTP_INPUT_DIR"),query=original_directory("HYLE_GLM_MTP_QUERY_DIR");
  const auto indexer=original_directory("HYLE_GLM_MTP_INDEXER_DIR"),directory=original_directory("HYLE_GLM_MTP_SHARED_LM_DIR");
  if(input.empty()||query.empty()||indexer.empty()||directory.empty())
    GTEST_SKIP()<<"Set authenticated original GLM input/query/indexer and HYLE_GLM_MTP_SHARED_LM_DIR roles";
  auto vocabulary=std::make_shared<const shared_lm::Vocabulary>(shared_lm::Vocabulary::load(directory,owner));
  run_seed_handoff(context,owner,cuda,original_nextn(input,query,indexer,owner,std::move(vocabulary)));
}
void run_original_attention(c::Context &context,const c::StorageOwner &owner,bool cuda) {
  const auto input=original_directory("HYLE_GLM_MTP_INPUT_DIR"),query=original_directory("HYLE_GLM_MTP_QUERY_DIR");
  const auto indexer=original_directory("HYLE_GLM_MTP_INDEXER_DIR"),lm=original_directory("HYLE_GLM_MTP_SHARED_LM_DIR");
  const auto attention=original_directory("HYLE_GLM_MTP_ATTENTION_DIR");
  if(input.empty()||query.empty()||indexer.empty()||lm.empty()||attention.empty())
    GTEST_SKIP()<<"Set authenticated original GLM input/query/indexer/shared-LM and HYLE_GLM_MTP_ATTENTION_DIR roles";
  auto vocabulary=std::make_shared<const shared_lm::Vocabulary>(shared_lm::Vocabulary::load(lm,owner));
  auto prepared=prepare_original_attention(attention,context,owner,cuda);
  run_seed_handoff(context,owner,cuda,original_nextn(input,query,indexer,owner,std::move(vocabulary),std::move(prepared)));
}
void run_original_moe(c::Context &context,const c::StorageOwner &owner,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ) {
  const auto input=original_directory("HYLE_GLM_MTP_INPUT_DIR"),query=original_directory("HYLE_GLM_MTP_QUERY_DIR");
  const auto indexer=original_directory("HYLE_GLM_MTP_INDEXER_DIR"),lm=original_directory("HYLE_GLM_MTP_SHARED_LM_DIR");
  const auto attention=original_directory("HYLE_GLM_MTP_ATTENTION_DIR"),directory=original_directory("HYLE_GLM_MTP_MOE_DIR");
  if(input.empty()||query.empty()||indexer.empty()||lm.empty()||attention.empty()||directory.empty())
    GTEST_SKIP()<<"Set authenticated original GLM input/query/indexer/LM/attention and HYLE_GLM_MTP_MOE_DIR roles";
  auto vocabulary=std::make_shared<const shared_lm::Vocabulary>(shared_lm::Vocabulary::load(lm,owner));
  auto prepared=prepare_original_attention(attention,context,owner,cuda);
  auto moe=std::make_shared<OriginalMoe>();
  auto data=std::make_shared<const original_moe::Data>(original_moe::Data::load(directory));
  moe->roles=original_moe::bind(std::move(data),owner);
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("actual CUDA expert supply requires the same typed compiled IO device");
    moe->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(resident_moe::CudaSupply::create(context,owner,*device),"one shared GLM CUDA expert supply"));
  } else
#endif
  moe->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(resident_moe::CpuSupply::create(owner),"one shared GLM CPU expert supply"));
  run_seed_handoff(context,owner,cuda,original_nextn(input,query,indexer,owner,std::move(vocabulary),std::move(prepared),moe));
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(moe->cuda_supply)check(moe->cuda_supply->close(),"close completed shared GLM CUDA uploads after every Invocation retired");
#endif
  if(moe->cpu_supply) {
    EXPECT_EQ(moe->cpu_supply->stats().live_blocks,0u);EXPECT_EQ(moe->cpu_supply->stats().active_uses,0u);
    EXPECT_EQ(moe->cpu_supply->stats().capacity_bytes,2*resident_moe::kSlotBytes);
  }
}

void run_original_target_verifier(c::Context &context,const c::StorageOwner &owner,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ,bool use_target44=false,bool use_target43=false,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1) {
  if(use_target43&&!use_target44)throw std::runtime_error("original target43 requires original target44 in the verifier chain");
  const auto input=original_directory("HYLE_GLM_MTP_INPUT_DIR"),query=original_directory("HYLE_GLM_MTP_QUERY_DIR");
  const auto indexer=original_directory("HYLE_GLM_MTP_INDEXER_DIR"),lm=original_directory("HYLE_GLM_MTP_SHARED_LM_DIR");
  const auto attention=original_directory("HYLE_GLM_MTP_ATTENTION_DIR"),directory=original_directory("HYLE_GLM_MTP_MOE_DIR");
  if(input.empty()||query.empty()||indexer.empty()||lm.empty()||attention.empty()||directory.empty())
    GTEST_SKIP()<<"Set all authenticated original GLM input/query/indexer/LM/attention/MoE directories for the complete NextN target verifier";
  const auto target44_directory=original_directory("HYLE_GLM_MTP_TARGET44_DIR");
  const auto target44_moe_directory=original_directory("HYLE_GLM_MTP_TARGET44_MOE_DIR");
  if(use_target44&&(target44_directory.empty()||target44_moe_directory.empty()))
    GTEST_SKIP()<<"Set authenticated HYLE_GLM_MTP_TARGET44_DIR and HYLE_GLM_MTP_TARGET44_MOE_DIR for distinct original target44 parameters";
  const auto target43_directory=original_directory("HYLE_GLM_MTP_TARGET43_DIR");
  const auto target43_moe_directory=original_directory("HYLE_GLM_MTP_TARGET43_MOE_DIR");
  if(use_target43&&(target43_directory.empty()||target43_moe_directory.empty()))
    GTEST_SKIP()<<"Set authenticated HYLE_GLM_MTP_TARGET43_DIR and HYLE_GLM_MTP_TARGET43_MOE_DIR for distinct original target43 parameters";
  auto vocabulary=std::make_shared<const shared_lm::Vocabulary>(shared_lm::Vocabulary::load(lm,owner));
  auto prepared=prepare_original_attention(attention,context,owner,cuda);
  auto moe=std::make_shared<OriginalMoe>();
  auto data=std::make_shared<const original_moe::Data>(original_moe::Data::load(directory));
  moe->roles=original_moe::bind(std::move(data),owner);
#if defined(HYLE_TEST_GLM_STATE_CUDA)
  if(cuda) {
    if(!device)throw std::runtime_error("target verifier requires its actual shared compiled IO/expert CUDA resource");
    moe->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(resident_moe::CudaSupply::create(context,owner,*device),"same complete original NextN CUDA expert supply"));
  } else
#endif
  moe->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(resident_moe::CpuSupply::create(owner),"same complete original NextN CPU expert supply"));
  std::shared_ptr<OriginalTarget44> target44;
  if(use_target44) {
    target44=std::make_shared<OriginalTarget44>();
    auto parameters=std::make_shared<const original_target44::Data>(original_target44::Data::load(target44_directory));
    target44->parameters=original_target44::bind(std::move(parameters),owner);
    target44->moe=std::make_shared<OriginalMoe>();
    auto source=std::make_shared<const original_moe::Data>(original_target44::load_moe(target44_moe_directory));
    target44->moe->roles=original_moe::bind(std::move(source),owner);
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if(cuda) {
      if(!device)throw std::runtime_error("original target44 requires the same actual Graph/expert CUDA resource");
      target44->moe->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(resident_moe::CudaSupply::create(context,owner,*device),"same original target44 CUDA expert supply"));
    } else
#endif
    target44->moe->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(resident_moe::CpuSupply::create(owner),"same original target44 CPU expert supply"));
  }
  std::shared_ptr<OriginalTarget43> target43;
  if(use_target43) {
    target43=std::make_shared<OriginalTarget43>();
    auto parameters=std::make_shared<const original_target43::Data>(original_target43::Data::load(target43_directory));
    target43->parameters=original_target43::bind(parameters,owner);
    target43->sparse=std::make_shared<OriginalSparseLayer>();
    target43->sparse->indexer=std::shared_ptr<const original_index::Data>(parameters,&parameters->indexer);
    target43->sparse->indexer_roles=original_index::bind(*target43->sparse->indexer,owner);
    target43->sparse->attention=prepare_original_attention(
        std::shared_ptr<const original_attention::Data>(parameters,&parameters->attention),context,owner,cuda);
    target43->moe=std::make_shared<OriginalMoe>();
    auto source=std::make_shared<const original_moe::Data>(original_target43::load_moe(target43_moe_directory));
    target43->moe->roles=original_moe::bind(std::move(source),owner);
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if(cuda) {
      if(!device)throw std::runtime_error("original target43 requires the same actual Graph/expert CUDA resource");
      target43->moe->cuda_supply=std::make_unique<resident_moe::CudaSupply>(take(resident_moe::CudaSupply::create(context,owner,*device),"same original target43 CUDA expert supply"));
    } else
#endif
    target43->moe->cpu_supply=std::make_unique<resident_moe::CpuSupply>(take(resident_moe::CpuSupply::create(owner),"same original target43 CPU expert supply"));
  }
  run_target_verifier(context,owner,cuda,original_nextn(input,query,indexer,owner,std::move(vocabulary),std::move(prepared),moe),target44,target43,stochastic,true,limits,maximum_drafts);
  for(const auto &source:{moe,target44?target44->moe:std::shared_ptr<OriginalMoe>{},target43?target43->moe:std::shared_ptr<OriginalMoe>{}}) {
    if(!source)continue;
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if(source->cuda_supply)check(source->cuda_supply->close(),"close verifier layer-owned expert uploads after all actual readers retired");
#endif
    if(source->cpu_supply) {
      EXPECT_EQ(source->cpu_supply->stats().live_blocks,0u);EXPECT_EQ(source->cpu_supply->stats().active_uses,0u);
      EXPECT_EQ(source->cpu_supply->stats().capacity_bytes,2*resident_moe::kSlotBytes);
    }
  }
}

#if defined(HYLE_TEST_GLM_Q30_ACTUAL_PARAMETERS)
void run_quark_target_verifier(c::Context &context,const c::StorageOwner &owner,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1) {
  namespace nextn_pins=glm_original_artifacts::q30::nextn_static;
  namespace pins43=glm_original_artifacts::q30::target43_static;
  namespace pins44=glm_original_artifacts::q30::target44_static;
  const std::filesystem::path root=original_directory("HYLE_GLM_Q30_ORIGINAL_DIR");
  if(root.empty())GTEST_SKIP()<<"Set HYLE_GLM_Q30_ORIGINAL_DIR to the actual same-artifact static and selected expert cache";
  if(nextn_pins::kRepository!=pins43::kRepository||nextn_pins::kRepository!=pins44::kRepository||
      nextn_pins::kRevision!=pins43::kRevision||nextn_pins::kRevision!=pins44::kRevision||
      nextn_pins::kProfile!=pins43::kProfile||nextn_pins::kProfile!=pins44::kProfile)
    throw std::runtime_error("Q30 target/NextN/vocabulary generated artifact profiles differ");
  const auto files45=artifact_source::files(nextn_pins::kRoles);
  const auto files43=artifact_source::files(pins43::kRoles),files44=artifact_source::files(pins44::kRoles);
  const glm_fp8_arithmetic::Arithmetic arithmetic{c::model_ops::Float8ScalePolicy::kMultiplyReciprocal,1.0e-10F};
  const auto identity=[&](unsigned layer) {return original_moe::SourceIdentity{std::string(nextn_pins::kRepository),
      std::string(nextn_pins::kRevision),"model.language_model.layers."+std::to_string(layer)+".","weight_scale"};};
  const artifact_source::Manifest manifest45{identity(45),arithmetic,"model.language_model.embed_tokens.weight","lm_head.weight",files45};
  const artifact_source::Manifest manifest43{identity(43),arithmetic,"model.language_model.embed_tokens.weight","lm_head.weight",files43};
  const artifact_source::Manifest manifest44{identity(44),arithmetic,"model.language_model.embed_tokens.weight","lm_head.weight",files44};
  const bool wait=original_directory("HYLE_GLM_Q30_WAIT_FOR_EXPERTS")=="1";
  const std::filesystem::path request=original_directory("HYLE_GLM_Q30_FRONTIER_FILE");
  const auto hook=[&](unsigned layer,std::string_view plan,original_moe::WeightStorage storage) {
    const auto experts=layer==45?root/"original-experts":root/("original-target"+std::to_string(layer))/"experts";
    glm_original_expert_frontier::Frontier frontier{{identity(layer),std::string(plan),storage,layer==45?"nextn":"target"+std::to_string(layer)},experts,request,wait};
    return OriginalExpertHook([frontier=std::move(frontier)](auto source,auto ids){frontier.ensure(std::move(source),ids);});
  };
  auto data=artifact_source::Data::load(root/"original-static",manifest45);
  auto nextn=original_nextn(data,context,owner,cuda,nextn_pins::kProfile,
      hook(45,nextn_pins::kExpertPlanSha256,original_moe::WeightStorage::kFloat8Block128)
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  auto targets=artifact_targets(nextn,root/"original-target43"/"static",manifest43,
      root/"original-target44"/"static",manifest44,context,owner,cuda,
      hook(43,pins43::kExpertPlanSha256,original_moe::WeightStorage::kMxfp4Group32),
      hook(44,pins44::kExpertPlanSha256,original_moe::WeightStorage::kMxfp4Group32)
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  run_target_verifier(context,owner,cuda,nextn,targets.target44,targets.target43,stochastic,false,limits,maximum_drafts);
  for(const auto &moe:{nextn->moe,targets.target43->moe,targets.target44->moe}) {
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if(moe->cuda_supply)check(moe->cuda_supply->close(),"close actual artifact expert uploads after all original readers retire");
#endif
    if(moe->cpu_supply) {
      EXPECT_EQ(moe->cpu_supply->stats().live_blocks,0u);EXPECT_EQ(moe->cpu_supply->stats().active_uses,0u);
      EXPECT_EQ(moe->cpu_supply->stats().capacity_bytes,2*moe->cpu_supply->layout().slot_bytes);
    }
  }
  ::testing::Test::RecordProperty("artifact_repository",std::string(nextn_pins::kRepository));
  ::testing::Test::RecordProperty("artifact_revision",std::string(nextn_pins::kRevision));
  ::testing::Test::RecordProperty("nextn_static_manifest_sha256",std::string(nextn_pins::kManifestSha256));
  ::testing::Test::RecordProperty("target43_static_manifest_sha256",std::string(pins43::kManifestSha256));
  ::testing::Test::RecordProperty("target44_static_manifest_sha256",std::string(pins44::kManifestSha256));
}
#endif


#if defined(HYLE_TEST_GLM_MLX_ACTUAL_PARAMETERS)
void run_mlx_target_verifier(c::Context &context,const c::StorageOwner &owner,bool cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    ,const c::cuda::MemoryResource *device=nullptr
#endif
    ,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1) {
  namespace pin=glm_mlx_original_artifacts;namespace target_pin=glm_mlx_target_artifacts;
  const auto nextn_dir=original_directory("HYLE_GLM_MLX_NEXTN_DIR"),target_dir=original_directory("HYLE_GLM_MLX_TARGET_DIR"),
      expert_dir=original_directory("HYLE_GLM_MLX_EXPERT_DIR");
  if(nextn_dir.empty()||target_dir.empty()||expert_dir.empty())GTEST_SKIP()<<"Set actual original Q32 NextN41, target76 and selected9-plane expert cache directories";
  if(pin::kRepository!=mlx_roles::kRepository||pin::kRevision!=mlx_roles::kRevision||target_pin::kRepository!=pin::kRepository||target_pin::kRevision!=pin::kRevision)
    throw std::runtime_error("MLX generated actual source identities differ");
  struct Cached final {std::filesystem::path nextn_path,target_path;std::shared_ptr<mlx_roles::Data> nextn;std::array<std::shared_ptr<mlx_roles::Data>,2> targets;};
  static const auto cached=[&] {
    auto r=std::make_shared<Cached>();r->nextn_path=nextn_dir;r->target_path=target_dir;
    r->nextn=mlx_roles::Data::load(nextn_dir,pin::kStaticCatalogueSha256,pin::kExpertSourcePlanSha256);
    r->targets=mlx_roles::Data::load_target(target_dir,target_pin::kTargetStaticCatalogueSha256,target_pin::kTargetExpertSourcePlanSha256);return r;
  }();
  if(cached->nextn_path!=nextn_dir||cached->target_path!=target_dir)throw std::runtime_error("MLX source directories changed between related backend cases");
  auto nextn=std::make_shared<OriginalNextn>();nextn->profile_description=mlx_runtime::kProfile;nextn->profile=take(c::ModelArtifactProfile::create(mlx_runtime::kProfile),"Q32 actual artifact profile");
  nextn->mlx=prepare_mlx_source(mlx_roles::bind(cached->nextn,owner),context,owner,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  auto bound=mlx_roles::bind(cached->targets[0],owner);std::array<mlx_roles::Bound,2> target_bounds;
  for(std::size_t stage=0;stage<2;++stage)target_bounds[stage].source=cached->targets[stage];
  for(auto &role:bound.roles){const auto stage=role.name.starts_with("model.language_model.layers.43.")?0U:1U;target_bounds[stage].roles.push_back(std::move(role));}
  auto target43=std::make_shared<OriginalTarget43>();auto target44=std::make_shared<OriginalTarget44>();
  target43->artifact_owner=nextn;target44->artifact_owner=nextn;
  target43->logical_source_bytes=129522520;target44->logical_source_bytes=299878232+8192;
  target43->mlx=prepare_mlx_source(std::move(target_bounds[0]),context,owner,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  target44->mlx=prepare_mlx_source(std::move(target_bounds[1]),context,owner,cuda
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,device
#endif
      );
  const auto request=original_directory("HYLE_GLM_MLX_FRONTIER_FILE");const auto wait=original_directory("HYLE_GLM_MLX_WAIT_FOR_EXPERTS")=="1";
  for(const auto &source:{nextn->mlx,target43->mlx,target44->mlx}) {
    auto frontier=std::make_shared<glm_mlx_expert_frontier::Frontier>();frontier->selected=source->parameters.source;frontier->root=expert_dir;frontier->request=request;frontier->wait=wait;
    frontier->section=frontier->selected->prefix()==mlx_roles::kPrefix?"nextn":frontier->selected->prefix()=="model.language_model.layers.43."?"target43":"target44";
    source->before_upload=[frontier](std::shared_ptr<mlx_roles::Data> data,std::span<const std::int32_t> ids){
      if(data!=frontier->selected)throw std::runtime_error("MLX frontier source owner differs");frontier->ensure(ids);};
  }
  run_target_verifier(context,owner,cuda,nextn,target44,target43,stochastic,false,limits,maximum_drafts);
  std::size_t selected_experts=0;
  for(const auto &source:{nextn->mlx,target43->mlx,target44->mlx}) {
    std::set<std::int32_t> selected;for(const auto &selection:source->selections)selected.insert(selection.globals.begin(),selection.globals.end());selected_experts+=selected.size();
#if defined(HYLE_TEST_GLM_STATE_CUDA)
    if(source->cuda_supply)check(source->cuda_supply->close(),"close retired original MLX plane uploads");
#endif
    if(source->cpu_supply){EXPECT_EQ(source->cpu_supply->stats().active_uses,0U);EXPECT_EQ(source->cpu_supply->stats().live_blocks,0U);}
  }
  ::testing::Test::RecordProperty("static_original_roles",117);::testing::Test::RecordProperty("static_original_bytes","3162045744");
  ::testing::Test::RecordProperty("expert_domain",288);::testing::Test::RecordProperty("resident_experts",8);::testing::Test::RecordProperty("expert_planes",9);
  ::testing::Test::RecordProperty("actual_selected_original_experts",int(selected_experts));::testing::Test::RecordProperty("actual_selected_original_bytes",std::to_string(selected_experts*mlx_roles::kExpertBytes));
  ::testing::Test::RecordProperty("nextn_catalogue_sha256",std::string(pin::kStaticCatalogueSha256));::testing::Test::RecordProperty("target_catalogue_sha256",std::string(target_pin::kTargetStaticCatalogueSha256));
}
#endif

void run_gguf_flash_target_verifier(c::Context &context,const c::StorageOwner &owner,bool cuda,
    const gguf_source::Fixture &artifact,const char *static_env,const char *expert_env,bool stochastic=false,hyle::inference::mtp::SamplingLimits limits={},std::size_t maximum_drafts=1) {
  const auto directory=original_directory(static_env),experts=original_directory(expert_env);
  if(directory.empty()||experts.empty())throw std::runtime_error("requested converted Flash case requires its own actual static/expert directories");
  const auto data=gguf_source::load(directory,artifact);auto bound=original_named_role_source::bind(data,owner);
  const auto request=original_directory("HYLE_GLM_GGUF_FLASH_FRONTIER_FILE");const bool wait=original_directory("HYLE_GLM_GGUF_FLASH_WAIT_FOR_EXPERTS")=="1";
  auto nextn=std::make_shared<OriginalNextn>();nextn->profile_description=artifact.profile;nextn->profile=take(c::ModelArtifactProfile::create(artifact.profile),"converted Flash own actual profile");
  nextn->converted=gguf_source::prepare(bound,artifact,2,owner,experts,request,wait);
  auto target43=std::make_shared<OriginalTarget43>();target43->artifact_owner=nextn;target43->converted=gguf_source::prepare(bound,artifact,0,owner,experts,request,wait);
  auto target44=std::make_shared<OriginalTarget44>();target44->artifact_owner=nextn;target44->converted=gguf_source::prepare(std::move(bound),artifact,1,owner,experts,request,wait);
  run_target_verifier(context,owner,cuda,nextn,target44,target43,stochastic,false,limits,maximum_drafts);
  std::size_t selected=0,selected_bytes=0;
  for(const auto &source:{target43->converted,target44->converted,nextn->converted}) {
    EXPECT_TRUE(source->full_checked);EXPECT_GT(source->episodes,0u);std::set<std::int32_t> ids;
    for(const auto &selection:source->selections)ids.insert(selection.globals.begin(),selection.globals.end());selected+=ids.size();
    for(const auto id:ids)for(std::size_t leg=0;leg<3;++leg)selected_bytes+=source->experts.source->expert_bytes(glm_gguf_runtime_source::Leg(leg),id).size();
    EXPECT_EQ(source->cpu_supply->stats().active_uses,0U);EXPECT_EQ(source->cpu_supply->stats().live_blocks,0U);
  }
  EXPECT_EQ(target43->checked_calls,target43->converted->episodes);EXPECT_EQ(target44->checked_calls,target44->converted->episodes);
  ::testing::Test::RecordProperty("static_original_roles",int(artifact.static_specs.size()));::testing::Test::RecordProperty("static_original_bytes",std::to_string(artifact.static_bytes));
  ::testing::Test::RecordProperty("static_source_plan_sha256",std::string(artifact.static_plan_sha256));::testing::Test::RecordProperty("expert_source_plan_sha256",std::string(artifact.expert_plan_sha256));
  ::testing::Test::RecordProperty("expert_domain",288);::testing::Test::RecordProperty("resident_experts",8);::testing::Test::RecordProperty("expert_planes",3);
  ::testing::Test::RecordProperty("actual_selected_original_experts",int(selected));::testing::Test::RecordProperty("actual_selected_original_bytes",std::to_string(selected_bytes));
}
#include "glm_flash_early_history_case.h"
#include "glm_flash_ordinary_startup_case.h"

#if defined(HYLE_TEST_GLM_STATE_CUDA)
struct SharedCudaDevice final {
  std::shared_ptr<c::cuda::MemoryResource> resource;
  m::MemoryResourceRef ref() noexcept {return resource->ref();}
};
#if defined(HYLE_TEST_GLM_B12_ARTIFACTS_HEADER)
TEST(GlmMtpStateConsumer,CudaB12OriginalGgufTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"B12 actual converted Flash CUDA Context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B12 actual CUDA allocation domain")),"B12 same original source/Graph/expert owner");
  run_gguf_flash_target_verifier(context,owner,true,glm_gguf_b12_artifacts::fixture(),"HYLE_GLM_B12_ORIGINAL_DIR","HYLE_GLM_B12_EXPERT_DIR");
}
#endif
#if defined(HYLE_TEST_GLM_B13_ARTIFACTS_HEADER)
TEST(GlmMtpStateConsumer,CudaB13OriginalGgufTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"B13 actual converted Flash CUDA Context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"B13 actual CUDA allocation domain")),"B13 same original source/Graph/expert owner");
  run_gguf_flash_target_verifier(context,owner,true,glm_gguf_b13_artifacts::fixture(),"HYLE_GLM_B13_ORIGINAL_DIR","HYLE_GLM_B13_EXPERT_DIR");
}
#endif
#if defined(HYLE_TEST_GLM_MLX_ACTUAL_PARAMETERS)
TEST(GlmMtpStateConsumer,CudaMlxOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"Q32 actual MLX CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice),"Q32 actual CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"Q32 same original source/Graph/expert owner");run_mlx_target_verifier(context,owner,true,device.get());
}
#endif
#if defined(HYLE_TEST_GLM_Q30_ACTUAL_PARAMETERS)
TEST(GlmMtpStateConsumer,CudaQuarkOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"Q30 actual original target/NextN CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"Q30 same actual CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"Q30 shared original source/Graph/expert owner");
  run_quark_target_verifier(context,owner,true,device.get());
}
TEST(GlmMtpStateConsumer,CudaQuarkOriginalTarget43And44AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cuda::create_context(),"Q30 actual original target/NextN CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"Q30 same actual CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"Q30 shared original source/Graph/expert owner");
  run_quark_target_verifier(context,owner,true,device.get(),true);
}
#endif
TEST(GlmMtpStateConsumer,CudaOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"original target43/44/NextN verifier CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"same actual original target43/44/NextN allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared original target43/44/NextN owner");
  run_original_target_verifier(context,owner,true,device.get(),true,true);
}
TEST(GlmMtpStateConsumer,CudaOriginalTarget43And44AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cuda::create_context(),"original random target43/44/NextN CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"actual random GLM source allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared random GLM source/Graph/expert owner");
  run_original_target_verifier(context,owner,true,device.get(),true,true,true);
}
TEST(GlmMtpStateConsumer,CudaOriginalTarget43And44AndNextNRandomK2PublishesCompletePrefixes) {
  auto context=take(g::cuda::create_context(),"original K2 target43/44/NextN CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"actual K2 GLM source allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared K2 GLM source/Graph/expert owner");
  run_original_target_verifier(context,owner,true,device.get(),true,true,true,{},2);
}
TEST(GlmMtpStateConsumer,CudaOriginalTarget44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cuda::create_context(),"original target44/NextN verifier CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"same actual original target44/NextN allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared original target44/NextN owner");
  run_original_target_verifier(context,owner,true,device.get(),true);
}
TEST(GlmMtpStateConsumer,CudaOriginalNextNFullVocabularyTargetVerifierPublishesActualPending) {
  auto context=take(g::cuda::create_context(),"complete original NextN target verifier CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"one actual target/verifier/draft allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"same CUDA target/draft/verifier readonly owner");
  run_original_target_verifier(context,owner,true,device.get());
}
TEST(GlmMtpStateConsumer,CudaOriginalMoeCompletesNextNAndFeedsBackAcrossSeedAndCatchup) {
  auto context=take(g::cuda::create_context(),"original complete NextN CUDA Context");
  auto device=std::make_shared<c::cuda::MemoryResource>(take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"same actual CUDA allocation domain"));
  const auto owner=take(c::StorageOwner::adopt(context,SharedCudaDevice{device}),"shared CUDA Graph and expert owner");
  run_original_moe(context,owner,true,device.get());
}
TEST(GlmMtpStateConsumer,CudaOriginalAttentionSharedLmSeedsSkipsCatchesUpAndFeedsBack) {
  auto context=take(g::cuda::create_context(),"original complete-MLA CUDA Context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original complete-MLA CUDA memory")),"original complete-MLA storage owner");
  run_original_attention(context,owner,true);
}
TEST(GlmMtpStateConsumer,CudaOriginalSharedLmPredictsAndFeedsBackAcrossSeedAndCatchup) {
  auto context=take(g::cuda::create_context(),"original shared-LM CUDA Context");
  const auto owner=take(c::StorageOwner::adopt(context,take(c::cuda::MemoryResource::create(
      context,c::cuda::MemoryKind::kDevice),"original shared-LM CUDA memory")),"original shared-LM storage owner");
  run_original_shared_lm(context,owner,true);
}
TEST(GlmMtpStateConsumer, CudaOriginalIndexerPublishesCatchesUpAndReseeds) {
  auto context = take(g::cuda::create_context(), "ordinary original-indexer CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "original-indexer CUDA memory")), "original-indexer storage owner");
  run_original_indexer(context, owner, true);
}
TEST(GlmMtpStateConsumer, CudaTarget43MlaFlowsThroughTarget44AndNextN) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "ordinary CUDA memory")), "retained CUDA allocation domain");
  run_target43(context, owner, true);
}
TEST(GlmMtpStateConsumer, CudaWholeNextNPrefixPublishesAndContinuesWithoutRejectedTail) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "ordinary CUDA memory")), "retained CUDA allocation domain");
  run(context, owner, true);
}
TEST(GlmMtpStateConsumer, CudaAcceptedJournalCatchesUpAndReseedsNextIteration) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "ordinary CUDA memory")), "retained CUDA allocation domain");
  run_catchup(context, owner, true);
}
TEST(GlmMtpStateConsumer, CudaTargetPrefixAndNextNPublishAndReseedTogether) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "ordinary CUDA memory")), "retained CUDA allocation domain");
  run_target(context, owner, true);
}
TEST(GlmMtpStateConsumer, CudaDraftExtendLastAcceptedSeedSkipsDecodeStepZero) {
  auto context = take(g::cuda::create_context(), "ordinary CUDA context");
  const auto owner = take(c::StorageOwner::adopt(context, take(c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice), "ordinary CUDA memory")), "retained CUDA allocation domain");
  run_seed_handoff(context, owner, true);
}
#endif
#if defined(HYLE_TEST_GLM_STATE_CPU)
struct SystemResource final { m::MemoryResourceRef ref() noexcept { return m::system_resource(); } };
#if defined(HYLE_TEST_GLM_B12_ARTIFACTS_HEADER)
TEST(GlmMtpStateConsumer,CpuB12OriginalGgufTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"B12 actual converted Flash CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B12 same original source/Graph/expert owner");
  run_gguf_flash_target_verifier(context,owner,false,glm_gguf_b12_artifacts::fixture(),"HYLE_GLM_B12_ORIGINAL_DIR","HYLE_GLM_B12_EXPERT_DIR");
}
#endif
#if defined(HYLE_TEST_GLM_B13_ARTIFACTS_HEADER)
TEST(GlmMtpStateConsumer,CpuB13OriginalGgufTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"B13 actual converted Flash CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"B13 same original source/Graph/expert owner");
  run_gguf_flash_target_verifier(context,owner,false,glm_gguf_b13_artifacts::fixture(),"HYLE_GLM_B13_ORIGINAL_DIR","HYLE_GLM_B13_EXPERT_DIR");
}
#endif
#if defined(HYLE_TEST_GLM_MLX_ACTUAL_PARAMETERS)
TEST(GlmMtpStateConsumer,CpuMlxOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"Q32 actual MLX CPU Context");const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"Q32 same original source/Graph/expert CPU owner");
  run_mlx_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      );
}
#endif
#if defined(HYLE_TEST_GLM_Q30_ACTUAL_PARAMETERS)
TEST(GlmMtpStateConsumer,CpuQuarkOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"Q30 actual original target/NextN CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"Q30 shared original source/Graph/expert CPU owner");
  run_quark_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      );
}
TEST(GlmMtpStateConsumer,CpuQuarkOriginalTarget43And44AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cpu::create_context(),"Q30 actual original target/NextN CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"Q30 shared original source/Graph/expert CPU owner");
  run_quark_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,true);
}
#endif
TEST(GlmMtpStateConsumer,CpuOriginalTarget43And44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"original target43/44/NextN verifier CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"shared original target43/44/NextN CPU owner");
  run_original_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,true,true);
}
TEST(GlmMtpStateConsumer,CpuOriginalTarget43And44AndNextNRandomVerifierContinuesAtomically) {
  auto context=take(g::cpu::create_context(),"original random target43/44/NextN CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"shared original random GLM CPU owner");
  run_original_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,true,true,true);
}
TEST(GlmMtpStateConsumer,CpuOriginalTarget44AndNextNFullVocabularyVerifier) {
  auto context=take(g::cpu::create_context(),"original target44/NextN verifier CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"shared original target44/NextN CPU owner");
  run_original_target_verifier(context,owner,false
#if defined(HYLE_TEST_GLM_STATE_CUDA)
      ,nullptr
#endif
      ,true);
}
TEST(GlmMtpStateConsumer,CpuOriginalNextNFullVocabularyTargetVerifierPublishesActualPending) {
  auto context=take(g::cpu::create_context(),"complete original NextN target verifier CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"same CPU target/draft/verifier readonly owner");
  run_original_target_verifier(context,owner,false);
}
TEST(GlmMtpStateConsumer,CpuOriginalMoeCompletesNextNAndFeedsBackAcrossSeedAndCatchup) {
  auto context=take(g::cpu::create_context(),"original complete NextN CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original complete NextN storage owner");
  run_original_moe(context,owner,false);
}
TEST(GlmMtpStateConsumer,CpuOriginalAttentionSharedLmSeedsSkipsCatchesUpAndFeedsBack) {
  auto context=take(g::cpu::create_context(),"original complete-MLA CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original complete-MLA storage owner");
  run_original_attention(context,owner,false);
}
TEST(GlmMtpStateConsumer,CpuOriginalSharedLmPredictsAndFeedsBackAcrossSeedAndCatchup) {
  auto context=take(g::cpu::create_context(),"original shared-LM CPU Context");
  const auto owner=take(c::StorageOwner::adopt(context,SystemResource{}),"original shared-LM storage owner");
  run_original_shared_lm(context,owner,false);
}
TEST(GlmMtpStateConsumer, CpuOriginalIndexerPublishesCatchesUpAndReseeds) {
  auto context = take(g::cpu::create_context(), "ordinary original-indexer CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "original-indexer storage owner");
  run_original_indexer(context, owner, false);
}
TEST(GlmMtpStateConsumer, CpuTarget43MlaFlowsThroughTarget44AndNextN) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run_target43(context, owner, false);
}
TEST(GlmMtpStateConsumer, CpuWholeNextNPrefixPublishesAndContinuesWithoutRejectedTail) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run(context, owner, false);
}
TEST(GlmMtpStateConsumer, CpuAcceptedJournalCatchesUpAndReseedsNextIteration) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run_catchup(context, owner, false);
}
TEST(GlmMtpStateConsumer, CpuTargetPrefixAndNextNPublishAndReseedTogether) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run_target(context, owner, false);
}
TEST(GlmMtpStateConsumer, CpuDraftExtendLastAcceptedSeedSkipsDecodeStepZero) {
  auto context = take(g::cpu::create_context(), "ordinary CPU context");
  const auto owner = take(c::StorageOwner::adopt(context, SystemResource{}), "retained CPU allocation domain");
  run_seed_handoff(context, owner, false);
}
#endif
} // namespace
