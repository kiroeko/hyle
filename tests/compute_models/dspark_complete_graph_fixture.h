#pragma once

#include <functional>
#include <string_view>

#include "cpu_c4_model_branches.h"

#include <bit>
#include <cstring>
#include <span>

// Test-only composition of the existing full Flash draft and its actual
// five-row greedy feedback. No executor, teacher tape or model-specific API.
namespace dspark_complete_graph {
namespace base = cpu_c4_model_branches;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace c = hyle::compute;
using sem::ValueId;
inline constexpr std::int64_t hidden = 4096, rank = 256;
inline constexpr auto rows = base::block, vocabulary = base::vocabulary;

enum Input : std::size_t {
  kMain, kFirst, kMainPosition, kDraftPositions, kWriting, kMask,
  kCache0, kCache1, kCache2, kInputs
};
enum Output : std::size_t {
  kState, kNextCache0, kNextCache1, kNextCache2, kHidden, kNormalized,
  kIds, kMarkov, kLogits, kConfidence, kOutputs
};
struct HeadParameters final {
  std::span<const std::uint16_t> markov_w1_bf16{}, markov_w2_bf16{}, confidence_bf16{};
  // Optional original immutable BF16 banks. Cold construction only; default
  // generated/previous paths retain their existing widening and arithmetic.
  std::function<ValueId(sem::Fragment &,std::string_view)> readonly_parameter{};
};
// Optional cold construction hooks for the actual target-owned vocabulary.
// Both hooks are required together; no generated fallback fills missing rows.
struct SharedParameters final {
  base::DraftEmbeddingTable embedding_table;
  std::function<ValueId(sem::Fragment &,ValueId)> logits;
};
struct ColdWidening final {
  std::size_t output;
  ValueId source;
};
struct CompleteProgram final {
  sem::Fragment fragment;
  // Cold F32 outputs can be checked in bounded chunks against their actual
  // BF16 source, without an additional 2GiB host F32 LM reference bank.
  std::vector<ColdWidening> cold_widenings;
  struct EncodedPorts final {
    ValueId pending,positions,mask;
    std::array<ValueId,3> histories;
    ValueId state,hidden,normalized,ids,markov,logits,confidence;
  };
  std::optional<EncodedPorts> encoded{};
  std::optional<base::DraftObservations> observations{};
};
inline std::uint16_t encode(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<std::uint16_t>(bits >> 16u);
}
inline float widen(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t{bits} << 16u);
}

// The old sparse BF16-source construction rule, applied to one complete
// vocabulary bank shared by all five rows. Shard-local seeds are not reused.
// This is a
// generated test LM, never claimed to be an original trained LM payload.
inline std::size_t lm_column(std::size_t token) {
  return (17 * token + 491) % hidden;
}
inline float lm_coefficient(std::size_t token) {
  return widen(encode(0.5F + float((token + 491) % 13) / 32.0F));
}
inline std::vector<float> lm_reference(std::span<const float> normalized) {
  if (normalized.size() != std::size_t(rows * hidden))
    throw std::runtime_error("DSpark LM reference input extent");
  std::vector<float> result(rows * vocabulary);
  for (std::size_t row = 0; row < std::size_t(rows); ++row)
    for (std::size_t token = 0; token < std::size_t(vocabulary); ++token)
      result[row * vocabulary + token] = float(
          double(normalized[row * hidden + lm_column(token)]) * lm_coefficient(token));
  return result;
}

// Remove only the former teacher-fed Markov/confidence branch. The original
// three stages, their constants, six outputs and independent oracle remain.
inline void remove_teacher_branch(sem::Fragment &f,bool encoded=false,std::size_t resident_inputs=0,
    std::span<const ValueId> observation_roots={}) {
  f.outputs.resize(encoded?3:kIds);
  std::vector<bool> live(f.descriptions.size());
  for (auto value : f.outputs) live[value.ordinal()] = true;
  // Declared diagnostics are real consumers, including metadata operands
  // not used by an alternative typed kernel (BF16 output-a has no group IDs).
  for (auto value : observation_roots) {
    if(value.ordinal()>=live.size())throw std::runtime_error("draft observation root is outside its Fragment");
    live[value.ordinal()]=true;
  }
  std::vector<bool> needed(f.nodes.size());
  for (std::size_t i = f.nodes.size(); i-- > 0;) {
    for (auto output : f.nodes[i].outputs) needed[i] = needed[i] || live[output.ordinal()];
    if (needed[i]) for (auto input : f.nodes[i].inputs) live[input.ordinal()] = true;
  }
  std::vector<sem::recipe::DirectNode> nodes;
  for (std::size_t i = 0; i < f.nodes.size(); ++i)
    if (needed[i]) nodes.push_back(std::move(f.nodes[i]));
  f.nodes = std::move(nodes);
  std::vector<ValueId> constants;
  std::vector<std::vector<std::byte>> payloads;
  const auto old_prepared = f.prepared_constants;
  std::vector<std::size_t> slots(f.constants.size(),std::numeric_limits<std::size_t>::max());
  for (std::size_t i = 0; i < f.constants.size(); ++i) {
    const auto found = std::find(f.prepared_constants.begin(), f.prepared_constants.end(), i);
    if (!live[f.constants[i].ordinal()]) {
      if (found != f.prepared_constants.end())
        throw std::runtime_error("teacher removal would discard a stage cold parameter");
      continue;
    }
    slots[i]=constants.size();
    constants.push_back(f.constants[i]);
    payloads.push_back(std::move(f.constant_bytes[i]));
  }
  f.constants = std::move(constants); f.constant_bytes = std::move(payloads);
  f.prepared_constants.clear();
  for (const auto old_slot : old_prepared) f.prepared_constants.push_back(slots[old_slot]);
  const auto teacher=encoded?1u:2u;
  if (f.inputs.size() != (encoded?7u:10u)+resident_inputs || live[f.inputs[teacher].ordinal()])
    throw std::runtime_error("unexpected complete draft teacher boundary");
  f.inputs.erase(f.inputs.begin() + teacher);
  f.state_ports.clear();
  if(!encoded) f.state_ports = {{kCache0, kNextCache0}, {kCache1, kNextCache1}, {kCache2, kNextCache2}};
  f.exact_outputs.clear();
  for (auto &replay : f.replays) {
    replay.inputs.erase(replay.inputs.begin() + teacher);
    replay.outputs.resize(encoded?3:kIds);
  }
}

inline std::vector<std::byte> parameter_bytes(std::span<const std::uint16_t> source,
                                             std::int64_t n, std::int64_t k, int seed) {
  if (!source.empty()) {
    if (source.size() != std::size_t(n * k))
      throw std::runtime_error("DSpark injected BF16 role extent mismatch");
    return sem::bytes(source);
  }
  std::vector<std::byte> result(std::size_t(n * k) * sizeof(std::uint16_t));
  for (std::int64_t row = 0; row < n; ++row) {
    const auto column = (17 * row + seed) % k;
    const auto value = encode(0.5F + float((row + seed) % 13) / 32.0F);
    std::memcpy(result.data() + (row * k + column) * sizeof(value), &value, sizeof(value));
  }
  return result;
}
inline ValueId cold_f32(CompleteProgram &program, std::int64_t n, std::int64_t k,
                        std::vector<std::byte> source) {
  auto &f = program.fragment;
  if (f.preparation && f.preparation.use_count() != 1)
    throw std::runtime_error("complete DSpark cold bank is not uniquely owned");
  auto cold = f.preparation ? std::const_pointer_cast<sem::Fragment>(f.preparation)
                           : std::make_shared<sem::Fragment>(sem::Fragment{.profile=f.profile,.region=0x0c04010b});
  if (cold->replays.empty()) cold->replays.push_back({{}, {}});
  const auto stored = cold->constant(sem::dense({n,k}, c::DataType::kBFloat16), std::move(source));
  const auto output = cold->one(c::Operation(c::Cast{c::DataType::kFloat32}), {stored});
  program.cold_widenings.push_back({cold->outputs.size(), stored});
  cold->outputs.push_back(output);
  // These large outputs use the chunked source-byte checker instead of
  // materializing a second dense F32 expected payload in Replay.
  cold->replays[0].outputs.emplace_back();
  f.preparation = std::move(cold);
  f.prepared_constants.push_back(f.constants.size());
  return f.constant(sem::dense({n,k}), {});
}

inline CompleteProgram build_flash(HeadParameters parameters = {},
    const base::DraftHistoryInputs *histories=nullptr,
    const SharedParameters *shared=nullptr,
    const base::DraftParameters *draft_parameters=nullptr) {
  if(shared && (!shared->embedding_table || !shared->logits))
    throw std::runtime_error("DSpark needs both actual shared vocabulary suppliers");
  auto original = base::complete_draft(base::models[1],histories,
      shared?&shared->embedding_table:nullptr,draft_parameters);
  CompleteProgram result{std::move(original.fragment), {}};
  result.observations=std::move(original.observations);
  auto &f = result.fragment;
  std::vector<ValueId> observation_roots;
  if(result.observations)result.observations->visit_values(
      [&](ValueId &value){observation_roots.push_back(value);});
  remove_teacher_branch(f,histories!=nullptr,result.observations && result.observations->residency?12u:0u,
      observation_roots);
  const auto hidden_value=f.outputs[histories?1:kHidden];
  const auto normalized_value=f.outputs[histories?2:kNormalized];
  f.region = 0x0c04010a;
  const auto table = parameters.readonly_parameter?parameters.readonly_parameter(f,"markov_w1.weight"):
      f.constant(sem::dense({vocabulary,rank}, c::DataType::kBFloat16),
          parameter_bytes(parameters.markov_w1_bf16, vocabulary, rank, 419));
  ValueId lm;
  if(!shared) lm=base::ds::permute(f, cold_f32(result, vocabulary, hidden,
      parameter_bytes({}, vocabulary, hidden, 491)), {1,0});
  const auto mw = parameters.readonly_parameter?parameters.readonly_parameter(f,"markov_w2.weight"):
      base::ds::permute(f, cold_f32(result, vocabulary, rank,
          parameter_bytes(parameters.markov_w2_bf16, vocabulary, rank, 431)), {1,0});
  const auto cw = parameters.readonly_parameter?parameters.readonly_parameter(f,"conf_proj.weight"):
      base::ds::permute(f, cold_f32(result, 1, hidden+rank,
          parameter_bytes(parameters.confidence_bf16, 1, hidden+rank, 409)), {1,0});
  if(parameters.readonly_parameter && (f.description(table)!=sem::dense({vocabulary,rank},c::DataType::kBFloat16)||
      f.description(mw)!=sem::dense({vocabulary,rank},c::DataType::kBFloat16)||
      f.description(cw)!=sem::dense({1,hidden+rank},c::DataType::kBFloat16)))
    throw std::runtime_error("original Markov/confidence BF16 bank geometry differs");
  const auto base_logits = shared?shared->logits(f,normalized_value):
      f.one(c::Operation(c::MatMul{}), {normalized_value,lm});
  if(shared && (base_logits.ordinal()>=f.descriptions.size() ||
      f.description(base_logits)!=sem::dense({rows,vocabulary})))
    throw std::runtime_error("DSpark shared LM must produce the complete F32 vocabulary");
  std::array<ValueId,rows+1> ids;
  std::array<ValueId,rows> embeddings, logits, confidence;
  ids[0] = f.inputs[histories?0:kFirst];
  for (std::size_t row = 0; row < std::size_t(rows); ++row) {
    // This index is the real preceding TopK output within this same Graph.
    embeddings[row] = f.one(c::Operation(c::Gather{}), {table,ids[row]});
    const auto markov = f.one(c::Operation(c::Cast{c::DataType::kFloat32}), {embeddings[row]});
    const auto bias = parameters.readonly_parameter?dspark_stored_projection_fragment::linear(f,markov,mw):
        f.one(c::Operation(c::MatMul{}), {markov,mw});
    logits[row] = base::core::binary(f,c::BinaryKind::kAdd,
        base::ds::slice(f,base_logits,0,std::int64_t(row),1),bias);
    auto top = f.append(c::Operation(c::TopK{.count=1,.stable=true}), {logits[row]}, 2);
    ids[row+1] = base::core::reshape(f,top[1],{1});
    // Source confidence is raw F32; sigmoid is a separate sidecar policy.
    const auto joined = base::ds::concat(f,
        base::ds::slice(f,hidden_value,0,std::int64_t(row),1),markov,1);
    confidence[row] = parameters.readonly_parameter?dspark_stored_projection_fragment::linear(f,joined,cw):
        f.one(c::Operation(c::MatMul{}), {joined,cw});
  }
  auto concatenate = [&](const auto &values) {
    auto output = values[0];
    for (std::size_t i=1;i<values.size();++i) output=base::ds::concat(f,output,values[i],0);
    return output;
  };
  f.outputs.push_back(concatenate(ids));
  f.outputs.push_back(concatenate(embeddings));
  f.outputs.push_back(concatenate(logits));
  f.outputs.push_back(concatenate(confidence));
  if(histories) {
    result.encoded=CompleteProgram::EncodedPorts{f.inputs[0],f.inputs[1],f.inputs[2],
        {f.inputs[3],f.inputs[4],f.inputs[5]},f.outputs[0],hidden_value,normalized_value,
        f.outputs[3],f.outputs[4],f.outputs[5],f.outputs[6]};
  }
  // The original generated fixture publishes its updated main rings. The
  // prepared encoded-history component only reads them; proposal KV remains
  // episode-local and is never exposed as a persistent next-cache output.
  return result;
}
} // namespace dspark_complete_graph
