#pragma once

#include <hyle/compute/leased_operation.h>
#include <hyle/compute/state_versions.h>
#include <hyle/graph/cpu.h>
#include <hyle/graph/executable.h>

#include "../compute_models/d5_deepseek_official_attention.h"

#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>

namespace deepseek_sparse_prefix_fixture {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace ds = d5_deepseek_official_attention;
namespace sem = ds::sem;
using hyle::Status;
using sem::ValueId;
template <class T> using Result = hyle::Result<T>;

// This consumer starts at the projected attention boundary. Storage is F32,
// with the original explicit BF16/FP8/F4 numerical boundaries; it does not
// claim packed BF16 storage, a paged bridge, or the rest of a decoder block.
constexpr int kHeads = 64, kDepth = 512, kIndexHeads = 64, kIndexDepth = 128;
constexpr int kTop = 512, kTokens = 9, kStates = 11;
enum State : int { kPastKv, kCurrentKv, kPastScore, kCurrentScore,
                  kPastIndex, kCurrentIndex, kPastIndexScore, kCurrentIndexScore,
                  kCache, kIndex, kRaw };
// fill is the current projection-window occupancy: four rows for ratio4,
// sixty-four rows (a half-block) for ratio128. It is not position % ratio128.
struct Cursor { std::uint32_t position, visible, fill; };
struct Case {
  int ratio, chunk, channels, capacity, initial_visible, base_position;
  bool indexed() const { return ratio == 4; }
  std::vector<int> states() const {
    return indexed() ? std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
                     : std::vector<int>{kPastKv, kCurrentKv, kPastScore, kCurrentScore, kCache, kRaw};
  }
  std::vector<int> tokens() const {
    return indexed() ? std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}
                     : std::vector<int>{0, 1, 4, 5, 8};
  }
};
constexpr Case kIndexed{4, 4, 1024, 516, 512, 2048};
constexpr Case kWide{128, 64, 512, 20, 17, 17 * 128 + 126};

inline Status require(bool value, const char *message) {
  return value ? Status::Ok() : Status::Internal(message);
}
inline c::TensorDesc dense(std::initializer_list<std::int64_t> dims,
                    c::DataType type = c::DataType::kFloat32) {
  return c::TensorDesc::dense(type, {dims.begin(), dims.size()}).value();
}
inline std::array<c::TensorDesc, kStates> state_descriptions(const Case &mode) {
  return {dense({mode.chunk, mode.channels}), dense({mode.chunk, mode.channels}),
          dense({mode.chunk, mode.channels}), dense({mode.chunk, mode.channels}),
          dense({4, 256}), dense({4, 256}), dense({4, 256}), dense({4, 256}),
          dense({mode.capacity, kDepth}), dense({mode.capacity, kIndexDepth}), dense({127, kDepth})};
}
inline std::array<c::TensorDesc, 9> token_descriptions(const Case &mode, int rows) {
  return {dense({rows, mode.channels}), dense({rows, mode.channels}), dense({rows, 256}), dense({rows, 256}),
          dense({rows, kDepth}), dense({1, rows, kHeads, kDepth}),
          dense({rows, kIndexHeads, kIndexDepth}), dense({rows, kIndexHeads}),
          dense({rows}, c::DataType::kInt32)};
}
inline ValueId copy(sem::Fragment &f, ValueId value) {
  return f.one(c::Operation(c::Copy{}), {value});
}
inline ValueId write_row(sem::Fragment &f, ValueId window, ValueId row, int index, int rows) {
  auto result = row;
  if (index) result = ds::concat(f, ds::slice(f, window, 0, 0, index), result, 0);
  if (index != rows - 1) result = ds::concat(f, result, ds::slice(f, window, 0, index + 1, rows - index - 1), 0);
  return result;
}
inline ValueId replace_cache(sem::Fragment &f, ValueId cache, ValueId row, int index, int capacity) {
  std::vector<std::uint8_t> writing(capacity);
  writing[index] = 1;
  const auto mask = f.constant(dense({capacity, 1}, c::DataType::kBool), sem::bytes(writing));
  return f.one(c::Operation(c::Select{}), {mask, row, cache});
}

inline sem::Fragment make_fragment(const Case &mode, int count, Cursor cursor) {
  sem::Fragment f{.profile = nullptr, .region = 0x53505245464958};
  const auto descriptions = state_descriptions(mode);
  std::array<ValueId, kStates> state;
  for (const auto role : mode.states()) state[role] = f.input(descriptions[role]);
  const auto token_desc = token_descriptions(mode, count);
  std::array<ValueId, 9> tokens;
  for (const auto role : mode.tokens()) tokens[role] = f.input(token_desc[role]);
  std::array<ValueId, 2> norm, ape;
  for (int role = 0; role < (mode.indexed() ? 2 : 1); ++role) {
    const int depth = role ? kIndexDepth : kDepth, channels = role ? 256 : mode.channels;
    std::vector<float> scales(depth), positions(mode.ratio * channels);
    for (int d = 0; d < depth; ++d) scales[d] = .75F + float(d % 7) / 16;
    for (int i = 0; i < mode.ratio * channels; ++i) positions[i] = float((i * 7 + 3) % 23 - 11) / 16;
    norm[role] = f.constant(dense({depth}), sem::bytes(scales));
    ape[role] = f.constant(dense({mode.ratio, channels}), sem::bytes(positions));
  }
  const auto empty_kv = ds::filled(f, dense({mode.chunk, mode.channels}), 0);
  const auto empty_score = ds::filled(f, dense({mode.chunk, mode.channels}), -INFINITY);
  ValueId empty_index, empty_index_score;
  if (mode.indexed()) {
    empty_index = ds::filled(f, dense({4, 256}), 0);
    empty_index_score = ds::filled(f, dense({4, 256}), -INFINITY);
  }
  std::vector<float> sinks(kHeads);
  for (int h = 0; h < kHeads; ++h) sinks[h] = .75F + float(h % 11) / 16;
  const auto sink = f.constant(dense({kHeads}), sem::bytes(sinks));
  std::vector<ValueId> ids_rows, attended_rows;
  for (int t = 0; t < count; ++t) {
    const auto main = ds::slice(f, tokens[0], 0, t, 1);
    const auto ape_row = f.constant(dense({1}, c::DataType::kInt32),
        sem::bytes(std::vector<std::int32_t>{std::int32_t(cursor.position % mode.ratio)}));
    const auto main_score = ds::binary(f, c::BinaryKind::kAdd,
        ds::slice(f, tokens[1], 0, t, 1), f.one(c::Operation(c::Gather{}), {ape[0], ape_row}));
    state[kCurrentKv] = write_row(f, state[kCurrentKv], main, cursor.fill, mode.chunk);
    state[kCurrentScore] = write_row(f, state[kCurrentScore], main_score, cursor.fill, mode.chunk);
    if (mode.indexed()) {
      const auto index = ds::slice(f, tokens[2], 0, t, 1);
      const auto index_score = ds::binary(f, c::BinaryKind::kAdd,
          ds::slice(f, tokens[3], 0, t, 1), f.one(c::Operation(c::Gather{}), {ape[1], ape_row}));
      state[kCurrentIndex] = write_row(f, state[kCurrentIndex], index, cursor.fill, mode.chunk);
      state[kCurrentIndexScore] = write_row(f, state[kCurrentIndexScore], index_score, cursor.fill, mode.chunk);
    }
    if (cursor.fill == unsigned(mode.chunk - 1)) {
      if ((cursor.position + 1) % mode.ratio == 0) {
        const auto position = ds::slice(f, tokens[8], 0, t, 1);
        const auto compressed = ds::compress(f, state[kPastKv], state[kPastScore],
            state[kCurrentKv], state[kCurrentScore], norm[0], position, mode.ratio, kDepth, false);
        state[kCache] = replace_cache(f, state[kCache], compressed, cursor.visible, mode.capacity);
        if (mode.indexed()) {
          const auto compressed_index = ds::compress(f, state[kPastIndex], state[kPastIndexScore],
              state[kCurrentIndex], state[kCurrentIndexScore], norm[1], position, mode.ratio, kIndexDepth, true);
          state[kIndex] = replace_cache(f, state[kIndex], compressed_index, cursor.visible, mode.capacity);
        }
        ++cursor.visible;
      }
      state[kPastKv] = state[kCurrentKv]; state[kPastScore] = state[kCurrentScore];
      state[kCurrentKv] = empty_kv; state[kCurrentScore] = empty_score;
      if (mode.indexed()) {
        state[kPastIndex] = state[kCurrentIndex]; state[kPastIndexScore] = state[kCurrentIndexScore];
        state[kCurrentIndex] = empty_index; state[kCurrentIndexScore] = empty_index_score;
      }
    }
    const auto raw = ds::concat(f, state[kRaw], ds::slice(f, tokens[4], 0, t, 1), 0);
    state[kRaw] = ds::slice(f, raw, 0, 1, 127);
    std::vector<float> visible(mode.capacity, -INFINITY);
    std::fill_n(visible.begin(), cursor.visible, 0.F);
    const auto visibility = f.constant(dense({1, mode.capacity}), sem::bytes(visible));
    auto compressed = state[kCache], compressed_mask = visibility;
    if (mode.indexed()) {
      auto iq = ds::reshape(f, ds::slice(f, tokens[6], 0, t, 1), {64, 128});
      const auto weight = ds::reshape(f, ds::slice(f, tokens[7], 0, t, 1), {64, 1});
      const auto dot = ds::bfloat_boundary(f, f.one(c::Operation(c::MatMul{}),
          {iq, ds::permute(f, state[kIndex], {1, 0})}));
      const auto positive = f.one(c::model_ops::clamp(0, std::numeric_limits<float>::max()), {dot});
      const auto weighted = ds::bfloat_boundary(f, ds::binary(f, c::BinaryKind::kMultiply, positive, weight));
      const auto scores = ds::bfloat_boundary(f, ds::reshape(f, ds::moe::sum(f, weighted, 0), {1, mode.capacity}));
      const auto ids = f.append(c::Operation(c::TopK{.count = kTop}),
          {ds::binary(f, c::BinaryKind::kAdd, scores, visibility)}, 2)[1];
      compressed = ds::reshape(f, f.one(c::Operation(c::Gather{}), {state[kCache], ids}), {kTop, kDepth});
      compressed_mask = ds::reshape(f, f.one(c::Operation(c::Gather{}),
          {ds::reshape(f, visibility, {mode.capacity, 1}), ids}), {1, kTop});
      ids_rows.push_back(copy(f, ids));
    }
    const int key_count = 128 + (mode.indexed() ? kTop : mode.capacity);
    const auto keys = ds::reshape(f, ds::concat(f, raw, compressed, 0), {1, key_count, 1, kDepth});
    const auto mask = ds::concat(f, ds::filled(f, dense({1, 128}), 0), compressed_mask, 1);
    const auto query = ds::slice(f, tokens[5], 1, t, 1);
    const auto attention = ds::bfloat_boundary(f, f.one(c::model_ops::sink_attention(1.F / std::sqrt(512.F)),
        {query, keys, keys, mask, sink}));
    for (const auto role : mode.states()) f.outputs.push_back(copy(f, state[role]));
    attended_rows.push_back(copy(f, attention));
    ++cursor.position; cursor.fill = (cursor.fill + 1) % mode.chunk;
  }
  auto attention = attended_rows[0];
  ValueId ids;
  if (mode.indexed()) ids = ids_rows[0];
  for (int t = 1; t < count; ++t) {
    if (mode.indexed()) ids = ds::concat(f, ids, ids_rows[t], 0);
    attention = ds::concat(f, attention, attended_rows[t], 1);
  }
  if (mode.indexed()) f.outputs.push_back(copy(f, ids));
  f.outputs.push_back(copy(f, attention));
  return f;
}

struct GraphProgram {
  g::CompiledGraph compiled;
  std::vector<g::ValueId> inputs, outputs;
  std::vector<c::TensorDesc> input_desc, output_desc;
  std::vector<c::ReadLease> constants;
};
struct Route {
  c::Context context;
  c::Queue queue;
  c::LeasePool pool;
  Result<c::ReadLease> load(std::span<const std::byte> data) {
    HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(data.size()));
    HYLE_RETURN_IF_ERROR(writer.view()->write(data)); return writer.publish();
  }
  template <class T> Result<std::vector<T>> read(const c::ReadLease &value) {
    std::vector<T> result(value.size() / sizeof(T));
    HYLE_RETURN_IF_ERROR(value.view()->read(std::as_writable_bytes(std::span(result)))); return result;
  }
  Result<GraphProgram> prepare(const sem::Fragment &f) {
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    GraphProgram result; std::vector<g::ValueId> values(f.descriptions.size());
    for (const auto id : f.inputs) {
      HYLE_ASSIGN_OR_RETURN(values[id.ordinal()], builder.input(f.description(id)));
      result.inputs.push_back(values[id.ordinal()]); result.input_desc.push_back(f.description(id));
    }
    // Constants are ordinary immutable leased inputs, not private provider data.
    for (std::size_t i = 0; i < f.constants.size(); ++i) {
      const auto id = f.constants[i];
      HYLE_ASSIGN_OR_RETURN(values[id.ordinal()], builder.input(f.description(id)));
      result.inputs.push_back(values[id.ordinal()]); result.input_desc.push_back(f.description(id));
      HYLE_ASSIGN_OR_RETURN(auto value, load(f.constant_bytes[i])); result.constants.push_back(std::move(value));
    }
    for (const auto &node : f.nodes) {
      std::vector<g::ValueId> inputs;
      for (const auto id : node.inputs) inputs.push_back(values[id.ordinal()]);
      HYLE_ASSIGN_OR_RETURN(auto added, builder.add_node(node.operation, inputs));
      for (std::size_t i = 0; i < node.outputs.size(); ++i) {
        HYLE_ASSIGN_OR_RETURN(values[node.outputs[i].ordinal()], builder.result(added, i));
      }
    }
    for (const auto id : f.outputs) {
      HYLE_RETURN_IF_ERROR(builder.publish(values[id.ordinal()]));
      result.outputs.push_back(values[id.ordinal()]); result.output_desc.push_back(f.description(id));
    }
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(result.compiled, g::compile(graph, context)); return result;
  }
  Result<std::vector<c::ReadLease>> execute(GraphProgram &program, std::span<const c::ReadLease> data) {
    HYLE_ASSIGN_OR_RETURN(auto slot, program.compiled.executable.create_invocation({.resources = g::ResourceOptions{}}));
    std::vector<c::ReadLease> inputs(data.begin(), data.end());
    inputs.insert(inputs.end(), program.constants.begin(), program.constants.end());
    HYLE_RETURN_IF_ERROR(require(inputs.size() == program.inputs.size(), "sparse consumer input count"));
    for (std::size_t i = 0; i < inputs.size(); ++i)
      HYLE_RETURN_IF_ERROR(slot.bind(program.inputs[i], program.input_desc[i], inputs[i]));
    for (std::size_t i = 0; i < program.outputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto bytes, program.output_desc[i].required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto writer, pool.allocate(bytes));
      HYLE_RETURN_IF_ERROR(slot.bind(program.outputs[i], program.output_desc[i], std::move(writer)));
    }
    HYLE_ASSIGN_OR_RETURN(auto done, slot.submit(queue)); HYLE_RETURN_IF_ERROR(done.wait());
    std::vector<c::ReadLease> outputs;
    for (const auto id : program.outputs) {
      HYLE_ASSIGN_OR_RETURN(auto value, slot.result(id));
      HYLE_ASSIGN_OR_RETURN(auto lease, value.storage_lease()); outputs.push_back(std::move(lease));
    }
    return outputs;
  }
};

inline Cursor cursor(const Case &mode, int count) {
  const auto position = unsigned(mode.base_position + count);
  return {position, unsigned(mode.initial_visible) + position / mode.ratio - unsigned(mode.base_position / mode.ratio),
          position % mode.chunk};
}
inline double state_max_error = 0, oracle_max_error = 0;
inline Status close(std::span<const float> a, std::span<const float> b, const char *label,
             float absolute = 8e-5F, float relative = 4e-5F) {
  HYLE_RETURN_IF_ERROR(require(a.size() == b.size(), "sparse consumer comparison extent"));
  double maximum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i]) continue;
    maximum = std::max(maximum, double(std::abs(a[i] - b[i])));
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
        std::abs(a[i] - b[i]) > absolute + relative * std::abs(b[i])) {
      std::fprintf(stderr, "%s index=%zu actual=%.9g expected=%.9g\n", label, i, a[i], b[i]);
      return Status::Internal("sparse prefix state or output differs");
    }
  }
  auto &observed = relative == ds::sem::q::kExpertRelativeTolerance ? oracle_max_error : state_max_error;
  observed = std::max(observed, maximum);
  return Status::Ok();
}

// Explicit, untrained token->projected-port fixture. This closes the actual
// proposal IDs -> target inputs dependency without pretending to implement
// the checkpoint embedding/43 target layers. It has one I32[count] input and
// returns ports in Case::tokens() order. IDs from the full Flash head are
// exactly representable in F32; no host-selected replacement token is used.
// Every floating port is BF16-rounded after the finite affine projection.
inline sem::Fragment token_features(const Case &mode, int count, int first_position) {
  if (count <= 0 || first_position < 0 || mode.ratio <= 0 ||
      first_position > std::numeric_limits<std::int32_t>::max() - count)
    throw std::invalid_argument("invalid projected token interval");
  sem::Fragment f{.profile = nullptr, .region = 0x445350544f4b454e};
  const auto ids = f.input(dense({count}, c::DataType::kInt32));
  const auto real_ids = f.one(c::Operation(c::Cast{.output_type = c::DataType::kFloat32}), {ids});
  const auto scaled = ds::binary(f, c::BinaryKind::kMultiply,
      ds::reshape(f, real_ids, {count, 1}), ds::filled(f, dense({1}), 0x1p-18F));
  const auto descriptions = token_descriptions(mode, count);
  for (const auto role : mode.tokens()) {
    if (role == 8) {
      std::vector<std::int32_t> positions(std::size_t(count), 0);
      for (int t = 0; t < count; ++t)
        positions[t] = (first_position + t) / mode.ratio * mode.ratio;
      f.outputs.push_back(copy(f, f.constant(dense({count}, c::DataType::kInt32), sem::bytes(positions))));
      continue;
    }
    const auto width = *descriptions[role].element_count() / std::size_t(count);
    std::vector<float> coefficients(width), bias(width);
    constexpr std::array<float, 4> signs{-1.F, -.5F, .5F, 1.F};
    for (std::size_t col = 0; col < width; ++col) {
      coefficients[col] = signs[(col + std::size_t(role)) % signs.size()];
      bias[col] = float(int((col * 5 + std::size_t(role) * 11) % 31) - 15) / 32;
    }
    const auto weight = f.constant(dense({1, std::int64_t(width)}), sem::bytes(coefficients));
    const auto offset = f.constant(dense({1, std::int64_t(width)}), sem::bytes(bias));
    auto value = ds::bfloat_boundary(f, ds::binary(f, c::BinaryKind::kAdd,
        ds::binary(f, c::BinaryKind::kMultiply, scaled, weight), offset));
    if (role == 5) value = ds::reshape(f, value, {1, count, kHeads, kDepth});
    if (role == 6) value = ds::reshape(f, value, {count, kIndexHeads, kIndexDepth});
    f.outputs.push_back(copy(f, value));
  }
  return f;
}

// Independent scalar F32/BF16 reference for the finite adapter above. For the
// actual full-head ID domain the dyadic intermediates are exact in F32.
inline std::vector<std::vector<std::byte>> token_feature_reference(const Case &mode,
    std::span<const std::int32_t> ids, int first_position) {
  if (ids.empty() || ids.size() > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
      first_position < 0 || mode.ratio <= 0 ||
      ids.size() > std::size_t(std::numeric_limits<std::int32_t>::max() - first_position))
    throw std::invalid_argument("invalid projected token reference interval");
  for (const auto id : ids)
    if (id < 0 || id > (1 << 24)) throw std::invalid_argument("projected token ID is not F32 exact");
  const auto count = int(ids.size());
  const auto descriptions = token_descriptions(mode, count);
  std::vector<std::vector<std::byte>> result;
  for (const auto role : mode.tokens()) {
    if (role == 8) {
      std::vector<std::int32_t> positions(ids.size());
      for (int t = 0; t < count; ++t)
        positions[t] = (first_position + t) / mode.ratio * mode.ratio;
      result.push_back(sem::bytes(positions));
      continue;
    }
    const auto width = *descriptions[role].element_count() / ids.size();
    std::vector<float> values(ids.size() * width);
    constexpr std::array<float, 4> signs{-1.F, -.5F, .5F, 1.F};
    for (std::size_t t = 0; t < ids.size(); ++t)
      for (std::size_t col = 0; col < width; ++col) {
        const float scaled = float(ids[t]) * 0x1p-18F;
        const float coefficient = signs[(col + std::size_t(role)) % signs.size()];
        const float offset = float(int((col * 5 + std::size_t(role) * 11) % 31) - 15) / 32;
        const float value = scaled * coefficient + offset;
        auto bits = std::bit_cast<std::uint32_t>(value);
        bits += 0x7fffu + ((bits >> 16u) & 1u);
        values[t * width + col] = std::bit_cast<float>(bits & 0xffff0000u);
      }
    result.push_back(sem::bytes(values));
  }
  return result;
}

struct PreparedInputs final {
  std::array<std::vector<float>, kStates> initial;
  std::vector<c::ReadLease> base_values;
  std::array<std::vector<float>, 8> features;
  std::array<c::ReadLease, 9> tokens;
  Result<std::vector<c::ReadLease>> inputs(const Case &mode,
      std::span<const c::ReadLease> state, int first, int count) const {
    HYLE_RETURN_IF_ERROR(require(first >= 0 && count > 0 && first + count <= kTokens,
                                "projected target token interval"));
    std::vector<c::ReadLease> result(state.begin(), state.end());
    for (const auto role : mode.tokens()) {
      const auto row_bytes = tokens[role].size() / kTokens;
      HYLE_ASSIGN_OR_RETURN(auto selected, tokens[role].subview(
          std::size_t(first) * row_bytes, std::size_t(count) * row_bytes));
      result.push_back(std::move(selected));
    }
    return result;
  }
};
inline Result<PreparedInputs> prepare_inputs(Route &route, const Case &mode) {
  PreparedInputs result;
  const auto state_roles = mode.states(), token_roles = mode.tokens();
  const auto descriptions = state_descriptions(mode);
  for (const auto s : state_roles) {
    result.initial[s].resize(*descriptions[s].element_count());
    for (std::size_t i = 0; i < result.initial[s].size(); ++i)
      result.initial[s][i] = ds::fp8::bfloat(float(int((i * 7 + s * 3) % 29) - 14) / 32);
    if (s == kPastIndex) std::fill(result.initial[s].begin(), result.initial[s].end(), -1.F);
    if (s == kCurrentKv || s == kCurrentIndex) {
      const auto initialized = std::size_t(cursor(mode, 0).fill) * (s == kCurrentKv ? mode.channels : 256);
      std::fill(result.initial[s].begin() + initialized, result.initial[s].end(), 0.F);
    }
    if (s == kCurrentScore || s == kCurrentIndexScore) {
      const auto initialized = std::size_t(cursor(mode, 0).fill) * (s == kCurrentScore ? mode.channels : 256);
      std::fill(result.initial[s].begin() + initialized, result.initial[s].end(), -INFINITY);
    }
    if (s == kIndex) {
      std::fill(result.initial[s].begin(), result.initial[s].end(), 0.F);
      for (int r = 0; r < kTop; ++r) result.initial[s][r * kIndexDepth] = float(r + 1) / 64;
      result.initial[s] = ds::quant::reference(ds::bfloat_boundary(std::move(result.initial[s])), true);
    }
    if (s == kCache || s == kRaw) result.initial[s] = ds::cache_reference(std::move(result.initial[s]), kDepth, false);
    HYLE_ASSIGN_OR_RETURN(auto value, route.load(std::as_bytes(std::span(result.initial[s])))); result.base_values.push_back(std::move(value));
  }
  const auto token_desc = token_descriptions(mode, kTokens);
  for (const auto p : token_roles) {
    if (p == 8) continue;
    result.features[p].resize(*token_desc[p].element_count());
    for (std::size_t i = 0; i < result.features[p].size(); ++i)
      result.features[p][i] = ds::fp8::bfloat(float(int((i * 5 + p * 11) % 31) - 15) / 32);
  }
  if (mode.indexed()) {
    std::fill(result.features[2].begin(), result.features[2].end(), -1.F);
    std::fill(result.features[6].begin(), result.features[6].end(), 0.F);
    std::fill(result.features[7].begin(), result.features[7].end(), 1.F);
    for (int t = 0; t < kTokens; ++t)
      for (int h = 0; h < kIndexHeads; ++h)
        result.features[6][(t * kIndexHeads + h) * kIndexDepth] = t % 2 ? -1.F : 1.F;
  }
  for (const auto p : token_roles) {
    if (p == 8) continue;
    HYLE_ASSIGN_OR_RETURN(result.tokens[p], route.load(std::as_bytes(std::span(result.features[p]))));
  }
  std::vector<std::int32_t> positions(kTokens);
  for (int t = 0; t < kTokens; ++t) positions[t] = (mode.base_position + t) / mode.ratio * mode.ratio;
  HYLE_ASSIGN_OR_RETURN(result.tokens[8], route.load(std::as_bytes(std::span(positions))));
  return result;
}

inline Status run(const Case &mode) {
  state_max_error = oracle_max_error = 0;
  const auto state_roles = mode.states(), token_roles = mode.tokens();
  const int states = int(state_roles.size());
  const auto ordinal = [&](int role) { return int(std::find(state_roles.begin(), state_roles.end(), role) - state_roles.begin()); };
  HYLE_ASSIGN_OR_RETURN(auto context, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto pool, c::LeasePool::create(m::system_resource(), 256u << 20,
      {.max_blocks = 2048}));
  Route route{std::move(context), std::move(queue), std::move(pool)};
  const auto descriptions = state_descriptions(mode);
  HYLE_ASSIGN_OR_RETURN(auto prepared, prepare_inputs(route, mode));
  auto &initial = prepared.initial;
  auto &base_values = prepared.base_values;
  auto &features = prepared.features;
  const auto inputs = [&](const std::vector<c::ReadLease> &state, int first, int count) {
    return prepared.inputs(mode, state, first, count);
  };
  std::vector<c::StateEntrySpec> specs;
  for (const auto role : state_roles) specs.push_back({.description = descriptions[role]});
  specs.push_back({.kind = c::StateEntryKind::kHostMetadata, .metadata_bytes = sizeof(Cursor)});
  HYLE_ASSIGN_OR_RETURN(auto schema, c::StateSchema::create(specs));
  // StateStore charges the complete shared LeasePool root plus its separately
  // owned metadata snapshots, rather than just the visible resource slices.
  HYLE_ASSIGN_OR_RETURN(auto store, c::StateStore::create(schema,
      {.max_versions = 64, .max_drafts = 4, .max_retained_blocks = 1024,
       .max_backing_bytes = (256u << 20) + (64u << 10)}));
  std::array<c::StateEntryId, kStates + 1> ids;
  for (int i = 0; i <= states; ++i) { HYLE_ASSIGN_OR_RETURN(ids[i], schema.entry(i)); }
  std::vector<c::StateMember> members;
  for (int i = 0; i < states; ++i) members.push_back({ids[i], base_values[i]});
  const auto zero_cursor = cursor(mode, 0);
  members.push_back({ids[states], {}, std::as_bytes(std::span(&zero_cursor, 1))});
  HYLE_ASSIGN_OR_RETURN(auto base, store.import(members));
  const auto publish = [&](const c::StateVersion &from, std::span<const c::ReadLease> values, int accepted) -> Result<c::StateVersion> {
    HYLE_ASSIGN_OR_RETURN(auto draft, store.begin(from));
    for (int i = 0; i < states; ++i) HYLE_RETURN_IF_ERROR(draft.replace(ids[i], values[i]));
    const auto position = cursor(mode, accepted);
    HYLE_RETURN_IF_ERROR(draft.replace_metadata(ids[states], std::as_bytes(std::span(&position, 1)))); return draft.finalize();
  };
  const auto state = [&](const c::StateVersion &version) -> Result<std::vector<c::ReadLease>> {
    std::vector<c::ReadLease> values;
    for (int i = 0; i < states; ++i) { HYLE_ASSIGN_OR_RETURN(auto value, version.member(ids[i])); values.push_back(std::move(value)); }
    return values;
  };
  std::printf("CPU compression ratio%d: preparing Graph verify8 + 9 continuation graphs, states=%d tokenPorts=%zu basePosition=%d halfFill=%u\n",
              mode.ratio, states, token_roles.size(), mode.base_position, zero_cursor.fill); std::fflush(stdout);
  HYLE_ASSIGN_OR_RETURN(auto block, route.prepare(make_fragment(mode, 8, cursor(mode, 0))));
  std::vector<GraphProgram> single;
  for (int t = 0; t < kTokens; ++t) { HYLE_ASSIGN_OR_RETURN(auto program, route.prepare(make_fragment(mode, 1, cursor(mode, t)))); single.push_back(std::move(program)); }
  HYLE_ASSIGN_OR_RETURN(auto block_inputs, inputs(base_values, 0, 8));
  HYLE_ASSIGN_OR_RETURN(auto verified, route.execute(block, block_inputs));
  std::vector<c::StateVersion> prefixes{base};
  for (int t = 0; t < 8; ++t) {
    HYLE_ASSIGN_OR_RETURN(auto value, publish(base, std::span(verified).subspan(t * states, states), t + 1)); prefixes.push_back(std::move(value));
  }
  if (!mode.indexed()) {
    // Independent token accounting checks absolute APE, the 64-row roll, raw
    // history and the one 128-boundary publication, rather than only comparing
    // two graphs built from the same helper.
    auto expected = initial;
    std::vector<float> completed_cache;
    for (int t = 0; t < 8; ++t) {
      const auto row = cursor(mode, t).fill;
      std::copy_n(features[0].begin() + t * mode.channels, mode.channels,
                  expected[kCurrentKv].begin() + row * mode.channels);
      for (int d = 0; d < mode.channels; ++d) {
        const auto index = ((mode.base_position + t) % mode.ratio) * mode.channels + d;
        const float ape = float((index * 7 + 3) % 23 - 11) / 16;
        expected[kCurrentScore][row * mode.channels + d] = features[1][t * mode.channels + d] + ape;
      }
      if (row == unsigned(mode.chunk - 1)) {
        expected[kPastKv] = expected[kCurrentKv]; expected[kPastScore] = expected[kCurrentScore];
        std::fill(expected[kCurrentKv].begin(), expected[kCurrentKv].end(), 0.F);
        std::fill(expected[kCurrentScore].begin(), expected[kCurrentScore].end(), -INFINITY);
      }
      std::rotate(expected[kRaw].begin(), expected[kRaw].begin() + kDepth, expected[kRaw].end());
      std::copy_n(features[4].begin() + t * kDepth, kDepth, expected[kRaw].end() - kDepth);
      for (const auto role : {kPastKv, kCurrentKv, kPastScore, kCurrentScore, kRaw}) {
        HYLE_ASSIGN_OR_RETURN(auto actual, route.read<float>(verified[t * states + ordinal(role)]));
        HYLE_RETURN_IF_ERROR(close(actual, expected[role], "ratio128 independent window/raw/APE"));
      }
      HYLE_ASSIGN_OR_RETURN(auto cache, route.read<float>(verified[t * states + ordinal(kCache)]));
      if (t == 1) {
        completed_cache = cache;
        HYLE_RETURN_IF_ERROR(require(!std::equal(cache.begin() + mode.initial_visible * kDepth,
            cache.begin() + (mode.initial_visible + 1) * kDepth,
            initial[kCache].begin() + mode.initial_visible * kDepth), "ratio128 completion did not update its cache row"));
      }
      HYLE_RETURN_IF_ERROR(close(cache, t == 0 ? initial[kCache] : completed_cache,
                                 "ratio128 exact publication boundary"));
      for (int r = 0; r < mode.capacity; ++r) if (r != mode.initial_visible) {
        HYLE_RETURN_IF_ERROR(close(std::span(cache).subspan(r * kDepth, kDepth),
            std::span(initial[kCache]).subspan(r * kDepth, kDepth), "ratio128 untouched cache row"));
      }
    }
  }
  std::vector<std::int32_t> block_ids;
  std::set<std::vector<std::int32_t>> distinct;
  if (mode.indexed()) {
    HYLE_ASSIGN_OR_RETURN(block_ids, route.read<std::int32_t>(verified[8 * states]));
    for (int t = 0; t < 8; ++t) {
      std::vector<std::int32_t> row(block_ids.begin() + t * kTop, block_ids.begin() + (t + 1) * kTop);
      std::sort(row.begin(), row.end());
      HYLE_RETURN_IF_ERROR(require(std::adjacent_find(row.begin(), row.end()) == row.end() && row.front() >= 0 && row.back() < int(cursor(mode, t + 1).visible), "TopK duplicate/future index"));
      distinct.insert(std::move(row));
    }
    HYLE_RETURN_IF_ERROR(require(distinct.size() > 1, "queries did not select distinct TopK sets"));
  }
  const int attention_port = 8 * states + (mode.indexed() ? 1 : 0);
  const int selected_rows = mode.indexed() ? kTop : mode.capacity;
  HYLE_ASSIGN_OR_RETURN(auto block_attention, route.read<float>(verified[attention_port]));
  std::vector<float> sink_values(kHeads);
  for (int h = 0; h < kHeads; ++h) sink_values[h] = .75F + float(h % 11) / 16;
  for (int t = 0; t < 8; ++t) {
    HYLE_ASSIGN_OR_RETURN(auto cache, route.read<float>(verified[t * states + ordinal(kCache)]));
    std::vector<float> raw(initial[kRaw]);
    raw.insert(raw.end(), features[4].begin(), features[4].begin() + (t + 1) * kDepth);
    std::vector<float> keys(raw.end() - 128 * kDepth, raw.end());
    std::vector<float> mask(128 + selected_rows);
    for (int k = 0; k < selected_rows; ++k) {
      const auto row = mode.indexed() ? block_ids[t * kTop + k] : k;
      keys.insert(keys.end(), cache.begin() + row * kDepth, cache.begin() + (row + 1) * kDepth);
      if (row >= int(cursor(mode, t + 1).visible)) mask[128 + k] = -INFINITY;
    }
    const auto query = std::span(features[5]).subspan(t * kHeads * kDepth, kHeads * kDepth);
    const auto expected = ds::bfloat_boundary(ds::old::sink_reference(query, keys, keys,
        mask, sink_values, 1, 128 + selected_rows, 1.F / std::sqrt(512.F)));
    HYLE_RETURN_IF_ERROR(close(std::span(block_attention).subspan(t * kHeads * kDepth,
        kHeads * kDepth), expected, "independent FP64 Sink",
        ds::sem::q::kExpertAbsoluteTolerance, ds::sem::q::kExpertRelativeTolerance));
  }
  if (mode.indexed()) std::printf("CPU sparse: verify8 PASS, distinct TopK sets=%zu\n", distinct.size());
  else std::printf("CPU ratio128: verify8 PASS, no indexer/TopK, visible17->18 at token2, halfFill62->0->6\n");
  std::fflush(stdout);
  for (int accepted = 0; accepted <= 8; ++accepted) {
    auto reference = base;
    std::vector<c::ReadLease> reference_output;
    for (int t = 0; t < accepted; ++t) {
      HYLE_ASSIGN_OR_RETURN(auto before, state(reference)); HYLE_ASSIGN_OR_RETURN(auto args, inputs(before, t, 1));
      HYLE_ASSIGN_OR_RETURN(auto output, route.execute(single[t], args));
      HYLE_ASSIGN_OR_RETURN(reference, publish(reference, output, t + 1));
      reference_output = std::move(output);
    }
    if (accepted) {
      if (mode.indexed()) {
        HYLE_ASSIGN_OR_RETURN(auto actual_ids, route.read<std::int32_t>(reference_output[states]));
        HYLE_RETURN_IF_ERROR(require(std::equal(actual_ids.begin(), actual_ids.end(),
            block_ids.begin() + (accepted - 1) * kTop), "block/reference per-query IDs differ"));
      }
      HYLE_ASSIGN_OR_RETURN(auto actual_attention, route.read<float>(reference_output[states + (mode.indexed() ? 1 : 0)]));
      HYLE_RETURN_IF_ERROR(close(actual_attention, std::span(block_attention).subspan(
          (accepted - 1) * kHeads * kDepth, kHeads * kDepth), "block/reference per-query Sink"));
    }
    for (int s = 0; s <= states; ++s) {
      HYLE_ASSIGN_OR_RETURN(auto a, prefixes[accepted].member(ids[s])); HYLE_ASSIGN_OR_RETURN(auto b, reference.member(ids[s]));
      if (s == states) {
        HYLE_ASSIGN_OR_RETURN(auto x, route.read<std::uint32_t>(a)); HYLE_ASSIGN_OR_RETURN(auto y, route.read<std::uint32_t>(b));
        HYLE_RETURN_IF_ERROR(require(x == y, "accepted cursor differs"));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto x, route.read<float>(a)); HYLE_ASSIGN_OR_RETURN(auto y, route.read<float>(b)); HYLE_RETURN_IF_ERROR(close(x, y, "accepted state"));
      }
    }
    std::array<std::vector<c::ReadLease>, 2> continued;
    int arm = 0;
    for (const auto &version : {prefixes[accepted], reference}) {
      HYLE_ASSIGN_OR_RETURN(auto before, state(version)); HYLE_ASSIGN_OR_RETURN(auto args, inputs(before, accepted, 1));
      HYLE_ASSIGN_OR_RETURN(continued[arm], route.execute(single[accepted], args)); ++arm;
    }
    HYLE_ASSIGN_OR_RETURN(auto committed, publish(prefixes[accepted], continued[0], accepted + 1));
    HYLE_ASSIGN_OR_RETURN(auto reference_committed, publish(reference, continued[1], accepted + 1));
    for (int s = 0; s <= states; ++s) {
      HYLE_ASSIGN_OR_RETURN(auto x, committed.member(ids[s]));
      HYLE_ASSIGN_OR_RETURN(auto y, reference_committed.member(ids[s]));
      HYLE_ASSIGN_OR_RETURN(auto a, route.read<std::byte>(x));
      HYLE_ASSIGN_OR_RETURN(auto b, route.read<std::byte>(y));
      HYLE_RETURN_IF_ERROR(require(a == b, "committed continued resource/metadata differs"));
    }
    for (int s = 0; s < states + (mode.indexed() ? 2 : 1); ++s) {
      if (mode.indexed() && s == states) {
        HYLE_ASSIGN_OR_RETURN(auto x, route.read<std::int32_t>(continued[0][s])); HYLE_ASSIGN_OR_RETURN(auto y, route.read<std::int32_t>(continued[1][s]));
        HYLE_RETURN_IF_ERROR(require(x == y, "continued TopK IDs differ"));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto x, route.read<float>(continued[0][s])); HYLE_ASSIGN_OR_RETURN(auto y, route.read<float>(continued[1][s])); HYLE_RETURN_IF_ERROR(close(x, y, "continued state/attention"));
      }
    }
    std::printf("CPU ratio%d accept=%d full state%s/Sink continuation PASS\n", mode.ratio, accepted,
                mode.indexed() ? "/IDs" : "/visible/raw/half-window"); std::fflush(stdout);
    HYLE_RETURN_IF_ERROR(store.collect());
  }
  for (int s = 0; s < states; ++s) {
    HYLE_ASSIGN_OR_RETURN(auto value, base.member(ids[s])); HYLE_ASSIGN_OR_RETURN(auto actual, route.read<float>(value)); HYLE_RETURN_IF_ERROR(close(actual, initial[state_roles[s]], "base preserved"));
  }
  std::printf("CPU sparse numeric: paired state/Sink maxAbs=%.9g; independent FP64 Sink+BF16 boundary maxAbs=%.9g\n",
              state_max_error, oracle_max_error);
  std::printf("CPU ratio%d compressor: H64 D512 %s, accept0..8 PASS; projected-boundary dense composition only\n",
              mode.ratio, mode.indexed() ? "index64x128 Top512 across two completions" : "single-channel 64+64 windows across one128 completion");
  return Status::Ok();
}
} // namespace deepseek_sparse_prefix_fixture
