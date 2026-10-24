#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "dspark_target_binding.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/state_versions.h"

// Model-only assembly of the selected Flash0731 Q8 sidecar main histories.
// The caller supplies a verified-target main_x lease and explicit positions;
// proposal-token KV is never an input. This contains no execution loop.
namespace dspark_main_history {
namespace b = dspark_target_binding;
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::Result;
using hyle::Status;
inline constexpr std::int64_t kWindow = 128, kStages = 3;
inline constexpr auto kEncoding = c::RowCacheEncoding::float8_block64_tail64();

struct Placement final {
  std::vector<std::int32_t> positions, slots;
};
inline Result<Placement> placement(const b::CaptureFrame &frame) {
  if (frame.rows < 1 || frame.rows > 2 || frame.epoch == 0 || frame.first_position < 0 ||
      frame.first_position > std::numeric_limits<std::int32_t>::max() - (frame.rows - 1))
    return std::unexpected(Status::InvalidArgument("DSpark history requires one or two qualified I32 positions"));
  Placement result;
  for (std::int64_t row = 0; row < frame.rows; ++row) {
    const auto position = std::int32_t(frame.first_position + row);
    result.positions.push_back(position); result.slots.push_back(position % kWindow);
  }
  return result;
}

struct History final {
  g::Graph graph;
  std::int64_t rows = 0;
  c::TensorDesc main_description, ring_description, index_description, carry_description;
  c::TensorDesc key_description, entry_description;
  g::ValueId main, positions, slots;
  std::array<g::ValueId, 3> rings;
  std::array<g::ValueId, 7> parameters;
  // Each prefix publishes three independently backed physical rings and the
  // last consumed main row. Prefix0 is the old version, not another output.
  std::vector<std::array<g::ValueId, 4>> prefixes;
  // Real arithmetic boundaries for the finite consumer's layered checks:
  // postnorm F32, post-RoPE F32, encoded-cache BF16 entry, in stage order.
  // They are published after the prefix tuple and never enter model state.
  std::array<std::array<g::ValueId, 3>, 3> checkpoints;
  std::array<std::optional<g::ValueId>,3> raw_projections;
  std::array<std::optional<std::size_t>,3> raw_projection_ports;
  std::array<b::StageKvRoles, 3> roles;
  b::ReadonlyRole frequencies;

  Result<std::array<g::ConstantBinding, 7>> constants() const {
    std::array<g::ConstantBinding, 7> result;
    for (std::size_t stage = 0; stage < roles.size(); ++stage) {
      HYLE_ASSIGN_OR_RETURN(result[2 * stage], b::constant(parameters[2 * stage], roles[stage].projection));
      HYLE_ASSIGN_OR_RETURN(result[2 * stage + 1], b::constant(parameters[2 * stage + 1], roles[stage].norm));
    }
    HYLE_ASSIGN_OR_RETURN(result[6], b::constant(parameters[6], frequencies));
    return result;
  }
  Result<c::ReadLease> qualify(const b::CaptureFrame &expected, const b::CaptureFrame &produced,
                               const c::ReadLease &actual_main) const {
    HYLE_ASSIGN_OR_RETURN(const auto selected, placement(expected));
    (void)selected;
    HYLE_ASSIGN_OR_RETURN(const auto bytes, main_description.required_bytes());
    if (expected != produced || expected.target != roles[0].target || expected.rows != rows ||
        actual_main.size() != bytes)
      return std::unexpected(Status::InvalidArgument("DSpark main result belongs to another target/frame/frontier"));
    HYLE_ASSIGN_OR_RETURN(const auto tensor, actual_main.tensor(main_description));
    (void)tensor;
    return actual_main;
  }
};

inline Result<History> make_history(std::array<b::StageKvRoles, 3> roles,
                                    b::ReadonlyRole frequencies, std::int64_t rows) {
  if (rows < 1 || rows > 2)
    return std::unexpected(Status::InvalidArgument("DSpark finite accepted-prefix history supports rows1/2"));
  HYLE_ASSIGN_OR_RETURN(const auto frequency_description, b::plain(std::array<std::int64_t, 1>{32}));
  HYLE_RETURN_IF_ERROR(b::qualify_role(frequencies, "dflash.rope.frequencies", frequency_description));
  for (std::size_t stage = 0; stage < roles.size(); ++stage)
    if (roles[stage].stage != std::int32_t(stage) || roles[stage].target != roles[0].target ||
        roles[stage].sidecar.repository != roles[0].sidecar.repository ||
        roles[stage].sidecar.revision != roles[0].sidecar.revision || roles[stage].sidecar.file != roles[0].sidecar.file)
      return std::unexpected(Status::InvalidArgument("DSpark requires the three distinct declared stages from one target/sidecar"));
  History result{.rows = rows, .roles = std::move(roles), .frequencies = std::move(frequencies)};
  HYLE_ASSIGN_OR_RETURN(result.main_description, b::plain(std::array{rows, b::kHidden}));
  HYLE_ASSIGN_OR_RETURN(result.ring_description, c::row_cache_carrier_description(kEncoding, kWindow));
  HYLE_ASSIGN_OR_RETURN(result.carry_description, b::plain(std::array<std::int64_t, 2>{1, b::kHidden}));
  HYLE_ASSIGN_OR_RETURN(result.key_description, b::plain(std::array{rows, b::kKv}));
  HYLE_ASSIGN_OR_RETURN(result.entry_description, c::TensorDesc::dense(c::DataType::kBFloat16, std::array{rows, b::kKv}));
  HYLE_ASSIGN_OR_RETURN(result.index_description, c::TensorDesc::dense(c::DataType::kInt32, std::array{rows}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.main, builder.input(result.main_description));
  HYLE_ASSIGN_OR_RETURN(result.positions, builder.input(result.index_description));
  HYLE_ASSIGN_OR_RETURN(result.slots, builder.input(result.index_description));
  for (auto &ring : result.rings) {
    HYLE_ASSIGN_OR_RETURN(ring, builder.input(result.ring_description));
  }
  HYLE_ASSIGN_OR_RETURN(result.parameters[6], builder.constant(frequency_description));
  std::array<g::ValueId, 3> keys;
  for (std::size_t stage = 0; stage < result.roles.size(); ++stage) {
    HYLE_ASSIGN_OR_RETURN(auto part, b::append_main_kv(builder, result.roles[stage], rows, result.main));
    result.parameters[2 * stage] = part.parameters[0];
    result.parameters[2 * stage + 1] = part.parameters[1];
    if(result.roles[stage].projection.description.is_plain()) {
      HYLE_ASSIGN_OR_RETURN(result.raw_projections[stage],builder.add(c::Copy{},part.raw_projection));
    }
    // Selected Q8 projection/F32 norm, then the explicit BF16 cache contract:
    // round postnorm, rotate only the final64, round again, group64 QDQ.
    // This does not claim the upstream official BF16-weight arithmetic.
    HYLE_ASSIGN_OR_RETURN(auto rounded, builder.add(c::Cast{c::DataType::kBFloat16}, part.kv));
    HYLE_ASSIGN_OR_RETURN(auto rotation_input, builder.add(c::Cast{c::DataType::kFloat32}, rounded));
    HYLE_ASSIGN_OR_RETURN(auto shape, c::TensorShape::create(std::array<std::int64_t, 4>{1, rows, 1, b::kKv}));
    HYLE_ASSIGN_OR_RETURN(auto reshape, c::View::reshape(shape));
    HYLE_ASSIGN_OR_RETURN(auto shaped, builder.add(reshape, rotation_input));
    HYLE_ASSIGN_OR_RETURN(auto head_view, c::View::slice(3, 0, 448));
    HYLE_ASSIGN_OR_RETURN(auto tail_view, c::View::slice(3, 448, 64));
    HYLE_ASSIGN_OR_RETURN(auto head, builder.add(head_view, shaped));
    HYLE_ASSIGN_OR_RETURN(auto tail, builder.add(tail_view, shaped));
    HYLE_ASSIGN_OR_RETURN(auto rotated, builder.add(c::RoPE{.rotary_dimensions = 64,
        .sequence_axis = 1, .layout = c::RopeLayout::kInterleaved}, tail, result.positions, result.parameters[6]));
    HYLE_ASSIGN_OR_RETURN(auto joined, builder.add(c::Concat{.axis = 3}, head, rotated));
    HYLE_ASSIGN_OR_RETURN(auto flat_shape, c::TensorShape::create(std::array{rows, b::kKv}));
    HYLE_ASSIGN_OR_RETURN(auto flatten, c::View::reshape(flat_shape));
    HYLE_ASSIGN_OR_RETURN(auto flat, builder.add(flatten, joined));
    HYLE_ASSIGN_OR_RETURN(auto cache_input, builder.add(c::Cast{c::DataType::kBFloat16}, flat));
    HYLE_ASSIGN_OR_RETURN(result.checkpoints[stage][0], builder.add(c::Copy{}, part.kv));
    HYLE_ASSIGN_OR_RETURN(result.checkpoints[stage][1], builder.add(c::Copy{}, flat));
    HYLE_ASSIGN_OR_RETURN(result.checkpoints[stage][2], builder.add(c::Copy{}, cache_input));
    HYLE_ASSIGN_OR_RETURN(auto encoded, builder.add_node(c::model_ops::row_cache_encode(kEncoding), std::array{cache_input}));
    HYLE_ASSIGN_OR_RETURN(keys[stage], builder.result(encoded));
  }
  auto current = result.rings;
  for (std::int64_t row = 0; row < rows; ++row) {
    std::array<g::ValueId, 4> prefix;
    HYLE_ASSIGN_OR_RETURN(auto single, c::View::slice(0, row, 1));
    HYLE_ASSIGN_OR_RETURN(auto slot, builder.add(single, result.slots));
    for (std::size_t stage = 0; stage < current.size(); ++stage) {
      HYLE_ASSIGN_OR_RETURN(auto update, builder.add(single, keys[stage]));
      HYLE_ASSIGN_OR_RETURN(current[stage], builder.add(c::Scatter{.axis = 0}, current[stage], slot, update));
      HYLE_ASSIGN_OR_RETURN(prefix[stage], builder.add(c::Copy{}, current[stage]));
      HYLE_RETURN_IF_ERROR(builder.publish(prefix[stage]));
    }
    HYLE_ASSIGN_OR_RETURN(auto last, builder.add(single, result.main));
    HYLE_ASSIGN_OR_RETURN(prefix[3], builder.add(c::Copy{}, last));
    HYLE_RETURN_IF_ERROR(builder.publish(prefix[3]));
    result.prefixes.push_back(prefix);
  }
  for (const auto &stage : result.checkpoints)
    for (const auto value : stage) HYLE_RETURN_IF_ERROR(builder.publish(value));
  std::size_t raw_port=std::size_t(rows)*4+9;
  for(std::size_t stage=0;stage<3;++stage)if(result.raw_projections[stage]) {
    result.raw_projection_ports[stage]=raw_port++;
    HYLE_RETURN_IF_ERROR(builder.publish(*result.raw_projections[stage]));
  }
  HYLE_ASSIGN_OR_RETURN(result.graph, std::move(builder).finish());
  return result;
}

struct Boundary final {
  std::uint64_t session, branch, epoch;
  std::int64_t next_position;
  std::int32_t pending;
  std::uint32_t consumed;
};
struct HistoryState final {
  b::TargetIdentity target;
  c::StateSchema schema;
  c::StateStore store;
  std::array<c::StateEntryId, 5> entries; // three rings, main carry, metadata
  static Result<HistoryState> create(b::TargetIdentity target) {
    HYLE_RETURN_IF_ERROR(b::qualify_identity(target, {b::kSidecarRepository, b::kSidecarRevision, b::kSidecarFile}));
    HistoryState result{.target = std::move(target)};
    HYLE_ASSIGN_OR_RETURN(auto ring, c::row_cache_carrier_description(kEncoding, kWindow));
    HYLE_ASSIGN_OR_RETURN(auto carry, b::plain(std::array<std::int64_t, 2>{1, b::kHidden}));
    const std::array specs{c::StateEntrySpec{.description = ring}, c::StateEntrySpec{.description = ring},
        c::StateEntrySpec{.description = ring}, c::StateEntrySpec{.description = carry},
        c::StateEntrySpec{.kind = c::StateEntryKind::kHostMetadata, .metadata_bytes = sizeof(Boundary)}};
    HYLE_ASSIGN_OR_RETURN(result.schema, c::StateSchema::create(specs));
    HYLE_ASSIGN_OR_RETURN(result.store, c::StateStore::create(result.schema,
        {.max_versions = 32, .max_drafts = 4, .max_retained_blocks = 128, .max_backing_bytes = 24u << 20}));
    for (std::size_t i = 0; i < result.entries.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(result.entries[i], result.schema.entry(i));
    }
    return result;
  }
  Result<Boundary> boundary(const c::StateVersion &version) const {
    HYLE_ASSIGN_OR_RETURN(const auto lease, version.member(entries[4]));
    HYLE_ASSIGN_OR_RETURN(const auto view, lease.view());
    Boundary result{};
    HYLE_RETURN_IF_ERROR(view.read(std::as_writable_bytes(std::span(&result, 1))));
    return result;
  }
  Result<c::StateVersion> select(const c::StateVersion &from, std::span<const c::ReadLease> results,
                                const b::CaptureFrame &frame, std::uint32_t accepted,
                                std::int32_t pending) {
    HYLE_ASSIGN_OR_RETURN(auto facts, boundary(from));
    HYLE_ASSIGN_OR_RETURN(const auto selected, placement(frame));
    (void)selected;
    if (frame.target != target || facts.session != frame.session || facts.branch != frame.branch || facts.epoch > frame.epoch ||
        facts.next_position != frame.first_position || accepted > std::uint32_t(frame.rows) || pending < 0 ||
        results.size() != std::size_t(frame.rows) * 4 || accepted > std::numeric_limits<std::uint32_t>::max() - facts.consumed)
      return std::unexpected(Status::InvalidArgument("DSpark accepted history does not match one complete verified frontier"));
    if (!accepted) return from;
    HYLE_ASSIGN_OR_RETURN(auto draft, store.begin(from));
    for (std::size_t member = 0; member < 4; ++member)
      HYLE_RETURN_IF_ERROR(draft.replace(entries[member], results[(accepted - 1) * 4 + member]));
    facts.epoch = frame.epoch; facts.next_position += accepted; facts.consumed += accepted; facts.pending = pending;
    HYLE_RETURN_IF_ERROR(draft.replace_metadata(entries[4], std::as_bytes(std::span(&facts, 1))));
    return draft.finalize();
  }
};
} // namespace dspark_main_history
