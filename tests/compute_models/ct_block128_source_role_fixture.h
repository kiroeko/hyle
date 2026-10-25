#pragma once

#include "ct_block128_qdq_fixture.h"

namespace ct_block128_source_role_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace math = hyle::inference::ct_block128;
namespace f = ct_block128_qdq_test;

// A bounded original ordinary role, not the whole expert or checkpoint. These
// facts come from the fixed shard's original safetensors header. Full K and one
// complete N block preserve the original SF axes without repacking either bank.
struct SourceRole final {
  static constexpr std::string_view repository = "RedHatAI/Kimi-K3-FP8-BLOCK";
  static constexpr std::string_view revision = "6409c37cf440abeb76f0b44ab4fb7db8c8bccb4b";
  static constexpr std::string_view file = "model-00002-of-000096.safetensors";
  static constexpr std::string_view header_sha256 = "2c5e3704e8502af121eba27b9baaebdcce5bd753b89881649c531a25535ff386";
  static constexpr std::string_view weight = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight";
  static constexpr std::string_view scale = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale";
  static constexpr std::int64_t original_rows = 3072, columns = 128, depth = 3584;
  static constexpr std::uint64_t data_start = 794592, file_bytes = 30683682144;
  static constexpr std::uint64_t weight_data = 904377728, scale_data = 643584;
  static constexpr std::size_t weight_bytes = std::size_t(columns * depth);
  static constexpr std::size_t scale_bytes = std::size_t(depth / 128 * 2);
  static constexpr std::uint64_t weight_begin = data_start + weight_data;
  static constexpr std::uint64_t scale_begin = data_start + scale_data;
};

// The graph's W/SF are unbound declarations until the caller supplies ready
// original source leases. No default, zero or generated source W/SF is created.
// Authentication/acquisition of the exact ranges is the source owner's job.
// Tensor construction checks the declared extent; normal Graph compile keeps
// its existing owner/resource/allocator validation, without a new bypass.
// ConstantBinding itself owns the real leases; there is no second owner bank.
struct Case final {
  g::Graph graph;
  g::ValueId activation{}, output{};
  c::TensorDesc activation_description, output_description;
  std::array<g::ConstantBinding, 6> constants;

  explicit Case(std::int64_t tokens = 1) {
    activation_description = f::dense(c::DataType::kBFloat16, {tokens, SourceRole::depth});
    auto b = f::take(g::GraphBuilder::create(), "original CT source builder");
    activation = f::take(b.input(activation_description), "original-shaped BF16 activation");
    const std::array descriptions{
        f::dense(c::DataType::kFloat8E4M3FN, {SourceRole::columns, SourceRole::depth}),
        f::dense(c::DataType::kBFloat16, {1, SourceRole::depth / 128}),
        f::dense(c::DataType::kInt32, {SourceRole::columns}),
        f::dense(c::DataType::kBFloat16, {}), f::dense(c::DataType::kBFloat16, {}),
        f::dense(c::DataType::kBFloat16, {})};
    std::array<math::Operand, 6> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto value = f::take(b.constant(descriptions[i]), "CT source constant declaration");
      values[i] = {value, descriptions[i]}; constants[i].value = value;
      constants[i].policy = g::ConstantPolicy::kMaterialize;
    }
    const auto literal = [&](std::size_t index, std::span<const std::byte> bytes) {
      auto buffer = f::take(m::Buffer::create(m::system_resource(), bytes.size(), 256), "CT small literal");
      f::check(buffer.view().write(bytes), "CT small literal bytes");
      auto lease = f::take(c::ReadLease::adopt(std::move(buffer)), "CT small literal lease");
      constants[index].tensor = f::take(lease.tensor(descriptions[index]), "CT small literal view");
      constants[index].lease = std::move(lease);
    };
    // This tile starts at original row0: every one of its128 rows uses SF row0.
    const std::array<std::int32_t, SourceRole::columns> block_rows{};
    const std::array<std::uint16_t, 1> limit{0x43e0}, epsilon{0x3c00}, zero{0};
    literal(2, std::as_bytes(std::span(block_rows)));
    literal(3, std::as_bytes(std::span(limit)));
    literal(4, std::as_bytes(std::span(epsilon)));
    literal(5, std::as_bytes(std::span(zero)));
    const auto result = f::take(math::append_projection(b, {activation, activation_description},
        values[0], values[1], values[2], {values[3], values[4], values[5]}), "original-shaped CT projection");
    output = result.value; output_description = result.description;
    f::check(b.publish(output), "publish original-shaped CT projection");
    graph = f::take(std::move(b).finish(), "finish original-shaped CT graph");
  }

  // The two ranges may belong to independent source resources/owners. Check
  // both before publishing either binding. No SF widening, weight decode,
  // concatenation, host upload or extra permanent floating bank happens here.
  hyle::Status bind_source(const c::ReadLease &weight, const c::ReadLease &scale,
      c::StorageOwner weight_owner = {}, c::StorageOwner scale_owner = {}) noexcept {
    if (constants[0].tensor.valid() || constants[1].tensor.valid())
      return hyle::Status::FailedPrecondition("original CT source constants are already bound");
    if (!weight.valid() || !scale.valid() || weight.size() != SourceRole::weight_bytes ||
        scale.size() != SourceRole::scale_bytes)
      return hyle::Status::InvalidArgument("original CT source requires its exact W and BF16 SF tile extents");
    HYLE_ASSIGN_OR_RETURN(auto w_desc, c::TensorDesc::dense(c::DataType::kFloat8E4M3FN,
        std::array{SourceRole::columns, SourceRole::depth}));
    HYLE_ASSIGN_OR_RETURN(auto sf_desc, c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array{std::int64_t{1}, SourceRole::depth / 128}));
    HYLE_ASSIGN_OR_RETURN(auto w, weight.tensor(w_desc));
    HYLE_ASSIGN_OR_RETURN(auto sf, scale.tensor(sf_desc));
    g::ConstantBinding w_binding{constants[0].value, w, constants[0].policy,
        std::move(weight_owner), weight};
    g::ConstantBinding sf_binding{constants[1].value, sf, constants[1].policy,
        std::move(scale_owner), scale};
    constants[0] = std::move(w_binding); constants[1] = std::move(sf_binding);
    return {};
  }
};

} // namespace ct_block128_source_role_test
