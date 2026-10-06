#include "default_rules.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

#include "../compute/provider_internal.h"
#include "hyle/compute/operation.h"
#include "hyle/platform/cpu.h"

namespace hyle::compute::native::internal {
namespace {
enum class Family : std::uint8_t {
  Unary,
  Binary,
  Reduce,
  Norm,
  Softmax,
  Gate,
  Select,
  Concat,
  Fill,
  Gather,
  Assign,
  ScatterAdd,
  Copy,
  Cast,
  Rope,
  Attention
};
struct Domain {
  Family family;
  int variant;
  DataType type;
  DataType other;
  int rows;
};
struct Rule {
  Family family;
  int variant;
  DataType type;
  DataType other;
  unsigned rows_mask;
};
constexpr std::array kWins{
    Rule{Family::Assign, 0, DataType(4), DataType(12), 1},
    Rule{Family::Assign, 0, DataType(4), DataType(14), 1},
    Rule{Family::Assign, 0, DataType(5), DataType(14), 1},
    Rule{Family::Assign, 0, DataType(6), DataType(12), 1},
    Rule{Family::Binary, 0, DataType(4), DataType(6), 3},
    Rule{Family::Binary, 0, DataType(5), DataType(6), 2},
    Rule{Family::Binary, 0, DataType(6), DataType(6), 1},
    Rule{Family::Binary, 1, DataType(4), DataType(6), 3},
    Rule{Family::Binary, 1, DataType(5), DataType(6), 3},
    Rule{Family::Binary, 1, DataType(6), DataType(6), 3},
    Rule{Family::Binary, 2, DataType(4), DataType(6), 2},
    Rule{Family::Binary, 2, DataType(5), DataType(6), 3},
    Rule{Family::Binary, 2, DataType(6), DataType(6), 3},
    Rule{Family::Binary, 3, DataType(5), DataType(6), 1},
    Rule{Family::Binary, 3, DataType(6), DataType(6), 1},
    Rule{Family::Binary, 4, DataType(4), DataType(6), 3},
    Rule{Family::Binary, 4, DataType(5), DataType(6), 2},
    Rule{Family::Binary, 4, DataType(6), DataType(6), 1},
    Rule{Family::Binary, 5, DataType(4), DataType(6), 2},
    Rule{Family::Binary, 5, DataType(5), DataType(6), 3},
    Rule{Family::Cast, 0, DataType(2), DataType(2), 1},
    Rule{Family::Cast, 0, DataType(2), DataType(3), 3},
    Rule{Family::Cast, 0, DataType(2), DataType(4), 1},
    Rule{Family::Cast, 0, DataType(2), DataType(5), 3},
    Rule{Family::Cast, 0, DataType(2), DataType(6), 3},
    Rule{Family::Cast, 0, DataType(3), DataType(3), 1},
    Rule{Family::Cast, 0, DataType(3), DataType(4), 3},
    Rule{Family::Cast, 0, DataType(3), DataType(5), 3},
    Rule{Family::Cast, 0, DataType(3), DataType(6), 3},
    Rule{Family::Cast, 0, DataType(4), DataType(2), 2},
    Rule{Family::Cast, 0, DataType(4), DataType(3), 3},
    Rule{Family::Cast, 0, DataType(4), DataType(4), 1},
    Rule{Family::Cast, 0, DataType(4), DataType(5), 3},
    Rule{Family::Cast, 0, DataType(4), DataType(6), 3},
    Rule{Family::Cast, 0, DataType(5), DataType(3), 1},
    Rule{Family::Cast, 0, DataType(5), DataType(4), 3},
    Rule{Family::Cast, 0, DataType(5), DataType(6), 1},
    Rule{Family::Cast, 0, DataType(6), DataType(2), 2},
    Rule{Family::Cast, 0, DataType(6), DataType(3), 3},
    Rule{Family::Cast, 0, DataType(6), DataType(4), 1},
    Rule{Family::Cast, 0, DataType(6), DataType(5), 1},
    Rule{Family::Cast, 0, DataType(6), DataType(6), 1},
    Rule{Family::Concat, 0, DataType(4), DataType(6), 3},
    Rule{Family::Concat, 0, DataType(5), DataType(6), 3},
    Rule{Family::Concat, 0, DataType(6), DataType(6), 3},
    Rule{Family::Copy, 0, DataType(2), DataType(6), 3},
    Rule{Family::Copy, 0, DataType(3), DataType(6), 1},
    Rule{Family::Copy, 0, DataType(4), DataType(6), 2},
    Rule{Family::Copy, 0, DataType(5), DataType(6), 3},
    Rule{Family::Copy, 0, DataType(6), DataType(6), 3},
    Rule{Family::Fill, 0, DataType(4), DataType(6), 3},
    Rule{Family::Fill, 0, DataType(5), DataType(6), 3},
    Rule{Family::Fill, 0, DataType(6), DataType(6), 2},
    Rule{Family::Gate, 0, DataType(4), DataType(6), 1},
    Rule{Family::Gate, 1, DataType(4), DataType(6), 2},
    Rule{Family::Gather, 0, DataType(4), DataType(14), 2},
    Rule{Family::Gather, 0, DataType(5), DataType(14), 2},
    Rule{Family::Gather, 0, DataType(6), DataType(12), 3},
    Rule{Family::Gather, 0, DataType(6), DataType(14), 3},
    Rule{Family::Norm, 0, DataType(4), DataType(6), 3},
    Rule{Family::Norm, 0, DataType(5), DataType(6), 2},
    Rule{Family::Norm, 0, DataType(6), DataType(6), 1},
    Rule{Family::Norm, 1, DataType(4), DataType(6), 3},
    Rule{Family::Norm, 1, DataType(5), DataType(6), 3},
    Rule{Family::Norm, 1, DataType(6), DataType(6), 1},
    Rule{Family::Reduce, 0, DataType(4), DataType(6), 3},
    Rule{Family::Reduce, 0, DataType(5), DataType(6), 3},
    Rule{Family::Reduce, 0, DataType(6), DataType(6), 3},
    Rule{Family::Reduce, 1, DataType(4), DataType(6), 2},
    Rule{Family::Reduce, 1, DataType(5), DataType(6), 3},
    Rule{Family::Reduce, 1, DataType(6), DataType(6), 3},
    Rule{Family::Reduce, 2, DataType(4), DataType(6), 2},
    Rule{Family::Reduce, 2, DataType(5), DataType(6), 2},
    Rule{Family::Reduce, 2, DataType(6), DataType(6), 1},
    Rule{Family::Reduce, 3, DataType(4), DataType(6), 2},
    Rule{Family::Reduce, 3, DataType(5), DataType(6), 2},
    Rule{Family::Rope, 0, DataType(6), DataType(6), 3},
    Rule{Family::ScatterAdd, 0, DataType(6), DataType(14), 1},
    Rule{Family::Select, 0, DataType(4), DataType(6), 3},
    Rule{Family::Select, 0, DataType(5), DataType(6), 3},
    Rule{Family::Select, 0, DataType(6), DataType(6), 3},
    Rule{Family::Softmax, 0, DataType(4), DataType(6), 3},
    Rule{Family::Softmax, 0, DataType(5), DataType(6), 3},
    Rule{Family::Softmax, 0, DataType(6), DataType(6), 3},
    Rule{Family::Softmax, 1, DataType(4), DataType(6), 3},
    Rule{Family::Softmax, 1, DataType(5), DataType(6), 3},
    Rule{Family::Softmax, 1, DataType(6), DataType(6), 3},
    Rule{Family::Softmax, 2, DataType(4), DataType(6), 1},
    Rule{Family::Softmax, 2, DataType(5), DataType(6), 1},
    Rule{Family::Softmax, 3, DataType(5), DataType(6), 1},
    Rule{Family::Unary, 0, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 0, DataType(5), DataType(6), 3},
    Rule{Family::Unary, 0, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 1, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 1, DataType(5), DataType(6), 3},
    Rule{Family::Unary, 1, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 2, DataType(4), DataType(6), 1},
    Rule{Family::Unary, 2, DataType(5), DataType(6), 3},
    Rule{Family::Unary, 2, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 3, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 3, DataType(5), DataType(6), 2},
    Rule{Family::Unary, 3, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 4, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 4, DataType(5), DataType(6), 2},
    Rule{Family::Unary, 4, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 5, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 5, DataType(5), DataType(6), 3},
    Rule{Family::Unary, 5, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 6, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 6, DataType(5), DataType(6), 1},
    Rule{Family::Unary, 6, DataType(6), DataType(6), 1},
    Rule{Family::Unary, 7, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 7, DataType(5), DataType(6), 2},
    Rule{Family::Unary, 7, DataType(6), DataType(6), 3},
    Rule{Family::Unary, 8, DataType(4), DataType(6), 3},
    Rule{Family::Unary, 8, DataType(5), DataType(6), 3},
    Rule{Family::Unary, 8, DataType(6), DataType(6), 2},
    Rule{Family::Unary, 11, DataType(4), DataType(6), 2},
    Rule{Family::Unary, 11, DataType(5), DataType(6), 1},
    Rule{Family::Unary, 12, DataType(5), DataType(6), 1},
};
constexpr std::array kLosses{
    Rule{Family::Assign, 0, DataType(4), DataType(14), 2},
    Rule{Family::Assign, 0, DataType(5), DataType(12), 2},
    Rule{Family::Assign, 0, DataType(6), DataType(12), 2},
    Rule{Family::Attention, 0, DataType(6), DataType(6), 2},
    Rule{Family::Norm, 0, DataType(6), DataType(6), 2},
    Rule{Family::Softmax, 2, DataType(4), DataType(6), 2},
    Rule{Family::Softmax, 2, DataType(5), DataType(6), 2},
    Rule{Family::Softmax, 2, DataType(6), DataType(6), 2},
    Rule{Family::Softmax, 3, DataType(4), DataType(6), 2},
    Rule{Family::Softmax, 3, DataType(5), DataType(6), 2},
    Rule{Family::Softmax, 3, DataType(6), DataType(6), 2},
    Rule{Family::Unary, 12, DataType(4), DataType(6), 2},
    Rule{Family::Unary, 12, DataType(5), DataType(6), 2},
};

bool shape(const TensorDesc &d, std::initializer_list<std::int64_t> dims,
           DataType type) noexcept {
  return d.is_plain() && d.is_dense() && d.data_type() == type &&
         d.rank() == dims.size() &&
         std::equal(d.dimensions().begin(), d.dimensions().end(), dims.begin());
}
std::optional<Domain> domain(Operation op, std::span<const TensorDesc> in,
                             std::span<const TensorDesc> out) noexcept {
  if (out.size() != 1 || !out[0].is_plain() || !out[0].is_dense())
    return {};
  const auto t = out[0].data_type();
  const auto scalar_rows = [&]() -> int {
    for (int r : {1, 32})
      if (shape(out[0], {r, 2048}, t))
        return r;
    return 0;
  };
  const int r = scalar_rows();
  const auto tensor = [&](std::size_t n) {
    return n < in.size() && shape(in[n], {r, 2048}, t);
  };
  const auto result = [&](Family f, int v = 0,
                          DataType other = DataType::kFloat32) {
    return std::optional<Domain>(Domain{f, v, t, other, r});
  };
  if (auto p = op.get_if<Unary>(); p && r && in.size() == 1 && tensor(0))
    return result(Family::Unary, int(p->kind));
  if (auto p = op.get_if<Binary>();
      p && r && in.size() == 2 && tensor(0) && tensor(1))
    return result(Family::Binary, int(p->kind));
  if (auto p = op.get_if<RmsNorm>();
      p && r && in.size() == 2 && tensor(0) && shape(in[1], {2048}, t) &&
      p->epsilon == 1e-5F && p->accumulation_type == DataType::kFloat32)
    return result(Family::Norm);
  if (auto p = op.get_if<LayerNorm>();
      p && r && in.size() == 2 && tensor(0) && shape(in[1], {2048}, t) &&
      p->epsilon == 1e-5F && p->accumulation_type == DataType::kFloat32)
    return result(Family::Norm, 1);
  if (auto p = op.get_if<Softmax>();
      p && r && p->axis == -1 && p->scale == 1.F &&
      p->accumulation_type == DataType::kFloat32 && tensor(0)) {
    if (p->mask == MaskKind::kNone && in.size() == 1)
      return result(Family::Softmax, int(p->kind));
    if (p->mask == MaskKind::kBoolean && in.size() == 2 &&
        shape(in[1], {r, 2048}, DataType::kBool))
      return result(Family::Softmax, int(p->kind) + 2);
  }
  if (auto p = op.get_if<GatedActivation>();
      p && r && p->activation == GatedActivationKind::kSilu) {
    if (p->inputs == GatedInputKind::kSeparate && in.size() == 2 && tensor(0) &&
        tensor(1))
      return result(Family::Gate);
    if (p->inputs == GatedInputKind::kPackedHalves && in.size() == 1 &&
        shape(in[0], {r, 4096}, t))
      return result(Family::Gate, 1);
  }
  if (op.kind() == OperationKind::kSelect && r && in.size() == 3 &&
      shape(in[0], {r, 2048}, DataType::kBool) && tensor(1) && tensor(2))
    return result(Family::Select);
  if (op.kind() == OperationKind::kCopy && r && in.size() == 1 && tensor(0))
    return result(Family::Copy);
  if (auto p = op.get_if<Fill>(); p && r && in.empty() && p->output_type == t &&
                                  p->value.data_type() == t) {
    // The qualified request fills 1.25; do not extrapolate a scalar-sensitive
    // implementation to zero, NaN, or another unmeasured attribute.
    const auto bits = t == DataType::kFloat32    ? 0x3fa00000ULL
                      : t == DataType::kFloat16  ? 0x3d00ULL
                      : t == DataType::kBFloat16 ? 0x3fa0ULL
                                                 : 0ULL;
    if (bits && p->value.bits() == bits)
      return result(Family::Fill);
  }
  if (op.kind() == OperationKind::kCast && r && in.size() == 1 &&
      shape(in[0], {r, 2048}, in[0].data_type()))
    return Domain{Family::Cast, 0, in[0].data_type(), t, r};
  if (auto p = op.get_if<Concat>(); p && p->axis == 1 && in.size() == 2) {
    for (int rows : {1, 32})
      if (shape(out[0], {rows, 4096}, t) && shape(in[0], {rows, 2048}, t) &&
          shape(in[1], {rows, 2048}, t))
        return Domain{Family::Concat, 0, t, DataType::kFloat32, rows};
  }
  if (auto p = op.get_if<Reduce>();
      p && p->accumulation_type == DataType::kFloat32 && !p->keep_dimensions &&
      in.size() == 1 && p->axes.values().size() == 1 &&
      p->axes.values()[0] == 1) {
    for (int rows : {1, 32})
      if (shape(in[0], {rows, 2048}, t) && shape(out[0], {rows}, t))
        return Domain{Family::Reduce, int(p->kind), t, DataType::kFloat32,
                      rows};
  }
  if (auto p = op.get_if<Gather>();
      p && r && p->axis == 0 && p->bounds == IndexBoundsPolicy::kChecked &&
      in.size() == 2 && shape(in[0], {64, 2048}, t) &&
      shape(in[1], {r}, in[1].data_type()))
    return result(Family::Gather, 0, in[1].data_type());
  if (auto p = op.get_if<Scatter>();
      p && p->axis == 0 && p->bounds == IndexBoundsPolicy::kChecked &&
      in.size() == 3 && shape(out[0], {64, 2048}, t) &&
      shape(in[0], {64, 2048}, t)) {
    for (int rows : {1, 32})
      if (shape(in[1], {rows}, in[1].data_type()) &&
          shape(in[2], {rows, 2048}, t))
        return Domain{p->kind == ScatterKind::kAssign ? Family::Assign
                                                      : Family::ScatterAdd,
                      0, t, in[1].data_type(), rows};
  }
  if (auto p = op.get_if<RoPE>();
      p && p->rotary_dimensions == 64 && p->sequence_axis == 1 &&
      p->layout == RopeLayout::kInterleaved && in.size() == 3) {
    for (int rows : {1, 32})
      if (shape(in[0], {1, rows, 8, 256}, t) &&
          shape(out[0], {1, rows, 8, 256}, t) &&
          shape(in[1], {rows}, DataType::kInt32) &&
          shape(in[2], {32}, DataType::kFloat32))
        return Domain{Family::Rope, 0, t, DataType::kFloat32, rows};
  }
  if (auto p = op.get_if<ScaledDotProductAttention>();
      p && p->layout == AttentionLayout::kBatchSequenceHeadDepth &&
      p->mask == MaskKind::kNone && p->scale == .125F && !p->causal &&
      p->accumulation_type == DataType::kFloat32 && in.size() == 3) {
    for (int rows : {1, 32})
      if (shape(in[0], {1, rows, 4, 64}, t) &&
          shape(out[0], {1, rows, 4, 64}, t) &&
          shape(in[1], {1, 128, 4, 64}, t) && shape(in[2], {1, 128, 4, 64}, t))
        return Domain{Family::Attention, 0, t, DataType::kFloat32, rows};
  }
  return {};
}
template <std::size_t N>
bool contains(const std::array<Rule, N> &rules, const Domain &d) noexcept {
  const unsigned bit = d.rows == 1 ? 1U : 2U;
  return std::any_of(rules.begin(), rules.end(), [&](const Rule &r) {
    return r.family == d.family && r.variant == d.variant && r.type == d.type &&
           r.other == d.other && (r.rows_mask & bit);
  });
}
} // namespace

bool qualified_default_environment() noexcept {
#if defined(HYLE_NATIVE_DEFAULT_PEER_456172EC) && defined(_WIN32) && \
    defined(NDEBUG) && defined(_MSC_FULL_VER) && \
    _MSC_FULL_VER == 195035728
  const auto &identity = platform::cpu_identity();
  std::string_view brand(identity.brand.data());
  while (!brand.empty() && brand.front() == ' ')
    brand.remove_prefix(1);
  while (!brand.empty() && brand.back() == ' ')
    brand.remove_suffix(1);
  return std::string_view(identity.vendor.data()) == "GenuineIntel" &&
         identity.family == 6 && identity.model == 183 &&
         identity.stepping == 1 &&
         brand == "13th Gen Intel(R) Core(TM) i9-13900K" &&
         platform::cpu_features().has(platform::CpuFeature::kX86Avx2);
#else
  return false;
#endif
}
bool retired_default_domain(Operation op, std::span<const TensorDesc> in,
                            std::span<const TensorDesc> out) noexcept {
  const auto d = domain(op, in, out);
  return d && contains(kLosses, *d);
}
RoutePolicyId
qualified_default_policy(Operation op, std::span<const TensorDesc> in,
                         std::span<const TensorDesc> out) noexcept {
  const auto d = domain(op, in, out);
  if (!d || !contains(kWins, *d))
    return {};
  const auto key = (std::uint64_t(d->family) << 40) |
                   (std::uint64_t(d->variant) << 32) |
                   (std::uint64_t(d->type) << 24) |
                   (std::uint64_t(d->other) << 16) | std::uint64_t(d->rows);
  return {0x48594c4543505552ULL, key + 1, 1};
}
} // namespace hyle::compute::native::internal
