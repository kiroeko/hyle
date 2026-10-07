#include "default_rules.h"
#include "hyle/core/environment.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

#include "hyle/platform/cpu.h"

namespace hyle::compute::native::internal {
namespace {
enum class Family : std::uint8_t { Unary, Binary, Reduce, Norm, Softmax, Gate, Fill, Gather, Cast };
struct Domain {
  Family family;
  int variant;
  DataType type;
  DataType other;
  int rows;
  int width;
};
using Rule = Domain;
// Complete current-candidate choices, not partial pair wins or old 187/17 rules.
// Evidence: evidence/performance/cpu-final-f4-p1/complete-provider-choices.json.
constexpr std::array kWins{
    Rule{Family::Binary, 0, DataType::kFloat32, DataType::kFloat32, 32, 2048},
    Rule{Family::Binary, 1, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Binary, 1, DataType::kFloat32, DataType::kFloat32, 1, 2048},
    Rule{Family::Binary, 3, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Binary, 3, DataType::kFloat16, DataType::kFloat32, 1, 4096},
    Rule{Family::Binary, 3, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Binary, 3, DataType::kFloat32, DataType::kFloat32, 1, 4096},
    Rule{Family::Binary, 4, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Cast, 0, DataType::kFloat8E4M3FN, DataType::kFloat16, 1, 4096},
    Rule{Family::Cast, 0, DataType::kFloat8E5M2, DataType::kBFloat16, 1, 2048},
    Rule{Family::Cast, 0, DataType::kFloat8E5M2, DataType::kFloat8E5M2, 1, 4096},
    Rule{Family::Cast, 0, DataType::kFloat8E5M2, DataType::kFloat8E4M3FN, 32, 2048},
    Rule{Family::Fill, 0, DataType::kFloat32, DataType::kFloat32, 1, 2048},
    Rule{Family::Gather, 0, DataType::kFloat16, DataType::kInt32, 1, 4096},
    Rule{Family::Gather, 0, DataType::kFloat16, DataType::kInt64, 1, 4096},
    Rule{Family::Gather, 0, DataType::kBFloat16, DataType::kInt64, 1, 4096},
    Rule{Family::Gather, 0, DataType::kFloat32, DataType::kInt32, 1, 2048},
    Rule{Family::Norm, 0, DataType::kBFloat16, DataType::kFloat32, 1, 4096},
    Rule{Family::Reduce, 2, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Reduce, 3, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Reduce, 3, DataType::kBFloat16, DataType::kFloat32, 1, 4096},
    Rule{Family::Gate, 0, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Softmax, 2, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 1, DataType::kBFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 1, DataType::kBFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 11, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 2, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 2, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 2, DataType::kBFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 2, DataType::kFloat32, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 2, DataType::kFloat32, DataType::kFloat32, 1, 4096},
    Rule{Family::Unary, 2, DataType::kFloat32, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 3, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 4, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 4, DataType::kBFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 5, DataType::kFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 5, DataType::kFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 5, DataType::kBFloat16, DataType::kFloat32, 32, 2048},
    Rule{Family::Unary, 6, DataType::kFloat16, DataType::kFloat32, 1, 4096},
    Rule{Family::Unary, 8, DataType::kBFloat16, DataType::kFloat32, 1, 2048},
    Rule{Family::Unary, 8, DataType::kBFloat16, DataType::kFloat32, 32, 2048},
};

bool shape(const TensorDesc &d, std::initializer_list<std::int64_t> dims,
           DataType type) noexcept {
  return d.is_plain() && d.is_dense() && d.data_type() == type &&
         d.rank() == dims.size() &&
         std::equal(d.dimensions().begin(), d.dimensions().end(), dims.begin());
}
std::optional<Domain> domain(Operation op, std::span<const TensorDesc> in,
                             std::span<const TensorDesc> out, int width) noexcept {
  if (out.size() != 1 || !out[0].is_plain() || !out[0].is_dense())
    return {};
  const auto t = out[0].data_type();
  const auto scalar_rows = [&]() -> int {
    for (int r : {1, 32})
      if (shape(out[0], {r, width}, t))
        return r;
    return 0;
  };
  const int r = scalar_rows();
  const auto tensor = [&](std::size_t n) {
    return n < in.size() && shape(in[n], {r, width}, t);
  };
  const auto result = [&](Family f, int v = 0,
                          DataType other = DataType::kFloat32) {
    return std::optional<Domain>(Domain{f, v, t, other, r, width});
  };
  if (auto p = op.get_if<Unary>(); p && r && in.size() == 1 && tensor(0))
    return result(Family::Unary, int(p->kind));
  if (auto p = op.get_if<Binary>();
      p && r && in.size() == 2 && tensor(0) && tensor(1))
    return result(Family::Binary, int(p->kind));
  if (auto p = op.get_if<RmsNorm>();
      p && r && in.size() == 2 && tensor(0) && shape(in[1], {width}, t) &&
      (p->epsilon == 1e-5F ||
       (t == DataType::kBFloat16 && p->epsilon == 1e-6F)) &&
      p->accumulation_type == DataType::kFloat32)
    return result(Family::Norm, p->epsilon == 1e-6F ? 2 : 0);
  if (auto p = op.get_if<LayerNorm>();
      p && r && in.size() == 2 && tensor(0) && shape(in[1], {width}, t) &&
      p->epsilon == 1e-5F && p->accumulation_type == DataType::kFloat32)
    return result(Family::Norm, 1);
  if (auto p = op.get_if<Softmax>();
      p && r && p->axis == -1 && p->scale == 1.F &&
      p->accumulation_type == DataType::kFloat32 && tensor(0)) {
    if (p->mask == MaskKind::kNone && in.size() == 1)
      return result(Family::Softmax, int(p->kind));
    if (p->mask == MaskKind::kBoolean && in.size() == 2 &&
        shape(in[1], {r, width}, DataType::kBool))
      return result(Family::Softmax, int(p->kind) + 2);
  }
  if (auto p = op.get_if<GatedActivation>();
      p && r && p->activation == GatedActivationKind::kSilu) {
    if (p->inputs == GatedInputKind::kSeparate && in.size() == 2 && tensor(0) &&
        tensor(1))
      return result(Family::Gate);
    if (p->inputs == GatedInputKind::kPackedHalves && in.size() == 1 &&
        shape(in[0], {r, 2 * width}, t))
      return result(Family::Gate, 1);
  }
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
      shape(in[0], {r, width}, in[0].data_type()))
    return Domain{Family::Cast, 0, in[0].data_type(), t, r, width};
  if (auto p = op.get_if<Reduce>();
      p && p->accumulation_type == DataType::kFloat32 && !p->keep_dimensions &&
      in.size() == 1 && p->axes.values().size() == 1 &&
      p->axes.values()[0] == 1) {
    for (int rows : {1, 32})
      if (shape(in[0], {rows, width}, t) && shape(out[0], {rows}, t))
        return Domain{Family::Reduce, int(p->kind), t, DataType::kFloat32,
                      rows, width};
  }
  if (auto p = op.get_if<Gather>();
      p && r && p->axis == 0 && p->bounds == IndexBoundsPolicy::kChecked &&
      in.size() == 2 && shape(in[0], {64, width}, t) &&
      shape(in[1], {r}, in[1].data_type()))
    return result(Family::Gather, 0, in[1].data_type());
  return {};
}

bool contains(const Domain &d) noexcept {
  return std::any_of(kWins.begin(), kWins.end(), [&](const Rule &r) {
    return r.family == d.family && r.variant == d.variant && r.type == d.type &&
           r.other == d.other && r.rows == d.rows && r.width == d.width;
  });
}
RoutePolicyId policy(const Domain &d) noexcept {
  const auto key = (std::uint64_t(d.family) << 48) |
                   (std::uint64_t(d.variant) << 40) |
                   (std::uint64_t(d.type) << 32) |
                   (std::uint64_t(d.other) << 24) |
                   (std::uint64_t(d.width) << 8) | std::uint64_t(d.rows);
  return {0x48594c4543505552ULL, key + 1, 3};
}
} // namespace

bool qualified_default_environment() noexcept {
#if defined(HYLE_NATIVE_DEFAULT_PEER_F4) && HYLE_OS_WINDOWS && \
    HYLE_BUILD_RELEASE && defined(_MSC_FULL_VER) && _MSC_FULL_VER == 195035728
  const auto &identity = platform::cpu_identity();
  std::string_view brand(identity.brand.data());
  while (!brand.empty() && brand.front() == ' ')
    brand.remove_prefix(1);
  while (!brand.empty() && brand.back() == ' ')
    brand.remove_suffix(1);
  // W1 qualification used this single process-available P-core. An unrestricted
  // hybrid process must not acquire that placement's WIN merely from its brand.
  unsigned caller_cpu = 0;
  const auto available = platform::available_cpu_ids(&caller_cpu, 1);
  if (!available || *available != 1 || caller_cpu != 2)
    return false;
  return std::string_view(identity.vendor.data()) == "GenuineIntel" &&
         identity.family == 6 && identity.model == 183 && identity.stepping == 1 &&
         brand == "13th Gen Intel(R) Core(TM) i9-13900K" &&
         platform::cpu_features().has(platform::CpuFeature::kX86Avx2);
#else
  return false;
#endif
}

DefaultRuleSelection qualified_default_selection(
    Operation op, std::span<const TensorDesc> in,
    std::span<const TensorDesc> out) noexcept {
  for (int width : {2048, 4096}) {
    const auto d = domain(op, in, out, width);
    if (!d)
      continue;
    if (contains(*d))
      return {DefaultRuleProvider::kNative, policy(*d)};
    if (d->family == Family::Softmax && d->variant == 2 &&
        d->type == DataType::kFloat16 && d->rows == 32 && d->width == 2048)
      return {DefaultRuleProvider::kAten, policy(*d)};
  }
  // Exact physical Qwen projection, including intermediate/output precision.
  if (const auto *p = op.get_if<MatMul>();
      p && p->accumulation_type == DataType::kFloat32 && in.size() == 2 && out.size() == 1 &&
      shape(in[0], {32, 2048}, DataType::kBFloat16) &&
      shape(out[0], {32, 6144}, DataType::kBFloat16)) {
    const auto weight = TensorDesc::dense(DataType::kBFloat16, std::array<std::int64_t, 2>{6144, 2048});
    if (weight) {
      const auto transposed = weight->transposed();
      if (transposed && in[1] == *transposed)
        return {DefaultRuleProvider::kGgml, {0x48594c4543505552ULL, 0xff00000000000001ULL, 3}};
    }
  }
  return {};
}

RoutePolicyId qualified_default_policy(Operation op, std::span<const TensorDesc> in,
                                       std::span<const TensorDesc> out) noexcept {
  const auto selection = qualified_default_selection(op, in, out);
  return selection.provider == DefaultRuleProvider::kNative ? selection.policy : RoutePolicyId{};
}
} // namespace hyle::compute::native::internal
