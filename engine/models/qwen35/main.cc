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
#include <type_traits>
#include <utility>
#include <vector>

#include "schedule.h"
#include "session_schedule.h"
#include "../../inference/session_cache.h"
#include "session_state.h"
#include "prefix_cache.h"
#include "batch_schedule.h"
#include "mtp_graph.h"
#include "../../inference/mtp_acceptance.h"
#include "../../inference/sampling_graph.h"
#include "../../inference/ct_block128_graph.h"
#include "../../inference/execution_resources.h"
#include "../../inference/retirement_observation.h"
#include "../../inference/evaluation_failure.h"
#include "../../inference/prepared_output.h"
#include "../../inference/request_progress.h"
#include "../../inference/request_stop_policy.h"
#include "../../inference/request_window.h"
#include "../../inference/version_admission.h"
#include "../../inference/batch_indices.h"
#include "../../inference/execution_observation.h"
#include "../../reporting/execution_report.h"
#include "generation_schedule.h"
#include "prepared_roles.h"
#include "ffn_placement.h"
#include "projection_binding.h"
#include "fp8_weight_import.h"
#include "precision_profile.h"
#include "cache_precision.h"
#include "nf4_weight_import.h"
#include "hyle/simd/dtype.h"
#include "hyle/container/small_vector.h"

#include "../../inference/backend.h"
#include "../../inference/readback.h"
#include "driver.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_budget.h"
#include "hyle/graph/batch.h"
#include "hyle/graph/progress.h"
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
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using V = g::ValueId;
using D = g::Dimension;
constexpr D f(std::int64_t value) noexcept { return D::fixed(value); }
using Type = c::DataType;
using Clock = std::chrono::steady_clock;
enum class ReplayPolicy { kAuto, kCommands, kPreferCapture };
constexpr bool is_end_of_generation(std::int32_t token) noexcept {
  return token == 248044 || token == 248046; // Original endoftext and tokenizer im_end.
}
constexpr std::array<const char *, 9> layer0_probe_names{
    "01-gdn-output", "02-out-proj-input", "03-out-proj-output",
    "04-down-proj-input", "05-layer-output", "06-post-attn-residual",
    "07-post-attn-norm", "08-gate-proj-output", "09-up-proj-output"};

void check(hyle::Status status) {
  if (!status.ok()) throw inference::EvaluationFailure(std::move(status));
}
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw inference::EvaluationFailure(static_cast<hyle::Status>(result.error()));
  return static_cast<T &&>(*result);
}
c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, std::span(shape.begin(), shape.size())));
}
struct Tensor final {
  c::TensorDesc description;
  mem::Buffer buffer;
  c::LeasePool pool;
  c::WriteLease writer;
  c::ReadLease ready;
  Tensor() = default;
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      // Drop the previous payload and its original domain together, before
      // adopting a replacement. Escaped readers keep that domain's charge.
      ready.reset(); writer.reset(); pool = {}; buffer = {};
      description = other.description;
      buffer = std::move(other.buffer); pool = std::move(other.pool);
      writer = std::move(other.writer); ready = std::move(other.ready);
    }
    return *this;
  }
  Tensor(mem::MemoryResourceRef resource, c::TensorDesc desc, bool caller = true,
         const c::ResourceBudget &budget = {})
      : description(desc) {
    if (!caller) return;
    const auto bytes = take(desc.required_bytes());
    if (budget && bytes) {
      pool = take(c::LeasePool::create(resource, bytes, {.max_blocks = 1, .resource_budget = budget}));
      writer = take(pool.allocate(bytes));
    } else buffer = take(mem::Buffer::create(resource, bytes, 256));
  }
  bool caller_owned() const noexcept { return bool(buffer) || bool(writer); }
  mem::BufferView storage() { return writer ? take(writer.view()) : buffer.view(); }
  c::TensorView view() { return take(c::TensorView::create(description, storage())); }
  void freeze(const Backend &backend) {
    if (!writer) { check(backend.freeze(buffer, ready)); return; }
    auto owner = take(backend.storage_owner_for(storage().resource()));
    auto retained = owner ? std::make_unique<c::StorageOwner>(std::move(owner)) : nullptr;
    ready = take(pool.freeze(std::move(writer)));
    if (retained) c::detail::LeaseAccess::retain_resource_owner(ready, retained.release(),
        +[](void *value) noexcept { delete static_cast<c::StorageOwner *>(value); });
  }
  c::ConstTensorView read() {
    return ready ? take(ready.tensor(description)) : view().as_const();
  }
  void zero() { check(storage().fill(std::byte{})); }
  template<class T> void write(std::span<const T> values) {
    check(storage().write(std::as_bytes(values)));
  }
  void dump(const std::filesystem::path &path) {
    std::vector<std::byte> bytes(take(description.required_bytes()));
    check(storage().read(bytes));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write trace: " + path.string());
  }
};

// Source-reader annotations belong only to a loaded physical weight, never
// to activation/literal/state storage. Bindings borrow its stable tensor child.
struct LoadedWeight final {
  Tensor tensor;
  prepared::ProjectionCodec codec = prepared::ProjectionCodec::kUnspecified;
  std::uint32_t group_size = 0;
};

namespace projection_detail {
using Desc = hyle::compute::TensorDesc;
using Type = hyle::compute::DataType;

[[noreturn]] void fail(std::string_view name, std::string_view reason) {
  throw std::runtime_error("invalid projection " + std::string(name) + ": " + std::string(reason));
}
LoadedWeight *find(std::map<std::string, LoadedWeight> &bank, const std::string &name) {
  const auto found = bank.find(name);
  return found == bank.end() ? nullptr : &found->second;
}
Tensor &require(std::map<std::string, LoadedWeight> &bank, const std::string &name,
                std::string_view logical_name) {
  if (auto *weight = find(bank, name)) return weight->tensor;
  fail(logical_name, "missing physical role " + name);
}
void matrix(const Tensor &tensor, std::string_view name) {
  const auto &d = tensor.description;
  if (!d.valid() || d.rank() != 2 || d.dimensions()[0] <= 0 || d.dimensions()[1] <= 0)
    fail(name, "principal needs a positive rank-two descriptor");
}
std::int64_t packed_width(std::int64_t packed, std::int64_t factor,
                          std::string_view name) {
  if (packed <= 0 || packed > (std::numeric_limits<std::int64_t>::max)() / factor)
    fail(name, "packed K overflows");
  return packed * factor;
}
std::uint32_t group_size(const LoadedWeight &codes, std::int64_t k, std::int64_t groups,
                         std::string_view name) {
  if (groups <= 0 || k % groups || k / groups > UINT32_MAX)
    fail(name, "scale groups do not partition K");
  const auto result = static_cast<std::uint32_t>(k / groups);
  if (codes.group_size && codes.group_size != result)
    fail(name, "source group annotation differs from physical geometry");
  return result;
}
void fixed_group(const LoadedWeight &codes, std::uint32_t group, std::string_view name) {
  if (codes.group_size && codes.group_size != group)
    fail(name, "source group annotation differs from the typed equation");
}
void auxiliaries(std::map<std::string, LoadedWeight> &bank, const std::string &base,
                 std::initializer_list<std::string_view> allowed, std::string_view name) {
  // Recognized auxiliaries of another representation must not silently select
  // the first matching branch. Source inventory validation owns unknown names.
  constexpr std::array<std::string_view, 17> known{
      ".qzeros", ".scales", ".g_idx", ".weight_scale", ".weight_shape",
      ".weight_global_scale", ".input_global_scale", ".weight_scale_2",
      ".input_scale", ".weight_scale_inv", ".weight_zero_point",
      ".weight.absmax", ".weight.nested_absmax", ".weight.nested_quant_map",
      ".weight.nf4_offset", ".weight.quant_map", ".biases"};
  for (const auto suffix : known) if (bank.contains(base + std::string(suffix)) &&
      std::ranges::find(allowed, suffix) == allowed.end())
    fail(name, "unexpected physical auxiliary " + base + std::string(suffix));
}
Desc activation(std::int64_t k, std::string_view name) {
  auto result = Desc::dense(Type::kBFloat16, std::array<std::int64_t, 2>{1, k});
  if (!result) fail(name, result.error().message());
  return *result;
}
template<class Equation, std::size_t N>
void validate(const Equation &equation, const std::array<Desc, N> &inputs,
               std::string_view name) {
  std::array<Desc, 1> outputs;
  const auto status = equation.infer_output_descriptions(inputs, outputs);
  if (!status.ok()) fail(name, status.message());
}
std::array<std::int64_t, 2> logical_shape(const ProjectionBinding &binding) {
  return std::visit([](const auto &value) -> std::array<std::int64_t, 2> {
    using Entry = std::remove_cvref_t<decltype(value)>;
    if constexpr (std::is_same_v<Entry, StoredProjection>) {
      const auto d = value.weight->description.dimensions(); return {d[0], d[1]};
    } else if constexpr (std::is_same_v<Entry, GptqProjection>) {
      const auto d = value.codes->description.dimensions(); return {d[1], d[0] * 8};
    } else if constexpr (std::is_same_v<Entry, MlxAffineProjection>) {
      const auto d = value.scales->description.dimensions(); return {d[0], d[1] * value.equation.group_size};
    } else if constexpr (std::is_same_v<Entry, Nf4Projection>) {
      const auto d = value.original[0]->description.dimensions(); return {d[0], d[1] * 2};
    } else if constexpr (std::is_same_v<Entry, CtPackedProjection>) {
      const auto d = value.codes->description.dimensions(); return {d[0], d[1] * 8};
    } else if constexpr (std::is_same_v<Entry, ModelOptNvfp4Projection> ||
                         std::is_same_v<Entry, LocalNvfp4Projection>) {
      const auto d = value.codes->description.dimensions(); return {d[0], d[1] * 2};
    } else if constexpr (std::is_same_v<Entry, BlockFp8Projection>) {
      const auto d = value.carrier->description.dimensions(); return {d[1], d[0]};
    } else {
      static_assert(std::is_same_v<Entry, ChannelTokenFp8Projection> ||
                    std::is_same_v<Entry, ModelOptStaticFp8Projection> ||
                    std::is_same_v<Entry, CtBlock128Projection>,
                    "missing logical shape for projection binding");
      const auto d = value.codes->description.dimensions(); return {d[0], d[1]};
    }
  }, binding);
}
} // namespace projection_detail

ProjectionBinding resolve_projection(std::map<std::string, LoadedWeight> &bank,
                                     std::string_view logical_name) {
  namespace detail = projection_detail;
  namespace ops = hyle::compute::model_ops;
  using Type = hyle::compute::DataType;
  using Codec = prepared::ProjectionCodec;
  if (!logical_name.ends_with(".weight") || logical_name.size() <= 7)
    detail::fail(logical_name, "logical name must end in .weight");
  const std::string name(logical_name), base(logical_name.substr(0, logical_name.size() - 7));
  auto *stored = detail::find(bank, name);
  auto *gptq = detail::find(bank, base + ".qweight");
  auto *packed = detail::find(bank, base + ".weight_packed");
  const auto principals = static_cast<unsigned>(stored != nullptr) +
      static_cast<unsigned>(gptq != nullptr) + static_cast<unsigned>(packed != nullptr);
  if (principals != 1) detail::fail(name, principals ? "ambiguous physical principals" : "missing physical principal");
  auto &principal = *(stored ? stored : (gptq ? gptq : packed));
  detail::matrix(principal.tensor, name);

  if (gptq) {
    if (gptq->codec != Codec::kGptqV1Int4)
      detail::fail(name, "qweight is not an identified original GPTQ group");
    detail::auxiliaries(bank, base, {".qzeros", ".scales", ".g_idx"}, name);
    auto &zeros = detail::require(bank, base + ".qzeros", name);
    auto &scales = detail::require(bank, base + ".scales", name);
    auto &groups = detail::require(bank, base + ".g_idx", name);
    detail::matrix(scales, name);
    const auto k = detail::packed_width(gptq->tensor.description.dimensions()[0], 8, name);
    const auto group = detail::group_size(*gptq, k, scales.description.dimensions()[0], name);
    const ops::ArtifactWeightDecode decode{ops::WeightArtifactFormat::kGptqV1Int4, group, Type::kFloat16};
    detail::validate(decode, std::array{gptq->tensor.description, zeros.description, scales.description, groups.description}, name);
    return GptqProjection{&gptq->tensor, &zeros, &scales, &groups, decode};
  }
  if (packed) {
    if (packed->codec == Codec::kCtLocalNvfp4 || packed->codec == Codec::kCtLocalNvfp4Bfloat16) {
      detail::auxiliaries(bank, base, {".weight_scale", ".weight_global_scale", ".input_global_scale"}, name);
      auto &scales = detail::require(bank, base + ".weight_scale", name);
      auto &weight_global = detail::require(bank, base + ".weight_global_scale", name);
      auto &input_global = detail::require(bank, base + ".input_global_scale", name);
      const auto k = detail::packed_width(packed->tensor.description.dimensions()[1], 2, name);
      detail::fixed_group(*packed, 16, name);
      const ops::LocalNvfp4Linear equation{packed->codec == Codec::kCtLocalNvfp4Bfloat16 ?
          ops::Fp4InputRounding::kBFloat16 : ops::Fp4InputRounding::kFloat32};
      detail::validate(equation, std::array{detail::activation(k, name), packed->tensor.description,
          scales.description, weight_global.description, input_global.description}, name);
      return LocalNvfp4Projection{&packed->tensor, &scales, &weight_global, &input_global, equation};
    }
    const bool asymmetric = packed->codec == Codec::kCtAsymmetric4;
    if (!asymmetric && packed->codec != Codec::kCtSymmetric4)
      detail::fail(name, "weight_packed is not an identified CT group");
    if (asymmetric) detail::auxiliaries(bank, base, {".weight_scale", ".weight_zero_point", ".weight_shape"}, name);
    else detail::auxiliaries(bank, base, {".weight_scale", ".weight_shape"}, name);
    auto &scales = detail::require(bank, base + ".weight_scale", name);
    detail::matrix(scales, name);
    const auto k = detail::packed_width(packed->tensor.description.dimensions()[1], 8, name);
    const auto group = detail::group_size(*packed, k, scales.description.dimensions()[1], name);
    if (asymmetric && (group != 32 || scales.description.data_type() != Type::kBFloat16))
      detail::fail(name, "original asymmetric CT requires group32 and BF16 scales");
    const ops::ArtifactWeightDecode decode{asymmetric ? ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 :
        ops::WeightArtifactFormat::kCompressedTensorsSymmetric4, group, Type::kBFloat16};
    Tensor *zeros = nullptr;
    if (asymmetric) {
      zeros = &detail::require(bank, base + ".weight_zero_point", name);
      detail::validate(decode, std::array{packed->tensor.description, zeros->description, scales.description}, name);
      (void)detail::require(bank, base + ".weight_shape", name);
    } else detail::validate(decode, std::array{packed->tensor.description, scales.description}, name);
    // weight_shape is an authenticated source role, not an equation operand.
    // Full loader validation already checks its actual I64 [N,K] payload.
    if (const auto *shape = detail::find(bank, base + ".weight_shape")) {
      const auto expected = hyle::compute::TensorDesc::dense(Type::kInt64, std::array<std::int64_t, 1>{2});
      if (!expected || shape->tensor.description != *expected) detail::fail(name, "invalid CT shape auxiliary descriptor");
    }
    return CtPackedProjection{&packed->tensor, zeros, &scales, decode};
  }

  if (stored->codec == Codec::kMlxAffine) {
    detail::auxiliaries(bank, base, {".scales", ".biases"}, name);
    auto &scales = detail::require(bank, base + ".scales", name);
    auto &biases = detail::require(bank, base + ".biases", name);
    detail::matrix(scales, name);
    const auto group = stored->group_size;
    if (group != 64 || scales.description.data_type() != Type::kBFloat16)
      detail::fail(name, "selected MLX source requires original BF16 affine group64");
    const auto k = detail::packed_width(scales.description.dimensions()[1], group, name);
    const auto bits_numerator = detail::packed_width(stored->tensor.description.dimensions()[1], 32, name);
    if (bits_numerator % k || (bits_numerator / k != 4 && bits_numerator / k != 8))
      detail::fail(name, "selected MLX packed width requires exactly four or eight bits");
    const ops::MlxQuantized equation{.format = ops::MlxWeightFormat::kAffine,
        .kind = ops::MlxQuantizedKind::kLinear,
        .bits = static_cast<std::uint32_t>(bits_numerator / k), .group_size = group};
    detail::validate(equation, std::array{detail::activation(k, name), stored->tensor.description,
        scales.description, biases.description}, name);
    return MlxAffineProjection{&stored->tensor, &scales, &biases, equation};
  }
  if (stored->codec == Codec::kModelOptStaticFp8) {
    detail::auxiliaries(bank, base, {".weight_scale", ".input_scale"}, name);
    auto &sa = detail::require(bank, base + ".input_scale", name);
    auto &sw = detail::require(bank, base + ".weight_scale", name);
    const auto &codes = stored->tensor.description;
    const auto scalar = [](const Tensor &value) {
      const auto &d = value.description;
      return d.is_plain() && d.is_dense() && d.data_type() == Type::kFloat32 &&
          (d.rank() == 0 || (d.rank() == 1 && d.dimensions()[0] == 1));
    };
    if (!codes.is_plain() || !codes.is_dense() || codes.data_type() != Type::kFloat8E4M3FN ||
        stored->group_size || !scalar(sa) || !scalar(sw))
      detail::fail(name, "static FP8 requires original dense E4 [N,K] and separate F32 scalar SA/SW");
    // Metadata/explicit reader chooses this codec; never inspect scalar values
    // or infer this math just because a weight has an FP8 storage name.
    return ModelOptStaticFp8Projection{&stored->tensor, &sa, &sw};
  }
  if (stored->codec == Codec::kNvfp4) {
    detail::auxiliaries(bank, base, {".weight_scale", ".weight_scale_2", ".input_scale"}, name);
    auto &scales = detail::require(bank, base + ".weight_scale", name);
    auto &weight_global = detail::require(bank, base + ".weight_scale_2", name);
    auto &input_global = detail::require(bank, base + ".input_scale", name);
    const auto k = detail::packed_width(stored->tensor.description.dimensions()[1], 2, name);
    detail::fixed_group(*stored, 16, name);
    detail::validate(ops::Nvfp4Linear{}, std::array{detail::activation(k, name), stored->tensor.description,
        scales.description, weight_global.description, input_global.description}, name);
    return ModelOptNvfp4Projection{&stored->tensor, &scales, &weight_global, &input_global};
  }
  if (stored->codec == Codec::kFloat8Block128) {
    // load_weight_bank has already combined original codes/scales into the
    // exact encoded [K,N] carrier. Do not look up a second resident scale bank.
    detail::auxiliaries(bank, base, {}, name);
    detail::fixed_group(*stored, 128, name);
    const ops::Float8BlockLinear equation{ops::Float8ScalePolicy::kDivide, prepared::Float8BlockPolicy::floor};
    detail::validate(equation, std::array{detail::activation(stored->tensor.description.dimensions()[0], name), stored->tensor.description}, name);
    return BlockFp8Projection{&stored->tensor, equation};
  }
  if (stored->codec == Codec::kCtFloat8Block128) {
    detail::auxiliaries(bank, base, {".weight_scale"}, name);
    auto &scales = detail::require(bank, base + ".weight_scale", name);
    const auto &codes = stored->tensor.description;
    detail::fixed_group(*stored, 128, name);
    const auto n = codes.dimensions()[0], k = codes.dimensions()[1];
    const auto expected = c::TensorDesc::dense(Type::kBFloat16,
        std::array<std::int64_t, 2>{(n - 1) / 128 + 1, k / 128});
    if (!codes.is_plain() || !codes.is_dense() || codes.data_type() != Type::kFloat8E4M3FN ||
        k % 128 || (n - 1) / 128 > INT32_MAX || !expected || scales.description != *expected)
      detail::fail(name, "CT block128 requires original dense E4 [N,K] and BF16 [ceil(N/128),K/128]");
    return CtBlock128Projection{&stored->tensor, &scales};
  }
  if (stored->codec == Codec::kCtFloat8ChannelToken) {
    detail::auxiliaries(bank, base, {".weight_scale"}, name);
    auto &scales = detail::require(bank, base + ".weight_scale", name);
    const auto &codes = stored->tensor.description;
    const auto expected_scales = hyle::compute::TensorDesc::dense(Type::kBFloat16,
        std::array<std::int64_t, 2>{codes.dimensions()[0], 1});
    if (!codes.is_plain() || !codes.is_dense() || codes.data_type() != Type::kFloat8E4M3FN ||
        !expected_scales || scales.description != *expected_scales || stored->group_size)
      detail::fail(name, "channel/token FP8 requires dense E4 [N,K] and original BF16 [N,1] scales");
    return ChannelTokenFp8Projection{&stored->tensor, &scales};
  }
  if (stored->codec == Codec::kNf4DoubleQuant) {
    detail::auxiliaries(bank, base, {".weight.absmax", ".weight.nested_absmax",
        ".weight.nested_quant_map", ".weight.nf4_offset", ".weight.quant_map"}, name);
    detail::fixed_group(*stored, 64, name);
    std::array<Tensor *, 6> original{&stored->tensor,
        &detail::require(bank, name + ".absmax", name),
        &detail::require(bank, name + ".nested_absmax", name),
        &detail::require(bank, name + ".nested_quant_map", name),
        &detail::require(bank, name + ".nf4_offset", name),
        &detail::require(bank, name + ".quant_map", name)};
    const ops::ArtifactWeightDecode decode{ops::WeightArtifactFormat::kNf4DoubleQuant, 64, Type::kFloat16};
    std::array<detail::Desc, 6> descriptions;
    for (std::size_t i = 0; i < original.size(); ++i) descriptions[i] = original[i]->description;
    detail::validate(decode, descriptions, name);
    return Nf4Projection{original, decode};
  }
  if (stored->codec != Codec::kDense && stored->codec != Codec::kUnspecified)
    detail::fail(name, "stored principal has an incompatible source codec");
  detail::auxiliaries(bank, base, {}, name);
  if (stored->group_size) detail::fail(name, "stored floating/GGUF weight has a packed group annotation");
  const auto &description = stored->tensor.description;
  if (!hyle::compute::gguf_quantized_type(description.storage_format())) {
    if (!description.is_plain() || !description.is_dense() ||
        (description.data_type() != Type::kBFloat16 && description.data_type() != Type::kFloat16 &&
         description.data_type() != Type::kFloat32))
      detail::fail(name, "stored weight is neither dense BF16/F16/F32 nor original GGUF storage");
  } else {
    // GGUF bank descriptors are the output-major transpose of the public
    // compressed [K,N] description; validate it without expanding its payload.
    const auto original = hyle::compute::gguf_weight_description(
        *hyle::compute::gguf_quantized_type(description.storage_format()), Type::kFloat32,
        description.dimensions()[1], description.dimensions()[0]);
    if (!original) detail::fail(name, original.error().message());
    const auto output_major = original->transposed();
    if (!output_major || description != *output_major)
      detail::fail(name, "GGUF weight differs from its original encoded output-major layout");
  }
  return StoredProjection{&stored->tensor};
}

ProjectionBindings prepare_projection_bindings(std::map<std::string, LoadedWeight> &bank,
                                                bool mtp, const ModelGeometry &geometry) {
  const auto inventory = prepared::parameter_inventory(mtp, geometry);
  ProjectionBindings bindings;
  const auto add = [&](const std::string &name, const prepared::Parameter &parameter) {
    auto binding = resolve_projection(bank, name);
    const auto shape = projection_detail::logical_shape(binding);
    if (parameter.shape.size() != 2 || shape[0] != parameter.shape[0] || shape[1] != parameter.shape[1])
      projection_detail::fail(name, "logical projection shape differs from source parameter inventory");
    bindings.emplace(name, std::move(binding));
  };
  for (const auto &[name, parameter] : inventory) if (parameter.linear) add(name, parameter);
  // A quantized embedding must have a cold binding even with an untied head.
  const std::string embedding("model.language_model.embed_tokens.weight");
  if (bank.at(embedding).codec == prepared::ProjectionCodec::kMlxAffine && !bindings.contains(embedding))
    add(embedding, inventory.at(embedding));
  const std::string head(geometry.head_role());
  if (!bindings.contains(head)) {
    const auto parameter = inventory.find(head);
    if (parameter == inventory.end()) projection_detail::fail(head, "head is absent from the source inventory");
    add(head, parameter->second);
  }
  return bindings;
}

std::string_view score_policy(const ProjectionBinding &binding, PrecisionProfile precision) noexcept {
  return std::visit([precision](const auto &entry) noexcept -> std::string_view {
    using Entry = std::remove_cvref_t<decltype(entry)>;
    if constexpr (std::is_same_v<Entry, StoredProjection>) {
      const auto *stored = &entry;
      if (!stored->weight) return "invalid_projection_binding";
      const auto &description = stored->weight->description;
      if (hyle::compute::gguf_quantized_type(description.storage_format()))
        return "gguf_encoded_weight_projection_f32_logits";
      if (precision.is_f16()) return "f16_inputs_f16_projection_f32_logits";
      if (description.data_type() == hyle::compute::DataType::kFloat16) return "f16_weight_f32_inputs_f32_output";
      if (description.data_type() == hyle::compute::DataType::kFloat32) return "f32_weight_f32_inputs_f32_output";
      return "bf16_inputs_f32_output";
    } else if constexpr (std::is_same_v<Entry, MlxAffineProjection>) {
      return "mlx_affine_f32_weight_dot_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, Nf4Projection>) {
      return "nf4_double_quant_f16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, GptqProjection>) {
      return "gptq_f16_weight_to_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, CtPackedProjection>) {
      const auto *projection = &entry;
      return projection->decode.format == hyle::compute::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4
          ? "ct_asym4_bf16_subtract_product_projection_f32_logits" : "ct_sym4_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, ModelOptNvfp4Projection>) {
      return "nvfp4_qdq_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, LocalNvfp4Projection>) {
      return entry.equation.rounding == hyle::compute::model_ops::Fp4InputRounding::kBFloat16
          ? "ct_local_nvfp4_bf16_preround_bf16_projection_f32_logits"
          : "ct_local_nvfp4_f32_preround_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, ModelOptStaticFp8Projection>) {
      return "static_fp8_stored_SA_SW_f32_code_dot_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, BlockFp8Projection>) {
      return "fp8_block128_dynamic_divide_bf16_projection_f32_logits";
    } else if constexpr (std::is_same_v<Entry, CtBlock128Projection>) {
      return "ct_block128_bf16_qdq_projection_f32_logits";
    } else {
      static_assert(std::is_same_v<Entry, ChannelTokenFp8Projection>,
                    "missing score policy for projection binding");
      return "ct_channel_token_bf16_qdq_projection_f32_logits";
    }
  }, binding);
}

struct StateSchema final {
  c::TensorDesc history, recurrent;
};
std::vector<StateSchema> state_schema_for(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision = {}) {
  geometry.validate();
  if (capacity == 0 || capacity > static_cast<std::size_t>(geometry.context_limit)) throw std::runtime_error("capacity outside model context limit");
  std::vector<StateSchema> result(geometry.layers.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    auto &layer = result[i];
    if (geometry.is_gdn(i)) {
      layer.history = dense(precision.activation_type(), {1, geometry.conv_kernel - 1, geometry.gdn_qkv_width()});
      layer.recurrent = dense(Type::kFloat32, {1, geometry.gdn_value_heads, geometry.gdn_key_depth, geometry.gdn_value_depth});
    }
  }
  return result;
}
std::size_t recurrent_bytes(std::span<const StateSchema> schema, const ModelGeometry &geometry) {
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
struct PackedProjectionNodes final {
  std::array<g::NodeId, 4> chain;
  std::size_t count = 4;
};
using WeightMeanings = std::map<std::string, prepared::ParameterMeaning>;
mem::MemoryResourceRef weight_resource(const Backend &backend, bool host) {
  if (!host) return backend.device();
  if (backend.has_cpu_assist()) return backend.host_device();
  throw std::runtime_error("CPU FFN placement requires a configured CPU plus GPU endpoint");
}
void validate_mtp_source(const ModelGeometry &geometry, bool enabled) {
  if (enabled && geometry.mtp_layers != 1)
    throw std::runtime_error("MTP requires one NextN layer in the prepared source geometry");
}
BackendSelection select_execution(BackendSelection selection, std::size_t cpu_layers) {
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
Backend create_execution(const ModelGeometry &geometry, const cache::Plan &plan,
    std::size_t cpu_layers, BackendSelection selection) {
  geometry.validate();
  validate_mtp_source(geometry, plan.draft_cache);
  FfnPlacement::validate(cpu_layers, geometry.layers.size());
  selection = select_execution(selection, cpu_layers);
  if (selection.primary != BackendKind::kCpu && !cpu_layers) return take(Backend::create(selection));
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
    return take(Backend::create(selection, {.host = compute}));
  }
  const auto staging = cache::add(64u << 20,
      cache::multiply(cache::multiply(rows, cache::multiply(static_cast<std::size_t>(geometry.hidden), 8)), cpu_layers));
  std::cerr << "hybrid_workspace_admission device=" << compute << " host=" << compute
      << " staging=" << staging << " scope=known_per_invocation_workspace native_private_unknown=1 physical_vram_cap=0\n";
  return take(Backend::create(selection, {compute, compute, staging}));
}
// Cold graph resources and placement, independent of the model equations.
// The original prepared graph owns the moved literals; Backend outlives both.
struct GraphAssembly final {
  struct Location final {
    g::ExecutionTargetId target;
    g::StorageDomainId storage;
  };
  struct Scope final {
    GraphAssembly &assembly;
    Location previous;
    Scope(GraphAssembly &selected, Location location)
        : assembly(selected), previous(selected.location_) { assembly.location_ = location; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { assembly.location_ = previous; }
  };

  std::vector<g::ConstantBinding> constants;
  std::vector<Tensor> literals;
  std::vector<g::NodePlacement> node_placements;
  std::vector<g::ExternalResourceContract> external_resources;

  explicit GraphAssembly(const Backend &environment, PrecisionProfile precision = {},
                         const c::ResourceBudget &budget = {})
      : backend_(environment), budget_(budget), precision_(precision), mixed_(environment.has_cpu_assist()),
        predicate_type_(environment.native_generated_storage() ? Type::kFloat32 : Type::kBool) {}
  GraphAssembly(const GraphAssembly &) = delete;
  GraphAssembly &operator=(const GraphAssembly &) = delete;
  static Location primary() { return {g::ExecutionTargetId{0}, g::StorageDomainId{0}}; }
  Location cpu() const {
    if (!mixed_) throw std::runtime_error("CPU FFN requires the configured CPU endpoint");
    return {g::ExecutionTargetId{1}, g::StorageDomainId{1}};
  }
  Tensor &literal_at(Location location, c::TensorDesc description) {
    literals.emplace_back(take(backend_.resource_for(location.storage)), description, true, budget_);
    return literals.back();
  }
  Tensor &literal(c::TensorDesc description) { return literal_at(location_, description); }
  Type predicate_type() const { return predicate_type_; }
  Type activation_type() const noexcept { return precision_.activation_type(); }
  void node(g::NodeId node) {
    if (mixed_) node_placements.push_back({node, location_.target, g::PlacementStrength::kRequired});
  }
  void external(V value, g::StorageDomainId storage = g::StorageDomainId{0}) {
    if (!mixed_) return;
    const auto found = std::ranges::find(external_resources, value, &g::ExternalResourceContract::value);
    if (found == external_resources.end()) external_resources.push_back({value, storage});
    else if (found->storage != storage) throw std::runtime_error("conflicting external tensor resource contracts");
  }
  g::CompileOptions options(g::CompileOptions options) const {
    options.node_placements = node_placements; options.external_resources = external_resources; return options;
  }
  void bind_constant(V value, Tensor &tensor) {
    tensor.freeze(backend_);
    const auto read = tensor.read();
    const auto owner = take(backend_.storage_owner_for(read.storage().resource()));
    if (mixed_) external(value, g::StorageDomainId{read.storage().resource() == backend_.device() ? 0u : 1u});
    // Native resources are explicitly borrowed from Model's Backend. The
    // bank/literal owns its lease through every compiled child. A native lease
    // requires its actual StorageOwner; only process-system storage can be
    // retained without one. Graph validates that allocator identity separately.
    const auto retained = owner || read.storage().resource() == mem::system_resource()
        ? tensor.ready : c::ReadLease{};
    constants.push_back({value, read, g::ConstantPolicy::kBorrow, owner, retained});
  }

private:
  const Backend &backend_;
  const c::ResourceBudget budget_;
  const PrecisionProfile precision_;
  const bool mixed_;
  const Type predicate_type_;
  Location location_ = primary();
};

struct Recipe final {
  g::GraphBuilder builder = take(g::GraphBuilder::create());
  std::vector<g::TensorSpec> specs;
  std::map<Tensor *, V> operands;
  std::vector<PackedProjectionNodes> packed_projection_nodes;
  std::map<std::string, LoadedWeight> &bank;
  const ProjectionBindings &projections;
  const WeightMeanings *meanings = nullptr;
  GraphAssembly &assembly;
  const ModelGeometry &geometry;
  const std::int64_t hidden;
  g::DimensionId last_row;
  D token_rows, packed_rows, sample_rows;
  Recipe(std::map<std::string, LoadedWeight> &selected, const ProjectionBindings &selected_projections,
         GraphAssembly &graph_assembly, const ModelGeometry &model_geometry,
         std::size_t max_query_rows, std::size_t requests = 1,
         const WeightMeanings *parameter_meanings = nullptr)
      : bank(selected), projections(selected_projections), meanings(parameter_meanings), assembly(graph_assembly), geometry(model_geometry), hidden(model_geometry.hidden) {
    // L is a zero-based last-token position, independently bound at startup.
    // T=L+1 avoids subtraction or a second independent (possibly inconsistent)
    // shape binding. All other numeric axes remain fixed model dimensions.
    last_row = take(builder.dimension({0, static_cast<std::int64_t>(max_query_rows - 1), 1}));
    token_rows = D(take(builder.dimension_sum(D(last_row), f(1))));
    packed_rows = requests == 1 ? token_rows :
        D(take(builder.dimension_product(f(static_cast<std::int64_t>(requests)), token_rows)));
    sample_rows = D(take(builder.dimension_sum(token_rows, f(geometry.conv_kernel - 1))));
  }
  static g::TensorSpec dense_spec(Type type, std::span<const D> dimensions) {
    return take(g::TensorSpec::dense(type, dimensions));
  }
  static g::TensorSpec dense_spec(Type type, std::initializer_list<D> dimensions) {
    return dense_spec(type, std::span(dimensions.begin(), dimensions.size()));
  }
  Type activation_type() const noexcept { return assembly.activation_type(); }
  const g::TensorSpec &spec(V value) const { return specs.at(value.ordinal()); }
  V remember(V value, g::TensorSpec description) {
    if (specs.size() <= value.ordinal()) specs.resize(value.ordinal() + 1);
    specs[value.ordinal()] = description;
    return value;
  }
  V input(g::TensorSpec description) {
    const auto value = remember(take(builder.input(description)), description); assembly.external(value); return value;
  }
  V input(c::TensorDesc description) { return input(take(g::TensorSpec::concrete(description))); }
  V constant(Tensor &tensor) {
    const auto value = remember(take(builder.constant(tensor.description)),
                                take(g::TensorSpec::concrete(tensor.description)));
    assembly.bind_constant(value, tensor);
    return value;
  }
  V operand(Tensor &tensor) {
    const auto found = operands.find(&tensor);
    if (found != operands.end()) return found->second;
    const auto value = constant(tensor);
    operands.emplace(&tensor, value);
    return value;
  }
  V weight_absolute(const std::string &name) {
    const auto original = bank.find(name);
    if (original == bank.end()) throw std::runtime_error("missing original model role: " + name);
    return operand(original->second.tensor);
  }
  V weight(const std::string &name) {
    return weight_absolute("model.language_model." + name);
  }
  prepared::ParameterMeaning meaning(const std::string &name) const {
    if (meanings) if (const auto found = meanings->find(name); found != meanings->end()) return found->second;
    return prepared::ParameterMeaning::kHfRaw;
  }
  V embedding(V tokens, const std::string &name) {
    const auto ids = spec(tokens);
    if (const auto found = projections.find(name); found != projections.end())
      if (const auto *mlx = std::get_if<MlxAffineProjection>(&found->second)) {
        if (activation_type() != Type::kBFloat16 || ids.rank() != 1 || ids.data_type() != Type::kInt32)
          throw std::runtime_error("MLX source embedding requires BF16 and checked I32 rows");
        const auto shape = projection_detail::logical_shape(found->second);
        if (shape[1] != hidden) throw std::runtime_error("MLX embedding logical width differs");
        // The embedding has typed BF16 product/add boundaries. Its tied head
        // reuses these original operands as a linear operation, not this table.
        return one(c::model_ops::mlx_quantized_embedding(
            mlx->equation.bits, mlx->equation.group_size),
            {operand(*mlx->codes), operand(*mlx->scales), operand(*mlx->biases), tokens},
            dense_spec(Type::kBFloat16, {ids.dimensions()[0], f(hidden)}));
      }
    const auto original = weight_absolute(name);
    const auto stored = spec(original);
    if (ids.rank() != 1 || ids.data_type() != Type::kInt32 || stored.rank() != 2 ||
        stored.dimensions()[1] != f(hidden))
      throw std::runtime_error("embedding requires I32 rows and original [vocabulary,hidden] storage: " + name);
    // Gather decodes selected original rows only, then applies the model's
    // activation boundary. The shared vocabulary bank is never expanded.
    // Encoded row decoders carry checked IDs as part of their storage contract.
    const auto bounds = c::gguf_quantized_type(stored.storage_format())
        ? c::IndexBoundsPolicy::kChecked : c::IndexBoundsPolicy::kUnchecked;
    return cast(one(c::Gather{.bounds = bounds}, {original, tokens},
        dense_spec(stored.data_type(), {ids.dimensions()[0], stored.dimensions()[1]})), activation_type());
  }
  V scalar(float value, Type type = Type::kFloat32) {
    auto &literal = assembly.literal(dense(type, {1}));
    if (type == Type::kFloat32) literal.write<float>(std::span(&value, 1));
    else if (type == Type::kFloat16) {
      const auto f16 = hyle::simd::f32_to_f16(value);
      literal.write<hyle::simd::float16_t>(std::span(&f16, 1));
    } else if (type == Type::kBFloat16) {
      auto bits = std::bit_cast<std::uint32_t>(value);
      bits += 0x7fffU + ((bits >> 16) & 1U);
      const auto bf16 = static_cast<std::uint16_t>(bits >> 16);
      literal.write<std::uint16_t>(std::span(&bf16, 1));
    } else throw std::runtime_error("floating scalar requires F32/F16/BF16");
    return constant(literal);
  }
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    std::initializer_list<g::TensorSpec> outputs,
                    g::NodeId *node_id = nullptr) {
    const auto node = take(builder.add_node(operation,
        std::span(arguments.begin(), arguments.size()),
        std::span(outputs.begin(), outputs.size())));
    assembly.node(node);
    if (node_id) *node_id = node;
    std::vector<V> result;
    std::size_t index = 0;
    for (const auto &output : outputs)
      result.push_back(remember(take(builder.result(node, index++)), output));
    return result;
  }
  // Only concrete constant transforms use the Compute-only inference shortcut.
  // Symbolic operations declare their real logical outputs; each exact child
  // still runs the authoritative Compute inference during Graph compilation.
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments,
                    g::NodeId *node_id = nullptr) {
    std::vector<c::TensorDesc> in;
    for (auto value : arguments) in.push_back(take(spec(value).concrete_description()));
    std::vector<c::TensorDesc> out(c::operation_traits(operation).outputs);
    check(c::infer_output_descriptions(operation, in, out));
    const auto node = take(builder.add_node(operation, std::span(arguments.begin(), arguments.size())));
    assembly.node(node);
    if (node_id) *node_id = node;
    std::vector<V> result;
    for (std::size_t i = 0; i < out.size(); ++i)
      result.push_back(remember(take(builder.result(node, i)), take(g::TensorSpec::concrete(out[i]))));
    return result;
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, node_id).at(0);
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments,
                          g::TensorSpec output, g::NodeId *node_id = nullptr) {
    return op(c::Operation(operation), arguments, {output}, node_id).at(0);
  }
  template<class O> V same(O operation, std::initializer_list<V> arguments, V source) {
    return one(operation, arguments, dense_spec(spec(source).data_type(), spec(source).dimensions()));
  }
  V cast(V x, Type type) {
    return spec(x).data_type() == type ? x :
        one(c::Cast{type}, {x}, dense_spec(type, spec(x).dimensions()));
  }
  V host_predicate(V value) {
    return cast(value, assembly.predicate_type());
  }
  V unary(c::UnaryKind kind, V x) { return same(c::Unary{kind}, {x}, x); }
  V binary(c::BinaryKind kind, V x, V y) {
    const auto left = spec(x), right = spec(y);
    if (left.data_type() != right.data_type()) throw std::runtime_error("binary dtype mismatch");
    const auto rank = std::max(left.rank(), right.rank());
    std::vector<D> dimensions(rank, f(1));
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto a = axis + left.rank() < rank ? f(1) : left.dimensions()[axis + left.rank() - rank];
      const auto b = axis + right.rank() < rank ? f(1) : right.dimensions()[axis + right.rank() - rank];
      if (a == b || b == f(1)) dimensions[axis] = a;
      else if (a == f(1)) dimensions[axis] = b;
      else dimensions[axis] = D(take(builder.dimension_broadcast(a, b)));
    }
    return one(c::Binary{kind}, {x, y}, dense_spec(left.data_type(), dimensions));
  }
  V reshape(V x, std::initializer_list<D> dimensions) {
    // A fixed constant may still need the original stride-chunk Copy. Dynamic
    // views preserve inference and receive explicit Copy only where the actual
    // model layout requires it (the self-attention interleaved gate below).
    bool fixed = true;
    std::vector<std::int64_t> concrete;
    for (auto dimension : dimensions) {
      fixed = fixed && dimension.is_fixed();
      concrete.push_back(dimension.fixed_value());
    }
    if (auto input = spec(x).concrete_description(); input && fixed) {
      auto view = take(c::View::reshape(take(c::TensorShape::create(concrete))));
      auto inferred = c::infer_single_output(c::Operation(view), std::array{*input});
      if (!inferred && inferred.error().IsUnimplemented()) x = copy(x);
    }
    const auto view = take(g::ViewSpec::reshape(std::span(dimensions.begin(), dimensions.size())));
    const auto node = take(builder.add_node(view, x)); assembly.node(node);
    const auto output = take(builder.result(node));
    return remember(output, take(g::TensorSpec::inferred(spec(x).data_type(),
        std::span(dimensions.begin(), dimensions.size()), spec(x).storage_format())));
  }
  V copy(V x) { return same(c::Copy{}, {x}, x); }
  V concat(std::span<const V> inputs, int axis, g::TensorSpec output) {
    if (inputs.size() == 1) return inputs.front();
    const std::array<g::TensorSpec, 1> outputs{output};
    const auto node = take(builder.add_node(c::Operation(c::Concat{.axis = axis}), inputs, outputs));
    assembly.node(node);
    return remember(take(builder.result(node)), std::move(output));
  }
  V slice(V x, int axis, D first, D count) {
    const auto source = spec(x);
    std::vector<D> dimensions(source.dimensions().begin(), source.dimensions().end());
    dimensions.at(static_cast<std::size_t>(axis)) = count;
    const auto node = take(builder.add_node(take(g::ViewSpec::slice(axis, first, count)), x)); assembly.node(node);
    const auto output = take(builder.result(node));
    return remember(output, take(g::TensorSpec::inferred(source.data_type(), dimensions,
                                                       source.storage_format())));
  }
  V linear(V x, const std::string &name) {
    return linear_absolute(x, "model.language_model." + name);
  }
  const ProjectionBinding &projection(const std::string &name) const {
    const auto found = projections.find(name);
    if (found == projections.end()) throw std::runtime_error("missing cold model projection binding: " + name);
    return found->second;
  }
  V emit_projection(V x, const ProjectionBinding &binding, bool scores) {
    if (spec(x).rank() != 2 || spec(x).data_type() != activation_type())
      throw std::runtime_error("model projection differs from source activation [rows,K] precision");
    auto dimensions = std::vector<D>(spec(x).dimensions().begin(), spec(x).dimensions().end());
    // Cold bindings already validate the source logical [N,K] layout.
    dimensions.back() = f(projection_detail::logical_shape(binding)[0]);
    const auto result = std::visit([&](const auto &projection) -> V {
      using P = std::decay_t<decltype(projection)>;
      if constexpr (std::is_same_v<P, StoredProjection>) {
        const auto original = operand(*projection.weight);
        const auto stored = spec(original);
        if (stored.rank() != 2 || spec(x).dimensions()[1] != stored.dimensions()[1])
          throw std::runtime_error("original stored projection type/shape differs");
        if (activation_type() == Type::kFloat16) {
          if (stored.data_type() != Type::kFloat16 || !projection.weight->description.is_plain())
            throw std::runtime_error("F16 source projection requires a dense F16 operand");
          constexpr std::array<std::int32_t, 2> axes{1, 0};
          return one(c::MatMul{.accumulation_type = Type::kFloat32},
              {x, one(take(c::View::permute(axes)), {original})}, dense_spec(Type::kFloat16, dimensions));
        }
        if (c::gguf_quantized_type(stored.storage_format())) {
          constexpr std::array<std::int32_t, 2> axes{1, 0};
          const auto rhs = one(take(c::View::permute(axes)), {original});
          const auto output = one(c::MatMul{}, {cast(x, Type::kFloat32), rhs}, dense_spec(Type::kFloat32, dimensions));
          return scores ? output : cast(output, Type::kBFloat16);
        }
        if (scores || stored.data_type() == Type::kFloat16 || stored.data_type() == Type::kFloat32) {
          const auto output = one(c::model_ops::dense_linear_f32(),
              {stored.data_type() == Type::kBFloat16 ? x : cast(x, Type::kFloat32), original},
              dense_spec(Type::kFloat32, dimensions));
          return scores ? output : cast(output, Type::kBFloat16);
        }
        if (stored.data_type() != Type::kBFloat16)
          throw std::runtime_error("original dense projection requires a supported floating operand");
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {original});
        return one(c::MatMul{}, {x, rhs}, dense_spec(Type::kBFloat16, dimensions));
      } else if constexpr (std::is_same_v<P, MlxAffineProjection>) {
        if (activation_type() != Type::kBFloat16) throw std::runtime_error("MLX source projection requires BF16 activation");
        return one(c::Operation::custom(projection.equation),
            {x, operand(*projection.codes), operand(*projection.scales), operand(*projection.biases)},
            dense_spec(Type::kBFloat16, dimensions));
      } else if constexpr (std::is_same_v<P, Nf4Projection>) {
        if (activation_type() != Type::kFloat16) throw std::runtime_error("NF4 source equation requires F16 activation");
        PackedProjectionNodes nodes; nodes.count = 3;
        const auto decoded = one(c::Operation::custom(projection.decode),
            {operand(*projection.original[0]), operand(*projection.original[1]),
             operand(*projection.original[2]), operand(*projection.original[3]),
             operand(*projection.original[4]), operand(*projection.original[5])}, &nodes.chain[0]);
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {decoded}, &nodes.chain[1]);
        const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
            dense_spec(Type::kFloat16, dimensions), &nodes.chain[2]);
        packed_projection_nodes.push_back(nodes);
        return output;
      } else if constexpr (std::is_same_v<P, GptqProjection>) {
        if (activation_type() != Type::kBFloat16) throw std::runtime_error("GPTQ source equation requires BF16 activation");
        const auto codes = operand(*projection.codes), zeros = operand(*projection.zeros);
        const auto scales = operand(*projection.scales), groups = operand(*projection.groups);
        // Preserve F16 decode, BF16 operand and the same four-node fusion region.
        PackedProjectionNodes nodes;
        const auto decoded = one(c::Operation::custom(projection.decode), {codes, zeros, scales, groups}, &nodes.chain[0]);
        const auto converted = one(c::Cast{Type::kBFloat16}, {decoded}, &nodes.chain[1]);
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {converted}, &nodes.chain[2]);
        const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
            dense_spec(Type::kBFloat16, dimensions), &nodes.chain[3]);
        packed_projection_nodes.push_back(nodes);
        return output;
      } else if constexpr (std::is_same_v<P, CtPackedProjection>) {
        if (activation_type() != Type::kBFloat16) throw std::runtime_error("CT packed source equation requires BF16 activation");
        PackedProjectionNodes nodes; nodes.count = 3;
        const auto decoded = projection.zeros ? one(c::Operation::custom(projection.decode),
            {operand(*projection.codes), operand(*projection.zeros), operand(*projection.scales)}, &nodes.chain[0]) :
            one(c::Operation::custom(projection.decode), {operand(*projection.codes), operand(*projection.scales)}, &nodes.chain[0]);
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {decoded}, &nodes.chain[1]);
        const auto output = one(c::MatMul{.accumulation_type = Type::kFloat32}, {x, rhs},
            dense_spec(Type::kBFloat16, dimensions), &nodes.chain[2]);
        packed_projection_nodes.push_back(nodes);
        return output;
      } else if constexpr (std::is_same_v<P, LocalNvfp4Projection>) {
        const auto codes = operand(*projection.codes);
        return one(c::Operation::custom(projection.equation),
            {x, codes, operand(*projection.scales), operand(*projection.weight_global), operand(*projection.input_global)},
            dense_spec(Type::kBFloat16, dimensions));
      } else if constexpr (std::is_same_v<P, ModelOptNvfp4Projection>) {
        const auto codes = operand(*projection.codes);
        return one(c::model_ops::nvfp4_linear(),
            {x, codes, operand(*projection.scales), operand(*projection.weight_global), operand(*projection.input_global)},
            dense_spec(Type::kBFloat16, dimensions));
      } else if constexpr (std::is_same_v<P, BlockFp8Projection>) {
        const auto carrier = operand(*projection.carrier);
        return one(c::Operation::custom(projection.equation), {x, carrier}, dense_spec(Type::kBFloat16, dimensions));
      } else if constexpr (std::is_same_v<P, ModelOptStaticFp8Projection>) {
        if (activation_type() != Type::kBFloat16)
          throw std::runtime_error("selected static FP8 source reader requires BF16 activation");
        const auto original = operand(*projection.codes);
        const auto sa = operand(*projection.activation_scale), sw = operand(*projection.weight_scale);
        // This exact 12-node public graph is the existing StaticFloat8Pattern.
        // Cold selection owns contraction or decomposition; no provider switch,
        // decoded bank, second executor or private Compute operation is added.
        const auto a = cast(x, Type::kFloat32);
        const auto one_value = one(c::Fill{
            .output_shape = take(c::TensorShape::create(projection.activation_scale->description.dimensions())),
            .output_type = Type::kFloat32, .value = c::ScalarValue(1.F)}, {}, spec(sa));
        const auto inverse = binary(c::BinaryKind::kDivide, one_value, sa);
        const auto normalized = binary(c::BinaryKind::kMultiply, a, inverse);
        const auto encoded = cast(normalized, Type::kFloat8E4M3FN);
        const auto q = cast(encoded, Type::kFloat32);
        const auto w = cast(original, Type::kFloat32);
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        const auto rhs = one(take(c::View::permute(axes)), {w});
        const auto dot = one(c::MatMul{.accumulation_type = Type::kFloat32},
            {q, rhs}, dense_spec(Type::kFloat32, dimensions));
        const auto scaled = binary(c::BinaryKind::kMultiply,
            binary(c::BinaryKind::kMultiply, dot, sa), sw);
        return cast(scaled, Type::kBFloat16);
      } else if constexpr (std::is_same_v<P, CtBlock128Projection>) {
        namespace math = inference::ct_block128;
        if (activation_type() != Type::kBFloat16)
          throw std::runtime_error("CT block128 source reader requires BF16 activation");
        const auto raw = operand(*projection.codes), sf = operand(*projection.block_scales);
        const auto n = projection.codes->description.dimensions()[0];
        auto &row_map = assembly.literal(dense(Type::kInt32, {n}));
        std::vector<std::int32_t> rows(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<std::int32_t>(i / 128);
        row_map.write<std::int32_t>(rows);
        const auto row_value = constant(row_map);
        const auto literal = [&](std::uint16_t bits) -> math::Operand {
          auto &value = assembly.literal(dense(Type::kBFloat16, {}));
          value.write<std::uint16_t>(std::span(&bits, 1));
          const auto id = constant(value); return {id, spec(id)};
        };
        const math::Constants literals{literal(0x43e0), literal(0x3c00), literal(0)};
        math::Nodes nodes;
        const auto projected = take(math::append_projection(builder, {x, spec(x)},
            {raw, spec(raw)}, {sf, spec(sf)}, {row_value, spec(row_value)}, literals, {}, &nodes));
        for (std::size_t i = 0; i < nodes.count; ++i) assembly.node(nodes.values[i]);
        return remember(projected.value, projected.description);
      } else if constexpr (std::is_same_v<P, ChannelTokenFp8Projection>) {
        const auto original = operand(*projection.codes);
        // The selected channel/token reader uses typed BF16 QDQ. Only an
        // exactly-zero scale gets epsilon; all original rounding stays here.
        const auto maximum = scalar(448.F, Type::kBFloat16);
        const auto raw_scale = binary(c::BinaryKind::kDivide,
            one(c::Reduce{.kind = c::ReductionKind::kMaximum,
                .axes = take(c::Axes::from(std::array<std::int32_t, 1>{1})),
                .keep_dimensions = true, .accumulation_type = Type::kFloat32},
                {unary(c::UnaryKind::kAbs, x)},
                dense_spec(Type::kBFloat16, {spec(x).dimensions()[0], f(1)})), maximum);
        const auto scale = one(c::Select{}, {cast(raw_scale, Type::kBool), raw_scale,
            scalar(1.F / 128.F, Type::kBFloat16)}, spec(raw_scale));
        const auto normalized = binary(c::BinaryKind::kDivide, x, scale);
        const auto clipped = binary(c::BinaryKind::kMinimum,
            binary(c::BinaryKind::kMaximum, normalized, scalar(-448.F, Type::kBFloat16)), maximum);
        const auto activation = binary(c::BinaryKind::kMultiply,
            cast(cast(clipped, Type::kFloat8E4M3FN), Type::kBFloat16), scale);
        const auto decoded = binary(c::BinaryKind::kMultiply, cast(original, Type::kBFloat16), operand(*projection.channel_scales));
        constexpr std::array<std::int32_t, 2> axes{1, 0};
        return one(c::MatMul{.accumulation_type = Type::kFloat32},
            {activation, one(take(c::View::permute(axes)), {decoded})}, dense_spec(Type::kBFloat16, dimensions));
      }
    }, binding);
    return scores ? cast(result, Type::kFloat32) : result;
  }
  V linear_absolute(V x, const std::string &name) {
    return emit_projection(x, projection(name), false);
  }
  V scores(V x, const std::string &name) {
    return emit_projection(x, projection(name), true);
  }
  V norm_absolute(V x, const std::string &name) {
    const auto type = spec(x).data_type();
    auto scale = cast(weight_absolute(name), Type::kFloat32);
    if (meaning(name) != prepared::ParameterMeaning::kRmsEffectiveGamma)
      scale = binary(c::BinaryKind::kAdd, scale, scalar(1));
    const auto converted = cast(x, Type::kFloat32);
    return cast(same(c::RmsNorm{.epsilon = geometry.epsilon}, {converted, scale}, converted), type);
  }
  V norm(V x, const std::string &name) { return norm_absolute(x, "model.language_model." + name); }
  V decay(const std::string &name) {
    auto value = cast(weight_absolute(name), Type::kFloat32);
    return meaning(name) == prepared::ParameterMeaning::kGdnNegativeExpDecay ? value :
        unary(c::UnaryKind::kNeg, unary(c::UnaryKind::kExp, value));
  }
  // Literal BF16 boundaries from the independent local reference, including
  // BF16 square/sum/epsilon/rsqrt. GDN applies its own 1/sqrt(128) exactly once.
  V l2(V x) {
    const auto square = binary(c::BinaryKind::kMultiply, x, x);
    const std::array<std::int32_t, 1> axes{-1};
    auto dimensions = std::vector<D>(spec(square).dimensions().begin(), spec(square).dimensions().end());
    dimensions.back() = f(1);
    auto sum = one(c::Reduce{.kind = c::ReductionKind::kSum,
        .axes = take(c::Axes::from(axes)), .keep_dimensions = true,
        .accumulation_type = Type::kFloat32}, {square}, dense_spec(spec(square).data_type(), dimensions));
    sum = binary(c::BinaryKind::kAdd, sum, scalar(1.e-6F, spec(x).data_type()));
    return binary(c::BinaryKind::kMultiply, x, unary(c::UnaryKind::kRsqrt, sum));
  }
  V ffn(V x, const std::string &prefix, std::span<V> probes = {}) {
    auto normalized = norm(x, prefix + "post_attention_layernorm.weight");
    auto projected_gate = linear(normalized, prefix + "mlp.gate_proj.weight");
    auto gate = unary(c::UnaryKind::kSilu, projected_gate);
    auto up = linear(normalized, prefix + "mlp.up_proj.weight");
    auto product = binary(c::BinaryKind::kMultiply, gate, up);
    if (!probes.empty()) {
      probes[3] = product; probes[5] = x; probes[6] = normalized;
      probes[7] = projected_gate; probes[8] = up;
    }
    auto result = linear(product, prefix + "mlp.down_proj.weight");
    return binary(c::BinaryKind::kAdd, x, result);
  }
  V gdn_key_rows(V rows) {
    if (geometry.gdn_layout == GdnHeadLayout::kCyclic || geometry.gdn_key_heads == geometry.gdn_value_heads)
      return rows;
    // HF value heads are grouped by key head. Converted GGUF has already
    // reordered its V/conv/gates/output weights into the native cyclic order.
    // Preserve either source; only grouped HF needs this explicit boundary.
    std::vector<std::int32_t> expansion(static_cast<std::size_t>(geometry.gdn_value_heads));
    for (std::size_t head = 0; head < expansion.size(); ++head)
      expansion[head] = static_cast<std::int32_t>(head / static_cast<std::size_t>(geometry.gdn_value_heads / geometry.gdn_key_heads));
    auto &literal = assembly.literal(dense(Type::kInt32, {geometry.gdn_value_heads}));
    literal.write<std::int32_t>(expansion);
    const auto indices = constant(literal);
    auto dimensions = std::vector<D>(spec(rows).dimensions().begin(), spec(rows).dimensions().end());
    dimensions.at(2) = f(geometry.gdn_value_heads);
    return one(c::Gather{.axis = 2, .bounds = c::IndexBoundsPolicy::kUnchecked}, {rows, indices},
        dense_spec(spec(rows).data_type(), dimensions));
  }
  void publish(V value) { check(builder.publish(value)); assembly.external(value); }
};

// Distinguish direct packed contractions from the ordinary Decode/Cast route.
// The latter has per-projection Invocation workspace and may use managed
// provider temporaries; it does not replace the original constant weight bank.
void report_packed_projections(const g::Graph &graph, const g::CompileReport &report,
                               std::span<const PackedProjectionNodes> projections, std::size_t graph_workspace_bytes) {
  std::size_t packed = 0;
  for (const auto &projection : projections) {
    if (projection.count != 3 && projection.count != 4)
      throw std::runtime_error("packed projection requires its actual three/four-node recipe");
    std::size_t containing = 0;
    for (std::size_t step_index = 0; step_index < report.step_count(); ++step_index) {
      const auto step = take(report.step(step_index));
      std::array<bool, 4> included{};
      for (std::size_t i = 0; i < step.node_count; ++i) {
        const auto node = take(report.step_node(step_index, i));
        for (std::size_t j = 0; j < projection.count; ++j) included[j] = included[j] || node == projection.chain[j];
      }
      if (!included[projection.count - 1]) continue;
      ++containing;
      const bool contracted = step.policy == g::kArtifactPackedProjectionPolicy && step.node_count == projection.count &&
          std::ranges::all_of(std::span(included).first(projection.count), [](bool value) { return value; });
      if (contracted) {
        for (std::size_t j = 0; j + 1 < projection.count; ++j) {
          const auto intermediate = take(graph.result(projection.chain[j]));
          for (std::size_t i = 0; i < report.physical_value_count(); ++i)
            if (take(report.physical_value(i)).value == intermediate)
              throw std::runtime_error("packed projection materialized a floating weight intermediate");
        }
        ++packed;
      }
    }
    if (containing != 1) throw std::runtime_error("original packed projection is missing from its selected plan");
  }
  if (!projections.empty())
    std::cerr << "original_quantized_projections=" << projections.size()
              << " packed_contractions=" << packed
              << " decomposed_projections=" << projections.size() - packed
              << " graph_workspace_bytes=" << graph_workspace_bytes << '\n';
}

enum class GraphRole { kText, kVerify, kMtpPrefill, kMtpRound, kMtpCatchupRound,
                       kMtpSingle, kMtpCatchupSingle, kMtpCatchupPrefill };
constexpr bool has_mtp(GraphRole role) noexcept { return role >= GraphRole::kMtpPrefill; }
constexpr bool is_mtp_prefill(GraphRole role) noexcept {
  return role == GraphRole::kMtpPrefill || role == GraphRole::kMtpCatchupPrefill;
}
constexpr bool drafts_token(GraphRole role) noexcept {
  return role == GraphRole::kMtpRound || role == GraphRole::kMtpCatchupRound;
}
constexpr bool verifies_prefix(GraphRole role) noexcept {
  return role == GraphRole::kVerify || drafts_token(role);
}
constexpr std::size_t draft_lag(GraphRole role) noexcept {
  return role == GraphRole::kMtpCatchupRound || role == GraphRole::kMtpCatchupSingle ||
      role == GraphRole::kMtpCatchupPrefill ? 1u : 0u;
}
enum class TextCompletion { kEmitNext, kStateOnly };
struct SamplingConfiguration final {
  bool enabled = false;
  sampling::Policy policy;
};
struct PreparedSampling;
struct ColdResources;
void validate_sampling_configuration(const SamplingConfiguration &value, std::int64_t vocabulary) {
  if (vocabulary <= 0 || vocabulary > INT32_MAX || !std::isfinite(value.policy.temperature) || value.policy.temperature <= 0 ||
      !std::isfinite(value.policy.top_p) || value.policy.top_p <= 0 || value.policy.top_p > 1 ||
      value.policy.top_k > static_cast<std::size_t>(vocabulary) ||
      (!value.enabled && (value.policy.temperature != 1 || value.policy.top_k || value.policy.top_p != 1)))
    throw std::runtime_error("invalid finite sampling configuration for vocabulary");
}
bool same_sampling_configuration(const SamplingConfiguration &a, const SamplingConfiguration &b) noexcept {
  return a.enabled == b.enabled && a.policy.temperature == b.policy.temperature &&
      a.policy.top_k == b.policy.top_k && a.policy.top_p == b.policy.top_p;
}

struct GenerationBoundary final {
  std::size_t length = 0, frontier = 0;
  std::uint32_t lag = 0;
  std::int32_t next_id = -1, catchup_token = -1;
  cache::SamplingCursor sampling{};
  friend bool operator==(const GenerationBoundary &, const GenerationBoundary &) = default;
};
GenerationBoundary generation_boundary(const cache::Snapshot &head) noexcept {
  return {head.length, head.draft ? head.draft->frontier : 0,
      head.draft ? head.draft->lag : 0, head.next_id,
      head.draft ? head.draft->catchup_token : -1, head.sampling};
}
struct GenerationCommitRecord final {
  GraphRole role = GraphRole::kMtpPrefill;
  bool generation = false, committed = false, user_output = true;
  bool imported = false, prefill_complete = true;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::shared_ptr<const void> input_owner; // Immutable host data, never a Snapshot/lease.
  std::span<const std::int32_t> prompt_ids;
  std::int32_t pending_input = -1;
  std::size_t quota = 0, turn = 0, accepted = 0, draft_rows = 0;
  std::size_t draft_limit = 0, verified_rows = 0, consumed_drafts = 0;
  std::size_t auxiliary_graph_calls = 0;
  std::int32_t draft = -1, target_first = -1, target_second = -1;
  bool agreed = false, accepted_draft = false, stopped = false;
  hyle::container::SmallVector<std::int32_t, 2> emitted{-1, -1};
  hyle::container::SmallVector<std::int32_t, 2> accepted_inputs{-1, -1};
  hyle::container::SmallVector<std::int32_t, 2> verification_inputs;
  std::size_t requests = 1, member_index = 0;
  std::size_t emitted_count = 0;
  GenerationBoundary before, after;
  double step_seconds = 0;
  Clock::time_point endpoint{};
  std::span<const std::int32_t> ids() const noexcept {
    return generation ? std::span(&pending_input, 1) : prompt_ids;
  }
  std::span<const std::int32_t> committed_ids() const noexcept {
    if (generation && (accepted == 0 || accepted > accepted_inputs.size())) return {};
    return generation ? std::span<const std::int32_t>(accepted_inputs).first(accepted) : prompt_ids;
  }
  bool same_result(const GenerationCommitRecord &other) const noexcept {
    return requests == other.requests && member_index == other.member_index &&
        role == other.role && generation == other.generation && quota == other.quota &&
        completion == other.completion && prefill_complete == other.prefill_complete && input_begin == other.input_begin &&
        before == other.before && after == other.after && accepted == other.accepted &&
        draft_limit == other.draft_limit && verified_rows == other.verified_rows && consumed_drafts == other.consumed_drafts &&
        auxiliary_graph_calls == other.auxiliary_graph_calls &&
        draft_rows == other.draft_rows && draft == other.draft && target_first == other.target_first &&
        target_second == other.target_second && agreed == other.agreed && accepted_draft == other.accepted_draft &&
        stopped == other.stopped && emitted == other.emitted && emitted_count == other.emitted_count &&
        accepted_inputs == other.accepted_inputs && verification_inputs == other.verification_inputs;
  }
};
struct GenerationLedger final {
  std::vector<GenerationCommitRecord> records;
  std::size_t committed_count = 0;
private:
  // Value-owned immutable configuration survives GenerationRequest relocation.
  // This is copied only at initial cold binding, never per token/record.
  std::optional<inference::RequestStopPolicy> stop_policy_;
public:
  const inference::RequestStopPolicy &stop_policy() const noexcept {
    return stop_policy_ ? *stop_policy_ : inference::default_request_stop_policy;
  }
  void validate_stop_policy(const inference::RequestStopPolicy &value) const {
    if ((stop_policy_ || committed_count) && stop_policy() != value)
      throw std::runtime_error("request stop policy differs from its committed generation ledger");
  }
  void bind_stop_policy(const inference::RequestStopPolicy &value, std::int64_t vocabulary) {
    value.validate(vocabulary); validate_stop_policy(value);
    if (!stop_policy_) stop_policy_.emplace(value);
  }
  const PreparedSampling *sampling = nullptr; // Borrowed immutable cold policy, never a cursor.
  std::weak_ptr<ColdResources> sampling_owner; // Identity only: cannot pin weights or admit an address-reused Model.
  void validate_sampling(const PreparedSampling &value, const std::weak_ptr<ColdResources> &owner) const {
    if (sampling && (sampling_owner.expired() || sampling_owner.owner_before(owner) || owner.owner_before(sampling_owner) || sampling != &value))
      throw std::runtime_error("generation ledger sampling belongs to another model or configuration");
  }
  void bind_sampling(const PreparedSampling &value, const std::weak_ptr<ColdResources> &owner) {
    validate_sampling(value, owner); sampling = &value; sampling_owner = owner;
  }
  explicit GenerationLedger(std::size_t capacity, std::size_t maximum_width = 2) : records(capacity) {
    for (auto &record : records)
      if (!record.emitted.try_reserve(maximum_width) || !record.accepted_inputs.try_reserve(maximum_width) ||
          !record.verification_inputs.try_reserve(maximum_width))
        throw std::bad_alloc{};
  }
  GenerationCommitRecord &next() {
    if (committed_count >= records.size() || records[committed_count].committed)
      throw std::runtime_error("MTP replay ledger has no uncommitted reserved record");
    return records[committed_count];
  }
};
struct FamilyKey final {
  std::size_t requests, verification_rows;
  GraphRole role;
  TextCompletion completion;
  std::size_t sampling_index;
  FamilyKey(std::size_t count, GraphRole selected = GraphRole::kText,
            std::size_t rows = 0, TextCompletion mode = TextCompletion::kEmitNext, std::size_t sampling = 0) noexcept
      : requests(count), verification_rows(selected == GraphRole::kVerify ? rows : 0), role(selected), completion(mode), sampling_index(sampling) {}
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
  auto operator<=>(const ExecutionShape &) const = default;
};
struct StatePorts final {
  V history, recurrent;
  // Cold, chronological material outputs. Ordinary execution has one final
  // boundary; verification has one independent output for every input prefix.
  std::vector<V> history_prefix, recurrent_prefix;
};
struct Ports final {
  V key, value;
  std::vector<StatePorts> sequences;
};
struct FamilyPorts {
  std::size_t prefix_rows = 1;
  V token_port, position_port, last_ids_port, metadata_port, logits_port, id_port, hidden_port;
  V sampling_identity_port, sampling_valid_port, sampling_accept_port;
  std::vector<g::NodeId> attention_nodes, append_nodes;
  V draft_key_port, draft_value_port, draft_metadata_port, draft_position_port;
  V carry_input_port, prior_input_port, catchup_token_port, draft_id_port;
  std::vector<std::array<V, 2>> draft_hidden_inputs;
  g::NodeId draft_attention_node;
  std::vector<V> state_resource_ports;
  std::vector<std::vector<V>> carry_ports;
  std::array<V, 9> probe_ports;
  std::vector<Ports> layers;
};
struct FamilyChild final {
  g::CompiledGraph compiled; // One immutable selected child shared by every session.
  std::vector<PreparedOutputFacts> outputs;
  std::vector<std::uint32_t> output_indices;
  const PreparedOutputFacts &output(V value) const {
    if (!value || value.ordinal() >= output_indices.size())
      throw std::runtime_error("value is outside the prepared child output index");
    const auto index = output_indices[value.ordinal()];
    if (index >= outputs.size() || outputs[index].value != value)
      throw std::runtime_error("value is not a prepared output of this exact child");
    return outputs[index];
  }
};
struct Family final : FamilyPorts {
  const PreparedSampling *sampling = nullptr;
  std::size_t requests = 1;
  GraphRole role = GraphRole::kText;
  TextCompletion completion = TextCompletion::kEmitNext;
  bool admitted_exact = false;
  std::vector<Tensor> literals; // Backing survives every child and its slots.
  std::vector<g::TensorSpec> specs;
  c::BatchSchema attention_schema;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::DimensionId last_row;
  std::vector<std::size_t> chunks;
  std::vector<std::unique_ptr<FamilyChild>> children;
};
// Model admission holds the whole carrier through native retirement and final
// readback. Batch width and retained state boundaries do not add in-flight work.
constexpr std::size_t execution_slot_count = 1;
using ResourceRetention = inference::RetainedResources<ColdResources>;
struct Phase final : FamilyPorts {
  inference::QueueBindings queues;
  std::size_t query_rows = 0;
  c::BatchSchema attention_schema;
  c::BatchLayoutVersion attention_layout;
  // The selected child is shared immutable model data, not copied per session.
  const FamilyChild *child = nullptr;
  const Family *family = nullptr;
  std::size_t variant_index = 0;
  std::array<g::Invocation, execution_slot_count> slots;
  std::array<ResourceRetention, execution_slot_count> retirement;
  std::array<std::size_t, execution_slot_count> planned_slot_uses{};
  std::array<std::size_t, execution_slot_count> preparation_uses{};
  std::array<g::ReplayMode, execution_slot_count> replay_modes{};
  Tensor tokens, positions, last_ids, logits, final_hidden;
  const PreparedOutputFacts *logits_facts = nullptr, *hidden_facts = nullptr;
  Tensor sampling_identities, draft_positions, catchup_token;
  PreparedReadback next_id, draft_id, sampling_valid, sampling_accept;
  std::vector<std::uint8_t> sampling_valid_values, sampling_accept_values;
  std::vector<float> sampling_scratch;
  std::vector<std::int32_t> draft_ids, prefix_candidates;
  std::array<Tensor, 9> probe_outputs;
  std::array<const PreparedOutputFacts *, 9> probe_facts{};
};

// One original Graph episode owns all dirty pages and candidate state. Until
// finish replaces the Session head, neither a ready layer nor an error changes
// the committed snapshot. Failure recovery remains attached to that Session.
// One host-only upload workspace for the original serial Model carrier.
// Its integer ranges never escape synchronous Tensor::write; retained model
// state, page permissions and submitted resources keep their existing owners.
struct InputPreparationScratch final {
  std::vector<std::int32_t> packed_ids, positions, draft_positions, catchups;
  std::vector<std::uint32_t> sampling_identities;
  // BatchSchema::version synchronously copies this integer source range.
  // Keep page readers, immutable layout versions and NodeLayouts outside scratch.
  std::vector<std::int32_t> target_page_ids;
  void prepare(const Phase &phase) {
    const auto reserve = [](auto &values, const Tensor &tensor) {
      using Element = typename std::decay_t<decltype(values)>::value_type;
      values.reserve(take(tensor.description.required_bytes()) / sizeof(Element));
    };
    target_page_ids.reserve(phase.attention_schema.capacity().max_page_refs);
    if (phase.family->requests > 1) reserve(packed_ids, phase.tokens);
    reserve(positions, phase.positions);
    if (phase.sampling_identities.caller_owned()) reserve(sampling_identities, phase.sampling_identities);
    if (phase.draft_positions.caller_owned()) reserve(draft_positions, phase.draft_positions);
    if (phase.catchup_token.caller_owned()) reserve(catchups, phase.catchup_token);
  }
};
class Session;
struct StepMember final {
  std::shared_ptr<Session> session;
  cache::SnapshotRef base;
  // Synchronous episode borrow, like its original request input. No native
  // callback reads this policy; the original owner keeps it through finish.
  const inference::RequestStopPolicy *stop_policy = nullptr;
  std::shared_ptr<cache::Snapshot> next{};
  hyle::container::SmallVector<c::PagedPageAppender, 32> next_tails{};
  std::array<c::PagedPageAppender, 2> next_draft_tails{};
  c::BatchLayoutVersion layout{};
  c::BatchLayoutVersion draft_layout{};
  std::size_t accepted_tokens = 0;
  std::uint32_t verified_last_page = 0;
  std::int32_t first_input = -1, draft_token = -1;
  std::vector<std::int32_t> verification_inputs{};
  std::vector<c::ReadLease> carry_prefix{};
  std::size_t discarded_draft_rows = 0;
  bool sampled_accept = false;
  GenerationCommitRecord *commit_record = nullptr;
  GenerationLedger *commit_ledger = nullptr;
};
struct StepState final {
  // Model owns the episode; these strong pins outlive all accepted native work.
  // Session's pending link borrows this owner and never creates a cycle.
  hyle::container::SmallVector<StepMember, 1> members;
  hyle::container::SmallVector<std::vector<c::PagedPageWriter>, 34> dirty;
  // One inherited partial page per request/bank. The common R1 case stays inline.
  hyle::container::SmallVector<hyle::container::SmallVector<c::PagedPageAppender, 1>, 34> tails;
  hyle::container::SmallVector<g::batch::PagedAppendPages, 34> appends;
  hyle::container::SmallVector<g::NodeId, 34> nodes;
  hyle::container::SmallVector<std::uint8_t, 34> bound;
  void configure(std::size_t banks) {
    if (!dirty.try_resize(banks) || !tails.try_resize(banks) || !appends.try_resize(banks) ||
        !nodes.try_resize(banks) || !bound.try_resize(banks)) throw std::bad_alloc{};
  }
  c::BatchLayoutVersion draft_layout;
  g::Invocation *run = nullptr;
  Phase *phase = nullptr;
  c::Completion completion;
  hyle::Status terminal;
  Clock::time_point step_start{}, start{};
  double submit_seconds = 0, cold_batch_seconds = 0;
  std::size_t input_tokens = 0, capture_recorded = 0, capture_reused = 0;
  g::ReplayRoute replay_route = g::ReplayRoute::kUnprepared;
  bool prepared = false, submitted = false, cancelled = false;
  // Model admission is serial even when a cancelled native episode has not
  // retired. Return it only after publication and final result reads, or after
  // complete abort cleanup.
  bool *candidate_admission = nullptr;
  void release_admission() noexcept {
    if (candidate_admission) *std::exchange(candidate_admission, nullptr) = false;
  }

  hyle::Result<bool> try_abort() noexcept {
    if (completion.valid()) {
      auto observed = completion.poll();
      if (observed && *observed == c::CompletionPoll::kPending) return false;
      if (!observed) inference::record_retirement_status(terminal, observed.error());
      completion = {}; // Error/abandonment keeps actual native guards retained.
    }
    if (run) {
      if (run->in_flight()) return false;
      for (std::size_t i = 0; i < appends.size(); ++i) if (bound[i]) {
        auto recovered = g::batch::take_append(*run, nodes[i]);
        if (!recovered) {
          if (recovered.error().IsUnavailable()) return false;
          return std::unexpected(recovered.error());
        }
        appends[i] = std::move(*recovered); bound[i] = false;
      }
      *run = {}; // Release failed state-output writers and old input bindings.
    }
    const auto recover = [](c::PagedPageWriter &page) noexcept -> hyle::Status {
      if (!page.valid()) return hyle::Status::Ok(); // Already successfully published.
      page.lease().reset();
      HYLE_RETURN_IF_ERROR(page.discard()); // Unavailable retains reservation.
      page = {}; // Return the recovered pristine writer; retries skip this page.
      return hyle::Status::Ok();
    };
    for (auto &append : appends) if (append.valid()) {
      for (std::size_t i = 0; i < append.appender_count(); ++i) {
        HYLE_ASSIGN_OR_RETURN(auto &tail, append.appender(i));
        if (!tail.valid() || !tail.reserved_rows()) continue;
        auto rolled_back = tail.rollback();
        if (rolled_back.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(rolled_back);
      }
      for (std::size_t page = 0; page < append.page_count(); ++page) {
        HYLE_ASSIGN_OR_RETURN(auto &writer, append.page(page));
        auto recovered = recover(writer);
        if (recovered.IsUnavailable()) return false;
        HYLE_RETURN_IF_ERROR(recovered);
      }
    }
    for (auto &bank : tails) for (auto &tail : bank) {
      if (!tail.valid() || !tail.reserved_rows()) continue;
      auto rolled_back = tail.rollback();
      if (rolled_back.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(rolled_back);
    }
    for (auto &bank : dirty) for (auto &writer : bank) {
      auto recovered = recover(writer);
      if (recovered.IsUnavailable()) return false;
      HYLE_RETURN_IF_ERROR(recovered);
    }
    appends = {};
    for (auto &bank : dirty) bank.clear();
    tails = {};
    for (auto &member : members) { member.next.reset(); member.base.reset(); }
    run = nullptr; phase = nullptr;
    release_admission();
    return true;
  }
};

// Each original device allocation becomes one retained lease pool;
// bounded pinned staging is the only extra payload storage.
struct WeightSourceFile final {
  hyle::io::File file;
  std::uint64_t capacity = 0;
};
hyle::sched::Task<hyle::Result<WeightSourceFile>> open_weight_source(
    mem::AllocatorRef, hyle::io::IoContext *io, std::string path) {
  auto opened = co_await hyle::io::open_file(*io, 0, path,
      {.access = hyle::io::FileAccess::kRead});
  if (!opened) co_return std::unexpected(std::move(opened).error());
  auto file = std::move(*opened);
  auto information = co_await file.info();
  if (!information) co_return std::unexpected(std::move(information).error());
  if (!information->is_regular())
    co_return std::unexpected(hyle::Status::InvalidArgument("checkpoint is not a regular file"));
  co_return WeightSourceFile{std::move(file), information->size};
}
hyle::sched::Task<hyle::Status> close_weight_source(
    mem::AllocatorRef, hyle::io::File *file) {
  co_return co_await file->close();
}
struct WeightLoadReport final {
  std::size_t bytes = 0;
  c::FileStreamReport stream;
  std::string identity;
  WeightMeanings meanings;
  std::size_t device_bytes = 0, host_bytes = 0;
  std::size_t physical_text_roles = 0, physical_mtp_roles = 0;
  std::size_t canonical_bytes = 0, fp8_groups = 0, import_peak_bytes = 0;
  bool float8_policy = false;
  std::size_t ct_channel_groups = 0, ct_local_groups = 0, ct_block_groups = 0, omitted_cache_roles = 0;
  std::size_t nf4_groups = 0, nf4_original_bytes = 0, nf4_input_copy_bytes = 0, nf4_known_group_peak_bytes = 0;
};
enum class WeightSelection { kText, kTextAndMtp };
ModelGeometry read_weight_geometry(const std::filesystem::path &manifest) {
  std::ifstream input(manifest); std::string signature, identity, line;
  if (!input) throw std::runtime_error("cannot open prepared input: " + manifest.string());
  prepared::read_manifest_line(input, signature, 128); prepared::read_manifest_line(input, identity, 65);
  if (!prepared::is_role_signature(signature)) throw std::runtime_error("input requires a prepared Qwen V2 manifest; rerun prepare.py");
  if (identity.ends_with('\r')) identity.pop_back();
  prepared::validate_identity(identity); prepared::read_manifest_line(input, line);
  return read_model_geometry(line);
}
prepared::Roles read_weight_roles(const std::filesystem::path &manifest, bool include_mtp) {
  return prepared::read_roles(manifest, include_mtp);
}
struct PreparedWeightSource {
  prepared::Roles roles;
  std::vector<CacheStoragePair> cache_storage;
  cache::KvRowBytes cache_row_bytes;
  double metadata_prepare_seconds = 0;
};
PreparedWeightSource prepare_weight_source(const std::filesystem::path &manifest, bool include_mtp) {
  const auto start = Clock::now();
  auto roles = read_weight_roles(manifest, include_mtp);
  auto storage = read_cache_calibration(roles);
  const auto plain = take(c::CacheStorage::plain(PrecisionProfile::from_roles(roles).activation_type()));
  const auto rows = cache_storage_row_bytes(roles.geometry, storage.empty() ? CacheStoragePair{plain, plain} : storage.front());
  for (const auto &pair : storage)
    if (cache_storage_row_bytes(roles.geometry, pair) != rows)
      throw std::runtime_error("selected cache layers require different physical row extents");
  return {std::move(roles), std::move(storage), {rows[0], rows[1]},
      std::chrono::duration<double>(Clock::now() - start).count()};
}
Type role_type(std::string_view dtype) {
  if (dtype == "BF16") return Type::kBFloat16;
  if (dtype == "F16") return Type::kFloat16;
  if (dtype == "F32") return Type::kFloat32;
  if (dtype == "I32") return Type::kInt32;
  if (dtype == "U32") return Type::kUInt32;
  if (dtype == "I64") return Type::kInt64;
  if (dtype == "U8") return Type::kUInt8;
  if (dtype == "F8_E4M3") return Type::kFloat8E4M3FN;
  throw std::runtime_error("unsupported original role dtype: " + std::string(dtype));
}
c::TensorDesc role_description(const prepared::Role &role) {
  const auto codec = [&]() -> std::optional<c::GgufQuantizedType> {
    if (role.dtype == "Q3_K") return c::GgufQuantizedType::kQ3K;
    if (role.dtype == "IQ3_S") return c::GgufQuantizedType::kIq3S;
    if (role.dtype == "IQ4_NL") return c::GgufQuantizedType::kIq4Nl;
    if (role.dtype == "IQ4_XS") return c::GgufQuantizedType::kIq4Xs;
    if (role.dtype == "Q4_K") return c::GgufQuantizedType::kQ4K;
    if (role.dtype == "Q5_K") return c::GgufQuantizedType::kQ5K;
    if (role.dtype == "Q6_K") return c::GgufQuantizedType::kQ6K;
    if (role.dtype == "Q8_0") return c::GgufQuantizedType::kQ8_0;
    if (role.dtype == "NVFP4") return c::GgufQuantizedType::kNvfp4;
    return std::nullopt;
  }();
  if (!codec) return take(c::TensorDesc::dense(role_type(role.dtype), role.shape));
  if (role.shape.size() != 2) throw std::runtime_error("encoded original role must be [N,K]: " + role.name);
  return take(take(c::gguf_weight_description(*codec, Type::kFloat32, role.shape[1], role.shape[0])).transposed());
}
// The complete model wrapper validates its full source inventory first. A
// source-bound component may select an explicit subset of those same roles;
// both paths use this one IO/lease/shape-authentication implementation.
WeightLoadReport load_selected_weight_roles(const prepared::Roles &roles,
    Backend &backend, std::map<std::string, LoadedWeight> &weights,
    std::vector<c::LeasePool> &weight_pools, const FfnPlacement &placement,
    const c::ResourceBudget &budget = {}) {
  roles.geometry.validate();
  if (placement.target_layers() != roles.geometry.layers.size())
    throw std::runtime_error("weight source differs from the cold FFN placement");
  (void)PrecisionProfile::from_roles(roles);
  if (roles.selected.empty() || roles.selected.size() > UINT32_MAX)
    throw std::runtime_error("invalid prepared physical role count");
  // Source groups borrow existing checkpoint/cache/run files. No stitched
  // checkpoint or duplicate packed bank is created. NF4 keeps its six typed
  // decoder operands; the ordinary cold planner decides contraction or decode.
  std::map<std::pair<std::filesystem::path, bool>, std::vector<std::size_t>> sources;
  std::array<std::size_t, 2> largest{}, largest_group{}, source_counts{}, domain_bytes{};
  std::set<std::string> fp8_roles, nf4_roles;
  for (const auto &role : roles.selected) if (role.projection_codec == prepared::ProjectionCodec::kFloat8Block128) {
    fp8_roles.insert(role.name);
    fp8_roles.insert(role.name.substr(0, role.name.size() - 7) + ".weight_scale_inv");
  }
  constexpr std::array<std::string_view, 6> nf4_suffixes{
      "", ".absmax", ".nested_absmax", ".nested_quant_map", ".quant_map", prepared::nf4_state_suffix};
  for (const auto &role : roles.selected) if (role.projection_codec == prepared::ProjectionCodec::kNf4DoubleQuant)
    for (const auto suffix : nf4_suffixes) nf4_roles.insert(role.name + std::string(suffix));
  std::size_t physical_mtp_roles = 0;
  for (std::size_t i = 0; i < roles.selected.size(); ++i) {
    const auto &role = roles.selected[i];
    const bool host = placement.role_on_cpu(role.name);
    domain_bytes[host] = cache::add(domain_bytes[host], role.bytes);
    physical_mtp_roles += role.name.starts_with("mtp.");
    // FP8 pairs enter one bounded cold import at a time below. Do not first
    // allocate their original codes/scales across the complete model bank.
    if (fp8_roles.contains(role.name) || nf4_roles.contains(role.name)) continue;
    sources[{role.source, host}].push_back(i); largest[host] = (std::max)(largest[host], role.bytes);
  }
  for (const auto &[key, indices] : sources) {
    const auto &[path, host] = key;
    ++source_counts[host]; largest_group[host] = (std::max)(largest_group[host], indices.size());
    std::error_code error;
    const auto capacity = std::filesystem::file_size(path, error);
    if (error) throw std::runtime_error("cannot inspect prepared weight source: " + path.string() + ": " + error.message());
    for (const auto index : indices) {
      const auto &role = roles.selected[index];
      if (role.offset > capacity || role.bytes > capacity - role.offset)
        throw std::runtime_error("short prepared weight source for " + role.name);
    }
  }
  auto scheduler_status = hyle::Status::Ok();
  hyle::sched::Scheduler loader_scheduler(2, mem::system_allocator(), 256, &scheduler_status);
  check(scheduler_status);
  hyle::io::IoContext io(loader_scheduler, {.max_blocking_threads = 2});
  struct Shutdown final {
    hyle::sched::Scheduler &scheduler;
    ~Shutdown() { scheduler.shutdown(); }
  } shutdown{loader_scheduler};
  check(io.status());
  constexpr std::size_t staging_per_slot = 8u * 1024u * 1024u;
  constexpr std::uint32_t slots = 2;
  if (roles.bytes > (std::numeric_limits<std::size_t>::max)() - slots * staging_per_slot)
    throw std::runtime_error("weight loader backing capacity overflows");
  // Exported ready weights intentionally retain their request's read owners.
  // Reserve one record per actual source; the two staging slots stay shared.
  std::array<c::FileResourcePipeline, 2> pipelines;
  for (std::size_t domain = 0; domain < pipelines.size(); ++domain) if (source_counts[domain]) {
    const std::array targets{c::CopyResourceSpec{weight_resource(backend, domain != 0), largest[domain]}};
    const c::FileStreamOptions options{
        .resource_budget = budget,
        .max_requests = static_cast<std::uint32_t>(source_counts[domain]),
        .max_groups_per_request = static_cast<std::uint32_t>(largest_group[domain]),
        .max_ranges_per_group = 1, .in_flight_slots = slots, .staging_bytes_per_slot = staging_per_slot,
        .max_active_backing_bytes = cache::add(domain_bytes[domain], slots * staging_per_slot)};
    if (!domain) pipelines[domain] = take(backend.create_file_pipeline(io, targets, options));
    else {
      pipelines[domain] = take(backend.create_host_file_pipeline(io, targets, options));
    }
  }
  struct Drain final {
    std::array<c::FileResourcePipeline, 2> &pipelines;
    ~Drain() { for (auto &pipeline : pipelines) if (pipeline.valid()) pipeline.drain().IgnoreError(); }
  } drain{pipelines};
  weight_pools.reserve(roles.selected.size());
  std::size_t loaded = 0;
  for (const auto &[key, indices] : sources) {
    const auto &[path, host] = key; auto &pipeline = pipelines[host];
    auto open_task = open_weight_source(mem::system_allocator(), &io, path.string());
    if (!open_task.valid()) throw std::runtime_error("cannot allocate weight open task");
    auto file = take(take(hyle::sched::spawn_task_on(loader_scheduler, 0,
        mem::system_allocator(), std::move(open_task)).join()));
    auto source = take(c::StreamFile::borrow(file.file, file.capacity));
    // A failed host allocation or result check can unwind after read() has
    // accepted IO. The outer pipeline guard runs after this local File dies;
    // retire this source first, while its borrowed File address is still valid.
    hyle::util::ScopeFail retire_source{[&]() noexcept { pipeline.drain().IgnoreError(); }};
    std::vector<c::StreamReadRange> ranges;
    ranges.reserve(indices.size());
    std::size_t source_bytes = 0;
    for (const auto index : indices) {
      const auto &role = roles.selected[index];
      if (role.offset > file.capacity || role.bytes > file.capacity - role.offset)
        throw std::runtime_error("short checkpoint tensor: " + role.name);
      const auto description = role_description(role);
      if (take(description.required_bytes()) != role.bytes)
        throw std::runtime_error("physical weight extent differs: " + role.name);
      Tensor tensor(weight_resource(backend, host), description, !budget);
      auto pool = budget
          ? take(c::LeasePool::create(weight_resource(backend, host), role.bytes,
                {.max_blocks = 1, .resource_budget = budget}))
          : take(c::LeasePool::adopt(std::move(tensor.buffer), {.max_blocks = 1}));
      auto writer = take(pool.allocate(role.bytes));
      ranges.push_back({.file_offset = role.offset, .destination = std::move(writer),
          .bytes = role.bytes, .description = description});
      weight_pools.push_back(std::move(pool));
      if (!weights.emplace(role.name, LoadedWeight{std::move(tensor), role.projection_codec, role.projection_group_size}).second)
        throw std::runtime_error("duplicate selected weight: " + role.name);
      source_bytes += role.bytes;
    }
    std::vector<c::StreamReadGroup> groups;
    groups.reserve(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) groups.push_back({i, std::span(ranges).subspan(i, 1)});
    auto ticket = take(pipeline.read(c::StreamSource::file(source), groups));
    std::vector<bool> delivered(groups.size(), false);
    for (std::size_t n = 0; n < groups.size(); ++n) {
      auto group = take(ticket.wait_next_ready());
      if (group.tag() >= indices.size() || group.range_count() != 1)
        throw std::runtime_error("weight stream returned an unexpected group");
      const auto index = static_cast<std::size_t>(group.tag());
      if (delivered[index]) throw std::runtime_error("weight stream returned a duplicate group");
      auto ready = take(group.output(0));
      if (ready.size() != ranges[index].bytes) throw std::runtime_error("incomplete original weight tensor");
      weights.at(roles.selected[indices[index]].name).tensor.ready = std::move(ready);
      delivered[index] = true;
    }
    const auto result = ticket.wait();
    check(result.status);
    if (!result.complete || !result.retirement_proven || result.read_bytes != source_bytes ||
        result.copied_bytes != source_bytes || result.ready_groups != groups.size())
      throw std::runtime_error("weight stream byte inventory or retirement differs");
    ticket = {};
    check(pipeline.collect());
    source.reset();
    auto close_task = close_weight_source(mem::system_allocator(), &file.file);
    if (!close_task.valid()) throw std::runtime_error("cannot allocate weight close task");
    check(take(hyle::sched::spawn_task_on(loader_scheduler, 0,
        mem::system_allocator(), std::move(close_task)).join()));
    loaded += source_bytes;
  }
  std::size_t fp8_groups = 0, import_peak_bytes = 0;
  for (const auto &role : roles.selected) if (role.projection_codec == prepared::ProjectionCodec::kFloat8Block128) {
    const auto scale_name = role.name.substr(0, role.name.size() - 7) + ".weight_scale_inv";
    const auto scale = std::ranges::find(roles.selected, scale_name, &prepared::Role::name);
    if (scale == roles.selected.end() || !roles.float8_policy.enabled)
      throw std::runtime_error("FP8 block128 import requires its complete group and reader policy");
    auto imported = take(import_float8_block128_roles(role, *scale, budget));
    const auto carrier_bytes = take(imported.description.required_bytes());
    // Includes the mapped original group as a conservative host-residency
    // upper bound, its exact F32 scale grid, and this one canonical carrier.
    import_peak_bytes = (std::max)(import_peak_bytes, cache::add(cache::add(role.bytes, scale->bytes),
        cache::add(carrier_bytes, imported.temporary_scale_bytes)));
    const bool host = placement.role_on_cpu(role.name);
    Tensor tensor; tensor.description = imported.description;
    if (budget) {
      if (weight_resource(backend, host) == mem::system_resource()) {
        tensor.ready = std::move(imported.ready);
        weight_pools.push_back(std::move(imported.pool));
      } else {
        auto pool = take(c::LeasePool::create(weight_resource(backend, host), carrier_bytes,
            {.max_blocks = 1, .resource_budget = budget}));
        auto writer = take(pool.allocate(carrier_bytes));
        auto mapped = take(take(imported.ready.view()).map_host());
        check(mapped.invalidate()); check(backend.write(take(writer.view()), mapped.bytes()));
        check(mapped.close());
        tensor.ready = take(writer.publish());
        weight_pools.push_back(std::move(pool));
      }
    } else {
      if (weight_resource(backend, host) == mem::system_resource()) tensor.buffer = std::move(imported.storage);
      else {
        tensor.buffer = take(mem::Buffer::create(weight_resource(backend, host), carrier_bytes, 256));
        auto mapped = take(imported.storage.view().map_host(mem::HostAccess::kRead));
        check(mapped.invalidate()); check(backend.write(tensor.buffer.view(), mapped.bytes()));
        check(mapped.close());
      }
      tensor.freeze(backend);
    }
    if (!weights.emplace(role.name, LoadedWeight{std::move(tensor), role.projection_codec, 128}).second)
      throw std::runtime_error("duplicate selected FP8 projection: " + role.name);
    loaded = cache::add(loaded, cache::add(role.bytes, scale->bytes));
    ++fp8_groups;
  }
  std::size_t nf4_groups = 0, nf4_original_bytes = 0, nf4_input_copy_bytes = 0, nf4_known_group_peak_bytes = 0;
  for (const auto &role : roles.selected) if (role.projection_codec == prepared::ProjectionCodec::kNf4DoubleQuant) {
    const bool host = placement.role_on_cpu(role.name);
    const auto resource = weight_resource(backend, host);
    const auto domain = g::StorageDomainId{host ? 1u : 0u};
    std::array<mem::FileMapping, 6> original;
    std::array<std::span<const std::byte>, 6> raw;
    std::array<c::TensorDesc, 6> physical;
    std::size_t group_bytes = 0, copied_bytes = 0;
    for (std::size_t i = 0; i < nf4_suffixes.size(); ++i) {
      const auto name = role.name + std::string(nf4_suffixes[i]);
      const auto found = std::ranges::find(roles.selected, name, &prepared::Role::name);
      if (found == roles.selected.end()) throw std::runtime_error("NF4 cold import is missing original role: " + name);
      physical[i] = role_description(*found);
      original[i] = take(mem::FileMapping::open(found->source.string().c_str(), mem::MappingAccess::kReadOnly,
          found->offset, found->bytes));
      raw[i] = take(original[i].bytes());
      group_bytes = cache::add(group_bytes, found->bytes);
    }
    const auto n = role.nf4_state.logical_shape[0], k = role.nf4_state.logical_shape[1];
    if (k % 2) throw std::runtime_error("NF4 cold import requires complete original packed rows");
    Nf4DecodeInputs inputs;
    inputs.original_json = raw[5];
    inputs.descriptions = {dense(Type::kUInt8, {n, k / 2}), physical[1], physical[2], physical[3],
        dense(Type::kFloat32, {}), physical[4]};
    const auto offset = std::bit_cast<float>(role.nf4_state.offset_f32_bits);
    const std::array<std::span<const std::byte>, 6> payloads{
        raw[0], raw[1], raw[2], raw[3], std::as_bytes(std::span(&offset, 1)), raw[4]};
    for (std::size_t i = 0; i < payloads.size(); ++i) {
      if (budget) {
        auto pool = take(c::LeasePool::create(resource, payloads[i].size(),
            {.max_blocks = 1, .resource_budget = budget}));
        auto writer = take(pool.allocate(payloads[i].size()));
        check(backend.write(take(writer.view()), payloads[i]));
        inputs.ready[i] = take(writer.publish());
        weight_pools.push_back(std::move(pool));
      } else {
        auto buffer = take(mem::Buffer::create(resource, payloads[i].size(), 256));
        check(backend.write(buffer.view(), payloads[i]));
        inputs.ready[i] = take(backend.adopt_buffer(std::move(buffer)));
      }
      copied_bytes = cache::add(copied_bytes, payloads[i].size());
    }
    (void)take(validate_nf4_decode_inputs(backend, role.nf4_state, inputs, domain));
    constexpr std::array<std::string_view, 6> binding_suffixes{
        "", ".absmax", ".nested_absmax", ".nested_quant_map", ".nf4_offset", ".quant_map"};
    for (std::size_t i = 0; i < inputs.ready.size(); ++i) {
      Tensor tensor; tensor.description = inputs.descriptions[i]; tensor.ready = std::move(inputs.ready[i]);
      const auto name = role.name + std::string(binding_suffixes[i]);
      if (!weights.emplace(name, LoadedWeight{std::move(tensor), i == 0 ? role.projection_codec :
          prepared::ProjectionCodec::kUnspecified, i == 0 ? 64u : 0u}).second)
        throw std::runtime_error("duplicate selected NF4 operand: " + name);
    }
    loaded = cache::add(loaded, group_bytes);
    nf4_original_bytes = cache::add(nf4_original_bytes, group_bytes);
    nf4_input_copy_bytes = cache::add(nf4_input_copy_bytes, copied_bytes);
    nf4_known_group_peak_bytes = (std::max)(nf4_known_group_peak_bytes, cache::add(group_bytes, copied_bytes));
    ++nf4_groups;
  } // Original mappings retire here; the six owned numeric inputs remain resident.
  if (loaded != roles.bytes || weights.size() + fp8_groups != roles.selected.size())
    throw std::runtime_error("selected model weight inventory differs");
  for (const auto &role : roles.selected) if (role.name.ends_with(".weight_shape")) {
    const auto principal_name = role.name.substr(0, role.name.rfind('.')) + ".weight_packed";
    const auto principal = std::ranges::find(roles.selected, principal_name, &prepared::Role::name);
    if (principal == roles.selected.end()) throw std::runtime_error("compressed-tensors shape has no identified packed projection");
    std::array<std::byte, 16> original{};
    check(backend.read(take(weights.at(role.name).tensor.ready.view()), original));
    prepared::validate_ct_shape_values(*principal, original);
  }
  c::FileStreamReport report;
  for (auto &pipeline : pipelines) if (pipeline.valid()) {
    check(pipeline.drain()); const auto part = pipeline.report();
    report.staging_bytes += part.staging_bytes; report.metadata_bytes += part.metadata_bytes;
    report.peak_backing_bytes += part.peak_backing_bytes;
    report.io_calls += part.io_calls; report.mapping_copy_bytes += part.mapping_copy_bytes;
    report.copied_physical_bytes += part.copied_physical_bytes;
    report.native_calls += part.native_calls; report.native_waits += part.native_waits;
    report.phase_completions += part.phase_completions; report.host_waits += part.host_waits;
    report.native_metadata_unknown = report.native_metadata_unknown || part.native_metadata_unknown;
    pipeline = {};
  }
  WeightMeanings meanings;
  for (const auto &role : roles.selected) meanings.emplace(role.name, role.meaning);
  // Domain zero is the primary route, which is host memory in a CPU-only
  // build. These fields describe original storage, not physical VRAM usage.
  const auto device_bytes = backend.host_only() ? 0 : domain_bytes[0];
  const auto host_bytes = backend.host_only() ? domain_bytes[0] : domain_bytes[1];
  std::size_t canonical_bytes = 0;
  for (const auto &[name, weight] : weights) {
    (void)name; canonical_bytes = cache::add(canonical_bytes, take(weight.tensor.description.required_bytes()));
  }
  return {loaded, report, roles.identity, std::move(meanings), device_bytes, host_bytes,
      roles.selected.size() - physical_mtp_roles, physical_mtp_roles,
      canonical_bytes, fp8_groups, import_peak_bytes, roles.float8_policy.enabled,
      static_cast<std::size_t>(std::ranges::count(roles.selected, prepared::ProjectionCodec::kCtFloat8ChannelToken, &prepared::Role::projection_codec)),
      static_cast<std::size_t>(std::ranges::count(roles.selected, prepared::ProjectionCodec::kCtLocalNvfp4, &prepared::Role::projection_codec) +
          std::ranges::count(roles.selected, prepared::ProjectionCodec::kCtLocalNvfp4Bfloat16, &prepared::Role::projection_codec)),
      static_cast<std::size_t>(std::ranges::count(roles.selected, prepared::ProjectionCodec::kCtFloat8Block128, &prepared::Role::projection_codec)),
      (roles.cache_policy == prepared::CachePolicy::kBFloat16OmitCalibration ? roles.cache_calibration.size() : 0), nf4_groups, nf4_original_bytes, nf4_input_copy_bytes, nf4_known_group_peak_bytes};
}

WeightLoadReport load_weight_bank(const std::filesystem::path &manifest,
    Backend &backend, std::map<std::string, LoadedWeight> &weights,
    std::vector<c::LeasePool> &weight_pools,
    const ModelGeometry &geometry, WeightSelection selection = WeightSelection::kText,
    std::size_t cpu_ffn_layers = 0) {
  if (selection != WeightSelection::kText && selection != WeightSelection::kTextAndMtp)
    throw std::runtime_error("invalid weight selection");
  const auto roles = read_weight_roles(manifest, selection == WeightSelection::kTextAndMtp);
  if (roles.geometry != geometry) throw std::runtime_error("source geometry changed before weight loading");
  const FfnPlacement placement(roles, cpu_ffn_layers);
  return load_selected_weight_roles(roles, backend, weights, weight_pools, placement);
}

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
  cache::SnapshotRef head;
  hyle::container::SmallVector<c::PagedPageAppender, 32> tails;
  std::array<c::PagedPageAppender, 2> draft_tails;
  const cache::Snapshot *tail_head = nullptr;
  const cache::Snapshot *draft_tail_head = nullptr;
  StepState *pending = nullptr; // Borrowed from Model; episode pins this Session.
  bool failed = false, stopping = false;
  hyle::Status retirement_status;
  std::size_t length() const noexcept { return head ? head->length : 0; }
  const PreparedSampling &sampling_configuration() const noexcept { return sampling_; }
  std::weak_ptr<ColdResources> sampling_owner() const noexcept { return resources_; }
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  // Stopping any member aborts the whole shared batch. No head is published.
  hyle::Status stop() noexcept;
  inference::RetirementObservation try_retire() noexcept;
  inference::RetirementObservation drain() noexcept;
  ~Session();
};
struct EvaluationRequest final {
  std::shared_ptr<Session> session;
  std::span<const std::int32_t> ids;
  const inference::RequestStopPolicy *stop_policy = nullptr;
};
struct EvaluationOutput final {
  std::shared_ptr<Session> session;
  bool capture_tensors = false, capture_probes = false;
  double *complete_seconds = nullptr;
  Clock::time_point *host_endpoint = nullptr;
  inference::StepObservation *observation = nullptr;
  inference::PreparationObservation *preparation = nullptr;
  GenerationLedger *generation_ledger = nullptr;
  const GenerationCommitRecord *expected_episode = nullptr;
};
// Borrowed only during the synchronous post-commit observer. Original output
// leases and the carrier admission guard remain in finish's stack scope.
struct CompletedEvaluationView final {
  inference::StepObservation &observation;
  const cache::Snapshot &head;
  std::size_t member_index = 0, requests = 1;
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
struct DraftProgram final {
  inference::QueueBindings queues;
  std::size_t rows = 0;
  bool head = false;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  ResourceRetention retirement;
  V tokens, positions, key, value, metadata, hidden, id, identity, log_probabilities, valid, probability;
  std::vector<V> hidden_inputs, resource_ports;
  mtp::AppendNodes append;
  g::NodeId attention;
  Tensor token_values, position_values, identity_values;
  PreparedReadback id_value, valid_value, probability_value;
  c::LeasePool probability_pool;
  PreparedOutputFacts hidden_output, probabilities_output;
};
struct DraftWork final {
  cache::DraftState state;
  std::array<c::PagedPageAppender, 2> tails;
};
struct DraftProposal final {
  std::int32_t id = -1;
  float probability = 0;
  c::ReadLease hidden, log_probabilities;
};
struct PrefixSamplingProgram final {
  inference::QueueBindings queues;
  std::size_t proposals = 0;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::Invocation run;
  ResourceRetention retirement;
  V logits, identities, ids;
  std::vector<V> q, accepted, final_ids, valid;
  std::vector<V> resource_ports;
  Tensor identity_values, id_values;
  std::vector<PreparedReadback> accepted_values, final_values, valid_values;
};
struct PrefixSamplingResult final {
  std::vector<std::uint8_t> accepted;
  std::vector<std::int32_t> final_ids;
};

struct SamplingFrontier final {
  inference::QueueBindings queues;
  std::vector<Tensor> literals;
  g::Graph graph;
  g::CompiledGraph compiled;
  g::DimensionId last_row;
  V hidden_port, identity_port, id_port, valid_port;
  Tensor identity;
  PreparedReadback id, valid;
  ResourceRetention retirement;
};

struct PreparedSampling final {
  const std::size_t index;
  const SamplingConfiguration configuration;
  std::unique_ptr<SamplingFrontier> frontier;
  PreparedSampling(std::size_t ordinal, SamplingConfiguration value) : index(ordinal), configuration(value) {}
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
  WeightMeanings weight_meanings;
  std::string source_identity;
  std::vector<StateSchema> state_schema;
  const prepared::CachePolicy cache_policy;
  const std::vector<CacheStoragePair> attention_storage;
  c::ResourceBudget cache_budget, execution_budget, allocation_budget;
  cache::VersionCount snapshot_versions;
  bool carrier_borrowed = false;
  cache::PageInventory page_inventory, draft_inventory;
  std::vector<cache::StatePools> state_pools;
  c::LeasePool hidden_pool;
  std::vector<std::unique_ptr<PreparedSampling>> sampling_configurations;
  const bool retain_sampling_frontier;
  std::map<FamilyKey, std::unique_ptr<Family>> families, probe_families;
  std::map<ExecutionShape, std::unique_ptr<Phase>> phases, probe_phases;
  std::map<std::pair<std::size_t, bool>, std::unique_ptr<DraftProgram>> draft_programs;
  std::map<std::size_t, std::unique_ptr<PrefixSamplingProgram>> prefix_sampling_programs;

  explicit ColdResources(const ModelGeometry &geometry, std::size_t capacity, PrecisionProfile precision,
      SamplingConfiguration first, std::span<const SamplingConfiguration> additional,
      prepared::CachePolicy policy, std::vector<CacheStoragePair> storage)
      : state_schema(state_schema_for(geometry, capacity, precision)),
        cache_policy(policy), attention_storage(std::move(storage)), state_pools(geometry.layers.size()),
        retain_sampling_frontier(first.enabled || std::ranges::any_of(additional, [](const auto &value) { return value.enabled; })) {
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
  bool pending() const noexcept {
    for (const auto *cache : {&phases, &probe_phases})
      for (const auto &[shape, phase] : *cache) {
        (void)shape;
        for (const auto &pin : phase->retirement) if (pin.pending()) return true;
      }
    for (const auto &[shape, program] : draft_programs) {
      (void)shape; if (program->retirement.pending()) return true;
    }
    for (const auto &[width, program] : prefix_sampling_programs) {
      (void)width; if (program->retirement.pending()) return true;
    }
    return std::ranges::any_of(sampling_configurations, [](const auto &entry) {
      return entry->frontier && entry->frontier->retirement.pending();
    });
  }
  void release_invocation_roots() noexcept {
    // The accepted attachment is the only temporary scope->Graph->scope
    // cycle. Drop these application roots at handoff; the original native
    // Completion still retains its Graph until actual retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[shape, phase] : *cache) {
        (void)shape; for (auto &run : phase->slots) run = {};
      }
    for (auto &[shape, program] : draft_programs) { (void)shape; program->run = {}; }
    for (auto &[width, program] : prefix_sampling_programs) { (void)width; program->run = {}; }
  }
  inference::RetirementObservation retire(bool wait) noexcept {
    hyle::Status status;
    if (retirement_backend) {
      g::Invocation empty;
      status = wait ? retirement_backend->drain_retired(empty) : retirement_backend->collect_retired();
    }
    // A pending original guard is not an observation failure. Preserve actual
    // backend errors independently; neither fact substitutes for the other.
    return {!pending(), status};
  }
};

class Model final {
public:
  struct Sampling final {
    bool enabled = false;
    sampling::Policy policy;
    std::uint64_t seed = 0;
  };
  struct DraftPolicy final {
    std::size_t max_drafts = 1;
    float minimum_probability = 0;
  };
private:
  DraftPolicy draft_policy_;
  bool preparation_started_ = false;
public:
  const std::uint64_t sampling_seed;
  std::size_t max_drafts() const noexcept { return draft_policy_.max_drafts; }
  float min_draft_probability() const noexcept { return draft_policy_.minimum_probability; }
  void configure_draft_policy(DraftPolicy policy) {
    if (preparation_started_)
      throw std::runtime_error("configure draft policy before the first model preparation");
    if (!std::isfinite(policy.minimum_probability) || policy.minimum_probability < 0 || policy.minimum_probability > 1)
      throw std::runtime_error("MTP minimum draft probability must be finite and within [0,1]");
    if (policy.max_drafts >= capacity ||
        (cache_plan.draft_cache && policy.max_drafts >= cache_plan.candidate_boundaries))
      throw std::runtime_error("MTP maximum drafts exceed the original context or candidate capacity");
    draft_policy_ = policy;
  }
  bool recursive_mtp() const noexcept { return max_drafts() && (max_drafts() > 1 || min_draft_probability() > 0); }
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
  const SamplingConfiguration &default_sampler; // The immutable default, shared by original MTP paths.
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
  std::vector<bool> history_writers, recurrent_writers;
  bool hidden_writers = false;
  bool generated_state_outputs = false;
  const cache::Plan cache_plan;
  c::BatchSchema attention_schema;
  std::vector<c::BatchSchema> attention_schemas;
  const c::BatchSchema &target_attention_schema(std::size_t layer) const {
    if (layer >= target_kv_layers) throw std::runtime_error("invalid target attention cache slot");
    return attention_schemas.empty() ? attention_schema : attention_schemas.at(layer);
  }
  cache::VersionCount &snapshot_versions;
  bool &carrier_borrowed;
  static constexpr std::size_t slot_index = 0;
  cache::PageInventory &page_inventory, &draft_inventory;
  std::vector<cache::StatePools> &state_pools;
  c::LeasePool &hidden_pool;
  // Family/child owners are shared by sessions; no session addresses enter compile.
  std::map<FamilyKey, std::unique_ptr<Family>> &families, &probe_families;
  std::vector<FamilyKey> admitted_exact_keys;
  std::size_t exact_admissions = 0, exact_evictions = 0;
  // One carrier per actually prepared finite shape. Serial admission borrows
  // its one slot and shared IO through native retirement and all result reads.
  std::map<ExecutionShape, std::unique_ptr<Phase>> &phases, &probe_phases;
  std::map<ExecutionShape, std::array<std::size_t, execution_slot_count>> planned_slot_uses;
  std::map<std::pair<std::size_t, bool>, std::unique_ptr<DraftProgram>> &draft_programs;
  std::map<std::size_t, std::unique_ptr<PrefixSamplingProgram>> &prefix_sampling_programs;
  std::size_t draft_program_graph_calls = 0, prefix_decision_graph_calls = 0;
  const PreparedSampling &sampling_configuration(std::size_t index = 0) const { return resources_->sampling_configuration(index); }
  std::size_t sampling_configuration_count() const noexcept { return resources_->sampling_configurations.size(); }
  std::size_t sampling_index(const PreparedSampling *value) const { return resources_->sampling_index(value); }
  std::size_t sampling_index(const GenerationLedger &ledger) const {
    if (ledger.sampling && (ledger.sampling_owner.expired() || ledger.sampling_owner.owner_before(resources_) || resources_.owner_before(ledger.sampling_owner)))
      throw std::runtime_error("generation ledger belongs to another or destroyed model");
    return sampling_index(ledger.sampling);
  }
  bool retains_sampling_frontier() const noexcept {
    return resources_->retain_sampling_frontier;
  }
  std::vector<std::weak_ptr<Session>> session_links;
  std::size_t frontier_sampling_graph_calls = 0;
  double frontier_sampling_seconds = 0;
  // Destroy episode pins/borrowed links before carriers and the raw policy
  // owners referenced by Snapshot/PageRef, including exceptional shutdown.
  InputPreparationScratch input_scratch;
  std::unique_ptr<StepState> active_step;
  const std::size_t capacity;
  const ReplayPolicy replay_policy;
  const std::filesystem::path layer0_probe;
  std::filesystem::path trace_routes;
  explicit Model(const std::filesystem::path &manifest, const ModelGeometry &model_geometry, std::size_t selected_capacity,
                 cache::Plan selected_cache,
                 std::filesystem::path probe = {}, bool progress = false,
                  ReplayPolicy replay = ReplayPolicy::kAuto,
                  std::size_t max_cache_bytes = (std::numeric_limits<std::size_t>::max)(),
                  Sampling sampling_configuration = {false, {}, 0},
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
              SamplingConfiguration{sampling_configuration.enabled, sampling_configuration.policy}, additional_sampling,
              source.roles.cache_policy, std::move(source.cache_storage))),
          default_sampler(resources_->sampling_configurations[0]->configuration),
         weight_pools(resources_->weight_pools), weights(resources_->weights), projections(resources_->projections),
         weight_meanings(resources_->weight_meanings), source_identity(resources_->source_identity),
         state_schema(resources_->state_schema), cache_budget(resources_->cache_budget), execution_budget(resources_->execution_budget),
         allocation_budget(resources_->allocation_budget),
         history_writers(geometry.layers.size(), false), recurrent_writers(geometry.layers.size(), false),
         cache_plan(selected_cache), snapshot_versions(resources_->snapshot_versions), carrier_borrowed(resources_->carrier_borrowed),
         page_inventory(resources_->page_inventory), draft_inventory(resources_->draft_inventory),
         state_pools(resources_->state_pools), hidden_pool(resources_->hidden_pool),
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
    const bool calibrated = resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration;
    if ((calibrated && resources_->attention_storage.size() != target_kv_layers) ||
        (!calibrated && !resources_->attention_storage.empty()) || (calibrated && cache_plan.draft_cache))
      throw std::runtime_error("prepared E4 cache storage requires its complete ordinary target pairs");
    if (cache_plan.row_bytes != source.cache_row_bytes ||
        cache_plan.key_bank_bytes != cache::add(cache::multiply(cache::multiply(cache_plan.max_pages,
            cache::Plan::page_rows), source.cache_row_bytes.key), 1) ||
        cache_plan.value_bank_bytes != cache::add(cache::multiply(cache::multiply(cache_plan.max_pages,
            cache::Plan::page_rows), source.cache_row_bytes.value), 1))
      throw std::runtime_error("cache backing plan differs from prepared physical storage");
    if (cache_plan.hidden_row_bytes != geometry.hidden_bytes() || cache_plan.attention_layers != target_kv_layers ||
        cache_plan.recurrent_layers != geometry.recurrent_layers())
      throw std::runtime_error("cache layout differs from source model geometry");
    if (cache_plan.draft_cache && !additional_sampling.empty())
      throw std::runtime_error("nondefault sampling configurations are currently ordinary Text only");
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
    else if (execution.native_generated_storage() || execution.host_only() || ffn_placement.cpu_layers()) {
      // Native checked operations can own staging status/metadata buffers.
      // Register actual resource identities once, including CPU's shared
      // payload/metadata resource. Execution workspace has its own scope,
      // separate from retained session-cache storage.
      std::vector<c::ResourceBudgetLimit> native_limits;
      native_limits.reserve(3);
      for (const auto resource : {execution.device(), execution.staging(), mem::system_resource()})
        if (std::ranges::none_of(native_limits, [resource](const auto &limit) {
              return limit.resource == resource;
            }))
          native_limits.push_back({resource});
      if (ffn_placement.cpu_layers()) {
        const auto host = weight_resource(execution, true);
        if (std::ranges::none_of(native_limits, [host](const auto &limit) { return limit.resource == host; }))
          native_limits.push_back({host});
      }
      execution_budget = take(c::ResourceBudget::create({.limits = native_limits, .allow_unattributed = true}));
    }
    if (progress) {
      auto status = hyle::Status::Ok();
      scheduler = std::make_unique<hyle::sched::Scheduler>(1, mem::system_allocator(), 64, &status);
      check(status);
      workers = std::make_unique<hyle::sched::BlockingPool>(mem::system_allocator(), 1,
                                                          std::chrono::seconds(10), &status);
      check(status);
    }
    const auto weight_load_start = Clock::now();
    auto loading = load_selected_weight_roles(source_roles, execution, weights, weight_pools, ffn_placement, allocation_budget);
    source_identity = loading.identity;
    weight_meanings = std::move(loading.meanings);
    projections = prepare_projection_bindings(weights, cache_plan.draft_cache, geometry);
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
    // Stop the host driver before joining its original scheduler/pool. Native
    // cancellation remains advisory and never serves as retirement proof.
    for (auto &weak : session_links) if (auto session = weak.lock()) session->stop().IgnoreError();
    (void)abort_episode();
    // Attempt actual native retirement before detaching application links.
    // Scheduler idleness alone does not prove native/completion retirement.
    for (auto *cache : {&phases, &probe_phases})
      for (auto &[tokens, phase] : *cache) {
        (void)tokens;
        for (auto &run : phase->slots) execution.drain_retired(run).IgnoreError();
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
      if (active_step) inference::record_retirement_status(session->retirement_status, active_step->terminal);
    }
    resources_->release_invocation_roots();
    if (active_step) {
      active_step->candidate_admission = nullptr; // Host admission is gone.
      detach_episode();
    }
  }

  c::Completion submit_retained(g::Invocation &run, std::span<const g::QueueBinding> queues,
                                ResourceRetention &retention) {
    check(retention.reserve(resources_));
    auto accepted = run.submit(queues);
    if (!accepted) { retention.reject(); return take(std::move(accepted)); }
    check(retention.attach(*accepted));
    return std::move(*accepted);
  }

  std::string_view score_precision_policy() const {
    return score_policy(projections.at(std::string(geometry.head_role())), precision);
  }
  std::string weight_inventory_report() const {
    auto result = ",\"weight_count_scope\":\"selected_physical_roles\",\"text_tensors\":" +
        std::to_string(physical_text_roles) + ",\"mtp_tensors\":" + std::to_string(physical_mtp_roles);
    if (float8_weight_policy) result += ",\"canonical_weight_tensors\":" + std::to_string(weights.size()) +
        ",\"canonical_weight_bytes\":" + std::to_string(canonical_weight_bytes) +
        ",\"fp8_block128_groups\":" + std::to_string(fp8_weight_groups) +
        ",\"fp8_import_temporary_group_peak_bytes\":" + std::to_string(fp8_import_peak_bytes) +
        ",\"fp8_block128_policy\":{\"activation_scale\":\"divide\",\"activation_floor_f32_bits\":\"2edbe6ff\","
        "\"input_dtype\":\"BF16\",\"accumulation_dtype\":\"F32\",\"output_dtype\":\"BF16\","
        "\"weight_scale_conversion\":\"bf16_to_f32_exact\",\"reader\":\"" + std::string(prepared::fp8_reader) + "\"}";
    if (ct_channel_groups || ct_local_groups) result += ",\"ct_channel_token_groups\":" + std::to_string(ct_channel_groups) +
        ",\"ct_local_nvfp4_groups\":" + std::to_string(ct_local_groups);
    if (ct_block_groups) result += ",\"ct_block128_bf16_groups\":" + std::to_string(ct_block_groups);
    if (omitted_cache_roles) result += ",\"kv_cache_dtype\":\"BF16\",\"omitted_original_cache_calibration_roles\":" +
        std::to_string(omitted_cache_roles);
    if (resources_->cache_policy == prepared::CachePolicy::kFloat8OriginalCalibration)
      result += ",\"kv_cache_dtype\":\"E4M3FN\",\"original_cache_calibration_roles\":" +
          std::to_string(resources_->attention_storage.size() * 2);
    return result;
  }
  Tensor caller_tensor(mem::MemoryResourceRef resource, c::TensorDesc description, bool caller = true) const {
    return Tensor(resource, description, caller, allocation_budget);
  }

  c::ResourceBudgetUsage cache_usage() const {
    const auto usage = take(cache_budget.usage(execution.device()));
    const auto metadata = usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)];
    const auto payload = usage.live_bytes - metadata;
    if (payload < known_cache_backing_bytes ||
        (!generated_state_outputs && payload != known_cache_backing_bytes))
      throw std::runtime_error("actual cache backing differs: payload=" + std::to_string(payload) +
          " known=" + std::to_string(known_cache_backing_bytes) + " metadata=" + std::to_string(metadata));
    return usage;
  }

  // Report the same accounting scope for ordinary, MTP and scheduled runs.
  // Native execution allocates before a completed result is retained; its
  // temporaries/private allocations do not become a physical-memory cap here.
  std::string cache_report() const {
    const auto usage = cache_usage();
    std::ostringstream out;
    if (sampling_configuration_count() > 1) out << ",\"sampling_scope\":\"default_configuration\"";
    out << ",\"sampling\":\"" << (default_sampler.enabled ? "random" : "greedy") << '"'
        << ",\"sampling_policy_order\":\"temperature_top_k_top_p\""
        << ",\"sampling_temperature\":" << default_sampler.policy.temperature
        << ",\"sampling_top_k\":" << default_sampler.policy.top_k
        << ",\"sampling_top_p\":" << default_sampler.policy.top_p
        << ",\"sampling_seed\":" << sampling_seed
        << ",\"mtp_acceptance_policy\":\"" << (!cache_plan.draft_cache ? "disabled" : default_sampler.enabled ? "min_one_p_over_q" : "equal_greedy_ids") << '"'
        << ",\"frontier_sampling_graph_calls\":" << frontier_sampling_graph_calls
        << ",\"mtp_max_drafts\":" << (cache_plan.draft_cache ? max_drafts() : 0)
        << ",\"mtp_min_draft_probability\":" << (cache_plan.draft_cache ? min_draft_probability() : 0)
        << ",\"draft_program_graph_calls\":" << draft_program_graph_calls
        << ",\"prefix_decision_graph_calls\":" << prefix_decision_graph_calls
        << ",\"cold_exact_admissions\":" << exact_admissions
        << ",\"cold_exact_evictions\":" << exact_evictions
        << ",\"backend\":\"" << execution.name()
        << "\",\"cache_budget_scope\":\"shared_pools_retained_outputs_and_resource_metadata\""
        << ",\"cpu_ffn_layers\":" << ffn_placement.cpu_layers()
        << ",\"original_host_weight_bytes\":" << host_weight_bytes
        << ",\"original_device_weight_bytes\":" << device_weight_bytes
        << ",\"cache_shared_backing_bytes\":" << known_cache_backing_bytes
        << ",\"cache_retained_output_backing_bytes\":" << usage.live_bytes -
            usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] - known_cache_backing_bytes
        << ",\"cache_metadata_requested_bytes\":" <<
            usage.live_by_category[static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)]
        << ",\"cache_live_requested_bytes\":" << usage.live_bytes
        << ",\"cache_peak_requested_bytes\":" << usage.peak_bytes
        << ",\"cache_requested_byte_limit\":";
    if (usage.limit_bytes == SIZE_MAX) out << "null";
    else out << usage.limit_bytes;
    out << ",\"native_execution_private_bytes_unknown\":" <<
        ((execution.native_generated_storage() || execution.host_only() || ffn_placement.cpu_layers()) ? "true" : "false")
        << ",\"physical_vram_limit_known\":false";
    if (execution.host_only() || ffn_placement.cpu_layers()) {
      out << ",\"execution_shared_requested_byte_limit\":null"
          << ",\"workspace_limit_scope\":\"known_per_invocation_graph_workspace\""
          << ",\"workspace_domains\":[";
      const auto domain_count = execution.host_only() ? 1u : known_workspace_peak.size();
      for (std::size_t i = 0; i < domain_count; ++i) {
        if (i) out << ',';
        out << "{\"domain\":" << i << ",\"known_peak_bytes\":" << known_workspace_peak[i]
            << ",\"limit_bytes\":" << known_workspace_limits[i] << '}';
      }
      out << ']';
    }
    return out.str();
  }

  void record_execution_plan(const g::CompileReport &report, std::span<const g::ConstantBinding> constants) {
    if (execution.host_only()) {
      if (report.storage_domain_count() != 1 || report.target_count() != 1)
        throw std::runtime_error("CPU graph lost its single host allocation domain");
      const auto domain = take(report.storage_domain(0));
      if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
        throw std::runtime_error("CPU known workspace lacks its finite domain admission limit");
      known_workspace_peak[0] = (std::max)(known_workspace_peak[0], domain.workspace_bytes);
      known_workspace_limits[0] = domain.workspace_budget_bytes;
      std::cerr << "cpu_plan known_workspace_host=" << domain.workspace_bytes
          << " workspace_host_limit=" << domain.workspace_budget_bytes
          << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_memory_cap=0\n";
      return;
    }
    if (!ffn_placement.cpu_layers()) return;
    if (report.storage_domain_count() != 3 || report.target_count() != 2)
      throw std::runtime_error("CPU FFN graph lost its configured CPU/GPU domains");
    for (std::size_t i = 0; i < known_workspace_peak.size(); ++i) {
      const auto domain = take(report.storage_domain(i));
      if (domain.workspace_budget_bytes == SIZE_MAX || domain.workspace_bytes > domain.workspace_budget_bytes)
        throw std::runtime_error("hybrid known workspace lacks its finite domain admission limit");
      known_workspace_peak[i] = (std::max)(known_workspace_peak[i], domain.workspace_bytes);
      known_workspace_limits[i] = domain.workspace_budget_bytes;
    }
    std::size_t host_constants = 0;
    for (std::size_t i = 0; i < report.constant_count(); ++i) {
      const auto constant = take(report.constant(i));
      const auto source = std::ranges::find(constants, constant.value, &g::ConstantBinding::value);
      if (source == constants.end()) throw std::runtime_error("hybrid constant lost its original binding");
      if (source->tensor.storage().resource() != weight_resource(execution, true)) continue;
      if (constant.storage != g::StorageDomainId{1} || constant.materialized_bytes)
        throw std::runtime_error("CPU FFN constant acquired an unexpected canonical replica");
      for (std::size_t value = 0; value < report.physical_value_count(); ++value) {
        const auto physical = take(report.physical_value(value));
        if (physical.value == constant.value && physical.constant && physical.storage != g::StorageDomainId{1})
          throw std::runtime_error("CPU FFN constant escaped its host allocation domain");
      }
      ++host_constants;
    }
    std::cerr << "hybrid_plan host_constants=" << host_constants << " materialized_constant_bytes="
        << report.materialized_constant_bytes() << " activation_copy_bytes=" << report.copy_bytes()
        << " known_workspace_device=" << take(report.storage_domain(0)).workspace_bytes
        << " known_workspace_host=" << take(report.storage_domain(1)).workspace_bytes
        << " known_workspace_staging=" << take(report.storage_domain(2)).workspace_bytes
        << " native_private_unknown=1 shared_requested_byte_limit=unlimited physical_vram_cap=0\n";
  }

  void validate_state_output(const PreparedOutputFacts &output, const c::TensorDesc &expected) const {
    if (output.description != expected)
      throw std::runtime_error("prepared state output differs from its model state description");
    if (output.caller_owned() && take(execution.resource_for(output.storage)) != execution.device())
      throw std::runtime_error("prepared state output is outside its original state pool resource");
  }

  c::ReadLease retain_state_storage(g::Invocation &run, const PreparedOutputFacts &output) const {
    auto retained = inference::retain_output(run, output, cache_budget, allocation_budget);
    if (retained) return std::move(*retained);
    const auto usage = cache_usage();
    throw std::runtime_error("cache retained-state admission failed before snapshot publication: " +
        std::string(retained.error().message()) + "; live=" + std::to_string(usage.live_bytes) +
        ", limit=" + std::to_string(usage.limit_bytes) +
        "; adjust the exhausted cache/shared byte limit or reduce retained sessions/context; old snapshots remain valid");
  }

  void prepare_cache() {
    preparation_started_ = true;
    if (attention_schema.valid()) {
      return;
    }
    const auto storage = take(c::CacheStorage::plain(precision.activation_type()));
    auto prepared_schema = take(c::BatchSchema::create({
        .storage = c::BatchStorage::kPaged, .query_type = precision.activation_type(),
        .output_type = precision.activation_type(), .key_storage = storage, .value_storage = storage,
        .max_requests = cache_plan.max_requests, .max_query_rows = cache_plan.max_query_rows,
        .max_kv_rows = cache_plan.max_kv_rows, .max_pages = cache_plan.max_pages,
        .max_page_refs = cache_plan.max_page_refs, .query_heads = static_cast<std::uint32_t>(geometry.query_heads), .kv_heads = static_cast<std::uint32_t>(geometry.kv_heads),
        .qk_head_dim = static_cast<std::uint32_t>(geometry.key_depth), .value_head_dim = static_cast<std::uint32_t>(geometry.value_depth), .page_size = cache::Plan::page_rows},
        {.resource_budget = cache_budget}));
    std::vector<c::BatchSchema> prepared_schemas;
    prepared_schemas.reserve(resources_->attention_storage.size());
    for (const auto &pair : resources_->attention_storage)
      prepared_schemas.push_back(take(prepared_schema.with_cache_storage(pair[0], pair[1])));
    cache::PageInventory prepared_inventory, prepared_draft;
    prepared_inventory.prepare(cache_plan.max_pages, target_kv_layers);
    for (std::size_t i = 0; i < prepared_inventory.keys.size(); ++i) {
      const auto &schema = prepared_schemas.empty() ? prepared_schema : prepared_schemas[i];
      prepared_inventory.keys[i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_inventory.values[i] = take(c::PagedArena::create(schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    if (cache_plan.draft_cache) {
      prepared_draft.prepare(cache_plan.max_pages, 1);
      prepared_draft.keys[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kKey,
          execution.device(), {.max_backing_bytes = cache_plan.key_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
      prepared_draft.values[0] = take(c::PagedArena::create(prepared_schema, c::BatchCacheRole::kValue,
          execution.device(), {.max_backing_bytes = cache_plan.value_bank_bytes, .resource_budget = cache_budget}, allocation_budget));
    }
    const auto prepared_bytes = cache::multiply(cache::add(cache_plan.key_bank_bytes, cache_plan.value_bank_bytes),
        target_kv_layers + (cache_plan.draft_cache ? 1u : 0u));
    // No PageRef exists until this complete cold source tuple is published.
    // Leave the schema sentinel invalid on failure so released quota can retry.
    static_assert(std::is_nothrow_move_assignable_v<cache::PageInventory>);
    static_assert(std::is_nothrow_move_assignable_v<c::BatchSchema>);
    page_inventory = std::move(prepared_inventory);
    draft_inventory = std::move(prepared_draft);
    known_cache_backing_bytes = prepared_bytes;
    snapshot_versions.capacity = cache_plan.snapshot_slots;
    attention_schemas = std::move(prepared_schemas);
    attention_schema = std::move(prepared_schema);
  }

  void prepare_state_storage() {
    preparation_started_ = true;
    if (state_storage_prepared) return;
    prepare_sampling_frontier();
    // Generated results keep their original native owners. These pools cover
    // actual initial zero inputs, and only selected caller-written outputs.
    // Publish together so a failed allocation releases its temporary backing
    // and leaves this Model ready for a later admission attempt.
    std::vector<cache::StatePools> prepared_pools(geometry.layers.size());
    c::LeasePool prepared_hidden;
    auto prepared_bytes = known_cache_backing_bytes;
    const auto initial_slots = static_cast<std::uint32_t>(cache_plan.resident_sessions);
    for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
      const auto &description = state_schema[i]; auto &pools = prepared_pools[i];
      const auto history_slots = history_writers[i] ? cache_plan.state_slots : initial_slots;
      const auto recurrent_slots = recurrent_writers[i] ? cache_plan.state_slots : initial_slots;
      const auto history_bytes = cache::state_pool_backing_bytes(take(description.history.required_bytes()), history_slots);
      const auto recurrent_bytes = cache::state_pool_backing_bytes(take(description.recurrent.required_bytes()), recurrent_slots);
      pools.history = take(c::LeasePool::create(execution.device(), history_bytes,
          {.max_blocks = history_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      pools.recurrent = take(c::LeasePool::create(execution.device(), recurrent_bytes,
          {.max_blocks = recurrent_slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      prepared_bytes = cache::add(prepared_bytes, cache::add(history_bytes, recurrent_bytes));
    }
    if (cache_plan.hidden_bytes) {
      const auto slots = hidden_writers ? cache_plan.hidden_slots : initial_slots;
      const auto bytes = cache::state_pool_backing_bytes(geometry.hidden_bytes(), slots);
      prepared_hidden = take(c::LeasePool::create(execution.device(), bytes,
          {.max_blocks = slots, .backing_alignment = cache::state_pool_alignment,
           .resource_budget = cache_budget}, allocation_budget));
      prepared_bytes = cache::add(prepared_bytes, bytes);
    }
    const auto usage = take(cache_budget.usage(execution.device()));
    if (usage.live_bytes - usage.live_by_category[
            static_cast<std::size_t>(c::ResourceBudgetCategory::kMetadata)] != prepared_bytes)
      throw std::runtime_error("shared cache backing inventory differs from the actual reserved bytes");
    state_pools = std::move(prepared_pools); hidden_pool = std::move(prepared_hidden);
    known_cache_backing_bytes = prepared_bytes;
    state_storage_prepared = true;
  }

  void prepare_family(std::span<const std::size_t> planned_chunks, bool diagnostic = false,
                      std::size_t requests = 1, GraphRole role = GraphRole::kText,
                      bool admitted_exact = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && sampling_index) throw std::runtime_error("nondefault sampling is ordinary Text only");
    if ((completion != TextCompletion::kEmitNext && completion != TextCompletion::kStateOnly) ||
        (completion == TextCompletion::kStateOnly && role != GraphRole::kText))
      throw std::runtime_error("state-only completion requires an ordinary Text family");
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
        const std::array chunk{width}; prepare_family(chunk, diagnostic, requests, role);
      }
      return;
    }
    const std::size_t prefix_rows = role == GraphRole::kVerify ? planned_chunks.front()
        : drafts_token(role) ? 2 : 1;
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
    const auto family_key = admitted_exact ? FamilyKey::exact(requests, role, planned_chunks.front(), completion, sampling_index) : FamilyKey(requests, role, prefix_rows, completion, sampling_index);
    auto &selected = diagnostic ? probe_families[family_key] : families[family_key];
    if (selected) throw std::runtime_error("text finite family was already prepared");
    auto result = std::make_unique<Family>();
    auto &definition = *result;
    definition.layers.resize(geometry.layers.size()); definition.attention_nodes.resize(target_kv_layers);
    definition.append_nodes.resize(draft_bank + 2);
    definition.sampling = &prepared_sampling;
    definition.requests = requests; definition.role = role; definition.prefix_rows = prefix_rows; definition.completion = completion;
    definition.admitted_exact = admitted_exact;
    definition.state_resource_ports.reserve(ports_per_request * requests);
    if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) {
      definition.carry_ports.resize(requests);
      for (auto &ports : definition.carry_ports) ports.resize(prefix_rows);
    }
    for (const auto tokens : planned_chunks) {
      if (tokens == 0 || tokens > capacity)
        throw std::runtime_error("finite text chunk outside nonempty context capacity");
      if (std::find(definition.chunks.begin(), definition.chunks.end(), tokens) == definition.chunks.end())
        definition.chunks.push_back(tokens);
    }
    const auto start = Clock::now();
    const auto max_query_rows = *std::max_element(definition.chunks.begin(), definition.chunks.end());
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, max_query_rows, requests, &weight_meanings);
    const auto t = r.token_rows, packed = r.packed_rows;
    const auto batch = f(static_cast<std::int64_t>(requests));
    definition.last_row = r.last_row;
    if (requests > cache_plan.active_sequences || cache::multiply(requests, max_query_rows) > cache_plan.max_query_rows)
      throw std::runtime_error("finite family exceeds the configured batch capacity");
    prepare_cache();
    definition.attention_schema = attention_schema;
    definition.token_port = r.input(r.dense_spec(Type::kInt32,
        {has_mtp(role) && !is_mtp_prefill(role) ? batch : packed}));
    definition.position_port = requests == 1 ? r.input(r.dense_spec(Type::kInt32, {t})) :
        r.input(r.dense_spec(Type::kInt32, {batch, t}));
    if (requests != 1 && !verifies_prefix(role)) definition.last_ids_port = r.input(r.dense_spec(Type::kInt32, {batch}));
    definition.metadata_port = r.input(take(definition.attention_schema.metadata_description()));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    V target_tokens = definition.token_port;
    std::vector<sampling::Distribution> draft_distributions;
    std::vector<V> proposal_ids;
    if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) definition.sampling_identity_port = r.input(
        dense(Type::kUInt32, {static_cast<std::int64_t>(requests), 3, 4}));
    const auto sampling_identity = [&](std::size_t request, std::size_t draw) {
      return r.copy(r.reshape(r.slice(r.slice(definition.sampling_identity_port, 0,
          f(static_cast<std::int64_t>(request)), f(1)), 1,
          f(static_cast<std::int64_t>(draw)), f(1)), {f(1), f(4)}));
    };
    const auto sampling_row = [&](V values, std::size_t row) {
      return r.copy(r.slice(values, 0, f(static_cast<std::int64_t>(row)), f(1)));
    };
    D draft_sequence_rows = f(0), draft_packed_rows = f(0);
    if (has_mtp(role)) {
      std::vector<V> carries, priors;
      definition.draft_hidden_inputs.resize(requests);
      for (auto &inputs : definition.draft_hidden_inputs) {
        inputs[0] = r.input(dense(precision.activation_type(), {1, hidden}));
        definition.state_resource_ports.push_back(inputs[0]); carries.push_back(inputs[0]);
        if (draft_lag(role)) {
          inputs[1] = r.input(dense(precision.activation_type(), {1, hidden}));
          definition.state_resource_ports.push_back(inputs[1]); priors.push_back(inputs[1]);
        }
      }
      definition.carry_input_port = r.concat(carries, 0, r.dense_spec(precision.activation_type(), {batch, f(hidden)}));
      const auto draft_rows = is_mtp_prefill(role)
          ? (draft_lag(role) ? D(take(r.builder.dimension_sum(t, f(1)))) : t)
          : f(static_cast<std::int64_t>(draft_lag(role) + 1));
      const auto packed_draft_rows = !draft_lag(role) && is_mtp_prefill(role) ? packed : requests == 1 ? draft_rows :
          draft_rows.is_fixed() ? f(static_cast<std::int64_t>(cache::multiply(requests, static_cast<std::size_t>(draft_rows.fixed_value())))) :
          D(take(r.builder.dimension_product(draft_rows, batch)));
      draft_sequence_rows = draft_rows; draft_packed_rows = packed_draft_rows;
      definition.draft_position_port = r.input(requests == 1 ? r.dense_spec(Type::kInt32, {draft_rows})
          : r.dense_spec(Type::kInt32, {batch, draft_rows}));
      if (draft_lag(role)) {
        definition.prior_input_port = r.concat(priors, 0, r.dense_spec(precision.activation_type(), {batch, f(hidden)}));
        definition.catchup_token_port = r.input(r.dense_spec(Type::kInt32, {batch}));
      }
      if (drafts_token(role) || !is_mtp_prefill(role)) {
        V ids = definition.token_port, hidden_rows = definition.carry_input_port;
        if (draft_lag(role)) {
          if (requests == 1) {
            ids = r.one(c::Concat{.axis = 0}, {definition.catchup_token_port, ids}, r.dense_spec(Type::kInt32, {f(2)}));
            hidden_rows = r.one(c::Concat{.axis = 0}, {definition.prior_input_port, hidden_rows}, r.dense_spec(precision.activation_type(), {f(2), f(hidden)}));
          } else {
            ids = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.catchup_token_port, {batch, f(1)}), r.reshape(ids, {batch, f(1)})},
                r.dense_spec(Type::kInt32, {batch, f(2)})), {packed_draft_rows});
            hidden_rows = r.reshape(r.one(c::Concat{.axis = 1},
                {r.reshape(definition.prior_input_port, {batch, f(1), f(hidden)}), r.reshape(hidden_rows, {batch, f(1), f(hidden)})},
                r.dense_spec(precision.activation_type(), {batch, f(2), f(hidden)})), {packed_draft_rows, f(hidden)});
          }
        }
        const auto projection = mtp::input_projection(r, ids, hidden_rows);
        const mtp::Geometry row_geometry{requests, draft_rows, packed_draft_rows};
        const auto kv = mtp::project_kv(r, projection, definition.draft_position_port, frequencies, row_geometry);
        mtp::AppendNodes append;
        if (drafts_token(role)) {
          definition.draft_key_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
          definition.draft_value_port = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
          definition.draft_metadata_port = r.input(take(attention_schema.metadata_description()));
          const auto head = mtp::full_head(r, projection, kv, definition.draft_position_port, frequencies,
              {attention_schema, definition.draft_key_port, definition.draft_value_port, definition.draft_metadata_port},
              f(static_cast<std::int64_t>(draft_lag(role))), row_geometry);
          append = head.append_nodes; definition.draft_attention_node = head.attention_node;
          if (!selected_sampler.enabled) definition.draft_id_port = r.op(c::Operation(c::TopK{.count = 1}), {head.logits},
              {r.dense_spec(Type::kFloat32, {batch, f(1)}), r.dense_spec(Type::kInt32, {batch, f(1)})})[1];
          if (selected_sampler.enabled) {
            for (std::size_t request = 0; request < requests; ++request) {
              draft_distributions.push_back(sampling::distribution(r, sampling_row(head.logits, request), selected_sampler.policy));
              proposal_ids.push_back(r.one(c::sampling_ops::categorical(),
                  {draft_distributions.back().log_probabilities, sampling_identity(request, 0)}));
            }
            definition.draft_id_port = r.reshape(r.concat(proposal_ids, 0,
                r.dense_spec(Type::kInt32, {batch})), {batch, f(1)});
          }
          r.publish(definition.draft_id_port);
          target_tokens = requests == 1 ? r.one(c::Concat{.axis = 0},
              {definition.token_port, r.reshape(definition.draft_id_port, {f(1)})}, r.dense_spec(Type::kInt32, {packed})) :
              r.reshape(r.one(c::Concat{.axis = 1},
                  {r.reshape(definition.token_port, {batch, f(1)}), r.reshape(definition.draft_id_port, {batch, f(1)})},
                  r.dense_spec(Type::kInt32, {batch, f(2)})), {packed});
        } else append = mtp::append_kv(r, attention_schema, kv);
        definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
      }
    }
    auto x = r.embedding(target_tokens, "model.language_model.embed_tokens.weight");
    // Writer-only draft maintenance has no attention data edge into target.
    // Order its real cache effects before target's first append as well.
    g::NodeId prior_append = has_mtp(role) && !is_mtp_prefill(role)
        ? definition.append_nodes[draft_bank + 1] : g::NodeId{};
    bool has_prior_append = has_mtp(role) && !is_mtp_prefill(role);
    for (std::size_t i = 0; i < state_schema.size(); ++i) {
      const auto prefix = "layers." + std::to_string(i) + ".";
      auto normalized = r.norm(x, prefix + "input_layernorm.weight");
      auto &ports = definition.layers[i];
      V mixed;
      if (geometry.is_gdn(i)) {
        ports.sequences.resize(requests);
        std::vector<V> history_inputs, recurrent_inputs;
        history_inputs.reserve(requests); recurrent_inputs.reserve(requests);
        for (auto &sequence : ports.sequences) {
          sequence.history = r.input(state_schema[i].history);
          sequence.recurrent = r.input(state_schema[i].recurrent);
          history_inputs.push_back(sequence.history);
          recurrent_inputs.push_back(sequence.recurrent);
        }
        const auto history = r.concat(history_inputs, 0,
            r.dense_spec(precision.activation_type(), {batch, f(geometry.conv_kernel - 1), f(geometry.gdn_qkv_width())}));
        const auto state = r.concat(recurrent_inputs, 0,
            r.dense_spec(Type::kFloat32, {batch, f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)}));
        auto qkv = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_qkv.weight"), {batch, t, f(geometry.gdn_qkv_width())});
        auto samples = r.one(c::Concat{.axis = 1}, {history, qkv},
            r.dense_spec(precision.activation_type(), {batch, r.sample_rows, f(geometry.gdn_qkv_width())}));
        // Every published state has independent per-session material storage.
        // R1 preserves its original Copy and direct GDN state result.
        const auto next_history = r.slice(samples, 1, t, f(geometry.conv_kernel - 1));
        for (std::size_t request = 0; request < requests; ++request) {
          auto &sequence = ports.sequences[request];
          sequence.history_prefix.resize(prefix_rows);
          sequence.recurrent_prefix.resize(prefix_rows);
          sequence.history_prefix.back() = r.copy(requests == 1 ? next_history :
              r.slice(next_history, 0, f(static_cast<std::int64_t>(request)), f(1)));
          for (std::size_t row = 0; row + 1 < prefix_rows; ++row) {
            auto selected_history = r.slice(samples, 1, f(static_cast<std::int64_t>(row + 1)), f(geometry.conv_kernel - 1));
            if (requests != 1) selected_history = r.slice(selected_history, 0, f(static_cast<std::int64_t>(request)), f(1));
            sequence.history_prefix[row] = r.copy(selected_history);
          }
        }
        auto convolution = r.one(c::model_ops::causal_conv1d(), {r.cast(samples, Type::kFloat32),
            r.cast(r.reshape(r.weight(prefix + "linear_attn.conv1d.weight"), {f(geometry.gdn_qkv_width()), f(geometry.conv_kernel)}), Type::kFloat32)},
            r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_qkv_width())}));
        auto activated = r.unary(c::UnaryKind::kSilu, r.cast(convolution, precision.activation_type()));
        auto q = r.reshape(r.slice(activated, 2, f(0), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto k = r.reshape(r.slice(activated, 2, f(geometry.gdn_key_width()), f(geometry.gdn_key_width())), {batch, t, f(geometry.gdn_key_heads), f(geometry.gdn_key_depth)});
        auto v = r.reshape(r.slice(activated, 2, f(2 * geometry.gdn_key_width()), f(geometry.gdn_value_width())), {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        auto a = r.reshape(r.cast(r.linear(normalized, prefix + "linear_attn.in_proj_a.weight"), Type::kFloat32), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto b = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_b.weight"), {batch, t, f(geometry.gdn_value_heads), f(1)});
        auto dt = r.reshape(r.cast(r.weight(prefix + "linear_attn.dt_bias"), Type::kFloat32), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto decay = r.reshape(r.decay("model.language_model." + prefix + "linear_attn.A_log"), {f(1), f(1), f(geometry.gdn_value_heads), f(1)});
        auto g = r.binary(c::BinaryKind::kMultiply, decay,
            r.unary(c::UnaryKind::kSoftplus, r.binary(c::BinaryKind::kAdd, a, dt)));
        auto beta = r.cast(r.unary(c::UnaryKind::kSigmoid, b), Type::kFloat32);
        auto recurrent = r.op(c::Operation(c::GatedDeltaNet{
            .snapshot_count = static_cast<std::uint32_t>(prefix_rows)}),
            {r.cast(r.gdn_key_rows(r.l2(q)), Type::kFloat32), r.cast(r.gdn_key_rows(r.l2(k)), Type::kFloat32),
             r.cast(v, Type::kFloat32), g, beta, state},
            {r.dense_spec(Type::kFloat32, {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)}),
             prefix_rows > 1
                 ? r.dense_spec(Type::kFloat32, {batch, f(static_cast<std::int64_t>(prefix_rows)), f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)}) : r.spec(state)});
        for (std::size_t request = 0; request < requests; ++request) {
          if (prefix_rows > 1) {
            auto per_request = r.slice(recurrent[1], 0, f(static_cast<std::int64_t>(request)), f(1));
            for (std::size_t row = 0; row < prefix_rows; ++row)
              ports.sequences[request].recurrent_prefix[row] = r.copy(r.reshape(
                  r.slice(per_request, 1, f(static_cast<std::int64_t>(row)), f(1)), {f(1), f(geometry.gdn_value_heads), f(geometry.gdn_key_depth), f(geometry.gdn_value_depth)}));
          } else ports.sequences[request].recurrent_prefix[0] = requests == 1 ? recurrent[1] :
              r.copy(r.slice(recurrent[1], 0, f(static_cast<std::int64_t>(request)), f(1)));
        }
        auto y = r.cast(recurrent[0], precision.activation_type());
        if (i == 0) definition.probe_ports[0] = y;
        const auto y_f32 = r.cast(y, Type::kFloat32);
        y = r.cast(r.same(c::RmsNorm{.epsilon = geometry.epsilon}, {y_f32}, y_f32), precision.activation_type());
        // Original storage is F32 in the official checkpoint and BF16 in
        // these quantized artifacts. Preserve its actual product rounding.
        const auto norm_scale = r.weight(prefix + "linear_attn.norm.weight");
        y = r.binary(c::BinaryKind::kMultiply, r.cast(y, r.spec(norm_scale).data_type()), norm_scale);
        auto z = r.reshape(r.linear(normalized, prefix + "linear_attn.in_proj_z.weight"), {batch, t, f(geometry.gdn_value_heads), f(geometry.gdn_value_depth)});
        y = r.cast(r.binary(c::BinaryKind::kMultiply, r.cast(y, Type::kFloat32),
            r.unary(c::UnaryKind::kSilu, r.cast(z, Type::kFloat32))), precision.activation_type());
        auto projection_input = r.reshape(y, {packed, f(geometry.gdn_value_width())});
        if (i == 0) definition.probe_ports[1] = y; // Same BF16 payload before its metadata-only out_proj reshape.
        mixed = r.linear(projection_input, prefix + "linear_attn.out_proj.weight");
        if (i == 0) definition.probe_ports[2] = mixed;
        for (const auto &sequence : ports.sequences) {
          // Preserve the default T2 publication order (final, then earlier),
          // with independently releasable storage for every possible prefix.
          for (auto row = prefix_rows; row != 0; --row) {
            r.publish(sequence.history_prefix[row - 1]); r.publish(sequence.recurrent_prefix[row - 1]);
            if (row != prefix_rows) {
              definition.state_resource_ports.push_back(sequence.history_prefix[row - 1]);
              definition.state_resource_ports.push_back(sequence.recurrent_prefix[row - 1]);
            }
          }
          for (const auto port : {sequence.history, sequence.recurrent, sequence.history_prefix.back(), sequence.recurrent_prefix.back()})
            definition.state_resource_ports.push_back(port);
        }
      } else {
        const auto layer = geometry.attention_slot(i);
        const auto &schema = target_attention_schema(layer);
        ports.key = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kKey)));
        ports.value = r.input(take(c::batch_cache_description(schema, c::BatchCacheRole::kValue)));
        auto projected = r.reshape(r.linear(normalized, prefix + "self_attn.q_proj.weight"), {batch, t, f(geometry.query_heads), f(geometry.key_depth + geometry.value_depth)});
        auto q = r.norm(r.slice(projected, 3, f(0), f(geometry.key_depth)), prefix + "self_attn.q_norm.weight");
        auto gate = r.reshape(r.copy(r.slice(projected, 3, f(geometry.key_depth), f(geometry.value_depth))), {packed, f(geometry.attention_width())});
        auto k = r.norm(r.reshape(r.linear(normalized, prefix + "self_attn.k_proj.weight"), {batch, t, f(geometry.kv_heads), f(geometry.key_depth)}), prefix + "self_attn.k_norm.weight");
        auto v = r.linear(normalized, prefix + "self_attn.v_proj.weight");
        const c::RoPE rope{.rotary_dimensions = static_cast<std::size_t>(geometry.rotary), .sequence_axis = 1, .layout = c::RopeLayout::kHalfSplit};
        q = r.same(rope, {q, definition.position_port, frequencies}, q);
        k = r.same(rope, {k, definition.position_port, frequencies}, k);
        auto &key_append = definition.append_nodes[layer * 2];
        auto &value_append = definition.append_nodes[layer * 2 + 1];
        auto key_rows = r.reshape(k, {packed, f(geometry.kv_heads), f(geometry.key_depth)});
        auto value_rows = r.reshape(v, {packed, f(geometry.kv_heads), f(geometry.value_depth)});
        const auto cache_capacity = schema.capacity();
        if (cache_capacity.key_storage.data_type() == Type::kFloat8E4M3FN)
          key_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.key_storage}),
              {key_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.key_depth)}));
        if (cache_capacity.value_storage.data_type() == Type::kFloat8E4M3FN)
          value_rows = r.one(c::Operation::custom(c::model_ops::CacheEncode{cache_capacity.value_storage}),
              {value_rows}, r.dense_spec(Type::kFloat8E4M3FN, {packed, f(geometry.kv_heads), f(geometry.value_depth)}));
        r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kKey))),
            {key_rows}, {}, &key_append);
        r.op(c::Operation::custom(take(g::batch::paged_append(schema, c::BatchCacheRole::kValue))),
            {value_rows}, {}, &value_append);
        if (has_prior_append) check(r.builder.depends_on(key_append, prior_append));
        check(r.builder.depends_on(value_append, key_append));
        prior_append = value_append; has_prior_append = true;
        auto attention = r.one(c::Operation::custom(c::PagedAttention{
            .capacity = cache_capacity, .scale = 1.F / std::sqrt(static_cast<float>(geometry.key_depth)), .causal = true}),
            {r.reshape(q, {packed, f(geometry.query_heads), f(geometry.key_depth)}), ports.key, ports.value, definition.metadata_port},
            r.dense_spec(precision.activation_type(), {packed, f(geometry.query_heads), f(geometry.value_depth)}), &definition.attention_nodes[layer]);
        check(r.builder.depends_on(definition.attention_nodes[layer], key_append));
        check(r.builder.depends_on(definition.attention_nodes[layer], value_append));
        auto gated = r.binary(c::BinaryKind::kMultiply, r.reshape(attention, {packed, f(geometry.attention_width())}), r.unary(c::UnaryKind::kSigmoid, gate));
        mixed = r.linear(gated, prefix + "self_attn.o_proj.weight");
      }
      const auto ffn_input = r.binary(c::BinaryKind::kAdd, x, mixed);
      {
        GraphAssembly::Scope scope(assembly, ffn_placement.ffn_on_cpu(i) ? assembly.cpu() : assembly.primary());
        x = r.ffn(ffn_input, prefix, i == 0 ? std::span<V>(definition.probe_ports) : std::span<V>{});
      }
      if (i == 0) definition.probe_ports[4] = x;
    }
    if (diagnostic) for (std::size_t i = 0; i < definition.probe_ports.size(); ++i) {
      r.publish(definition.probe_ports[i]);
      auto description = r.spec(definition.probe_ports[i]);
      if (description.data_type() != precision.activation_type()) throw std::runtime_error("layer0 diagnostic differs from source activation precision");
    }
    // The selected child decides caller-written versus generated result storage.
    const auto normalized_last = r.norm(x, "norm.weight");
    if (is_mtp_prefill(role)) {
      const auto kv = mtp::prefill_kv(r, definition.token_port, normalized_last,
          definition.carry_input_port, definition.prior_input_port, definition.catchup_token_port,
          mtp::Geometry{requests, t, packed}, mtp::Geometry{requests, draft_sequence_rows, draft_packed_rows}, definition.draft_position_port,
          frequencies, draft_lag(role) != 0);
      const auto append = mtp::append_kv(r, attention_schema, kv);
      if (has_prior_append) check(r.builder.depends_on(append.key, prior_append));
      definition.append_nodes[draft_bank] = append.key; definition.append_nodes[draft_bank + 1] = append.value;
    }
    auto last = verifies_prefix(role) ? r.copy(normalized_last) :
        (requests == 1 ? r.copy(r.slice(normalized_last, 0, D(r.last_row), f(1))) :
        r.one(c::Gather{}, {normalized_last, definition.last_ids_port},
            r.dense_spec(precision.activation_type(), {batch, f(hidden)})));
    if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) for (std::size_t request = 0; request < requests; ++request)
      for (std::size_t row = 0; row < prefix_rows; ++row) {
        const auto carry = r.copy(r.slice(normalized_last, 0,
            verifies_prefix(role) ? f(static_cast<std::int64_t>(request * prefix_rows + row)) :
                (requests == 1 ? D(r.last_row) : D(take(r.builder.dimension_sum(
                    D(take(r.builder.dimension_product(t, f(static_cast<std::int64_t>(request))))), D(r.last_row))))), f(1)));
        definition.carry_ports[request][row] = carry;
        r.publish(carry); definition.state_resource_ports.push_back(carry);
      }
    definition.hidden_port = last;
    if (completion == TextCompletion::kEmitNext) {
      definition.logits_port = r.scores(last, std::string(geometry.head_role()));
      if (!selected_sampler.enabled || role == GraphRole::kVerify) definition.id_port = r.op(c::Operation(c::TopK{.count = 1}), {definition.logits_port},
          {r.dense_spec(Type::kFloat32, {verifies_prefix(role) ? packed : batch, f(1)}),
           r.dense_spec(Type::kInt32, {verifies_prefix(role) ? packed : batch, f(1)})})[1];
      else {
        std::vector<V> ids, valid, accepted;
        for (std::size_t request = 0; request < requests; ++request) {
          const auto p = sampling::distribution(r, sampling_row(definition.logits_port,
              verifies_prefix(role) ? request * 2 : request), selected_sampler.policy);
          if (drafts_token(role)) {
            const auto bonus = sampling::distribution(r, sampling_row(definition.logits_port, request * 2 + 1), selected_sampler.policy);
            const auto decision = sampling::decision(r, p.log_probabilities,
                draft_distributions[request].log_probabilities, bonus.log_probabilities,
                proposal_ids[request], sampling_identity(request, 1), sampling_identity(request, 2));
            ids.push_back(r.one(c::Select{}, {decision.accepted, proposal_ids[request], decision.final_id}));
            ids.push_back(decision.final_id); accepted.push_back(decision.accepted);
            auto ok = r.one(c::Select{}, {p.valid, draft_distributions[request].valid, r.cast(r.scalar(0), Type::kBool)});
            ok = r.one(c::Select{}, {ok, bonus.valid, r.cast(r.scalar(0), Type::kBool)});
            valid.push_back(r.one(c::Select{}, {ok, decision.valid, r.cast(r.scalar(0), Type::kBool)}));
          } else {
            ids.push_back(r.one(c::sampling_ops::categorical(), {p.log_probabilities, sampling_identity(request, 0)}));
            valid.push_back(p.valid);
          }
        }
        definition.id_port = r.reshape(r.concat(ids, 0, r.dense_spec(Type::kInt32,
            {f(static_cast<std::int64_t>(ids.size()))})), {verifies_prefix(role) ? packed : batch, f(1)});
        definition.sampling_valid_port = r.host_predicate(r.concat(valid, 0, r.dense_spec(Type::kBool, {batch})));
        r.publish(definition.sampling_valid_port);
        if (drafts_token(role)) {
          definition.sampling_accept_port = r.host_predicate(r.concat(accepted, 0, r.dense_spec(Type::kBool, {batch})));
          r.publish(definition.sampling_accept_port);
        }
      }
    }
    r.publish(definition.hidden_port);
    if (completion == TextCompletion::kEmitNext) { r.publish(definition.logits_port); r.publish(definition.id_port); }
    definition.graph = take(std::move(r.builder).finish());
    definition.literals = std::move(assembly.literals);
    definition.specs = std::move(r.specs);
    std::vector<std::array<g::ShapeBinding, 1>> bindings;
    bindings.reserve(definition.chunks.size());
    for (const auto tokens : definition.chunks)
      bindings.push_back({g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(tokens - 1)}});
    std::vector<g::ExactShapeVariant> variants;
    variants.reserve(bindings.size());
    for (const auto &point : bindings) variants.push_back({point});
    g::CompileOptions compile_options{.shape_policy = g::CompileShapePolicy::kFiniteExact,
        .exact_variants = variants, .constants = assembly.constants};
    // The library's cold operation admission cap covers the whole family.
    // Budget the requested finite children explicitly at the per-graph default;
    // this is host search/storage policy, not device or execution capacity.
    const auto per_graph_operations = g::CompileOptions{}.max_operation_proposals;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_operations)
      throw std::runtime_error("finite text operation proposal budget overflows");
    compile_options.max_operation_proposals = per_graph_operations * variants.size();
    const auto per_graph_preparations = g::CompileOptions{}.max_placement_attempts;
    if (variants.size() > (std::numeric_limits<std::size_t>::max)() / per_graph_preparations)
      throw std::runtime_error("finite text artifact preparation budget overflows");
    compile_options.max_placement_attempts = per_graph_preparations * variants.size();
    std::cerr << "compile full text finite family requests=" << requests << " variants=" << variants.size()
              << " max_query_rows=" << max_query_rows << " nodes=" << definition.graph.node_count()
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts << '\n';
    compile_options.resource_budget = execution_budget;
    definition.compiled = take(execution.compile(definition.graph, assembly.constants, assembly.options(compile_options)));
    const auto storage = definition.compiled.report.total_compiled_storage();
    std::cerr << "compiled text finite family variants=" << definition.compiled.report.exact_variant_count()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " max_operation_proposals=" << compile_options.max_operation_proposals
              << " operation_proposals=" << definition.compiled.report.total_search().operation_proposals
              << " max_placement_attempts=" << compile_options.max_placement_attempts
              << " placement_attempts=" << definition.compiled.report.total_search().placement_attempts
              << " cold_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    definition.children.reserve(definition.chunks.size());
    for (std::size_t index = 0; index < definition.chunks.size(); ++index) {
      auto child = std::make_unique<FamilyChild>();
      child->compiled.executable = take(definition.compiled.executable.exact_variant(index));
      child->compiled.report = take(definition.compiled.report.exact_variant(index));
      record_execution_plan(child->compiled.report, assembly.constants);
      report_packed_projections(definition.graph, child->compiled.report, r.packed_projection_nodes,
          child->compiled.executable.workspace_bytes());
      child->outputs.resize(child->compiled.executable.output_count());
      child->output_indices.resize(definition.specs.size());
      check(inference::prepare_outputs(child->compiled, child->outputs, child->output_indices));
      for (std::size_t layer = 0; layer < state_schema.size(); ++layer) if (geometry.is_gdn(layer))
        for (const auto &ports : definition.layers[layer].sequences) {
          for (std::size_t row = 0; row < prefix_rows; ++row) {
            const auto &history = child->output(ports.history_prefix[row]);
            const auto &recurrent = child->output(ports.recurrent_prefix[row]);
            validate_state_output(history, state_schema[layer].history);
            validate_state_output(recurrent, state_schema[layer].recurrent);
            generated_state_outputs = generated_state_outputs || !history.caller_owned() || !recurrent.caller_owned();
            history_writers[layer] = history_writers[layer] || history.caller_owned();
            recurrent_writers[layer] = recurrent_writers[layer] || recurrent.caller_owned();
          }
        }
      for (const auto &ports : definition.carry_ports) for (const auto value : ports) {
        if (!value.valid()) continue;
        const auto &output = child->output(value);
        validate_state_output(output, dense(precision.activation_type(), {1, hidden}));
        hidden_writers = hidden_writers || output.caller_owned();
        generated_state_outputs = generated_state_outputs || !output.caller_owned();
      }
      definition.children.push_back(std::move(child));
    }
    selected = std::move(result);
  }

  Family *find_family(std::size_t requests, std::size_t tokens, GraphRole role,
                      bool diagnostic = false, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    auto &table = diagnostic ? probe_families : families;
    const auto primary = table.find({requests, role, tokens, completion, sampling_index});
    if (primary != table.end() && primary->second &&
        std::ranges::find(primary->second->chunks, tokens) != primary->second->chunks.end()) return primary->second.get();
    const auto exact = table.find(FamilyKey::exact(requests, role, tokens, completion, sampling_index));
    return exact == table.end() ? nullptr : exact->second.get();
  }

  // A generated-token checkpoint can expose a suffix whose length was unknown
  // at model setup. Specialize only that actual cold request. The same original
  // Family/Phase owners remain bounded by N*R; decoded hot-loop shapes stay in
  // their original immutable finite families.
  void admit_exact_shape(std::size_t tokens, std::size_t requests, GraphRole role,
                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (find_family(requests, tokens, role, false, completion, sampling_index)) return;
    if (!state_storage_prepared || active_step || carrier_borrowed || !tokens || tokens > capacity ||
        !requests || requests > cache_plan.active_sequences ||
        cache::multiply(tokens, requests) > cache_plan.max_query_rows || verifies_prefix(role))
      throw std::runtime_error("actual suffix admission exceeds the idle model's original query/state bounds");
    const auto limit = cache::multiply(cache_plan.resident_sessions, cache_plan.active_sequences);
    if (admitted_exact_keys.capacity() < limit) admitted_exact_keys.reserve(limit);
    if (admitted_exact_keys.size() == limit) {
      const auto victim = admitted_exact_keys.front();
      auto old = families.find(victim);
      if (old == families.end() || !old->second->admitted_exact || old->second->chunks.size() != 1)
        throw std::runtime_error("exact family inventory is inconsistent");
      const ExecutionShape shape{victim.requests, old->second->chunks[0], victim.role, victim.completion, victim.sampling_index};
      if (auto phase = phases.find(shape); phase != phases.end()) {
        for (auto &run : phase->second->slots) check(run.drain());
        phases.erase(phase);
      }
      planned_slot_uses.erase(shape); families.erase(old);
      admitted_exact_keys.erase(admitted_exact_keys.begin()); ++exact_evictions;
    }
    const auto before_history = history_writers, before_recurrent = recurrent_writers;
    const auto before_hidden = hidden_writers, before_generated = generated_state_outputs;
    const auto key = FamilyKey::exact(requests, role, tokens, completion, sampling_index);
    try {
      const std::array chunk{tokens}; prepare_family(chunk, false, requests, role, true, completion, sampling_index);
      for (std::size_t i = 0; i < state_schema.size(); ++i)
        if ((!before_history[i] && history_writers[i]) || (!before_recurrent[i] && recurrent_writers[i]))
          throw std::runtime_error("new suffix requires caller-written state storage absent from the original cache plan; reconfigure the idle model");
      if (!before_hidden && hidden_writers)
        throw std::runtime_error("new suffix requires caller-written hidden storage absent from the original cache plan; reconfigure the idle model");
      planned_slot_uses[{requests, tokens, role, completion, sampling_index}].fill(1);
      admitted_exact_keys.push_back(key); ++exact_admissions;
      std::cerr << "cold_exact_admission requests=" << requests << " T=" << tokens
                << " role=" << static_cast<int>(role) << " retained=" << admitted_exact_keys.size()
                << " limit=" << limit << '\n';
    } catch (...) {
      families.erase(key); history_writers = before_history; recurrent_writers = before_recurrent;
      hidden_writers = before_hidden; generated_state_outputs = before_generated;
      throw;
    }
  }

  void prepare_draft_program(std::size_t rows, bool head) {
    preparation_started_ = true;
    if (state_storage_prepared || !cache_plan.draft_cache || !rows || rows > capacity)
      throw std::runtime_error("prepare a positive NextN width before selected cache storage");
    const auto key = std::pair{rows, head};
    if (draft_programs.contains(key)) return;
    prepare_cache();
    auto prepared = std::make_unique<DraftProgram>(); auto &program = *prepared;
    program.rows = rows; program.head = head;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, rows, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(rows));
    program.tokens = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.positions = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    for (std::size_t row = 0; row < rows; ++row) {
      program.hidden_inputs.push_back(r.input(dense(precision.activation_type(), {1, hidden})));
      program.resource_ports.push_back(program.hidden_inputs.back());
    }
    const auto shifted = r.concat(program.hidden_inputs, 0, r.dense_spec(precision.activation_type(), {count, f(hidden)}));
    std::vector<float> frequency(static_cast<std::size_t>(geometry.rotary / 2));
    for (std::size_t i = 0; i < frequency.size(); ++i) frequency[i] = 1.F / std::pow(geometry.rope_theta, float(i) / static_cast<float>(frequency.size()));
    auto &frequency_literal = assembly.literal_at(assembly.primary(), dense(Type::kFloat32, {geometry.rotary / 2}));
    frequency_literal.write<float>(frequency);
    const auto frequencies = r.constant(frequency_literal);
    const mtp::Geometry row_geometry{1, count, count};
    const auto projection = mtp::input_projection(r, program.tokens, shifted);
    const auto kv = mtp::project_kv(r, projection, program.positions, frequencies, row_geometry);
    if (head) {
      program.key = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kKey)));
      program.value = r.input(take(c::batch_cache_description(attention_schema, c::BatchCacheRole::kValue)));
      program.metadata = r.input(take(attention_schema.metadata_description()));
      const auto result = mtp::full_head(r, projection, kv, program.positions, frequencies,
          {attention_schema, program.key, program.value, program.metadata}, f(static_cast<std::int64_t>(rows - 1)), row_geometry);
      program.append = result.append_nodes; program.attention = result.attention_node;
      program.hidden = result.selected_hidden;
      if (!default_sampler.enabled) program.id = r.op(c::Operation(c::TopK{.count = 1}), {result.logits},
          {r.dense_spec(Type::kFloat32, {f(1), f(1)}), r.dense_spec(Type::kInt32, {f(1), f(1)})})[1];
      if (default_sampler.enabled || min_draft_probability() > 0) {
        const auto distribution = sampling::distribution(r, result.logits,
            default_sampler.enabled ? default_sampler.policy : sampling::Policy{});
        program.log_probabilities = distribution.log_probabilities; program.valid = r.host_predicate(distribution.valid);
        if (default_sampler.enabled) {
          program.identity = r.input(dense(Type::kUInt32, {1, 4}));
          program.id = r.one(c::sampling_ops::categorical(), {program.log_probabilities, program.identity});
          r.publish(program.log_probabilities);
        }
        if (min_draft_probability() > 0) {
          const auto maximum = r.copy(r.reshape(r.one(c::Reduce{.kind = c::ReductionKind::kMaximum,
              .accumulation_type = Type::kFloat32}, {program.log_probabilities}), {f(1)}));
          program.probability = r.unary(c::UnaryKind::kExp, maximum); r.publish(program.probability);
        }
        r.publish(program.valid);
      }
      r.publish(program.hidden); r.publish(program.id);
    } else program.append = mtp::append_kv(r, attention_schema, kv);
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, static_cast<std::int64_t>(rows - 1)}};
    const std::array variants{g::ExactShapeVariant{binding}};
    program.compiled = take(execution.compile(program.graph, assembly.constants,
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants, .resource_budget = execution_budget})));
    program.compiled.executable = take(program.compiled.executable.exact_variant(0));
    program.compiled.report = take(program.compiled.report.exact_variant(0));
    program.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), program.compiled.report));
    record_execution_plan(program.compiled.report, assembly.constants);
    program.literals = std::move(assembly.literals);
    program.token_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    program.position_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(rows)}));
    if (head) {
      program.hidden_output = take(inference::prepare_output(program.compiled, program.hidden));
      validate_state_output(program.hidden_output, dense(precision.activation_type(), {1, hidden}));
      if (default_sampler.enabled) program.probabilities_output =
          take(inference::prepare_output(program.compiled, program.log_probabilities));
    }
    for (std::size_t output = 0; output < program.compiled.executable.output_count(); ++output) {
      if (take(program.compiled.executable.output_ownership(output)) != g::OutputOwnership::kCaller) continue;
      const auto port = take(program.compiled.executable.output_id(output));
      if (port == program.hidden) {
        hidden_writers = true; program.resource_ports.push_back(port);
      } else if (port == program.log_probabilities) {
        program.resource_ports.push_back(port);
        const auto bytes = cache::multiply(vocabulary * sizeof(float), cache_plan.candidate_boundaries);
        program.probability_pool = take(c::LeasePool::create(
            take(execution.resource_for(program.probabilities_output.storage)), cache::add(bytes, 1),
            {.max_blocks = cache_plan.candidate_boundaries, .resource_budget = execution_budget}));
      }
    }
    if (head) {
      program.id_value = take(PreparedReadback::prepare(program.compiled, program.id, execution, allocation_budget));
      if (default_sampler.enabled || min_draft_probability() > 0)
        program.valid_value = take(PreparedReadback::prepare(program.compiled, program.valid, execution, allocation_budget));
      if (min_draft_probability() > 0)
        program.probability_value = take(PreparedReadback::prepare(program.compiled, program.probability, execution, allocation_budget));
    }
    initialize_draft_program(program);
    draft_programs.emplace(key, std::move(prepared));
  }

  void initialize_draft_program(DraftProgram &program) {
    program.run = take(program.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(program.run, *scheduler, *workers));
    check(program.run.bind(program.tokens, program.token_values.read()));
    check(program.run.bind(program.positions, program.position_values.read()));
    if (program.head) check(program.id_value.bind(program.run));
    if (program.head && default_sampler.enabled) {
      if (!program.identity_values.caller_owned()) program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
      check(program.run.bind(program.identity, program.identity_values.read()));
    }
    if (program.head && (default_sampler.enabled || min_draft_probability() > 0)) check(program.valid_value.bind(program.run));
    if (program.head && min_draft_probability() > 0) check(program.probability_value.bind(program.run));
  }

  // Append to an unpublished draft only. The first append uses COW; later
  // recursive rows may extend that private tail. Published Session tails are
  // never borrowed, so aborting proposal/verification leaves the old head valid.
  DraftProposal execute_draft(DraftWork &work, std::span<const std::int32_t> ids,
      std::span<const c::ReadLease> shifted, bool head, const std::array<std::uint32_t, 4> &identity = {}) {
    if (ids.empty() || ids.size() != shifted.size() || !state_storage_prepared ||
        work.state.frontier > capacity || ids.size() > capacity - work.state.frontier)
      throw std::runtime_error("NextN input rows exceed the prepared context or hidden geometry");
    auto found = draft_programs.find({ids.size(), head});
    if (found == draft_programs.end()) throw std::runtime_error("NextN finite program was not cold prepared");
    auto &program = *found->second;
    if (!program.run) initialize_draft_program(program);
    auto &run = program.run;
    StepState guard; guard.configure(draft_bank + 2); guard.run = &run;
    guard.nodes[draft_bank] = program.append.key; guard.nodes[draft_bank + 1] = program.append.value;
    cache::DraftState next;
    DraftProposal output;
    try {
      const auto length = work.state.frontier;
      next.frontier = cache::add(length, ids.size());
      const auto prefix_pages = length / cache::Plan::page_rows;
      const auto page_count = (next.frontier + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      const auto dirty_count = page_count - prefix_pages;
      const auto tail_rows = length % cache::Plan::page_rows;
      bool reuse = tail_rows && !work.state.pages.empty();
      if (reuse) for (const auto &tail : work.tails)
        reuse = reuse && tail.valid() && !tail.reserved_rows() && tail.initialized_rows() == tail_rows &&
            tail.page() == work.state.pages.back().page();
      if (reuse) {
        const auto rows = static_cast<std::uint32_t>((std::min)(ids.size(), cache::Plan::page_rows - tail_rows));
        std::size_t reserved = 0; hyle::Status status;
        for (; reserved < work.tails.size(); ++reserved) {
          status = work.tails[reserved].reserve_rows(rows); if (!status.ok()) break;
        }
        if (!status.ok()) {
          for (std::size_t i = 0; i < reserved; ++i) check(work.tails[i].rollback());
          if (!status.IsUnavailable()) check(status);
          reuse = false;
        }
      }
      next.pages.reserve(page_count);
      for (std::size_t i = 0; i < prefix_pages; ++i) next.pages.push_back(work.state.pages.at(i));
      if (reuse) {
        next.pages.push_back(work.state.pages.back());
        for (std::size_t bank = 0; bank < 2; ++bank) guard.tails[draft_bank + bank].push_back(std::move(work.tails[bank]));
      } else work.tails = {};
      for (std::size_t i = reuse ? 1 : 0; i < dirty_count; ++i) {
        auto page = take(draft_inventory.acquire()); next.pages.push_back(std::move(page.reference));
        guard.dirty[draft_bank].push_back(std::move(page.keys[0])); guard.dirty[draft_bank + 1].push_back(std::move(page.values[0]));
      }
      std::vector<std::int32_t> page_ids, positions;
      for (const auto &page : next.pages) page_ids.push_back(static_cast<std::int32_t>(page.page()));
      for (std::size_t i = 0; i < ids.size(); ++i) positions.push_back(static_cast<std::int32_t>(length + i));
      const std::array<std::int32_t, 2> query{0, static_cast<std::int32_t>(ids.size())},
          kv{0, static_cast<std::int32_t>(page_ids.size())};
      const std::array last{static_cast<std::int32_t>((next.frontier - 1) % cache::Plan::page_rows + 1)};
      const auto layout = take(attention_schema.version({query, kv, page_ids, last}));
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        auto &arena = bank == draft_bank ? draft_inventory.keys[0] : draft_inventory.values[0];
        guard.appends[bank] = take(g::batch::PagedAppendPages::create(arena, layout,
            bank == draft_bank ? work.state.key : work.state.value, guard.dirty[bank], guard.tails[bank],
            {.resource_budget = cache_budget}));
      }
      program.token_values.write<std::int32_t>(ids); program.position_values.write<std::int32_t>(positions);
      for (std::size_t row = 0; row < ids.size(); ++row)
        check(run.bind(program.hidden_inputs[row], dense(precision.activation_type(), {1, hidden}), shifted[row]));
      if (program.hidden_output.value && program.hidden_output.caller_owned())
        check(run.bind(program.hidden_output.value, program.hidden_output.description,
            take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment))));
      if (head && default_sampler.enabled) {
        program.identity_values.write<std::uint32_t>(identity);
        if (program.probabilities_output.caller_owned()) check(run.bind(program.probabilities_output.value,
            program.probabilities_output.description, take(program.probability_pool.allocate(vocabulary * sizeof(float)))));
      }
      const std::array layouts{g::batch::NodeLayout{program.attention, layout}};
      const auto queues = inference::queue_span(program.queues);
      check(g::batch::prepare(run, queues, head ? std::span<const g::batch::NodeLayout>(layouts) : std::span<const g::batch::NodeLayout>{},
          {.host_workspace = execution.staging()}));
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        if (head) check(g::batch::bind_append(run, program.graph, guard.nodes[bank],
            bank == draft_bank ? program.key : program.value, std::move(guard.appends[bank])));
        else check(g::batch::bind_append(run, program.graph, guard.nodes[bank], std::move(guard.appends[bank])));
        guard.bound[bank] = true;
      }
      check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
      guard.completion = submit_retained(run, queues, program.retirement); check(guard.completion.wait()); guard.completion = {};
      ++draft_program_graph_calls;
      if (head) {
        if (default_sampler.enabled || min_draft_probability() > 0) {
          const auto flag = take(program.valid_value.predicate(execution, run));
          if (flag != 1) throw std::runtime_error("recursive draft distribution is invalid; no head or RNG was published");
          if (default_sampler.enabled) output.log_probabilities =
              take(inference::retain_output(run, program.probabilities_output, execution_budget));
          if (min_draft_probability() > 0) {
            check(program.probability_value.read(execution, run, std::as_writable_bytes(std::span(&output.probability, 1))));
            if (!std::isfinite(output.probability) || output.probability < 0 || output.probability > 1)
              throw std::runtime_error("draft maximum probability is outside its declared probability domain");
          }
        }
        check(program.id_value.read(execution, run, std::as_writable_bytes(std::span(&output.id, 1))));
        if (output.id < 0 || output.id >= vocabulary) throw std::runtime_error("NextN proposal is outside vocabulary");
        output.hidden = retain_state_storage(run, program.hidden_output);
      }
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        guard.appends[bank] = take(g::batch::take_append(run, guard.nodes[bank])); guard.bound[bank] = false;
      }
      next.key = take(guard.appends[draft_bank].ready()); next.value = take(guard.appends[draft_bank + 1].ready());
      std::array<c::PagedPageAppender, 2> tails;
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) for (std::size_t i = 0; i < guard.appends[bank].appender_count(); ++i) {
        auto &tail = take(guard.appends[bank].appender(i)); if (!tail.valid()) continue;
        if (tail.page() != next.pages.back().page() || tail.reserved_rows() || tails[bank - draft_bank].valid() ||
            tail.initialized_rows() != next.frontier % cache::Plan::page_rows)
          throw std::runtime_error("NextN append returned an invalid private tail");
        tails[bank - draft_bank] = std::move(tail);
      }
      check(run.unbind(program.resource_ports));
      work.tails = std::move(tails); work.state = std::move(next);
      return output;
    } catch (...) {
      // An unsuccessful private append is never a partially published head.
      run.close().IgnoreError(); execution.drain_retired(run).IgnoreError();
      auto recovered = guard.try_abort();
      if (!recovered || !*recovered) run = {};
      throw;
    }
  }

  void prepare_prefix_sampling(std::size_t proposals) {
    preparation_started_ = true;
    if (!default_sampler.enabled || state_storage_prepared || proposals >= cache_plan.candidate_boundaries)
      throw std::runtime_error("prepare bounded speculative-sampling widths before cache storage");
    if (prefix_sampling_programs.contains(proposals)) return;
    auto prepared = std::make_unique<PrefixSamplingProgram>(); auto &program = *prepared;
    program.proposals = proposals;
    GraphAssembly assembly(execution, precision, allocation_budget);
    Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
    const auto count = f(static_cast<std::int64_t>(proposals + 1));
    program.logits = r.input(r.dense_spec(Type::kFloat32, {count, f(vocabulary)}));
    program.identities = r.input(r.dense_spec(Type::kUInt32, {count, f(4)}));
    if (proposals) program.ids = r.input(dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    auto assembled = sampling::speculative_prefix(r, program.logits, program.identities,
        program.ids, default_sampler.policy);
    program.q = std::move(assembled.q); program.accepted = std::move(assembled.accepted);
    program.final_ids = std::move(assembled.final_ids); program.valid = std::move(assembled.valid);
    for (std::size_t row = 0; row < program.final_ids.size(); ++row) {
      if (proposals) {
        program.accepted[row] = r.host_predicate(program.accepted[row]);
        r.publish(program.accepted[row]);
      }
      program.valid[row] = r.host_predicate(program.valid[row]);
      r.publish(program.final_ids[row]); r.publish(program.valid[row]);
    }
    program.graph = take(std::move(r.builder).finish());
    const std::array binding{g::ShapeBinding{r.last_row, 0}};
    const std::array variants{g::ExactShapeVariant{binding}};
    program.compiled = take(execution.compile(program.graph, assembly.constants,
        assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
         .constants = assembly.constants, .resource_budget = execution_budget})));
    program.compiled.executable = take(program.compiled.executable.exact_variant(0));
    program.compiled.report = take(program.compiled.report.exact_variant(0));
    program.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), program.compiled.report));
    record_execution_plan(program.compiled.report, assembly.constants);
    program.literals = std::move(assembly.literals);
    program.identity_values = caller_tensor(execution.device(), dense(Type::kUInt32, {static_cast<std::int64_t>(proposals + 1), 4}));
    input_scratch.sampling_identities.reserve(take(program.identity_values.description.required_bytes()) / sizeof(std::uint32_t));
    if (proposals) program.id_values = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(proposals)}));
    const auto prepare = [&](const std::vector<V> &ports, std::vector<PreparedReadback> &values) {
      values.reserve(ports.size());
      for (const auto port : ports) values.push_back(take(PreparedReadback::prepare(program.compiled, port, execution, allocation_budget)));
    };
    if (proposals) prepare(program.accepted, program.accepted_values);
    prepare(program.final_ids, program.final_values); prepare(program.valid, program.valid_values);
    program.resource_ports = program.q; program.resource_ports.push_back(program.logits);
    initialize_prefix_sampling(program);
    prefix_sampling_programs.emplace(proposals, std::move(prepared));
  }

  void initialize_prefix_sampling(PrefixSamplingProgram &program) {
    program.run = take(program.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(program.run, *scheduler, *workers));
    check(program.run.bind(program.identities, program.identity_values.read()));
    if (program.proposals) check(program.run.bind(program.ids, program.id_values.read()));
    for (auto &output : program.accepted_values) check(output.bind(program.run));
    for (auto &output : program.final_values) check(output.bind(program.run));
    for (auto &output : program.valid_values) check(output.bind(program.run));
  }

  // Target family widths and replay counts stay with the finite request plan;
  // every recursive consumer shares this model's actual NextN/sampler programs.
  void prepare_recursive_round_programs() {
    for (std::size_t rows = 1; rows <= cache_plan.candidate_boundaries; ++rows) {
      prepare_draft_program(rows, false);
      if (default_sampler.enabled && (rows > 1 || min_draft_probability() > 0)) prepare_prefix_sampling(rows - 1);
    }
    prepare_draft_program(1, true);
    if (capacity > 1) prepare_draft_program(2, true);
  }

  PrefixSamplingResult sample_recursive_prefix(c::ConstTensorView logits,
      std::span<const std::int32_t> proposals, std::span<const c::ReadLease> q,
      cache::SamplingCursor cursor) {
    if (!default_sampler.enabled || proposals.size() != q.size())
      throw std::runtime_error("speculative sampling requires every actual proposal distribution");
    auto found = prefix_sampling_programs.find(proposals.size());
    if (found == prefix_sampling_programs.end()) throw std::runtime_error("speculative-sampling width was not cold prepared");
    auto &program = *found->second;
    if (!program.run) initialize_prefix_sampling(program);
    auto &run = program.run;
    const auto count = proposals.size();
    (void)cursor.advanced(cache::add(cache::multiply(2, count), 1));
    auto &identities = input_scratch.sampling_identities;
    identities.clear();
    for (std::size_t row = 0; row <= count; ++row) {
      const auto identity = cursor.identity(row == count ? 2 * count : 2 * row + 1);
      identities.insert(identities.end(), identity.begin(), identity.end());
    }
    program.identity_values.write<std::uint32_t>(identities);
    if (count) program.id_values.write<std::int32_t>(proposals);
    try {
    check(run.bind(program.logits, logits));
    for (std::size_t row = 0; row < count; ++row)
      check(run.bind(program.q[row], dense(Type::kFloat32, {1, vocabulary}), q[row]));
    const auto queues = inference::queue_span(program.queues);
    check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
    auto done = submit_retained(run, queues, program.retirement); check(done.wait());
    ++prefix_decision_graph_calls;
    PrefixSamplingResult result; result.accepted.resize(count); result.final_ids.resize(program.final_ids.size());
    for (std::size_t row = 0; row < program.final_ids.size(); ++row) {
      const auto valid = take(program.valid_values[row].predicate(execution, run));
      if (valid != 1) throw std::runtime_error("recursive target/q/residual distribution is invalid; no head or RNG was published");
      if (count) result.accepted[row] = take(program.accepted_values[row].predicate(execution, run));
      check(program.final_values[row].read(execution, run, std::as_writable_bytes(std::span(&result.final_ids[row], 1))));
    }
    check(run.unbind(program.resource_ports));
    return result;
    } catch (...) {
      // Failed p/q validation or native work must release the same transient
      // input pins as success. A later explicit retry recreates only this
      // original finite invocation, never a Session or an RNG cursor.
      run.close().IgnoreError(); execution.drain_retired(run).IgnoreError(); run = {};
      throw;
    }
  }

  std::shared_ptr<cache::Snapshot> reserve_snapshot() {
    auto version = take(cache::VersionRef::acquire(snapshot_versions));
    auto result = std::make_shared<cache::Snapshot>();
    result->resource_owner = resources_;
    if (!result->keys.try_resize(target_kv_layers) || !result->values.try_resize(target_kv_layers) ||
        !result->state.try_resize(geometry.layers.size())) throw std::bad_alloc{};
    result->version = std::move(version); result->model = this; result->math = math_identity();
    return result;
  }

  // A request's pending token/cursor is not part of the reusable model state.
  // Wrapper origins are canonical so aliases share exactly one finite version;
  // mutable page-tail capabilities stay with the original Session only.
  std::shared_ptr<cache::Snapshot> sampling_fork(cache::SnapshotRef base,
      cache::SamplingCursor cursor, std::int32_t pending) const {
    if (!base || base->model != this || base->math != math_identity())
      throw std::runtime_error("sampling fork belongs to another model or compiled policy");
    auto next = std::make_shared<cache::Snapshot>();
    next->state_origin = base->state_origin ? base->state_origin : base;
    next->pages = base->pages; next->keys = base->keys; next->values = base->values;
    next->state = base->state; next->carry_hidden = base->carry_hidden;
    next->prior_hidden = base->prior_hidden; next->draft = base->draft;
    next->model = base->model; next->math = base->math; next->length = base->length;
    next->next_id = pending; next->sampling = cursor; return next;
  }

  void prepare_sampling_frontier() {
    preparation_started_ = true;
    for (auto &entry : resources_->sampling_configurations) {
      const auto &selected_sampler = entry->configuration;
      if (!selected_sampler.enabled && sampling_configuration_count() == 1) continue;
      if (!entry->frontier) {
        auto prepared = std::make_unique<SamplingFrontier>();
        GraphAssembly assembly(execution, precision, allocation_budget);
        Recipe r(weights, projections, assembly, geometry, 1, 1, &weight_meanings);
        prepared->last_row = r.last_row;
        prepared->hidden_port = r.input(dense(precision.activation_type(), {1, hidden}));
        if (selected_sampler.enabled) prepared->identity_port = r.input(dense(Type::kUInt32, {1, 4}));
        if (selected_sampler.enabled) {
          const auto probabilities = sampling::distribution(r,
              r.scores(prepared->hidden_port, std::string(geometry.head_role())), selected_sampler.policy);
          prepared->id_port = r.one(c::sampling_ops::categorical(), {probabilities.log_probabilities, prepared->identity_port});
          prepared->valid_port = r.host_predicate(probabilities.valid);
        } else prepared->id_port = r.op(c::Operation(c::TopK{.count = 1}),
            {r.scores(prepared->hidden_port, std::string(geometry.head_role()))})[1];
        r.publish(prepared->id_port); if (selected_sampler.enabled) r.publish(prepared->valid_port);
        prepared->graph = take(std::move(r.builder).finish());
        // The frontier is a fixed [1,H] projection with no symbolic model rows.
        const std::array shape{g::ShapeBinding{r.last_row, 0}};
        const std::array variants{g::ExactShapeVariant{shape}};
        prepared->compiled = take(execution.compile(prepared->graph, assembly.constants,
            assembly.options({.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = variants,
             .constants = assembly.constants, .resource_budget = execution_budget})));
        prepared->compiled.executable = take(prepared->compiled.executable.exact_variant(0));
        prepared->compiled.report = take(prepared->compiled.report.exact_variant(0));
        prepared->queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), prepared->compiled.report));
        record_execution_plan(prepared->compiled.report, assembly.constants);
        prepared->literals = std::move(assembly.literals);
        if (selected_sampler.enabled) prepared->identity = caller_tensor(execution.device(), dense(Type::kUInt32, {1, 4}));
        prepared->id = take(PreparedReadback::prepare(prepared->compiled, prepared->id_port, execution, allocation_budget));
        if (selected_sampler.enabled) prepared->valid = take(PreparedReadback::prepare(prepared->compiled, prepared->valid_port, execution, allocation_budget));
        entry->frontier = std::move(prepared);
      }
    }
  }

  cache::SnapshotRef sample_prefix(cache::SnapshotRef base, cache::SamplingCursor cursor, std::size_t sampling_index = 0) {
    const auto &selected = sampling_configuration(sampling_index);
    const auto &selected_sampler = selected.configuration;
    if (!base || !base->length || !base->carry_hidden.valid() || active_step || carrier_borrowed || !selected.frontier)
      throw std::runtime_error("prefix sampling requires its cold-prepared graph and an idle complete model-state hidden frontier");
    const auto next_cursor = selected_sampler.enabled ? cursor.advanced(1) : cache::SamplingCursor{};
    const auto started = Clock::now();
    auto &frontier = *selected.frontier;
    if (frontier.retirement.pending())
      throw std::runtime_error("original sampling frontier resource scope is still retained");
    // This is the same Graph execution API/runtime as target/draft. It reads
    // the original immutable hidden lease and returns only two scalar results.
    auto run = take(frontier.compiled.executable.create_invocation(
        {.resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (scheduler) check(g::prepare_progress(run, *scheduler, *workers));
    if (selected_sampler.enabled) { const auto identity = cursor.identity(); frontier.identity.write<std::uint32_t>(identity); }
    check(run.bind(frontier.hidden_port, dense(precision.activation_type(), {1, hidden}), base->carry_hidden));
    if (selected_sampler.enabled) check(run.bind(frontier.identity_port, frontier.identity.read()));
    check(frontier.id.bind(run)); if (selected_sampler.enabled) check(frontier.valid.bind(run));
    const auto queues = inference::queue_span(frontier.queues);
    check(run.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
    auto done = submit_retained(run, queues, frontier.retirement); check(done.wait());
    ++frontier_sampling_graph_calls;
    const auto valid = selected_sampler.enabled ? take(frontier.valid.predicate(execution, run)) : std::uint8_t{1};
    std::int32_t pending = -1;
    check(frontier.id.read(execution, run, std::as_writable_bytes(std::span(&pending, 1))));
    if (valid != 1 || pending < 0 || pending >= vocabulary)
      throw std::runtime_error("cached frontier sampling is invalid; no request RNG cursor was published");
    auto next = sampling_fork(std::move(base), next_cursor, pending);
    frontier_sampling_seconds += std::chrono::duration<double>(Clock::now() - started).count();
    return next;
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
      if (!families.contains({requests, role, tokens, completion, sampling_index}) || std::find(families.at({requests, role, tokens, completion, sampling_index})->chunks.begin(), families.at({requests, role, tokens, completion, sampling_index})->chunks.end(), tokens) == families.at({requests, role, tokens, completion, sampling_index})->chunks.end())
        throw std::runtime_error("execution plan chunk is outside the shared finite family");
      auto &uses = planned_slot_uses[{requests, tokens, role, completion, sampling_index}];
      uses[slot_index] = cache::add(uses[slot_index], 1);
    }
  }

  // Finite wave/suffix plans can count many possible restores without
  // materializing repeated host call tapes or execution carriers.
  void plan_replay_uses(std::size_t tokens, std::size_t requests, std::size_t count,
                        GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (!phases.empty() || !probe_phases.empty() || carrier_borrowed)
      throw std::runtime_error("execution use plans must precede carrier preparation");
    if (!families.contains({requests, role, tokens, completion, sampling_index}) ||
        std::find(families.at({requests, role, tokens, completion, sampling_index})->chunks.begin(), families.at({requests, role, tokens, completion, sampling_index})->chunks.end(), tokens) == families.at({requests, role, tokens, completion, sampling_index})->chunks.end())
      throw std::runtime_error("execution use shape is outside the shared finite family");
    for (auto &uses : planned_slot_uses[{requests, tokens, role, completion, sampling_index}]) uses = cache::add(uses, count);
  }

  std::shared_ptr<Session> create_session(std::span<const std::size_t> planned_chunks,
                                        cache::SnapshotRef prefix = {},
                                        GraphRole role = GraphRole::kText,
                                        std::optional<cache::SamplingCursor> cursor = {},
                                        bool resample_frontier = false,
                                         TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    const auto &prepared_sampling = sampling_configuration(sampling_index);
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && sampling_index) throw std::runtime_error("nondefault sampling is ordinary Text only");
    prepare_state_storage();
    const auto initial_rows = planned_chunks.empty() ? 0 : planned_chunks.front();
    for (const auto rows : planned_chunks)
      if (!find_family(1, rows, role, false, completion, sampling_index)) admit_exact_shape(rows, 1, role, completion, sampling_index);
    if ((has_mtp(role) && !cache_plan.draft_cache) ||
        (cache_plan.draft_cache && role == GraphRole::kText))
      throw std::runtime_error("session must use its complete target/draft cache family");
    if ((planned_chunks.empty() ? !families.contains({1, role, 0, completion, sampling_index}) : !find_family(1, initial_rows, role, false, completion, sampling_index)) ||
        (!layer0_probe.empty() && !probe_families.contains({1, role, initial_rows, completion, sampling_index})))
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
      for (const auto *selected : {find_family(1, tokens, role, false, completion, sampling_index), layer0_probe.empty() ? nullptr : probe_families.at({1, role, tokens, completion, sampling_index}).get()}) {
        if (selected && std::find(selected->chunks.begin(), selected->chunks.end(), tokens) == selected->chunks.end())
          throw std::runtime_error("session chunk is outside the shared finite family");
      }
      total += tokens;
      if (!planned_slot_uses.contains({1, tokens, role, completion, sampling_index}))
        throw std::runtime_error("session chunk has no model execution use plan");
    }
    if (prefix) {
      if (resample_frontier && (selected_sampler.enabled || sampling_configuration_count() > 1))
        prefix = sample_prefix(std::move(prefix), cursor.value_or(cache::SamplingCursor{}), sampling_index);
      else if (selected_sampler.enabled && cursor) prefix = sampling_fork(std::move(prefix), *cursor, -1);
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
      session->head = std::move(prefix); // Exact evaluated-state fork; no KV/state payload copy.
    }
    else {
      auto initial = reserve_snapshot();
      initial->sampling = selected_sampler.enabled ? cursor.value_or(cache::SamplingCursor{sampling_seed, 0, 0})
          : cache::SamplingCursor{};
      const std::array<std::int32_t, 2> empty{0, 0};
      const std::array<std::int32_t, 1> tail{0};
      const auto layout = take(attention_schema.version({empty, empty, {}, tail}));
      for (std::size_t i = 0; i < initial->keys.size(); ++i) {
        initial->keys[i] = take(page_inventory.keys[i].read(layout, {}));
        initial->values[i] = take(page_inventory.values[i].read(layout, {}));
      }
      if (cache_plan.draft_cache) {
        initial->draft.emplace();
        initial->draft->key = take(draft_inventory.keys[0].read(layout, {}));
        initial->draft->value = take(draft_inventory.values[0].read(layout, {}));
        auto zero = take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment));
        check(zero.view()->fill(std::byte{})); initial->carry_hidden = take(zero.publish());
      }
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i)) {
        auto history = take(state_pools[i].history.allocate(take(state_schema[i].history.required_bytes()), cache::state_pool_alignment));
        auto recurrent = take(state_pools[i].recurrent.allocate(take(state_schema[i].recurrent.required_bytes()), cache::state_pool_alignment));
        check(history.view()->fill(std::byte{})); check(recurrent.view()->fill(std::byte{}));
        initial->state[i].history = take(history.publish());
        initial->state[i].recurrent = take(recurrent.publish());
      }
      session->head = std::move(initial);
    }
    // State admission does not allocate execution carriers. The first actual
    // explicit prepare creates its needed Phase and slot; idle/full-hit cached
    // Sessions never duplicate IO, workspace or native execution preparation.
    return session;
  }

  void restore_pending_boundary(Session &session, cache::SamplingCursor cursor, std::int32_t pending) const {
    if (session.pending || active_step || carrier_borrowed || pending < -1 || pending >= vocabulary)
      throw std::runtime_error("sampling recovery requires an idle committed boundary and historical pending token");
    auto replacement = sampling_fork(session.head, cursor, pending);
    for (auto &tail : session.tails) tail = {};
    session.draft_tails = {};
    session.tail_head = nullptr; session.draft_tail_head = nullptr;
    session.head = std::move(replacement);
  }

  void validate_mtp_snapshot(const cache::Snapshot &head) const {
    if (cache_plan.draft_cache != head.draft.has_value())
      throw std::runtime_error("snapshot does not contain the configured complete MTP state");
    if (!head.draft) return;
    const auto &draft = *head.draft;
    if (draft.lag > 1 || head.length < draft.lag || draft.frontier != head.length - draft.lag ||
        !draft.key.valid() || !draft.value.valid() || !head.carry_hidden.valid() ||
        (draft.lag && (!head.prior_hidden.valid() || draft.catchup_token < 0 || draft.catchup_token >= vocabulary)) ||
        (!draft.lag && draft.catchup_token != -1))
      throw std::runtime_error("inconsistent MTP frontier, catch-up token or owned target hidden state");
  }

  void initialize_slot(Phase &phase, std::size_t slot) {
    if (phase.slots[slot].valid()) return;
    const auto &definition = *phase.family;
    const auto &selected_sampler = definition.sampling->configuration;
    const std::array shape{g::ShapeBinding{definition.last_row, static_cast<std::int64_t>(phase.query_rows - 1)}};
    auto run = take(definition.compiled.executable.create_invocation(
        {.shape = shape, .resources = g::ResourceOptions{.resource_budget = execution_budget ? execution_budget : cache_budget}}));
    if (run.exact_variant_index() != phase.variant_index)
      throw std::runtime_error("finite family selected an unexpected exact child");
    if (scheduler) check(g::prepare_progress(run, *scheduler, *workers));
    check(run.bind(phase.token_port, phase.tokens.read()));
    check(run.bind(phase.position_port, phase.positions.read()));
    if (selected_sampler.enabled && definition.completion == TextCompletion::kEmitNext && definition.role != GraphRole::kVerify) {
      check(run.bind(phase.sampling_identity_port, phase.sampling_identities.read()));
      check(phase.sampling_valid.bind(run));
      if (drafts_token(definition.role)) check(phase.sampling_accept.bind(run));
    }
    if (definition.requests != 1 && !verifies_prefix(definition.role)) check(run.bind(phase.last_ids_port, phase.last_ids.read()));
    if (phase.logits.caller_owned()) check(run.bind(phase.logits_port, phase.logits.view()));
    if (definition.completion == TextCompletion::kEmitNext) check(phase.next_id.bind(run));
    if (phase.final_hidden.caller_owned()) check(run.bind(phase.hidden_port, phase.final_hidden.view()));
    if (has_mtp(definition.role)) {
      check(run.bind(phase.draft_position_port, phase.draft_positions.read()));
      if (draft_lag(definition.role)) check(run.bind(phase.catchup_token_port, phase.catchup_token.read()));
      if (drafts_token(definition.role)) check(phase.draft_id.bind(run));
    }
    if (!layer0_probe.empty()) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i)
      if (const auto *output = phase.probe_facts[i]; output && output->caller_owned())
        check(run.bind(output->value, phase.probe_outputs[i].view()));
    // Cold preparation can also run without a pending StepState. Publish only
    // the fully bound slot, so an allocation/binding failure cannot leave an
    // apparently valid half-prepared Invocation in the reusable Model pool.
    phase.slots[slot] = std::move(run);
    phase.preparation_uses[slot] = 0;
  }

  Phase &prepare_child(std::size_t tokens, bool diagnostic, std::size_t requests = 1, GraphRole role = GraphRole::kText,
                       TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    auto &cache = diagnostic ? probe_phases : phases;
    const auto found = cache.find({requests, tokens, role, completion, sampling_index});
    if (found != cache.end()) return *found->second;
    auto *selected = find_family(requests, tokens, role, diagnostic, completion, sampling_index);
    if (!selected) throw std::runtime_error("text finite family has not been prepared");
    auto &definition = *selected;
    const auto &selected_sampler = definition.sampling->configuration;
    const auto point = std::find(definition.chunks.begin(), definition.chunks.end(), tokens);
    if (point == definition.chunks.end()) throw std::runtime_error("unlisted text chunk has no prepared exact child");
    const auto start = Clock::now();
    const auto index = static_cast<std::size_t>(point - definition.chunks.begin());
    auto result = std::make_unique<Phase>();
    auto &phase = *result;
    phase.query_rows = tokens;
    static_cast<FamilyPorts &>(phase) = static_cast<const FamilyPorts &>(definition);
    phase.planned_slot_uses = planned_slot_uses.at({requests, tokens, role, completion, sampling_index});
    for (std::size_t slot = 0; slot < phase.replay_modes.size(); ++slot) {
      phase.replay_modes[slot] = selected_sampler.enabled || replay_policy == ReplayPolicy::kCommands
          || (replay_policy == ReplayPolicy::kAuto && phase.planned_slot_uses[slot] <= 1)
          ? g::ReplayMode::kCommands : g::ReplayMode::kPreferCapture;
    }
    phase.attention_schema = definition.attention_schema;
    phase.child = definition.children.at(index).get();
    phase.queues = take(inference::prepare_queue_bindings(execution.queue_bindings(), phase.child->compiled.report));
    phase.family = &definition; phase.variant_index = index;
    std::erase_if(phase.state_resource_ports, [&](V value) {
      return value.kind() != g::ValueKind::kInput && !phase.child->output(value).caller_owned();
    });
    const auto t = static_cast<std::int64_t>(tokens);
    phase.tokens = caller_tensor(execution.device(), dense(Type::kInt32,
        {has_mtp(role) && !is_mtp_prefill(role) ? static_cast<std::int64_t>(requests) : t * static_cast<std::int64_t>(requests)}));
    phase.positions = caller_tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {t})
        : dense(Type::kInt32, {static_cast<std::int64_t>(requests), t}));
    if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) {
      phase.sampling_identities = caller_tensor(execution.device(), dense(Type::kUInt32,
          {static_cast<std::int64_t>(requests), 3, 4}));
      phase.sampling_valid = take(PreparedReadback::prepare(phase.child->compiled, phase.sampling_valid_port, execution, allocation_budget));
      phase.sampling_valid_values.resize(requests);
      bool float_predicates = phase.sampling_valid.facts.description.data_type() == Type::kFloat32;
      if (drafts_token(role)) {
        phase.sampling_accept = take(PreparedReadback::prepare(phase.child->compiled, phase.sampling_accept_port, execution, allocation_budget));
        phase.sampling_accept_values.resize(requests);
        float_predicates = float_predicates || phase.sampling_accept.facts.description.data_type() == Type::kFloat32;
      }
      if (float_predicates) phase.sampling_scratch.resize(requests);
    }
    if (has_mtp(role)) {
      const auto draft_rows = is_mtp_prefill(role) ? t + static_cast<std::int64_t>(draft_lag(role)) : static_cast<std::int64_t>(draft_lag(role) + 1);
      phase.draft_positions = caller_tensor(execution.device(), requests == 1 ? dense(Type::kInt32, {draft_rows})
          : dense(Type::kInt32, {static_cast<std::int64_t>(requests), draft_rows}));
      if (draft_lag(role)) phase.catchup_token = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
      if (drafts_token(role)) {
        phase.draft_id = take(PreparedReadback::prepare(phase.child->compiled, phase.draft_id_port, execution, allocation_budget));
        phase.draft_ids.resize(requests);
      }
    }
    if (requests != 1 && !verifies_prefix(role)) {
      phase.last_ids = caller_tensor(execution.device(), dense(Type::kInt32, {static_cast<std::int64_t>(requests)}));
      std::vector<std::int32_t> last_ids(requests);
      for (std::size_t j = 0; j < requests; ++j) last_ids[j] = static_cast<std::int32_t>((j + 1) * tokens - 1);
      phase.last_ids.write<std::int32_t>(last_ids);
    }
    if (completion == TextCompletion::kEmitNext) {
      phase.logits_facts = &phase.child->output(phase.logits_port);
      phase.logits = caller_tensor(take(execution.resource_for(phase.logits_facts->storage)),
          phase.logits_facts->description, phase.logits_facts->caller_owned());
      phase.next_id = take(PreparedReadback::prepare(phase.child->compiled, phase.id_port, execution, allocation_budget));
    }
    if (verifies_prefix(role)) phase.prefix_candidates.resize(cache::multiply(requests, phase.prefix_rows));
    phase.hidden_facts = &phase.child->output(phase.hidden_port);
    phase.final_hidden = caller_tensor(take(execution.resource_for(phase.hidden_facts->storage)),
        phase.hidden_facts->description, phase.hidden_facts->caller_owned());
    if (diagnostic) for (std::size_t i = 0; i < phase.probe_ports.size(); ++i) {
      phase.probe_facts[i] = &phase.child->output(phase.probe_ports[i]);
      const auto &output = *phase.probe_facts[i];
      phase.probe_outputs[i] = caller_tensor(take(execution.resource_for(output.storage)), output.description, output.caller_owned());
    }
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
              << " slot0_uses=" << phase.planned_slot_uses[0]
              << " slot0=" << (phase.replay_modes[0] == g::ReplayMode::kCommands ? "commands" : "prefer-capture") << '\n';
    std::cerr << "compiled T=" << tokens << " steps=" << phase.child->compiled.report.step_count()
              << " workspace/slot=" << phase.child->compiled.executable.workspace_bytes()
              << " compiled_known_bytes=" << storage.known_bytes()
              << " compiled_unknown_providers=" << storage.unknown_provider_count
              << " child_prepare_seconds=" << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
    input_scratch.prepare(phase);
    auto *address = result.get(); cache.emplace(ExecutionShape{requests, tokens, role, completion, sampling_index}, std::move(result)); return *address;
  }

  // An engine with a known ordered request can explicitly prepare its actual
  // carriers before admission/ready, preserving the original cold boundary.
  // Candidate prefix suffixes remain unmaterialized until a hit selects one.
  void prepare_execution(std::span<const std::size_t> calls,
                         std::size_t requests = 1, GraphRole role = GraphRole::kText,
                          TextCompletion completion = TextCompletion::kEmitNext, std::size_t sampling_index = 0) {
    if (carrier_borrowed)
      throw std::runtime_error("cannot cold-prepare carriers during an active execution loan");
    for (const auto tokens : calls) {
      if (!layer0_probe.empty()) prepare_child(tokens, false, requests, role, completion, sampling_index);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), requests, role, completion, sampling_index);
      initialize_slot(phase, slot_index);
    }
  }

  void detach_episode() noexcept {
    if (!active_step) return;
    // Only a caller-requested, actually retired shared cancellation is local
    // to this episode. An unchanged surviving Session remains runnable; do not
    // poison its later retirement with a peer's normal Graph cancellation.
    // Preserve native and all other failures, and the cancelled owner's fact.
    const bool cancelled_episode = active_step->cancelled &&
        active_step->terminal.IsCancelled() && !active_step->terminal.has_native_error();
    for (auto &member : active_step->members) {
      if (!cancelled_episode || member.session->stopping)
        inference::record_retirement_status(member.session->retirement_status, active_step->terminal);
      member.session->pending = nullptr;
    }
    active_step.reset();
  }

  inference::RetirementObservation try_retire_episode() noexcept {
    if (!active_step) return {true, {}};
    // Dropping the public Completion does not remove original native work.
    // Drive its owning environment before checking the unchanged Graph/page
    // retirement gates; an observation error alone never proves retirement.
    inference::record_retirement_status(active_step->terminal, execution.collect_retired());
    const auto retired = active_step->try_abort();
    if (!retired) {
      inference::record_retirement_status(active_step->terminal, retired.error());
      return {false, active_step->terminal};
    }
    const auto terminal = active_step->terminal;
    if (*retired) detach_episode();
    return {*retired, terminal};
  }

  inference::RetirementObservation abort_episode() noexcept {
    if (!active_step) return {true, {}};
    // Nonblocking collection cannot drain a partially accepted Vulkan prefix.
    // Hand off once to the original blocking owner, then check the unchanged
    // Graph/page gates. Unknown native failure keeps this candidate, Session
    // pins and admission intact for an explicit retry; it must not spin.
    if (active_step->run)
      inference::record_retirement_status(active_step->terminal, execution.drain_retired(*active_step->run));
    // Episode retirement alone does not release a Session's published head.
    return try_retire_episode();
  }

  // Selection and equal-T bucketing belong to the caller. A batch owns one
  // original Invocation, IO loan, Completion and publication transaction.
  void prepare_draft_append(StepState &step, GraphRole role) {
    auto &phase = *step.phase; auto &run = *step.run;
    const auto width = step.members.size();
    const auto rows = is_mtp_prefill(role) ? cache::add(step.input_tokens, draft_lag(role)) : draft_lag(role) + 1;
    hyle::container::SmallVector<std::int32_t, 2> query{0}, old_ptr{0}, next_ptr{0}, old_last, next_last;
    std::vector<std::int32_t> old_ids, next_ids;
    auto &positions = input_scratch.draft_positions, &catchups = input_scratch.catchups;
    positions.clear(); catchups.clear();
    for (auto *table : {&query, &old_ptr, &next_ptr})
      if (!table->try_reserve(width + 1)) throw std::bad_alloc{};
    for (auto *table : {&old_last, &next_last})
      if (!table->try_reserve(width)) throw std::bad_alloc{};
    std::size_t old_count = 0, next_count = 0;
    for (const auto &member : step.members) {
      old_count = cache::add(old_count, member.base->draft->pages.size());
      next_count = cache::add(next_count, (cache::add(member.base->draft->frontier, rows) + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
    }
    old_ids.reserve(old_count); next_ids.reserve(next_count);
    for (std::size_t j = 0; j < width; ++j) {
      auto &member = step.members[j]; auto &session = *member.session;
      const auto &old = *member.base->draft;
      member.next->draft.emplace(); auto &next = *member.next->draft;
      next.frontier = cache::add(old.frontier, rows);
      if (next.frontier > capacity) throw std::runtime_error("draft cache exceeds context capacity");
      const auto prefix_pages = old.frontier / cache::Plan::page_rows;
      const auto count = (next.frontier + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
      const auto dirty_count = count - prefix_pages, old_tail = old.frontier % cache::Plan::page_rows;
      bool reuse = old_tail && session.draft_tail_head == member.base.get() && !old.pages.empty();
      if (reuse) for (const auto &tail : session.draft_tails)
        reuse = reuse && tail.valid() && !tail.reserved_rows() &&
            tail.page() == old.pages.back().page() && tail.initialized_rows() == old_tail;
      next.pages.reserve(count);
      for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) step.dirty[bank].reserve(cache::add(step.dirty[bank].size(), dirty_count));
      if (reuse) {
        const auto added = static_cast<std::uint32_t>((std::min)(rows, cache::Plan::page_rows - old_tail));
        std::size_t reserved = 0; hyle::Status status;
        for (; reserved < session.draft_tails.size(); ++reserved) {
          status = session.draft_tails[reserved].reserve_rows(added); if (!status.ok()) break;
        }
        if (!status.ok()) {
          for (std::size_t i = 0; i < reserved; ++i) check(session.draft_tails[i].rollback());
          if (!status.IsUnavailable()) check(status);
          reuse = false;
        }
      }
      for (std::size_t page = 0; page < prefix_pages; ++page) next.pages.push_back(old.pages.at(page));
      if (reuse) {
        next.pages.push_back(old.pages.back());
        for (std::size_t bank = 0; bank < 2; ++bank) step.tails[draft_bank + bank].push_back(std::move(session.draft_tails[bank]));
      } else session.draft_tails = {};
      session.draft_tail_head = nullptr;
      for (std::size_t page = reuse ? 1 : 0; page < dirty_count; ++page) {
        auto reservation = take(draft_inventory.acquire()); next.pages.push_back(std::move(reservation.reference));
        step.dirty[draft_bank].push_back(std::move(reservation.keys[0])); step.dirty[draft_bank + 1].push_back(std::move(reservation.values[0]));
      }
      for (const auto &page : old.pages) old_ids.push_back(static_cast<std::int32_t>(page.page()));
      const auto begin = next_ids.size();
      for (const auto &page : next.pages) next_ids.push_back(static_cast<std::int32_t>(page.page()));
      query.push_back(static_cast<std::int32_t>((j + 1) * rows));
      old_ptr.push_back(static_cast<std::int32_t>(old_ids.size())); next_ptr.push_back(static_cast<std::int32_t>(next_ids.size()));
      old_last.push_back(old.frontier ? static_cast<std::int32_t>((old.frontier - 1) % cache::Plan::page_rows + 1) : 0);
      next_last.push_back(static_cast<std::int32_t>((next.frontier - 1) % cache::Plan::page_rows + 1));
      const std::array<std::int32_t, 2> one_query{0, static_cast<std::int32_t>(rows)}, one_kv{0, static_cast<std::int32_t>(count)};
      member.draft_layout = take(attention_schema.version({one_query, one_kv, std::span(next_ids).subspan(begin, count), std::span(&next_last.back(), 1)}));
      for (std::size_t i = 0; i < rows; ++i) positions.push_back(static_cast<std::int32_t>(old.frontier + i));
      const auto &ports = phase.draft_hidden_inputs[j];
      check(run.bind(ports[0], dense(precision.activation_type(), {1, hidden}), member.base->carry_hidden));
      if (draft_lag(role)) {
        catchups.push_back(old.catchup_token);
        check(run.bind(ports[1], dense(precision.activation_type(), {1, hidden}), member.base->prior_hidden));
      }
    }
    const auto layout = width == 1 ? step.members[0].draft_layout : take(attention_schema.version({query, next_ptr, next_ids, next_last}));
    step.draft_layout = layout;
    std::vector<std::int32_t> old_query(width == 1 ? 0 : width + 1, 0);
    const auto old_layout = width == 1 ? c::BatchLayoutVersion{} : take(attention_schema.version({old_query, old_ptr, old_ids, old_last}));
    std::vector<c::PagedArenaRead> sources(width);
    for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
      auto &arena = bank == draft_bank ? draft_inventory.keys[0] : draft_inventory.values[0];
      for (std::size_t j = 0; j < width; ++j) sources[j] = bank == draft_bank ? step.members[j].base->draft->key : step.members[j].base->draft->value;
      const auto previous = width == 1 ? sources[0] : take(arena.reframe(old_layout, sources));
      step.appends[bank] = take(g::batch::PagedAppendPages::create(arena, layout, previous,
          step.dirty[bank], step.tails[bank], {.resource_budget = cache_budget}));
    }
    phase.draft_positions.write<std::int32_t>(positions);
    if (draft_lag(role)) phase.catchup_token.write<std::int32_t>(catchups);
  }

  void prepare_evaluations(std::span<const EvaluationRequest> requests,
                           inference::PreparationObservation *preparation = nullptr,
                           GraphRole role = GraphRole::kText,
                           TextCompletion completion = TextCompletion::kEmitNext) {
    if ((completion != TextCompletion::kEmitNext && completion != TextCompletion::kStateOnly) ||
        (completion == TextCompletion::kStateOnly && role != GraphRole::kText))
      throw std::runtime_error("state-only evaluation requires ordinary Text");
    if (requests.empty() || requests.size() > cache_plan.active_sequences)
      throw std::runtime_error("batch width exceeds the configured active sequence capacity");
    if (carrier_borrowed || active_step)
      throw std::runtime_error("the original execution carrier has not retired");
    if (!requests.front().session || requests.front().session->model_ != this)
      throw std::runtime_error("batch session belongs to another model");
    const auto &prepared_sampling = requests.front().session->sampling_configuration();
    const auto sampling_index = prepared_sampling.index;
    const auto &selected_sampler = prepared_sampling.configuration;
    if (role != GraphRole::kText && sampling_index) throw std::runtime_error("nondefault sampling is ordinary Text only");
    const auto supplied_tokens = requests.front().ids.size();
    const auto tokens = drafts_token(role) ? 2u : supplied_tokens;
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
      if (role != GraphRole::kText && !stop_policy.is_default())
        throw std::runtime_error("nondefault request stop policy is ordinary Text only");
      if (&session.sampling_configuration() != &prepared_sampling) throw std::runtime_error("batch requires one cold sampling configuration");
      if (session.stopping || session.failed || session.pending)
        throw std::runtime_error("batch session is stopped, failed or already pending");
      if (!supplied_tokens || request.ids.size() != supplied_tokens || tokens > capacity - session.length())
        throw std::runtime_error("batch requires equal nonempty chunks within each remaining context");
      validate_mtp_snapshot(*session.head);
      if (has_mtp(role)) {
        if (session.head->draft->lag != draft_lag(role) ||
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
        (void)session.head->sampling.advanced(1);
    }
    if ((role == GraphRole::kText || is_mtp_prefill(role)) &&
        !find_family(requests.size(), tokens, role, false, completion, sampling_index))
      admit_exact_shape(tokens, requests.size(), role, completion, sampling_index);
    const auto step_start = Clock::now();
    auto candidate = std::make_unique<StepState>(); candidate->configure(draft_bank + 2);
    if (!candidate->members.try_reserve(requests.size())) throw std::bad_alloc{};
    for (const auto &request : requests) {
      candidate->members.push_back({.session = request.session, .base = request.session->head,
          .stop_policy = request.stop_policy});
      candidate->members.back().first_input = request.ids.front();
      if (role == GraphRole::kVerify)
        candidate->members.back().verification_inputs.assign(request.ids.begin(), request.ids.end());
    }
    active_step = std::move(candidate);
    auto &step = *active_step;
    carrier_borrowed = true; step.candidate_admission = &carrier_borrowed;
    step.step_start = step_start; step.input_tokens = tokens;
    for (auto &member : step.members) member.session->pending = &step;
    try {
      const auto width = requests.size(), slot = slot_index;
      if (!layer0_probe.empty()) prepare_child(tokens, false, width, role, completion, sampling_index);
      auto &phase = prepare_child(tokens, !layer0_probe.empty(), width, role, completion, sampling_index);
      auto &run = phase.slots[slot];
      step.phase = &phase; step.run = &run; step.nodes.assign(phase.append_nodes.begin(), phase.append_nodes.end());
      initialize_slot(phase, slot);
      auto &packed_ids = input_scratch.packed_ids, &positions = input_scratch.positions;
      packed_ids.clear(); positions.clear();
      for (const auto &request : requests) {
        if (width != 1) packed_ids.insert(packed_ids.end(), request.ids.begin(), request.ids.end());
        for (std::size_t row = 0; row < tokens; ++row)
          positions.push_back(static_cast<std::int32_t>(request.session->length() + row));
      }
      phase.tokens.write<std::int32_t>(width == 1 ? requests[0].ids : std::span<const std::int32_t>(packed_ids));
      phase.positions.write<std::int32_t>(positions);
      if (selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify) {
        auto &identities = input_scratch.sampling_identities;
        identities.clear();
        for (const auto &member : step.members) {
          if (role != GraphRole::kText) (void)member.base->sampling.advanced(drafts_token(role) ? 3 : 1);
          for (std::size_t draw = 0; draw < 3; ++draw) {
            // Non-round roles bind three stable slots but consume only slot0.
            const auto identity = member.base->sampling.identity(drafts_token(role) ? draw : 0);
            identities.insert(identities.end(), identity.begin(), identity.end());
          }
        }
        phase.sampling_identities.write<std::uint32_t>(identities);
      }
      const auto cold_start = Clock::now();
      hyle::container::SmallVector<std::int32_t, 2> query_indptr{0}, old_indptr{0}, new_indptr{0}, old_last, new_last;
      for (auto *table : {&query_indptr, &old_indptr, &new_indptr, &old_last, &new_last})
        if (!table->try_reserve(width + 1)) throw std::bad_alloc{};
      std::vector<std::int32_t> old_ids;
      auto &new_ids = input_scratch.target_page_ids;
      new_ids.clear();
      std::size_t old_refs = 0, new_refs = 0;
      for (const auto &member : step.members) {
        old_refs = cache::add(old_refs, member.base->pages.size());
        new_refs = cache::add(new_refs, (member.base->length + tokens + cache::Plan::page_rows - 1) / cache::Plan::page_rows);
      }
      if (width != 1) old_ids.reserve(old_refs);
      if (new_refs > new_ids.capacity())
        throw std::runtime_error("target page-index scratch exceeds prepared schema capacity");
      for (auto &bank : step.tails) if (!bank.try_reserve(width)) throw std::bad_alloc{};
      for (std::size_t j = 0; j < width; ++j) {
        auto &member = step.members[j]; auto &session = *member.session;
        const auto length = session.length();
        member.next = reserve_snapshot();
        if (!member.next_tails.try_resize(draft_bank)) throw std::bad_alloc{};
        member.next->length = length + tokens;
        member.next->sampling = selected_sampler.enabled && completion == TextCompletion::kEmitNext && role != GraphRole::kVerify ? member.base->sampling.advanced(drafts_token(role) ? 3 : 1)
            : member.base->sampling;
        const auto full_prefix_pages = length / cache::Plan::page_rows;
        const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
        const auto dirty_count = page_count - full_prefix_pages;
        const auto tail_rows = length % cache::Plan::page_rows;
        bool reuse_tail = tail_rows && session.tail_head == member.base.get() && !member.base->pages.empty();
        if (reuse_tail) for (const auto &tail : session.tails)
          reuse_tail = reuse_tail && tail.valid() && !tail.reserved_rows() &&
              tail.page() == member.base->pages.back().page() && tail.initialized_rows() == tail_rows;
        member.next->pages.reserve(page_count);
        for (auto &bank : step.dirty) bank.reserve(cache::add(bank.size(), dirty_count - (reuse_tail ? 1 : 0)));
        if (reuse_tail) {
          const auto rows = static_cast<std::uint32_t>((std::min)(tokens, cache::Plan::page_rows - tail_rows));
          std::size_t reserved = 0; hyle::Status status;
          for (; reserved < session.tails.size(); ++reserved) {
            status = session.tails[reserved].reserve_rows(rows); if (!status.ok()) break;
          }
          if (!status.ok()) {
            for (std::size_t i = 0; i < reserved; ++i) check(session.tails[i].rollback());
            if (!status.IsUnavailable()) check(status);
            reuse_tail = false;
            for (auto &bank : step.dirty) bank.reserve(cache::add(bank.size(), dirty_count));
          }
        }
        for (std::size_t i = 0; i < full_prefix_pages; ++i) member.next->pages.push_back(member.base->pages.at(i));
        if (reuse_tail) {
          member.next->pages.push_back(member.base->pages.back());
          for (std::size_t bank = 0; bank < session.tails.size(); ++bank)
            step.tails[bank].push_back(std::move(session.tails[bank]));
        } else for (auto &tail : session.tails) tail = {};
        session.tail_head = nullptr;
        for (std::size_t page = reuse_tail ? 1 : 0; page < dirty_count; ++page) {
          auto reservation = take(page_inventory.acquire());
          member.next->pages.push_back(std::move(reservation.reference));
          for (std::size_t layer = 0; layer < reservation.keys.size(); ++layer) {
            step.dirty[layer * 2].push_back(std::move(reservation.keys[layer]));
            step.dirty[layer * 2 + 1].push_back(std::move(reservation.values[layer]));
          }
        }
        if (width != 1) for (const auto &page : member.base->pages) old_ids.push_back(static_cast<std::int32_t>(page.page()));
        const auto begin = new_ids.size();
        for (const auto &page : member.next->pages) new_ids.push_back(static_cast<std::int32_t>(page.page()));
        query_indptr.push_back(static_cast<std::int32_t>((j + 1) * tokens));
        old_indptr.push_back(static_cast<std::int32_t>(old_ids.size()));
        new_indptr.push_back(static_cast<std::int32_t>(new_ids.size()));
        old_last.push_back(length ? static_cast<std::int32_t>((length - 1) % cache::Plan::page_rows + 1) : 0);
        new_last.push_back(static_cast<std::int32_t>((member.next->length - 1) % cache::Plan::page_rows + 1));
        const std::array<std::int32_t, 2> one_query{0, static_cast<std::int32_t>(tokens)}, one_kv{0, static_cast<std::int32_t>(page_count)};
        member.layout = take(attention_schema.version({one_query, one_kv,
            std::span(new_ids).subspan(begin, page_count), std::span(&new_last.back(), 1)}));
      }
      phase.attention_layout = width == 1 ? step.members[0].layout
          : take(attention_schema.version({query_indptr, new_indptr, new_ids, new_last}));
      // Old layouts describe no new queries, but retain each independent history.
      const std::vector<std::int32_t> old_query(width == 1 ? 0 : width + 1, 0);
      const auto old_layout = width == 1 ? c::BatchLayoutVersion{}
          : take(attention_schema.version({old_query, old_indptr, old_ids, old_last}));
      std::vector<c::PagedArenaRead> sources(width == 1 ? 0 : width);
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        const auto compose = [&](c::PagedArena &arena, bool key) {
          if (width == 1) return key ? step.members[0].base->keys[layer] : step.members[0].base->values[layer];
          for (std::size_t j = 0; j < width; ++j)
            sources[j] = key ? step.members[j].base->keys[layer] : step.members[j].base->values[layer];
          return take(arena.reframe(old_layout, sources));
        };
        auto &key_arena = page_inventory.keys[layer]; auto &value_arena = page_inventory.values[layer];
        step.appends[layer * 2] = take(g::batch::PagedAppendPages::create(key_arena, phase.attention_layout,
            compose(key_arena, true), step.dirty[layer * 2], step.tails[layer * 2], {.resource_budget = cache_budget}));
        step.appends[layer * 2 + 1] = take(g::batch::PagedAppendPages::create(value_arena, phase.attention_layout,
            compose(value_arena, false), step.dirty[layer * 2 + 1], step.tails[layer * 2 + 1], {.resource_budget = cache_budget}));
      }
      if (has_mtp(role)) prepare_draft_append(step, role);
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i))
        for (std::size_t j = 0; j < width; ++j) {
          const auto &ports = phase.layers[i].sequences[j]; const auto &descriptions = state_schema[i];
          check(run.bind(ports.history, descriptions.history, step.members[j].base->state[i].history));
          check(run.bind(ports.recurrent, descriptions.recurrent, step.members[j].base->state[i].recurrent));
          const auto bind_state = [&](V port, c::LeasePool &pool) {
            const auto &output = phase.child->output(port);
            if (output.caller_owned())
              check(run.bind(output.value, output.description, take(pool.allocate(take(output.description.required_bytes()), cache::state_pool_alignment))));
          };
          for (auto row = phase.prefix_rows; row != 0; --row) {
            bind_state(ports.history_prefix[row - 1], state_pools[i].history);
            bind_state(ports.recurrent_prefix[row - 1], state_pools[i].recurrent);
          }
        }
      if (verifies_prefix(role) || has_mtp(role) || retains_sampling_frontier()) for (const auto &ports : phase.carry_ports)
        for (std::size_t i = 0; i < phase.prefix_rows; ++i)
          if (const auto &output = phase.child->output(ports[i]); output.caller_owned())
            check(run.bind(output.value, output.description, take(hidden_pool.allocate(geometry.hidden_bytes(), cache::state_pool_alignment))));
      std::vector<g::batch::NodeLayout> layouts;
      layouts.reserve(target_kv_layers + (drafts_token(role) ? 1 : 0));
      for (std::size_t i = 0; i < phase.attention_nodes.size(); ++i) layouts.push_back({phase.attention_nodes[i], phase.attention_layout});
      if (drafts_token(role)) layouts.push_back({phase.draft_attention_node, step.draft_layout});
      const auto queues = inference::queue_span(phase.queues);
      if (preparation) { preparation->slot = slot; preparation->phase_slot_prior_uses = phase.preparation_uses[slot]; }
      const auto batch_start = Clock::now();
      check(g::batch::prepare(run, queues, layouts, {.host_workspace = execution.staging()}));
      for (std::size_t layer = 0; layer < phase.attention_nodes.size(); ++layer) {
        const auto &ports = phase.layers[attention_layer_ids[layer]];
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2], ports.key,
            std::move(step.appends[layer * 2]))); step.bound[layer * 2] = true;
        check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[layer * 2 + 1], ports.value,
            std::move(step.appends[layer * 2 + 1]))); step.bound[layer * 2 + 1] = true;
      }
      if (has_mtp(role)) for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
        if (drafts_token(role)) check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank],
            bank == draft_bank ? phase.draft_key_port : phase.draft_value_port, std::move(step.appends[bank])));
        else check(g::batch::bind_append(run, phase.family->graph, phase.append_nodes[bank], std::move(step.appends[bank])));
        step.bound[bank] = true;
      }
      const auto batch_end = Clock::now();
      check(run.prepare_replay(queues, {.mode = phase.replay_modes[slot], .host_wait = g::HostWaitPolicy::kAllow}));
      const auto replay_end = Clock::now();
      if (preparation) {
        preparation->layout_seconds = std::chrono::duration<double>(batch_start - cold_start).count();
        preparation->batch_prepare_seconds = std::chrono::duration<double>(batch_end - batch_start).count();
        preparation->replay_prepare_seconds = std::chrono::duration<double>(replay_end - batch_end).count();
      }
      step.replay_route = run.replay_route();
      const auto replay_report = take(run.replay_report());
      step.capture_recorded = replay_report.recorded_capture_range_count;
      step.capture_reused = replay_report.reused_capture_range_count;
      step.cold_batch_seconds = std::chrono::duration<double>(Clock::now() - cold_start).count();
      ++phase.preparation_uses[slot]; step.prepared = true;
    } catch (inference::EvaluationFailure &failure) {
      failure.observe_retirement(abort_episode());
      throw;
    } catch (...) { (void)abort_episode(); throw; }
  }

  void prepare_evaluation(Session &session, std::span<const std::int32_t> ids,
                          inference::PreparationObservation *preparation = nullptr,
                          GraphRole role = GraphRole::kText, TextCompletion completion = TextCompletion::kEmitNext,
                          const inference::RequestStopPolicy *stop_policy = nullptr) {
    const std::array requests{EvaluationRequest{session.shared_from_this(), ids, stop_policy}};
    prepare_evaluations(requests, preparation, role, completion);
  }

  void submit_evaluations() {
    if (!active_step || !active_step->prepared || active_step->submitted || active_step->cancelled || !active_step->run)
      throw std::runtime_error("batch is not prepared or was already submitted/cancelled");
    for (const auto &member : active_step->members)
      if (member.session->stopping || member.session->failed)
        throw std::runtime_error("batch participant stopped before submission");
    const auto queues = inference::queue_span(active_step->phase->queues);
    auto &step = *active_step;
    step.start = Clock::now(); step.submitted = true;
    step.completion = submit_retained(*step.run, queues, step.phase->retirement[slot_index]);
    step.submit_seconds = std::chrono::duration<double>(Clock::now() - step.start).count();
  }

  void submit_evaluation(Session &session) {
    if (session.model_ != this || !active_step || session.pending != active_step.get() || active_step->members.size() != 1)
      throw std::runtime_error("single submission requires its original one-request episode");
    submit_evaluations();
  }

  template<class Select, class Complete, class Observe>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Complete complete, Observe observe) {
    if (!active_step || outputs.size() != active_step->members.size() || next_ids.size() != outputs.size() ||
        active_step->cancelled || !active_step->submitted || !active_step->completion.valid())
      throw std::runtime_error("finish requires its original complete noncancelled batch");
    for (std::size_t j = 0; j < outputs.size(); ++j)
      if (outputs[j].session != active_step->members[j].session || outputs[j].session->stopping || outputs[j].session->failed)
        throw std::runtime_error("finish participant does not match the original active batch");
    auto &step = *active_step; auto &phase = *step.phase; auto &run = *step.run;
    const auto step_start = step.step_start, start = step.start;
    const auto tokens = step.input_tokens;
    const auto &selected_sampler = phase.family->sampling->configuration;
    const bool state_only = phase.family->completion == TextCompletion::kStateOnly;
    const auto submit_seconds = step.submit_seconds, cold_batch_seconds = step.cold_batch_seconds;
    const auto replay_route = step.replay_route;
    const auto capture_recorded = step.capture_recorded, capture_reused = step.capture_reused;
    std::array<c::ConstTensorView, 9> probe_views;
    std::array<c::ReadLease, 9> probe_owners;
    c::ReadLease generated_logits, generated_hidden;
    struct ReturnReadback final { bool *admission = nullptr; ~ReturnReadback() { if (admission) *admission = false; } } return_readback;
    try {
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        if (outputs[j].generation_ledger) {
          if (cache_plan.draft_cache != (phase.family->role != GraphRole::kText) ||
              step.members[j].base->draft.has_value() != cache_plan.draft_cache)
            throw std::runtime_error("generation role differs from its complete target/draft participant");
          for (std::size_t k = 0; k < j; ++k) if (outputs[k].generation_ledger == outputs[j].generation_ledger)
            throw std::runtime_error("batch participants cannot share a mutable generation ledger");
          if (outputs[j].generation_ledger->stop_policy() !=
              inference::request_stop_policy(step.members[j].stop_policy))
            throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
          auto &record = outputs[j].generation_ledger->next();
          step.members[j].commit_ledger = outputs[j].generation_ledger; step.members[j].commit_record = &record;
          if (record.role != phase.family->role || record.completion != phase.family->completion || record.before != generation_boundary(*step.members[j].base) ||
              record.ids().size() != (record.generation ? 1u : tokens) ||
              record.ids().empty() || record.ids().front() != step.members[j].first_input)
            throw std::runtime_error("generation ledger entry differs from its actual episode");
          if (outputs[j].expected_episode && !outputs[j].expected_episode->committed)
            throw std::runtime_error("expected episode record is not committed");
        } else if (outputs[j].expected_episode)
          throw std::runtime_error("expected episode record requires a fresh commit witness");
      }
      check(step.completion.wait()); step.completion = {};
      // Trace storage must be pinned before take_append/unbind revoke Graph
      // output observation. Generated arrays keep their actual native owner.
      if (std::ranges::any_of(outputs, [](const auto &output) { return output.capture_tensors; })) {
        if (!state_only && !phase.logits.caller_owned()) generated_logits = take(inference::retain_output(run, *phase.logits_facts, allocation_budget));
        if (!phase.final_hidden.caller_owned()) generated_hidden = take(inference::retain_output(run, *phase.hidden_facts, allocation_budget));
      }
      if (drafts_token(phase.family->role)) {
        check(phase.draft_id.read(execution, run, std::as_writable_bytes(std::span(phase.draft_ids))));
        for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].draft_token = phase.draft_ids[j];
      }
      if (selected_sampler.enabled && !state_only && phase.family->role != GraphRole::kVerify) {
        check(phase.sampling_valid.predicates(execution, run, phase.sampling_valid_values, phase.sampling_scratch));
        const auto bad = std::ranges::find_if(phase.sampling_valid_values, [](auto value) { return value != 1; });
        if (bad != phase.sampling_valid_values.end()) {
          if (phase.family->role == GraphRole::kText) {
            std::optional<std::size_t> participant;
            // Everything before the first bad value is already proven 1.
            // Only the failure path inspects the remainder for a unique 0.
            if (*bad == 0 && std::all_of(bad + 1, phase.sampling_valid_values.end(),
                [](auto value) { return value == 1; }))
              participant = static_cast<std::size_t>(bad - phase.sampling_valid_values.begin());
            throw inference::EvaluationFailure(hyle::Status::InvalidArgument(
                "sampling distribution or proposal is invalid; no session or RNG cursor was published"), participant);
          }
          throw std::runtime_error("sampling distribution or proposal is invalid; no session or RNG cursor was published");
        }
        if (drafts_token(phase.family->role)) {
          check(phase.sampling_accept.predicates(execution, run, phase.sampling_accept_values, phase.sampling_scratch));
          for (std::size_t j = 0; j < outputs.size(); ++j) step.members[j].sampled_accept = phase.sampling_accept_values[j] != 0;
        }
      }
      // Selection observes only retired original Graph results. No head has
      // been published and all state/append guards still belong to this episode.
      if (verifies_prefix(phase.family->role)) {
        check(phase.next_id.read(execution, run, std::as_writable_bytes(std::span(phase.prefix_candidates))));
        select(std::span<const std::int32_t>(phase.prefix_candidates), step, next_ids);
      } else if (state_only) {
        std::ranges::fill(next_ids, -1);
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      } else {
        check(phase.next_id.read(execution, run, std::as_writable_bytes(next_ids)));
        select(std::span<const std::int32_t>(next_ids), step, next_ids);
      }
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        auto &member = step.members[j];
        if (!member.accepted_tokens || member.accepted_tokens > tokens ||
            (member.accepted_tokens != tokens && !verifies_prefix(phase.family->role)) ||
            (!state_only && (next_ids[j] < 0 || next_ids[j] >= vocabulary)))
          throw std::runtime_error("invalid accepted target boundary");
        member.verified_last_page = member.next->pages.back().page();
        if (member.accepted_tokens != tokens) {
          member.next->length = member.base->length + member.accepted_tokens;
          const auto page_count = (member.next->length + cache::Plan::page_rows - 1) / cache::Plan::page_rows;
          member.next->pages.resize(page_count);
          std::vector<std::int32_t> ids;
          ids.reserve(page_count);
          for (const auto &page : member.next->pages) ids.push_back(static_cast<std::int32_t>(page.page()));
          const std::array<std::int32_t, 2> query{0, static_cast<std::int32_t>(member.accepted_tokens)},
              kv{0, static_cast<std::int32_t>(page_count)};
          const std::array last{static_cast<std::int32_t>((member.next->length - 1) % cache::Plan::page_rows + 1)};
          member.layout = take(attention_schema.version({query, kv, ids, last}));
        }
      }
      // Export every participant's ordinary outputs before taking ANY append:
      // the first take_append revokes Graph output-ready for the entire batch.
      for (std::size_t i = 0; i < state_schema.size(); ++i) if (geometry.is_gdn(i))
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          const auto &ports = phase.layers[i].sequences[j];
          const auto row = verifies_prefix(phase.family->role) ? step.members[j].accepted_tokens - 1 : 0;
          step.members[j].next->state[i].history = retain_state_storage(run, phase.child->output(ports.history_prefix.at(row)));
          step.members[j].next->state[i].recurrent = retain_state_storage(run, phase.child->output(ports.recurrent_prefix.at(row)));
        }
      if (verifies_prefix(phase.family->role) || has_mtp(phase.family->role) || retains_sampling_frontier())
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          const auto &ports = phase.carry_ports[j];
          member.next->carry_hidden = retain_state_storage(run,
              phase.child->output(ports[verifies_prefix(phase.family->role) ? member.accepted_tokens - 1 : 0]));
          if (verifies_prefix(phase.family->role) && member.accepted_tokens > 1) {
            member.next->prior_hidden = retain_state_storage(run, phase.child->output(ports[member.accepted_tokens - 2]));
          }
          if (phase.family->role == GraphRole::kVerify && cache_plan.draft_cache) {
            member.carry_prefix.reserve(member.accepted_tokens);
            for (std::size_t row = 0; row < member.accepted_tokens; ++row)
              member.carry_prefix.push_back(retain_state_storage(run, phase.child->output(ports[row])));
          }
        }
      if (has_mtp(phase.family->role)) for (auto &member : step.members) {
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
      for (std::size_t layer = 0; layer < page_inventory.keys.size(); ++layer) {
        for (const auto bank : {layer * 2, layer * 2 + 1}) {
          step.appends[bank] = take(g::batch::take_append(run, step.nodes[bank])); step.bound[bank] = false;
        }
        auto key = take(step.appends[layer * 2].ready()), value = take(step.appends[layer * 2 + 1].ready());
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j];
          member.next->keys[layer] = outputs.size() == 1 && member.accepted_tokens == tokens ? key
              : take(page_inventory.keys[layer].reframe(member.layout, std::span(&key, 1)));
          member.next->values[layer] = outputs.size() == 1 && member.accepted_tokens == tokens ? value
              : take(page_inventory.values[layer].reframe(member.layout, std::span(&value, 1)));
        }
      }
      if (has_mtp(phase.family->role)) {
        for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank) {
          step.appends[bank] = take(g::batch::take_append(run, step.nodes[bank])); step.bound[bank] = false;
        }
        auto key = take(step.appends[draft_bank].ready()), value = take(step.appends[draft_bank + 1].ready());
        std::vector<std::pair<std::uint32_t, std::size_t>> owners; owners.reserve(outputs.size());
        for (std::size_t j = 0; j < outputs.size(); ++j) {
          auto &member = step.members[j]; auto &draft = *member.next->draft;
          draft.key = outputs.size() == 1 ? key : take(draft_inventory.keys[0].reframe(member.draft_layout, std::span(&key, 1)));
          draft.value = outputs.size() == 1 ? value : take(draft_inventory.values[0].reframe(member.draft_layout, std::span(&value, 1)));
          owners.emplace_back(draft.pages.back().page(), j);
        }
        std::sort(owners.begin(), owners.end());
        for (std::size_t bank = draft_bank; bank < draft_bank + 2; ++bank)
          for (std::size_t i = 0; i < step.appends[bank].appender_count(); ++i) {
            auto &tail = take(step.appends[bank].appender(i)); if (!tail.valid()) continue;
            const auto found = std::lower_bound(owners.begin(), owners.end(), std::pair{tail.page(), std::size_t{0}});
            if (found == owners.end() || found->first != tail.page())
              throw std::runtime_error("draft append returned a tail outside all final pages");
            auto &member = step.members[found->second]; auto &draft = *member.next->draft;
            if (tail.reserved_rows() || member.next_draft_tails[bank - draft_bank].valid() ||
                tail.initialized_rows() != draft.frontier % cache::Plan::page_rows)
              throw std::runtime_error("draft append returned an invalid committed frontier");
            member.next_draft_tails[bank - draft_bank] = std::move(tail);
          }
      }
      // Caller-supplied completion is confined to this unpublished episode.
      // In particular, recursive proposals must repair their retained KV from
      // actual target hidden rows here, before ledger checks or any head swap.
      complete(step);
      if (cache_plan.draft_cache)
        for (const auto &member : step.members) validate_mtp_snapshot(*member.next);
      for (std::size_t j = 0; j < outputs.size(); ++j) if (auto *record = step.members[j].commit_record) {
        const auto &member = step.members[j];
        record->requests = outputs.size(); record->member_index = j;
        record->accepted = member.accepted_tokens;
        record->draft_rows = member.next->draft
            ? member.next->draft->frontier - member.base->draft->frontier : 0;
        if (phase.family->role == GraphRole::kVerify)
          record->draft_rows = cache::add(record->draft_rows,
              tokens - 1 + member.base->draft->lag + member.discarded_draft_rows);
        record->after = generation_boundary(*member.next); record->after.next_id = next_ids[j];
        if (phase.family->role == GraphRole::kVerify)
          record->accepted_inputs.assign(member.verification_inputs.begin(),
              member.verification_inputs.begin() + member.accepted_tokens);
        else record->accepted_inputs = {member.first_input, member.accepted_tokens == 2 ? member.draft_token : -1};
        if (record->generation) {
          if (phase.family->role == GraphRole::kVerify) record->verification_inputs.assign(
              member.verification_inputs.begin(), member.verification_inputs.end());
          else if (drafts_token(phase.family->role)) record->verification_inputs = {member.first_input, member.draft_token};
          else record->verification_inputs = {member.first_input};
        }
        record->verified_rows = tokens;
        if (!record->generation || phase.family->role == GraphRole::kText) {
          record->draft = -1; record->target_first = next_ids[j]; record->target_second = -1;
          record->agreed = false; record->accepted_draft = false;
          record->stopped = !state_only && (phase.family->role == GraphRole::kText
              ? inference::request_stop_policy(member.stop_policy).stopped(next_ids[j], is_end_of_generation(next_ids[j]))
              : is_end_of_generation(next_ids[j]));
          if (state_only) { record->emitted.clear(); record->emitted_count = 0; }
          else { record->emitted = {next_ids[j], -1}; record->emitted_count = 1; }
        }
        if (outputs[j].expected_episode && !record->same_result(*outputs[j].expected_episode))
          throw std::runtime_error("MTP replay natural decision or state boundary diverged from its recorded episode");
      }
      hyle::container::SmallVector<std::pair<std::uint32_t, std::size_t>, 1> tail_owners;
      if (!tail_owners.try_reserve(step.members.size())) throw std::bad_alloc{};
      for (std::size_t j = 0; j < step.members.size(); ++j)
        tail_owners.emplace_back(step.members[j].verified_last_page, j);
      std::sort(tail_owners.begin(), tail_owners.end());
      for (std::size_t bank = 0; bank < draft_bank; ++bank)
        for (std::size_t i = 0; i < step.appends[bank].appender_count(); ++i) {
          auto &tail = take(step.appends[bank].appender(i)); if (!tail.valid()) continue;
          const auto found = std::lower_bound(tail_owners.begin(), tail_owners.end(), std::pair{tail.page(), std::size_t{0}});
          if (found == tail_owners.end() || found->first != tail.page())
            throw std::runtime_error("batch append returned a tail outside all final pages");
          auto &member = step.members[found->second];
          // A ready T2 tail already owns the verified physical frontier.
          // Partial acceptance only narrows read authority; discard its mutable
          // capability and use the original COW path next, never rewind it.
          if (member.accepted_tokens != tokens) { tail = {}; continue; }
          if (tail.reserved_rows() || member.next_tails[bank].valid() || tail.initialized_rows() != member.next->length % cache::Plan::page_rows)
            throw std::runtime_error("batch append returned an invalid committed tail");
          member.next_tails[bank] = std::move(tail);
        }
      check(run.unbind(phase.state_resource_ports));
      for (const auto &member : step.members)
        if (member.session->head != member.base || member.session->pending != &step || member.session->stopping)
          throw std::runtime_error("batch snapshot expected-base conflict");
      // All fallible native/export/metadata work is complete. The head swaps
      // below cannot throw and publish the whole batch in this host transaction.
      for (std::size_t j = 0; j < step.members.size(); ++j) {
        auto &member = step.members[j]; member.next->next_id = next_ids[j];
        member.session->tails = std::move(member.next_tails);
        member.session->draft_tails = std::move(member.next_draft_tails);
        member.session->head = std::move(member.next);
        member.session->tail_head = member.session->head.get();
        member.session->draft_tail_head = member.session->head.get();
      }
      // Every bound and expected decision was checked before publication.
      // The reserved host slot is now authoritative even if trace/report IO
      // below throws. No callback, allocation or permission is acquired here.
      for (auto &member : step.members) if (member.commit_record) {
        member.commit_record->committed = true;
        ++member.commit_ledger->committed_count;
      }
      return_readback.admission = std::exchange(step.candidate_admission, nullptr);
      detach_episode(); // Old states/transactions retire inside the host endpoint.
    } catch (inference::EvaluationFailure &failure) {
      failure.observe_retirement(abort_episode());
      throw;
    } catch (...) { (void)abort_episode(); throw; }
    const auto finished = Clock::now();
    for (const auto &output : outputs) if (output.generation_ledger) {
      auto &record = output.generation_ledger->records[output.generation_ledger->committed_count - 1];
      record.endpoint = finished;
      record.step_seconds = std::chrono::duration<double>(finished - step_start).count();
    }
    for (std::size_t j = 0; j < outputs.size(); ++j) {
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
      observation.input_tokens = tokens; observation.visible = session.length(); observation.next = next;
      observation.graph_executed = true; observation.graph_input_tokens = tokens;
      observation.seconds = std::chrono::duration<double>(finished - start).count();
      observation.submit_seconds = submit_seconds; observation.step_seconds = std::chrono::duration<double>(finished - step_start).count();
      observation.cold_batch_seconds = cold_batch_seconds; observation.replay_route = replay_route;
      observation.capture_recorded = capture_recorded; observation.capture_reused = capture_reused;
      observation.scheduled_progress = static_cast<bool>(scheduler);
      CompletedEvaluationView completed{.observation = observation, .head = *session.head,
          .member_index = j, .requests = outputs.size()};
      if (output.capture_tensors) {
        if (!state_only) completed.logits = phase.logits.caller_owned() ? phase.logits.view().as_const()
            : take(generated_logits.tensor(phase.logits.description));
        completed.hidden = phase.final_hidden.caller_owned() ? phase.final_hidden.view().as_const()
            : take(generated_hidden.tensor(phase.final_hidden.description));
      }
      if (output.capture_probes) completed.probes = probe_views;
      observe(j, completed);
    }
  }

  template<class Select, class Observe>
  void finish_selected_evaluations(std::span<const EvaluationOutput> outputs,
                                  std::span<std::int32_t> next_ids, Select select, Observe observe) {
    finish_selected_evaluations(outputs, next_ids, std::move(select), [](StepState &) {}, std::move(observe));
  }

  // Returns the actual verifier input [pending, proposal0, ...]. Approximate
  // hidden is local to this recursion and is deliberately discarded with its
  // private pages; it is never installed in a committed draft cache.
  std::vector<std::int32_t> propose_recursive(Session &session, std::size_t maximum,
      std::span<const std::int32_t> eos = {}, std::vector<c::ReadLease> *q = nullptr) {
    if (active_step || carrier_borrowed || session.pending || session.stopping || session.failed ||
        !session.head || !session.head->draft || (default_sampler.enabled && !q) ||
        maximum >= capacity - session.length() || maximum >= cache_plan.candidate_boundaries)
      throw std::runtime_error("recursive proposals require an idle prepared head, bounded width and every stochastic q row");
    validate_mtp_snapshot(*session.head);
    const auto ended = [&](auto token) { return std::find(eos.begin(), eos.end(), token) != eos.end(); };
    std::vector<std::int32_t> inputs; inputs.reserve(maximum + 1); inputs.push_back(session.head->next_id);
    if (q) { q->clear(); q->reserve(maximum); }
    if (!maximum) return inputs;
    if (inputs.front() < 0 || inputs.front() >= vocabulary || ended(inputs.front()))
      throw std::runtime_error("recursive proposals require a valid nonterminal emitted pending token");
    DraftWork work; work.state = *session.head->draft;
    c::ReadLease approximate;
    for (std::size_t i = 0; i < maximum; ++i) {
      std::vector<std::int32_t> ids;
      std::vector<c::ReadLease> shifted;
      if (!i && work.state.lag) { ids.push_back(work.state.catchup_token); shifted.push_back(session.head->prior_hidden); }
      ids.push_back(inputs.back()); shifted.push_back(i ? approximate : session.head->carry_hidden);
      auto proposal = execute_draft(work, ids, shifted, true,
          default_sampler.enabled ? session.head->sampling.identity(2 * i) : std::array<std::uint32_t, 4>{});
      // Stop on a distribution property, not the sampled proposal's own
      // probability. The stopping condition is independent of this draw; its
      // unpublished address may then be the target's final draw at 2*m.
      if (min_draft_probability() > 0 && proposal.probability < min_draft_probability()) break;
      inputs.push_back(proposal.id); approximate = std::move(proposal.hidden);
      if (q && default_sampler.enabled) q->push_back(std::move(proposal.log_probabilities));
      if (ended(proposal.id)) break;
    }
    return inputs;
  }

  void repair_recursive_draft(StepState &step) {
    if (step.phase->family->role != GraphRole::kVerify || !cache_plan.draft_cache)
      throw std::runtime_error("draft repair requires its unpublished target verifier");
    for (auto &member : step.members) {
      const auto n = member.accepted_tokens;
      if (!n || member.verification_inputs.size() < n || member.carry_prefix.size() != n)
        throw std::runtime_error("draft repair requires every actually consumed target prefix row");
      DraftWork repaired; repaired.state = *member.base->draft;
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

  template<class Observe>
  void finish_recursive_mtp(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe) {
    if (!active_step || selections.empty() || active_step->phase->family->role != GraphRole::kVerify ||
        !cache_plan.draft_cache || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size())
      throw std::runtime_error("recursive finish requires the complete original verification batch");
    const auto ended = [&](auto token) { return std::find(eos.begin(), eos.end(), token) != eos.end(); };
    std::vector<std::int32_t> next(selections.size(), -1);
    for (std::size_t j = 0; j < selections.size(); ++j) {
      const auto &selection = selections[j];
      if (!selection.output_quota || selection.session != active_step->members[j].session ||
          selection.session != outputs[j].session || ended(selection.expected_pending))
        throw std::runtime_error("recursive generation requires its nonterminal pending token and positive quota");
      results[j].emitted.clear(); results[j].emitted.reserve(active_step->input_tokens);
    }
    finish_selected_evaluations(outputs, next,
        [&](std::span<const std::int32_t> candidates, StepState &step, std::span<std::int32_t> selected) {
      c::ReadLease logits_owner;
      mem::ConstBufferView logits_storage;
      if (default_sampler.enabled) {
        if (step.phase->logits.caller_owned()) logits_storage = step.phase->logits.storage();
        else {
          logits_owner = take(inference::retain_output(*step.run, *step.phase->logits_facts, allocation_budget));
          logits_storage = take(logits_owner.view());
        }
      }
      for (std::size_t j = 0; j < selections.size(); ++j) {
        auto &member = step.members[j]; auto &result = results[j]; const auto &selection = selections[j];
        if (member.first_input != selection.expected_pending || selection.expected_pending != member.base->next_id)
          throw std::runtime_error("recursive verifier input differs from the emitted pending token");
        const auto width = step.input_tokens;
        const auto rows = candidates.subspan(j * width, width);
        result.pending = selection.expected_pending; result.proposal_count = width - 1; result.verified_rows = width;
        result.auxiliary_graph_calls = width + (default_sampler.enabled ? 1u : 0u);
        result.draft = width > 1 ? member.verification_inputs[1] : -1;
        result.target_first = rows[0]; result.target_second = width > 1 ? rows[1] : -1;
        hyle::inference::mtp::AcceptedPrefix accepted;
        if (default_sampler.enabled) {
          if (selection.draft_log_probabilities.size() != width - 1)
            throw std::runtime_error("stochastic verification requires every actual q row");
          const auto bytes = cache::multiply(cache::multiply(width, vocabulary), sizeof(float));
          const auto logits = take(c::ConstTensorView::create(dense(Type::kFloat32,
              {static_cast<std::int64_t>(width), vocabulary}), take(logits_storage.subview(j * bytes, bytes))));
          const auto decision = sample_recursive_prefix(logits,
              std::span<const std::int32_t>(member.verification_inputs).subspan(1),
              selection.draft_log_probabilities, member.base->sampling);
          accepted = hyle::inference::mtp::select_sampled_prefix(member.verification_inputs, decision.accepted, decision.final_ids,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
          member.next->sampling = member.base->sampling.advanced(cache::add(cache::multiply(2, width - 1), 1));
          result.target_first = result.emitted.front();
          result.target_second = result.emitted.size() > 1 ? result.emitted[1] : -1;
        } else {
          accepted = hyle::inference::mtp::select_greedy_prefix(member.verification_inputs, rows,
              selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
        }
        member.accepted_tokens = accepted.consumed; selected[j] = result.next_id = accepted.next_id;
        result.agreed = accepted.first_agrees; result.consumed_drafts = accepted.consumed_drafts;
        const bool clipped = min_draft_probability() > 0 && selection.proposal_limit > width - 1 &&
            !ended(member.verification_inputs.back());
        member.discarded_draft_rows = clipped ? (width == 1 ? member.base->draft->lag + 1 : 1) : 0;
        result.auxiliary_graph_calls += clipped ? 1u : 0u;
        result.accepted_draft = result.consumed_drafts != 0;
        result.processed_target = member.accepted_tokens;
        result.processed_draft = width - 1 + 2 * member.base->draft->lag +
            (member.accepted_tokens == 1 ? 1 : member.accepted_tokens - 1) + member.discarded_draft_rows;
        result.stopped = accepted.stopped;
        if (outputs[j].generation_ledger) {
          auto &record = outputs[j].generation_ledger->next();
          if (!record.generation || record.quota != selection.output_quota)
            throw std::runtime_error("recursive MTP ledger differs from its original output quota");
          record.draft = result.draft; record.target_first = result.target_first;
          record.target_second = result.target_second; record.agreed = result.agreed;
          record.accepted_draft = result.accepted_draft; record.consumed_drafts = result.consumed_drafts;
          record.auxiliary_graph_calls = result.auxiliary_graph_calls;
          record.stopped = result.stopped; record.emitted.assign(result.emitted.begin(), result.emitted.end());
          record.emitted_count = result.emitted.size();
        }
      }
    }, [&](StepState &step) { repair_recursive_draft(step); }, std::move(observe));
  }

  template<class Observe>
  void finish_evaluations(std::span<const EvaluationOutput> outputs,
                         std::span<std::int32_t> next_ids, Observe observe) {
    if (active_step && active_step->phase && verifies_prefix(active_step->phase->family->role))
      throw std::runtime_error("verification requires explicit accepted-boundary selection");
    finish_selected_evaluations(outputs, next_ids,
        [](std::span<const std::int32_t> ids, StepState &step, std::span<std::int32_t> next) {
          for (std::size_t j = 0; j < step.members.size(); ++j) {
            step.members[j].accepted_tokens = step.input_tokens; next[j] = ids[j];
          }
        }, std::move(observe));
  }

  // Generation policy observes only the original terminal graph. Pending has
  // already been emitted; an accepted draft and its bonus are new outputs.
  template<class Observe>
  void finish_mtp_rounds(std::span<const MtpGenerationSelection> selections,
      std::span<const std::int32_t> eos, std::span<const EvaluationOutput> outputs,
      std::span<MtpRoundResult> results, Observe observe) {
    if (!active_step || selections.empty() || selections.size() != active_step->members.size() ||
        outputs.size() != selections.size() || results.size() != selections.size() || !active_step->phase ||
        !has_mtp(active_step->phase->family->role) || is_mtp_prefill(active_step->phase->family->role))
      throw std::runtime_error("MTP finish requires its original complete generation batch");
    const auto role = active_step->phase->family->role;
    const auto ended = [&](std::int32_t id) { return std::find(eos.begin(), eos.end(), id) != eos.end(); };
    for (std::size_t j = 0; j < selections.size(); ++j) {
      if (!selections[j].output_quota || selections[j].session != active_step->members[j].session ||
          selections[j].session != outputs[j].session || selections[j].expected_pending < 0 ||
          selections[j].expected_pending >= vocabulary || ended(selections[j].expected_pending))
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
            if (default_sampler.enabled) {
              const std::array flags{std::uint8_t(member.sampled_accept)};
              // The original K1 graph's second ID is its residual/bonus final;
              // its first ID is already Select(accept, proposal, final).
              const std::array finals{drafts_token(role) ? result.target_second : result.target_first};
              accepted = hyle::inference::mtp::select_sampled_prefix(verified_inputs,
                  std::span<const std::uint8_t>(flags).first(rows - 1), finals,
                  selection.output_quota, eos, static_cast<std::int32_t>(vocabulary), result.emitted);
            } else {
              const std::array targets{result.target_first, result.target_second};
              accepted = hyle::inference::mtp::select_greedy_prefix(verified_inputs,
                  std::span<const std::int32_t>(targets).first(rows), selection.output_quota, eos,
                  static_cast<std::int32_t>(vocabulary), result.emitted);
            }
            member.accepted_tokens = accepted.consumed;
            selected[j] = result.next_id = accepted.next_id;
            result.agreed = accepted.first_agrees; result.accepted_draft = accepted.consumed_drafts != 0;
            result.processed_target = accepted.consumed; result.processed_draft = draft_lag(role) + 1;
            result.proposal_count = drafts_token(role) ? 1 : 0;
            result.consumed_drafts = accepted.consumed_drafts; result.verified_rows = step.input_tokens;
            result.stopped = accepted.stopped;
            if (outputs[j].generation_ledger) {
              auto &record = outputs[j].generation_ledger->next();
              if (!record.generation || record.quota != selection.output_quota)
                throw std::runtime_error("MTP ledger quota differs from its original selection");
              record.draft = result.draft; record.target_first = result.target_first;
              record.target_second = result.target_second; record.agreed = result.agreed;
              record.accepted_draft = result.accepted_draft; record.stopped = result.stopped;
              record.consumed_drafts = result.consumed_drafts;
              record.emitted = {result.emitted[0], result.emitted.size() == 2 ? result.emitted[1] : -1};
              record.emitted_count = result.emitted.size();
            }
          }
        }, std::move(observe));
  }

};

// Entry IO runs synchronously inside Model's original completed-output scope.
// Neither this view nor its borrowed storage may escape or reenter preparation.
void observe_completed_evaluation(Model &model, const EvaluationOutput &output,
    const EvaluationIO &io, const CompletedEvaluationView &completed) {
  auto &observation = completed.observation;
  if (!output.observation) { observation.name = io.name; hyle::engine::reporting::write_step_observation(std::cerr, observation); }
  const auto dump_bytes = [&](mem::ConstBufferView view, const std::filesystem::path &path) {
    std::vector<std::byte> bytes(view.size()); check(model.execution.read(view, bytes));
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("cannot write Qwen trace: " + path.string());
  };
  if (io.trace && !io.trace->empty()) {
    const auto directory = *io.trace / io.name; std::filesystem::create_directories(directory);
    const auto dump_row = [&](c::ConstTensorView tensor, const char *file) {
      const auto bytes = take(tensor.description().required_bytes()) / completed.requests;
      dump_bytes(take(tensor.storage().subview(completed.member_index * bytes, bytes)), directory / file);
    };
    if (completed.logits.valid()) dump_row(completed.logits, "logits.f32");
    dump_row(completed.hidden, model.precision.is_f16() ? "hidden.f16" : "hidden.bf16");
    if (io.trace_states) {
      for (std::size_t i = 0; i < model.state_schema.size(); ++i) if (model.geometry.is_gdn(i)) {
        const auto prefix = "layer" + std::to_string(i);
        dump_bytes(take(completed.head.state[i].history.view()), directory / (prefix + (model.precision.is_f16() ? ".conv.f16" : ".conv.bf16")));
        dump_bytes(take(completed.head.state[i].recurrent.view()), directory / (prefix + ".gdn.f32"));
      }
      std::ofstream layout_trace(directory / "kv-layout.json");
      layout_trace << "{\"visible_tokens\":" << completed.head.length << ",\"page_rows\":" << cache::Plan::page_rows << ",\"pages\":[";
      for (std::size_t i = 0; i < completed.head.pages.size(); ++i) layout_trace << (i ? "," : "") << completed.head.pages[i].page();
      layout_trace << "]}\n"; if (!layout_trace) throw std::runtime_error("cannot write KV layout trace");
    }
  }
  if (output.capture_probes) {
    const auto directory = io.probe_subdirectory.empty() ? model.layer0_probe / io.name
        : model.layer0_probe / std::filesystem::path(io.probe_subdirectory) / io.name;
    std::filesystem::create_directories(directory);
    for (std::size_t i = 0; i < completed.probes.size(); ++i) {
      const auto view = completed.probes[i]; const auto bytes = take(view.description().required_bytes());
      if ((completed.requests != 1 && !view.description().is_dense()) || bytes % completed.requests)
        throw std::runtime_error("published batch diagnostic is not request-major dense");
      dump_bytes(take(view.storage().subview(completed.member_index * (bytes / completed.requests), bytes / completed.requests)),
          directory / (std::string(layer0_probe_names[i]) + (model.precision.is_f16() ? ".f16" : ".bf16")));
    }
  }
}

EvaluationOutput evaluation_output(Model &model, const std::shared_ptr<Session> &session,
    const EvaluationIO &io, double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    GenerationLedger *ledger = nullptr, const GenerationCommitRecord *expected = nullptr) {
  return {.session = session, .capture_tensors = io.trace && !io.trace->empty(),
      .capture_probes = !model.layer0_probe.empty(), .complete_seconds = complete_seconds,
      .host_endpoint = host_endpoint, .observation = observation, .preparation = preparation,
      .generation_ledger = ledger, .expected_episode = expected};
}

MtpRoundResult finish_mtp_round(Model &model, Session &session, std::int32_t expected_pending,
    std::size_t output_quota, std::span<const std::int32_t> eos = {},
    const EvaluationOutput *output = nullptr, const EvaluationIO &io = {.name = "mtp"}) {
  const std::array selections{MtpGenerationSelection{session.shared_from_this(), expected_pending, output_quota}};
  const std::array outputs{output ? *output : evaluation_output(model, session.shared_from_this(), io)};
  std::array<MtpRoundResult, 1> results;
  model.finish_mtp_rounds(selections, eos, outputs, results, [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  });
  return std::move(results[0]);
}

std::int32_t finish_evaluation(Model &model, Session &session, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {},
    GenerationLedger *generation_ledger = nullptr, const GenerationCommitRecord *expected_episode = nullptr) {
  const EvaluationIO io{name, &trace, trace_states, probe_subdirectory};
  const std::array outputs{evaluation_output(model, session.shared_from_this(), io,
      complete_seconds, host_endpoint, observation, preparation, generation_ledger, expected_episode)};
  std::int32_t next = -1;
  model.finish_evaluations(outputs, std::span(&next, 1), [&](std::size_t j, const CompletedEvaluationView &view) {
    observe_completed_evaluation(model, outputs[j], io, view);
  });
  return next;
}

std::int32_t evaluate(Model &model, Session &session, std::span<const std::int32_t> ids, const std::string &name,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    double *complete_seconds = nullptr, Clock::time_point *host_endpoint = nullptr,
    inference::StepObservation *observation = nullptr, inference::PreparationObservation *preparation = nullptr,
    std::string_view probe_subdirectory = {}, const inference::RequestStopPolicy *stop_policy = nullptr) {
  model.prepare_evaluation(session, ids, preparation, GraphRole::kText, TextCompletion::kEmitNext, stop_policy);
  try { model.submit_evaluation(session); return finish_evaluation(model, session, name, trace, trace_states,
      complete_seconds, host_endpoint, observation, preparation, probe_subdirectory); }
  catch (inference::EvaluationFailure &failure) {
    failure.observe_retirement(model.abort_episode());
    throw;
  } catch (...) { (void)model.abort_episode(); throw; }
}

hyle::Status Session::stop() noexcept {
  if (stopping) return retirement_status;
  stopping = true;
  if (pending) {
    pending->cancelled = true;
    if (pending->completion.valid()) {
      const auto cancelled = pending->completion.request_cancel();
      if (!cancelled.IsUnimplemented()) inference::record_retirement_status(retirement_status, cancelled);
    }
    if (pending->run) pending->run->close().IgnoreError();
  }
  return retirement_status;
}
inference::RetirementObservation Session::try_retire() noexcept {
  if (!stopping) return {false, hyle::Status::FailedPrecondition("stop the Qwen session before retirement")};
  inference::RetirementObservation observed{true, {}};
  if (!model_) observed = resources_->retire(false);
  else if (pending) observed = model_->try_retire_episode();
  inference::record_retirement_status(retirement_status, observed.status);
  if (observed.retired) {
    tails = {}; draft_tails = {}; tail_head = nullptr; draft_tail_head = nullptr; head.reset();
  }
  return {observed.retired, retirement_status};
}
inference::RetirementObservation Session::drain() noexcept {
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
Session::~Session() { (void)drain(); }

struct GenerationRestoreStats final {
  std::size_t graph_calls = 0, target_rows = 0, draft_rows = 0;
  std::size_t normalized_lag_rows = 0;
  struct DecisionChange { std::size_t episode; std::int32_t historical, fresh; };
  std::vector<DecisionChange> decision_changes;
  double step_seconds = 0, seconds = 0;
};

enum class GenerationMode { kText, kMtp };

struct GenerationRequest {
  std::string id, memory_name;
  std::span<const std::int32_t> prompt;
  std::filesystem::path trace;
  inference::RequestProgress progress;
  inference::RequestKey request_key{};
  std::shared_ptr<Session> session;
  cache::TokenPlanRef input;
  cache::Checkpoint completed_checkpoint;
  std::shared_ptr<cache::ReplayHistory> history;
  std::vector<MtpRoundResult> rounds;
  std::vector<std::vector<std::int32_t>> round_emitted;
  Clock::time_point prefill_start{}, prefill_endpoint{}, endpoint{};
  double prefill_step_seconds = 0, decode_step_sum_seconds = 0;
  std::size_t verified_rows = 0, draft_rows = 0, draft_rounds = 0, agreements = 0, accepts = 0;
  std::size_t visible_tokens = 0, draft_frontier = 0, draft_lag = 0;
  std::size_t reused_tokens = 0, prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0;
  std::size_t prefill_graph_calls = 0, prefill_graph_tokens = 0, prefill_draft_rows = 0;
  std::size_t prefix_sampling_calls = 0;
  std::size_t auxiliary_graph_calls = 0;
  double prefix_sampling_seconds = 0;
  const inference::RequestStopPolicy stop_policy;
  std::unique_ptr<GenerationLedger> ledger;
  struct TurnObservation {
    Clock::time_point start{}, prefill_endpoint{}, endpoint{};
    std::size_t output_begin = 0, output_end = 0;
    std::size_t context_before = 0, context_after = 0;
  };
  std::vector<TurnObservation> turn_observations;
  std::vector<std::shared_ptr<std::vector<std::int32_t>>> warm_inputs;
  std::size_t last_idle = 0;
  const std::size_t sampling_index;
  std::size_t input_turn = SIZE_MAX, turn_input_consumed = 0;
  Clock::time_point admission_start{};
  GenerationRestoreStats restored;
  std::size_t restore_count = 0, idle_evictions = 0;
  std::size_t checkpoint_resume_hits = 0, checkpoint_resume_misses = 0;
  std::int32_t fresh_next_id = -1;

  GenerationRequest(std::string name, std::span<const std::int32_t> ids,
                       std::filesystem::path output, std::size_t steps, std::size_t sampling = 0,
                       inference::RequestStopPolicy stop = {})
      : id(std::move(name)), memory_name(id.empty() ? "generation" : "generation_" + id),
        prompt(ids), trace(std::move(output)), progress(steps), stop_policy(std::move(stop)), sampling_index(sampling) {
    rounds.reserve(steps - 1);
  }
};

// Cold episode count; imported source chunks have their own explicit bound.
std::size_t generation_record_capacity(std::span<const inference::GenerationTurn> turns,
    std::size_t max_prefill_rows = SIZE_MAX, std::size_t imported_records = 0) {
  if (!max_prefill_rows) throw std::runtime_error("prefill quantum must be positive");
  auto count = imported_records;
  for (std::size_t turn = 0; turn < turns.size(); ++turn) {
    const auto rows = cache::add(turns[turn].prompt.size(), turn ? 1u : 0u);
    if (!rows || !turns[turn].max_outputs) throw std::runtime_error("generation requires nonempty turn input and positive quota");
    const auto chunks = rows / max_prefill_rows + (rows % max_prefill_rows != 0);
    count = cache::add(count, cache::add(chunks, turns[turn].max_outputs - 1));
  }
  return count;
}

GenerationCommitRecord &prepare_generation_record(GenerationLedger &ledger, const Session &session,
    GraphRole role, std::span<const std::int32_t> ids, bool generation,
    std::size_t quota, std::size_t turn, std::shared_ptr<const void> owner = {}, std::size_t maximum = 0,
    TextCompletion completion = TextCompletion::kEmitNext, std::size_t input_begin = 0, bool prefill_complete = true) {
  if (completion == TextCompletion::kStateOnly && (role != GraphRole::kText || generation || prefill_complete))
    throw std::runtime_error("state-only ledger entry requires an incomplete ordinary prefill");
  if (ids.empty() || !session.head || (generation && ids.size() != 1))
    throw std::runtime_error("MTP replay record requires its actual input and complete base");
  auto &record = ledger.next();
  ledger.bind_sampling(session.sampling_configuration(), session.sampling_owner());
  auto emitted = std::move(record.emitted), consumed = std::move(record.accepted_inputs);
  auto verified = std::move(record.verification_inputs);
  record = {}; record.emitted = std::move(emitted); record.accepted_inputs = std::move(consumed);
  record.verification_inputs = std::move(verified); record.verification_inputs.clear();
  record.emitted.assign({-1, -1}); record.accepted_inputs.assign({-1, -1});
  record.role = role; record.generation = generation; record.quota = quota; record.turn = turn;
  record.completion = completion; record.input_begin = input_begin; record.prefill_complete = prefill_complete;
  record.user_output = completion == TextCompletion::kEmitNext;
  record.draft_limit = role == GraphRole::kVerify ? maximum : drafts_token(role) ? 1 : 0;
  record.before = generation_boundary(*session.head);
  if (generation) record.pending_input = ids.front();
  else { record.input_owner = std::move(owner); record.prompt_ids = ids; }
  return record;
}

void preflight_generation_prompt_ledger(const GenerationLedger &ledger, const cache::ReplayPrefix &prefix) {
  if (ledger.committed_count || !prefix.history || !prefix.chunks ||
      prefix.chunks > prefix.history->chunks.size() || prefix.chunks > ledger.records.size())
    throw std::runtime_error("MTP initial prompt import requires an empty bounded ledger");
  std::size_t covered = 0;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    if (ledger.records[i].committed || ledger.records[i].emitted.capacity() < 2 ||
        !part.input || !part.count || part.first != covered || part.first > part.input->ids.size() ||
        part.count > part.input->ids.size() - part.first || covered > prefix.length ||
        part.count > prefix.length - covered || (part.state_only ? part.next_id != -1 : part.next_id < 0))
      throw std::runtime_error("prompt import requires complete original ranges and uncommitted reserved records");
    covered += part.count;
  }
  if (covered != prefix.length)
    throw std::runtime_error("prompt import history does not cover its complete boundary");
}

void import_generation_prompt_ledger(GenerationLedger &ledger, const cache::ReplayPrefix &prefix,
                              bool full_hit, bool draft_cache) {
  preflight_generation_prompt_ledger(ledger, prefix);
  std::int32_t previous = -1;
  for (std::size_t i = 0; i < prefix.chunks; ++i) {
    const auto &part = prefix.history->chunks[i];
    auto &record = ledger.records[i];
    record.role = !draft_cache ? GraphRole::kText
        : part.count == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
    record.input_owner = part.input;
    record.prompt_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
    record.before = {part.first, draft_cache ? part.first : 0, 0, previous, -1};
    record.after = {part.first + part.count, draft_cache ? part.first + part.count : 0, 0, part.next_id, -1};
    record.accepted = part.count; record.draft_rows = draft_cache ? part.count : 0;
    record.completion = part.state_only ? TextCompletion::kStateOnly : TextCompletion::kEmitNext;
    record.prefill_complete = !part.state_only;
    record.input_begin = part.first;
    record.target_first = part.next_id; record.emitted = {part.next_id, -1}; record.emitted_count = part.state_only ? 0 : 1;
    record.stopped = !draft_cache
        ? ledger.stop_policy().stopped(part.next_id, is_end_of_generation(part.next_id))
        : is_end_of_generation(part.next_id);
    record.user_output = full_hit && i + 1 == prefix.chunks;
    record.imported = true;
    record.requests = 0; // Role-only source history does not claim original batch geometry.
    previous = part.next_id;
  }
  // All ranges/capacities were checked before population. shared_ptr/span and
  // scalar writes, including the two inline emitted IDs, allocate nothing.
  for (std::size_t i = 0; i < prefix.chunks; ++i) ledger.records[i].committed = true;
  ledger.committed_count = prefix.chunks;
}

// Consume committed host facts even when a subsequent diagnostic write threw.
// The authoritative ledger was installed by the original head-swap transaction.
void observe_generation_commits(GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  while (request.progress.observed_commits() < request.ledger->committed_count) {
    const auto ordinal = request.progress.observed_commits();
    const auto &record = request.ledger->records[ordinal];
    if (record.emitted_count > record.emitted.size())
      throw std::runtime_error("committed output count exceeds its original record storage");
    const auto emitted = std::span(record.emitted).first(record.emitted_count);
    const bool prefill_complete = !record.generation && !record.imported && record.prefill_complete;
    if (!record.generation && !record.imported &&
        (record.turn != request.progress.turn_index || request.turn_input_consumed > request.prompt.size() || record.input_begin != request.turn_input_consumed ||
         record.ids().size() > request.prompt.size() - request.turn_input_consumed ||
         record.ids().data() != request.prompt.data() + request.turn_input_consumed ||
         record.prefill_complete != (record.ids().size() == request.prompt.size() - request.turn_input_consumed)))
      throw std::runtime_error("committed prefill input is not the next actual turn slice");
    const bool stopped = record.role == GraphRole::kText ? record.stopped : is_end_of_generation(record.after.next_id);
    if (!request.progress.preflight_commit(ordinal, emitted, record.user_output, prefill_complete, stopped))
      continue;
    if (record.user_output && record.generation &&
        (request.rounds.size() == request.rounds.capacity() || request.rounds.size() >= request.round_emitted.size() ||
         record.emitted_count > request.round_emitted[request.rounds.size()].capacity()))
      throw std::runtime_error("generation observer requires its original cold-reserved round storage");
    const auto output_end = request.progress.generated.size() + (record.user_output ? emitted.size() : 0);
    if (record.user_output) {
      if (record.generation) {
        MtpRoundResult result;
        result.pending = record.pending_input; result.draft = record.draft;
        result.target_first = record.target_first; result.target_second = record.target_second;
        result.next_id = record.after.next_id; result.agreed = record.agreed;
        result.accepted_draft = record.accepted_draft; result.stopped = record.stopped;
        result.processed_target = record.accepted; result.processed_draft = record.draft_rows;
        result.proposal_count = record.verified_rows ? record.verified_rows - 1 : 0;
        result.verified_rows = record.verified_rows; result.consumed_drafts = record.consumed_drafts;
        result.auxiliary_graph_calls = record.auxiliary_graph_calls;
        result.emitted = std::move(request.round_emitted[request.rounds.size()]);
        result.emitted.assign(record.emitted.begin(), record.emitted.begin() + record.emitted_count);
        request.rounds.push_back(std::move(result));
        request.verified_rows += record.verified_rows;
        request.draft_rows += record.draft_rows; request.draft_rounds += record.draft_limit ? 1 : 0;
        request.agreements += record.agreed ? 1 : 0; request.accepts += record.consumed_drafts;
        request.auxiliary_graph_calls += record.auxiliary_graph_calls;
        request.decode_step_sum_seconds += record.step_seconds;
      }
    }
    if (!record.generation && !record.imported) {
      ++request.prefill_graph_calls; request.prefill_graph_tokens += record.ids().size();
      request.prefill_draft_rows += record.draft_rows;
      request.prefill_step_seconds += record.step_seconds;
    }
    request.visible_tokens = record.after.length; request.draft_frontier = record.after.frontier;
    request.draft_lag = record.after.lag; request.fresh_next_id = record.after.next_id;
    if (!record.imported) request.endpoint = record.endpoint;
    if (prefill_complete) {
      request.prefill_endpoint = record.endpoint;
      if (!request.turn_observations.empty()) {
        auto &turn = request.turn_observations[request.progress.turn_index];
        turn.start = request.prefill_start; turn.prefill_endpoint = record.endpoint;
      }
    }
    if (!request.turn_observations.empty()) {
      auto &turn = request.turn_observations[request.progress.turn_index];
      turn.endpoint = request.endpoint; turn.output_end = output_end;
      turn.context_after = request.visible_tokens;
    }
    // No fallible family bookkeeping follows the one host observation mark.
    (void)request.progress.observe_commit(ordinal, emitted, record.user_output, prefill_complete, stopped);
    if (!record.generation && !record.imported) request.turn_input_consumed += record.ids().size();
  }
}

// Only the serialized request owners and this directory retain committed
// Snapshots here. Hit/local Snapshot references are dropped before this gate.
inference::VersionHolder<Session, cache::Snapshot> version_holder(const Session *session) noexcept {
  return {session, session && session->head ? session->head->state_identity() : nullptr};
}

void make_generation_room(Model &model, cache::PrefixCache *prefix, Session *owner,
                    std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const std::array selected{version_holder(owner)};
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
  });
  const auto fits = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation admission requires retired committed heads within N");
    return inference::version_admission({live, limit}, selected, residents,
        [&](const auto *head) { return prefix && prefix->retains(head); }).fits;
  };
  while (!fits()) if (!prefix || !prefix->evict_one())
    throw std::runtime_error("generation checkpoints retain all committed versions; no unfinished session may be evicted");
}

std::shared_ptr<Session> restore_mtp_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = model.sampling_index(ledger);
  const auto started = Clock::now();
  if (sampling_index || !model.cache_plan.draft_cache || !ledger.stop_policy().is_default())
    throw std::runtime_error("strict MTP replay requires the original complete target/draft model and default stop policy");
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP restore requires a complete zero-origin ledger and an idle carrier");
  for (std::size_t i = 0; i < ledger.committed_count; ++i)
    if (ledger.records[i].requests != 1 || ledger.records[i].member_index != 0)
      throw std::runtime_error("strict MTP replay requires recorded original R1 geometry; use committed-token rebuild for cache recovery");
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before restoring within N committed versions");
  const auto &first = ledger.records[0];
  const std::array calls{first.ids().size()};
  auto session = model.create_session(calls, {}, first.role, first.before.sampling);
  GenerationLedger actual(ledger.committed_count, model.cache_plan.candidate_boundaries);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  const std::array<std::int32_t, 2> eos{248044, 248046};
  try {
    for (std::size_t index = 0; index < ledger.committed_count; ++index) {
      const auto &expected = ledger.records[index];
      if (!expected.committed) throw std::runtime_error("MTP restore ledger contains an uncommitted gap");
      const auto pending = expected.pending_input;
      const auto ids = expected.generation ? std::span<const std::int32_t>(&pending, 1) : expected.ids();
      prepare_generation_record(actual, *session, expected.role, ids, expected.generation,
          expected.quota, expected.turn, expected.input_owner, expected.draft_limit);
      make_generation_room(model, prefix, session.get(), requests);
      std::vector<c::ReadLease> q;
      const auto verification = expected.role == GraphRole::kVerify
          ? model.propose_recursive(*session, expected.draft_limit, eos, &q) : std::vector<std::int32_t>{};
      model.prepare_evaluation(*session, expected.role == GraphRole::kVerify ? std::span<const std::int32_t>(verification) : ids,
          nullptr, expected.role); model.submit_evaluation(*session);
      if (stats) {
        ++stats->graph_calls;
        stats->target_rows = cache::add(stats->target_rows, expected.verified_rows);
        stats->draft_rows = cache::add(stats->draft_rows, expected.draft_rows);
      }
      const auto name = "episode" + std::to_string(index);
      double step_seconds = 0;
      if (expected.generation) {
        const EvaluationIO io{name, &trace, trace_states};
        const auto output = evaluation_output(model, session, io, &step_seconds,
            nullptr, nullptr, nullptr, &actual, &expected);
        if (expected.role == GraphRole::kVerify) {
          const std::array selections{MtpGenerationSelection{session, pending, expected.quota, q, expected.draft_limit}};
          const std::array outputs{output}; std::array<MtpRoundResult, 1> results;
          model.finish_recursive_mtp(selections, eos, outputs, results,
              [&](std::size_t j, const CompletedEvaluationView &view) { observe_completed_evaluation(model, outputs[j], io, view); });
        } else (void)qwen35::finish_mtp_round(model, *session, pending, expected.quota, eos, &output, io);
      } else {
        (void)qwen35::finish_evaluation(model, *session, name, trace, trace_states, &step_seconds,
            nullptr, nullptr, nullptr, {}, &actual, &expected);
      }
      if (stats) stats->step_seconds += step_seconds;
    }
  } catch (...) {
    session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// Cache reconstruction consumes immutable committed token chunks. It does not
// resample past outputs or claim to replay an old batch's natural decisions.
std::shared_ptr<Session> rebuild_generation_session(Model &model, const GenerationLedger &ledger,
    const std::filesystem::path &trace = {}, bool trace_states = true,
    GenerationRestoreStats *stats = nullptr, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  const auto sampling_index = model.sampling_index(ledger);
  ledger.stop_policy().validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !ledger.stop_policy().is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  const auto started = Clock::now();
  if (!ledger.committed_count || ledger.committed_count > ledger.records.size() ||
      ledger.records[0].generation || ledger.records[0].before.length || model.active_step || model.carrier_borrowed)
    throw std::runtime_error("MTP rebuild requires a complete zero-origin committed history and idle carrier");
  std::size_t committed_length = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i];
    if (!record.committed || !record.accepted || (record.generation &&
        (record.accepted > record.accepted_inputs.size() || record.accepted_inputs[0] != record.pending_input ||
          record.consumed_drafts + 1 != record.accepted ||
          (record.accepted > 1 && (!record.accepted_draft || record.accepted_inputs[1] != record.draft)))))
      throw std::runtime_error("MTP rebuild has an invalid committed input boundary");
    const auto ids = record.committed_ids();
    if (record.before.length != committed_length || ids.empty() || ids.size() != record.accepted ||
        record.accepted > model.capacity - committed_length)
      throw std::runtime_error("MTP rebuild committed input chunks are not a continuous bounded prefix");
    committed_length = cache::add(committed_length, record.accepted);
    if (record.after.length != committed_length) throw std::runtime_error("MTP rebuild committed length is inconsistent");
    for (auto id : ids) if (id < 0 || id >= model.vocabulary) throw std::runtime_error("MTP rebuild token is outside vocabulary");
  }
  make_generation_room(model, prefix, nullptr, requests);
  if (model.snapshot_versions.live >= model.cache_plan.resident_sessions)
    throw std::runtime_error("retire the old MTP head before rebuilding within N committed versions");
  if (stats) stats->decision_changes.reserve(cache::add(stats->decision_changes.size(), ledger.committed_count));
  const auto first_ids = ledger.records[0].committed_ids();
  const std::array calls{first_ids.size()};
  const auto role_for = [&](std::size_t rows) {
    return !model.cache_plan.draft_cache ? GraphRole::kText
        : rows == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
  };
  auto session = model.create_session(calls, {}, role_for(first_ids.size()),
      ledger.records[0].before.sampling, false, ledger.records[0].completion, sampling_index);
  GenerationLedger actual(ledger.committed_count);
  actual.bind_stop_policy(ledger.stop_policy(), model.vocabulary);
  std::size_t observed = 0;
  const auto observe = [&] {
    while (observed < actual.committed_count) {
      const auto &fresh = actual.records[observed]; const auto &old = ledger.records[observed];
      if (stats) {
        ++stats->graph_calls; stats->target_rows = cache::add(stats->target_rows, fresh.accepted);
        stats->draft_rows = cache::add(stats->draft_rows, fresh.draft_rows);
        stats->step_seconds += fresh.step_seconds;
        if (fresh.after.next_id != old.after.next_id)
          stats->decision_changes.push_back({observed, old.after.next_id, fresh.after.next_id});
      }
      ++observed;
    }
  };
  try {
    for (std::size_t i = 0; i < ledger.committed_count; ++i) {
      const auto &source = ledger.records[i]; const auto ids = source.committed_ids();
      const auto role = role_for(ids.size());
      prepare_generation_record(actual, *session, role, ids, false, 0, source.turn, source.input_owner, 0,
          source.completion, source.input_begin, source.prefill_complete);
      make_generation_room(model, prefix, session.get(), requests);
      model.prepare_evaluation(*session, ids, nullptr, role, source.completion, &actual.stop_policy()); model.submit_evaluation(*session);
      (void)qwen35::finish_evaluation(model, *session, "episode" + std::to_string(i), trace, trace_states,
          nullptr, nullptr, nullptr, nullptr, {}, &actual);
      observe();
      // Rebuilding committed tokens is not a fresh user sampling episode.
      // The historical pending token and cursor remain authoritative even if
      // a differently partitioned recomputation produces different logits.
      model.restore_pending_boundary(*session, source.after.sampling, source.after.next_id);
    }
    if (session->length() != ledger.records[ledger.committed_count - 1].after.length ||
        (session->head->draft && session->head->draft->lag))
      throw std::runtime_error("MTP committed-token rebuild has an inconsistent complete boundary");
    if (stats) stats->normalized_lag_rows = cache::add(stats->normalized_lag_rows, ledger.records[ledger.committed_count - 1].after.lag);
  } catch (...) {
    observe(); session->stop().IgnoreError(); (void)session->drain();
    (void)session->try_retire();
    if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
    throw;
  }
  if (stats) stats->seconds += std::chrono::duration<double>(Clock::now() - started).count();
  return session;
}

// A completed checkpoint is only a possible recovery shortcut. A newer
// mid-turn commit leaves that old complete key behind; compare it with the
// actual immutable ledger at this cold recovery boundary, without a new key.
bool checkpoint_covers_generation_ledger(const cache::Checkpoint &checkpoint,
                                         const GenerationLedger &ledger) noexcept {
  if (!checkpoint.input || !checkpoint.replay.history || !ledger.committed_count ||
      ledger.committed_count > ledger.records.size()) return false;
  const auto &last = ledger.records[ledger.committed_count - 1];
  if (last.after.next_id < 0 || checkpoint.input->ids.size() != last.after.length ||
      checkpoint.replay.length != last.after.length || checkpoint.replay.chunks != ledger.committed_count ||
      checkpoint.replay.chunks > checkpoint.replay.history->chunks.size() ||
      checkpoint.replay.history->chunks[checkpoint.replay.chunks - 1].next_id != last.after.next_id)
    return false;
  std::size_t offset = 0;
  for (std::size_t i = 0; i < ledger.committed_count; ++i) {
    const auto &record = ledger.records[i]; const auto ids = record.committed_ids();
    if (!record.committed || ids.empty() || ids.size() != record.accepted || record.before.length != offset ||
        offset > checkpoint.input->ids.size() || ids.size() > checkpoint.input->ids.size() - offset ||
        !std::equal(ids.begin(), ids.end(), checkpoint.input->ids.begin() + static_cast<std::ptrdiff_t>(offset)))
      return false;
    offset += ids.size();
    if (record.after.length != offset) return false;
  }
  return offset == last.after.length;
}

// Prove the whole remaining ordinary turn against its authoritative cursor.
// This is admission arithmetic only: no RNG reservation or mutable copy. A new
// prefix hit must start from this request, never from its cached publisher.
void preflight_generation_sampling(const Model &model, const GenerationRequest &request) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (model.cache_plan.draft_cache ||
      !model.sampling_configuration(request.sampling_index).configuration.enabled) return;
  const auto &progress = request.progress;
  std::size_t outputs = 0;
  if (progress.turn_started()) outputs = progress.remaining_outputs();
  else {
    if (progress.generated.size() > progress.maximum_outputs())
      throw std::runtime_error("request outputs exceed its prepared output limit");
    const auto turns = progress.turns();
    if (turns.empty()) outputs = progress.maximum_outputs() - progress.generated.size();
    else {
      if (progress.turn_index >= turns.size())
        throw std::runtime_error("request turn is outside its prepared schedule");
      outputs = turns[progress.turn_index].max_outputs;
      if (outputs > progress.maximum_outputs() - progress.generated.size())
        throw std::runtime_error("request turn exceeds its prepared output limit");
    }
  }
  if (!outputs) return;
  auto cursor = progress.initial_sampling;
  if (request.session) {
    if (!request.session->head || request.session->stopping || request.session->failed || request.session->pending)
      throw std::runtime_error("ordinary sampling admission requires a live idle Session head");
    cursor = request.session->head->sampling;
  }
  else if (request.ledger->committed_count) {
    if (request.ledger->committed_count > request.ledger->records.size())
      throw std::runtime_error("request recovery exceeds its original committed ledger");
    cursor = request.ledger->records[request.ledger->committed_count - 1].after.sampling;
  }
  // Includes the prefill/full-hit output; the last pending token needs no
  // additional draw. State-only chunks do not consume this checked budget.
  (void)cursor.advanced(outputs);
}

// Only restore the original Session. Request progress, turn input/offset,
// output delivery and its key remain with the serialized owner. In particular,
// this function never begins a turn or observes/emits historical output again.
void restore_generation_session(Model &model, GenerationRequest &request,
    cache::PrefixCache *prefix = nullptr, bool trace_states = true,
    std::span<const GenerationRequest> requests = {}) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !request.stop_policy.is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (request.session) {
    if (&request.session->sampling_configuration() != &prepared_sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    return;
  }
  if (!request.ledger->committed_count) return;
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  if (model.sampling_index(*request.ledger) != request.sampling_index)
    throw std::runtime_error("request recovery sampling differs from its committed ledger");
  preflight_generation_sampling(model, request);
  if (model.active_step || model.carrier_borrowed)
    throw std::runtime_error("request recovery requires the original idle execution carrier");
  const auto &ledger = *request.ledger;
  if (ledger.committed_count > ledger.records.size())
    throw std::runtime_error("request recovery exceeds its original committed ledger");
  const auto &last = ledger.records[ledger.committed_count - 1];
  std::shared_ptr<Session> restored;
  if (prefix && request.completed_checkpoint.input) {
    if (ledger.sampling == &prepared_sampling &&
        checkpoint_covers_generation_ledger(request.completed_checkpoint, ledger)) {
      const auto length = request.completed_checkpoint.input->ids.size();
      auto hit = prefix->lookup(*request.completed_checkpoint.input, length, length, &model, model.math_identity());
      if (hit) {
        // A peer may have published the same consumed key with another pending
        // or RNG cursor. Restore only those historical scalar facts; the fork
        // cannot repair an old physical length/frontier/lag/catchup boundary.
        auto head = model.sampling_fork(std::move(hit.payload.snapshot), last.after.sampling, last.after.next_id);
        hit.reset();
        if (generation_boundary(*head) == last.after &&
            bool(head->draft) == model.cache_plan.draft_cache) {
          restored = model.create_session({}, std::move(head),
              model.cache_plan.draft_cache ? GraphRole::kMtpSingle : GraphRole::kText,
              {}, false, TextCompletion::kEmitNext, request.sampling_index);
          ++request.checkpoint_resume_hits;
        }
      }
    }
    if (!restored) ++request.checkpoint_resume_misses;
  }
  if (!restored) {
    const auto trace = request.trace.empty() ? request.trace
        : request.trace / "restores" / ("restore" + std::to_string(request.restore_count));
    restored = rebuild_generation_session(model, ledger, trace, trace_states, &request.restored, prefix, requests);
    ++request.restore_count;
  }
  // All fallible recovery precedes installing the new original Session owner.
  request.session = std::move(restored);
}

struct PreparedGenerationCall final {
  GenerationRequest *request = nullptr;
  GraphRole role = GraphRole::kText;
  std::span<const std::int32_t> prompt;
  std::int32_t pending = -1;
  std::size_t quota = 0, steps = 0;
  std::size_t maximum = 0;
  std::vector<std::int32_t> verification_ids;
  std::vector<c::ReadLease> draft_log_probabilities;
  bool generation = false, full_hit = false, initial_prefix = false, prefill_complete = true;
  TextCompletion completion = TextCompletion::kEmitNext;
  std::size_t input_begin = 0;
  std::string name;
  double step_seconds = 0, proposal_seconds = 0;
  std::span<const std::int32_t> ids() const noexcept {
    if (generation && role == GraphRole::kVerify && !verification_ids.empty()) return verification_ids;
    return generation ? std::span(&pending, 1) : prompt;
  }
  std::size_t planned_rows() const noexcept { return role == GraphRole::kVerify ? maximum + 1 : drafts_token(role) ? 2 : ids().size(); }
};

PreparedGenerationCall prepare_generation_call(Model &model, GenerationRequest &request, std::size_t steps,
    cache::PrefixCache *prefix, std::span<const GenerationRequest> requests,
    std::size_t output_credit = SIZE_MAX, std::size_t max_prefill_rows = SIZE_MAX) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  request.stop_policy.validate(model.vocabulary);
  if (model.cache_plan.draft_cache && !request.stop_policy.is_default())
    throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  request.ledger->validate_stop_policy(request.stop_policy);
  const auto &prepared_sampling = model.sampling_configuration(request.sampling_index);
  if (model.cache_plan.draft_cache && request.sampling_index)
    throw std::runtime_error("nondefault request sampling is ordinary Text only");
  if (request.session && &request.session->sampling_configuration() != &prepared_sampling)
    throw std::runtime_error("request sampling cannot change after Session admission");
  request.ledger->validate_sampling(prepared_sampling, model.resources_);
  const auto &sampler = prepared_sampling.configuration;
  if (!max_prefill_rows || (model.cache_plan.draft_cache && max_prefill_rows != SIZE_MAX))
    throw std::runtime_error("bounded prefill rows require a positive ordinary Text quantum");
  // Credit belongs to this episode, not the turn's immutable output quota.
  // Reject before prefix lookup, session creation or any other admission work.
  if (!output_credit) throw std::runtime_error("generation must wait for output credit before preparation");
  if (steps != request.progress.output_limit())
    throw std::runtime_error("generation call quota differs from its prepared request turn");
  preflight_generation_sampling(model, request);
  if (!request.progress.turn_started()) (void)request.progress.begin_turn();
  if (request.progress.turn_done) throw std::runtime_error("MTP request has already completed");
  if (request.round_emitted.size() < steps - 1)
    throw std::runtime_error("MTP observer slots must be reserved for every turn before execution");
  request.ledger->bind_stop_policy(request.stop_policy, model.vocabulary);
  PreparedGenerationCall call; call.request = &request; call.steps = steps;
  call.generation = request.progress.prefilled;
  if (!call.generation) {
    if (request.input_turn != request.progress.turn_index) {
      const auto before = request.session ? request.session->length() : 0;
      if (before > model.capacity || request.prompt.empty() || request.prompt.size() > model.capacity - before)
        throw std::runtime_error("complete turn input exceeds the remaining context");
      // The turn's final output remains pending. Admit its full maximum journey
      // before any input chunk, prefix import or Session is published.
      const auto outputs = request.progress.remaining_outputs();
      if (!outputs || outputs - 1 > model.capacity - before - request.prompt.size())
        throw std::runtime_error("complete turn input and output quota exceed the remaining context");
      for (auto id : request.prompt) if (id < 0 || id >= model.vocabulary)
        throw std::runtime_error("token ID outside vocabulary");
      request.input_turn = request.progress.turn_index; request.turn_input_consumed = 0;
    }
    request.ledger->validate_sampling(prepared_sampling, model.resources_);
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_begin\n";
    if (!request.turn_input_consumed) request.prefill_start = request.admission_start == Clock::time_point{} ? Clock::now()
        : std::exchange(request.admission_start, Clock::time_point{});
    call.initial_prefix = prefix && request.progress.turn_index == 0;
    if (call.initial_prefix && !request.session) {
      auto hit = prefix->lookup(*request.input, request.prompt.size(), 1, &model, model.math_identity());
      if (hit) {
        if (hit.payload.snapshot->draft.has_value() != model.cache_plan.draft_cache)
          throw std::runtime_error("prefix directory contains a different target/draft state contract");
        const auto cached_lag = hit.payload.snapshot->draft ? hit.payload.snapshot->draft->lag : 0;
        const auto reused_tokens = hit.payload.snapshot->length;
        const bool full_hit = reused_tokens == request.prompt.size();
        auto history = cache::copy_history(hit.payload.replay, model.capacity);
        preflight_generation_prompt_ledger(*request.ledger, hit.payload.replay);
        request.progress.preflight_adopt_committed_history();
        const auto suffix = request.prompt.size() - reused_tokens;
        const auto suffix_rows = (std::min)(suffix, max_prefill_rows);
        const auto suffix_completion = suffix > suffix_rows ? TextCompletion::kStateOnly : TextCompletion::kEmitNext;
        std::string hit_name = "prefill";
        if (suffix > suffix_rows) hit_name += "_chunk" + std::to_string(reused_tokens);
        const std::array chunks{suffix_rows};
        const auto prior_sampling_seconds = model.frontier_sampling_seconds;
        const auto suffix_role = !model.cache_plan.draft_cache ? GraphRole::kText : (suffix == 1 || request.prompt.size() == 1)
            ? (cached_lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
            : (cached_lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
        // Keep every new request fact local until the actual Session/frontier
        // succeeds. Failed native/resource admission must permit the same retry.
        const auto admission_role = !model.cache_plan.draft_cache ? GraphRole::kText
            : suffix ? suffix_role : request.prompt.size() == 1 ? GraphRole::kMtpSingle : GraphRole::kMtpPrefill;
        auto adopted = model.create_session(suffix ? std::span<const std::size_t>(chunks) : std::span<const std::size_t>{},
            hit.payload.snapshot, admission_role,
            request.progress.initial_sampling, suffix == 0, suffix_completion, request.sampling_index);
        if (!suffix) history->chunks.back().next_id = adopted->head->next_id;
        // Only original reserved metadata remains. Preconditions are unchanged
        // since preflight; no allocation or native work follows this commit.
        import_generation_prompt_ledger(*request.ledger, hit.payload.replay, full_hit, model.cache_plan.draft_cache);
        if (sampler.enabled || model.sampling_configuration_count() > 1) {
          for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
            auto &record = request.ledger->records[i];
            record.before.sampling = sampler.enabled ? request.progress.initial_sampling : cache::SamplingCursor{};
            record.after.sampling = record.before.sampling;
            record.emitted_count = 0; record.emitted = {-1, -1};
          }
          request.ledger->records[request.ledger->committed_count - 1].after.sampling = adopted->head->sampling;
        }
        if (!suffix) {
          auto &last = request.ledger->records[request.ledger->committed_count - 1];
          last.after.sampling = adopted->head->sampling;
          last.after.next_id = adopted->head->next_id;
          last.target_first = last.after.next_id; last.emitted = {last.after.next_id, -1};
          last.emitted_count = 1; last.stopped = !model.cache_plan.draft_cache
              ? request.stop_policy.stopped(last.after.next_id, is_end_of_generation(last.after.next_id))
              : is_end_of_generation(last.after.next_id);
        }
        request.ledger->bind_sampling(prepared_sampling, model.resources_);
        request.progress.adopt_committed_history(request.ledger->committed_count);
        request.history = std::move(history);
        request.session = std::move(adopted);
        request.reused_tokens = reused_tokens; request.prefix_hits = 1;
        request.prefix_full_hits = full_hit ? 1 : 0;
        request.turn_input_consumed = reused_tokens;
        call.name = std::move(hit_name);
        if (sampler.enabled || model.sampling_configuration_count() > 1) {
          request.prefix_sampling_calls = suffix == 0 ? 1 : 0;
          request.prefix_sampling_seconds = model.frontier_sampling_seconds - prior_sampling_seconds;
        }
        hit.reset();
      } else { request.prefix_misses = 1; request.history = std::make_shared<cache::ReplayHistory>(model.capacity); }
    }
    call.input_begin = request.turn_input_consumed;
    const auto remaining_input = request.prompt.size() - call.input_begin;
    const auto rows = (std::min)(remaining_input, max_prefill_rows);
    call.prefill_complete = rows == remaining_input;
    call.completion = call.prefill_complete ? TextCompletion::kEmitNext : TextCompletion::kStateOnly;
    call.prompt = request.prompt.subspan(call.input_begin, rows);
    const auto lag = request.session && request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : call.prompt.size() == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
    if (!request.session) {
      make_generation_room(model, prefix, nullptr, requests);
      const std::array chunks{call.prompt.size()}; request.session = model.create_session(chunks, {}, call.role, request.progress.initial_sampling, false, call.completion, request.sampling_index);
    }
    if (!request.ledger->sampling) request.ledger->bind_sampling(prepared_sampling, model.resources_);
    if (!request.turn_observations.empty() &&
        call.input_begin == (request.progress.turn_index ? 0 : request.reused_tokens))
      request.turn_observations[request.progress.turn_index].context_before = request.session->length();
    call.full_hit = call.prompt.empty();
    if (call.name.empty()) {
      call.name = request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_prefill" : "prefill";
      if (!call.prefill_complete || call.input_begin > request.reused_tokens)
        call.name += "_chunk" + std::to_string(call.input_begin);
    }
  } else {
    call.quota = (std::min)(request.progress.remaining_outputs(), output_credit);
    call.pending = request.session->head->next_id;
    if (request.progress.generated.empty() || call.pending != request.progress.generated.back())
      throw std::runtime_error("generation pending differs from its last observed output");
    const auto remaining = model.capacity - request.session->length();
    if (!remaining) throw std::runtime_error("MTP has exhausted the configured context capacity");
    request.ledger->bind_sampling(prepared_sampling, model.resources_);
    call.maximum = model.cache_plan.draft_cache
        ? (std::min)({model.max_drafts(), call.quota - 1, remaining - 1}) : 0;
    const bool one = call.maximum == 0;
    const auto lag = request.session->head->draft ? request.session->head->draft->lag : 0;
    call.role = !model.cache_plan.draft_cache ? GraphRole::kText
        : one ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : !model.recursive_mtp() ? (lag ? GraphRole::kMtpCatchupRound : GraphRole::kMtpRound) : GraphRole::kVerify;
    call.name = (request.progress.turn_index ? "round" + std::to_string(request.progress.turn_index + 1) + "_" : "") +
        (model.cache_plan.draft_cache ? "mtp" : "decode") + std::to_string(request.rounds.size() + 1);
  }
  return call;
}

bool make_generation_batch_room(Model &model, cache::PrefixCache *prefix,
    std::span<const PreparedGenerationCall> calls, std::span<const GenerationRequest> requests) {
  const auto limit = model.cache_plan.resident_sessions;
  const auto selected = calls | std::views::transform([](const auto &call) {
    return version_holder(call.request->session.get());
  });
  const auto residents = requests | std::views::transform([](const auto &request) {
    return version_holder(request.session.get());
  });
  const auto fits = [&] {
    const auto live = std::size_t(model.snapshot_versions.live);
    if (live > limit || model.active_step) throw std::runtime_error("generation batch admission requires retired heads within N");
    return inference::version_admission({live, limit}, selected, residents,
        [&](const auto *head) { return prefix && prefix->retains(head); }).fits;
  };
  while (!fits()) if (!prefix || !prefix->evict_one()) return false;
  return true;
}

void finish_generation_call(Model &model, PreparedGenerationCall &call) {
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (call.full_hit && request.progress.prefilled) return; // A committed cache hit is observed once.
  if (call.full_hit) {
    const auto next_id = request.session->head->next_id;
    const auto emitted = std::span<const std::int32_t>(&next_id, 1);
    const bool stopped = call.role == GraphRole::kText
        ? request.stop_policy.stopped(next_id, is_end_of_generation(next_id)) : is_end_of_generation(next_id);
    request.progress.preflight_outputs(emitted, true, true, stopped);
    // The imported request ledger already adopted source history. The new
    // frontier output is observed directly, without inventing a model ordinal.
    request.progress.apply_outputs(emitted, true, true, stopped);
    request.prefill_endpoint = Clock::now(); request.endpoint = request.prefill_endpoint;
  } else observe_generation_commits(request);
  if (!call.generation && call.prefill_complete) {
    request.endpoint = request.prefill_endpoint;
  }
  request.visible_tokens = request.session->length();
  request.draft_frontier = request.session->head->draft ? request.session->head->draft->frontier : 0;
  request.draft_lag = request.session->head->draft ? request.session->head->draft->lag : 0;
  request.fresh_next_id = request.session->head->next_id;
  if (!request.turn_observations.empty()) {
    auto &turn = request.turn_observations[request.progress.turn_index];
    if (!call.generation && call.prefill_complete) { turn.start = request.prefill_start; turn.prefill_endpoint = request.prefill_endpoint; }
    turn.endpoint = request.endpoint;
    turn.output_end = request.progress.generated.size();
    turn.context_after = request.visible_tokens;
  }
  if (call.full_hit)
    std::cerr << "prefix_hit name=prefill reused_tokens=" << request.reused_tokens
              << " prompt_graph_calls=0 frontier_sampling_graph_calls=" << request.prefix_sampling_calls
              << " visible=" << request.session->length() << '\n';
  if (!call.generation && call.prefill_complete)
    std::cerr << "memory_phase=" << request.memory_name << "_prefill_end\n"
              << "memory_phase=" << request.memory_name << "_decode_begin\n";
  if (request.progress.turn_done) std::cerr << "memory_phase=" << request.memory_name << "_decode_end\n";
  if (call.full_hit) {
    if (!request.trace.empty()) {
      const auto path = request.trace / "prefill"; std::filesystem::create_directories(path);
      std::ofstream metadata(path / "cache-hit.json");
      if (!metadata) throw std::runtime_error("cannot write MTP prefix-hit metadata");
      metadata << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << request.prefix_sampling_calls
               << ",\"graph_calls\":" << request.prefix_sampling_calls << ",\"tensor_payload_available\":false,\"reused_tokens\":"
               << request.reused_tokens << ",\"next_id\":" << request.progress.generated.back() << "}\n";
    }
  }
  (void)model;
}

void execute_generation_calls(Model &model, std::span<PreparedGenerationCall> calls, bool trace_states,
    cache::PrefixCache *prefix = nullptr, std::span<const GenerationRequest> requests = {}) {
  if (calls.empty()) return;
  // Validate every request's cold metadata before prefix eviction, proposal
  // work or preparing even the first uncommitted ledger slot.
  for (std::size_t member = 0; member < calls.size(); ++member) {
    const auto &call = calls[member];
    if (!call.request || !call.request->ledger)
      throw std::runtime_error("generation request requires its cold-prepared commit ledger");
    const auto &request = *call.request;
    if (!request.session || !request.session->head || request.session->stopping ||
        request.session->failed || request.session->pending)
      throw std::runtime_error("generation batch requires live idle Session heads");
    for (const auto &prior : calls.first(member))
      if (prior.request->session.get() == request.session.get())
        throw std::runtime_error("generation batch requires distinct Session owners");
    const auto &sampling = model.sampling_configuration(request.sampling_index);
    if (&request.session->sampling_configuration() != &sampling)
      throw std::runtime_error("request sampling cannot change after Session admission");
    request.ledger->validate_sampling(sampling, model.resources_);
    if (request.ledger->stop_policy() != request.stop_policy)
      throw std::runtime_error("generation ledger stop policy differs from its admitted episode");
    if ((call.full_hit && calls.size() != 1) ||
        call.request->sampling_index != calls[0].request->sampling_index || call.role != calls[0].role ||
        call.completion != calls[0].completion || call.generation != calls[0].generation ||
        (call.role != GraphRole::kVerify && call.ids().size() != calls[0].ids().size()))
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    if (!call.full_hit) {
      if (call.completion == TextCompletion::kStateOnly &&
          (call.role != GraphRole::kText || call.generation || call.prefill_complete))
        throw std::runtime_error("state-only ledger entry requires an incomplete ordinary prefill");
      if (call.ids().empty() || (call.role == GraphRole::kVerify && !call.generation))
        throw std::runtime_error("generation call requires its actual prepared input");
      (void)request.ledger->next();
      if (!call.generation) (void)request.warm_inputs.at(request.progress.turn_index);
    }
  }
  if (calls.size() == 1 && calls[0].full_hit) { finish_generation_call(model, calls[0]); return; }
  if (!make_generation_batch_room(model, prefix, calls, requests)) throw std::runtime_error("MTP batch exceeds committed head budget");
  const std::array<std::int32_t, 2> eos{248044, 248046};
  if (calls[0].role == GraphRole::kVerify) {
    for (auto &call : calls) {
      if (!call.generation || call.role != GraphRole::kVerify)
        throw std::runtime_error("recursive MTP batch requires generation verifiers only");
      if (call.verification_ids.empty()) {
        const auto begin = Clock::now();
        call.verification_ids = model.propose_recursive(*call.request->session, call.maximum, eos, &call.draft_log_probabilities);
        call.proposal_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      }
    }
    // EOS can shorten different requests to different exact widths. Preserve
    // real query geometry rather than padding or consuming a terminal token.
    std::map<std::size_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < calls.size(); ++i) groups[calls[i].ids().size()].push_back(i);
    if (groups.size() > 1) {
      for (const auto &[rows, members] : groups) {
        (void)rows;
        std::vector<PreparedGenerationCall> selected;
        for (const auto index : members) selected.push_back(std::move(calls[index]));
        execute_generation_calls(model, selected, trace_states, prefix, requests);
        for (std::size_t j = 0; j < members.size(); ++j) calls[members[j]] = std::move(selected[j]);
      }
      return;
    }
  }
  const auto role = calls[0].role; const auto generation = calls[0].generation;
  hyle::container::SmallVector<EvaluationRequest, 2> inputs;
  hyle::container::SmallVector<EvaluationOutput, 2> outputs;
  hyle::container::SmallVector<MtpGenerationSelection, 2> selections;
  hyle::container::SmallVector<MtpRoundResult, 2> results;
  hyle::container::SmallVector<std::int32_t, 2> next;
  if (!inputs.try_reserve(calls.size()) || !outputs.try_reserve(calls.size()) ||
      !selections.try_reserve(calls.size()) || !results.try_resize(generation && role != GraphRole::kText ? calls.size() : 0) || !next.try_resize(calls.size()))
    throw std::bad_alloc{};
  for (auto &call : calls) {
    auto &request = *call.request;
    // Recursive proposals may have shortened since the initial metadata pass.
    if (role == GraphRole::kVerify && call.ids().size() != calls[0].ids().size())
      throw std::runtime_error("MTP batch requires equal role, kind and input rows without padding");
    prepare_generation_record(*request.ledger, *request.session, role,
        generation ? std::span<const std::int32_t>(&call.pending, 1) : call.ids(), generation,
        call.quota, request.progress.turn_index, generation ? std::shared_ptr<const void>{} : std::shared_ptr<const void>(request.warm_inputs.at(request.progress.turn_index)), call.maximum, call.completion, call.input_begin, call.prefill_complete);
    inputs.push_back({request.session, call.ids(), &request.ledger->stop_policy()});
    outputs.push_back(evaluation_output(model, request.session, {call.name, &request.trace, trace_states},
        &call.step_seconds, generation ? &request.endpoint : &request.prefill_endpoint,
        nullptr, nullptr, request.ledger.get()));
    if (generation && role != GraphRole::kText) selections.push_back({request.session, call.pending, call.quota, call.draft_log_probabilities, call.maximum});
  }
  const auto observe = [&](std::size_t j, const CompletedEvaluationView &view) {
    const auto &call = calls[j];
    observe_completed_evaluation(model, outputs[j], {call.name, &call.request->trace, trace_states}, view);
  };
  model.prepare_evaluations(std::span(inputs.data(), inputs.size()), nullptr, role, calls[0].completion); model.submit_evaluations();
  try {
    if (generation && role == GraphRole::kVerify) model.finish_recursive_mtp(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else if (generation && role != GraphRole::kText) model.finish_mtp_rounds(std::span(selections.data(), selections.size()), eos,
        std::span(outputs.data(), outputs.size()), std::span(results.data(), results.size()), observe);
    else model.finish_evaluations(std::span(outputs.data(), outputs.size()), std::span(next.data(), next.size()), observe);
    for (std::size_t j = 0; j < calls.size(); ++j) {
      calls[j].step_seconds += calls[j].proposal_seconds;
      if (calls[j].proposal_seconds) {
        auto &ledger = *calls[j].request->ledger;
        ledger.records[ledger.committed_count - 1].step_seconds += calls[j].proposal_seconds;
      }
      finish_generation_call(model, calls[j]);
    }
  } catch (...) {
    for (auto &call : calls) observe_generation_commits(*call.request);
    throw;
  }
}

void publish_generation_prompt(Model &model, PreparedGenerationCall &call, cache::PrefixCache *prefix) {
  if (!call.initial_prefix || call.full_hit || !call.prefill_complete) return;
  auto &request = *call.request;
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  for (const auto &record : std::span(request.ledger->records).first(request.ledger->committed_count)) {
    if (record.imported || record.generation || record.turn != 0) continue;
    if (record.input_begin != request.history->length)
      throw std::runtime_error("complete prompt is missing its actual committed prefill chunk");
    request.history->append(request.input, record.input_begin, record.accepted, record.after.next_id,
        record.completion == TextCompletion::kStateOnly);
  }
  if (request.history->length != request.prompt.size())
    throw std::runtime_error("committed prefill chunks do not cover the complete prompt");
  if (request.session->head->draft && request.session->head->draft->lag)
    throw std::runtime_error("only a complete lag0 MTP prompt may enter the directory");
  prefix->publish(request.input, request.session->head, {request.history, request.history->chunks.size(), request.prompt.size()});
  (void)model;
}

void publish_generation_completed(Model &model, GenerationRequest &request, cache::PrefixCache *prefix) {
  if (!request.ledger) throw std::runtime_error("generation request requires its cold-prepared commit ledger");
  if (!prefix || !request.progress.turn_done || !request.session || !request.session->head)
    return;
  std::vector<cache::ConsumedChunk> chunks;
  chunks.reserve(request.ledger->committed_count);
  for (std::size_t i = 0; i < request.ledger->committed_count; ++i) {
    const auto &record = request.ledger->records[i];
    if (!record.committed) throw std::runtime_error("completed checkpoint has a gap in its original committed ledger");
    chunks.push_back({record.committed_ids(), record.after.next_id, record.completion == TextCompletion::kStateOnly});
  }
  request.completed_checkpoint = cache::consumed_checkpoint(chunks, *request.session->head, model.capacity);
  prefix->publish(request.completed_checkpoint.input, request.session->head, request.completed_checkpoint.replay);
}

void advance_generation_request(Model &model, GenerationRequest &request,
    std::size_t steps, bool trace_states, cache::PrefixCache *prefix = nullptr,
    std::span<const GenerationRequest> requests = {}) {
  std::array calls{prepare_generation_call(model, request, steps, prefix, requests)};
  execute_generation_calls(model, calls, trace_states, prefix, requests);
  publish_generation_prompt(model, calls[0], prefix);
}

inference::RetirementObservation retire_generation_request(GenerationRequest &request) {
  // Stop/observe even on error; the runtime owner returns payload/key capacity
  // only after proof, before reporting the original observation status.
  (void)request.session->stop();
  return request.session->try_retire();
}

void print_generation_rebuild(const GenerationRestoreStats &stats, bool draft_cache) {
  std::cout << ",\"restore_contract\":\"recompute_committed_inputs\",\"restored_graph_calls\":" << stats.graph_calls
      << ",\"restored_target_rows\":" << stats.target_rows;
  if (draft_cache) std::cout << ",\"restored_draft_rows\":" << stats.draft_rows
      << ",\"normalized_lag_rows\":" << stats.normalized_lag_rows;
  std::cout << ",\"decision_changes\":[";
  for (std::size_t i = 0; i < stats.decision_changes.size(); ++i) {
    const auto &change = stats.decision_changes[i];
    std::cout << (i ? "," : "") << "{\"episode\":" << change.episode << ",\"historical_next\":" << change.historical
        << ",\"fresh_next\":" << change.fresh << '}';
  }
  std::cout << ']';
}

void write_request_sampling(std::ostream &out, const Model &model, const GenerationRequest &request) {
  if (model.sampling_configuration_count() == 1) return;
  const auto &configuration = model.sampling_configuration(request.sampling_index).configuration;
  out << ",\"sampling_configuration_index\":" << request.sampling_index
      << ",\"request_sampling\":\"" << (configuration.enabled ? "random" : "greedy") << '\"'
      << ",\"request_temperature\":" << configuration.policy.temperature
      << ",\"request_top_k\":" << configuration.policy.top_k << ",\"request_top_p\":" << configuration.policy.top_p;
}

void print_generation_request(const Model &model, const GenerationRequest &request, std::size_t steps,
                       Clock::time_point startup, Clock::time_point ready) {
  const auto &prompt = request.prompt;
  const auto &generated = request.progress.generated;
  const auto &rounds = request.rounds;
  const auto prefill_start = request.turn_observations.empty() ? request.prefill_start : request.turn_observations.front().start;
  const auto prefill_endpoint = request.turn_observations.empty() ? request.prefill_endpoint : request.turn_observations.front().prefill_endpoint;
  const auto endpoint = request.endpoint;
  const auto prefill_step_seconds = request.prefill_step_seconds,
      decode_step_sum_seconds = request.decode_step_sum_seconds;
  const auto verified_rows = request.verified_rows, draft_rows = request.draft_rows,
      draft_rounds = request.draft_rounds, agreements = request.agreements, accepts = request.accepts;
  const auto &trace = request.trace;
  // The original finish endpoint follows publication and host token visibility,
  // before its trailing trace/log output. Inter-call work remains in elapsed.
  auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
  auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
  std::size_t logical_prompt = prompt.size(), turn_count = 1;
  if (!request.turn_observations.empty()) {
    prefill_seconds = 0; decode_seconds = 0; logical_prompt = request.progress.turns().size() - 1; turn_count = request.progress.turns().size();
    for (std::size_t i = 0; i < request.progress.turns().size(); ++i) {
      logical_prompt = cache::add(logical_prompt, request.progress.turns()[i].prompt.size());
      const auto &turn = request.turn_observations[i];
      prefill_seconds += std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count();
      decode_seconds += std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count();
    }
  }
  const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
  std::cout << std::setprecision(17)
            << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
            << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << model.weight_inventory_report()
            << model.cache_report() << ",\"visible_tokens\":" << request.visible_tokens;
  write_request_sampling(std::cout, model, request);
  if (model.cache_plan.draft_cache) std::cout << ",\"draft_frontier\":" << request.draft_frontier
      << ",\"draft_lag\":" << request.draft_lag;
  std::cout << ",\"generated_ids\":[";
  for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
  std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
            << ",\"session_seconds\":" << session_seconds
            << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
            << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
            << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
            << ",\"prefill_step_seconds\":" << prefill_step_seconds + request.prefix_sampling_seconds
            << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
            << ",\"step_sum_seconds\":" << prefill_step_seconds + decode_step_sum_seconds + request.restored.step_seconds + request.prefix_sampling_seconds
            << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
            << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated.size() - turn_count) / decode_seconds : 0)
            << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
            << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
            << ",\"generated_decode_tokens\":" << generated.size() - turn_count << ",\"verified_target_rows\":" << verified_rows
            << (request.turn_observations.empty() ? ",\"actual_graph_calls\":" : ",\"graph_participations\":")
            << rounds.size() + request.prefill_graph_calls + request.restored.graph_calls + request.prefix_sampling_calls + request.auxiliary_graph_calls
            << ",\"prefix_sampling_graph_calls\":" << request.prefix_sampling_calls
            << ",\"sampling_stream\":" << request.progress.initial_sampling.stream
            << ",\"sampling_draw_cursor\":" << (request.ledger && request.ledger->committed_count
                ? request.ledger->records[request.ledger->committed_count - 1].after.sampling.draw
                : request.session ? request.session->head->sampling.draw : 0)
            << ",\"prefill_graph_calls\":" << request.prefill_graph_calls << ",\"decode_graph_calls\":" << rounds.size()
            << ",\"prefill_target_rows\":" << request.prefill_graph_tokens
            << ",\"actual_target_rows\":" << request.prefill_graph_tokens + verified_rows + request.restored.target_rows
            << ",\"committed_target_rows\":" << request.visible_tokens
            << ",\"committed_target_decode_rows\":" << request.visible_tokens - logical_prompt;
  if (model.cache_plan.draft_cache) std::cout << ",\"prefill_draft_rows\":" << request.prefill_draft_rows
      << ",\"actual_draft_rows\":" << request.prefill_draft_rows + draft_rows + request.restored.draft_rows
      << ",\"draft_rows\":" << draft_rows << ",\"draft_rounds\":" << draft_rounds
      << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts;
  std::cout << ",\"diagnostic\":" << (trace.empty() ? "false" : "true");
  if (!request.turn_observations.empty()) {
    std::cout << ",\"emitted_pending\":" << generated.back() << ",\"fresh_next_id\":" << request.fresh_next_id;
    print_generation_rebuild(request.restored, model.cache_plan.draft_cache);
  }
  if (!request.id.empty())
    std::cout << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"reused_tokens\":" << request.reused_tokens
              << ",\"prefix_hits\":" << request.prefix_hits << ",\"prefix_misses\":" << request.prefix_misses
              << ",\"prefix_full_hits\":" << request.prefix_full_hits;
  if (!request.turn_observations.empty()) {
    std::cout << ",\"between_turn_seconds\":" << (std::max)(0.0, session_seconds - prefill_seconds - decode_seconds)
              << ",\"phase_timing\":\"sum_per_turn_latency_including_interleaving\",\"turns\":[";
    for (std::size_t i = 0; i < request.turn_observations.size(); ++i) {
      const auto &turn = request.turn_observations[i];
      std::cout << (i ? "," : "") << "{\"turn\":" << i + 1 << ",\"requested_steps\":" << request.progress.turns()[i].max_outputs
          << ",\"new_user_tokens\":" << request.progress.turns()[i].prompt.size() << ",\"pending_flushed\":" << (i ? 1 : 0)
          << ",\"context_before\":" << turn.context_before << ",\"context_after\":" << turn.context_after
          << ",\"output_begin\":" << turn.output_begin << ",\"output_end\":" << turn.output_end
          << ",\"prefill_seconds\":" << std::chrono::duration<double>(turn.prefill_endpoint - turn.start).count()
          << ",\"decode_seconds\":" << std::chrono::duration<double>(turn.endpoint - turn.prefill_endpoint).count()
          << ",\"input_ids\":[";
      for (std::size_t k = 0; k < request.warm_inputs[i]->size(); ++k)
        std::cout << (k ? "," : "") << (*request.warm_inputs[i])[k];
      std::cout << "]}";
    }
    std::cout << ']';
  }
  std::cout << ",\"rounds\":[";
  for (std::size_t i = 0; i < rounds.size(); ++i) {
    const auto &row = rounds[i];
    std::cout << (i ? "," : "") << "{\"pending\":" << row.pending;
    if (!model.cache_plan.draft_cache) {
      std::cout << ",\"next_id\":" << row.next_id << ",\"committed_target_rows\":" << row.processed_target << '}';
      continue;
    }
    std::cout << ",\"draft\":" << row.draft
              << ",\"target_first\":" << row.target_first << ",\"target_second\":" << row.target_second
              << ",\"accepted_draft\":" << (row.accepted_draft ? "true" : "false")
              << ",\"proposals\":" << row.proposal_count << ",\"verified_rows\":" << row.verified_rows
              << ",\"consumed_drafts\":" << row.consumed_drafts << ",\"auxiliary_graph_calls\":" << row.auxiliary_graph_calls
              << ",\"committed_target_rows\":" << row.processed_target << ",\"draft_rows\":" << row.processed_draft << '}';
  }
  std::cout << "]}";
}

template<GenerationMode Mode>
int run_generation_sessions(Model &model, std::span<const inference::GenerationSessionSchedule> schedules,
    const std::filesystem::path &trace, bool trace_states, Clock::time_point startup,
    Clock::time_point constructed, bool prefix_enabled, std::span<const cache::TokenPlanRef> token_plans,
    std::span<const std::size_t> sampling_indices = {}, std::span<const inference::RequestStopPolicy> stop_policies = {}) {
  constexpr bool mtp = Mode == GenerationMode::kMtp;
  if (!stop_policies.empty() && stop_policies.size() != schedules.size())
    throw std::runtime_error("one immutable stop policy is required per request");
  for (const auto &policy : stop_policies) {
    policy.validate(model.vocabulary);
    if (mtp && !policy.is_default()) throw std::runtime_error("nondefault request stop policy is ordinary Text only");
  }
  if (!sampling_indices.empty() && sampling_indices.size() != schedules.size())
    throw std::runtime_error("one cold sampling choice is required per request");
  for (const auto index : sampling_indices) {
    (void)model.sampling_configuration(index);
    if (mtp && index) throw std::runtime_error("nondefault request sampling is ordinary Text only");
  }
  if (schedules.empty() || model.cache_plan.draft_cache != mtp)
    throw std::runtime_error("natural generation mode differs from its complete model/cache plan");
  if (prefix_enabled && token_plans.size() != schedules.size())
    throw std::runtime_error("natural generation requires one validated prefix input per request");
  const auto prompt_role = [](std::size_t rows, bool lag = false) {
    if constexpr (!mtp) { (void)rows; (void)lag; return GraphRole::kText; }
    else return rows == 1 ? (lag ? GraphRole::kMtpCatchupSingle : GraphRole::kMtpSingle)
        : (lag ? GraphRole::kMtpCatchupPrefill : GraphRole::kMtpPrefill);
  };
  auto prefix = prefix_enabled ? std::make_unique<cache::PrefixCache>(model.cache_plan.resident_sessions) : nullptr;
  std::vector<GenerationRequest> requests;
  requests.reserve(schedules.size());
  std::size_t total_turns = 0, requested_outputs = 0, decode_bound = 0, restore_bound = 0;
  for (const auto &schedule : schedules) {
    if (schedule.turns.empty()) throw std::runtime_error("natural request requires a nonempty turn list");
    if (schedule.turns.front().prompt.empty())
      throw std::runtime_error("natural request requires a nonempty initial prompt");
    std::size_t outputs = 0;
    for (const auto &turn : schedule.turns) {
      if (!turn.max_outputs)
        throw std::runtime_error("natural turns require positive output quota");
      outputs = cache::add(outputs, turn.max_outputs);
    }
    const auto &first = schedule.turns.front().prompt;
    requests.emplace_back(schedule.id, first, trace.empty() ? trace : trace /
        ("session" + std::to_string(requests.size()) + "_" + schedule.id), outputs, sampling_indices.empty() ? 0 : sampling_indices[requests.size()],
        stop_policies.empty() ? inference::RequestStopPolicy{} : stop_policies[requests.size()]);
    auto &request = requests.back(); request.progress.bind_turns(schedule.turns);
    request.progress.initial_sampling = {model.sampling_seed, static_cast<std::uint64_t>(requests.size() - 1), 0};
    request.turn_observations.resize(schedule.turns.size());
    request.warm_inputs.reserve(schedule.turns.size());
    for (std::size_t i = 0; i < schedule.turns.size(); ++i) {
      const auto &ids = schedule.turns[i].prompt;
      auto owner = std::make_shared<std::vector<std::int32_t>>(cache::add(ids.size(), i ? 1 : 0));
      std::copy(ids.begin(), ids.end(), owner->begin() + (i ? 1 : 0));
      request.warm_inputs.push_back(std::move(owner));
    }
    request.prompt = *request.warm_inputs[0];
    const auto ledger_capacity = generation_record_capacity(schedule.turns, SIZE_MAX, first.size());
    request.ledger = std::make_unique<GenerationLedger>(ledger_capacity, model.cache_plan.candidate_boundaries);
    request.round_emitted.resize(outputs - 1);
    for (auto &emitted : request.round_emitted) emitted.reserve(model.cache_plan.candidate_boundaries);
    if (prefix) request.input = token_plans[requests.size() - 1];
    total_turns = cache::add(total_turns, schedule.turns.size());
    requested_outputs = cache::add(requested_outputs, outputs);
    decode_bound = cache::add(decode_bound, outputs - schedule.turns.size());
    if (model.cache_plan.resident_sessions < schedules.size())
      restore_bound = cache::add(restore_bound, cache::multiply(schedule.turns.size() - 1, ledger_capacity));
  }
  const std::array pair{std::size_t{2}};
  std::map<GraphRole, std::map<std::size_t, std::size_t>> prompt_uses;
  std::map<GraphRole, std::map<std::size_t, std::vector<std::size_t>>> support;
  const auto note_prompt = [&](GraphRole role, std::size_t tokens, std::size_t member) {
    auto &count = prompt_uses[role][tokens]; count = cache::add(count, 1);
    auto &members = support[role][tokens];
    if (std::find(members.begin(), members.end(), member) == members.end()) members.push_back(member);
  };
  for (std::size_t i = 0; i < requests.size(); ++i) {
    std::vector<std::size_t> suffixes{requests[i].prompt.size()};
    if (prefix) for (std::size_t j = 0; j < i; ++j) {
      const auto length = requests[j].prompt.size();
      if (length < requests[i].prompt.size() && cache::same_prefix(*requests[i].input, *requests[j].input, length))
        suffixes.push_back(requests[i].prompt.size() - length);
    }
    std::sort(suffixes.begin(), suffixes.end()); suffixes.erase(std::unique(suffixes.begin(), suffixes.end()), suffixes.end());
    for (auto tokens : suffixes) note_prompt(prompt_role(tokens), tokens, i);
    for (std::size_t turn = 1; turn < requests[i].progress.turns().size(); ++turn) {
      const auto tokens = cache::add(requests[i].progress.turns()[turn].prompt.size(), 1);
      note_prompt(prompt_role(tokens), tokens, i);
      if constexpr (mtp) note_prompt(prompt_role(tokens, true), tokens, i);
    }
  }
  const std::array roles{GraphRole::kMtpRound, GraphRole::kMtpCatchupRound,
      GraphRole::kMtpSingle, GraphRole::kMtpCatchupSingle};
  // Normal rebuild preserves original committed chunks, not original proposals.
  // Every consumed verifier prefix can become a teacher-forced prefill chunk.
  prompt_uses[prompt_role(1)].try_emplace(1, 0);
  if constexpr (mtp) for (std::size_t rows = 2; rows <= model.cache_plan.candidate_boundaries; ++rows)
    prompt_uses[GraphRole::kMtpPrefill].try_emplace(rows, 0);
  for (std::size_t sampling_index = 0; sampling_index < model.sampling_configuration_count(); ++sampling_index) {
    if (std::ranges::none_of(requests, [&](const auto &request) { return request.sampling_index == sampling_index; })) continue;
    const auto policy_decoding_requests = static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [&](const auto &request) {
      return request.sampling_index == sampling_index && std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; });
    }));
  for (std::size_t width = 1; width <= model.cache_plan.active_sequences; ++width) {
    std::map<GraphRole, std::map<std::size_t, std::size_t>> shapes;
    for (const auto &[role, tokens] : prompt_uses) for (const auto &[count, uses] : tokens) {
      const auto found_role = support.find(role);
      const auto eligible = found_role == support.end() || !found_role->second.contains(count) ? 0 : static_cast<std::size_t>(std::count_if(found_role->second.at(count).begin(), found_role->second.at(count).end(), [&](auto member) { return requests[member].sampling_index == sampling_index; }));
      if (width == 1 || width <= eligible) shapes[role][count] = cache::add(uses, width == 1 ? restore_bound : 0);
    }
    if (decode_bound && width <= policy_decoding_requests) {
      if constexpr (mtp) for (auto role : roles) {
        if (drafts_token(role) && (model.max_drafts() != 1 || model.recursive_mtp())) continue;
        auto &uses = shapes[role][drafts_token(role) ? 2 : 1]; uses = cache::add(uses, decode_bound);
      }
      else shapes[GraphRole::kText][1] = cache::add(shapes[GraphRole::kText][1], decode_bound);
    }
    if (mtp && decode_bound && width <= policy_decoding_requests && model.recursive_mtp())
      for (std::size_t rows = 1; rows <= model.cache_plan.candidate_boundaries; ++rows)
        shapes[GraphRole::kVerify][rows] = decode_bound;
    // R1 restoration also needs lag0 versions of each original warm chunk.
    if (mtp && width == 1) for (const auto &[tokens, uses] : prompt_uses[GraphRole::kMtpCatchupPrefill]) {
      (void)uses; shapes[GraphRole::kMtpPrefill].try_emplace(tokens, restore_bound);
    }
    for (const auto &[role, entries] : shapes) {
      std::vector<std::size_t> chunks; chunks.reserve(entries.size());
      for (const auto &[tokens, uses] : entries) { (void)uses; chunks.push_back(tokens); }
      model.prepare_family(chunks, false, width, role, false, TextCompletion::kEmitNext, sampling_index);
      for (const auto &[tokens, uses] : entries) model.plan_replay_uses(tokens, width,
          cache::add(uses, width == 1 ? restore_bound : 0), role, TextCompletion::kEmitNext, sampling_index);
    }
  }
  }
  if (mtp && decode_bound && model.recursive_mtp()) model.prepare_recursive_round_programs();
  const std::array first_prefill{requests.front().prompt.size()};
  model.prepare_execution(first_prefill, 1, prompt_role(first_prefill[0]), TextCompletion::kEmitNext, requests.front().sampling_index);
  if constexpr (!mtp) if (decode_bound) {
    const std::array decode{std::size_t{1}};
    for (std::size_t i = 0; i < model.sampling_configuration_count(); ++i)
      if (std::ranges::any_of(requests, [i](const auto &request) { return request.sampling_index == i &&
          std::ranges::any_of(request.progress.turns(), [](const auto &turn) { return turn.max_outputs > 1; }); }))
        model.prepare_execution(decode, 1, GraphRole::kText, TextCompletion::kEmitNext, i);
  }
  if (mtp && decode_bound && model.max_drafts() == 1 && !model.recursive_mtp()) for (auto role : roles) if (drafts_token(role)) model.prepare_execution(pair, 1, role);
  inference::RequestWindow window;
  window.prepare(requests.size(), model.cache_plan.resident_sessions);
  for (std::size_t index = 0; index < requests.size(); ++index)
    requests[index].request_key = window.enqueue(index);
  std::vector<std::size_t> dispatch, dispatch_turns;
  dispatch.reserve(requested_outputs); dispatch_turns.reserve(requested_outputs);
  std::vector<std::size_t> evictions; evictions.reserve(total_turns);
  std::size_t completed = 0, admitted = 0;
  struct BatchObservation { std::size_t tokens; GraphRole role; bool generation; hyle::container::SmallVector<std::size_t, 2> members; };
  std::vector<BatchObservation> batches; batches.reserve(requested_outputs);
  std::size_t actual_prefill_calls = 0, actual_decode_calls = 0;
  double actual_step_sum = 0;
  std::size_t peak_resident = 0, peak_committed = 0;
  const auto resident_count = [&] {
    return static_cast<std::size_t>(std::count_if(requests.begin(), requests.end(), [](const auto &request) { return bool(request.session); }));
  };
  inference::BatchIndices batch_indices;
  batch_indices.prepare(model.cache_plan.resident_sessions, model.cache_plan.active_sequences);
  const auto ready = Clock::now();
  std::cerr << "generation_shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
            << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n'
            << "measure_memory name=ready_before_first\n";
  const auto start = Clock::now(); auto endpoint = start;
  double prefill_seconds = 0, decode_seconds = 0;
  while (completed != requests.size()) {
    while (const auto next = window.waiting_front()) {
      const auto index = next->slot; auto &request = requests[index];
      preflight_generation_sampling(model, request);
      request.admission_start = Clock::now();
      if (!request.session) {
        // Earlier new requests in this wave have no head until prepare below.
        // Their reserved slots still count when admitting/restoring a peer.
        const auto active = window.active();
        const auto reserved_heads = static_cast<std::size_t>(std::count_if(active.begin(), active.end(),
            [&](auto member) { return !requests[member.slot].session; }));
        while (cache::add(resident_count(), reserved_heads) >= model.cache_plan.resident_sessions) {
          const auto victim = window.idle_victim([&](auto member) { return bool(requests[member].session); },
              [&](auto member) { return requests[member].last_idle; });
          if (!victim) throw std::runtime_error("all MTP heads are generating; no idle owner may be evicted");
          const auto retired = retire_generation_request(requests[*victim]);
          if (retired.retired) {
            requests[*victim].session.reset();
            ++requests[*victim].idle_evictions; evictions.push_back(*victim);
          }
          check(retired.status);
          if (!retired.retired)
            throw std::runtime_error("idle generation request still retains an in-flight episode");
        }
        restore_generation_session(model, request, prefix.get(), trace_states, requests);
      }
      if (request.progress.turn_index && !request.session)
        throw std::runtime_error("later generation turn requires its rebuilt complete head");
      if (!request.progress.turn_started()) {
        request.prompt = request.progress.begin_turn(*request.warm_inputs[request.progress.turn_index],
            request.progress.turn_index ? std::optional<std::int32_t>(request.session->head->next_id) : std::nullopt);
        request.turn_observations[request.progress.turn_index].output_begin = request.progress.turn_output_begin;
      }
      window.admitted(*next); ++admitted;
    }
    const auto active = window.active();
    if (active.empty()) throw std::runtime_error("MTP admission has no runnable resident request");
    // Resolve every lookup/admission in this wave before any checkpoint is
    // published. A same-wave prompt cannot observe another member's new head.
    hyle::container::SmallVector<PreparedGenerationCall, 2> calls;
    if (!calls.try_reserve(active.size())) throw std::bad_alloc{};
    for (auto key : active) {
      auto &request = requests[key.slot];
      std::cerr << "generation_session id=" << request.id << " turn=" << request.progress.turn_index + 1 << " episode=" << request.rounds.size() << '\n';
      calls.push_back(prepare_generation_call(model, request, request.progress.output_limit(), prefix.get(), requests));
    }
    batch_indices.reset(calls.size());
    for (std::size_t first = 0; first < calls.size(); ++first) {
      if (batch_indices.completed(first)) continue;
      const auto selected = batch_indices.select(first,
          calls[first].full_hit ? std::size_t{1} : model.cache_plan.active_sequences, [&](std::size_t j) {
        return calls[j].full_hit == calls[first].full_hit &&
            calls[j].request->sampling_index == calls[first].request->sampling_index &&
            calls[j].generation == calls[first].generation && calls[j].role == calls[first].role &&
            calls[j].completion == calls[first].completion &&
            calls[j].planned_rows() == calls[first].planned_rows();
      });
      hyle::container::SmallVector<PreparedGenerationCall, 2> bucket;
      if (!bucket.try_reserve(selected.size())) throw std::bad_alloc{};
      for (auto j : selected) bucket.push_back(calls[j]);
      while (bucket.size() > 1 && !bucket[0].full_hit && !make_generation_batch_room(model, prefix.get(), std::span(bucket.data(), bucket.size()), requests)) {
        bucket.pop_back();
      }
      std::cerr << "generation_batch requests=" << bucket.size() << " T=" << bucket[0].planned_rows()
                << " role=" << static_cast<int>(bucket[0].role) << " full_hit=" << bucket[0].full_hit << '\n';
      execute_generation_calls(model, std::span(bucket.data(), bucket.size()), trace_states, prefix.get(), requests);
      auto finished = endpoint;
      hyle::container::SmallVector<std::size_t, 2> members;
      if (!members.try_reserve(bucket.size())) throw std::bad_alloc{};
      batch_indices.mark_selected(bucket.size());
      for (std::size_t j = 0; j < bucket.size(); ++j) {
        const auto index = active[selected[j]].slot;
        dispatch.push_back(index); dispatch_turns.push_back(requests[index].progress.turn_index); members.push_back(index);
        finished = (std::max)(finished, requests[index].endpoint);
      }
      const auto seconds = std::chrono::duration<double>(finished - endpoint).count();
      (bucket[0].generation ? decode_seconds : prefill_seconds) += seconds; endpoint = finished;
      if (!bucket[0].full_hit) {
        (bucket[0].generation ? actual_decode_calls : actual_prefill_calls)++;
        actual_step_sum += bucket[0].step_seconds;
        batches.push_back({drafts_token(bucket[0].role) ? 2 : bucket[0].ids().size(), bucket[0].role, bucket[0].generation, std::move(members)});
      }
      peak_resident = (std::max)(peak_resident, resident_count());
      peak_committed = (std::max)(peak_committed, std::size_t(model.snapshot_versions.live));
      if (model.snapshot_versions.live > model.cache_plan.resident_sessions) throw std::runtime_error("MTP committed version count exceeds N after publication");
    }
    for (auto &call : calls) publish_generation_prompt(model, call, prefix.get());
    for (std::size_t at = 0; at < window.active().size();) {
      const auto index = window.active()[at].slot; auto &request = requests[index];
      if (!request.progress.turn_done) { ++at; continue; }
      publish_generation_completed(model, request, prefix.get());
      const auto more = request.progress.more_turns();
      const auto idle_order = window.release_turn(request.request_key, more);
      if (!more) {
        const auto retired = retire_generation_request(request);
        if (retired.retired) {
          request.session.reset();
          window.release(request.request_key);
          ++completed;
        }
        check(retired.status);
        if (!retired.retired)
          throw std::runtime_error("finished generation request still retains an in-flight episode");
      }
      else { request.last_idle = idle_order; request.progress.advance_turn(); }
    }
  }
  const auto post_endpoint_seconds = std::chrono::duration<double>(Clock::now() - endpoint).count();
  const auto session_seconds = std::chrono::duration<double>(endpoint - start).count();
  std::size_t logical_prompt = 0, user_prompt = 0, prefill_tokens = 0, prefill_draft = 0, prefill_calls = 0;
  std::size_t generated = 0, decode_calls = 0, verified = 0, drafted = 0, committed = 0;
  std::size_t draft_rounds = 0, agreements = 0, accepts = 0, hits = 0, misses = 0, full_hits = 0, reused = 0;
  std::size_t restores = 0, recomputed_changes = 0; GenerationRestoreStats restored; double step_sum_seconds = 0;
  for (const auto &request : requests) {
    for (const auto &turn : request.progress.turns()) user_prompt = cache::add(user_prompt, turn.prompt.size());
    prefill_tokens = cache::add(prefill_tokens, request.prefill_graph_tokens);
    prefill_draft = cache::add(prefill_draft, request.prefill_draft_rows);
    prefill_calls = cache::add(prefill_calls, request.prefill_graph_calls);
    generated = cache::add(generated, request.progress.generated.size()); decode_calls = cache::add(decode_calls, request.rounds.size());
    verified = cache::add(verified, request.verified_rows); drafted = cache::add(drafted, request.draft_rows);
    committed = cache::add(committed, request.visible_tokens); draft_rounds = cache::add(draft_rounds, request.draft_rounds);
    agreements = cache::add(agreements, request.agreements); accepts = cache::add(accepts, request.accepts);
    hits = cache::add(hits, request.prefix_hits); misses = cache::add(misses, request.prefix_misses);
    full_hits = cache::add(full_hits, request.prefix_full_hits); reused = cache::add(reused, request.reused_tokens);
    step_sum_seconds += request.prefill_step_seconds + request.decode_step_sum_seconds;
    restores = cache::add(restores, request.restore_count);
    restored.graph_calls = cache::add(restored.graph_calls, request.restored.graph_calls);
    restored.target_rows = cache::add(restored.target_rows, request.restored.target_rows);
    restored.draft_rows = cache::add(restored.draft_rows, request.restored.draft_rows);
    restored.normalized_lag_rows = cache::add(restored.normalized_lag_rows, request.restored.normalized_lag_rows);
    recomputed_changes = cache::add(recomputed_changes, request.restored.decision_changes.size());
    restored.step_seconds += request.restored.step_seconds; restored.seconds += request.restored.seconds;
  }
  logical_prompt = cache::add(user_prompt, total_turns - requests.size());
  const auto graph_calls = cache::add(cache::add(cache::add(actual_prefill_calls, actual_decode_calls), restored.graph_calls),
      cache::add(model.frontier_sampling_graph_calls, cache::add(model.draft_program_graph_calls, model.prefix_decision_graph_calls)));
  std::cout << std::setprecision(17) << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
      << "\",\"mtp\":" << (model.cache_plan.draft_cache ? "true" : "false") << model.weight_inventory_report()
      << ",\"session_count\":" << requests.size() << ",\"turn_count\":" << total_turns << ",\"active_sequences\":" << model.cache_plan.active_sequences << ",\"max_sessions\":" << model.cache_plan.resident_sessions
      << model.cache_report() << ",\"prefix_cache\":" << (prefix_enabled ? "true" : "false")
      << ",\"admitted_requests\":" << requests.size() << ",\"admitted_turns\":" << admitted << ",\"retired_requests\":" << completed
      << ",\"peak_resident_requests\":" << peak_resident << ",\"peak_committed_versions\":" << peak_committed
      << ",\"live_committed_versions\":" << model.snapshot_versions.live << ",\"idle_evictions\":" << evictions.size()
      << ",\"prefix_hits\":" << hits << ",\"prefix_misses\":" << misses << ",\"prefix_full_hits\":" << full_hits << ",\"reused_tokens\":" << reused
      << ",\"prefix_entries\":" << (prefix ? prefix->size() : 0) << ",\"prefix_evictions\":" << (prefix ? prefix->evictions() : 0)
      << ",\"requested_output_tokens\":" << requested_outputs << ",\"generated_tokens\":" << generated << ",\"generated_decode_tokens\":" << generated - total_turns
      << ",\"actual_graph_calls\":" << graph_calls << ",\"prefill_graph_calls\":" << actual_prefill_calls << ",\"decode_graph_calls\":" << actual_decode_calls
      << ",\"graph_participations\":" << prefill_calls + decode_calls + model.frontier_sampling_graph_calls + model.draft_program_graph_calls + model.prefix_decision_graph_calls
      << ",\"logical_dispatches\":" << dispatch.size() << ",\"logical_prompt_tokens\":" << logical_prompt << ",\"logical_user_prompt_tokens\":" << user_prompt
      << ",\"pending_flushed_tokens\":" << total_turns - requests.size() << ",\"prefill_target_rows\":" << prefill_tokens
      << ",\"verified_target_rows\":" << verified
      << ",\"actual_target_rows\":" << cache::add(cache::add(prefill_tokens, verified), restored.target_rows)
      << ",\"committed_target_rows\":" << committed << ",\"committed_target_decode_rows\":" << committed - logical_prompt
      << ",\"restore_count\":" << restores << ",\"restored_graph_calls\":" << restored.graph_calls << ",\"restored_target_rows\":" << restored.target_rows
      << ",\"restore_step_seconds\":" << restored.step_seconds << ",\"restore_seconds\":" << restored.seconds
      << ",\"restore_contract\":\"recompute_committed_inputs\"";
  if constexpr (mtp) std::cout << ",\"prefill_draft_rows\":" << prefill_draft << ",\"draft_rows\":" << drafted
      << ",\"actual_draft_rows\":" << cache::add(cache::add(prefill_draft, drafted), restored.draft_rows)
      << ",\"draft_rounds\":" << draft_rounds << ",\"agreements\":" << agreements << ",\"consumed_drafts\":" << accepts
      << ",\"restored_draft_rows\":" << restored.draft_rows << ",\"normalized_lag_rows\":" << restored.normalized_lag_rows;
  std::cout << ",\"recomputed_decision_changes\":" << recomputed_changes
      << ",\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds << ",\"session_seconds\":" << session_seconds
      << ",\"step_sum_seconds\":" << actual_step_sum + restored.step_seconds + model.frontier_sampling_seconds
      << ",\"participation_step_sum_seconds\":" << step_sum_seconds << ",\"post_endpoint_seconds\":" << post_endpoint_seconds
      << ",\"phase_timing\":\"disjoint_continuous_episode_intervals\",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
      << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count() << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
      << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(requests.front().turn_observations.front().prefill_endpoint - startup).count()
      << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? logical_prompt / prefill_seconds : 0)
      << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? (generated - total_turns) / decode_seconds : 0)
      << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated / session_seconds : 0)
      << ",\"diagnostic\":" << (trace.empty() ? "false" : "true") << ",\"dispatch\":[";
  for (std::size_t i = 0; i < dispatch.size(); ++i)
    std::cout << (i ? "," : "") << "{\"session_id\":\"" << requests[dispatch[i]].id << "\",\"turn\":" << dispatch_turns[i] + 1 << '}';
  std::cout << "],\"batches\":[";
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto &batch = batches[i];
    std::cout << (i ? "," : "") << "{\"index\":" << i << ",\"requests\":" << batch.members.size()
        << ",\"tokens\":" << batch.tokens << ",\"role\":" << static_cast<int>(batch.role)
        << ",\"generation\":" << (batch.generation ? "true" : "false") << ",\"session_ids\":[";
    for (std::size_t j = 0; j < batch.members.size(); ++j)
      std::cout << (j ? "," : "") << '"' << requests[batch.members[j]].id << '"';
    std::cout << "]}";
  }
  std::cout << "],\"sessions\":[";
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto &request = requests[i]; std::size_t quota = 0;
    for (const auto &turn : request.progress.turns()) quota = cache::add(quota, turn.max_outputs);
    std::cout << (i ? "," : "") << "{\"session_id\":\"" << request.id << "\",\"result\":";
    print_generation_request(model, request, quota, startup, ready);
    std::cout << ",\"restore_count\":" << request.restore_count << ",\"idle_evictions\":" << request.idle_evictions
        << ",\"checkpoint_resume_hits\":" << request.checkpoint_resume_hits
        << ",\"checkpoint_resume_misses\":" << request.checkpoint_resume_misses << ",\"ledger\":[";
    std::size_t round = 0;
    for (std::size_t j = 0; j < request.ledger->committed_count; ++j) {
      const auto &record = request.ledger->records[j]; if (record.generation) ++round;
      std::uint64_t hash = 1469598103934665603ULL;
      for (const auto token : record.ids()) hash = (hash ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
      const auto prefix_name = record.turn ? "round" + std::to_string(record.turn + 1) + "_" : "";
      const auto name = prefix_name + (record.generation ? (mtp ? "mtp" : "decode") + std::to_string(round) : "prefill");
      std::cout << (j ? "," : "") << "{\"index\":" << j << ",\"turn\":" << record.turn + 1 << ",\"role\":" << static_cast<int>(record.role)
          << ",\"requests\":" << record.requests << ",\"member_index\":" << record.member_index
          << ",\"generation\":" << (record.generation ? "true" : "false") << ",\"input_rows\":" << record.ids().size()
          << ",\"verified_rows\":" << record.verified_rows << ",\"draft_limit\":" << record.draft_limit
          << ",\"consumed_drafts\":" << record.consumed_drafts << ",\"auxiliary_graph_calls\":" << record.auxiliary_graph_calls
          << ",\"input_hash\":" << hash << ",\"first_input\":" << record.ids().front() << ",\"quota\":" << record.quota
          << ",\"before_length\":" << record.before.length << ",\"before_frontier\":" << record.before.frontier << ",\"before_lag\":" << record.before.lag
          << ",\"before_next_id\":" << record.before.next_id << ",\"draft\":" << record.draft << ",\"target_first\":" << record.target_first << ",\"target_second\":" << record.target_second
          << ",\"sampling_seed\":" << record.before.sampling.seed << ",\"sampling_stream\":" << record.before.sampling.stream
          << ",\"sampling_draw_before\":" << record.before.sampling.draw << ",\"sampling_draw_after\":" << record.after.sampling.draw
          << ",\"accepted\":" << record.accepted << ",\"draft_rows\":" << record.draft_rows << ",\"after_length\":" << record.after.length
          << ",\"after_frontier\":" << record.after.frontier << ",\"after_lag\":" << record.after.lag << ",\"next_id\":" << record.after.next_id
          << ",\"committed_input_ids\":[";
      if (record.generation) for (std::size_t k = 0; k < record.accepted; ++k)
        std::cout << (k ? "," : "") << record.accepted_inputs[k];
      std::cout << "],\"verification_input_ids\":[";
      for (std::size_t k = 0; k < record.verification_inputs.size(); ++k)
        std::cout << (k ? "," : "") << record.verification_inputs[k];
      std::cout << "],\"imported\":" << (record.imported ? "true" : "false")
          << ",\"source_geometry\":\"" << (record.imported ? "unknown" : "recorded")
          << "\",\"trace_name\":\"" << (record.imported ? "" : name) << "\"}";
    }
    std::cout << "]}";
  }
  std::cout << "]}\n"; return 0;
}
} // namespace qwen35

int qwen35::run(int argc, char **argv) {
  if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
    std::cout << "Hyle inference engine (Qwen text)\n"
        "Usage: hyle --inputs DIR [options]\n"
        "  --inputs DIR                Locally prepared checkpoint and token IDs\n"
        "  --backend cpu|cuda|apple|vulkan  Primary backend; required when multiple GPU routes are compiled\n"
        "  --mtp off|on                Multi-token prediction (default: off)\n"
        "  --mtp-max-drafts N          Maximum proposals; 0 disables drafting (default: 1)\n"
        "  --mtp-min-probability FLOAT  Stop below max draft probability, [0,1] (default: 0)\n"
        "  --sampling greedy|random    Token selection (default: greedy)\n"
        "  --temperature FLOAT         Random policy temperature > 0 (default: 1)\n"
        "  --top-k N                    Random policy top-k; 0 disables (default: 0)\n"
        "  --top-p FLOAT                Random policy nucleus in (0,1] (default: 1)\n"
        "  --seed N                     Unsigned 64-bit random seed (default: 0)\n"
        "  --steps N                   Maximum output tokens (default: 4)\n"
        "  --capacity N                Total context capacity (default: 128)\n"
        "  --cpu-ffn-layers N          First N target FFNs on CPU; needs CPU+GPU build (default: 0)\n"
        "  --schedule FILE             One session's prompt/generate turns\n"
        "  --session-schedules FILE    Multiple named session schedules\n"
        "  --max-sessions auto|N       Bounded retained session states\n"
        "  --active-sequences auto|N   Maximum actual execution batch width\n"
        "  --cache-bytes auto|N        Session cache backing budget\n"
        "  --prefix-cache off|auto     Reuse completed prompt/turn checkpoints\n"
        "  --replay auto|commands|prefer-capture\n"
        "Prepare inputs with engine/models/qwen35/prepare.py; no model is downloaded at runtime.\n";
    return 0;
  }
  const auto startup = qwen35::Clock::now();
  try {
    std::filesystem::path input, continuation, trace, layer0_probe, decode_ids, schedule_file, session_file;
    std::size_t steps = 4, capacity = 128, max_sessions = 0, active_sequences = 0, cache_bytes = 0, max_drafts = 1, cpu_ffn_layers = 0;
    bool progress = false, trace_states = true, explicit_steps = false;
    bool mtp_enabled = false;
    bool explicit_mtp_drafts = false;
    float min_draft_probability = 0;
    bool explicit_mtp_probability = false;
    qwen35::Model::Sampling sampler;
    bool explicit_sampling_parameters = false;
    bool single_schedule = false, multiple_schedules = false;
    std::string prefix_policy;
    bool explicit_continuation = false, explicit_decode_ids = false, preparation_diagnostics = false;
    auto replay = qwen35::ReplayPolicy::kAuto;
    qwen35::BackendSelection backend_selection;
    const auto positive_size = [](const std::string &option, const std::string &value) {
      std::size_t result = 0;
      const auto converted = std::from_chars(value.data(), value.data() + value.size(), result);
      if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || result == 0)
        throw std::runtime_error(option + " requires a positive integer");
      return result;
    };
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("option needs a value");
      const std::string value = argv[++i];
      if (option == "--inputs") input = value;
      else if (option == "--backend") {
        if (value == "cpu") backend_selection.primary = qwen35::BackendKind::kCpu;
        else if (value == "cuda") backend_selection.primary = qwen35::BackendKind::kCuda;
        else if (value == "apple") backend_selection.primary = qwen35::BackendKind::kApple;
        else if (value == "vulkan") backend_selection.primary = qwen35::BackendKind::kVulkan;
        else throw std::runtime_error("--backend requires cpu, cuda, apple or vulkan");
      }
      else if (option == "--sampling") {
        if (value != "greedy" && value != "random") throw std::runtime_error("--sampling requires greedy or random");
        sampler.enabled = value == "random";
      }
      else if (option == "--temperature" || option == "--top-p" || option == "--mtp-min-probability") {
        float number = 0;
        std::istringstream parsed(value); parsed.imbue(std::locale::classic());
        parsed >> std::noskipws >> number;
        const bool probability = option == "--mtp-min-probability";
        if (probability && (!parsed || parsed.peek() != std::char_traits<char>::eof() ||
            !std::isfinite(number) || number < 0 || number > 1))
          throw std::runtime_error("--mtp-min-probability requires a finite probability within [0,1]");
        if (!probability && (!parsed || parsed.peek() != std::char_traits<char>::eof() || !std::isfinite(number) ||
            number <= 0 || (option == "--top-p" && number > 1)))
          throw std::runtime_error(option + " requires a finite positive number" + (option == "--top-p" ? " at most one" : ""));
        if (probability) { min_draft_probability = number; explicit_mtp_probability = true; }
        else {
          if (option == "--temperature") sampler.policy.temperature = number; else sampler.policy.top_p = number;
          explicit_sampling_parameters = true;
        }
      }
      else if (option == "--top-k" || option == "--seed") {
        std::uint64_t number = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            (option == "--top-k" && number > INT32_MAX))
          throw std::runtime_error(option + " requires an unsigned integer" + (option == "--top-k" ? " no greater than the vocabulary" : ""));
        if (option == "--seed") sampler.seed = number; else sampler.policy.top_k = static_cast<std::size_t>(number);
        explicit_sampling_parameters = true;
      }
      else if (option == "--mtp") {
        if (value != "on" && value != "off") throw std::runtime_error("--mtp requires on or off");
        mtp_enabled = value == "on";
      }
      else if (option == "--mtp-max-drafts") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), max_drafts);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--mtp-max-drafts requires a nonnegative integer within the supported size range");
        explicit_mtp_drafts = true;
      }
      else if (option == "--cpu-ffn-layers") {
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), cpu_ffn_layers);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw std::runtime_error("--cpu-ffn-layers requires a nonnegative integer within the supported size range");
      }
      else if (option == "--continuation") { continuation = value; explicit_continuation = true; }
      else if (option == "--schedule") { schedule_file = value; single_schedule = true; }
      else if (option == "--session-schedules") { session_file = value; multiple_schedules = true; }
      else if (option == "--max-sessions") max_sessions = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--active-sequences") active_sequences = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--cache-bytes") cache_bytes = value == "auto" ? 0 : positive_size(option, value);
      else if (option == "--prefix-cache") {
        if (value != "off" && value != "auto") throw std::runtime_error("--prefix-cache requires off or auto");
        prefix_policy = value;
      }
      else if (option == "--trace") trace = value;
      else if (option == "--preparation-diagnostics") {
        if (value != "0" && value != "1") throw std::runtime_error("--preparation-diagnostics requires 0 or 1");
        preparation_diagnostics = value == "1";
      }
      else if (option == "--trace-level") {
        if (value != "full" && value != "decisions")
          throw std::runtime_error("--trace-level requires full or decisions");
        trace_states = value == "full";
      }
      else if (option == "--decode-ids") { decode_ids = value; explicit_decode_ids = true; }
      else if (option == "--layer0-probe") layer0_probe = value;
      else if (option == "--steps") { steps = positive_size(option, value); explicit_steps = true; }
      else if (option == "--capacity") capacity = positive_size(option, value);
      else if (option == "--progress") {
        if (value != "direct" && value != "scheduler")
          throw std::runtime_error("--progress requires direct or scheduler");
        progress = value == "scheduler";
      }
      else if (option == "--replay") {
        if (value != "auto" && value != "commands" && value != "prefer-capture")
          throw std::runtime_error("--replay requires auto, commands or prefer-capture");
        replay = value == "auto" ? qwen35::ReplayPolicy::kAuto
            : value == "prefer-capture" ? qwen35::ReplayPolicy::kPreferCapture
                                         : qwen35::ReplayPolicy::kCommands;
      }
      else throw std::runtime_error("unknown option: " + option);
    }
    const bool scheduled = single_schedule || multiple_schedules;
    if (input.empty()) throw std::runtime_error("--inputs DIR is required");
    backend_selection = qwen35::select_execution(backend_selection, cpu_ffn_layers);
    auto weight_source = qwen35::prepare_weight_source(input / "weights.tsv", mtp_enabled);
    const auto geometry = weight_source.roles.geometry;
    qwen35::validate_mtp_source(geometry, mtp_enabled);
    qwen35::FfnPlacement::validate(cpu_ffn_layers, geometry.layers.size());
    if (sampler.policy.top_k > static_cast<std::size_t>(geometry.vocabulary))
      throw std::runtime_error("--top-k exceeds the source vocabulary");
    if (explicit_mtp_drafts && !mtp_enabled) throw std::runtime_error("--mtp-max-drafts requires --mtp on");
    if (explicit_mtp_probability && !mtp_enabled) throw std::runtime_error("--mtp-min-probability requires --mtp on");
    if (explicit_sampling_parameters && !sampler.enabled)
      throw std::runtime_error("--temperature, --top-k, --top-p and --seed require --sampling random");
    if (sampler.enabled && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("random sampling does not support layer or preparation diagnostics");
    const bool prefix_enabled = prefix_policy == "auto" || (prefix_policy.empty() && multiple_schedules);
    if (mtp_enabled) {
      if (explicit_continuation || explicit_decode_ids || !layer0_probe.empty())
        throw std::runtime_error("--mtp on uses natural generation; --continuation, --decode-ids and --layer0-probe are not supported");
      if (preparation_diagnostics)
        throw std::runtime_error("--preparation-diagnostics is not supported with --mtp on");

    }
    if (prefix_enabled && !scheduled) throw std::runtime_error("--prefix-cache auto requires a schedule");
    if (single_schedule && multiple_schedules)
      throw std::runtime_error("--schedule and --session-schedules are exclusive");
    using Grammar = qwen35::benchmark::ScheduleGrammar;
    const auto grammar = single_schedule ? qwen35::benchmark::schedule_grammar(schedule_file)
        : multiple_schedules ? qwen35::benchmark::session_schedule_grammar(session_file,
            std::vector<qwen35::benchmark::SessionManifestEntry>{}.max_size()) : Grammar::kPromptOnly;
    const bool natural_schedule = scheduled && grammar != Grammar::kFixedInput;
    const bool natural_single = !scheduled && !explicit_continuation && !explicit_decode_ids &&
        layer0_probe.empty() && !preparation_diagnostics;
    const bool natural_generation = natural_schedule || natural_single;
    if (mtp_enabled && grammar == Grammar::kFixedInput)
      throw std::runtime_error("--mtp on requires natural prompt/generate requests; decode IDSFILE is fixed replay");
    if (natural_schedule && (!layer0_probe.empty() || preparation_diagnostics))
      throw std::runtime_error("natural requests do not support layer or preparation diagnostics");
    if (scheduled && (explicit_continuation || explicit_decode_ids || (explicit_steps && !natural_schedule)))
      throw std::runtime_error("schedule inputs are exclusive with continuation, decode IDs and fixed-replay --steps");
    if ((single_schedule && schedule_file.empty()) || (multiple_schedules && session_file.empty()))
      throw std::runtime_error("schedule requires a nonempty manifest path");
    if (preparation_diagnostics && !scheduled)
      throw std::runtime_error("--preparation-diagnostics requires a fixed replay schedule");
    std::vector<qwen35::benchmark::SessionSchedule> schedules;
    std::vector<hyle::inference::GenerationSessionSchedule> generation_schedules;
    if (natural_schedule && single_schedule)
      generation_schedules.push_back({"default", qwen35::benchmark::read_generation_schedule(schedule_file,
          geometry.vocabulary, capacity, steps, explicit_steps)});
    else if (natural_schedule && multiple_schedules)
      generation_schedules = qwen35::benchmark::read_generation_session_schedules(session_file, geometry.vocabulary,
          capacity, std::vector<hyle::inference::GenerationSessionSchedule>{}.max_size(), steps, explicit_steps);
    else if (single_schedule)
      schedules.push_back({"default", qwen35::benchmark::read_schedule(schedule_file, geometry.vocabulary, capacity, true)});
    else if (multiple_schedules)
      schedules = qwen35::benchmark::read_session_schedules(session_file, geometry.vocabulary, capacity,
          std::vector<qwen35::benchmark::SessionSchedule>{}.max_size(), true);
    const auto input_sessions = natural_schedule ? generation_schedules.size() : scheduled ? schedules.size() : std::size_t{1};
    const auto recurrent_payload = qwen35::recurrent_bytes(qwen35::state_schema_for(geometry, capacity), geometry);
    // Configuration describes a maximum, never permission to overrun context.
    // Actual calls additionally clip against output quota and remaining rows.
    max_drafts = (std::min)(max_drafts, capacity - 1);
    if (max_sessions == 0) max_sessions = input_sessions;
    const auto cached_slots = (std::min)(max_sessions, input_sessions);
    if (active_sequences > max_sessions)
      throw std::runtime_error("--active-sequences cannot exceed --max-sessions");
    const auto forced = decode_ids.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(decode_ids, geometry.vocabulary, capacity);
    if (!decode_ids.empty()) {
      if (forced.empty()) throw std::runtime_error("--decode-ids requires a nonempty token tape");
      const auto forced_steps = qwen35::cache::add(forced.size(), 1);
      if (explicit_steps && steps != forced_steps)
        throw std::runtime_error("--steps must equal decode input count plus one");
      steps = forced_steps;
    }
    if (input.empty())
      throw std::runtime_error("usage: hyle --inputs DIR [--schedule FILE | --session-schedules FILE | [--continuation IDS] [--steps 4 | --decode-ids IDS]] [--mtp off|on] [--capacity 128] [--max-sessions auto|N] [--active-sequences auto|N] [--cache-bytes auto|N] [--prefix-cache off|auto] [--progress direct|scheduler] [--replay auto|commands|prefer-capture] [--trace DIR] [--trace-level full|decisions] [--preparation-diagnostics 0|1] [--layer0-probe DIR]");
    const auto prompt = scheduled ? std::vector<std::int32_t>{} : qwen35::read_ids(input / "prompt.ids", geometry.vocabulary, capacity);
    const auto extra = continuation.empty() ? std::vector<std::int32_t>{} : qwen35::read_ids(continuation, geometry.vocabulary, capacity);
    if (!scheduled) {
      if (prompt.empty()) throw std::runtime_error("generation requires a nonempty prompt before loading weights");
      if (prompt.size() > capacity || extra.size() > capacity - prompt.size() ||
          steps - 1 > capacity - prompt.size() - extra.size())
        throw std::runtime_error("prompt, continuation and requested generation exceed --capacity");
    }
    if (natural_single) generation_schedules.push_back({"default", {{prompt, steps}}});
    std::vector<std::size_t> planned_chunks;
    std::size_t mtp_schema_rows = 0;
    bool mtp_has_decode = !scheduled && steps > 1;
    if (natural_generation) {
      for (const auto &request : generation_schedules) for (std::size_t i = 0; i < request.turns.size(); ++i) {
        const auto tokens = qwen35::cache::add(request.turns[i].prompt.size(), i ? 1 : 0);
        planned_chunks.push_back(tokens);
        if (mtp_enabled) mtp_schema_rows = (std::max)(mtp_schema_rows, qwen35::cache::add(tokens, i ? 1 : 0));
        mtp_has_decode = mtp_has_decode || request.turns[i].max_outputs > 1;
      }
    } else if (scheduled) {
      for (const auto &request : schedules) for (const auto &phase : request.phases) {
        const auto tokens = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? phase.ids.size() : std::size_t{1};
        if (std::find(planned_chunks.begin(), planned_chunks.end(), tokens) == planned_chunks.end())
          planned_chunks.push_back(tokens);
      }
    } else {
      planned_chunks.push_back(prompt.size());
      if (!extra.empty()) planned_chunks.push_back(extra.size());
      planned_chunks.insert(planned_chunks.end(), steps - 1, 1);
    }
    // Immutable host token owners are bounded by the validated input batch.
    // Only suffixes at real prompt/turn boundaries enter the cold shape union.
    std::vector<qwen35::cache::TokenPlanRef> token_plans;
    if (prefix_enabled) {
      token_plans.reserve(input_sessions);
      if (natural_generation) for (const auto &request : generation_schedules) {
        token_plans.push_back(qwen35::cache::token_plan(request.turns.front().prompt, capacity));
      } else for (const auto &request : schedules) {
        std::vector<std::span<const std::int32_t>> chunks;
        chunks.reserve(request.phases.size());
        for (const auto &phase : request.phases) chunks.push_back(phase.ids);
        token_plans.push_back(qwen35::cache::token_plan(chunks, capacity));
      }
    }
    const auto max_query_rows = (std::max)({*std::max_element(planned_chunks.begin(), planned_chunks.end()), mtp_schema_rows,
        mtp_enabled && mtp_has_decode ? max_drafts + 1 : std::size_t{1}});
    const auto requested_active = (std::min)(active_sequences, input_sessions);
    const auto plan_for = [&](std::size_t residents) {
      return qwen35::cache::plan(geometry, weight_source.cache_row_bytes, capacity, residents,
          mtp_enabled && mtp_has_decode ? (std::max)(max_query_rows, std::size_t{2}) : max_query_rows, recurrent_payload,
          active_sequences == 0 ? residents : requested_active,
          mtp_enabled ? max_drafts + 1 : 1, mtp_enabled, sampler.enabled);
    };
    const auto minimum_residents = active_sequences == 0 ? std::size_t{1} : requested_active;
    auto selected_cache = plan_for(cached_slots);
    // Caller-written pools have an exact cold formula. Native generated
    // results instead enter the same byte scope after real completion, before
    // any append is taken or head published. Do not estimate native N from the
    // CUDA pool formula or pretend auto bounds opaque execution allocations.
    // CPU cache metadata shares the payload's actual host resource. The GPU
    // payload formula is not an upper bound for that combined byte scope.
    // Auto keeps the configured version/batch bounds; an explicit byte cap
    // still accounts for and limits every cache allocation, including metadata.
    const bool native_generated_storage = backend_selection.primary == qwen35::BackendKind::kApple ||
        backend_selection.primary == qwen35::BackendKind::kVulkan;
    if (cache_bytes == 0) cache_bytes = (native_generated_storage || backend_selection.primary == qwen35::BackendKind::kCpu)
        ? SIZE_MAX : selected_cache.backing_bytes;
    else if (!native_generated_storage && selected_cache.backing_bytes > cache_bytes) {
      const auto minimum = plan_for(minimum_residents);
      if (minimum.backing_bytes > cache_bytes)
        throw std::runtime_error("--cache-bytes cannot hold the configured active sequences: requires " +
            std::to_string(minimum.backing_bytes) + " shared KV/RNN backing bytes");
      std::size_t low = minimum_residents, high = cached_slots;
      while (low < high) {
        const auto middle = low + (high - low + 1) / 2;
        if (plan_for(middle).backing_bytes <= cache_bytes) low = middle;
        else high = middle - 1;
      }
      selected_cache = plan_for(low);
    }
    const auto cache_slots = selected_cache.resident_sessions;
    const auto active_slots = selected_cache.active_sequences;
    const auto metadata_prepare_seconds = weight_source.metadata_prepare_seconds;
    const auto model_start = qwen35::Clock::now();
    qwen35::Model model(std::move(weight_source), geometry, capacity, selected_cache, layer0_probe, progress, replay, cache_bytes, sampler, cpu_ffn_layers, backend_selection);
    model.configure_draft_policy({max_drafts, min_draft_probability});
    const auto constructed = qwen35::Clock::now();
    if (!trace.empty()) model.trace_routes = trace / "routes";
    if (natural_generation) {
      if (mtp_enabled) return qwen35::run_generation_sessions<qwen35::GenerationMode::kMtp>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans);
      return qwen35::run_generation_sessions<qwen35::GenerationMode::kText>(model,
          generation_schedules, trace, trace_states, startup, constructed, prefix_enabled, token_plans);
    }

    // These owners are declared after Model, including on exception paths.
    // The schedule is consumer policy; every call uses the same Model/Graph chain.
    struct PhaseSummary {
      std::size_t context_before = 0, context_after = 0;
      double seconds = 0, step_sum_seconds = 0;
      std::string name, memory_name;
      qwen35::Clock::time_point start{};
    };
    struct Call { std::size_t phase, local; };
    // Count/idle policy only. Actual shared arena and state-pool backings own
    // the byte reservations; charging this directory per Session would double
    // count shared pages and immutable snapshots.
    hyle::inference::SessionCache resident_cache({cache_slots, (std::numeric_limits<std::size_t>::max)()});
    // Destroy runs (and their real slots) before this directory and Model.
    auto prefix_cache = prefix_enabled ? std::make_unique<qwen35::cache::PrefixCache>(cache_slots) : nullptr;
    struct SessionRun {
      struct RecomputedDecision { std::size_t chunk; std::int32_t previous, actual; };
      std::shared_ptr<qwen35::Session> session;
      hyle::inference::SessionCache::Pin pin;
      std::vector<std::size_t> chunks;
      std::shared_ptr<qwen35::cache::ReplayHistory> history;
      std::vector<PhaseSummary> summaries;
      std::vector<Call> calls;
      std::vector<std::int32_t> generated, consumed;
      std::vector<inference::StepObservation> observations;
      std::vector<inference::PreparationObservation> preparation;
      qwen35::cache::SamplingCursor saved_sampling;
      std::int32_t saved_pending = -1;
      std::filesystem::path trace;
      std::string trace_subdirectory;
      qwen35::Clock::time_point start{}, endpoint{};
      std::size_t visible_tokens = 0, cache_hits = 0, cache_misses = 0;
      std::size_t restored_calls = 0, restored_tokens = 0;
      std::vector<RecomputedDecision> recomputed_decisions;
      std::size_t graph_participations = 0, actual_graph_tokens = 0;
      std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
      std::size_t prefix_sampling_calls = 0;
      double step_sum_seconds = 0, first_token_elapsed_seconds = 0, cache_prepare_seconds = 0;
    };
    std::vector<SessionRun> runs(schedules.size());
    std::size_t peak_committed_versions = 0;
    const auto note_committed_versions = [&] {
      peak_committed_versions = (std::max)(peak_committed_versions, std::size_t(model.snapshot_versions.live));
    };
    std::size_t all_calls = 0;
    for (std::size_t s = 0; s < schedules.size(); ++s) {
      const auto &request = schedules[s];
      auto &run = runs[s];
      run.saved_sampling = {model.sampling_seed, static_cast<std::uint64_t>(s), 0};
      run.summaries.resize(request.phases.size());
      if (prefix_enabled) run.history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
      run.trace_subdirectory = "session" + std::to_string(s) + "_" + request.id;
      run.trace = trace.empty() || !multiple_schedules ? trace : trace / run.trace_subdirectory;
      for (std::size_t i = 0; i < request.phases.size(); ++i) {
        const auto &phase = request.phases[i];
        auto &summary = run.summaries[i];
        summary.name = "round" + std::to_string(phase.turn) +
            (phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? "_prefill" : "_decode");
        summary.memory_name = multiple_schedules ? request.id + "_" + summary.name : summary.name;
        const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
        const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
        for (std::size_t local = 0; local < count; ++local) {
          run.calls.push_back({i, local});
          run.chunks.push_back(is_prompt ? phase.ids.size() : std::size_t{1});
        }
      }
      if (run.calls.size() > (std::numeric_limits<std::size_t>::max)() - all_calls)
        throw std::runtime_error("total schedule call count overflows");
      all_calls += run.calls.size();
      run.generated.reserve(run.calls.size()); run.consumed.reserve(run.calls.size());
      run.observations.resize(run.calls.size());
      if (preparation_diagnostics) run.preparation.resize(run.calls.size());
      for (std::size_t i = 0; i < run.calls.size(); ++i) {
        const auto call = run.calls[i];
        run.observations[i].name = qwen35::benchmark::phase_name(request.phases[call.phase], call.local);
      }
    }
    const auto waves = scheduled ? qwen35::benchmark::make_schedule_waves(schedules, active_slots)
        : std::vector<qwen35::benchmark::ScheduleWave>{};
    std::vector<std::pair<std::size_t, std::size_t>> dispatch;
    dispatch.reserve(all_calls);
    std::vector<std::vector<std::size_t>> query_options(all_calls);
    std::vector<std::map<std::size_t, std::size_t>> published_at(schedules.size());
    std::vector<std::size_t> prior_tokens(schedules.size());
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      dispatch.emplace_back(call.session, call.call);
      prior_tokens[call.session] = qwen35::cache::add(prior_tokens[call.session], call.query_rows);
      const auto &phase = schedules[call.session].phases[call.phase];
      if (phase.kind == qwen35::benchmark::PhaseKind::kPrompt || call.local + 1 == phase.ids.size())
        published_at[call.session].emplace(prior_tokens[call.session], w);
    }
    std::fill(prior_tokens.begin(), prior_tokens.end(), 0);
    std::size_t restore_call_bound = 0;
    for (std::size_t w = 0; w < waves.size(); ++w) for (const auto &call : waves[w].calls) {
      auto &before = prior_tokens[call.session];
      const auto target = qwen35::cache::add(before, call.query_rows);
      if (call.kind == qwen35::benchmark::PhaseKind::kPrompt) {
        if (schedules.size() > cache_slots) restore_call_bound = qwen35::cache::add(restore_call_bound, before);
        if (prefix_enabled) for (std::size_t candidate = 0; candidate < token_plans.size(); ++candidate)
          for (const auto boundary : token_plans[candidate]->checkpoints) {
            const auto producer = published_at[candidate].find(boundary);
            if (boundary < before || boundary > target || producer == published_at[candidate].end() || producer->second >= w ||
                !qwen35::cache::same_prefix(*token_plans[candidate], *token_plans[call.session], boundary)) continue;
            query_options[call.ordinal].push_back(target - boundary);
          }
      }
      before = target;
    }
    const auto shape_uses = qwen35::benchmark::plan_batch_shape_uses(waves, query_options);
    if (!scheduled) {
      model.prepare_family(planned_chunks);
      if (!layer0_probe.empty()) model.prepare_family(planned_chunks, true);
      model.plan_execution(planned_chunks);
    } else {
      std::map<std::size_t, std::vector<std::size_t>> chunks_by_width;
      for (const auto &shape : shape_uses) chunks_by_width[shape.requests].push_back(shape.query_rows);
      for (const auto &[width, chunks] : chunks_by_width) {
        model.prepare_family(chunks, false, width);
        if (!layer0_probe.empty()) model.prepare_family(chunks, true, width);
      }
      if (!multiple_schedules && !prefix_enabled) model.plan_execution(runs.front().chunks);
      else for (const auto &shape : shape_uses)
        model.plan_replay_uses(shape.query_rows, shape.requests,
            qwen35::cache::add(shape.max_wave_calls, shape.requests == 1 ? restore_call_bound : 0));
    }
    struct BatchObservation {
      qwen35::benchmark::PhaseKind kind;
      std::size_t tokens;
      bool restore;
      double seconds;
      hyle::container::SmallVector<std::size_t, 4> sessions;
    };
    struct WaveObservation {
      int kind;
      std::size_t outputs = 0, logical_tokens = 0, actual_tokens = 0, graph_calls = 0;
      double seconds = 0;
    };
    std::vector<BatchObservation> batches;
    batches.reserve(all_calls);
    std::vector<WaveObservation> wave_observations;
    wave_observations.reserve(waves.size());
    std::size_t executed_graph_calls = 0, executed_graph_tokens = 0, max_batch_requests = 0;
    const auto record_batch = [&](qwen35::benchmark::PhaseKind kind, std::size_t tokens,
                                   std::span<const std::size_t> members, bool restore, double seconds) {
      BatchObservation value{kind, tokens, restore, seconds, {}};
      if (!value.sessions.try_reserve(members.size())) throw std::bad_alloc{};
      for (auto s : members) value.sessions.push_back(s);
      batches.push_back(std::move(value));
      executed_graph_calls = qwen35::cache::add(executed_graph_calls, 1);
      executed_graph_tokens = qwen35::cache::add(executed_graph_tokens, qwen35::cache::multiply(tokens, members.size()));
      max_batch_requests = (std::max)(max_batch_requests, members.size());
    };
    std::vector<std::size_t> evicted_sessions;
    evicted_sessions.reserve(all_calls);
    const auto make_initial_room = [&] {
      if (model.active_step) throw std::runtime_error("initial admission requires retired sessions");
      const std::array selected{qwen35::version_holder(nullptr)};
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto fits = [&] {
        return hyle::inference::version_admission({std::size_t(model.snapshot_versions.live), cache_slots},
            selected, residents, [&](const auto *head) { return prefix_cache && prefix_cache->retains(head); }).fits;
      };
      while (!fits()) {
        if (!prefix_cache || !prefix_cache->evict_one())
          throw std::runtime_error("prefix/session heads retain the committed version budget before initial admission");
      }
    };
    const auto make_batch_room = [&](std::span<qwen35::Session *const> owners) {
      if (owners.empty() || model.active_step || model.snapshot_versions.live > cache_slots)
        throw std::runtime_error("batch advance requires retired sessions within the committed version budget");
      for (auto *owner : owners) if (!owner || !owner->head || owner->pending || owner->failed || owner->stopping)
        throw std::runtime_error("batch advance contains a stopped, failed or pending session");
      // With prefix reuse disabled, each resident Session has its own sole
      // committed head. Advancing it replaces one version; no directory scan.
      if (!prefix_cache) return;
      const auto selected = owners | std::views::transform([](const auto *owner) {
        return qwen35::version_holder(owner);
      });
      const auto residents = runs | std::views::transform([](const auto &run) {
        return qwen35::version_holder(run.session.get());
      });
      const auto fits = [&] {
        const auto live = std::size_t(model.snapshot_versions.live);
        return hyle::inference::version_admission({live, cache_slots}, selected, residents,
            [&](const auto *head) { return prefix_cache->retains(head); }).fits;
      };
      // Dropping a directory alias need not free a version. Re-read both live
      // inventory and external retainers after EVERY eviction, including full hits.
      while (!fits()) if (!prefix_cache->evict_one())
        throw std::runtime_error("batch commit exceeds N retained snapshots; release a checkpoint/session or increase the cache");
    };
    const auto make_advance_room = [&](qwen35::Session &owner) {
      const std::array owners{&owner}; make_batch_room(owners);
    };
    const auto ensure_resident = [&](std::size_t s, bool defer_restore) {
      auto &run = runs[s];
      if (run.pin) return;
      const auto begin = qwen35::Clock::now();
      const auto admission = resident_cache.admit(s, 0, [&](std::size_t victim) {
        auto &old = runs[victim];
        if (old.pin) return hyle::inference::RetirementObservation{};
        hyle::inference::RetirementObservation retired{true, {}};
        if (old.session) {
          (void)old.session->stop();
          retired = old.session->try_retire();
          if (!retired.retired) return retired;
        }
        old.session.reset(); // Actual state leases retire before quota returns.
        evicted_sessions.push_back(victim);
        return retired;
      });
      check(admission.retirement_status);
      if (!admission.accepted())
        throw std::runtime_error(admission.status == hyle::inference::CacheAdmissionStatus::kRetirementUnproven
            ? "session cache cannot prove retirement; the original owner remains charged"
            : "session cache admission exhausted; reduce --active-sequences or increase --max-sessions/--cache-bytes");
      run.pin = resident_cache.pin(s);
      if (!run.pin) throw std::runtime_error("session cache could not pin the admitted owner");
      if (run.session) ++run.cache_hits;
      else {
        ++run.cache_misses;
        if (!defer_restore) {
          make_initial_room();
          run.session = model.create_session(run.chunks, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute the original token/chunk tape; never resample or replace
          // already returned tokens. A different legal batch reduction may
          // change logits/greedy decisions. The rebuilt real head is authority
          // for new work; old decisions remain diagnostics, not state payload.
          for (std::size_t i = 0; i < run.generated.size(); ++i) {
            const auto call = run.calls[i];
            const auto &phase = schedules[s].phases[call.phase];
            const auto ids = phase.kind == qwen35::benchmark::PhaseKind::kPrompt
                ? std::span<const std::int32_t>(phase.ids)
                : std::span<const std::int32_t>(&phase.ids[call.local], 1);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_call" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != run.generated[i]) run.recomputed_decisions.push_back({i, run.generated[i], next});
            ++run.restored_calls; run.restored_tokens += ids.size();
            ++run.graph_participations; run.actual_graph_tokens += ids.size();
            const std::array members{s};
            record_batch(phase.kind, ids.size(), members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
            note_committed_versions();
          }
          if (run.session->length() != run.visible_tokens)
            throw std::runtime_error("restored cache context differs from retained consumer history");
          if (run.saved_pending >= 0) model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
        }
      }
      (void)model.cache_usage();
      run.cache_prepare_seconds += std::chrono::duration<double>(qwen35::Clock::now() - begin).count();
    };
    // Prefix mode admits at the timed request boundary, before any initial
    // state allocation. A full hit must not create or evaluate an empty state.
    if (scheduled && !prefix_enabled) {
      // Exact nominal buckets are known before admission. Suffix alternatives
      // stay cold metadata until an actual prefix hit selects one.
      if (!multiple_schedules) model.prepare_execution(runs.front().chunks);
      else for (const auto &wave : waves) {
        std::vector<std::size_t> rows;
        rows.reserve(wave.calls.size());
        for (const auto &call : wave.calls) rows.push_back(call.query_rows);
        for (const auto &bucket : qwen35::benchmark::bucket_schedule_wave(wave, rows)) {
          const std::array chunks{bucket.query_rows};
          model.prepare_execution(chunks, bucket.calls.size());
        }
      }
      if (schedules.size() > cache_slots) for (const auto &run : runs)
        model.prepare_execution(run.chunks);
    }
    if (!prefix_enabled) for (std::size_t s = 0; s < (std::min)(active_slots, runs.size()); ++s) ensure_resident(s, false);
    auto session = scheduled ? std::shared_ptr<qwen35::Session>{} : model.create_session(planned_chunks);
    if (session) model.prepare_execution(planned_chunks);
    inference::BatchIndices batch_indices;
    if (scheduled) batch_indices.prepare(active_slots, active_slots);
    std::vector<std::int32_t> generated, consumed;
    if (!scheduled) { generated.reserve(steps); consumed.reserve(steps - 1); }
    const auto ready = qwen35::Clock::now();
    std::cerr << "initialized model_seconds=" << metadata_prepare_seconds +
                  std::chrono::duration<double>(constructed - model_start).count()
              << " shape_prepare_seconds=" << std::chrono::duration<double>(ready - constructed).count()
              << " startup_to_ready_seconds=" << std::chrono::duration<double>(ready - startup).count() << '\n';
    std::cerr << "cache_limits context=" << capacity << " max_sessions=" << max_sessions
              << " active_sequences=" << active_slots << " effective_cache_sessions=" << cache_slots
              << " recurrent_payload_bytes_per_snapshot=" << recurrent_payload
              << " backend=" << model.execution.name()
              << " shared_cache_backing_bytes=" << model.known_cache_backing_bytes
              << " native_execution_private_bytes_unknown=" << model.execution.native_generated_storage()
              << " cache_budget_scope=shared_pools_and_retained_output_backing";
    if (cache_bytes == SIZE_MAX) std::cerr << " cache_requested_byte_limit=none";
    else std::cerr << " cache_requested_byte_limit=" << cache_bytes;
    std::cerr << " physical_vram_limit_known=false\n";
    if (scheduled) {
      qwen35::Clock::time_point journey_start{}, endpoint{};
      struct PreparedCall {
        std::span<const std::int32_t> ids, logical_ids;
        std::size_t before = 0, reused = 0;
        bool used_prefix = false;
        qwen35::Clock::time_point call_start{};
        double complete_seconds = 0;
        std::int32_t next = -1;
      };
      std::vector<PreparedCall> prepared;
      std::vector<std::size_t> actual_rows, members;
      std::vector<qwen35::Session *> owners;
      std::vector<qwen35::EvaluationRequest> requests;
      std::vector<qwen35::EvaluationOutput> outputs;
      std::vector<std::int32_t> next_ids;
      prepared.reserve(active_slots); actual_rows.reserve(active_slots); members.reserve(active_slots);
      owners.reserve(active_slots); requests.reserve(active_slots); outputs.reserve(active_slots); next_ids.reserve(active_slots);
      for (const auto &wave : waves) {
        const auto wave_start = qwen35::Clock::now();
        if (journey_start == qwen35::Clock::time_point{}) journey_start = wave_start;
        const auto graphs_before = executed_graph_calls, tokens_before = executed_graph_tokens;
        WaveObservation wave_observation{wave.calls.front().kind == qwen35::benchmark::PhaseKind::kPrompt ? 0 : 1};
        prepared.clear(); prepared.resize(wave.calls.size()); actual_rows.clear();
        // Resolve the whole wave against PREVIOUS publications. A full hit
        // stays pinned and charged while the other members are evaluated.
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s];
          const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase];
          auto &summary = run.summaries[call.phase];
          if (call.local == 0) {
            summary.context_before = run.visible_tokens;
            std::cerr << "memory_phase=" << summary.memory_name << "_begin\n";
            summary.start = qwen35::Clock::now();
            if (index == 0) run.start = summary.start;
          }
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j];
          work.call_start = qwen35::Clock::now();
          if (is_prompt) ensure_resident(s, prefix_enabled);
          const auto logical_ids = is_prompt ? std::span<const std::int32_t>(phase.ids)
              : std::span<const std::int32_t>(&phase.ids[call.local], 1);
          auto ids = logical_ids;
          const auto before = run.visible_tokens;
          const auto target = before + logical_ids.size();
          std::size_t reused = 0;
          bool used_prefix = false;
        if (prefix_enabled && is_prompt) {
          // An equal-length checkpoint is useful after eviction, but must not
          // replace a resident same-boundary Session and lose its recordings.
          auto hit = prefix_cache->lookup(*token_plans[s], target,
              before + (run.session ? std::size_t{1} : std::size_t{0}), &model, model.math_identity());
          if (hit) {
            if (hit.payload.snapshot->next_id >= geometry.vocabulary)
              throw std::runtime_error("prefix next_id is outside the model vocabulary");
            reused = hit.payload.snapshot->length - before;
            const auto suffix = target - hit.payload.snapshot->length;
            std::vector<std::size_t> remaining;
            remaining.reserve(run.chunks.size() - index);
            if (suffix) remaining.push_back(suffix);
            remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index + 1), run.chunks.end());
            // All fallible host/IO preparation precedes withdrawal of the old
            // Session. These are real original Model slots, not shadow state.
            auto history = qwen35::cache::copy_history(hit.payload.replay, capacity);
            const auto cursor = run.session ? run.session->head->sampling : run.saved_sampling;
            auto adopted = model.create_session(remaining, hit.payload.snapshot, qwen35::GraphRole::kText, cursor, suffix == 0);
            if (model.default_sampler.enabled && !suffix) {
              history->chunks.back().next_id = adopted->head->next_id; ++run.prefix_sampling_calls;
            }
            if (run.session) {
              (void)run.session->stop();
              const auto retired = run.session->try_retire();
              if (retired.retired) run.session.reset();
              check(retired.status);
              if (!retired.retired)
                throw std::runtime_error("prefix adoption cannot retire the original session");
            }
            run.session = std::move(adopted);
            run.history = std::move(history);
            ids = logical_ids.subspan(reused);
            hit.reset(); // No temporary Snapshot owner may hide from advance admission.
            used_prefix = true; ++run.prefix_hits; run.reused_tokens += reused;
            if (ids.empty()) ++run.prefix_full_hits;
          } else ++run.prefix_misses;
        }
        if (prefix_enabled && !run.session) {
          make_initial_room();
          std::vector<std::size_t> remaining;
          remaining.reserve(run.history->chunks.size() + run.chunks.size() - index);
          for (const auto &part : run.history->chunks) remaining.push_back(part.count);
          remaining.insert(remaining.end(), run.chunks.begin() + static_cast<std::ptrdiff_t>(index), run.chunks.end());
          run.session = model.create_session(remaining, {}, qwen35::GraphRole::kText,
              qwen35::cache::SamplingCursor{model.sampling_seed, static_cast<std::uint64_t>(s), 0});
          note_committed_versions();
          // Recompute actual evaluated chunks, including imported prefixes.
          // Keep any old shared history immutable: new checkpoint metadata
          // must describe the REAL rebuilt head and its recomputed decisions.
          auto rebuilt_history = std::make_shared<qwen35::cache::ReplayHistory>(capacity);
          for (std::size_t i = 0; i < run.history->chunks.size(); ++i) {
            const auto &part = run.history->chunks[i];
            const auto restored_ids = std::span<const std::int32_t>(part.input->ids).subspan(part.first, part.count);
            inference::StepObservation restored;
            const auto name = "restore" + std::to_string(run.cache_misses) + "_chunk" + std::to_string(i);
            const auto restore_start = qwen35::Clock::now();
            make_advance_room(*run.session);
            const auto next = qwen35::evaluate(model, *run.session, restored_ids, name,
                run.trace.empty() ? std::filesystem::path{} : run.trace / "restores", false, nullptr, nullptr,
                &restored, nullptr, multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{});
            if (next != part.next_id) run.recomputed_decisions.push_back({i, part.next_id, next});
            rebuilt_history->append(part.input, part.first, part.count, next);
            ++run.restored_calls; run.restored_tokens += restored_ids.size();
            ++run.graph_participations; run.actual_graph_tokens += restored_ids.size();
            const std::array restored_members{s};
            record_batch(phase.kind, restored_ids.size(), restored_members, true,
                std::chrono::duration<double>(qwen35::Clock::now() - restore_start).count());
          }
          if (run.session->length() != before)
            throw std::runtime_error("actual-chunk recompute differs from the retained logical boundary");
          if (run.saved_pending >= 0) {
            model.restore_pending_boundary(*run.session, run.saved_sampling, run.saved_pending);
            if (model.default_sampler.enabled && !rebuilt_history->chunks.empty())
              rebuilt_history->chunks.back().next_id = run.saved_pending;
          }
          run.history = std::move(rebuilt_history);
        }
          work.ids = ids; work.logical_ids = logical_ids;
          work.before = before; work.reused = reused; work.used_prefix = used_prefix;
          actual_rows.push_back(ids.size());
          wave_observation.logical_tokens += logical_ids.size();
          if (wave_observation.kind != (is_prompt ? 0 : 1)) wave_observation.kind = -1;
          if (ids.empty()) {
            work.next = run.session->head->next_id;
            run.endpoint = qwen35::Clock::now();
            auto &observation = run.observations[index];
            observation.graph_executed = false; observation.graph_input_tokens = 0;
            observation.visible = run.session->length(); observation.next = work.next;
            observation.scheduled_progress = progress;
          }
        }
        // Validate the entire resolved wave before any bucket can submit.
        // A later incompatible member must not turn an old preflight error
        // into a partially committed wave.
        if (actual_rows.size() != wave.calls.size())
          throw std::runtime_error("batch wave query lengths do not match its calls");
        for (std::size_t j = 0; j < wave.calls.size(); ++j)
          if (actual_rows[j] > wave.calls[j].query_rows)
            throw std::runtime_error("batch wave actual query exceeds its logical input");
        batch_indices.reset(wave.calls.size());
        for (std::size_t first = 0; first < wave.calls.size(); ++first) {
          if (batch_indices.completed(first) || !actual_rows[first]) continue;
          const auto kind = wave.calls[first].kind;
          const auto query_rows = actual_rows[first];
          const auto selected = batch_indices.select(first, active_slots, [&](std::size_t j) {
            return wave.calls[j].kind == kind && actual_rows[j] == query_rows;
          });
          requests.clear(); outputs.clear(); owners.clear(); members.clear(); next_ids.resize(selected.size());
          for (const auto j : selected) {
            const auto &call = wave.calls[j]; auto &run = runs[call.session]; auto &work = prepared[j];
            requests.push_back({run.session, work.ids});
            owners.push_back(run.session.get()); members.push_back(call.session);
            outputs.push_back(qwen35::evaluation_output(model, run.session,
                {run.observations[call.call].name, &run.trace, trace_states,
                    multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}},
                &work.complete_seconds, &run.endpoint, &run.observations[call.call],
                preparation_diagnostics ? &run.preparation[call.call] : nullptr));
          }
          const auto batch_start = qwen35::Clock::now();
          make_batch_room(owners);
          inference::PreparationObservation preparation;
          model.prepare_evaluations(requests, preparation_diagnostics ? &preparation : nullptr);
          if (preparation_diagnostics) for (const auto j : selected)
            runs[wave.calls[j].session].preparation[wave.calls[j].call] = preparation;
          model.submit_evaluations();
          model.finish_evaluations(outputs, next_ids, [&](std::size_t at, const qwen35::CompletedEvaluationView &view) {
            const auto &call = wave.calls[selected[at]]; const auto &run = runs[call.session];
            qwen35::observe_completed_evaluation(model, outputs[at],
                {run.observations[call.call].name, &run.trace, trace_states,
                    multiple_schedules ? std::string_view(run.trace_subdirectory) : std::string_view{}}, view);
          });
          const auto batch_endpoint = runs[members.front()].endpoint;
          record_batch(kind, query_rows, members, false,
              std::chrono::duration<double>(batch_endpoint - batch_start).count());
          for (std::size_t at = 0; at < selected.size(); ++at) {
            const auto j = selected[at]; const auto &call = wave.calls[j];
            auto &run = runs[call.session]; auto &work = prepared[j]; work.next = next_ids[at];
            ++run.graph_participations; run.actual_graph_tokens += work.ids.size();
            if (prefix_enabled) run.history->append(token_plans[call.session], work.before + work.reused, work.ids.size(), work.next);
          }
          // No temporary strong Session owner may retain an old head across
          // next-wave adoption or eviction outside the charged directory.
          requests.clear(); outputs.clear();
          if (model.snapshot_versions.live > cache_slots)
            throw std::runtime_error("committed batch versions exceed the finite cache budget");
          note_committed_versions();
          batch_indices.mark_selected(selected.size());
        }
        for (std::size_t j = 0; j < wave.calls.size(); ++j) {
          const auto &scheduled_call = wave.calls[j];
          const auto s = scheduled_call.session, index = scheduled_call.call;
          auto &run = runs[s]; const auto call = run.calls[index];
          const auto &phase = schedules[s].phases[call.phase]; auto &summary = run.summaries[call.phase];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          auto &work = prepared[j]; auto &observation = run.observations[index];
          const auto logical_ids = work.logical_ids;
          const auto call_start = work.call_start;
          const auto next = work.next; const auto reused = work.reused; const auto used_prefix = work.used_prefix;
          // Per-session latency includes time waiting for its batch. Aggregate
          // throughput uses the wave wall once, never the sum of these spans.
          auto complete_seconds = std::chrono::duration<double>(run.endpoint - call_start).count();
          observation.step_seconds = complete_seconds;
          if (prefix_enabled) {
            observation.input_tokens = logical_ids.size();
            observation.prefix_hit = used_prefix; observation.reused_tokens = reused;
            if (model.snapshot_versions.live > cache_slots)
              throw std::runtime_error("committed prefix/session versions exceed the finite cache budget");
            if (!observation.graph_executed) {
              observation.seconds = complete_seconds;
              const auto write_hit_trace = [&](const std::filesystem::path &base) {
                if (base.empty()) return;
                const auto directory = base / observation.name;
                std::filesystem::create_directories(directory);
                std::ofstream output(directory / "cache-hit.json");
                output << "{\"prefix_hit\":true,\"prompt_graph_calls\":0,\"frontier_sampling_graph_calls\":" << (sampler.enabled ? 1 : 0)
                       << ",\"graph_calls\":" << (sampler.enabled ? 1 : 0) << ",\"tensor_payload_available\":false,\"visible_tokens\":"
                       << run.session->length() << ",\"next_id\":" << next << "}\n";
                if (!output) throw std::runtime_error("cannot write prefix-hit metadata trace");
              };
              write_hit_trace(run.trace);
              if (!layer0_probe.empty()) write_hit_trace(multiple_schedules ? layer0_probe / run.trace_subdirectory : layer0_probe);
            }
          }
          note_committed_versions();
          endpoint = (std::max)(endpoint, run.endpoint);
          run.visible_tokens = run.session->length();
          summary.step_sum_seconds += complete_seconds; run.step_sum_seconds += complete_seconds;
          run.generated.push_back(next);
          run.saved_sampling = run.session->head->sampling; run.saved_pending = next;
          if (!is_prompt) run.consumed.push_back(phase.ids[call.local]);
          if (index == 0) run.first_token_elapsed_seconds = std::chrono::duration<double>(run.endpoint - startup).count();
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 == count) {
            summary.context_after = run.visible_tokens;
            summary.seconds = std::chrono::duration<double>(run.endpoint - summary.start).count();
            std::cerr << "memory_phase=" << summary.memory_name << "_end\n";
            // Terminal prompts are real completed requests too.
            if (!is_prompt || call.phase + 1 == schedules[s].phases.size()) run.pin.reset();
          }
        }
        // Publication is a wave boundary. Same-wave calls never consume each
        // other's new checkpoint, matching the dependency and cold shape plan.
        if (prefix_enabled) for (const auto &call : wave.calls) {
          const auto &phase = schedules[call.session].phases[call.phase];
          const auto count = phase.kind == qwen35::benchmark::PhaseKind::kPrompt ? std::size_t{1} : phase.ids.size();
          if (call.local + 1 != count) continue;
          auto &run = runs[call.session];
          const qwen35::cache::ReplayPrefix history{run.history, run.history->chunks.size(), run.history->length};
          (void)prefix_cache->publish(token_plans[call.session], run.session->head, history);
        }
        wave_observation.outputs = wave.calls.size();
        wave_observation.actual_tokens = executed_graph_tokens - tokens_before;
        wave_observation.graph_calls = executed_graph_calls - graphs_before;
        endpoint = qwen35::Clock::now(); // Include all result/checkpoint publication once.
        wave_observation.seconds = std::chrono::duration<double>(endpoint - wave_start).count();
        wave_observations.push_back(wave_observation);
      }
      for (std::size_t s = 0; s < runs.size(); ++s) {
        const auto &run = runs[s];
        if (multiple_schedules) std::cerr << "session_steps id=" << schedules[s].id << '\n';
        for (const auto &observation : run.observations) hyle::engine::reporting::write_step_observation(std::cerr, observation);
        if (preparation_diagnostics) for (std::size_t i = 0; i < run.observations.size(); ++i)
          if (run.observations[i].graph_executed)
            hyle::engine::reporting::write_preparation_observation(std::cerr, run.observations[i], run.preparation[i]);
      }
      // Define after the execution observations. Call once in the single
      // schedule's top-level object, or once in the multi-schedule top-level
      // object; do not call for each nested Session. Existing output sets
      // setprecision(17). Session IDs were validated as ASCII [A-Za-z0-9_-].
      const auto print_execution = [&] {
        std::cout << ",\"actual_graph_calls\":" << executed_graph_calls + model.frontier_sampling_graph_calls
                  << ",\"actual_graph_tokens\":" << executed_graph_tokens
                  << ",\"max_batch_requests\":" << max_batch_requests
                  << ",\"batches\":[";
        for (std::size_t i = 0; i < batches.size(); ++i) {
          const auto &batch = batches[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (batch.kind == qwen35::benchmark::PhaseKind::kPrompt ? "prefill" : "decode")
                    << "\",\"restore\":" << (batch.restore ? "true" : "false")
                    << ",\"query_rows\":" << batch.tokens
                    << ",\"requests\":" << batch.sessions.size()
                    << ",\"seconds\":" << batch.seconds << ",\"session_ids\":[";
          for (std::size_t j = 0; j < batch.sessions.size(); ++j)
            std::cout << (j ? "," : "") << '"' << schedules[batch.sessions[j]].id << '"';
          std::cout << "]}";
        }
        std::cout << "],\"waves\":[";
        for (std::size_t i = 0; i < wave_observations.size(); ++i) {
          const auto &wave = wave_observations[i];
          std::cout << (i ? "," : "") << "{\"kind\":\""
                    << (wave.kind == 0 ? "prefill" : wave.kind == 1 ? "decode" : "mixed")
                    << "\",\"outputs\":" << wave.outputs
                    << ",\"logical_tokens\":" << wave.logical_tokens
                    << ",\"actual_tokens\":" << wave.actual_tokens
                    << ",\"graph_calls\":" << wave.graph_calls
                    << ",\"seconds\":" << wave.seconds << '}';
        }
        std::cout << ']';
      };
      const auto print_result = [&](std::size_t s) {
        const auto &run = runs[s];
        const auto &phases = schedules[s].phases;
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << model.weight_inventory_report() << ",\"visible_tokens\":"
                  << run.visible_tokens << ",\"generated_ids\":[";
        for (std::size_t i = 0; i < run.generated.size(); ++i) std::cout << (i ? "," : "") << run.generated[i];
        std::cout << "],\"fixed_decode\":true,\"decode_input_ids\":[";
        for (std::size_t i = 0; i < run.consumed.size(); ++i) std::cout << (i ? "," : "") << run.consumed[i];
        std::cout << "],\"schedule\":true,\"phase_count\":" << phases.size()
                  << ",\"turn_count\":" << phases.back().turn
                  << ",\"processed_tokens\":" << run.visible_tokens
                  << ",\"generated_tokens\":" << run.generated.size()
                  << ",\"session_seconds\":" << std::chrono::duration<double>(run.endpoint - run.start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(run.endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << run.step_sum_seconds
                  << ",\"cache_hits\":" << run.cache_hits << ",\"cache_misses\":" << run.cache_misses
                  << ",\"restored_calls\":" << run.restored_calls << ",\"restored_tokens\":" << run.restored_tokens
                  << ",\"restore_contract\":\"recompute_original_inputs\""
                  << ",\"cache_prepare_seconds\":" << run.cache_prepare_seconds
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_hits\":" << run.prefix_hits << ",\"prefix_misses\":" << run.prefix_misses
                  << ",\"prefix_full_hits\":" << run.prefix_full_hits << ",\"reused_tokens\":" << run.reused_tokens
                  << ",\"graph_participations\":" << run.graph_participations + run.prefix_sampling_calls
                  << ",\"prefix_sampling_graph_calls\":" << run.prefix_sampling_calls
                  << ",\"sampling_stream\":" << run.saved_sampling.stream
                  << ",\"sampling_draw_cursor\":" << run.saved_sampling.draw
                  << ",\"graph_input_tokens\":" << run.actual_graph_tokens
                  << model.cache_report()
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false");
        std::cout << ",\"recomputed_decision_changes\":[";
        for (std::size_t i = 0; i < run.recomputed_decisions.size(); ++i) {
          const auto &change = run.recomputed_decisions[i];
          std::cout << (i ? "," : "") << "{\"chunk\":" << change.chunk
                    << ",\"previous\":" << change.previous << ",\"actual\":" << change.actual << '}';
        }
        std::cout << ']';
        if (multiple_schedules) std::cout << ",\"session_id\":\"" << schedules[s].id
            << "\",\"trace_subdirectory\":\"" << run.trace_subdirectory << '"';
        if (!multiple_schedules) print_execution();
        std::cout << ",\"phases\":[";
        for (std::size_t i = 0; i < phases.size(); ++i) {
          const auto &phase = phases[i]; const auto &summary = run.summaries[i];
          const bool is_prompt = phase.kind == qwen35::benchmark::PhaseKind::kPrompt;
          const auto count = is_prompt ? std::size_t{1} : phase.ids.size();
          std::cout << (i ? "," : "") << "{\"turn\":" << phase.turn
                    << ",\"kind\":\"" << (is_prompt ? "prefill" : "decode") << "\",\"name\":\"" << summary.name
                    << "\",\"input_tokens\":" << phase.ids.size() << ",\"calls\":" << count << ",\"generated_tokens\":" << count
                    << ",\"context_before\":" << summary.context_before << ",\"context_after\":" << summary.context_after
                    << ",\"seconds\":" << summary.seconds << ",\"step_sum_seconds\":" << summary.step_sum_seconds << '}';
        }
        std::cout << "]}";
      };
      std::cerr << "first_token_elapsed_seconds=" << runs.front().first_token_elapsed_seconds << '\n';
      std::cout << std::setprecision(17);
      if (!multiple_schedules) print_result(0);
      else {
        std::size_t processed = 0, graph_participations = 0;
        std::size_t prefix_hits = 0, prefix_misses = 0, prefix_full_hits = 0, reused_tokens = 0;
        double step_sum = 0;
        for (const auto &run : runs) {
          processed += run.visible_tokens; step_sum += run.step_sum_seconds;
          graph_participations += run.graph_participations + run.prefix_sampling_calls;
          prefix_hits += run.prefix_hits; prefix_misses += run.prefix_misses;
          prefix_full_hits += run.prefix_full_hits; reused_tokens += run.reused_tokens;
        }
        std::cout << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
                  << '"' << model.weight_inventory_report() << ",\"session_count\":" << runs.size()
                  << ",\"execution\":\"equal_T_batched\",\"processed_tokens\":" << processed
                  << ",\"generated_tokens\":" << all_calls
                  << ",\"graph_participations\":" << graph_participations
                  << ",\"prefix_hits\":" << prefix_hits << ",\"prefix_misses\":" << prefix_misses
                  << ",\"prefix_full_hits\":" << prefix_full_hits << ",\"reused_tokens\":" << reused_tokens
                  << ",\"interleaved_seconds\":" << std::chrono::duration<double>(endpoint - journey_start).count()
                  << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
                  << ",\"step_sum_seconds\":" << step_sum
                  << ",\"max_cached_sessions\":" << max_sessions << ",\"active_sequences\":" << active_slots
                  << ",\"effective_cache_sessions\":" << cache_slots
                  << ",\"cache_recurrent_payload_bytes_per_snapshot\":" << recurrent_payload
                  << model.cache_report()
                  << ",\"cache_page_rows\":" << qwen35::cache::Plan::page_rows
                  << ",\"cache_max_physical_pages\":" << selected_cache.max_pages
                  << ",\"cache_live_snapshot_versions\":" << model.snapshot_versions.live
                  << ",\"cache_peak_committed_versions\":" << peak_committed_versions
                  << ",\"cache_committed_version_limit\":" << cache_slots
                  << ",\"cache_retained_sessions\":" << resident_cache.retained_sessions()
                  << ",\"prefix_cache\":\"" << (prefix_enabled ? "auto" : "off") << '"'
                  << ",\"prefix_entry_capacity\":" << (prefix_enabled ? cache_slots : 0)
                  << ",\"prefix_entries\":" << (prefix_cache ? prefix_cache->size() : 0)
                  << ",\"prefix_evictions\":" << (prefix_cache ? prefix_cache->evictions() : 0)
                  << ",\"diagnostic\":" << (preparation_diagnostics || !trace.empty() || !layer0_probe.empty() ? "true" : "false")
                  << ",\"logical_dispatch_session_ids\":[";
        for (std::size_t i = 0; i < dispatch.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[dispatch[i].first].id << '"';
        std::cout << "],\"evicted_session_ids\":[";
        for (std::size_t i = 0; i < evicted_sessions.size(); ++i)
          std::cout << (i ? "," : "") << '"' << schedules[evicted_sessions[i]].id << '"';
        std::cout << ']';
        print_execution();
        std::cout << ",\"sessions\":[";
        for (std::size_t s = 0; s < runs.size(); ++s) { if (s) std::cout << ','; print_result(s); }
        std::cout << "]}";
      }
      std::cout << '\n';
      return 0;
    }
    std::cerr << "measure_memory name=ready_before_first\nmemory_phase=generation_prefill_begin\n";
    double prefill_step_seconds = 0, continuation_step_seconds = 0, continuation_seconds = 0, decode_step_sum_seconds = 0;
    qwen35::Clock::time_point prefill_endpoint{};
    const auto prefill_start = qwen35::Clock::now();
    auto next = qwen35::evaluate(model, *session, prompt, "prefill", trace, trace_states,
        &prefill_step_seconds, &prefill_endpoint);
    auto endpoint = prefill_endpoint;
    std::cerr << "first_token_elapsed_seconds=" << std::chrono::duration<double>(prefill_endpoint - startup).count() << '\n';
    std::cerr << "memory_phase=generation_prefill_end\nmemory_phase=generation_decode_begin\n";
    if (!extra.empty()) {
      const auto continuation_start = qwen35::Clock::now();
      next = qwen35::evaluate(model, *session, extra, "continuation", trace, trace_states,
          &continuation_step_seconds, &endpoint);
      continuation_seconds = std::chrono::duration<double>(endpoint - continuation_start).count();
    }
    generated.push_back(next);
    for (std::size_t step = 1; step < steps && (!forced.empty() || !qwen35::is_end_of_generation(next)); ++step) {
      const auto previous = forced.empty() ? next : forced[step - 1];
      double step_seconds = 0;
      next = qwen35::evaluate(model, *session, std::span(&previous, 1), "decode" + std::to_string(step), trace, trace_states,
          &step_seconds, &endpoint);
      decode_step_sum_seconds += step_seconds;
      consumed.push_back(previous); generated.push_back(next);
    }
    std::cerr << "memory_phase=generation_decode_end\n";
    const auto prefill_seconds = std::chrono::duration<double>(prefill_endpoint - prefill_start).count();
    const auto decode_seconds = std::chrono::duration<double>(endpoint - prefill_endpoint).count();
    const auto session_seconds = std::chrono::duration<double>(endpoint - prefill_start).count();
    const auto graph_calls = 1 + std::size_t(!extra.empty()) + consumed.size();
    std::cout << std::setprecision(17)
              << "{\"source_id\":\"" << model.source_identity << "\",\"score_dtype\":\"F32\",\"score_policy\":\"" << model.score_precision_policy()
              << "\",\"mtp\":false" << model.weight_inventory_report() << model.cache_report()
              << ",\"sampling_stream\":" << session->head->sampling.stream
              << ",\"sampling_draw_cursor\":" << session->head->sampling.draw
              << ",\"visible_tokens\":" << session->length() << ",\"generated_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) std::cout << (i ? "," : "") << generated[i];
    std::cout << "],\"fixed_decode\":" << (forced.empty() ? "false" : "true") << ",\"decode_input_ids\":[";
    for (std::size_t i = 0; i < consumed.size(); ++i) std::cout << (i ? "," : "") << consumed[i];
    std::cout << "],\"prefill_seconds\":" << prefill_seconds << ",\"decode_seconds\":" << decode_seconds
              << ",\"session_seconds\":" << session_seconds
              << ",\"startup_total_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_endpoint_seconds\":" << std::chrono::duration<double>(endpoint - startup).count()
              << ",\"startup_to_ready_seconds\":" << std::chrono::duration<double>(ready - startup).count()
              << ",\"first_token_elapsed_seconds\":" << std::chrono::duration<double>(prefill_endpoint - startup).count()
              << ",\"prefill_step_seconds\":" << prefill_step_seconds
              << ",\"continuation_seconds\":" << continuation_seconds << ",\"continuation_step_seconds\":" << continuation_step_seconds
              << ",\"decode_step_sum_seconds\":" << decode_step_sum_seconds
              << ",\"step_sum_seconds\":" << prefill_step_seconds + continuation_step_seconds + decode_step_sum_seconds
              << ",\"prefill_tokens_per_second\":" << (prefill_seconds > 0 ? prompt.size() / prefill_seconds : 0)
              << ",\"decode_output_tokens_per_second\":" << (decode_seconds > 0 ? consumed.size() / decode_seconds : 0)
              << ",\"output_tokens_per_second\":" << (session_seconds > 0 ? generated.size() / session_seconds : 0)
              << ",\"requested_steps\":" << steps << ",\"generated_tokens\":" << generated.size()
              << ",\"generated_decode_tokens\":" << consumed.size() << ",\"actual_graph_calls\":" << graph_calls
              << ",\"prefill_graph_calls\":1,\"decode_graph_calls\":" << consumed.size()
              << ",\"continuation_graph_calls\":" << std::size_t(!extra.empty())
              << ",\"prefill_target_rows\":" << prompt.size() << ",\"prefill_draft_rows\":0"
              << ",\"continuation_target_rows\":" << extra.size() << ",\"verified_target_rows\":" << consumed.size()
              << ",\"actual_target_rows\":" << prompt.size() + extra.size() + consumed.size() << ",\"actual_draft_rows\":0"
              << ",\"committed_target_rows\":" << session->length()
              << ",\"committed_target_decode_rows\":" << consumed.size()
              << ",\"draft_rows\":0,\"draft_rounds\":0,\"agreements\":0,\"consumed_drafts\":0"
              << ",\"decode_seconds_includes_continuation\":" << (extra.empty() ? "false" : "true")
              << ",\"diagnostic\":" << (!trace.empty() || !layer0_probe.empty() ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen3.5 text consumer: " << error.what() << '\n'; return 1;
  }
}
