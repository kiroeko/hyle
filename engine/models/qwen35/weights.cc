// Cold Qwen binding and original-source loading. Compile this same source
// under each consumer's existing backend capability/SDK contract.
#include "weights.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <type_traits>

#include "../../inference/backend.h"
#include "fp8_weight_import.h"
#include "nf4_weight_import.h"
#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/util/scope_guard.h"

namespace qwen35 {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using hyle::inference::Backend;
using Type = c::DataType;
using Clock = std::chrono::steady_clock;

void Tensor::freeze(const Backend &backend) {
    if (!writer) { check(backend.freeze(buffer, ready)); return; }
    auto owner = take(backend.storage_owner_for(storage().resource()));
    ready = owner ? take(owner.freeze(pool, std::move(writer)))
                  : take(pool.freeze(std::move(writer)));
  }
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

mem::MemoryResourceRef weight_resource(const Backend &backend, bool host) {
  if (!host) return backend.device();
  if (backend.has_cpu_assist()) return backend.host_device();
  throw std::runtime_error("CPU FFN placement requires a configured CPU plus GPU endpoint");
}
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
namespace {
// Apple/Vulkan copies require complete four-byte words. Keep the same owned
// carrier on every route; the file range and tensor descriptor remain logical.
// Only the stream initializes the extra tail, without reading a neighbour role.
std::size_t streamed_weight_capacity(std::size_t bytes) {
  return cache::add(bytes, (4 - bytes % 4) % 4);
}
} // namespace
// The complete model wrapper validates its full source inventory first. A
// source-bound component may select an explicit subset of those same roles;
// both paths use this one IO/lease/shape-authentication implementation.
WeightLoadReport load_selected_weight_roles(const prepared::Roles &roles,
    Backend &backend, std::map<std::string, LoadedWeight> &weights,
    std::vector<c::LeasePool> &weight_pools, const FfnPlacement &placement,
    const c::ResourceBudget &budget) {
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
  std::array<std::size_t, 2> largest{}, largest_group{}, source_counts{}, domain_bytes{}, stream_domain_bytes{};
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
    const auto capacity = streamed_weight_capacity(role.bytes);
    stream_domain_bytes[host] = cache::add(stream_domain_bytes[host], capacity);
    sources[{role.source, host}].push_back(i); largest[host] = (std::max)(largest[host], capacity);
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
        .max_active_backing_bytes = cache::add(stream_domain_bytes[domain], slots * staging_per_slot)};
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
      const auto capacity = streamed_weight_capacity(role.bytes);
      Tensor tensor(weight_resource(backend, host), description, false);
      auto pool = take(c::LeasePool::create(weight_resource(backend, host), capacity,
          {.max_blocks = 1, .resource_budget = budget}));
      auto writer = take(pool.allocate(capacity));
      ranges.push_back({.file_offset = role.offset, .destination = std::move(writer),
          .bytes = role.bytes, .physical_bytes = capacity, .description = description});
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
      auto ready = take(group.initialized_storage(0));
      if (ready.size() != ranges[index].physical_bytes) throw std::runtime_error("incomplete initialized weight storage");
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
    const ModelGeometry &geometry, WeightSelection selection,
    std::size_t cpu_ffn_layers) {
  if (selection != WeightSelection::kText && selection != WeightSelection::kTextAndMtp)
    throw std::runtime_error("invalid weight selection");
  const auto roles = read_weight_roles(manifest, selection == WeightSelection::kTextAndMtp);
  if (roles.geometry != geometry) throw std::runtime_error("source geometry changed before weight loading");
  const FfnPlacement placement(roles, cpu_ffn_layers);
  return load_selected_weight_roles(roles, backend, weights, weight_pools, placement);
}

} // namespace qwen35
