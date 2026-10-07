#include "torch_cpu_program.h"
#include "hyle/core/environment.h"

#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
// c10 2.8 irange's inlined iterator produces C4702 under MSVC /O2, even with
// SYSTEM headers. Keep this SDK diagnostic local to its adapter translation unit.
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#include <array>
#include <cstdint>
#include <new>
#include <mutex>
#include <string_view>
#include <utility>

#include <ATen/core/grad_mode.h>
#include <c10/util/Exception.h>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "native_plan.h"
#include "torchao_cpu.h"
#include "bitsandbytes_cpu.h"
#include "awq_cpu.h"
#include "compressed_tensors_cpu.h"
#include "mlx_cpu_quant.h"

namespace hyle::compute::native::internal {
namespace {
enum class Kind { kDecode, kLinear, kEncodedMatMul, kGrouped8, kGrouped4, kRoundtrip,
                  kArtifactDecode, kMlx, kCausalConv };
struct Shape final {
  Kind kind{};
  std::int64_t e = 1, n = 0, k = 0, tokens = 0, slots = 1, activations = 1;
  float floor = 1.0e-10F;
  bool power = false;
  at::ScalarType output = at::kFloat;
  model_ops::ArtifactWeightDecode artifact{};
  model_ops::MlxQuantized mlx{};
};
struct Program final {
  using Run = Status (*)(const Program &, std::span<const std::byte *const>, std::byte *) noexcept;
  Shape shape{};
  std::array<TensorDesc, 6> inputs{};
  TensorDesc output{};
  std::size_t input_count = 0;
  Run run = nullptr;
};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = true,
    .supports_concurrent_invocations = true,
    .capture_compatible = false};

Status translate_error(const c10::Error &error) noexcept {
  const std::string_view message = error.what_without_backtrace();
  // Pinned c10 alloc_cpu uses a plain Error for CPU allocation failure.
  // Match its two allocation diagnostics; other ATen errors remain Internal.
  if (message.find("DefaultCPUAllocator: not enough memory:") != message.npos ||
      message.find("DefaultCPUAllocator: can't allocate memory:") != message.npos)
    return Status::ResourceExhausted(message);
  return Status::Internal(message);
}

Result<Shape> select(const Operation &op, std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  if (!is_torch_cpu_operation(op, inputs))
    return std::unexpected(Status::Unimplemented("original ATen CPU low precision operation is unavailable"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(op, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return std::unexpected(Status::InvalidArgument("ATen CPU output differs from inference"));
  if (outputs[0].required_bytes().value() > static_cast<std::uint64_t>(INT64_MAX))
    return std::unexpected(Status::OutOfRange("ATen CPU output exceeds native int64 capacity"));
  for (const auto &desc : inputs)
    if (desc.required_bytes().value() > static_cast<std::uint64_t>(INT64_MAX))
      return std::unexpected(Status::OutOfRange("ATen CPU carrier exceeds native int64 capacity"));
  Shape s;
  s.output = outputs[0].data_type() == DataType::kBFloat16 ? at::kBFloat16 :
      (outputs[0].data_type() == DataType::kFloat16 ? at::kHalf : at::kFloat);
  if (op.custom_as<model_ops::CausalConv1d>()) {
    s.kind = Kind::kCausalConv;
    s.n = inputs[0].dimensions()[2]; s.k = inputs[1].dimensions()[1];
    for (const auto &desc : inputs)
      if (desc.element_count().value() > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
        return std::unexpected(Status::OutOfRange("ATen CPU convolution F32 temporary capacity overflows"));
    if (outputs[0].element_count().value() > static_cast<std::uint64_t>(INT64_MAX) / sizeof(float))
      return std::unexpected(Status::OutOfRange("ATen CPU convolution result capacity overflows"));
  } else if (const auto mlx = op.custom_as<model_ops::MlxQuantized>()) {
    s.kind = Kind::kMlx;s.mlx = *mlx;
    const auto &w = inputs[mlx->kind == model_ops::MlxQuantizedKind::kEmbedding ? 0 : 1];
    s.k = w.dimensions().back() * (mlx->format == model_ops::MlxWeightFormat::kMxfp8 ? 4 : 8);
    s.n = w.dimensions()[w.rank() - 2];
    s.e = mlx->kind == model_ops::MlxQuantizedKind::kGrouped ? w.dimensions()[0] : 1;
    // ATen's current-group codes/decoded matrices require more bytes than the
    // caller's final F16/BF16 output. Check that bounded native capacity too.
    const auto count = outputs[0].element_count().value();
    const std::uint64_t factor = mlx->kind == model_ops::MlxQuantizedKind::kEmbedding
        ? 32 : std::uint64_t{32} * mlx->group_size;
    if (count > static_cast<std::uint64_t>(INT64_MAX) / factor)
      return std::unexpected(Status::OutOfRange("MLX selected-group native temporary capacity overflows"));
  } else if (const auto artifact = op.custom_as<model_ops::ArtifactWeightDecode>()) {
    s.kind = Kind::kArtifactDecode; s.artifact = *artifact;
    s.n = outputs[0].dimensions()[0]; s.k = outputs[0].dimensions()[1];
  } else if (op.custom_as<model_ops::Float8WeightDecode>()) {
    s.kind = Kind::kDecode;
    s.k = inputs[0].dimensions()[0]; s.n = inputs[0].dimensions()[1];
  } else if (op.custom_as<model_ops::BlockQuantizationRoundtrip>()) {
    s.kind = Kind::kRoundtrip; s.power = true; s.floor = 1.0e-4F;
    s.tokens = inputs[0].dimensions()[0]; s.k = inputs[0].dimensions()[1];
  } else if (op.custom_as<model_ops::GroupedFloat8Linear>() ||
             op.custom_as<model_ops::GroupedFloat4Linear>()) {
    const auto grouped = op.custom_as<model_ops::GroupedFloat8Linear>();
    s.kind = grouped ? Kind::kGrouped8 : Kind::kGrouped4;
    s.floor = grouped ? grouped->absmax_floor : 1.0e-4F;
    s.power = !grouped;
    const auto w = inputs[0].dimensions(), a = inputs[1].dimensions();
    s.e = w[0]; s.n = w[1]; s.k = w[2]; s.tokens = a[0]; s.activations = a[1];
    s.slots = inputs[2].dimensions()[1];
  } else {
    const auto linear = op.custom_as<model_ops::Float8BlockLinear>();
    s.kind = linear ? Kind::kLinear : Kind::kEncodedMatMul;
    if (linear) { s.floor = linear->absmax_floor;
      s.power = linear->scale_rounding == model_ops::Float8ScaleRounding::kPowerOfTwo; }
    s.tokens = inputs[0].dimensions()[0]; s.k = inputs[0].dimensions()[1];
    s.n = inputs[1].dimensions()[1];
    auto weight = float8_weight_description(s.k, s.n, Float8ScaleMode::kFloat32Block128);
    if (!weight || inputs[1] != *weight)
      return std::unexpected(Status::Unimplemented("ATen CPU requires canonical block128 weights"));
    if (!linear) {
      auto activation = float8_activation_description(s.tokens, s.k, Float8ScaleMode::kFloat32Block128);
      if (!activation || inputs[0] != *activation ||
          op.get_if<MatMul>()->accumulation_type != DataType::kFloat32)
        return std::unexpected(Status::Unimplemented("ATen encoded MatMul requires canonical block128 activation and F32 accumulation"));
    }
  }
  const auto product = [](std::int64_t a, std::int64_t b) {
    return math::try_multiply(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b));
  };
  auto rows = product(s.tokens, s.slots);
  auto matrix = product(s.n, s.k);
  auto elements = rows && matrix ? math::try_multiply(*rows, *matrix) : std::nullopt;
  if (!matrix || (s.kind != Kind::kDecode && s.kind != Kind::kArtifactDecode && s.kind != Kind::kMlx && s.kind != Kind::kCausalConv &&
                  (!elements || *elements > INT64_MAX / 32)))
    return std::unexpected(Status::OutOfRange("ATen CPU selected tensor capacity overflows"));
  return s;
}

struct Quantized final { at::Tensor codes, scales; };
Quantized quantize(const at::Tensor &input, const Shape &s) {
  auto data = input.to(at::kFloat).reshape({-1, 128});
  at::Tensor scale;
  if (s.power) {
    // Original TorchAO _to_mx_rceil receives the model's explicit absmax floor.
    auto maximum = data.abs().amax(-1, true).clamp_min(s.floor);
    auto exponent = torchao_cpu::f32_to_e8m0_rceil(maximum * (1.0 / 448.0));
    scale = torchao_cpu::e8m0_scale_to_reciprocal_fp32(exponent).reciprocal();
  } else scale = torchao_cpu::choose_scale_float8(data, s.floor);
  return {torchao_cpu::quantize_affine_float8(data, scale), std::move(scale)};
}
Quantized weight(const at::Tensor &carrier, const Shape &s, const at::Tensor &ids) {
  const bool four = s.kind == Kind::kGrouped4;
  const auto payload = s.e * s.n * (s.k / (four ? 2 : 1));
  auto codes = carrier.slice(0, 0, payload).reshape({s.e, s.n, s.k / (four ? 2 : 1)});
  auto scales = carrier.slice(0, payload).view(at::kFloat).reshape(
      {s.e, four ? s.n : (s.n + 127) / 128, s.k / (four ? 32 : 128)});
  if (ids.defined()) {
    codes = codes.index_select(0, ids).contiguous();
    scales = scales.index_select(0, ids);
  }
  // Retain encoded weights. The original FP4 codec creates several tensors;
  // decoding an entire selected expert batch magnifies its temporary memory.
  if (!four) scales = scales.repeat_interleave(128, 1).slice(1, 0, s.n);
  return {std::move(codes), std::move(scales)};
}
at::Tensor block_dot(Quantized a, Quantized w, const Shape &s) {
  const auto rows = s.tokens * s.slots;
  const auto group = s.kind == Kind::kGrouped4 ? 32 : 128;
  const auto blocks = s.k / group;
  auto lhs = a.codes.to(at::kFloat).reshape({rows, blocks, 1, group});
  auto scales = a.scales.reshape({rows, s.k / 128});
  if (group == 32) scales = scales.repeat_interleave(4, 1);
  auto accumulated = at::zeros({rows, 1, s.n}, at::kFloat);
  // Fixed original K-block order. Decode and matrix temporaries are bounded
  // by one original block; all codec and payload arithmetic remains in ATen.
  const auto encoded_group = s.kind == Kind::kGrouped4 ? group / 2 : group;
  for (std::int64_t block = 0; block < blocks; ++block) {
    auto encoded = w.codes.slice(2, block * encoded_group,
                                 (block + 1) * encoded_group).contiguous();
    auto decoded = s.kind == Kind::kGrouped4
        ? torchao_cpu::f4_unpacked_to_f32(torchao_cpu::unpack_uint4(encoded))
        : encoded.view(at::kFloat8_e4m3fn).to(at::kFloat);
    auto dots = at::matmul(lhs.select(1, block), decoded.transpose(1, 2));
    auto scaled = dots * scales.select(1, block).reshape({rows, 1, 1});
    accumulated = accumulated + scaled * w.scales.select(2, block).unsqueeze(1);
  }
  return accumulated.reshape({s.tokens, s.slots, s.n}).to(s.output);
}
template <Kind Selected>
Status execute(const Program &p, std::span<const std::byte *const> pointers,
               std::byte *destination) noexcept {
  try {
    at::NoGradGuard no_grad;
    std::array<at::Tensor, 6> input;
    for (std::size_t i = 0; i < p.input_count; ++i) {
      const auto &desc = p.inputs[i];
      if (!desc.is_plain()) {
        input[i] = at::from_blob(const_cast<std::byte *>(pointers[i]),
            {static_cast<std::int64_t>(desc.required_bytes().value())}, at::kByte);
      } else {
        const auto type = desc.data_type() == DataType::kUInt32 ? at::kUInt32
            : (desc.data_type() == DataType::kInt32 ? at::kInt
            : (desc.data_type() == DataType::kUInt8 ? at::kByte
            : (desc.data_type() == DataType::kFloat16 ? at::kHalf
            : (desc.data_type() == DataType::kBFloat16 ? at::kBFloat16 : at::kFloat))));
        input[i] = at::from_blob(const_cast<std::byte *>(pointers[i]),
            at::IntArrayRef(desc.dimensions().data(), desc.rank()), type);
      }
    }
    const auto &s = p.shape;
    constexpr auto activation_index = (Selected == Kind::kGrouped8 || Selected == Kind::kGrouped4) ? 1u : 0u;
    if (Selected != Kind::kDecode && Selected != Kind::kArtifactDecode && Selected != Kind::kMlx &&
        Selected != Kind::kEncodedMatMul && Selected != Kind::kCausalConv &&
        !at::isfinite(input[activation_index]).all().item<bool>())
      return Status::InvalidArgument("original ATen CPU quantization requires finite activation");
    at::Tensor result;
    if constexpr (Selected == Kind::kCausalConv) {
      // Keep the declared F32 accumulator and one final storage rounding.
      // Layout/casts are original ATen operations; no HYLE numerical kernel.
      auto samples = input[0].permute({0, 2, 1}).to(at::kFloat).contiguous();
      auto weights = input[1].to(at::kFloat).reshape({s.n, 1, s.k});
      const std::array<std::int64_t,1> stride{1}, padding{0}, dilation{1};
      result = at::conv1d(samples, weights, std::nullopt, stride, padding, dilation, s.n)
                   .permute({0, 2, 1}).to(s.output).contiguous();
    } else if constexpr (Selected == Kind::kMlx) {
      const bool embedding = s.mlx.kind == model_ops::MlxQuantizedKind::kEmbedding;
      const bool grouped = s.mlx.kind == model_ops::MlxQuantizedKind::kGrouped;
      const bool affine = s.mlx.format == model_ops::MlxWeightFormat::kAffine4;
      at::Tensor ids;
      if (embedding || grouped) {
        ids = input[p.input_count - 1].reshape({-1}).to(at::kLong);
        if (at::logical_or(ids < 0, ids >= (embedding ? s.n : s.e)).any().item<bool>())
          return Status::OutOfRange("MLX original selected row/expert id is outside its source");
      }
      if (embedding) {
        result = borrowed::mlx_cpu_quant::embedding(input[0], input[1], input[2],
                                                    ids, s.mlx.group_size);
      } else {
        using Format = borrowed::mlx_cpu_quant::Format;
        const auto format = affine ? Format::affine4 :
            (s.mlx.format == model_ops::MlxWeightFormat::kMxfp4 ? Format::mxfp4 : Format::mxfp8);
        result = borrowed::mlx_cpu_quant::matmul(input[0], input[1], input[2],
            affine ? input[3] : at::Tensor{}, ids, format, s.mlx.group_size, s.output);
      }
    } else if constexpr (Selected == Kind::kArtifactDecode) {
      switch (s.artifact.format) {
      case model_ops::WeightArtifactFormat::kNf4:
        result = borrowed::bitsandbytes_cpu::decode_nf4(input[0], input[1],
            input[2], s.artifact.group_size, s.n, s.k, s.output); break;
      case model_ops::WeightArtifactFormat::kNf4DoubleQuant: {
        auto scales = borrowed::bitsandbytes_cpu::nested_scales(input[1],
            input[2], input[3], input[4]);
        result = borrowed::bitsandbytes_cpu::decode_nf4(input[0], scales,
            input[5], s.artifact.group_size, s.n, s.k, s.output); break;
      }
      case model_ops::WeightArtifactFormat::kAwqGemm4:
        result = borrowed::awq_cpu::decode_gemm4(input[0], input[1], input[2],
                                               s.artifact.group_size).to(s.output); break;
      case model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4:
        result = borrowed::compressed_tensors_cpu::decode_symmetric4(
            input[0], input[1], s.artifact.group_size).to(s.output); break;
      }
    } else if constexpr (Selected == Kind::kDecode) {
      auto w = weight(input[0], s, {});
      result = (w.codes.view(at::kFloat8_e4m3fn).to(at::kFloat)
                    .reshape({1, s.n, s.k / 128, 128}) * w.scales.unsqueeze(-1))
          .reshape({s.n, s.k}).to(s.output);
    } else if constexpr (Selected == Kind::kRoundtrip) {
      auto q = quantize(input[0], s);
      result = (q.codes.to(at::kFloat) * q.scales).reshape({s.tokens, s.k}).to(s.output);
    } else if constexpr (Selected == Kind::kGrouped8 || Selected == Kind::kGrouped4) {
      auto ids = input[2].reshape({-1}).to(at::kLong);
      if (at::logical_or(ids < 0, ids >= s.e).any().item<bool>())
        return Status::OutOfRange("ATen CPU selected expert ID is out of range");
      auto w = weight(input[0], s, ids);
      auto q = quantize(input[1], s);
      q.codes = q.codes.reshape({s.tokens, s.activations, s.k});
      q.scales = q.scales.reshape({s.tokens, s.activations, s.k / 128});
      if (s.activations == 1 && s.slots != 1) {
        q.codes = q.codes.expand({s.tokens, s.slots, s.k});
        q.scales = q.scales.expand({s.tokens, s.slots, s.k / 128});
      }
      result = block_dot(std::move(q), std::move(w), s);
    } else {
      auto w = weight(input[1], s, {});
      // A single matrix is shared across the token batch; no all-expert decode.
      w.codes = w.codes.expand({s.tokens, s.n, s.k});
      w.scales = w.scales.expand({s.tokens, s.n, s.k / 128});
      Quantized q;
      if constexpr (Selected == Kind::kLinear) q = quantize(input[0], s);
      else {
        const auto bytes = s.tokens * s.k;
        q.codes = input[0].slice(0, 0, bytes).view(at::kFloat8_e4m3fn);
        q.scales = input[0].slice(0, bytes).view(at::kFloat);
      }
      result = block_dot(std::move(q), std::move(w), s).reshape({s.tokens, s.n});
    }
    auto output = at::from_blob(destination,
        at::IntArrayRef(p.output.dimensions().data(), p.output.rank()), s.output);
    output.copy_(result);
    return Status::Ok();
  } catch (const c10::OutOfMemoryError &e) {
    return Status::ResourceExhausted(e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ATen CPU private allocation failed");
  } catch (const c10::Error &e) {
    return translate_error(e);
  } catch (const std::exception &e) {
    return Status::Internal(e.what());
  }
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *p = static_cast<Program *>(state);
  if (!p) return;
  p->~Program(); allocator.deallocate(p, sizeof(Program), alignof(Program));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView, void **out) noexcept {
  if (!out) return Status::InvalidArgument("ATen CPU invocation output is null");
  *out = nullptr; return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, Queue &,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Program *>(state);
  if (inputs.size() != p.input_count || outputs.size() != 1)
    return completed(Status::Internal("ATen CPU binding count differs"));
  std::array<mem::HostMapping, 6> mappings;
  std::array<const std::byte *, 6> pointers{};
  mem::HostMapping mapped_output;
  Status status = Status::Ok();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto buffer = inputs[i].storage();
    if (!buffer.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      status = buffer.invalidate();
    if (!status.ok()) break;
    auto mapping = buffer.map_host();
    if (!mapping) { status = std::move(mapping).error(); break; }
    mappings[i] = std::move(*mapping); pointers[i] = mappings[i].bytes().data();
  }
  if (status.ok()) {
    auto mapping = outputs[0].storage().map_host(mem::HostAccess::kWrite);
    if (!mapping) status = std::move(mapping).error();
    else {
      mapped_output = std::move(*mapping);
      auto writable = mapped_output.writable_bytes();
      if (!writable) status = std::move(writable).error();
      else status = p.run(p, std::span(pointers).first(p.input_count), writable->data());
      if (!outputs[0].storage().resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
        preserve_first_error(status, mapped_output.flush());
    }
  }
  preserve_first_error(status, mapped_output.close());
  for (auto &mapping : mappings) preserve_first_error(status, mapping.close());
  return completed(std::move(status));
}
const detail::ProviderPreparedVTable kVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
detail::ResourceBindingRequirement host(const TensorDesc &desc) noexcept {
  return {.required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = desc.required_alignment()};
}
} // namespace

bool is_torch_cpu_operation(const Operation &operation,
                           std::span<const TensorDesc> inputs) noexcept {
  if (operation.custom_as<model_ops::CausalConv1d>())
    return inputs.size() == 2 &&
        (inputs[0].data_type() == DataType::kFloat16 || inputs[0].data_type() == DataType::kBFloat16);
  if (const auto q = operation.custom_as<model_ops::BlockQuantizationRoundtrip>())
    return q->format == model_ops::BlockQuantizationFormat::kFloat8E4M3Block128;
  return operation.custom_as<model_ops::MlxQuantized>() ||
      operation.custom_as<model_ops::ArtifactWeightDecode>() ||
      operation.custom_as<model_ops::Float8WeightDecode>() ||
      operation.custom_as<model_ops::Float8BlockLinear>() ||
      operation.custom_as<model_ops::GroupedFloat8Linear>() ||
      operation.custom_as<model_ops::GroupedFloat4Linear>() ||
      (operation.get_if<MatMul>() && inputs.size() == 2 &&
       inputs[1].storage_format().id() == kFloat8WeightFloat32Block128);
}
Status propose_torch_cpu(const detail::ProviderOperationRequest &request,
    std::span<detail::ResourceBindingRequirement> in,
    std::span<detail::ResourceBindingRequirement> out,
    detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact()) return Status::Unimplemented("ATen CPU requires exact shapes");
  HYLE_ASSIGN_OR_RETURN(auto shape, select(request.operation, request.inputs, request.outputs));
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = host(request.inputs[i]);
  out[0] = host(request.outputs[0]);
  *proposal = {.route = static_cast<std::uint64_t>(request.operation.kind()),
      .runtime = kRuntime, .workspace_resource = host(request.outputs[0]),
      .cost = {.workspace = WorkspaceRequirement{}, .persistent_bytes = sizeof(Program),
          .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
          .synchronization_count = 0, .dispatch_count = 1},
      .implementation = ProviderImplementationId{0x544f524348000001ULL + static_cast<std::uint64_t>(shape.kind)}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare_torch_cpu(const detail::ProviderPrepareRequest &request,
    mem::AllocatorRef allocator, detail::ProviderPrepared *out) noexcept {
  auto shape = select(request.operation, request.inputs, request.outputs);
  if (!shape) return shape.error().IsUnimplemented()
      ? ProviderPrepareOutcome::unsupported(std::move(shape).error())
      : ProviderPrepareOutcome::error(std::move(shape).error());
  // Initialize the library's persistent CPU workers/dispatch on isolated data.
  // No user payload, model state or provider competition is evaluated here.
  static std::once_flag initialized;
  try {
    std::call_once(initialized, [] {
      at::NoGradGuard no_grad;
      auto a = at::zeros({64, 128}, at::kFloat);
      auto b = at::zeros({128, 128}, at::kFloat);
      (void)at::matmul(a, b);
      (void)torchao_cpu::quantize_affine_float8(a, at::ones({64, 1}, at::kFloat));
      (void)torchao_cpu::f4_unpacked_to_f32(at::zeros({512}, at::kByte));
    });
  } catch (const std::bad_alloc &) {
    return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("ATen CPU isolated initialization allocation failed"));
  } catch (const c10::Error &e) {
    auto status = translate_error(e);
    return status.IsResourceExhausted()
        ? ProviderPrepareOutcome::resource_declined(std::move(status))
        : ProviderPrepareOutcome::error(std::move(status));
  } catch (const std::exception &e) {
    return ProviderPrepareOutcome::error(Status::Internal(e.what()));
  }
  auto *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage) return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("ATen CPU program allocation failed"));
  auto *p = ::new (storage) Program{.shape = *shape, .output = request.outputs[0],
                                  .input_count = request.inputs.size()};
  switch (shape->kind) {
  case Kind::kDecode: p->run = &execute<Kind::kDecode>; break;
  case Kind::kLinear: p->run = &execute<Kind::kLinear>; break;
  case Kind::kEncodedMatMul: p->run = &execute<Kind::kEncodedMatMul>; break;
  case Kind::kGrouped8: p->run = &execute<Kind::kGrouped8>; break;
  case Kind::kGrouped4: p->run = &execute<Kind::kGrouped4>; break;
  case Kind::kRoundtrip: p->run = &execute<Kind::kRoundtrip>; break;
  case Kind::kArtifactDecode: p->run = &execute<Kind::kArtifactDecode>; break;
  case Kind::kMlx: p->run = &execute<Kind::kMlx>; break;
  case Kind::kCausalConv: p->run = &execute<Kind::kCausalConv>; break;
  }
  for (std::size_t i = 0; i < p->input_count; ++i) p->inputs[i] = request.inputs[i];
  *out = {.vtable = &kVTable, .state = p, .workspace = {}, .runtime = kRuntime,
          .persistent_bytes = sizeof(Program), .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::native::internal

#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(pop)
#endif
