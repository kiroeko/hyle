#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>

#include "hyle/compute/ops_visibility.h"
#include "hyle/compute/tensor_desc.h"

namespace hyle::compute {

enum class OperationKind : std::uint8_t {
  kUnknown = 0,
  kView,
  kCopy,
  kFill,
  kCast,
  kUnary,
  kBinary,
  kSelect,
  kReduce,
  kRmsNorm,
  kLayerNorm,
  kMatMul,
  kGatedActivation,
  kGather,
  kScatter,
  kConcat,
  kRoPE,
  kSoftmax,
  kTopK,
  kScaledDotProductAttention,
  kQuantizedMatMul,
  kCustom,
};

enum class ViewKind : std::uint8_t {
  kReshape = 0,
  kSlice,
  kPermute,
  kBroadcast,
};

enum class UnaryKind : std::uint8_t {
  kAbs = 0,
  kNeg,
  kExp,
  kLog,
  kSqrt,
  kRsqrt,
  kTanh,
  kSigmoid,
  kRelu,
  kGelu,
  kGeluTanh,
  kSilu,
  // log(1 + exp(x)), beta=1, evaluated without overflow for large positive x.
  // Floating input/output; NaN propagates, -infinity maps to +0.
  kSoftplus,
};

enum class BinaryKind : std::uint8_t {
  kAdd = 0,
  kSubtract,
  kMultiply,
  kDivide,
  kMinimum,
  kMaximum,
};

enum class ReductionKind : std::uint8_t {
  kSum = 0,
  kMean,
  kMinimum,
  kMaximum,
};

enum class GatedActivationKind : std::uint8_t {
  kSilu = 0,
  kGelu,
  kGeluTanh,
};

enum class GatedInputKind : std::uint8_t {
  kPackedHalves = 0,
  kSeparate,
};

enum class IndexBoundsPolicy : std::uint8_t {
  kChecked = 0,
  kUnchecked,
};

enum class ScatterKind : std::uint8_t {
  kAssign = 0,
  kAdd,
};

enum class RopeLayout : std::uint8_t {
  kInterleaved = 0,
  kHalfSplit,
};

enum class SoftmaxKind : std::uint8_t {
  kSoftmax = 0,
  kLogSoftmax,
};

enum class MaskKind : std::uint8_t {
  kNone = 0,
  kAdditive,
  kBoolean,
};

enum class AttentionLayout : std::uint8_t {
  kBatchSequenceHeadDepth = 0,
  kBatchHeadSequenceDepth,
};

enum class AliasPolicy : std::uint8_t {
  kNone = 0,
  kMayAliasInput0,
  kMayAliasAnyInput,
  kMustAliasInput0,
};

enum class EffectKind : std::uint8_t {
  kPure = 0,
  // Unknown external effects; not a claim that any resources are independent.
  kExternalSideEffect,
  // All external accesses are described by the resource ports below. Reading
  // mutable external state is not pure, even when no resource is written.
  kResourceAccess,
};

enum class ResourceAccess : std::uint8_t {
  kRead = 0,
  kWrite,
  kReadWrite,
};

enum class PrecisionPolicy : std::uint8_t {
  kExact = 0,
  kAccumulateInOutputType,
  kAccumulateInDeclaredType,
};

struct InputArity final {
  static constexpr std::uint16_t kVariadic =
      static_cast<std::uint16_t>(~std::uint16_t{0});

  std::uint16_t minimum = 0;
  std::uint16_t maximum = 0;

  constexpr bool accepts(std::size_t count) const noexcept {
    return count >= minimum && (maximum == kVariadic || count <= maximum);
  }
};

struct OperationTraits final {
  InputArity inputs{};
  std::uint8_t outputs = 0;
  AliasPolicy alias = AliasPolicy::kNone;
  EffectKind effect = EffectKind::kPure;
  PrecisionPolicy precision = PrecisionPolicy::kExact;
  DataType accumulation_type = DataType::kUnknown;
  bool materializes = true;
  // Semantic resource ports, not tensor input ordinals or runtime handles.
  // Must be nonzero exactly for kResourceAccess. Graph binds these ports to
  // graph-local identities; direct callers own resource ordering/lifetime.
  std::uint8_t resource_count = 0;
};

// Exact scalar payload for Fill. Floating storage values use their native bit
// representation; low-precision formats can be supplied with from_bits().
class ScalarValue final {
public:
  constexpr ScalarValue() noexcept = default;
  constexpr explicit ScalarValue(bool value) noexcept
      : type_(DataType::kBool), bits_(value ? 1u : 0u) {}
  constexpr explicit ScalarValue(std::int8_t value) noexcept
      : type_(DataType::kInt8), bits_(static_cast<std::uint8_t>(value)) {}
  constexpr explicit ScalarValue(std::uint8_t value) noexcept
      : type_(DataType::kUInt8), bits_(value) {}
  constexpr explicit ScalarValue(std::int16_t value) noexcept
      : type_(DataType::kInt16), bits_(static_cast<std::uint16_t>(value)) {}
  constexpr explicit ScalarValue(std::uint16_t value) noexcept
      : type_(DataType::kUInt16), bits_(value) {}
  constexpr explicit ScalarValue(std::int32_t value) noexcept
      : type_(DataType::kInt32), bits_(static_cast<std::uint32_t>(value)) {}
  constexpr explicit ScalarValue(std::uint32_t value) noexcept
      : type_(DataType::kUInt32), bits_(value) {}
  constexpr explicit ScalarValue(std::int64_t value) noexcept
      : type_(DataType::kInt64), bits_(static_cast<std::uint64_t>(value)) {}
  constexpr explicit ScalarValue(std::uint64_t value) noexcept
      : type_(DataType::kUInt64), bits_(value) {}
  constexpr explicit ScalarValue(float value) noexcept
      : type_(DataType::kFloat32), bits_(std::bit_cast<std::uint32_t>(value)) {}
  constexpr explicit ScalarValue(double value) noexcept
      : type_(DataType::kFloat64), bits_(std::bit_cast<std::uint64_t>(value)) {}

  static constexpr ScalarValue from_bits(DataType type,
                                         std::uint64_t bits) noexcept {
    ScalarValue result;
    result.type_ = type;
    result.bits_ = bits;
    return result;
  }

  constexpr bool valid() const noexcept { return type_ != DataType::kUnknown; }
  constexpr DataType data_type() const noexcept { return type_; }
  constexpr std::uint64_t bits() const noexcept { return bits_; }

private:
  DataType type_ = DataType::kUnknown;
  std::uint64_t bits_ = 0;
};

class HYLE_COMPUTE_OPS_API Axes final {
public:
  constexpr Axes() noexcept = default;

  static Axes all() noexcept { return {}; }
  static Result<Axes> from(std::span<const std::int32_t> axes) noexcept;

  constexpr bool is_all() const noexcept { return all_; }
  constexpr std::span<const std::int8_t> values() const noexcept {
    return {values_.data(), count_};
  }

private:
  std::array<std::int8_t, kMaxTensorRank> values_{};
  std::uint8_t count_ = 0;
  bool all_ = true;
};

class HYLE_COMPUTE_OPS_API View final {
public:
  constexpr View() noexcept = default;

  static Result<View> reshape(TensorShape shape) noexcept;
  static Result<View> slice(std::int32_t axis, std::int64_t start,
                            std::int64_t count, std::int64_t step = 1) noexcept;
  static Result<View> permute(std::span<const std::int32_t> axes) noexcept;
  static Result<View> broadcast_to(TensorShape shape) noexcept;

  constexpr ViewKind kind() const noexcept { return kind_; }
  constexpr const TensorShape &shape() const noexcept { return shape_; }
  constexpr std::int32_t axis() const noexcept { return axis_; }
  constexpr std::int64_t start() const noexcept { return start_; }
  constexpr std::int64_t count() const noexcept { return count_; }
  constexpr std::int64_t step() const noexcept { return step_; }
  constexpr std::span<const std::int8_t> permutation() const noexcept {
    return {permutation_.data(), permutation_count_};
  }

private:
  ViewKind kind_ = ViewKind::kReshape;
  TensorShape shape_{};
  std::array<std::int8_t, kMaxTensorRank> permutation_{};
  std::int64_t start_ = 0;
  std::int64_t count_ = 0;
  std::int64_t step_ = 1;
  std::int32_t axis_ = 0;
  std::uint8_t permutation_count_ = 0;
};

struct Copy final {};

struct Fill final {
  TensorShape output_shape{};
  DataType output_type = DataType::kUnknown;
  ScalarValue value{};
};

struct Cast final {
  DataType output_type = DataType::kUnknown;
};

struct Unary final {
  UnaryKind kind = UnaryKind::kAbs;
};

struct Binary final {
  BinaryKind kind = BinaryKind::kAdd;
};

struct Select final {};

struct Reduce final {
  ReductionKind kind = ReductionKind::kSum;
  Axes axes = Axes::all();
  bool keep_dimensions = false;
  DataType accumulation_type = DataType::kUnknown;
};

struct RmsNorm final {
  float epsilon = 1.0e-5F;
  DataType accumulation_type = DataType::kFloat32;
};

struct LayerNorm final {
  float epsilon = 1.0e-5F;
  DataType accumulation_type = DataType::kFloat32;
};

// Logical [...,M,K] * [...,K,N] with NumPy-style batch broadcasting.
// Physical orientation remains a TensorDesc stride/storage concern.
struct MatMul final {
  DataType accumulation_type = DataType::kFloat32;
};

enum class QuantizedWeightMode : std::uint8_t {
  kSignedSymmetric4 = 0,
  kUnsignedAffine4,
};

// Grouped Q4 linear projection: [..., K] * transpose([N, K]) -> [..., N].
// Inputs are activation, packed weights, scales and (affine only) biases.
// Scales/biases have logical shape [N, K / group_size]. Signed weights mean
// q * scale for q in [-8, 7]; affine weights mean q * scale + bias for q in
// [0, 15]. The weight StorageFormatId separately identifies physical packing.
// All inputs share one floating logical dtype; outputs keep that dtype. The
// declared accumulation allows normal arithmetic reassociation, not implicit
// activation conversion or bitwise equivalence to materialized weights.
struct QuantizedMatMul final {
  QuantizedWeightMode mode = QuantizedWeightMode::kSignedSymmetric4;
  std::uint32_t group_size = 64;
  DataType accumulation_type = DataType::kFloat32;
};

struct GatedActivation final {
  GatedActivationKind activation = GatedActivationKind::kSilu;
  GatedInputKind inputs = GatedInputKind::kPackedHalves;
};

struct Gather final {
  std::int32_t axis = 0;
  IndexBoundsPolicy bounds = IndexBoundsPolicy::kChecked;
};

struct Scatter final {
  std::int32_t axis = 0;
  ScatterKind kind = ScatterKind::kAssign;
  IndexBoundsPolicy bounds = IndexBoundsPolicy::kChecked;
};

struct Concat final {
  std::int32_t axis = 0;
};

struct RoPE final {
  std::size_t rotary_dimensions = 0;
  std::int32_t sequence_axis = -2;
  RopeLayout layout = RopeLayout::kInterleaved;
};

struct Softmax final {
  std::int32_t axis = -1;
  float scale = 1.0F;
  SoftmaxKind kind = SoftmaxKind::kSoftmax;
  MaskKind mask = MaskKind::kNone;
  DataType accumulation_type = DataType::kFloat32;
};

struct TopK final {
  std::size_t count = 1;
  std::int32_t axis = -1;
  bool largest = true;
  bool sorted = true;
  bool stable = true;
};

struct ScaledDotProductAttention final {
  AttentionLayout layout = AttentionLayout::kBatchSequenceHeadDepth;
  MaskKind mask = MaskKind::kNone;
  float scale = 1.0F;
  bool causal = false;
  DataType accumulation_type = DataType::kFloat32;
};

struct OperationId final {
  std::uint64_t namespace_id = 0;
  std::uint64_t operation_id = 0;

  // For a custom operation this identity names the complete schema version,
  // including the inline descriptor layout. Reusing an ID after changing the
  // descriptor or its semantics is a contract violation.
  constexpr bool valid() const noexcept {
    return namespace_id != 0 || operation_id != 0;
  }
  friend constexpr bool operator==(OperationId, OperationId) noexcept = default;
};

namespace detail {

inline constexpr std::size_t kCustomOperationInlineBytes = 64;

struct CustomOperationVTable final {
  std::string_view name{};
  OperationTraits (*traits)(const void *descriptor) noexcept = nullptr;
  Status (*infer)(const void *descriptor, std::span<const TensorDesc> inputs,
                  std::span<TensorDesc> outputs) noexcept = nullptr;
  Result<ResourceAccess> (*resource_access)(
      const void *descriptor, std::size_t index) noexcept = nullptr;
};

struct CustomOperationValue final {
  OperationId id{};
  const CustomOperationVTable *vtable = nullptr;
  std::uint8_t descriptor_size = 0;
  alignas(std::max_align_t)
      std::array<std::byte, kCustomOperationInlineBytes> descriptor{};
};

template <class Descriptor>
concept InlineCustomOperationDescriptor =
    std::is_trivially_copyable_v<Descriptor> &&
    sizeof(Descriptor) <= kCustomOperationInlineBytes &&
    alignof(Descriptor) <= alignof(std::max_align_t) &&
    requires(const Descriptor &descriptor, std::span<const TensorDesc> inputs,
             std::span<TensorDesc> outputs) {
      { Descriptor::operation_id } -> std::convertible_to<OperationId>;
      { Descriptor::operation_name } -> std::convertible_to<std::string_view>;
      { descriptor.operation_traits() } -> std::same_as<OperationTraits>;
      {
        descriptor.infer_output_descriptions(inputs, outputs)
      } -> std::same_as<Status>;
    };

template <InlineCustomOperationDescriptor Descriptor>
Descriptor load_custom_descriptor(const void *storage) noexcept {
  std::array<std::byte, sizeof(Descriptor)> bytes{};
  std::memcpy(bytes.data(), storage, bytes.size());
  return std::bit_cast<Descriptor>(bytes);
}

template <InlineCustomOperationDescriptor Descriptor>
struct CustomOperationModel final {
  static OperationTraits traits(const void *storage) noexcept {
    const Descriptor descriptor = load_custom_descriptor<Descriptor>(storage);
    return descriptor.operation_traits();
  }

  static Status infer(const void *storage, std::span<const TensorDesc> inputs,
                      std::span<TensorDesc> outputs) noexcept {
    const Descriptor descriptor = load_custom_descriptor<Descriptor>(storage);
    return descriptor.infer_output_descriptions(inputs, outputs);
  }

  static Result<ResourceAccess> resource_access(const void *storage,
                                                std::size_t index) noexcept {
    if constexpr (requires(const Descriptor &descriptor) {
                    {
                      descriptor.resource_access(index)
                    } noexcept -> std::same_as<ResourceAccess>;
                  }) {
      const Descriptor descriptor = load_custom_descriptor<Descriptor>(storage);
      return descriptor.resource_access(index);
    } else {
      return std::unexpected(Status::InvalidArgument(
          "resource operation requires resource_access(index) const noexcept"));
    }
  }

  inline static const CustomOperationVTable vtable{
      std::string_view(Descriptor::operation_name), &traits, &infer,
      &resource_access};
};

} // namespace detail

// Provider-neutral, heap-free typed schema. It is a cold build/prepare value;
// PreparedOperation owns the hot execution artifact in a separate header.
class Operation final {
public:
  constexpr Operation() noexcept = default;
  constexpr explicit Operation(View value) noexcept : value_(value) {}
  constexpr explicit Operation(Copy value) noexcept : value_(value) {}
  constexpr explicit Operation(Fill value) noexcept : value_(value) {}
  constexpr explicit Operation(Cast value) noexcept : value_(value) {}
  constexpr explicit Operation(Unary value) noexcept : value_(value) {}
  constexpr explicit Operation(Binary value) noexcept : value_(value) {}
  constexpr explicit Operation(Select value) noexcept : value_(value) {}
  constexpr explicit Operation(Reduce value) noexcept : value_(value) {}
  constexpr explicit Operation(RmsNorm value) noexcept : value_(value) {}
  constexpr explicit Operation(LayerNorm value) noexcept : value_(value) {}
  constexpr explicit Operation(MatMul value) noexcept : value_(value) {}
  constexpr explicit Operation(QuantizedMatMul value) noexcept : value_(value) {}
  constexpr explicit Operation(GatedActivation value) noexcept
      : value_(value) {}
  constexpr explicit Operation(Gather value) noexcept : value_(value) {}
  constexpr explicit Operation(Scatter value) noexcept : value_(value) {}
  constexpr explicit Operation(Concat value) noexcept : value_(value) {}
  constexpr explicit Operation(RoPE value) noexcept : value_(value) {}
  constexpr explicit Operation(Softmax value) noexcept : value_(value) {}
  constexpr explicit Operation(TopK value) noexcept : value_(value) {}
  constexpr explicit Operation(ScaledDotProductAttention value) noexcept
      : value_(value) {}

  template <detail::InlineCustomOperationDescriptor Descriptor>
  static Operation custom(Descriptor descriptor) noexcept {
    constexpr OperationId id = Descriptor::operation_id;
    constexpr std::string_view name = Descriptor::operation_name;
    static_assert(id.valid(), "custom operation identity must be non-empty");
    static_assert(!name.empty(), "custom operation name must be non-empty");

    detail::CustomOperationValue value;
    value.id = id;
    value.vtable = &detail::CustomOperationModel<Descriptor>::vtable;
    value.descriptor_size = static_cast<std::uint8_t>(sizeof(Descriptor));
    std::memcpy(value.descriptor.data(), &descriptor, sizeof(descriptor));
    return Operation(value);
  }

  constexpr bool valid() const noexcept {
    return kind() != OperationKind::kUnknown;
  }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr OperationKind kind() const noexcept {
    return static_cast<OperationKind>(value_.index());
  }

  constexpr std::optional<OperationId> custom_id() const noexcept {
    const auto *value = std::get_if<detail::CustomOperationValue>(&value_);
    return value != nullptr ? std::optional<OperationId>(value->id)
                            : std::nullopt;
  }

  template <class Descriptor>
  constexpr const Descriptor *get_if() const noexcept {
    return std::get_if<Descriptor>(&value_);
  }

  template <detail::InlineCustomOperationDescriptor Descriptor>
  std::optional<Descriptor> custom_as() const noexcept {
    const auto *value = std::get_if<detail::CustomOperationValue>(&value_);
    // A template-static vtable can have a different address in another shared
    // factor. OperationId is the schema/version identity; the exact byte size
    // catches accidental stale descriptor declarations before decoding.
    if (value == nullptr || value->id != Descriptor::operation_id ||
        value->descriptor_size != sizeof(Descriptor)) {
      return std::nullopt;
    }
    return detail::load_custom_descriptor<Descriptor>(value->descriptor.data());
  }

private:
  using Value =
      std::variant<std::monostate, View, Copy, Fill, Cast, Unary, Binary,
                   Select, Reduce, RmsNorm, LayerNorm, MatMul, GatedActivation,
                   Gather, Scatter, Concat, RoPE, Softmax, TopK,
                   ScaledDotProductAttention, QuantizedMatMul,
                   detail::CustomOperationValue>;
  static_assert(std::variant_size_v<Value> ==
                static_cast<std::size_t>(OperationKind::kCustom) + 1);

  explicit Operation(detail::CustomOperationValue value) noexcept
      : value_(value) {}

  friend HYLE_COMPUTE_OPS_API std::string_view
  operation_name(const Operation &operation) noexcept;
  friend HYLE_COMPUTE_OPS_API OperationTraits
  operation_traits(const Operation &operation) noexcept;
  friend HYLE_COMPUTE_OPS_API Result<ResourceAccess>
  operation_resource_access(const Operation &operation,
                            std::size_t index) noexcept;
  friend HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      const Operation &operation, std::span<const TensorDesc> inputs,
      std::span<TensorDesc> outputs) noexcept;

  Value value_{};
};

HYLE_COMPUTE_OPS_API std::string_view
operation_name(OperationKind operation) noexcept;
HYLE_COMPUTE_OPS_API std::string_view
operation_name(const Operation &operation) noexcept;
HYLE_COMPUTE_OPS_API OperationTraits
operation_traits(const Operation &operation) noexcept;
// Cold schema queries. The callback must be deterministic, allocation-free,
// and must not read payload or execute work. Results are values, never spans
// into the temporary inline descriptor decoded by the producer callback.
HYLE_COMPUTE_OPS_API Result<ResourceAccess>
operation_resource_access(const Operation &operation,
                          std::size_t index) noexcept;
HYLE_COMPUTE_OPS_API Status
validate_operation_effects(const Operation &operation) noexcept;
HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
    const Operation &operation, std::span<const TensorDesc> inputs,
    std::span<TensorDesc> outputs) noexcept;
HYLE_COMPUTE_OPS_API Result<TensorDesc>
infer_single_output(const Operation &operation,
                    std::span<const TensorDesc> inputs) noexcept;

static_assert(static_cast<std::size_t>(OperationKind::kCustom) + 1 == 22);
static_assert(std::is_trivially_copyable_v<Operation>);
static_assert(sizeof(Operation) <= 256);

} // namespace hyle::compute
