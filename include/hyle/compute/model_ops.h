#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/compute/operation.h"
#include "hyle/compute/low_precision.h"

namespace hyle::compute::model_ops {

// Gather each index from its corresponding source row/slice, rather than
// forming the Cartesian product used by Gather. Non-axis extents must match;
// there is no broadcasting or negative-index wrapping. For scores [T,E] and
// indices [T,K], output[t,k] = scores[t,indices[t,k]], with only T*K values.
// I32 sources preserve integer IDs bitwise, without a floating conversion.
struct GatherElements final {
  std::int32_t axis = -1;
  IndexBoundsPolicy bounds = IndexBoundsPolicy::kChecked;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x47415448454c0001ULL};
  static constexpr std::string_view operation_name = "gather-elements";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2,2},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    if(in.size()!=2 || out.size()!=1 || !in[0].valid() || !in[1].valid() ||
        !in[0].is_plain() || !in[1].is_plain() || !in[0].is_dense() || !in[1].is_dense() ||
        !in[0].rank() || in[0].rank()!=in[1].rank() || in[1].data_type()!=DataType::kInt32 ||
        (in[0].data_type()!=DataType::kFloat32 && in[0].data_type()!=DataType::kFloat16 &&
         in[0].data_type()!=DataType::kBFloat16 && in[0].data_type()!=DataType::kInt32) ||
        (bounds!=IndexBoundsPolicy::kChecked && bounds!=IndexBoundsPolicy::kUnchecked))
      return Status::InvalidArgument("GatherElements needs plain dense floating or I32 data and same-rank I32 indices");
    const auto rank=static_cast<std::int32_t>(in[0].rank());
    const auto dimension=axis<0?axis+rank:axis;
    if(dimension<0 || dimension>=rank)
      return Status::InvalidArgument("GatherElements axis is outside the input rank");
    for(std::int32_t d=0;d<rank;++d)
      if(d!=dimension && in[0].dimensions()[d]!=in[1].dimensions()[d])
        return Status::InvalidArgument("GatherElements non-axis extents must match without broadcasting");
    HYLE_ASSIGN_OR_RETURN(auto count,in[1].element_count());
    if(count && !in[0].dimensions()[dimension])
      return Status::InvalidArgument("GatherElements nonempty indices need a nonempty source axis");
    HYLE_ASSIGN_OR_RETURN(out[0],TensorDesc::dense(in[0].data_type(),in[1].dimensions()));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<GatherElements>);
inline Operation gather_elements(std::int32_t axis=-1,
    IndexBoundsPolicy bounds=IndexBoundsPolicy::kChecked) noexcept {
  return Operation::custom(GatherElements{axis,bounds});
}

// Encode only the supplied BF16 rows. Group absmax/floor, power-of-two scale
// and saturated nearest-even codes match the original QAT arithmetic:
// E4M3FN group64 uses floor1e-4/max448; E2M1 group32 uses
// floor7.052966104933725e-38/max6. Codes retain the source sign, including
// negative zero; the FP8 tail is copied bitwise. Input NaN/Inf is invalid,
// but a finite input may quantize/decode to BF16 Inf. Indexer Hadamard and
// its BF16 boundary remain explicit upstream operations.
struct RowCacheEncode final {
  RowCacheEncoding encoding{};
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x524f57454e430001ULL};
  static constexpr std::string_view operation_name = "row-cache-encode";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1,1},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    if (in.size()!=1 || out.size()!=1 || !in[0].valid() ||
        !in[0].is_plain() || !in[0].is_dense() || in[0].rank()!=2 ||
        in[0].data_type()!=DataType::kBFloat16 || encoding.depth()==0 ||
        in[0].dimensions()[1]!=encoding.depth())
      return Status::InvalidArgument("row cache encoder requires matching dense BF16 rows");
    HYLE_ASSIGN_OR_RETURN(out[0],row_cache_carrier_description(encoding,in[0].dimensions()[0]));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<RowCacheEncode>);
inline Operation row_cache_encode(RowCacheEncoding encoding) noexcept {
  return Operation::custom(RowCacheEncode{encoding});
}

// Read/decode only the indexed rows; no Cartesian product, negative wrapping
// or implicit zero-fill. The bank is the matching plain U8 carrier above.
// Checked IDs are all validated before writing any output. Unchecked IDs
// must already be valid. Decode preserves the encoding's IEEE values:
// E4M3FN reserved codes or UE8M0=255 decode to NaN, not an index error.
struct RowCacheGatherDecode final {
  RowCacheEncoding encoding{};
  IndexBoundsPolicy bounds = IndexBoundsPolicy::kChecked;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x524f574741540001ULL};
  static constexpr std::string_view operation_name = "row-cache-gather-decode";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2,2},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    if (in.size()!=2 || out.size()!=1 || encoding.depth()==0 ||
        !in[0].valid() || !in[1].valid() || !in[0].is_plain() ||
        !in[1].is_plain() || !in[0].is_dense() || !in[1].is_dense() ||
        in[0].rank()!=2 || in[0].data_type()!=DataType::kUInt8 ||
        in[0].dimensions()[1]!=encoding.row_bytes() ||
        in[1].data_type()!=DataType::kInt32 || in[1].rank()>=kMaxTensorRank ||
        (bounds!=IndexBoundsPolicy::kChecked && bounds!=IndexBoundsPolicy::kUnchecked))
      return Status::InvalidArgument("row cache gather requires matching byte rows and dense I32 IDs");
    HYLE_ASSIGN_OR_RETURN(auto count,in[1].element_count());
    if (count!=0 && in[0].dimensions()[0]==0)
      return Status::InvalidArgument("nonempty row cache IDs need a nonempty bank");
    std::array<std::int64_t,kMaxTensorRank> shape{};
    for (std::size_t i=0;i<in[1].rank();++i) shape[i]=in[1].dimensions()[i];
    shape[in[1].rank()]=encoding.depth();
    HYLE_ASSIGN_OR_RETURN(out[0],TensorDesc::dense(DataType::kBFloat16,
        std::span(shape).first(in[1].rank()+1)));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<RowCacheGatherDecode>);
inline Operation row_cache_gather_decode(RowCacheEncoding encoding,
    IndexBoundsPolicy bounds=IndexBoundsPolicy::kChecked) noexcept {
  return Operation::custom(RowCacheGatherDecode{encoding,bounds});
}

// Checked padding-aware decode. Only -1 is padding: it produces a BF16
// positive-zero row without accessing the bank. Other invalid IDs fail before
// any output write. A zero-row bank is valid when all runtime IDs are -1.
// Use the corresponding additive mask when padding participates in attention.
struct RowCacheGatherDecodePadded final {
  RowCacheEncoding encoding{};
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,0x524f574744500001ULL};
  static constexpr std::string_view operation_name="row-cache-gather-decode-padded";
  constexpr OperationTraits operation_traits()const noexcept{
    return {{2,2},1,AliasPolicy::kNone,EffectKind::kPure,PrecisionPolicy::kExact};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,std::span<TensorDesc> out)const noexcept{
    if(in.size()!=2 || out.size()!=1 || !encoding.depth() ||
       !in[0].valid() || !in[1].valid() || !in[0].is_plain() || !in[1].is_plain() ||
       !in[0].is_dense() || !in[1].is_dense() || in[0].rank()!=2 ||
       in[0].data_type()!=DataType::kUInt8 || in[0].dimensions()[1]!=encoding.row_bytes() ||
       in[1].data_type()!=DataType::kInt32 || in[1].rank()>=kMaxTensorRank)
      return Status::InvalidArgument("padded row decode requires matching byte rows and dense I32 IDs");
    std::array<std::int64_t,kMaxTensorRank> shape{};
    for(std::size_t i=0;i<in[1].rank();++i)shape[i]=in[1].dimensions()[i];
    shape[in[1].rank()]=encoding.depth();
    HYLE_ASSIGN_OR_RETURN(out[0],TensorDesc::dense(DataType::kBFloat16,std::span(shape).first(in[1].rank()+1)));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<RowCacheGatherDecodePadded>);
inline Operation row_cache_gather_decode_padded(RowCacheEncoding encoding)noexcept{
  return Operation::custom(RowCacheGatherDecodePadded{encoding});
}

// DeepSeek indexer scoring over HYLE's row-inline FP4/UE8M0 cache. Query
// [T,64,128] is already Hadamard/FP4-QAT rounded BF16; weights [T,64] are
// already prescaled BF16. The U8 bank is [L,68], not a ggml block or a
// native page-footer layout. I32 ranges [T,2] select half-open [begin,end)
// intervals in that bank; every range is checked before any output write.
// Each dot accumulates in F32 and rounds to BF16 before ReLU; each weighted
// product rounds to BF16, and the F32 head sum rounds once to BF16. Output
// [T,L] widens those scores to F32, with -Inf outside each query's interval.
// No implicit query quantization, head scaling or TopK is performed.
struct RowCacheIndexerScores final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x524f574944580001ULL};
  static constexpr std::string_view operation_name = "row-cache-indexer-scores";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4,4},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> in,
                                   std::span<TensorDesc> out) const noexcept {
    if (in.size()!=4 || out.size()!=1)
      return Status::InvalidArgument("row cache indexer requires four inputs and one output");
    for (const auto &desc : in)
      if (!desc.valid() || !desc.is_plain() || !desc.is_dense())
        return Status::InvalidArgument("row cache indexer requires plain dense inputs");
    if (in[0].rank()!=3 || in[1].rank()!=2 || in[2].rank()!=2 ||
        in[3].rank()!=2 || in[0].data_type()!=DataType::kBFloat16 ||
        in[1].data_type()!=DataType::kUInt8 ||
        in[2].data_type()!=DataType::kBFloat16 ||
        in[3].data_type()!=DataType::kInt32)
      return Status::InvalidArgument("row cache indexer requires BF16 query/weights, U8 bank and I32 ranges");
    const auto q=in[0].dimensions(), bank=in[1].dimensions();
    const auto w=in[2].dimensions(), ranges=in[3].dimensions();
    if (q[1]!=64 || q[2]!=128 || bank[1]!=68 || w[0]!=q[0] ||
        w[1]!=64 || ranges[0]!=q[0] || ranges[1]!=2)
      return Status::InvalidArgument("row cache indexer dimensions differ from H64/D128 row-inline FP4 scoring");
    HYLE_ASSIGN_OR_RETURN(out[0],TensorDesc::dense(DataType::kFloat32,
        std::array{q[0],bank[0]}));
    return Status::Ok();
  }
};
static_assert(detail::InlineCustomOperationDescriptor<RowCacheIndexerScores>);
inline Operation row_cache_indexer_scores() noexcept {
  return Operation::custom(RowCacheIndexerScores{});
}

// Encode only the supplied new rows, never mutate or decode an existing cache.
// Q8_0: block32 absmax/127, F16 scale, roundf codes in [-127,127], all-zero
// blocks stay zero. If the rounded F16 scale is zero, all 34 block bytes are
// zero; otherwise codes use the original F32 scale/inverse before F16 storage.
// Inputs are finite with absmax/127 <= 65504.
// FP8: input/scale, finite-saturating nearest-even E4M3FN. The same immutable
// CacheStorage must describe subsequent cache consumption. Output is its real
// plain copy carrier; page ownership/publication is handled by original F2.
struct CacheEncode final {
  CacheStorage storage{};
  static constexpr OperationId operation_id{0x48594c4543414348ULL,
                                             0x45454e434f444501ULL};
  static constexpr std::string_view operation_name = "cache-encode";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                  std::span<TensorDesc> outputs) const noexcept {
    return infer_cache_encode(storage, inputs, outputs);
  }
};
static_assert(detail::InlineCustomOperationDescriptor<CacheEncode>);

enum class WeightArtifactFormat : std::uint32_t {
  kNf4,
  kNf4DoubleQuant,
  kAwqGemm4,
  kCompressedTensorsSymmetric4,
  kCompressedTensorsAsymmetric4,
  kGptqV1Int4,
};

// Explicit cold preparation of dense original model weights. Run once and
// retain the resulting [N,K] tensor as a constant for mature MatMul/Gather.
// This operation neither loads a model nor caches mutable input payloads.
// NF4 inputs: U8[N,K/2], F32[G] absmax, F32[16] codebook, G=N*K/group.
// Nested NF4: U8[N,K/2], U8[G], F32[ceil(G/256)], F32[256], F32 scalar
// offset, F32[16]. AWQ: I32[K,N/8], I32[K/group,N/8], float[K/group,N].
// CT symmetric: I32[N,K/8], float[N,K/group], original low-first q minus8.
// CT asymmetric: I32[N,K/8], I32[ceil(N/8),K/group], float[N,K/group].
// CT codes pack low-first along K; zero points pack low-first along N. Natural
// groups only, without g_idx. Subtract/multiply round in the scale dtype before
// the final requested output conversion; an explicit F32 output keeps this.
// GPTQ-v1 INT4: I32 Q[K/8,N], I32 Z[K/group,N/8], F16 SF[K/group,N],
// I32 g_idx[K]. Q packs low-first along K; Z low-first along N stores zero-1.
// The decoded zero is raw nibble+1, including15 ->16, never masked again.
// g_idx selects a scale/zero group for original K (not a permutation); every
// index is checked before index-derived access on each submission. Integer
// subtraction is exact, multiplication rounds to F16 before output conversion.
// Default output is F16 [N,K]; no inference of packing from desc_act/filename.
// Packing/group/precision come from the artifact, never its filename.
struct ArtifactWeightDecode final {
  WeightArtifactFormat format = WeightArtifactFormat::kNf4;
  std::uint32_t group_size = 64;
  // Unknown chooses the original scale dtype for AWQ/CT, F16 for GPTQ-v1,
  // and BF16 for the admitted NF4 artifact. F16/BF16/F32 overrides are explicit numeric choices.
  DataType output_type = DataType::kUnknown;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x57474445434f0001ULL};
  static constexpr std::string_view operation_name = "artifact-weight-decode";
  constexpr OperationTraits operation_traits() const noexcept {
    const std::uint16_t count = format == WeightArtifactFormat::kNf4DoubleQuant ? 6 :
        (format == WeightArtifactFormat::kGptqV1Int4 ? 4 :
        (format == WeightArtifactFormat::kCompressedTensorsSymmetric4 ? 2 : 3));
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};

inline Operation artifact_weight_decode(WeightArtifactFormat format,
    std::uint32_t group_size, DataType output_type = DataType::kUnknown) noexcept {
  return Operation::custom(ArtifactWeightDecode{format, group_size, output_type});
}

// ModelOpt's software-reference NVFP4 W4A4 linear, not the hardware FP4 GEMM
// accumulation contract. Inputs: BF16 A[M,K], original adjacent low-first U8
// W[N,K/2], E4M3FN scales[N,K/16], F32 weight_global and input_global scalars.
// K contains complete groups of16. No implicit dense weight preparation.
//
// For each current A group, b=absmax(A), g=input_global and
// s=F32(RNE_E4M3(min(b/F32(6*g),448))*g); replace s<1e-5 by1.
// Round abs(A)/s to E2M1 nearest-even, multiply by s and store BF16. Its sign
// uses A>=0, as the original Hopper reference does (-0 becomes +0).
// W is BF16(E2M1(code)*F32(E4M3(scale)*weight_global)), with the non-fast
// producer's scale-first order; both zero codes decode to positive zero.
// Dot products accumulate in F32 with the backend's reduction tree, followed
// by one BF16 output rounding. These pointwise operand boundaries must stay
// before the dot; native scaled FP4 GEMM is not presumed equivalent.
//
// A must be finite, scales finite/nonnegative, and both globals positive and
// finite. Runtime violations fail Completion; failed outputs are unusable,
// without an all-output rollback guarantee. Finite scaling may overflow.
struct Nvfp4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4e5646344c494e01ULL};
  static constexpr std::string_view operation_name = "nvfp4-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{5,5},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<Nvfp4Linear>);
inline Operation nvfp4_linear() noexcept {
  return Operation::custom(Nvfp4Linear{});
}

// Expert-selected version of the same software-reference group16 contract.
// Inputs: BF16 A[T,A_slots,K], U8 W[E,N,K/2], E4M3FN SF[E,N,K/16],
// independent F32 WG[E]/IG[E], and checked I32 IDs[T,S]; Y is BF16[T,S,N].
// All dimensions are positive and K is divisible16. A_slots is1 orS; A may
// also share a real zero-slot-stride view with contiguous K and nonoverlapping,
// element-aligned token rows. Other ports use plain dense storage.
//
// IDs must all lie in[0,E) before expert addressing. Only selected experts'
// WG/IG/SF values are validated/consumed; unselected metadata may be invalid.
// Original A is shared without expansion, but QDQ is per selected slot because
// IG[e] participates in its block-scale rounding. Duplicate IDs retain slots.
// Nvfp4Linear's operand rounding, scale-first W tree, finite-input domain and
// failed-Completion contract apply independently to every selected expert.
// This is not a hardware-native FP4 GEMM or an artifact/producer selector.
struct GroupedNvfp4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4e56463447525001ULL};
  static constexpr std::string_view operation_name = "grouped-nvfp4-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{6,6},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<GroupedNvfp4Linear>);
inline Operation grouped_nvfp4_linear() noexcept {
  return Operation::custom(GroupedNvfp4Linear{});
}

// Precision immediately before the E2M1 nearest-even threshold operation.
// This selects arithmetic, not a checkpoint, producer version or backend.
enum class Fp4InputRounding : std::uint32_t { kFloat32, kBFloat16 };

// Local group16 NVFP4 software reference with the original CT-style global
// quantization multipliers. Ports and geometry match Nvfp4Linear, but the
// original independent positive-finite F32 WG/IG scalars are divisors when
// reconstructing a group scale. They are not pre-inverted by the caller.
//
// For each finite BF16 A group, b=absmax(A), u=BF16_RNE(F32(b/6)),
// sf=RNE_E4M3(clamp(F32(IG*u),0,448)); replace exactly-zero sf by0.125.
// s=F32(sf/IG), v=clamp(F32(A/s),-6,6). Optionally round v to BF16,
// then E2M1 nearest-even. The portable reference uses sign(v): explicit
// signed zero becomes+0, while negative nonzero values rounding to0 retain
// -0. A_operand=BF16_RNE(F32(E2M1(v)*s)). There is no tiny-scale guard.
//
// W_operand=BF16_RNE(F32(E2M1(code)*F32(E4M3(SF)/WG))); code8 is -0.
// Stored SF may be zero, but must be finite/nonnegative. Both BF16 operands
// precede the backend F32 dot reduction and final BF16 output rounding.
// Finite source scales can overflow during reconstruction; IEEE results
// remain meaningful. Runtime domain violations fail Completion, as above.
// This contract does not claim hardware scaled-FP4 GEMM equivalence.
struct LocalNvfp4Linear final {
  Fp4InputRounding rounding = Fp4InputRounding::kFloat32;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4c4e5646344c4901ULL};
  static constexpr std::string_view operation_name = "local-nvfp4-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{5,5},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<LocalNvfp4Linear>);
inline Operation local_nvfp4_linear(
    Fp4InputRounding rounding = Fp4InputRounding::kFloat32) noexcept {
  return Operation::custom(LocalNvfp4Linear{rounding});
}

// Same local arithmetic for selected experts. Geometry and six original
// ports match GroupedNvfp4Linear, including shared/independent A slots.
// Each selected expert's IG participates in its own A-group rounding;
// complete ID validation precedes any expert address computation.
struct GroupedLocalNvfp4Linear final {
  Fp4InputRounding rounding = Fp4InputRounding::kFloat32;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4c4e564634474c01ULL};
  static constexpr std::string_view operation_name = "grouped-local-nvfp4-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{6,6},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<GroupedLocalNvfp4Linear>);
inline Operation grouped_local_nvfp4_linear(
    Fp4InputRounding rounding = Fp4InputRounding::kFloat32) noexcept {
  return Operation::custom(GroupedLocalNvfp4Linear{rounding});
}

enum class Mxfp4ActivationMode : std::uint32_t {
  kFloat4Group32,
  kBFloat16,
};

// Portable group32 MXFP4 software reference. Three original ports:
// finite BF16 A[M,K], adjacent low-even/high-odd E2M1 U8 W[N,K/2],
// U8 E8M0 SF[N,K/32]; BF16 Y[M,N]. Positive M/N/K, complete group32 K.
//
// kBFloat16 uses A directly, without activation quantization or scratch storage.
// kFloat4Group32 (the default W4A4 mode) computes each A group's finite BF16
// absmax bits b, then r=(b+0x20)&0x7f80.
// r=0 selects exponent -127; r=+Inf selects127; otherwise take r's unbiased
// exponent minus2, clamped to[-127,127]. With s=2^exponent, round A/s to
// BF16, saturating E2M1 nearest-even (preserving sign, including signed zero),
// then BF16(code*s). In both modes W operands are
// BF16(E2M1(code)*BF16(2^(SF-127))); code8=-0. Weight decoding precedes
// backend F32 dot reduction and BF16 output, never post-dot group scaling.
//
// SF0 is2^-127; SF255 and nonfinite source A fail Completion. Legal finite
// source arithmetic may overflow to IEEE Inf/NaN. Failed output is unusable;
// no all-output rollback is promised. This explicit zero/extreme definition
// does not emulate upstream log2(0)-to-int behavior or optional Triton ties.
// No globals, implicit weight expansion or native FP4 GEMM equivalence.
struct Mxfp4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4d5846344c494e02ULL};
  static constexpr std::string_view operation_name = "mxfp4-reference-linear";
  Mxfp4ActivationMode activation_mode = Mxfp4ActivationMode::kFloat4Group32;
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<Mxfp4Linear>);
inline Operation mxfp4_linear(
    Mxfp4ActivationMode mode = Mxfp4ActivationMode::kFloat4Group32) noexcept {
  return Operation::custom(Mxfp4Linear{mode});
}

// Same arithmetic with four original ports: A[T,1-or-S,K], W[E,N,K/2],
// SF[E,N,K/32], checked I32 IDs[T,S] -> BF16[T,S,N]. All dimensions positive.
// A may also be a true zero-slot-stride view with contiguous K and aligned,
// nonoverlapping token rows; other ports are dense. Validate every ID before
// any selected expert address, consuming SF only from selected experts.
// In W4A4 mode shared A is quantized once per token (no expert-dependent input
// globals); independent slots each quantize once. W4A16 reads the original A
// rows at their declared strides. Duplicate expert IDs retain slots.
struct GroupedMxfp4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4d58463447525002ULL};
  static constexpr std::string_view operation_name = "grouped-mxfp4-reference-linear";
  Mxfp4ActivationMode activation_mode = Mxfp4ActivationMode::kFloat4Group32;
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4,4},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kFloat32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<GroupedMxfp4Linear>);
inline Operation grouped_mxfp4_linear(
    Mxfp4ActivationMode mode = Mxfp4ActivationMode::kFloat4Group32) noexcept {
  return Operation::custom(GroupedMxfp4Linear{mode});
}

// Group512 INT4-weight / INT8-activation software reference. Original ports:
// BF16 A[M,K], I32 W[N,K/8], F32 SF[N,K/512] -> BF16 Y[M,N]. W packs eight
// offset-binary signed codes per word, low nibble first: code=nibble-8.
// M/N/K are positive and K is a multiple of512; all ports are plain dense.
//
// Each BF16 A block computes s=BF16(absmax(A)*BF16(1/127)), then
// inv=s==0?0:BF16(1/s), q=truncate_to_I8(BF16(A*inv)). Each group computes
// d=BF16(I32_dot(q,W)), p=BF16(d*s), v=BF16(p*BF16(SF)), and updates a BF16
// accumulator in increasing K-group order: y=BF16(y+v), starting from+0.
// Every BF16 conversion is RNE, including subnormal values; these boundaries
// cannot be replaced by a dequantized F32 dot or one final BF16 conversion.
//
// Source A must be finite and original SF finite/positive. Derived q must be
// finite with its truncated value in[-128,127]; undefined float-to-I8 source
// conversions fail Completion rather than saturating or wrapping. A zero
// scale has inv=0. Legal arithmetic after quantization may yield Inf/NaN,
// including BF16(SF) overflow/underflow. Failed output is unusable and no
// all-output rollback is promised. No implicit floating weight bank is made.
struct BlockwiseW4A8Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x42573441384c4901ULL};
  static constexpr std::string_view operation_name = "blockwise-w4a8-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3,3},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kInt32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<BlockwiseW4A8Linear>);
inline Operation blockwise_w4a8_linear() noexcept {
  return Operation::custom(BlockwiseW4A8Linear{});
}

// Same arithmetic with four original ports: BF16 A[T,1-or-S,K],
// I32 W[E,N,K/8], F32 SF[E,N,K/512], I32 IDs[T,S] -> BF16 Y[T,S,N].
// A may also use a real zero-slot-stride view with contiguous K and aligned,
// nonoverlapping token rows; other ports are dense. All dimensions positive.
// Validate every ID in[0,E) before any expert address. SF is checked only for
// selected experts. Shared A is quantized once per token, independent A once
// per slot. Duplicate IDs preserve output slots; original strides are kept.
struct GroupedBlockwiseW4A8Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4257344138475201ULL};
  static constexpr std::string_view operation_name = "grouped-blockwise-w4a8-reference-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{4,4},1,AliasPolicy::kNone,EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType,DataType::kInt32,true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<GroupedBlockwiseW4A8Linear>);
inline Operation grouped_blockwise_w4a8_linear() noexcept {
  return Operation::custom(GroupedBlockwiseW4A8Linear{});
}

enum class Float8ScalePolicy : std::uint32_t {
  kDivide = 0,
  kPowerOfTwo = 1,
  kMultiplyReciprocal = 2,
};

enum class MlxWeightFormat : std::uint32_t { kAffine, kMxfp4, kMxfp8 };
enum class MlxQuantizedKind : std::uint32_t { kLinear, kGrouped, kEmbedding };
enum class MlxMatmulOrientation : std::uint32_t {
  kTransposeWeight,
  kWeightAsStored,
};

// Original MLX packed CPU formulas with F32 accumulation and explicit source
// storage. Codes are U32 words with low bits first. Affine scale/bias retain
// their float dtype; MX scales are original U8 E8M0 group32 values.
// Affine has explicit bits4/5/6/8; MX formats keep their fixed4/8 bits.
// Each row is a continuous bitstream: words=(K/32)*bits, including codes
// crossing a word boundary. Complete groups are32/64/128 for affine,32 for MX.
// Linear: A[M,K], W[N,words], scales[N,K/group], optional affine bias.
// Grouped: A[T,A,K], W[E,N,words], scales[E,N,K/group], optional bias,
// I32 ids[T,S] -> output[T,S,N], A=1 or S. A=1 and a real zero-slot-stride
// activation view share each token without expanding the activation. All other
// ports remain dense; a shared view has contiguous K and element-aligned,
// nonoverlapping token rows. Embedding: affine W/scales/bias and I32 ids[M].
// kTransposeWeight (default) computes A*W^T as above. Affine-only
// kWeightAsStored computes A[M,N]*W[N,K] -> Y[M,K], or grouped
// A[T,1 or S,N] -> Y[T,S,K]. Physical W/scale/bias layout and K grouping do
// not change; N need not contain complete groups. Each decoded affine value
// stays F32 before the F32 dot and the usual final output cast. There is no
// full floating-point W bank or implicit physical transpose. Embedding and
// MX formats only admit the default direction.
// Only selected experts/rows and bounded weight tiles are expanded during execution.
struct MlxQuantized final {
  MlxWeightFormat format = MlxWeightFormat::kAffine;
  MlxQuantizedKind kind = MlxQuantizedKind::kLinear;
  std::uint32_t bits = 4;
  std::uint32_t group_size = 64;
  MlxMatmulOrientation orientation = MlxMatmulOrientation::kTransposeWeight;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                            0x4d4c585150550003ULL};
  static constexpr std::string_view operation_name = "mlx-packed-quantized";
  constexpr OperationTraits operation_traits() const noexcept {
    const std::uint16_t count = kind == MlxQuantizedKind::kEmbedding ? 4 :
        static_cast<std::uint16_t>((format == MlxWeightFormat::kAffine ? 4 : 3) +
                                   (kind == MlxQuantizedKind::kGrouped ? 1 : 0));
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  HYLE_COMPUTE_OPS_API Status infer_output_descriptions(
      std::span<const TensorDesc> inputs, std::span<TensorDesc> outputs) const noexcept;
};
static_assert(detail::InlineCustomOperationDescriptor<MlxQuantized>);
inline Operation mlx_quantized_matmul(MlxWeightFormat format, std::uint32_t bits,
    std::uint32_t group,
    MlxMatmulOrientation orientation = MlxMatmulOrientation::kTransposeWeight) noexcept {
  return Operation::custom(MlxQuantized{format, MlxQuantizedKind::kLinear, bits, group, orientation});
}
inline Operation mlx_grouped_matmul(MlxWeightFormat format, std::uint32_t bits,
    std::uint32_t group,
    MlxMatmulOrientation orientation = MlxMatmulOrientation::kTransposeWeight) noexcept {
  return Operation::custom(MlxQuantized{format, MlxQuantizedKind::kGrouped, bits, group, orientation});
}
inline Operation mlx_quantized_embedding(std::uint32_t bits, std::uint32_t group) noexcept {
  return Operation::custom(MlxQuantized{MlxWeightFormat::kAffine,
                                      MlxQuantizedKind::kEmbedding, bits, group});
}

enum class BlockQuantizationFormat : std::uint32_t {
  kFloat8E4M3Block64,
  kFloat4E2M1Block32,
  kFloat8E4M3Block128
};

// Quantize and dequantize finite BF16 activations, preserving a model's QAT
// boundary without exposing packed cache storage. Per contiguous last-axis
// group, scale = 2^ceil(log2(max(absmax, floor)/max_code)), in F32.
// E4M3FN uses groups of64 or128, max_code448, floor1e-4; E2M1 uses groups of32,
// max_code6, floor6*2^-126. Codes use saturated nearest-even rounding; decode
// is F32 followed by BF16 storage. Finite inputs may overflow after rounding.
// Input/output are distinct dense BF16 matrices of identical shape. This is
// not the round-even scale policy used by some other MX quantizers.
struct BlockQuantizationRoundtrip final {
  BlockQuantizationFormat format = BlockQuantizationFormat::kFloat8E4M3Block64;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x514e545254500001ULL};
  static constexpr std::string_view operation_name = "block-quantization-roundtrip";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        (format != BlockQuantizationFormat::kFloat8E4M3Block64 &&
         format != BlockQuantizationFormat::kFloat4E2M1Block32 &&
         format != BlockQuantizationFormat::kFloat8E4M3Block128))
      return Status::InvalidArgument("Quantization roundtrip requires one input and a known format");
    const auto &input = inputs[0];
    const int group = format == BlockQuantizationFormat::kFloat4E2M1Block32 ? 32
        : (format == BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128);
    if (!input.valid() || !input.is_plain() || !input.is_dense() || input.rank() != 2 ||
        input.data_type() != DataType::kBFloat16 || input.dimensions()[0] <= 0 ||
        input.dimensions()[1] <= 0 || input.dimensions()[1] % group != 0)
      return Status::InvalidArgument("Quantization roundtrip requires a positive dense BF16 matrix of whole groups");
    outputs[0] = input;
    return Status::Ok();
  }
};

inline Operation block_quantization_roundtrip(BlockQuantizationFormat format) noexcept {
  return Operation::custom(BlockQuantizationRoundtrip{format});
}

// Decode an existing block128 weight for one-time native weight preparation.
// Input is the logical [K,N] encoded weight; output is physical row-major
// [N,K] in BF16/F32. This explicit layout is ready for model-owned splitting
// or packing. Execute once, then bind the resulting storage as a constant;
// descriptor preparation itself never reads payloads or creates a cache.
struct Float8WeightDecode final {
  DataType output_type = DataType::kBFloat16;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384445430001ULL};
  static constexpr std::string_view operation_name = "float8-weight-decode";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        (output_type != DataType::kBFloat16 && output_type != DataType::kFloat32))
      return Status::InvalidArgument("FP8 weight decode requires one block128 matrix and BF16/F32 output");
    const auto k = inputs[0].dimensions()[0], n = inputs[0].dimensions()[1];
    auto encoded = float8_weight_description(k, n, Float8ScaleMode::kFloat32Block128);
    if (!encoded || inputs[0] != *encoded)
      return Status::InvalidArgument("FP8 weight decode requires the exact E4M3FN/F32 block128 storage");
    auto output = TensorDesc::dense(output_type, std::array{n, k});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation float8_weight_decode(DataType output_type = DataType::kBFloat16) noexcept {
  return Operation::custom(Float8WeightDecode{output_type});
}

// Complete model linear: dynamically quantize finite [M,K] activations by
// groups of 128 to E4M3FN, then multiply the already encoded block128 weight.
// First take the F32 absolute maximum and clamp it by absmax_floor.
// kDivide stores F32(maximum/448). kMultiplyReciprocal stores
// F32(maximum*F32(1/448)), then normalizes with two separate F32 stages:
// reciprocal=F32(1/scale), normalized=F32(activation*reciprocal).
// kPowerOfTwo forms the same multiply-based raw scale, then rounds it
// upward to a power of two and retains the original normalization.
// No BF16 intermediate is inserted. Codes use saturated nearest-even
// conversion. Scales remain F32; this is not a group32 storage format.
// BF16/F32 output preserves activation dtype after F32 accumulation. The
// floor is explicit, for example 1e-10 or official DeepSeek's 1e-4.
struct Float8BlockLinear final {
  Float8ScalePolicy scale_policy = Float8ScalePolicy::kDivide;
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384c494e0002ULL};
  static constexpr std::string_view operation_name = "float8-block-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 ||
        (scale_policy != Float8ScalePolicy::kDivide &&
         scale_policy != Float8ScalePolicy::kPowerOfTwo &&
         scale_policy != Float8ScalePolicy::kMultiplyReciprocal) ||
        !std::isfinite(absmax_floor) || absmax_floor <= 0 ||
        !std::isnormal(absmax_floor / 448.0F))
      return Status::InvalidArgument("FP8 linear requires a known scale policy and positive normal scale floor");
    const auto &a = inputs[0];
    const auto &b = inputs[1];
    if (!a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 2 ||
        b.rank() != 2 || (a.data_type() != DataType::kFloat32 &&
                         a.data_type() != DataType::kBFloat16))
      return Status::InvalidArgument("FP8 linear requires dense BF16/F32 [M,K] and block128 weight [K,N]");
    const auto m = a.dimensions()[0], k = a.dimensions()[1], n = b.dimensions()[1];
    auto weight = float8_weight_description(k, n, Float8ScaleMode::kFloat32Block128);
    if (m <= 0 || !weight || b != *weight)
      return Status::InvalidArgument("FP8 linear requires positive dimensions and the exact E4M3FN/block128 weight layout");
    auto output = TensorDesc::dense(a.data_type(), std::array{m, n});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};
static_assert(sizeof(Float8BlockLinear) == 8);

inline Operation float8_block_linear(
    Float8ScalePolicy policy = Float8ScalePolicy::kDivide,
    float absmax_floor = 1.0e-10F) noexcept {
  return Operation::custom(Float8BlockLinear{policy, absmax_floor});
}

// Expose the dynamic activation quantization of Float8BlockLinear without a
// dot product. Input is dense BF16/F32 [M,K], M>0, K>0 and K%128==0.
// Output0 is plain E4M3FN [M,K]; output1 is plain F32 [M,K/128], in row-major
// contiguous-group order, with positive finite scales. These are separate
// typed tensors, not an encoded
// carrier or decoded floating cache. Absmax clamping, the three scale policies
// and F32 normalization follow Float8BlockLinear exactly, with no implicit
// BF16 boundary. Codes use finite-saturating nearest-even rounding and retain
// signed zero. Every input value must be finite; a runtime violation fails
// Completion. Failed outputs must not be consumed; partial writes need not
// roll back. Both outputs are distinct from the input and from each other.
struct Float8BlockQuantize final {
  Float8ScalePolicy scale_policy = Float8ScalePolicy::kDivide;
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x465038514e540001ULL};
  static constexpr std::string_view operation_name = "float8-block-quantize";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 2 ||
        (scale_policy != Float8ScalePolicy::kDivide &&
         scale_policy != Float8ScalePolicy::kPowerOfTwo &&
         scale_policy != Float8ScalePolicy::kMultiplyReciprocal) ||
        !std::isfinite(absmax_floor) || absmax_floor <= 0 ||
        !std::isnormal(absmax_floor / 448.0F))
      return Status::InvalidArgument("FP8 quantize requires a known scale policy and positive normal scale floor");
    const auto &a = inputs[0];
    if (!a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 2 ||
        (a.data_type() != DataType::kFloat32 &&
         a.data_type() != DataType::kBFloat16))
      return Status::InvalidArgument("FP8 quantize requires dense BF16/F32 [M,K]");
    const auto m = a.dimensions()[0], k = a.dimensions()[1];
    if (m <= 0 || k <= 0 || k % 128 != 0)
      return Status::InvalidArgument("FP8 quantize requires positive dimensions and complete groups of 128");
    auto codes = TensorDesc::dense(DataType::kFloat8E4M3FN, std::array{m, k});
    if (!codes) return std::move(codes).error();
    auto scales = TensorDesc::dense(DataType::kFloat32, std::array{m, k / 128});
    if (!scales) return std::move(scales).error();
    outputs[0] = *codes;
    outputs[1] = *scales;
    return Status::Ok();
  }
};
static_assert(sizeof(Float8BlockQuantize) == 8);
static_assert(detail::InlineCustomOperationDescriptor<Float8BlockQuantize>);

inline Operation float8_block_quantize(
    Float8ScalePolicy policy = Float8ScalePolicy::kDivide,
    float absmax_floor = 1.0e-10F) noexcept {
  return Operation::custom(Float8BlockQuantize{policy, absmax_floor});
}

// Weights [E,N,K], finite BF16/F32 activations [T,A,K], I32 expert IDs [T,S]
// -> [T,S,N] in activation dtype. A=1 shares each token across selected
// experts; A=S preserves distinct per-expert activations for the down leg.
// Quantization follows Float8BlockLinear, including its explicit scale
// policy and F32 group128 scales.
// Duplicate IDs are valid. Negative or out-of-range IDs fail Completion;
// outputs of a failed invocation must not be consumed.
struct GroupedFloat8Linear final {
  Float8ScalePolicy scale_policy = Float8ScalePolicy::kDivide;
  float absmax_floor = 1.0e-10F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650384752500003ULL};
  static constexpr std::string_view operation_name = "grouped-float8-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 ||
        (scale_policy != Float8ScalePolicy::kDivide &&
         scale_policy != Float8ScalePolicy::kPowerOfTwo &&
         scale_policy != Float8ScalePolicy::kMultiplyReciprocal) ||
        !std::isfinite(absmax_floor) || absmax_floor <= 0 ||
        !std::isnormal(absmax_floor / 448.0F))
      return Status::InvalidArgument("grouped FP8 linear requires a known scale policy and positive normal scale floor");
    const auto &w = inputs[0];
    const auto &a = inputs[1];
    const auto &ids = inputs[2];
    if (w.rank() != 3 || !a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 3 ||
        (a.data_type() != DataType::kFloat32 && a.data_type() != DataType::kBFloat16) ||
        !ids.valid() || !ids.is_plain() || !ids.is_dense() || ids.rank() != 2 ||
        ids.data_type() != DataType::kInt32)
      return Status::InvalidArgument("grouped FP8 linear requires encoded [E,N,K], dense BF16/F32 [T,A,K], I32 IDs [T,S]");
    const auto t = a.dimensions()[0], slots = ids.dimensions()[1];
    auto weight = float8_grouped_weight_description(
        w.dimensions()[0], w.dimensions()[1], w.dimensions()[2]);
    if (!weight || w != *weight || t <= 0 || slots <= 0 || ids.dimensions()[0] != t ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != slots) ||
        a.dimensions()[2] != w.dimensions()[2])
      return Status::InvalidArgument("grouped FP8 dimensions or encoded layout do not match");
    auto output = TensorDesc::dense(a.data_type(), std::array{t, slots, w.dimensions()[1]});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};
static_assert(sizeof(GroupedFloat8Linear) == 8);

inline Operation grouped_float8_linear(
    Float8ScalePolicy policy = Float8ScalePolicy::kDivide,
    float absmax_floor = 1.0e-10F) noexcept {
  return Operation::custom(GroupedFloat8Linear{policy, absmax_floor});
}

// Original FP4 experts: encoded E2M1/F32 group32 [E,N,K], finite dense BF16
// activations [T,A,K], I32 IDs [T,S] -> BF16 [T,S,N]. A=1 or S; K%128=0.
// Each activation group128 is quantized to E4M3FN with absmax floor1e-4,
// scale rounded upward to power2 and saturated nearest-even codes. For each
// K-block32, dot the decoded E4M3FN/E2M1 codes with F32 accumulation, multiply
// that F32 dot by its activation group128 scale and then its weight group32
// scale in F32, and add the scaled block to the outer F32 accumulator. Round
// to BF16 nearest-even only at the final output store. F32 summation may be
// reordered; an implementation must preserve F32 effective accumulation
// precision rather than only using an F32 result container. Lossless code
// widening for the dot is permitted; no second Q8 activation conversion or
// intermediate BF16 partial/scale rounding is implied.
// Duplicate IDs are valid; out-of-range IDs fail Completion. Failed outputs
// are not consumable, and no cross-row rollback is promised.
struct GroupedFloat4Linear final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4650344752500001ULL};
  static constexpr std::string_view operation_name = "grouped-float4-linear";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1)
      return Status::InvalidArgument("grouped FP4 linear requires three inputs and one output");
    const auto &w = inputs[0];
    const auto &a = inputs[1];
    const auto &ids = inputs[2];
    if (w.rank() != 3 || !a.valid() || !a.is_plain() || !a.is_dense() || a.rank() != 3 ||
        a.data_type() != DataType::kBFloat16 || !ids.valid() || !ids.is_plain() ||
        !ids.is_dense() || ids.rank() != 2 || ids.data_type() != DataType::kInt32)
      return Status::InvalidArgument("grouped FP4 linear requires encoded [E,N,K], BF16 [T,A,K], I32 [T,S]");
    const auto t = a.dimensions()[0], slots = ids.dimensions()[1];
    auto weight = float4_grouped_weight_description(
        w.dimensions()[0], w.dimensions()[1], w.dimensions()[2]);
    if (!weight || w != *weight || w.dimensions()[2] % 128 != 0 || t <= 0 ||
        slots <= 0 || ids.dimensions()[0] != t ||
        (a.dimensions()[1] != 1 && a.dimensions()[1] != slots) ||
        a.dimensions()[2] != w.dimensions()[2])
      return Status::InvalidArgument("grouped FP4 dimensions or encoded layout do not match");
    auto output = TensorDesc::dense(DataType::kBFloat16, std::array{t, slots, w.dimensions()[1]});
    if (!output) return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_float4_linear() noexcept {
  return Operation::custom(GroupedFloat4Linear{});
}

// Native model clamp. Finite values and infinities are clamped to [min,max].
// NaN data and signed-zero bit patterns follow the selected native backend;
// this operation does not promise Binary minimum/maximum's NaN semantics.
struct Clamp final {
  float minimum;
  float maximum;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x434c414d50000001ULL};
  static constexpr std::string_view operation_name = "model-clamp";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{1, 1}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (std::isnan(minimum) || std::isnan(maximum) || minimum > maximum ||
        inputs.size() != 1 || outputs.size() != 1)
      return Status::InvalidArgument("Clamp requires ordered non-NaN bounds and one input/output");
    const auto &input = inputs[0];
    if (!input.valid() || !input.is_plain() || !input.is_dense() ||
        input.data_type() != DataType::kFloat32 || input.rank() < 1 ||
        input.rank() > 4 || !input.element_count() || *input.element_count() == 0)
      return Status::InvalidArgument("Clamp requires nonempty plain dense rank 1-4 F32");
    outputs[0] = input;
    return Status::Ok();
  }
};

inline Operation clamp(float minimum, float maximum) noexcept {
  return Operation::custom(Clamp{minimum, maximum});
}

// Complete DeepSeek score -> biased TopK -> unbiased normalized weights.
// Selected sqrt(softplus) scores are divided by their sum, then scaled;
// no epsilon or zero-sum fallback is applied (zero scores yield NaN).
// Ties may select any valid top-k member; this is not stable generic TopK.
// The CUDA SGLang carrier consumes finite F32 model logits and bias. Other
// routes may express this model fragment with their mature typed operations.
struct SqrtSoftplusRouting final {
  float scaling = 1.0F;
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x445334524f555409ULL};
  static constexpr std::string_view operation_name = "sqrt-softplus-routing";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (!std::isfinite(scaling) || scaling <= 0 || inputs.size() != 2 ||
        outputs.size() != 2)
      return Status::InvalidArgument("SqrtSoftplusRouting requires positive finite scale and two inputs/outputs");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense() ||
          input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument("SqrtSoftplusRouting requires plain dense F32");
    if (inputs[0].rank() != 2 || inputs[1].rank() != 1)
      return Status::InvalidArgument("SqrtSoftplusRouting requires logits [T,E] and bias [E]");
    const auto shape = inputs[0].dimensions();
    if (shape[0] <= 0 || (shape[1] != 256 && shape[1] != 384) ||
        inputs[1].dimensions()[0] != shape[1])
      return Status::InvalidArgument("SqrtSoftplusRouting requires nonempty tokens and E256/384");
    const std::array output_shape{shape[0], std::int64_t{6}};
    auto weights = TensorDesc::dense(DataType::kFloat32, output_shape);
    auto ids = TensorDesc::dense(DataType::kInt32, output_shape);
    if (!weights) return std::move(weights).error();
    if (!ids) return std::move(ids).error();
    outputs[0] = *weights;
    outputs[1] = *ids;
    return Status::Ok();
  }
};

inline Operation sqrt_softplus_routing(float scaling = 1.0F) noexcept {
  return Operation::custom(SqrtSoftplusRouting{scaling});
}

// DeepSeek hash layers select six experts from a token-to-expert table.
// Inputs: F32 logits [T,E], I64 token IDs [T], I32 table [V,6]. Outputs:
// F32 weights [T,6] and I32 IDs [T,6], in table order (duplicates retained).
// Weights divide selected sqrt(softplus) scores by their sum, without an
// epsilon or zero-sum fallback (zero scores yield NaN); scaling is separate.
// Invalid token/selected-expert IDs fail Completion without an invalid read.
// Failed outputs cannot be consumed; there is no cross-row rollback promise.
struct SqrtSoftplusHashRouting final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4453344841534802ULL};
  static constexpr std::string_view operation_name = "sqrt-softplus-hash-routing";
  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }
  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 2)
      return Status::InvalidArgument("Hash routing requires three inputs and two outputs");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense())
        return Status::InvalidArgument("Hash routing requires plain dense inputs");
    if (inputs[0].data_type() != DataType::kFloat32 || inputs[0].rank() != 2 ||
        inputs[1].data_type() != DataType::kInt64 || inputs[1].rank() != 1 ||
        inputs[2].data_type() != DataType::kInt32 || inputs[2].rank() != 2)
      return Status::InvalidArgument("Hash routing requires F32 logits, I64 tokens and an I32 table");
    const auto logits = inputs[0].dimensions();
    const auto table = inputs[2].dimensions();
    if (logits[0] <= 0 || (logits[1] != 256 && logits[1] != 384) ||
        inputs[1].dimensions()[0] != logits[0] || table[0] <= 0 || table[1] != 6)
      return Status::InvalidArgument("Hash routing requires nonempty T/V, E256/384 and six table columns");
    const std::array shape{logits[0], std::int64_t{6}};
    auto weights = TensorDesc::dense(DataType::kFloat32, shape);
    auto ids = TensorDesc::dense(DataType::kInt32, shape);
    if (!weights) return std::move(weights).error();
    if (!ids) return std::move(ids).error();
    outputs[0] = *weights;
    outputs[1] = *ids;
    return Status::Ok();
  }
};

inline Operation sqrt_softplus_hash_routing() noexcept {
  return Operation::custom(SqrtSoftplusHashRouting{});
}

// DeepSeek's complete masked attention with a learned, unmasked sink per head.
// Q/K/V are BSHD F32. The sink participates only in the softmax denominator.
// The explicit additive mask already describes causality and cache selection.
// [T,S] is shared across batches; [B,T,S] permits independent selected KV sets.
struct SinkAttention final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x53494e4b41545401ULL};
  static constexpr std::string_view operation_name = "sink-attention";
  float scale = 1.0F;

  constexpr OperationTraits operation_traits() const noexcept {
    return {{5, 5}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32, true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 5 || outputs.size() != 1 ||
        !std::isfinite(scale) || scale <= 0)
      return Status::InvalidArgument("SinkAttention requires five inputs, one output and positive finite scale");
    for (const auto &input : inputs)
      if (!input.valid() || !input.is_plain() || !input.is_dense() ||
          input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument("SinkAttention requires plain dense F32 inputs");
    if (inputs[0].rank() != 4 || inputs[1].rank() != 4 || inputs[2] != inputs[1] ||
        (inputs[3].rank() != 2 && inputs[3].rank() != 3) || inputs[4].rank() != 1)
      return Status::InvalidArgument("SinkAttention requires Q/K/V BSHD, mask [T,S] or [B,T,S] and sinks [H]");
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    const auto mask = inputs[3].dimensions();
    const auto mask_query = mask.size() - 2;
    if (q[0] <= 0 || k[0] != q[0] || q[1] <= 0 || k[1] <= 0 ||
        (q[2] != 64 && q[2] != 128) || k[2] != 1 || q[3] != 512 || k[3] != 512 ||
        (mask.size() == 3 && mask[0] != q[0]) ||
        mask[mask_query] != q[1] || mask.back() != k[1] || inputs[4].dimensions()[0] != q[2])
      return Status::InvalidArgument("SinkAttention requires matching positive B, H64/128, KV1, D512 and matching mask/sinks");
    const std::array score_shape{q[0], q[2], q[1], k[1]};
    auto scores = TensorDesc::dense(DataType::kFloat32, score_shape);
    if (!scores) return std::move(scores).error();
    auto score_bytes = scores->required_bytes();
    if (!score_bytes) return std::move(score_bytes).error();
    outputs[0] = inputs[0];
    return Status::Ok();
  }
};

inline Operation sink_attention(float scale) noexcept {
  return Operation::custom(SinkAttention{scale});
}

// Depthwise causal-window convolution used by recurrent LLM blocks. The
// caller supplies the K-1 history prefix explicitly, so replay is stateless:
//   samples [B, T + K - 1, C], weights [C, K] -> output [B, T, C].
// Providers keep their native layout and may fuse following bias/activation
// when the graph boundary and exact semantics permit it.
struct CausalConv1d final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x434f4e5631440001ULL};
  static constexpr std::string_view operation_name = "causal-conv1d";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 2 || !inputs[0].is_plain() ||
        !inputs[1].is_plain() || !inputs[0].is_dense() ||
        !inputs[1].is_dense()) {
      return Status::InvalidArgument(
          "CausalConv1d requires dense samples [B,W,C] and weights [C,K]");
    }
    const auto type = inputs[0].data_type();
    if ((type != DataType::kFloat16 && type != DataType::kBFloat16 &&
         type != DataType::kFloat32) ||
        inputs[1].data_type() != type) {
      return Status::InvalidArgument(
          "CausalConv1d requires matching F16/BF16/F32 inputs");
    }
    const auto samples = inputs[0].dimensions();
    const auto weights = inputs[1].dimensions();
    if (samples[0] <= 0 || samples[1] <= 0 || samples[2] <= 0 ||
        weights[0] != samples[2] || weights[1] <= 0 ||
        samples[1] < weights[1]) {
      return Status::InvalidArgument(
          "CausalConv1d channel/kernel dimensions do not match");
    }
    const std::array shape{samples[0], samples[1] - weights[1] + 1,
                           samples[2]};
    auto output = TensorDesc::dense(type, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation causal_conv1d() noexcept {
  return Operation::custom(CausalConv1d{});
}

// Scalar-gated recurrent region. Q/K are already normalized and use
// GatedDeltaNet's cyclic head mapping, QKV types and explicit state contract.
// Inputs are Q, K, V, raw a, raw b, decay coefficient, dt bias, state;
// outputs are attended values and state. Raw a/b are [B,T,Hv,1];
// coefficient and dt bias are shared per-head [1,1,Hv,1] parameters.
// snapshot_count follows GatedDeltaNet's state-output contract.
// The provider owns softplus(a + dt) * negative_a and sigmoid(b) internally.
// Their numerical qualification belongs to the complete recurrent region,
// independently of the generic Unary softplus contract.
struct RecurrentGatedDeltaNet final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x47444e4741540009ULL};
  static constexpr std::string_view operation_name = "recurrent-gated-delta-net";
  std::uint32_t snapshot_count = 1;
  std::optional<float> output_scale{};

  constexpr OperationTraits operation_traits() const noexcept {
    return {{8, 8}, 2, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 8 || outputs.size() != 2)
      return Status::InvalidArgument(
          "RecurrentGatedDeltaNet requires eight inputs and two outputs");
    // Ordinary GDN is the single authority for Q/K/V type, shape, cyclic
    // head grouping, state orientation and snapshot-count/output rules.
    const std::array recurrent{inputs[0], inputs[1], inputs[2], inputs[3],
                               inputs[4], inputs[7]};
    std::array<TensorDesc, 2> inferred;
    const auto status = compute::infer_output_descriptions(
        Operation(GatedDeltaNet{.snapshot_count = snapshot_count, .output_scale = output_scale}),
        recurrent, inferred);
    if (!status.ok()) return status;
    const auto value = inputs[2].dimensions();
    const std::array gate_shape{value[0], value[1], value[2], std::int64_t{1}};
    const std::array parameter_shape{std::int64_t{1}, std::int64_t{1},
                                     value[2], std::int64_t{1}};
    for (std::size_t port = 3; port != 7; ++port) {
      const auto &input = inputs[port];
      const auto &shape = port < 5 ? gate_shape : parameter_shape;
      if (!input.valid() || !input.is_plain() ||
          input.data_type() != DataType::kFloat32 || input.rank() != 4 ||
          !std::ranges::equal(input.dimensions(), shape))
        return Status::InvalidArgument(
            "RecurrentGatedDeltaNet requires F32 raw gates [B,T,Hv,1] and shared parameters [1,1,Hv,1]");
    }
    // Do not publish one output before all composite-port checks succeed.
    outputs[0] = inferred[0];
    outputs[1] = inferred[1];
    return Status::Ok();
  }
};

inline Operation recurrent_gated_delta_net(
    std::uint32_t snapshot_count = 1,
    std::optional<float> output_scale = {}) noexcept {
  return Operation::custom(RecurrentGatedDeltaNet{
      .snapshot_count = snapshot_count, .output_scale = output_scale});
}

// Expert-selecting matrix multiplication.  The expert id is chosen for every
// token/slot pair while the public tensors keep their natural model layout:
//   weights [E,N,K], activations [T,A,K], ids [T,S] -> [T,S,N], A=1 or S.
// A=1 shares each token across slots. Read-only activation layouts may also
// express broadcast metadata views; concrete layouts are provider capabilities.
// Plain dense expert weights may be F32 or BF16. Activations, accumulation and
// output remain F32; exact quantized storage mappings are separate provider
// capabilities rather than implicit conversions here.
struct GroupedMatMul final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4d554c4944000002ULL};
  static constexpr std::string_view operation_name = "grouped-matmul";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 3 || inputs[2].rank() != 2 ||
        !inputs[0].is_dense() || !inputs[1].valid() || !inputs[1].is_plain() ||
        !inputs[2].is_dense())
      return Status::InvalidArgument(
          "GroupedMatMul requires dense [E,N,K], plain [T,A,K], dense ids [T,S]");
    if ((inputs[0].data_type() != DataType::kFloat32 &&
         (inputs[0].data_type() != DataType::kBFloat16 ||
          !inputs[0].is_plain())) ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kInt32)
      return Status::InvalidArgument(
          "GroupedMatMul requires F32 or plain BF16 weights, F32 activations "
          "and I32 ids");
    const auto weights = inputs[0].dimensions();
    const auto activations = inputs[1].dimensions();
    const auto ids = inputs[2].dimensions();
    if (weights[0] <= 0 || weights[1] <= 0 || weights[2] <= 0 ||
        activations[0] <= 0 || activations[1] <= 0 ||
        activations[2] != weights[2] || ids[0] != activations[0] ||
        ids[1] <= 0 || (activations[1] != 1 && activations[1] != ids[1]))
      return Status::InvalidArgument("GroupedMatMul dimensions do not match");
    const std::array shape{activations[0], ids[1], weights[1]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

// DeepSeek V4 multi-stream hyper-connection primitives.  hc is deliberately
// fixed to four because that is the exact mature-provider/model contract.
struct HyperConnectionCombine final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843434f4d420003ULL};
  static constexpr std::string_view operation_name = "hyper-connection-combine";
  float epsilon = 1.0e-6F;
  std::int32_t iterations = 20;

  constexpr OperationTraits operation_traits() const noexcept {
    return {{3, 3}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 3 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        inputs[1].rank() != 1 || inputs[2].rank() != 1 ||
        !inputs[0].is_dense() || !inputs[1].is_dense() ||
        !inputs[2].is_dense())
      return Status::InvalidArgument(
          "HyperConnectionCombine requires dense [T,24], [3], [24]");
    if (inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kFloat32 || !(epsilon >= 0.0F) ||
        iterations <= 0)
      return Status::InvalidArgument(
          "HyperConnectionCombine requires F32 inputs and valid cold parameters");
    const auto mixes = inputs[0].dimensions();
    if (mixes[0] <= 0 || mixes[1] != 24 ||
        inputs[1].dimensions()[0] != 3 || inputs[2].dimensions()[0] != 24)
      return Status::InvalidArgument(
          "HyperConnectionCombine dimensions do not match hc=4");
    const std::array shape{mixes[0], std::int64_t{4}, std::int64_t{4}};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

struct HyperConnectionPre final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843505245000004ULL};
  static constexpr std::string_view operation_name = "hyper-connection-pre";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].rank() != 3 ||
        inputs[1].rank() != 2 || !inputs[0].is_dense() ||
        !inputs[1].is_dense() ||
        inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32)
      return Status::InvalidArgument(
          "HyperConnectionPre requires dense F32 [T,4,D], [T,4]");
    const auto value = inputs[0].dimensions();
    const auto weights = inputs[1].dimensions();
    if (value[0] <= 0 || value[1] != 4 || value[2] <= 0 ||
        weights[0] != value[0] || weights[1] != 4)
      return Status::InvalidArgument(
          "HyperConnectionPre dimensions do not match hc=4");
    const std::array shape{value[0], value[2]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

struct HyperConnectionPost final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4843504f53540005ULL};
  static constexpr std::string_view operation_name = "hyper-connection-post";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 4 || outputs.size() != 1 || inputs[0].rank() != 2 ||
        inputs[1].rank() != 3 || inputs[2].rank() != 2 ||
        inputs[3].rank() != 3)
      return Status::InvalidArgument(
          "HyperConnectionPost requires [T,D], [T,4,D], [T,4], [T,4,4]");
    for (const auto &input : inputs)
      if (!input.is_dense() || input.data_type() != DataType::kFloat32)
        return Status::InvalidArgument(
            "HyperConnectionPost requires dense F32 inputs");
    const auto value = inputs[0].dimensions();
    const auto residual = inputs[1].dimensions();
    const auto post = inputs[2].dimensions();
    const auto combine = inputs[3].dimensions();
    if (value[0] <= 0 || value[1] <= 0 || residual[0] != value[0] ||
        residual[1] != 4 || residual[2] != value[1] ||
        post[0] != value[0] || post[1] != 4 ||
        combine[0] != value[0] || combine[1] != 4 || combine[2] != 4)
      return Status::InvalidArgument(
          "HyperConnectionPost dimensions do not match hc=4");
    auto output = TensorDesc::dense(DataType::kFloat32, residual);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

// DeepSeek/GLM learned sparse-attention score region.  We expose model-order
// tensors while the provider retains its native score kernel:
//   query   [S,T,H,D]
//   key     [S,K,1,D]
//   weights [S,1,T,H] (already prescaled)
//   mask    [Sm,1,T,K] (F16, Sm broadcasts across S)
//   output  [S,1,T,K].
// The first common four-route domain matches the original Metal kernel;
// route-specific supersets remain provider qualifications.
struct LightningIndexer final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4c494e4458000006ULL};
  static constexpr std::string_view operation_name = "lightning-indexer";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{4, 4}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 4 || outputs.size() != 1)
      return Status::InvalidArgument("LightningIndexer requires four inputs");
    for (const auto &input : inputs)
      if (input.rank() != 4 || !input.is_dense())
        return Status::InvalidArgument(
            "LightningIndexer requires dense rank-four inputs");
    if (inputs[0].data_type() != DataType::kFloat32 ||
        inputs[1].data_type() != DataType::kFloat32 ||
        inputs[2].data_type() != DataType::kFloat32 ||
        inputs[3].data_type() != DataType::kFloat16)
      return Status::InvalidArgument(
          "LightningIndexer requires F32 query/key/weights and F16 mask");
    const auto q = inputs[0].dimensions();
    const auto k = inputs[1].dimensions();
    const auto weights = inputs[2].dimensions();
    const auto mask = inputs[3].dimensions();
    if (q[0] <= 0 || q[1] <= 0 || q[2] != 64 || q[3] != 128 ||
        k[0] != q[0] || k[1] <= 0 || k[2] != 1 || k[3] != q[3] ||
        weights[0] != q[0] || weights[1] != 1 || weights[2] != q[1] ||
        weights[3] != q[2] || mask[0] <= 0 || q[0] % mask[0] != 0 ||
        mask[1] != 1 || mask[2] != q[1] || mask[3] != k[1])
      return Status::InvalidArgument(
          "LightningIndexer dimensions do not match D128/H64 sparse scoring");
    const std::array shape{q[0], std::int64_t{1}, q[1], k[1]};
    auto output = TensorDesc::dense(DataType::kFloat32, shape);
    if (!output)
      return std::move(output).error();
    outputs[0] = *output;
    return Status::Ok();
  }
};

inline Operation grouped_matmul() noexcept {
  return Operation::custom(GroupedMatMul{});
}
// Model linear projection: BF16 or F32 activations [T,K], stored output-major
// BF16 weights [E,K] -> F32 [T,E]. Exact batched [B,T,K] x [B,E,K]
// additionally produces [B,T,E], without implicit batch broadcasting.
// Products and accumulation use F32; F32
// activations must not be implicitly narrowed. BF16 weights remain in their
// original storage and may be widened on load, without a full F32 weight bank.
// Ordinary MatMul retains its same-dtype contract.
struct BFloatLinearF32 final {
  static constexpr OperationId operation_id{0x48594c454d4f444cULL,
                                             0x4d41544d554c4633ULL};
  static constexpr std::string_view operation_name = "bfloat-linear-f32";

  constexpr OperationTraits operation_traits() const noexcept {
    return {{2, 2}, 1, AliasPolicy::kNone, EffectKind::kPure,
            PrecisionPolicy::kAccumulateInDeclaredType, DataType::kFloat32,
            true};
  }

  Status infer_output_descriptions(std::span<const TensorDesc> inputs,
                                   std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 2 || outputs.size() != 1 ||
        (inputs[0].rank() != 2 && inputs[0].rank() != 3) ||
        inputs[1].rank() != inputs[0].rank() || !inputs[0].is_plain() ||
        !inputs[1].is_plain() || !inputs[0].is_dense() ||
        !inputs[1].is_dense() ||
        (inputs[0].data_type() != DataType::kBFloat16 &&
         inputs[0].data_type() != DataType::kFloat32) ||
        inputs[1].data_type() != DataType::kBFloat16)
      return Status::InvalidArgument(
          "BFloatLinearF32 requires dense BF16/F32 [T,K]/[B,T,K] and aligned BF16 weights");
    const auto lhs = inputs[0].dimensions();
    const auto rhs = inputs[1].dimensions();
    const auto rank = inputs[0].rank();
    if (lhs[rank-2] <= 0 || lhs[rank-1] <= 0 || rhs[rank-2] <= 0 ||
        rhs[rank-1] != lhs[rank-1] ||
        (rank == 3 && (lhs[0] <= 0 || rhs[0] != lhs[0])))
      return Status::InvalidArgument("BFloatLinearF32 dimensions do not match");
    std::array<std::int64_t,3> shape{};
    for (std::size_t axis=0;axis<rank;++axis) shape[axis]=lhs[axis];
    shape[rank-1]=rhs[rank-2];
    auto result = TensorDesc::dense(DataType::kFloat32, std::span(shape).first(rank));
    if (!result)
      return std::move(result).error();
    outputs[0] = *result;
    return Status::Ok();
  }
};

inline Operation bfloat_linear_f32() noexcept {
  return Operation::custom(BFloatLinearF32{});
}
inline Operation hyper_connection_combine(float epsilon = 1.0e-6F,
                                           std::int32_t iterations = 20) noexcept {
  return Operation::custom(HyperConnectionCombine{epsilon, iterations});
}
inline Operation hyper_connection_pre() noexcept {
  return Operation::custom(HyperConnectionPre{});
}
inline Operation hyper_connection_post() noexcept {
  return Operation::custom(HyperConnectionPost{});
}
inline Operation lightning_indexer() noexcept {
  return Operation::custom(LightningIndexer{});
}

} // namespace hyle::compute::model_ops
