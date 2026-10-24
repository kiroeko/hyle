#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>
#include "hyle/compute/model_ops.h"
#include "hyle/simd/dtype.h"
#include "mlx_affine_bitstream.h"

namespace mlx_grouped_shared_test {
namespace c = hyle::compute;
template<class T> inline std::vector<std::byte> bytes(std::span<const T> values) {
  const auto raw = std::as_bytes(values);
  return {raw.begin(), raw.end()};
}
inline std::vector<std::byte> floating(c::DataType type, const std::vector<float> &values) {
  if (type == c::DataType::kFloat32) return bytes<float>(values);
  std::vector<hyle::simd::bfloat16_t> encoded;
  for (float value : values) encoded.push_back(hyle::simd::f32_to_bf16(value));
  return bytes<hyle::simd::bfloat16_t>(encoded);
}
// The oracle is closed-form: all64 codes in each expert/channel have the
// same value. Bit packing uses the independent single-bit test writer.
struct Case final {
  c::Operation operation = c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine,5,32);
  std::vector<c::TensorDesc> descriptions;
  std::vector<std::vector<std::byte>> payloads;
  std::vector<float> expected;
  Case(c::DataType type, bool zero_slot_stride) {
    constexpr std::int64_t k=64, experts=3, n=2, slots=3;
    const auto width=static_cast<std::int64_t>(c::data_type_size(type));
    const auto activation=zero_slot_stride
        ? c::TensorDesc::strided_bytes(type,std::array<std::int64_t,3>{2,slots,k},
            std::array<std::int64_t,3>{80*width,0,width}).value()
        : c::TensorDesc::dense(type,std::array<std::int64_t,3>{2,1,k}).value();
    descriptions={activation,
        c::TensorDesc::dense(c::DataType::kUInt32,std::array<std::int64_t,3>{experts,n,10}).value(),
        c::TensorDesc::dense(type,std::array<std::int64_t,3>{experts,n,2}).value(),
        c::TensorDesc::dense(type,std::array<std::int64_t,3>{experts,n,2}).value(),
        c::TensorDesc::dense(c::DataType::kInt32,std::array<std::int64_t,2>{2,slots}).value()};
    std::vector<float> a(zero_slot_stride?144:128,-777.F);
    std::fill_n(a.begin(),k,1.F);
    std::fill_n(a.begin()+(zero_slot_stride?80:64),k,2.F);
    std::vector<std::uint32_t> codes(experts*n*k);
    for(std::int64_t e=0;e<experts;++e)for(std::int64_t channel=0;channel<n;++channel)
      std::fill_n(codes.begin()+(e*n+channel)*k,k,static_cast<std::uint32_t>((e+1)*(channel+1)));
    const auto words=mlx_affine_test::pack(codes,experts*n,k,5);
    const std::array<std::int32_t,6> ids{2,0,2,1,1,0};
    payloads={floating(type,a),bytes<std::uint32_t>(words),
        floating(type,std::vector<float>(12,.5F)),floating(type,std::vector<float>(12,-.25F)),bytes<std::int32_t>(ids)};
    for(std::int64_t token=0;token<2;++token)for(std::int64_t slot=0;slot<slots;++slot)
      for(std::int64_t channel=0;channel<n;++channel)
        expected.push_back(64.F*float(token+1)*(.5F*float((ids[token*slots+slot]+1)*(channel+1))-.25F));
  }
};
} // namespace mlx_grouped_shared_test
