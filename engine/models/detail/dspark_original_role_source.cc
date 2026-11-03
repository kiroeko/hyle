#include "dspark_original_role_source.h"

namespace dspark_original_role_source {
void check(hyle::Status status,std::string_view where) {
  if(!status.ok())throw std::runtime_error(std::string(where)+": "+std::string(status.message()));
}
}

namespace dspark_original_role_source {
std::uint32_t u32(std::span<const std::byte> raw,std::size_t offset) {
  if(offset>raw.size()||raw.size()-offset<4)throw std::runtime_error("original LE32 extent");
  std::uint32_t result=0;for(std::size_t i=0;i<4;++i)result|=std::to_integer<std::uint32_t>(raw[offset+i])<<(8*i);
  return result;
}
}

namespace dspark_original_role_source {
std::uint64_t u64(std::span<const std::byte> raw,std::size_t offset) {
  if(offset>raw.size()||raw.size()-offset<8)throw std::runtime_error("original LE64 extent");
  std::uint64_t result=0;for(std::size_t i=0;i<8;++i)result|=std::to_integer<std::uint64_t>(raw[offset+i])<<(8*i);
  return result;
}
}

namespace dspark_original_role_source {
std::vector<std::uint16_t> decode_bfloat(std::span<const std::byte> raw) {
  if(raw.size()%2)throw std::runtime_error("original BF16 byte extent");
  std::vector<std::uint16_t> result(raw.size()/2);
  for(std::size_t i=0;i<result.size();++i) {
    result[i]=std::uint16_t(std::to_integer<unsigned>(raw[2*i])|(std::to_integer<unsigned>(raw[2*i+1])<<8U));
    if(!std::isfinite(std::bit_cast<float>(std::uint32_t{result[i]}<<16U)))
      throw std::runtime_error("original BF16 coefficient is nonfinite");
  }
  return result;
}
}

namespace dspark_original_role_source {
std::vector<float> widen_bfloat(std::span<const std::byte> raw) {
  const auto bits=decode_bfloat(raw);std::vector<float> result;result.reserve(bits.size());
  for(const auto value:bits)result.push_back(std::bit_cast<float>(std::uint32_t{value}<<16U));
  return result;
}
}

namespace dspark_original_role_source {
std::vector<float> decode_f32(std::span<const std::byte> raw) {
  if(raw.size()%4)throw std::runtime_error("original F32 byte extent");
  std::vector<float> result(raw.size()/4);
  for(std::size_t i=0;i<result.size();++i) {
    result[i]=std::bit_cast<float>(u32(raw,4*i));
    if(!std::isfinite(result[i]))throw std::runtime_error("original F32 coefficient is nonfinite");
  }
  return result;
}
}

namespace dspark_original_role_source {
Float8Matrix prepare_float8(const c::TensorDesc &codes_description,m::ConstBufferView codes_view,
    std::span<const std::byte> codes,const c::TensorDesc &scales_description,m::ConstBufferView scales_view,
    std::span<const std::byte> source_scales) {
  if(codes_description.rank()!=2||codes_description.data_type()!=c::DataType::kFloat8E4M3FN)
    throw std::runtime_error("original FP8 preparation requires plain E4[N,K]");
  for(const auto code:codes)if((std::to_integer<unsigned>(code)&127U)==127U)
    throw std::runtime_error("original E4 coefficient is nonfinite");
  Float8Matrix result;
  result.scales.reserve(source_scales.size());
  for(const auto encoded:source_scales) {
    const auto code=std::to_integer<unsigned>(encoded);
    if(code==255)throw std::runtime_error("original UE8M0 scale is nonfinite");
    result.scales.push_back(std::ldexp(1.F,int(code)-127));
  }
  const auto dimensions=codes_description.dimensions();
  result.description=take(c::float8_weight_description(dimensions[1],dimensions[0],c::Float8ScaleMode::kFloat32Block128),"original canonical FP8 description");
  result.buffer=take(m::Buffer::create(m::system_resource(),take(result.description.required_bytes(),"original canonical FP8 extent"),64),"original canonical FP8 allocation");
  const auto code_tensor=take(c::ConstTensorView::create(codes_description,codes_view),"original E4 source view");
  const auto scale_tensor=take(c::ConstTensorView::create(scales_description,scales_view),"original UE8 source view");
  auto destination=take(c::TensorView::create(result.description,result.buffer.view()),"original canonical FP8 destination");
  check(c::import_float8_block128_weight(code_tensor,scale_tensor,destination),"original FP8 canonical import");
  result.mapping=take(static_cast<const m::Buffer&>(result.buffer).view().map_host(),"retain original canonical read mapping");
  const auto packed=result.payload(),scale_bytes=std::as_bytes(std::span(result.scales));
  if(packed.size()!=codes.size()+scale_bytes.size()||!std::equal(codes.begin(),codes.end(),packed.begin())||
      !std::equal(scale_bytes.begin(),scale_bytes.end(),packed.begin()+std::ptrdiff_t(codes.size())))
    throw std::runtime_error("original FP8 import changed codes or exact UE8 values");
  return result;
}
}

namespace dspark_original_role_source {
ReadonlyRole bind(std::string_view name,const c::TensorDesc &description,
    std::span<const std::byte> payload,const c::StorageOwner &owner) {
  if(!owner)throw std::runtime_error("original immutable bind needs its Graph owner");
  const auto bytes=take(description.required_bytes(),"original immutable role extent");
  if(bytes!=payload.size())throw std::runtime_error("original immutable payload extent differs");
  auto bank=take(m::Buffer::create(owner.ref(),bytes,256),"original immutable bank allocation");
  check(bank.view().write(payload),"original once-only immutable upload");
  return {name,description,owner,take(owner.adopt_buffer(std::move(bank)),"original readonly bank lease")};
}
}
