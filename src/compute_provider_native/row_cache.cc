#include "native_plan.h"
#include "row_cache_codec.h"
#include "../compute_batch/row_cache_map.h"
#include "../compute_batch/row_cache_table.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4505) // The original header also defines unused whole-row entry points.
#endif
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "ds4_roundtrip.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include "hyle/compute/model_ops.h"
#include "hyle/mem/resource.h"
#include "numeric.h"

namespace hyle::compute::native::internal {
namespace {

Result<mem::HostMapping> read_mapping(mem::ConstBufferView storage) noexcept {
  if (!storage.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
    HYLE_RETURN_IF_ERROR(storage.invalidate());
  }
  return storage.map_host();
}

std::uint16_t bf16_bits(const std::byte *source) noexcept {
  std::uint16_t bits;
  std::memcpy(&bits, source, sizeof(bits));
  return bits;
}

// Only encode an already rounded positive format value. Rounding and ties are
// selected by the original DS4 helpers, while the source sign is kept in bytes.
std::uint8_t fp8_magnitude(float value) noexcept {
  if (value < 0.015625F)
    return static_cast<std::uint8_t>(value * 512.0F);
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint8_t>((((bits >> 23) - 120) << 3) |
                                   ((bits >> 20) & 7));
}

template <bool Four>
void encode_row(const std::byte *source, std::byte *output) noexcept {
  constexpr std::size_t count = Four ? 128 : 448;
  constexpr std::size_t group = Four ? 32 : 64;
  constexpr std::size_t scales = Four ? 64 : 576;
  constexpr float cap = Four ? 6.0F : 448.0F;
  constexpr float floor = Four ? 7.052966104933725e-38F : 1.0e-4F;
  for (std::size_t base = 0; base < count; base += group) {
    float maximum = 0;
    for (std::size_t i = 0; i < group; ++i) {
      const auto value = numeric::load<simd::bfloat16_t>(source + (base + i) * 2);
      maximum = (std::max)(maximum, std::abs(value));
    }
    if (maximum < floor) maximum = floor;
    const float scale = std::ldexp(1.0F,
        static_cast<int>(std::ceil(std::log2(maximum / cap))));
    output[scales + base / group] =
        std::byte((std::bit_cast<std::uint32_t>(scale) >> 23) & 255);
    for (std::size_t i = 0; i < group; ++i) {
      const auto offset = base + i;
      const auto raw = bf16_bits(source + offset * 2);
      const float value = std::abs(numeric::load<simd::bfloat16_t>(source + offset * 2) / scale);
      std::uint8_t code;
      if constexpr (Four) {
        const float rounded = ds4_quantization::dsv4_e2m1fn_dequant_cpu(value);
        code = 0;
        while (ds4_quantization::dsv4_e2m1fn_value_cpu(code) != rounded) ++code;
        code |= static_cast<std::uint8_t>((raw >> 12) & 8);
        if ((offset & 1) == 0) output[offset / 2] = std::byte(code);
        else output[offset / 2] |= std::byte(code << 4);
      } else {
        code = fp8_magnitude(ds4_quantization::dsv4_e4m3fn_dequant_cpu(value));
        code |= static_cast<std::uint8_t>((raw >> 8) & 128);
        output[offset] = std::byte(code);
      }
    }
  }
  if constexpr (!Four) {
    std::memcpy(output + 448, source + 448 * 2, 64 * 2);
    output[583] = std::byte{0};
  }
}

float scale_value(std::byte encoded) noexcept {
  const auto exponent = std::to_integer<unsigned>(encoded);
  return exponent == 255 ? std::numeric_limits<float>::quiet_NaN()
                         : std::ldexp(1.0F, static_cast<int>(exponent) - 127);
}

template <bool Four>
void decode_row(const std::byte *source, std::byte *output) noexcept {
  constexpr std::size_t count = Four ? 128 : 448;
  constexpr std::size_t group = Four ? 32 : 64;
  constexpr std::size_t scales = Four ? 64 : 576;
  for (std::size_t base = 0; base < count; base += group) {
    const float scale = scale_value(source[scales + base / group]);
    for (std::size_t i = 0; i < group; ++i) {
      const auto index = base + i;
      unsigned code = std::to_integer<unsigned>(source[Four ? index / 2 : index]);
      if constexpr (Four) code = (code >> ((index & 1) * 4)) & 15;
      const unsigned magnitude = code & (Four ? 7 : 127);
      float value;
      if constexpr (Four) value = ds4_quantization::dsv4_e2m1fn_value_cpu(magnitude);
      else value = magnitude == 127 ? std::numeric_limits<float>::quiet_NaN()
                                    : ds4_quantization::dsv4_e4m3fn_value_cpu(magnitude);
      if (code & (Four ? 8 : 128)) value = -value;
      numeric::store<simd::bfloat16_t>(output + index * 2, value * scale);
    }
  }
  if constexpr (!Four) std::memcpy(output + 448 * 2, source + 448, 64 * 2);
}

Status finish_output(mem::HostMapping &mapping, mem::BufferView output) noexcept {
  Status terminal;
  if (!output.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
    terminal = mapping.flush();
  preserve_first_error(terminal, mapping.close());
  return terminal;
}

template <bool Four>
detail::EnqueueResult submit_encode(const NativePlan &plan,
    std::span<const ConstTensorView> inputs, std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 1 || outputs.size() != 1)
    return completed(Status::Internal("row cache encode binding count differs"));
  const auto rows = plan.parameters[0], depth = plan.parameters[1], width = plan.parameters[2];
  if (rows == 0) return completed(Status::Ok());
  auto input = read_mapping(inputs[0].storage());
  if (!input) return completed(std::move(input).error());
  const auto *source = input->bytes().data();
  // Validate every input, including the bitwise tail, before opening output.
  for (std::size_t i = 0; i < rows * depth; ++i) {
    if ((bf16_bits(source + i * 2) & 0x7f80) == 0x7f80) {
      input->close().IgnoreError();
      return completed(Status::InvalidArgument("row cache encoder requires finite BF16"));
    }
  }
  auto output = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!output) { input->close().IgnoreError(); return completed(std::move(output).error()); }
  auto writable = output->writable_bytes();
  if (!writable) {
    output->close().IgnoreError(); input->close().IgnoreError();
    return completed(std::move(writable).error());
  }
  for (std::size_t row = 0; row < rows; ++row)
    encode_row<Four>(source + row * depth * 2, writable->data() + row * width);
  auto terminal = finish_output(*output, outputs[0].storage());
  preserve_first_error(terminal, input->close());
  return completed(std::move(terminal));
}

std::int32_t index_value(const std::byte *source, std::size_t index) noexcept {
  std::int32_t result;
  std::memcpy(&result, source + index * sizeof(result), sizeof(result));
  return result;
}

template <bool Four, bool Padded = false>
detail::EnqueueResult submit_gather(const NativePlan &plan,
    std::span<const ConstTensorView> inputs, std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 2 || outputs.size() != 1)
    return completed(Status::Internal("row cache gather binding count differs"));
  const auto count = plan.parameters[0], rows = plan.parameters[1], width = plan.parameters[2];
  if (count == 0) return completed(Status::Ok());
  auto ids = read_mapping(inputs[1].storage());
  if (!ids) return completed(std::move(ids).error());
  const auto *indices = ids->bytes().data();
  if (plan.index_bounds == IndexBoundsPolicy::kChecked) {
    for (std::size_t i = 0; i < count; ++i) {
      const auto index = index_value(indices, i);
      if ((Padded && index == -1)) continue;
      if (index < 0 || static_cast<std::uint64_t>(index) >= rows) {
        ids->close().IgnoreError();
        return completed(Status::OutOfRange("row cache ID is outside the bank"));
      }
    }
  }
  bool needs_bank = !Padded;
  if constexpr (Padded) for (std::size_t i=0;i<count && !needs_bank;++i)
    needs_bank=index_value(indices,i)>=0;
  Result<mem::HostMapping> source = mem::HostMapping{};
  if(needs_bank)source=read_mapping(inputs[0].storage());
  if (!source) { ids->close().IgnoreError(); return completed(std::move(source).error()); }
  auto output = outputs[0].storage().map_host(mem::HostAccess::kWrite);
  if (!output) {
    source->close().IgnoreError(); ids->close().IgnoreError();
    return completed(std::move(output).error());
  }
  auto writable = output->writable_bytes();
  if (!writable) {
    output->close().IgnoreError(); source->close().IgnoreError(); ids->close().IgnoreError();
    return completed(std::move(writable).error());
  }
  for (std::size_t i = 0; i < count; ++i) {
    const auto id=index_value(indices,i);auto*row=writable->data()+i*plan.row_output_stride*2;
    if constexpr(Padded){if(id==-1){std::memset(row,0,plan.row_output_stride*2);continue;}}
    decode_row<Four>(source->bytes().data()+static_cast<std::size_t>(id)*width,row);
  }
  auto terminal = finish_output(*output, outputs[0].storage());
  preserve_first_error(terminal, source->close());
  preserve_first_error(terminal, ids->close());
  return completed(std::move(terminal));
}
detail::EnqueueResult submit_page_map(const NativePlan&plan,
    std::span<const ConstTensorView> inputs,std::span<const TensorView> outputs)noexcept{
  const auto execute=[&]()noexcept->Status{
  const auto rows=plan.parameters[0],selected=plan.parameters[1],page_size=plan.parameters[2];
  if(!rows || !selected)return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto ids,read_mapping(inputs[0].storage()));
  HYLE_ASSIGN_OR_RETURN(auto metadata,read_mapping(inputs[1].storage()));
  HYLE_ASSIGN_OR_RETURN(auto output,outputs[0].storage().map_host(mem::HostAccess::kWrite));
  HYLE_ASSIGN_OR_RETURN(auto target,output.writable_bytes());
  const auto*bytes=metadata.bytes().data();
  const auto requests=std::uint32_t(index_value(bytes,4));
  const auto query_rows=std::uint32_t(index_value(bytes,5));
  const auto refs=std::uint32_t(index_value(bytes,7));
  const auto max_requests=std::uint32_t(plan.scalar_bits>>32),max_pages=std::uint32_t(plan.scalar_bits);
  const auto*q=bytes+plan.axis;const auto*kv=bytes+plan.row_first_stride;
  const auto*pages=bytes+plan.row_second_stride;const auto*tails=bytes+plan.row_output_stride;
  const bool valid=requests<=max_requests && query_rows==rows && refs<=(plan.row_output_stride-plan.row_second_stride)/4;
  for(std::size_t row=0;row<rows;++row){
    std::size_t lo=0,hi=requests;
    if(valid)while(lo<hi){const auto middle=lo+(hi-lo)/2;if(index_value(q,middle+1)<=std::int64_t(row))lo=middle+1;else hi=middle;}
    const bool request_ok=valid && lo<requests && index_value(q,lo)<=std::int64_t(row) && index_value(q,lo+1)>std::int64_t(row);
    const auto first=request_ok?index_value(kv,lo):0,end=request_ok?index_value(kv,lo+1):0,tail=request_ok?index_value(tails,lo):0;
    const bool span_ok=request_ok && first>=0 && end>=first && std::uint32_t(end)<=refs &&
        ((end==first && tail==0) || (end>first && tail>0 && std::size_t(tail)<=page_size));
    const auto length=span_ok && end>first?std::uint64_t(end-first-1)*page_size+std::uint32_t(tail):0;
    for(std::size_t k=0;k<selected;++k){
      const auto id=index_value(ids.bytes().data(),row*selected+k);std::int32_t physical=-1;
      if(span_ok && id>=0 && std::uint64_t(id)<length){
        const auto page=index_value(pages,std::size_t(first)+std::size_t(id)/page_size);
        if(page>=0 && std::uint32_t(page)<max_pages){const auto index=std::uint64_t(page)*page_size+std::size_t(id)%page_size;
          if(index<=INT32_MAX)physical=std::int32_t(index);}
      }
      std::memcpy(target.data()+(row*selected+k)*4,&physical,4);
    }
  }
  auto terminal=finish_output(output,outputs[0].storage());preserve_first_error(terminal,metadata.close());preserve_first_error(terminal,ids.close());return terminal;
  };return completed(execute());
}
detail::EnqueueResult submit_padded_page_map(const NativePlan&plan,
    std::span<const ConstTensorView> in,std::span<const TensorView> out)noexcept{
  const auto execute=[&]()noexcept->Status{
    const auto rows=plan.parameters[0],selected=plan.parameters[1];
    if(!rows)return Status::Ok();
    HYLE_ASSIGN_OR_RETURN(auto ids,read_mapping(in[0].storage()));
    HYLE_ASSIGN_OR_RETURN(auto ranges,read_mapping(in[1].storage()));
    HYLE_ASSIGN_OR_RETURN(auto metadata,read_mapping(in[2].storage()));
    detail::RowCacheMap geometry{plan.axis,plan.row_first_stride,plan.row_second_stride,
      plan.row_output_stride,std::uint64_t(in[2].description().dimensions()[0]),std::uint32_t(plan.parameters[2]),std::uint32_t(plan.scalar_bits)};
    const auto table=detail::RowCacheTable::create(geometry,metadata.bytes().data(),rows);
    bool valid=bool(table);
    if(valid)for(std::size_t q=0;q<rows;++q){const auto request=table->request(q);
      const auto b=index_value(ranges.bytes().data(),q*2),e=index_value(ranges.bytes().data(),q*2+1);
      if(request>=table->requests || b<0 || e<b || std::uint64_t(e)>table->length(request) ||
         std::uint64_t(e)>plan.alias_input0_offset){valid=false;break;}}
    HYLE_ASSIGN_OR_RETURN(auto output,out[0].storage().map_host(mem::HostAccess::kWrite));
    HYLE_ASSIGN_OR_RETURN(auto mask,out[1].storage().map_host(mem::HostAccess::kWrite));
    HYLE_ASSIGN_OR_RETURN(auto target,output.writable_bytes());HYLE_ASSIGN_OR_RETURN(auto masks,mask.writable_bytes());
    for(std::size_t q=0;q<rows;++q)for(std::size_t k=0;k<selected;++k){
      const auto id=index_value(ids.bytes().data(),q*selected+k);std::int32_t physical=-2;float bias=-std::numeric_limits<float>::infinity();
      if(valid && id>=-1 && (id==-1 || std::uint64_t(id)<plan.alias_input0_offset)){
        physical=-1;
        if(id>=index_value(ranges.bytes().data(),q*2) && id<index_value(ranges.bytes().data(),q*2+1)){
          physical=table->physical(table->request(q),std::uint32_t(id));bias=0;}
      }
      std::memcpy(target.data()+(q*selected+k)*4,&physical,4);std::memcpy(masks.data()+(q*selected+k)*4,&bias,4);
    }
    auto status=finish_output(output,out[0].storage());preserve_first_error(status,finish_output(mask,out[1].storage()));
    preserve_first_error(status,metadata.close());preserve_first_error(status,ranges.close());preserve_first_error(status,ids.close());return status;
  };return completed(execute());
}
} // namespace

void decode_row_cache_float4(const std::byte *source,
                            std::byte *output) noexcept {
  decode_row<true>(source, output);
}

bool inspect_row_cache_access(const NativePlan &plan,
                             detail::OperationPreparedAccess &access) noexcept {
  access = {};
  if (plan.submit_route == &submit_page_map ||
      plan.submit_route == &submit_padded_page_map) return true;
  const bool padded = plan.submit_route == &submit_gather<true, true> ||
                      plan.submit_route == &submit_gather<false, true>;
  const bool checked = plan.index_bounds == IndexBoundsPolicy::kChecked &&
      (plan.submit_route == &submit_gather<true> ||
       plan.submit_route == &submit_gather<false>);
  if (!padded && !checked) return false;
  access.projected_input = 0;
  access.allows_empty_footprint = padded || plan.parameters[0] == 0;
  return true;
}

Status prepare_row_cache(Operation operation, std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs, NativePlan &plan) noexcept {
  if(const auto map=operation.custom_as<detail::RowCachePaddedMap>()){
    std::array<TensorDesc,2> inferred;HYLE_RETURN_IF_ERROR(map->infer_output_descriptions(inputs,inferred));
    if(outputs.size()!=2 || outputs[0]!=inferred[0] || outputs[1]!=inferred[1])return Status::InvalidArgument("padded map outputs differ");
    const auto&m=map->mapping;plan.parameters={std::size_t(inputs[0].dimensions()[0]),std::size_t(inputs[0].dimensions()[1]),m.page_size};
    plan.axis=std::size_t(m.query_offset);plan.row_first_stride=std::size_t(m.kv_offset);
    plan.row_second_stride=std::size_t(m.page_offset);plan.row_output_stride=std::size_t(m.tail_offset);
    plan.scalar_bits=m.max_pages;plan.alias_input0_offset=map->score_columns;
    plan.submit_route=&submit_padded_page_map;return Status::Ok();
  }
  if (outputs.size() != 1) return Status::InvalidArgument("row cache output count differs");
  std::array<TensorDesc, 1> inferred;
  if(const auto map=operation.custom_as<detail::RowCacheMap>()){
    HYLE_RETURN_IF_ERROR(map->infer_output_descriptions(inputs,inferred));
    if(outputs[0]!=inferred[0])return Status::InvalidArgument("row cache map output differs");
    plan.parameters={std::size_t(inputs[0].dimensions()[0]),std::size_t(inputs[0].dimensions()[1]),map->page_size};
    plan.axis=std::size_t(map->query_offset);plan.row_first_stride=std::size_t(map->kv_offset);
    plan.row_second_stride=std::size_t(map->page_offset);plan.row_output_stride=std::size_t(map->tail_offset);
    plan.scalar_bits=(std::uint64_t(map->max_requests())<<32)|map->max_pages;
    plan.submit_route=&submit_page_map;return Status::Ok();
  }
  if (const auto encode = operation.custom_as<model_ops::RowCacheEncode>()) {
    HYLE_RETURN_IF_ERROR(encode->infer_output_descriptions(inputs, inferred));
    if (outputs[0] != inferred[0]) return Status::InvalidArgument("row cache encode output differs");
    plan.parameters = {static_cast<std::size_t>(inputs[0].dimensions()[0]),
                       encode->encoding.depth(), encode->encoding.row_bytes()};
    plan.submit_route = encode->encoding.format() == RowCacheFormat::kFloat4Block32
        ? &submit_encode<true> : &submit_encode<false>;
    return Status::Ok();
  }
  if(const auto gather=operation.custom_as<model_ops::RowCacheGatherDecodePadded>()){
    HYLE_RETURN_IF_ERROR(gather->infer_output_descriptions(inputs,inferred));
    if(outputs[0]!=inferred[0])return Status::InvalidArgument("padded row gather output differs");
    HYLE_ASSIGN_OR_RETURN(auto count,inputs[1].element_count());
    plan.parameters={count,std::size_t(inputs[0].dimensions()[0]),gather->encoding.row_bytes()};
    plan.row_output_stride=gather->encoding.depth();plan.index_bounds=IndexBoundsPolicy::kChecked;
    plan.submit_route=gather->encoding.format()==RowCacheFormat::kFloat4Block32?&submit_gather<true,true>:&submit_gather<false,true>;
    return Status::Ok();
  }
  if (const auto gather = operation.custom_as<model_ops::RowCacheGatherDecode>()) {
    HYLE_RETURN_IF_ERROR(gather->infer_output_descriptions(inputs, inferred));
    if (outputs[0] != inferred[0]) return Status::InvalidArgument("row cache gather output differs");
    HYLE_ASSIGN_OR_RETURN(auto count, inputs[1].element_count());
    plan.parameters = {count, static_cast<std::size_t>(inputs[0].dimensions()[0]),
                       gather->encoding.row_bytes()};
    plan.row_output_stride = gather->encoding.depth();
    plan.index_bounds = gather->bounds;
    plan.submit_route = gather->encoding.format() == RowCacheFormat::kFloat4Block32
        ? &submit_gather<true> : &submit_gather<false>;
    return Status::Ok();
  }
  return Status::Unimplemented("operation is not a row cache codec");
}
} // namespace hyle::compute::native::internal
