#include "apple_row_cache.h"

#include <array>
#include <cstdint>

#include <mlx/ops.h>
#include <mlx/fast.h>

#include "apple_checks.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
mx::array u(std::uint32_t value) { return mx::array(value); }

// BF16 max is ordered by its unsigned magnitude bits. Comparing the short
// significand against 448/2 or 6*32 computes ceil(log2(max/code_max)) exactly,
// including the original scale floors, without subnormal floating arithmetic.
mx::array scale_exponents(const mx::array &magnitude, bool four, mx::Stream s) {
  auto maximum = mx::max(magnitude, -1, true, s);
  auto exponent = mx::astype(mx::right_shift(maximum, u(7), s), mx::int32, s);
  auto significand = mx::bitwise_or(mx::bitwise_and(maximum, u(127), s), u(128), s);
  auto increment = mx::astype(mx::greater(significand, u(four ? 192 : 224), s), mx::int32, s);
  auto result = mx::maximum(mx::add(mx::subtract(exponent, mx::array(four ? 2 : 8), s),
      increment, s), mx::array(four ? 1 : 105), s);
  return mx::astype(result, mx::uint32, s);
}

mx::array quantized_codes(const mx::array &bits, const mx::array &magnitude,
                         const mx::array &exponent, bool four, mx::Stream s) {
  auto scale = mx::view(mx::left_shift(exponent, u(23), s), mx::float32, s);
  auto value = mx::view(mx::left_shift(magnitude, u(16), s), mx::float32, s);
  auto normalized = mx::divide(value, scale, s);
  mx::array codes = mx::zeros(bits.shape(), mx::uint32, s);
  if (four) {
    // Metal may flush BF16 subnormal values after conversion to F32. In the
    // only affected scale domain, exact integer significands stay small.
    auto source_exp = mx::right_shift(magnitude, u(7), s);
    auto significand = mx::bitwise_or(mx::bitwise_and(magnitude, u(127), s),
        mx::where(mx::equal(source_exp, u(0), s), u(0), u(128), s), s);
    auto shift = mx::subtract(mx::minimum(mx::maximum(source_exp, u(1), s), u(4), s), u(1), s);
    auto low = mx::multiply(mx::astype(mx::left_shift(significand, shift, s), mx::float32, s),
        mx::where(mx::equal(exponent, u(1), s), mx::array(1.F / 128),
            mx::array(1.F / 256), s), s);
    normalized = mx::minimum(mx::where(mx::less_equal(exponent, u(2), s),
        low, normalized, s), mx::array(6.F), s);
    constexpr std::array<float, 8> values{0, .5F, 1, 1.5F, 2, 3, 4, 6};
    auto distance = normalized;
    for (std::uint32_t code = 1; code < values.size(); ++code) {
      auto difference = mx::abs(mx::subtract(normalized, mx::array(values[code]), s), s);
      auto choose = mx::less(difference, distance, s);
      if ((code & 1) == 0)
        choose = mx::logical_or(choose, mx::logical_and(mx::equal(difference, distance, s),
            mx::not_equal(mx::bitwise_and(codes, u(1), s), u(0), s), s), s);
      codes = mx::where(choose, u(code), codes, s);
      distance = mx::where(choose, difference, distance, s);
    }
  } else {
    // The locked native converter uses saturated nearest-even E4M3FN.
    codes = mx::astype(mx::to_fp8(mx::minimum(normalized, mx::array(448.F), s), s), mx::uint32, s);
  }
  auto sign = mx::right_shift(mx::bitwise_and(bits, u(0x8000), s), u(four ? 12 : 8), s);
  return mx::astype(mx::bitwise_or(codes, sign, s), mx::uint8, s);
}

// All nonzero code magnitudes are normal F32 and exactly BF16-representable.
// Apply the UE8 exponent and RNE in integers so UE8=0 and small BF16 outputs
// never pass through Metal's floating subnormal/FTZ path.
mx::array decoded_bits(const mx::array &codes, const mx::array &exponents,
                       bool four, mx::Stream s) {
  const auto raw = mx::astype(codes, mx::uint32, s);
  auto magnitude = mx::bitwise_and(raw, u(four ? 7 : 127), s);
  mx::array value = mx::zeros(codes.shape(), mx::float32, s);
  if (four) {
    constexpr std::array<float, 8> values{0, .5F, 1, 1.5F, 2, 3, 4, 6};
    value = mx::take(mx::array(values.data(), {8}, mx::float32), magnitude, 0, s);
  } else {
    value = mx::from_fp8(mx::astype(magnitude, mx::uint8, s), mx::float32, s);
  }
  auto bits = mx::view(value, mx::uint32, s);
  auto exponent = mx::subtract(mx::add(
      mx::astype(mx::right_shift(bits, u(23), s), mx::int32, s),
      mx::astype(exponents, mx::int32, s), s), mx::array(127), s);
  auto normal_exp = mx::astype(mx::clip(exponent, mx::array(1), mx::array(254), s), mx::uint32, s);
  auto normal = mx::bitwise_or(mx::left_shift(normal_exp, u(7), s),
      mx::bitwise_and(mx::right_shift(bits, u(16), s), u(127), s), s);
  auto significand = mx::bitwise_or(mx::bitwise_and(bits, u(0x7fffff), s), u(0x800000), s);
  // Both where branches may be evaluated: every shift must remain defined.
  auto shift = mx::astype(mx::clip(mx::subtract(mx::array(17), exponent, s),
      mx::array(1), mx::array(31), s), mx::uint32, s);
  auto base = mx::right_shift(significand, shift, s);
  auto halfway = mx::left_shift(u(1), mx::subtract(shift, u(1), s), s);
  auto remainder = mx::bitwise_and(significand,
      mx::subtract(mx::left_shift(u(1), shift, s), u(1), s), s);
  auto increment = mx::logical_or(mx::greater(remainder, halfway, s),
      mx::logical_and(mx::equal(remainder, halfway, s),
          mx::not_equal(mx::bitwise_and(base, u(1), s), u(0), s), s), s);
  auto subnormal = mx::add(base, mx::astype(increment, mx::uint32, s), s);
  auto result = mx::where(mx::less_equal(exponent, mx::array(0), s), subnormal, normal, s);
  result = mx::where(mx::greater_equal(exponent, mx::array(255), s), u(0x7f80), result, s);
  result = mx::where(mx::equal(magnitude, u(0), s), u(0), result, s);
  auto sign = mx::left_shift(mx::bitwise_and(raw, u(four ? 8 : 128), s), u(four ? 12 : 8), s);
  result = mx::bitwise_or(result, sign, s);
  auto invalid = mx::equal(exponents, u(255), s);
  if (!four)
    invalid = mx::logical_or(invalid, mx::equal(magnitude, u(127), s), s);
  return mx::astype(mx::where(invalid, u(0x7fc0), result, s), mx::uint16, s);
}
mx::array decode_selected(RowCacheEncoding encoding,const mx::array&selected,const mx::Shape&output_shape,mx::Stream s){
  const bool four=encoding.format()==RowCacheFormat::kFloat4Block32;const int count=selected.shape(0);
  const int channels = four ? 128 : 448, group = four ? 32 : 64;
  auto payload = mx::slice(selected, {0, 0}, {count, four ? 64 : 448}, s);
  mx::array codes = mx::reshape(payload, {count, channels / group, four ? 16 : 64}, s);
  if (four) {
    auto low = mx::bitwise_and(codes, mx::array(std::uint8_t{7 | 8}), s);
    auto high = mx::right_shift(codes, mx::array(std::uint8_t{4}), s);
    codes = mx::reshape(mx::stack({low, high}, -1, s), {count, 4, 32}, s);
  }
  auto scales = mx::astype(mx::reshape(mx::slice(selected, {0, four ? 64 : 576},
      {count, four ? 68 : 583}, s), {count, channels / group, 1}, s), mx::uint32, s);
  auto bits = mx::reshape(decoded_bits(codes, scales, four, s), {count, channels}, s);
  if (!four) {
    // Byte loads retain the public U8 bank's valid odd byte origins.
    auto tail = mx::reshape(mx::slice(selected, {0, 448}, {count, 576}, s), {count, 64, 2}, s);
    auto low = mx::astype(mx::reshape(mx::slice(tail, {0, 0, 0}, {count, 64, 1}, s),
        {count, 64}, s), mx::uint16, s);
    auto high = mx::astype(mx::reshape(mx::slice(tail, {0, 0, 1}, {count, 64, 2}, s),
        {count, 64}, s), mx::uint16, s);
    bits = mx::concatenate({bits, mx::bitwise_or(low,
        mx::left_shift(high, mx::array(std::uint16_t{8}), s), s)}, -1, s);
  }
  return mx::view(mx::reshape(bits, output_shape, s), mx::bfloat16, s);
}
mx::array select_checked_rows(const mx::array &bank, const mx::array &ids,
                              const mx::array &valid, mx::Stream s) {
  const int count = int(ids.size());
  static const auto select=mx::fast::metal_kernel("hyle_padded_encoded_rows",
      {"bank","ids","all_valid"},{"selected"},R"METAL(
    const ulong row=thread_position_in_grid.y, col=thread_position_in_grid.x;
    if(row>=ulong(ids_shape[0]) || col>=ulong(bank_shape[1]))return;
    const int id=ids[row*ids_strides[0]];
    if(!all_valid || id==-1){selected[row*bank_shape[1]+col]=0;return;}
    selected[row*bank_shape[1]+col]=bank[ulong(id)*bank_strides[0]+col*bank_strides[1]];
  )METAL","",false,false,mx::CompileOptions{});
  return select({bank,ids,valid},{{count,bank.shape(1)}},{mx::uint8},
      {bank.shape(1),count,1},{128,1,1},{},std::nullopt,false,s).front();
}
} // namespace

Result<mx::array> encode_row_cache(RowCacheEncoding encoding,
                                  const mx::array &input, mx::Stream s) {
  const bool four = encoding.format() == RowCacheFormat::kFloat4Block32;
  const int rows = input.shape(0), width = static_cast<int>(encoding.row_bytes());
  if (rows == 0) return mx::zeros({0, width}, mx::uint8, s);
  const int channels = four ? 128 : 448, group = four ? 32 : 64;
  auto raw = mx::astype(mx::view(input, mx::uint16, s), mx::uint32, s);
  auto finite = mx::not_equal(mx::bitwise_and(raw, u(0x7f80), s), u(0x7f80), s);
  auto safe = mx::where(finite, raw, u(0), s);
  auto bits = mx::reshape(mx::slice(safe, {0, 0}, {rows, channels}, s),
      {rows, channels / group, group}, s);
  auto magnitude = mx::bitwise_and(bits, u(0x7fff), s);
  auto exponents = scale_exponents(magnitude, four, s);
  auto codes = quantized_codes(bits, magnitude, exponents, four, s);
  mx::array payload = mx::reshape(codes, {rows, channels}, s);
  if (four) {
    auto pairs = mx::reshape(payload, {rows, 64, 2}, s);
    auto low = mx::reshape(mx::slice(pairs, {0, 0, 0}, {rows, 64, 1}, s), {rows, 64}, s);
    auto high = mx::reshape(mx::slice(pairs, {0, 0, 1}, {rows, 64, 2}, s), {rows, 64}, s);
    payload = mx::bitwise_or(low, mx::left_shift(high, mx::array(std::uint8_t{4}), s), s);
  }
  auto scales = mx::astype(mx::reshape(exponents, {rows, channels / group}, s), mx::uint8, s);
  mx::array packed = four ? mx::concatenate({payload, scales}, -1, s)
      : mx::concatenate({payload,
          mx::view(mx::slice(input, {0, 448}, {rows, 512}, s), mx::uint8, s),
          scales, mx::zeros({rows, 1}, mx::uint8, s)}, -1, s);
  auto status = mx::where(mx::all(finite, false, s), mx::array(0),
      mx::array(kInvalidNumericInput), s);
  return with_device_status(std::move(packed), std::move(status), s);
}

Result<mx::array> gather_decode_row_cache(
    const model_ops::RowCacheGatherDecode &operation, const mx::array &bank,
    const mx::array &indices, mx::Stream s) {
  const auto encoding = operation.encoding;
  auto output_shape = indices.shape();
  output_shape.push_back(static_cast<int>(encoding.depth()));
  if (indices.size() == 0) return mx::zeros(output_shape, mx::bfloat16, s);
  if (operation.bounds == IndexBoundsPolicy::kChecked && bank.shape(0) == 0)
    return with_device_status(mx::zeros(output_shape, mx::bfloat16, s),
        mx::array(kIndexOutOfRange), s);
  const int count = static_cast<int>(indices.size());
  auto ids = mx::reshape(indices, {count}, s);
  auto invalid = mx::logical_or(mx::less(ids, mx::array(0), s),
      mx::greater_equal(ids, mx::array(bank.shape(0)), s), s);
  // Reuse the original padding-aware select kernel. A failed checked set
  // reads no bank row at all; it never falls back to an unauthorized row0.
  auto selected = operation.bounds == IndexBoundsPolicy::kChecked
      ? select_checked_rows(bank, ids, mx::logical_not(mx::any(invalid, s), s), s)
      : mx::take(bank, ids, 0, s);
  auto result=decode_selected(encoding,selected,output_shape,s);
  if (operation.bounds == IndexBoundsPolicy::kUnchecked) return result;
  auto status = mx::where(mx::any(invalid, s), mx::array(kIndexOutOfRange), mx::array(0), s);
  return with_device_status(std::move(result), std::move(status), s);
}
Result<mx::array> gather_decode_row_cache_padded(RowCacheEncoding encoding,const mx::array&bank,const mx::array&indices,mx::Stream s){
  auto shape=indices.shape();shape.push_back(int(encoding.depth()));
  if(!indices.size())return mx::zeros(shape,mx::bfloat16,s);
  const int count=int(indices.size());auto ids=mx::reshape(indices,{count},s);
  auto invalid=mx::logical_or(mx::less(ids,mx::array(-1),s),mx::greater_equal(ids,mx::array(bank.shape(0)),s),s);
  auto valid=mx::logical_not(mx::any(invalid,s),s);
  auto status=mx::where(valid,mx::array(0),mx::array(kIndexOutOfRange),s);
  if(!bank.shape(0))return with_device_status(mx::zeros(shape,mx::bfloat16,s),std::move(status),s);
  // A real control dependency on the complete checked ID set precedes every
  // bank read. Padding never falls back to an existing/unauthorized row0.
  auto selected=select_checked_rows(bank,ids,valid,s);
  auto result=decode_selected(encoding,selected,shape,s);
  return with_device_status(std::move(result),std::move(status),s);
}
Result<std::vector<mx::array>> map_row_cache_pages_padded(const compute::detail::RowCachePaddedMap&operation,
    const mx::array&indices,const mx::array&ranges,const mx::array&metadata,mx::Stream s){
  const auto&p=operation.mapping;
  if(p.metadata_bytes>INT32_MAX || !p.max_refs())return std::unexpected(Status::Unimplemented("Apple padded row metadata exceeds native I32 carrier"));
  if(!indices.size())return std::vector{mx::zeros(indices.shape(),mx::int32,s),mx::zeros(indices.shape(),mx::float32,s)};
  // Fixed original page metadata is read on device; no ID host observation.
  static const auto kernel=mx::fast::metal_kernel("hyle_padded_page_map",{"ids","ranges","metadata"},{"physical","mask"},R"METAL(
    const ulong row=thread_position_in_grid.y, col=thread_position_in_grid.x;
    if(row>=ulong(ids_shape[0]) || col>=ulong(ids_shape[1]))return;
    const uint requests=hyle_word(metadata,metadata_strides[0],16), query_rows=hyle_word(metadata,metadata_strides[0],20), refs=hyle_word(metadata,metadata_strides[0],28);
    bool valid=requests<=MAX_REQUESTS && query_rows==uint(ids_shape[0]) && refs<=MAX_REFS;
    uint lo=0,hi=requests;
    if(valid)while(lo<hi){const uint mid=lo+(hi-lo)/2;const int next=int(hyle_word(metadata,metadata_strides[0],(Q_OFFSET+mid+1)*4));if(next<=int(row))lo=mid+1;else hi=mid;}
    bool request_ok=valid && lo<requests;
    if(request_ok)request_ok=int(hyle_word(metadata,metadata_strides[0],(Q_OFFSET+lo)*4))<=int(row) && int(hyle_word(metadata,metadata_strides[0],(Q_OFFSET+lo+1)*4))>int(row);
    const int first=request_ok?int(hyle_word(metadata,metadata_strides[0],(KV_OFFSET+lo)*4)):0;
    const int end=request_ok?int(hyle_word(metadata,metadata_strides[0],(KV_OFFSET+lo+1)*4)):0;
    const int tail=request_ok?int(hyle_word(metadata,metadata_strides[0],(TAIL_OFFSET+lo)*4)):0;
    bool span=request_ok && first>=0 && end>=first && uint(end)<=refs && ((end==first && tail==0) || (end>first && tail>0 && uint(tail)<=PAGE_SIZE));
    const ulong length=span && end>first?ulong(end-first-1)*PAGE_SIZE+uint(tail):0;
    const int begin=ranges[row*ranges_strides[0]], stop=ranges[row*ranges_strides[0]+ranges_strides[1]];
    span=span && begin>=0 && stop>=begin && ulong(stop)<=length && uint(stop)<=SCORE_COLUMNS;
    const int id=ids[row*ids_strides[0]+col*ids_strides[1]];int out=-2;float bias=-INFINITY;
    if(span && id>=-1 && (id==-1 || uint(id)<SCORE_COLUMNS)){
      out=-1;
      if(id>=begin && id<stop){const int page=int(hyle_word(metadata,metadata_strides[0],(PAGE_OFFSET+uint(first)+uint(id)/PAGE_SIZE)*4));
        if(page>=0 && uint(page)<MAX_PAGES){out=int(uint(page)*PAGE_SIZE+uint(id)%PAGE_SIZE);bias=0;}else out=-2;}
    }
    physical[row*ids_shape[1]+col]=out;mask[row*ids_shape[1]+col]=bias;
  )METAL",R"METAL(
    inline uint hyle_word(const device uchar*p,ulong stride,uint at){return uint(p[ulong(at)*stride])|(uint(p[ulong(at+1)*stride])<<8)|(uint(p[ulong(at+2)*stride])<<16)|(uint(p[ulong(at+3)*stride])<<24);}
  )METAL",false,false,mx::CompileOptions{});
  return kernel({indices,ranges,metadata},{indices.shape(),indices.shape()},{mx::int32,mx::float32},
      {indices.shape(1),indices.shape(0),1},{128,1,1},
      {{"MAX_REQUESTS",int(p.max_requests())},{"MAX_REFS",int(p.max_refs())},{"MAX_PAGES",int(p.max_pages)},{"PAGE_SIZE",int(p.page_size)},
       {"Q_OFFSET",int(p.query_offset/4)},{"KV_OFFSET",int(p.kv_offset/4)},{"PAGE_OFFSET",int(p.page_offset/4)},
       {"TAIL_OFFSET",int(p.tail_offset/4)},{"SCORE_COLUMNS",int(operation.score_columns)}},std::nullopt,false,s);
}
Result<mx::array> map_row_cache_pages(const compute::detail::RowCacheMap&p,
    const mx::array&indices,const mx::array&metadata,mx::Stream s){
  if(p.metadata_bytes>INT32_MAX || !p.max_refs())return std::unexpected(Status::Unimplemented("Apple row page metadata exceeds native I32 carrier"));
  if(indices.size()==0)return mx::zeros(indices.shape(),mx::int32,s);
  const int rows=indices.shape(0),max_requests=int(p.max_requests());
  auto words=mx::view(metadata,mx::int32,s);
  const auto part=[&](std::uint64_t first,std::uint64_t count){return mx::slice(words,{int(first/4)},{int(first/4+count)},s);};
  auto q=part(p.query_offset,p.max_requests()+1),kv=part(p.kv_offset,p.max_requests()+1);
  auto pages=part(p.page_offset,p.max_refs()),tails=part(p.tail_offset,p.max_requests());
  auto requests=mx::take(words,mx::array(4),0,s),query_rows=mx::take(words,mx::array(5),0,s),refs=mx::take(words,mx::array(7),0,s);
  auto valid=mx::logical_and(mx::logical_and(mx::greater_equal(requests,mx::array(0),s),mx::less_equal(requests,mx::array(max_requests),s),s),
      mx::logical_and(mx::equal(query_rows,mx::array(rows),s),mx::logical_and(mx::greater_equal(refs,mx::array(0),s),mx::less_equal(refs,mx::array(int(p.max_refs())),s),s),s),s);
  auto row=mx::arange(rows,mx::int32,s),lo=mx::zeros({rows},mx::int32,s),hi=mx::broadcast_to(mx::clip(requests,mx::array(0),mx::array(max_requests),s),{rows},s);
  for(std::uint64_t span=1;span<=std::uint64_t(p.max_requests());span*=2){
    auto active=mx::less(lo,hi,s),middle=mx::add(lo,mx::floor_divide(mx::subtract(hi,lo,s),mx::array(2),s),s);
    auto upper=mx::take(q,mx::minimum(mx::add(middle,mx::array(1),s),mx::array(max_requests),s),0,s);
    auto right=mx::less_equal(upper,row,s);
    lo=mx::where(mx::logical_and(active,right,s),mx::add(middle,mx::array(1),s),lo,s);
    hi=mx::where(mx::logical_and(active,mx::logical_not(right,s),s),middle,hi,s);
  }
  auto r=mx::minimum(lo,mx::array(max_requests-1),s);
  auto first=mx::take(kv,r,0,s),end=mx::take(kv,mx::add(r,mx::array(1),s),0,s),tail=mx::take(tails,r,0,s);
  auto request_ok=mx::logical_and(valid,mx::logical_and(mx::less(lo,requests,s),mx::logical_and(mx::less_equal(mx::take(q,r,0,s),row,s),mx::greater(mx::take(q,mx::add(r,mx::array(1),s),0,s),row,s),s),s),s);
  auto span_ok=mx::logical_and(request_ok,mx::logical_and(mx::greater_equal(first,mx::array(0),s),mx::logical_and(mx::greater_equal(end,first,s),mx::less_equal(end,refs,s),s),s),s);
  span_ok=mx::logical_and(span_ok,mx::logical_or(mx::logical_and(mx::equal(end,first,s),mx::equal(tail,mx::array(0),s),s),
      mx::logical_and(mx::greater(end,first,s),mx::logical_and(mx::greater(tail,mx::array(0),s),mx::less_equal(tail,mx::array(int(p.page_size)),s),s),s),s),s);
  auto length=mx::where(mx::greater(end,first,s),mx::add(mx::multiply(mx::maximum(mx::subtract(mx::subtract(end,first,s),mx::array(1),s),mx::array(0),s),mx::array(int(p.page_size)),s),tail,s),mx::array(0),s);
  const auto column=[&](const mx::array&v){return mx::reshape(v,{rows,1},s);};
  auto safe_ids=mx::maximum(indices,mx::array(0),s);
  auto at=mx::clip(mx::add(column(first),mx::floor_divide(safe_ids,mx::array(int(p.page_size)),s),s),mx::array(0),mx::array(int(p.max_refs()-1)),s);
  auto page=mx::take(pages,at,0,s);
  auto good=mx::logical_and(column(span_ok),mx::logical_and(mx::greater_equal(indices,mx::array(0),s),mx::less(indices,column(length),s),s),s);
  good=mx::logical_and(good,mx::logical_and(mx::greater_equal(page,mx::array(0),s),mx::less(page,mx::array(int(p.max_pages)),s),s),s);
  auto physical=mx::add(mx::multiply(page,mx::array(int(p.page_size)),s),mx::remainder(safe_ids,mx::array(int(p.page_size)),s),s);
  return mx::where(good,physical,mx::array(-1),s);
}
} // namespace hyle::compute::apple::detail
