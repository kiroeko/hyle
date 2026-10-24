#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "prepared_roles.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/weight_import.h"
#include "hyle/mem/mapping.h"

namespace qwen35 {
// One cold projection group only. Original codes are mapped, never expanded or
// retained as a second weight bank. Only the small scale grid is widened.
struct ImportedFloat8Weight final {
  hyle::compute::TensorDesc description;
  hyle::mem::Buffer storage;
  std::size_t temporary_scale_bytes = 0;
};
inline hyle::Result<ImportedFloat8Weight> import_float8_block128_roles(
    const prepared::Role &codes, const prepared::Role &scales) {
  namespace c = hyle::compute;
  namespace m = hyle::mem;
  if (codes.projection_codec != prepared::ProjectionCodec::kFloat8Block128 ||
      codes.dtype != "F8_E4M3" || codes.shape.size() != 2 || codes.shape[0] <= 0 ||
      codes.shape[1] <= 0 || codes.shape[1] % 128 || scales.dtype != "BF16" ||
      scales.shape != std::vector<std::int64_t>{codes.shape[0] / 128 + (codes.shape[0] % 128 != 0), codes.shape[1] / 128})
    return std::unexpected(hyle::Status::InvalidArgument("invalid original FP8 block128 role group"));
  HYLE_ASSIGN_OR_RETURN(auto wd, c::TensorDesc::dense(c::DataType::kFloat8E4M3FN, codes.shape));
  HYLE_ASSIGN_OR_RETURN(auto sd, c::TensorDesc::dense(c::DataType::kFloat32, scales.shape));
  HYLE_ASSIGN_OR_RETURN(auto yd, c::float8_weight_description(codes.shape[1], codes.shape[0],
      c::Float8ScaleMode::kFloat32Block128));
  HYLE_ASSIGN_OR_RETURN(auto wb, wd.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto sb, sd.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto yb, yd.required_bytes());
  if (codes.bytes != wb || scales.bytes != sb / 2)
    return std::unexpected(hyle::Status::InvalidArgument("original FP8 role byte extent differs"));
  HYLE_ASSIGN_OR_RETURN(auto original_codes, m::FileMapping::open(codes.source.string().c_str(),
      m::MappingAccess::kReadOnly, codes.offset, codes.bytes));
  HYLE_ASSIGN_OR_RETURN(auto original_scales, m::FileMapping::open(scales.source.string().c_str(),
      m::MappingAccess::kReadOnly, scales.offset, scales.bytes));
  HYLE_ASSIGN_OR_RETURN(auto raw, original_scales.bytes());
  HYLE_ASSIGN_OR_RETURN(auto sf, m::Buffer::create(m::system_resource(), sb, 64));
  HYLE_ASSIGN_OR_RETURN(auto sf_map, sf.view().map_host(m::HostAccess::kWrite));
  HYLE_ASSIGN_OR_RETURN(auto sf_bytes, sf_map.writable_bytes());
  for (std::size_t i = 0; i < raw.size() / 2; ++i) {
    // Safetensors is little-endian. Construct native F32 bits exactly, without
    // multiplying weights or rounding any product through BF16.
    const auto bits = (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[2*i])) |
        (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[2*i+1])) << 8)) << 16;
    const auto native = std::bit_cast<std::array<std::byte, 4>>(bits);
    std::copy(native.begin(), native.end(), sf_bytes.begin() + static_cast<std::ptrdiff_t>(4*i));
  }
  HYLE_RETURN_IF_ERROR(sf_map.flush()); HYLE_RETURN_IF_ERROR(sf_map.close());
  HYLE_ASSIGN_OR_RETURN(auto destination, m::Buffer::create(m::system_resource(), yb, 256));
  HYLE_ASSIGN_OR_RETURN(auto cv, original_codes.view());
  HYLE_ASSIGN_OR_RETURN(auto wv, c::ConstTensorView::create(wd, cv));
  HYLE_ASSIGN_OR_RETURN(auto sv, c::ConstTensorView::create(sd, sf.view()));
  HYLE_ASSIGN_OR_RETURN(auto yv, c::TensorView::create(yd, destination.view()));
  HYLE_RETURN_IF_ERROR(c::import_float8_block128_weight(wv, sv, yv));
  return ImportedFloat8Weight{yd, std::move(destination), sb};
}
} // namespace qwen35
