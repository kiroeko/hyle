#include "hyle/compute/weight_import.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "hyle/compute/low_precision.h"
#include "hyle/container/small_vector.h"

namespace hyle::compute {
namespace {

bool host_identity(mem::ConstBufferView view) noexcept {
  const auto flags = view.resource().properties();
  return flags.has(mem::MemoryProperty::kHostAccessible) &&
         flags.has(mem::MemoryProperty::kHostAddressIdentity);
}

bool overlapping(std::size_t x, std::size_t nx, std::size_t y,
                 std::size_t ny) noexcept {
  return x <= y ? y - x < nx : x - y < ny;
}

bool metadata_overlap(mem::ConstBufferView a,
                      mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() &&
         a.native_handle() == b.native_handle() &&
         overlapping(a.offset(), a.size(), b.offset(), b.size());
}

bool address_bounded(const mem::HostMapping &mapping) noexcept {
  const auto bytes = mapping.bytes();
  const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
  return bytes.data() && bytes.size() <= UINTPTR_MAX - address;
}

bool address_overlap(const mem::HostMapping &a,
                     const mem::HostMapping &b) noexcept {
  return overlapping(reinterpret_cast<std::uintptr_t>(a.bytes().data()),
                     a.bytes().size(),
                     reinterpret_cast<std::uintptr_t>(b.bytes().data()),
                     b.bytes().size());
}

Status source_domain(ConstTensorView view, DataType type) noexcept {
  if (!view.valid() || !view.storage().valid() ||
      !view.description().is_plain() || view.description().rank() != 2 ||
      view.description().data_type() != type)
    return Status::InvalidArgument("FP8 import needs plain matrices of the declared E4 and U8/F32 types");
  for (const auto stride : view.description().byte_strides())
    if (!std::in_range<std::size_t>(stride))
      return Status::OutOfRange("FP8 import source stride is outside host size domain");
  return Status::Ok();
}

} // namespace

Status import_float8_block128_weight(ConstTensorView weight,
                                    ConstTensorView scales,
                                    TensorView output) noexcept {
  HYLE_RETURN_IF_ERROR(source_domain(weight, DataType::kFloat8E4M3FN));
  if (!scales.valid() || (scales.description().data_type() != DataType::kUInt8 &&
                          scales.description().data_type() != DataType::kFloat32))
    return Status::InvalidArgument("FP8 import scales must be U8 E8M0 or raw F32");
  const bool stored_f32 = scales.description().data_type() == DataType::kFloat32;
  HYLE_RETURN_IF_ERROR(source_domain(scales,
      stored_f32 ? DataType::kFloat32 : DataType::kUInt8));
  if (!output.valid() || !output.storage().valid())
    return Status::InvalidArgument("FP8 import output is invalid");
  const auto n = weight.description().dimensions()[0];
  const auto k = weight.description().dimensions()[1];
  HYLE_ASSIGN_OR_RETURN(auto canonical,
      float8_weight_description(k, n, Float8ScaleMode::kFloat32Block128));
  if (output.description() != canonical ||
      scales.description().dimensions()[0] != n / 128 + (n % 128 != 0) ||
      scales.description().dimensions()[1] != k / 128)
    return Status::InvalidArgument("FP8 import scale grid or canonical output differs");
  HYLE_ASSIGN_OR_RETURN(const auto weight_bytes, weight.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(const auto scale_bytes, scales.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(const auto output_bytes, canonical.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto weight_range, weight.storage().subview(0, weight_bytes));
  HYLE_ASSIGN_OR_RETURN(auto scale_range, scales.storage().subview(0, scale_bytes));
  HYLE_ASSIGN_OR_RETURN(auto output_range, output.storage().subview(0, output_bytes));
  if (!host_identity(weight_range) || !host_identity(scale_range) ||
      !host_identity(output_range))
    return Status::Unimplemented("FP8 import requires canonical host mappings without device transfer");
  if (metadata_overlap(weight_range, output_range) ||
      metadata_overlap(scale_range, output_range))
    return Status::InvalidArgument("FP8 import output overlaps source storage");

  mem::HostMapping weight_map, scale_map, output_map;
  bool wrote = false;
  const auto finish = [&](Status status) {
    if (wrote && !output_range.resource().properties().has(
                     mem::MemoryProperty::kHostDeviceCoherent))
      status.Update(output_map.flush());
    status.Update(output_map.close());
    status.Update(scale_map.close());
    status.Update(weight_map.close());
    return status;
  };
  auto mapped_weight = weight_range.map_host();
  if (!mapped_weight) return finish(std::move(mapped_weight).error());
  weight_map = std::move(*mapped_weight);
  auto mapped_scale = scale_range.map_host();
  if (!mapped_scale) return finish(std::move(mapped_scale).error());
  scale_map = std::move(*mapped_scale);
  auto mapped_output = output_range.map_host(mem::HostAccess::kWrite);
  if (!mapped_output) return finish(std::move(mapped_output).error());
  output_map = std::move(*mapped_output);
  if (!address_bounded(weight_map) || !address_bounded(scale_map) ||
      !address_bounded(output_map))
    return finish(Status::OutOfRange("FP8 import host address range overflows"));
  if (address_overlap(weight_map, output_map) || address_overlap(scale_map, output_map))
    return finish(Status::InvalidArgument("FP8 import mapped output overlaps source storage"));
  if (!weight_range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
    const auto status = weight_map.invalidate();
    if (!status.ok()) return finish(status);
  }
  if (!scale_range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
    const auto status = scale_map.invalidate();
    if (!status.ok()) return finish(status);
  }
  auto writable = output_map.writable_bytes();
  if (!writable) return finish(std::move(writable).error());
  const auto source = weight_map.bytes();
  const auto scale_source = scale_map.bytes();
  auto target = *writable;
  const auto rows = static_cast<std::size_t>(n);
  const auto columns = static_cast<std::size_t>(k);
  const auto wr = static_cast<std::size_t>(weight.description().byte_strides()[0]);
  const auto wc = static_cast<std::size_t>(weight.description().byte_strides()[1]);
  wrote = true;
  for (std::size_t row = 0; row < rows; ++row) {
    if (wc == 1)
      std::memcpy(target.data() + row * columns, source.data() + row * wr, columns);
    else
      for (std::size_t column = 0; column < columns; ++column)
        target[row * columns + column] = source[row * wr + column * wc];
  }
  const auto sr = static_cast<std::size_t>(scales.description().byte_strides()[0]);
  const auto sc = static_cast<std::size_t>(scales.description().byte_strides()[1]);
  const auto scale_columns = columns / 128;
  const auto scale_rows = rows / 128 + (rows % 128 != 0);
  for (std::size_t row = 0; row < scale_rows; ++row)
    for (std::size_t column = 0; column < scale_columns; ++column) {
      std::uint32_t bits;
      if (stored_f32) {
        std::memcpy(&bits, scale_source.data() + row * sr + column * sc, sizeof(bits));
      } else {
        const auto code = std::to_integer<std::uint32_t>(scale_source[row * sr + column * sc]);
        bits = code == 0 ? 0x00400000u : code == 255 ? 0x7f800001u : code << 23;
      }
      std::memcpy(target.data() + rows * columns +
                      (row * scale_columns + column) * sizeof(bits), &bits, sizeof(bits));
    }
  return finish(Status::Ok());
}

Status encode_e8m0_scales(ConstTensorView scales, TensorView output) noexcept {
  if (!scales.valid() || !scales.storage().valid() ||
      !output.valid() || !output.storage().valid())
    return Status::InvalidArgument("E8M0 scale import views are invalid");
  const auto &source_desc = scales.description();
  const auto &target_desc = output.description();
  if (!source_desc.is_plain() || source_desc.data_type() != DataType::kFloat32 ||
      !target_desc.is_plain() || !target_desc.is_dense() ||
      target_desc.data_type() != DataType::kUInt8 ||
      source_desc.rank() != target_desc.rank() ||
      !std::equal(source_desc.dimensions().begin(), source_desc.dimensions().end(),
                  target_desc.dimensions().begin()))
    return Status::InvalidArgument("E8M0 scale import needs matching F32 and dense U8 shapes");
  for (const auto stride : source_desc.byte_strides())
    if (!std::in_range<std::size_t>(stride))
      return Status::OutOfRange("E8M0 scale source stride is outside host size domain");
  HYLE_ASSIGN_OR_RETURN(const auto count, source_desc.element_count());
  if (count == 0)
    return Status::InvalidArgument("E8M0 scale import needs a nonempty grid");
  HYLE_ASSIGN_OR_RETURN(const auto source_bytes, source_desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(const auto target_bytes, target_desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto source_range, scales.storage().subview(0, source_bytes));
  HYLE_ASSIGN_OR_RETURN(auto target_range, output.storage().subview(0, target_bytes));
  if (!host_identity(source_range) || !host_identity(target_range))
    return Status::Unimplemented("E8M0 scale import requires canonical host mappings without device transfer");
  if (metadata_overlap(source_range, target_range))
    return Status::InvalidArgument("E8M0 scale output overlaps source storage");
  mem::HostMapping source_map, target_map;
  bool wrote = false;
  const auto finish = [&](Status status) {
    if (wrote && !target_range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      status.Update(target_map.flush());
    status.Update(target_map.close());
    status.Update(source_map.close());
    return status;
  };
  auto mapped_source = source_range.map_host();
  if (!mapped_source) return finish(std::move(mapped_source).error());
  source_map = std::move(*mapped_source);
  auto mapped_target = target_range.map_host(mem::HostAccess::kWrite);
  if (!mapped_target) return finish(std::move(mapped_target).error());
  target_map = std::move(*mapped_target);
  if (!address_bounded(source_map) || !address_bounded(target_map))
    return finish(Status::OutOfRange("E8M0 scale host address range overflows"));
  if (address_overlap(source_map, target_map))
    return finish(Status::InvalidArgument("E8M0 scale mapped output overlaps source storage"));
  if (!source_range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
    const auto status = source_map.invalidate();
    if (!status.ok()) return finish(status);
  }
  auto writable = target_map.writable_bytes();
  if (!writable) return finish(std::move(writable).error());
  const auto source = source_map.bytes();
  const auto dims = source_desc.dimensions(), strides = source_desc.byte_strides();
  const auto load_code = [&](std::size_t index, bool &valid) {
    std::size_t offset = 0;
    for (auto axis = dims.size(); axis > 0; --axis) {
      const auto extent = static_cast<std::size_t>(dims[axis - 1]);
      offset += (index % extent) * static_cast<std::size_t>(strides[axis - 1]);
      index /= extent;
    }
    std::uint32_t bits;
    std::memcpy(&bits, source.data() + offset, sizeof(bits));
    const auto exponent = (bits >> 23) & 255u;
    const bool up = (bits & 0x00400000u) &&
        ((bits & 0x003fffffu) || exponent != 0);
    const auto code = exponent + static_cast<unsigned>(up);
    valid = (bits & 0x80000000u) == 0 && bits != 0 &&
            exponent != 255 && code != 255;
    return static_cast<std::byte>(code);
  };
  // Validate the complete logical grid before writing even its first byte.
  for (std::size_t i = 0; i < count; ++i) {
    bool valid;
    (void)load_code(i, valid);
    if (!valid)
      return finish(Status::InvalidArgument("E8M0 scale import needs positive finite values rounding below255"));
  }
  wrote = true;
  for (std::size_t i = 0; i < count; ++i) {
    bool valid;
    (*writable)[i] = load_code(i, valid);
  }
  return finish(Status::Ok());
}

Status pack_int4_zero_rows(std::span<const ConstTensorView> sources,
                          std::int64_t rows_per_source,
                          TensorView output) noexcept {
  if (sources.empty() || rows_per_source <= 0 || !output.valid() ||
      !output.storage().valid())
    return Status::InvalidArgument("INT4 zero packing needs nonempty rows/sources/output");
  if (!std::in_range<std::int64_t>(sources.size()) ||
      sources.size() > static_cast<std::uint64_t>(INT64_MAX / rows_per_source))
    return Status::OutOfRange("INT4 zero row count overflows");
  const auto rows = static_cast<std::int64_t>(sources.size()) * rows_per_source;
  const auto source_words = rows_per_source / 8 + (rows_per_source % 8 != 0);
  std::int64_t groups = 0;
  for (const auto &source : sources) {
    const auto &desc = source.description();
    if (!source.valid() || !source.storage().valid() || !desc.is_plain() ||
        desc.data_type() != DataType::kInt32 || desc.rank() != 2 ||
        desc.dimensions()[0] != source_words || desc.dimensions()[1] <= 0 ||
        (groups != 0 && desc.dimensions()[1] != groups))
      return Status::InvalidArgument("INT4 zero source word/group grid differs");
    groups = desc.dimensions()[1];
    for (const auto stride : desc.byte_strides())
      if (!std::in_range<std::size_t>(stride))
        return Status::OutOfRange("INT4 zero source stride is outside host size domain");
  }
  HYLE_ASSIGN_OR_RETURN(auto expected, TensorDesc::dense(DataType::kInt32,
      std::array{rows / 8 + (rows % 8 != 0), groups}));
  if (output.description() != expected)
    return Status::InvalidArgument("INT4 zero output is not the tight concatenated grid");
  HYLE_ASSIGN_OR_RETURN(const auto output_bytes, expected.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto output_range, output.storage().subview(0, output_bytes));
  if (!host_identity(output_range))
    return Status::Unimplemented("INT4 zero packing needs canonical host output");
  for (const auto &source : sources) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, source.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto range, source.storage().subview(0, bytes));
    if (!host_identity(range))
      return Status::Unimplemented("INT4 zero packing needs canonical host sources");
    if (metadata_overlap(range, output_range))
      return Status::InvalidArgument("INT4 zero output overlaps a source");
  }
  container::SmallVector<mem::HostMapping, 4, mem::AllocatorRef> maps(mem::system_allocator());
  if (!maps.try_resize(sources.size()))
    return Status::ResourceExhausted("INT4 zero source mapping metadata");
  mem::HostMapping output_map;
  bool wrote = false;
  const auto finish = [&](Status status) {
    if (wrote && !output_range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent))
      status.Update(output_map.flush());
    status.Update(output_map.close());
    for (auto &mapping : maps) status.Update(mapping.close());
    return status;
  };
  for (std::size_t i = 0; i < sources.size(); ++i) {
    const auto &source = sources[i];
    HYLE_ASSIGN_OR_RETURN(const auto bytes, source.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto range, source.storage().subview(0, bytes));
    auto mapped = range.map_host();
    if (!mapped) return finish(std::move(mapped).error());
    maps[i] = std::move(*mapped);
    if (!address_bounded(maps[i]))
      return finish(Status::OutOfRange("INT4 zero source host address range overflows"));
    if (!range.resource().properties().has(mem::MemoryProperty::kHostDeviceCoherent)) {
      const auto status = maps[i].invalidate();
      if (!status.ok()) return finish(status);
    }
  }
  auto mapped = output_range.map_host(mem::HostAccess::kWrite);
  if (!mapped) return finish(std::move(mapped).error());
  output_map = std::move(*mapped);
  if (!address_bounded(output_map))
    return finish(Status::OutOfRange("INT4 zero output host address range overflows"));
  for (const auto &source_map : maps)
    if (address_overlap(source_map, output_map))
      return finish(Status::InvalidArgument("INT4 zero mapped output overlaps a source"));
  auto writable = output_map.writable_bytes();
  if (!writable) return finish(std::move(writable).error());
  wrote = true;
  std::fill(writable->begin(), writable->end(), std::byte{});
  const auto columns = static_cast<std::size_t>(groups);
  const auto source_rows = static_cast<std::size_t>(rows_per_source);
  for (std::size_t i = 0; i < sources.size(); ++i) {
    const auto source = maps[i].bytes();
    if (rows_per_source % 8 == 0 && sources[i].description().is_dense()) {
      const auto bytes = source_rows / 8 * columns * sizeof(std::uint32_t);
      std::memcpy(writable->data() + i * bytes, source.data(), bytes);
      continue;
    }
    const auto stride = sources[i].description().byte_strides();
    for (std::size_t row = 0; row < source_rows; ++row)
      for (std::size_t group = 0; group < columns; ++group) {
        std::uint32_t code, value;
        std::memcpy(&code, source.data() + (row / 8) * static_cast<std::size_t>(stride[0]) +
            group * static_cast<std::size_t>(stride[1]), sizeof(code));
        const auto global_row = i * source_rows + row;
        const auto offset = ((global_row / 8) * columns + group) * sizeof(value);
        std::memcpy(&value, writable->data() + offset, sizeof(value));
        value |= ((code >> (4 * (row % 8))) & 15u) << (4 * (global_row % 8));
        std::memcpy(writable->data() + offset, &value, sizeof(value));
      }
  }
  return finish(Status::Ok());
}

} // namespace hyle::compute
