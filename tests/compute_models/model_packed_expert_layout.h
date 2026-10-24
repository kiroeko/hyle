#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "hyle/compute/low_precision.h"
#include "hyle/compute/resource_leases.h"
// Mechanical extraction of the existing Q24 descriptor/stride arithmetic.
// Only capacity/domain and mutually-consistent H/I are explicit cold facts.
// This header owns no upload lifetime, model reader or execution loop.
namespace model_packed_expert_layout {
namespace c=hyle::compute;using hyle::Result;using hyle::Status;
class Layout final {
 public:
  static Result<Layout> create(const std::array<c::TensorDesc, 3> &sources,std::size_t capacity) {
    Layout result;
    if(!capacity||sources[0].rank()!=3||sources[0].dimensions()[0]<=0||
        capacity>std::size_t(sources[0].dimensions()[0])||capacity>(std::numeric_limits<std::size_t>::max()-1)/3)
      return std::unexpected(Status::InvalidArgument("invalid actual expert domain/resident capacity"));
    result.capacity_=capacity;result.domain_=sources[0].dimensions()[0];
    const auto hidden=sources[0].dimensions()[2],intermediate=sources[0].dimensions()[1];
    for (std::size_t leg = 0; leg < sources.size(); ++leg) {
      const auto &source = sources[leg];
      const auto codec = c::gguf_quantized_type(source.storage_format());
      const auto n = leg == 2 ? hidden : intermediate;
      const auto k = leg == 2 ? intermediate : hidden;
      if (source.rank() != 3 || source.dimensions()[0] != result.domain_ ||
          source.dimensions()[1] != n || source.dimensions()[2] != k ||
          source.byte_strides().size() != 3 || source.byte_strides()[0] <= 0)
        return std::unexpected(Status::InvalidArgument("packed expert source needs its actual full-domain grouped geometry"));
      if (std::uint64_t(source.byte_strides()[0]) > std::uint64_t(std::numeric_limits<std::size_t>::max()))
        return std::unexpected(Status::OutOfRange("packed original expert stride exceeds size_t"));
      c::TensorDesc resident;
      std::size_t plane_bytes = 0;
      if (codec && source.data_type() == c::DataType::kFloat32) {
        HYLE_ASSIGN_OR_RETURN(auto canonical, c::gguf_grouped_weight_description(*codec,
            c::DataType::kFloat32, result.domain_, n, k));
        if (source != canonical)
          return std::unexpected(Status::InvalidArgument("packed expert layout differs from its native GGUF codec"));
        HYLE_ASSIGN_OR_RETURN(resident, c::gguf_grouped_weight_description(*codec,
            c::DataType::kFloat32, std::int64_t(capacity), n, k));
        plane_bytes = std::size_t(source.byte_strides()[0]);
      } else if (source.is_plain() && source.data_type() == c::DataType::kBFloat16) {
        // A gate/up plane is a view of [E,2I,H]: expert stride includes the
        // other half. Copy only this plane, compactly, into the resident bank.
        HYLE_ASSIGN_OR_RETURN(auto plane, c::TensorDesc::dense(c::DataType::kBFloat16, std::array{n, k}));
        HYLE_ASSIGN_OR_RETURN(plane_bytes, plane.required_bytes());
        if (source.byte_strides()[2] != 2 || source.byte_strides()[1] != k * 2 ||
            std::uint64_t(source.byte_strides()[0]) < plane_bytes)
          return std::unexpected(Status::InvalidArgument("packed BF16 selected expert rows must be contiguous within each plane"));
        HYLE_ASSIGN_OR_RETURN(resident, c::TensorDesc::dense(c::DataType::kBFloat16,
            std::array{std::int64_t(capacity), n, k}));
      } else return std::unexpected(Status::InvalidArgument("unsupported packed selected expert storage"));
      HYLE_ASSIGN_OR_RETURN(auto bank_bytes, resident.required_bytes());
      const auto stride = std::uint64_t(source.byte_strides()[0]);
      if (stride > std::uint64_t(std::numeric_limits<std::size_t>::max()) ||
          stride > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
          plane_bytes > std::numeric_limits<std::size_t>::max() / capacity ||
          bank_bytes != plane_bytes * capacity)
        return std::unexpected(Status::OutOfRange("packed expert stride or resident extent overflows"));
      result.sources_[leg] = source;
      result.residents_[leg] = resident;
      // Native sliced assembly preserves a plain bank's scalar type. Only
      // encoded banks use raw U8 fragments; the bytes are never converted.
      const auto upload_type = resident.is_plain() ? resident.data_type() : c::DataType::kUInt8;
      const auto upload_width = c::data_type_size(upload_type);
      if (upload_width == 0 || plane_bytes % upload_width != 0)
        return std::unexpected(Status::InvalidArgument("packed upload plane has a partial scalar"));
      HYLE_ASSIGN_OR_RETURN(result.uploads_[leg], c::TensorDesc::dense(upload_type,
          std::array<std::int64_t, 1>{std::int64_t(plane_bytes / upload_width)}));
      result.expert_bytes_[leg] = plane_bytes;
      result.bank_bytes_[leg] = bank_bytes;
      HYLE_ASSIGN_OR_RETURN(auto aligned, align(bank_bytes));
      HYLE_RETURN_IF_ERROR(add(result.slot_bytes_, aligned));
      HYLE_RETURN_IF_ERROR(add(result.logical_bytes_, bank_bytes));
    }
    HYLE_ASSIGN_OR_RETURN(result.map_, c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{result.domain_}));
    HYLE_ASSIGN_OR_RETURN(result.map_bytes_, result.map_.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto aligned_map, align(result.map_bytes_));
    HYLE_RETURN_IF_ERROR(add(result.slot_bytes_, aligned_map));
    HYLE_RETURN_IF_ERROR(add(result.logical_bytes_, result.map_bytes_));
    if (result.slot_bytes_ > std::numeric_limits<std::size_t>::max() / 2)
      return std::unexpected(Status::OutOfRange("packed two-generation resident capacity overflows"));
    return result;
  }
  const c::TensorDesc &source(std::size_t leg) const { return sources_.at(leg); }
  const c::TensorDesc &resident(std::size_t leg) const { return residents_.at(leg); }
  const c::TensorDesc &upload_description(std::size_t leg) const { return uploads_.at(leg); }
  std::size_t expert_bytes(std::size_t leg) const { return expert_bytes_.at(leg); }
  std::size_t bank_bytes(std::size_t leg) const { return bank_bytes_.at(leg); }
  const c::TensorDesc &map_description() const noexcept { return map_; }
  std::size_t map_bytes() const noexcept { return map_bytes_; }
  std::size_t logical_bytes() const noexcept { return logical_bytes_; }
  std::size_t slot_bytes() const noexcept { return slot_bytes_; }
  std::size_t fragment_count() const noexcept { return 3 * capacity_ + 1; }
  std::size_t capacity() const noexcept { return capacity_; }
  std::int64_t domain() const noexcept { return domain_; }
  Status validate_descriptions(const std::array<c::TensorDesc,3> &sources) const {
    if(sources!=sources_)return Status::InvalidArgument("supplied original source differs from cold expert layout");
    return Status::Ok();
  }
 private:
  static Result<std::size_t> align(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - 255)
      return std::unexpected(Status::OutOfRange("packed resident alignment overflows"));
    return (bytes + 255) & ~std::size_t{255};
  }
  static Status add(std::size_t &total, std::size_t bytes) {
    if (total > std::numeric_limits<std::size_t>::max() - bytes)
      return Status::OutOfRange("packed resident byte capacity overflows");
    total += bytes; return Status::Ok();
  }
  std::size_t capacity_=0;std::int64_t domain_=0;
  std::array<c::TensorDesc, 3> sources_, residents_, uploads_;
  std::array<std::size_t, 3> expert_bytes_{}, bank_bytes_{};
  c::TensorDesc map_;
  std::size_t map_bytes_ = 0, logical_bytes_ = 0, slot_bytes_ = 0;
};
} // namespace model_packed_expert_layout
