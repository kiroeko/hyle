#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>

#include "hyle/compute/weight_assembly.h"

namespace hyle::compute::detail {

struct WeightAssemblySize final {
  std::size_t banks = 0, logical_bytes = 0, physical_bytes = 0;
};

// Shared metadata validation only: no executor, allocation, source mapping or
// transfer. Catalog is the original cold-sorted unique WeightSpec array. Entry
// is the original native {spec,offset,bytes} record, already reserved cold.
// Plain banks retain their scalar contract. Encoded banks accept already
// formatted bytes; neither path interprets codecs or repacks padded bit rows.
template<class Spec, class Entry>
Result<WeightAssemblySize> prepare_weight_slices(
    std::span<const Spec> catalog, std::span<const WeightSlice> slices,
    std::span<Entry> entries, std::size_t capacity,
    std::size_t alignment) noexcept {
  if (catalog.empty() || slices.empty() || !std::has_single_bit(alignment))
    return std::unexpected(Status::InvalidArgument("weight assembly catalog/fragments/alignment are invalid"));
  if (entries.size() < catalog.size())
    return std::unexpected(Status::ResourceExhausted("weight assembly bank metadata capacity is too small"));
  // Validate every source and destination extent before changing any native
  // staging bytes. Repeated bank IDs are intentional; destination byte ranges
  // rather than IDs must be unique. Read-only source aliases are allowed.
  for (std::size_t i = 0; i < slices.size(); ++i) {
    const auto &part = slices[i];
    const auto found = std::lower_bound(catalog.begin(), catalog.end(), part.id,
        [](const auto &spec, std::uint64_t id) { return spec.id < id; });
    if (found == catalog.end() || found->id != part.id || !part.source.valid() ||
        !part.source.storage().valid() || !found->description.valid() ||
        !part.source.description().is_plain() ||
        !part.source.description().is_dense() ||
        !part.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("weight assembly bank/type/dense host source differs"));
    const bool plain_bank = found->description.is_plain();
    if (plain_bank ? (!found->description.is_dense() ||
            found->description.data_type() != part.source.description().data_type())
        : part.source.description().data_type() != DataType::kUInt8)
      return std::unexpected(Status::InvalidArgument("weight assembly requires matching plain scalars or encoded-bank U8 bytes"));
    if (!part.source.storage().resource().properties().has(mem::MemoryProperty::kHostAddressIdentity))
      return std::unexpected(Status::Unimplemented("weight assembly needs canonical host source addresses"));
    HYLE_ASSIGN_OR_RETURN(const auto bytes, part.source.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(const auto bank_bytes, found->description.required_bytes());
    const auto width = plain_bank ? data_type_size(found->description.data_type()) : 1;
    if (bytes == 0 || width == 0 || part.destination_offset % width != 0 ||
        part.destination_offset > bank_bytes || bytes > bank_bytes - part.destination_offset ||
        bytes > part.source.storage().size())
      return std::unexpected(Status::OutOfRange("weight assembly fragment exceeds aligned bank/source extent"));
    for (std::size_t j = 0; j < i; ++j) {
      const auto &earlier = slices[j];
      if (earlier.id != part.id) continue;
      HYLE_ASSIGN_OR_RETURN(const auto earlier_bytes, earlier.source.description().required_bytes());
      const bool overlap = part.destination_offset <= earlier.destination_offset
          ? earlier.destination_offset - part.destination_offset < bytes
          : part.destination_offset - earlier.destination_offset < earlier_bytes;
      if (overlap)
        return std::unexpected(Status::InvalidArgument("weight assembly destination fragments overlap"));
    }
  }
  WeightAssemblySize result;
  for (const auto &bank : catalog) {
    std::size_t covered = 0;
    for (const auto &part : slices) if (part.id == bank.id) {
      HYLE_ASSIGN_OR_RETURN(const auto bytes, part.source.description().required_bytes());
      // Nonoverlap plus bounded extents above prove this sum cannot exceed
      // the bank, including before the coverage equality below.
      covered += bytes;
    }
    if (covered == 0) continue;
    HYLE_ASSIGN_OR_RETURN(const auto bytes, bank.description.required_bytes());
    if (covered != bytes)
      return std::unexpected(Status::InvalidArgument("weight assembly bank has unfilled bytes"));
    const auto bank_alignment = (std::max)(alignment, bank.description.required_alignment());
    const auto padding = (bank_alignment - result.physical_bytes % bank_alignment) % bank_alignment;
    if (result.physical_bytes > capacity || padding > capacity - result.physical_bytes ||
        bytes > capacity - result.physical_bytes - padding)
      return std::unexpected(Status::ResourceExhausted("assembled weights exceed fixed slot capacity"));
    entries[result.banks++] = {bank, result.physical_bytes + padding, bytes};
    result.logical_bytes += bytes;
    result.physical_bytes += padding + bytes;
  }
  return result;
}

template<class Entry>
Result<std::size_t> weight_slice_offset(
    std::span<const Entry> entries, std::uint64_t id) noexcept {
  const auto found = std::lower_bound(entries.begin(), entries.end(), id,
      [](const auto &entry, std::uint64_t wanted) { return entry.spec.id < wanted; });
  if (found == entries.end() || found->spec.id != id)
    return std::unexpected(Status::NotFound("weight assembly bank is not selected"));
  return found->offset;
}

} // namespace hyle::compute::detail
