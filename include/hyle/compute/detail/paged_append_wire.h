#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::compute::detail {
// Device wire format. Coordinates are rows, so independently permissioned
// banks with identical page mappings can share this immutable episode data.
// source_kind 0 reads packed new rows at source_row; 1 reads source_page in
// this same arena. Every instruction is validated by the typed page owner.
struct PagedAppendInstruction final {
  std::uint32_t source_kind=0,source_page=0,source_row=0,destination_page=0;
  std::uint32_t destination_row=0,rows=0,reserved0=0,reserved1=0;
  friend constexpr bool operator==(PagedAppendInstruction,PagedAppendInstruction) noexcept = default;
};
struct PagedAppendMetadata final {
  std::uint32_t active_count=0,reserved0=0,reserved1=0,reserved2=0;
  // Followed by instruction_capacity PagedAppendInstruction records.
};
static_assert(sizeof(PagedAppendInstruction)==32 && sizeof(PagedAppendMetadata)==16);

struct PagedAppendNativeSpec final {
  std::uint32_t page_rows=0,physical_pages=0,input_rows=0,instruction_capacity=0;
  std::size_t row_bytes=0;
};
} // namespace hyle::compute::detail
