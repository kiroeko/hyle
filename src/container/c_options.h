#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hyle::container::c_detail {

// Decode a versioned C options object without reading beyond its declared
// prefix. The caller default-initializes `output`; this copies the required
// prefix and reports the declared byte count for optional field-by-field reads.
template <class Options>
bool read_required_options_prefix(const Options* input,
                                  std::uint32_t required_size,
                                  Options& output,
                                  std::uint32_t& declared_size) noexcept {
  static_assert(std::is_standard_layout_v<Options>);
  static_assert(std::is_trivially_copyable_v<Options>);
  static_assert(offsetof(Options, struct_size) == 0);
  static_assert(sizeof(Options) <= UINT32_MAX);

  if (input == nullptr) return false;
  std::memcpy(&declared_size, input, sizeof(declared_size));
  if (declared_size < required_size) return false;
  std::memcpy(&output, input, required_size);
  output.struct_size = static_cast<std::uint32_t>(sizeof(Options));
  return true;
}

// A size that ends inside an optional field treats that field as absent. This
// is safer than manufacturing a pointer/function pointer from partial bytes;
// every historical ABI size naturally ends after a complete field.
template <class Options, class Field>
void read_optional_options_field(const Options* input,
                                 std::uint32_t declared_size,
                                 std::size_t field_offset,
                                 Field& output) noexcept {
  if (field_offset <= declared_size &&
      sizeof(Field) <= declared_size - field_offset) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(input);
    std::memcpy(&output, bytes + field_offset, sizeof(Field));
  }
}

}  // namespace hyle::container::c_detail
