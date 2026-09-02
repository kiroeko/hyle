#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace hyle::container::detail {

// C++ has no standard inverse for an arbitrary pointer-to-data-member. HYLE's
// supported 64-bit compiler ABIs represent a direct data-member pointer as an
// offset. Keep that ABI boundary in one factor-private helper and fail closed
// if a compiler selects a richer representation.
template <class Owner, auto Member> struct IntrusiveMemberAccess {
  using member_type =
      std::remove_reference_t<decltype(std::declval<Owner &>().*Member)>;

  static member_type *member_of(Owner *owner) noexcept {
    return &(owner->*Member);
  }
  static const member_type *member_of(const Owner *owner) noexcept {
    return &(owner->*Member);
  }

  static Owner *owner_of(member_type *member) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(member) -
                         static_cast<std::uintptr_t>(member_offset());
    return std::launder(reinterpret_cast<Owner *>(address));
  }
  static const Owner *owner_of(const member_type *member) noexcept {
    return owner_of(const_cast<member_type *>(member));
  }

private:
  static std::ptrdiff_t member_offset() noexcept {
#if defined(_MSC_VER)
    static_assert(sizeof(Member) == sizeof(int),
                  "unsupported MSVC data-member pointer representation");
    return static_cast<std::ptrdiff_t>(std::bit_cast<int>(Member));
#elif defined(__GNUC__) || defined(__clang__)
    static_assert(sizeof(Member) == sizeof(std::ptrdiff_t),
                  "unsupported Itanium data-member pointer representation");
    return std::bit_cast<std::ptrdiff_t>(Member);
#else
#  error "Intrusive containers require a supported data-member pointer ABI"
#endif
  }
};

} // namespace hyle::container::detail
