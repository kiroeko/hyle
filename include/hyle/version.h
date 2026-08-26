/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <cstdint>

// Checked-in product metadata is also parsed by the root build. These macros
// identify the headers; they are not a C product API or an ABI guarantee.
#define HYLE_VERSION_MAJOR 0
#define HYLE_VERSION_MINOR 1
#define HYLE_VERSION_PATCH 0

#define HYLE_DETAIL_VERSION_STRINGIZE_INNER(value) #value
#define HYLE_DETAIL_VERSION_STRINGIZE(value) \
  HYLE_DETAIL_VERSION_STRINGIZE_INNER(value)
#define HYLE_VERSION_STRING                     \
  HYLE_DETAIL_VERSION_STRINGIZE(HYLE_VERSION_MAJOR) \
  "." HYLE_DETAIL_VERSION_STRINGIZE(HYLE_VERSION_MINOR) \
  "." HYLE_DETAIL_VERSION_STRINGIZE(HYLE_VERSION_PATCH)

namespace hyle {

inline constexpr std::uint32_t kVersionMajor = HYLE_VERSION_MAJOR;
inline constexpr std::uint32_t kVersionMinor = HYLE_VERSION_MINOR;
inline constexpr std::uint32_t kVersionPatch = HYLE_VERSION_PATCH;
static_assert(kVersionMajor <= 0xffu && kVersionMinor <= 0xffu &&
              kVersionPatch <= 0xffu,
              "HYLE packed version fields are 8-bit values");
inline constexpr std::uint32_t kVersionPacked =
    ((kVersionMajor & 0xffu) << 16u) | ((kVersionMinor & 0xffu) << 8u) |
    (kVersionPatch & 0xffu);
inline constexpr char kVersionString[] = HYLE_VERSION_STRING;

}  // namespace hyle
