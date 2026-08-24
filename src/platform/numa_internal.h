#pragma once

#include "hyle/platform/numa.h"

#if defined(HYLE_TESTING) && defined(_WIN32)
namespace hyle::platform::detail {

// TEST-ONLY: inject runtime API availability into the Windows capability
// decision without replacing or invoking OS entry points.
HYLE_PLATFORM_API bool windows_selected_interleave_supported_for_testing(
    bool has_virtual_alloc_ex_numa, bool has_virtual_alloc2) noexcept;

}  // namespace hyle::platform::detail
#endif
