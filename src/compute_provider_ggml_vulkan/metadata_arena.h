#pragma once

#include "../compute_ggml_runtime/metadata_arena.h"

#if defined(HYLE_STATIC)
#define HYLE_GGML_VULKAN_INTERNAL_API
#elif defined(HYLE_COMPUTE_PROVIDER_GGML_VULKAN_BUILD)
#define HYLE_GGML_VULKAN_INTERNAL_API HYLE_EXPORT
#else
#define HYLE_GGML_VULKAN_INTERNAL_API HYLE_IMPORT
#endif

namespace hyle::compute::ggml_vulkan::detail {
using ggml_runtime::MetadataArena;
} // namespace hyle::compute::ggml_vulkan::detail
