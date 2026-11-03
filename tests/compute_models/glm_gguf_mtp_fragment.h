#pragma once

#include "glm_gguf_f32_cache_fragment.h"
#include "glm_original_moe_fragment.h"

// Converted packed weights with the explicit all-F32 runtime chosen for this
// consumer. These are pure ordinary Graph leaves; index algorithm, absolute
// IDs and accepted-journal semantics do not change with the cache encoding.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_gguf_mtp_fragment.h"
