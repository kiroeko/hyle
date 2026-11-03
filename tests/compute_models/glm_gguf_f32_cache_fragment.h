#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "glm_gguf_compute_fragment.h"

// Explicit converted F32 runtime. A plain index cache contains ONLY F32 keys;
// no fake scale member, Scatter1, E4 quantizer, BF16 store or implicit codec.
// These are cold descriptions/pure Graph producers, not another StateStore,
// StateHead, execution loop, acceptance policy or artifact loader.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_gguf_f32_cache_fragment.h"
