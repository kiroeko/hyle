#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "glm_gguf_model_types.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Converted GGUF projection leaves only. Original packed bytes and native
// activation quantization remain provider operands. No weight decoding,
// BF16 activation/store insertion, loading, state or execution lives here.
// In particular, the per-head K/V banks are ALREADY split and transposed by
// the converter; do not pass them through the original FP8 KV_b preparation.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_gguf_projection_fragment.h"
