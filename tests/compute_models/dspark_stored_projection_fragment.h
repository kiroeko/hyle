#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/graph.h"

// Cold typed projection leaves only. Plain BF16 weights explicitly consume
// BF16 activations and return F32 accumulation through DenseLinearF32. This
// is a Hyle precision policy, not equality to every peer's F32-A dispatch.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_stored_projection_fragment.h"
