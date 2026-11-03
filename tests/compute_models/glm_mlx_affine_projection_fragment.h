#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"

// Original affine planes, without a loader, executor or storage conversion.
// In the Q32 domain BF16/F32 A plus original F16 SF/bias promotes to F32.
// Keep code*scale+bias in F32; narrowing decoded W to BF16 changes the model.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_mlx_affine_projection_fragment.h"
