#pragma once
#include <numeric>
#include "glm_flash_selection_extent.h"
#include "glm_gguf_compute_fragment.h"
#include "glm_mlx_original_roles.h"
#include "glm_mlx_affine_projection_fragment.h"
#include "glm_original_moe_fragment.h"

// Original MLX affine/F16 metadata path, with an explicit F32 activation and
// plain F32 pooled-state contract. No quantizer, BF16 intermediate store,
// provider choice, loader or executor. Ordinary public mixed DenseLinearF32
// rank3 support is a prerequisite for the two original BF16 KV_b head banks.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_mlx_mtp_fragment.h"
