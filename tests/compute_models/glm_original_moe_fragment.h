#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_moe_semantic_fragments.h"
#include "glm_moe_geometry.h"
#include "glm_moe_storage.h"
#include "glm_fp8_arithmetic.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Pure original GLM MoE construction, with explicit model-local geometry. All parameters are actual immutable
// Graph constants or explicit compact expert inputs supplied by the caller.
// No weight loading, provider routing, host callback or execution lives here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_moe_fragment.h"
