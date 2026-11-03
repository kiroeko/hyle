#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "glm_original_target43_fragment.h"
#include "glm_original_target44_fragment.h"

// Fixed MLX-VLM f1a9b7 GLM Flash target shell. All streams and state inputs are
// actual F32; original BF16 dense/HC weights remain immutable BF16 banks.
// The caller appends its real MLA/MoE and owns the ordinary Graph execution.
// There is no role loader, cache owner, route choice or executor here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_mlx_target_fragment.h"
