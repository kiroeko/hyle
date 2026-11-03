#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_shared_vocabulary.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Distinct original Pro target77 shared-index MLA parameters and model.norm.
// There are no EH/shared_head/indexer roles in target77. The original target
// MoE uses its own authenticated catalogue with the same Pro geometry.
// This source loads only 15 static roles; it never executes a model.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_pro_target_roles.h"
