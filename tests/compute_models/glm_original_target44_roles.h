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

#include "glm_original_moe_roles.h"

// Pinned official Flash target44 parameters and final target model.norm.
// Stored BF16 matrices stay BF16. Fixed-source parameter loaders require only
// the two small HC matrices and three short convolution banks widened to F32.
// This supplier owns no Graph, executor, routing policy or mutable model state.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_target44_roles.h"
