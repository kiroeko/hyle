#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

// Immutable-by-use, model-local MoE math/storage geometry. Artifact identity
// and its authenticated catalogue stay in the original parameter supplier.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_moe_geometry.h"
