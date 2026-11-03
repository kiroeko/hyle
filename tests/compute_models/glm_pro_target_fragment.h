#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "glm_pro_mtp_fragment.h"

// Original last main layer77. Its shared DSA selection is an explicit input
// from layer74, carried through75/76; it has NO trained local indexer. Inputs
// hidden/residual name the separate BF16 outputs at the layer76 boundary.
// No loading, execution, provider selection or acceptance policy lives here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_pro_target_fragment.h"
