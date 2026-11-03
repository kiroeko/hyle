#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/model_ops.h"
#include "model_sampling_fragment.h"

// Pure model-local target-head construction. There is no
// loader, invocation, provider choice, or target model execution in this file.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_target_verifier_fragment.h"
