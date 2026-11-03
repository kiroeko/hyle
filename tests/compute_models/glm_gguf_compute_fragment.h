#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>

#include "glm_gguf_model_types.h"
#include "glm_gguf_projection_fragment.h"
#include "glm_target_verifier_fragment.h"

// Explicit F32 COMPUTE leaves around original converted packed parameters.
// No implicit cache, hidden/journal store, index quantization, selection or
// state controller is chosen here. A caller must apply its declared store
// boundary and pass the resulting ACTUAL operand into the next leaf.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_gguf_compute_fragment.h"
