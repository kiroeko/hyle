#pragma once

#include <array>
#include <stdexcept>

#include "glm_hyper_connection_fragment.h"
#include "glm_original_moe_fragment.h"

// Pure original GLM Flash layer43 shell. The caller appends this layer's
// actual MLA/indexer between prepare and finish in the same Fragment. This
// non-final target layer forwards four BF16 streams; it has no stream mean,
// model.norm, cache owner, parameter loader, provider choice or executor.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_target43_fragment.h"
