#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "glm_hyper_connection_fragment.h"
#include "glm_original_moe_fragment.h"
#include "glm_original_target44_roles.h"

// Pure original GLM Flash layer44 construction. The caller binds distinct
// authenticated44 constants, compact selected expert banks and actual state.
// HC uses fixed06992's explicit torch/unfused path, including BF16 collapse
// before learned RMS; it does not claim TileLang fused-norm bit equivalence.
// KDA normalization is F32 inside the recurrent boundary. The original wrapper
// retains BF16 convolution/recurrent/output-gate stores; the explicit MLX
// constructor retains F32 stores and its own q/k and recurrence boundaries.
// There is no loader/provider/executor here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_target44_fragment.h"
