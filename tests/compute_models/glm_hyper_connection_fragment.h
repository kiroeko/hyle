#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "d5_moe_semantic_fragments.h"
#include "hyle/compute/model_ops.h"

// Model-local GLM Flash hyper-connections shared by ordinary target layers.
// Fixed06992 torch/unfused contract: raw F32 mixing projection, model-RMS
// epsilon1e-5, HC/Sinkhorn epsilon1e-6 and20 iterations. Default storage has
// BF16 collapse/learned RMS/post boundaries; explicit F32 storage retains the
// original mixed-type F32 path. No parameter supplier or execution.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_hyper_connection_fragment.h"
