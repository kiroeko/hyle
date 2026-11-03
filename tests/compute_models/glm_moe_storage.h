#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "glm_moe_geometry.h"
#include "glm_shared_vocabulary.h"

// Two actual source encodings, independent of repository identity. MXFP4
// carriers concatenate unchanged U8 codes then U8 UE8 scales; they are not
// GGUF blocks, an FP8-A128 carrier, or an expanded floating weight bank.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_moe_storage.h"
