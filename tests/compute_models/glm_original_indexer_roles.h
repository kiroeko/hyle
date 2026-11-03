#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "d5_glm_remaining_fragments.h"
#include "original_role_input.h"

// Test-local immutable supplier for seven original Flash indexer roles.
// Default pins identify NextN45; another actual layer supplies all seven pins.
// The checkpoint stores BF16; fixed06992 widens APE, LayerNorm and head-gate
// coefficients to F32 when loading. Runtime banks below do that exact widening
// once, while the three dense projection banks retain their original BF16 bits.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_indexer_roles.h"
