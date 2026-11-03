#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "glm_official_index_cache.h"
#include "glm_original_moe_fragment.h"

// Model-local original GLM Pro NextN construction. Values name operands in
// this same ordinary Fragment; no loading, execution, provider or acceptance
// policy is hidden here. The selected official non-fused indexer keeps its
// Hadamard/UE8M0 boundary. Main cache rows contain BOTH CKV512 and rotated KPE64.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_pro_mtp_fragment.h"
