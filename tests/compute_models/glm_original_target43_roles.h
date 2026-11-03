#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_original_attention_roles.h"
#include "glm_original_indexer_roles.h"
#include "glm_original_moe_roles.h"

// Original Flash layer43. Its MLA/indexer data reuse the existing authenticated
// suppliers with explicit43 pins; only HC, input norm and q_a live here. EH,
// embedding/hidden norms and the shared LM are NextN concerns, absent from43.
// The owner retains mapped originals and once-prepared banks, never an executor.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_target43_roles.h"
