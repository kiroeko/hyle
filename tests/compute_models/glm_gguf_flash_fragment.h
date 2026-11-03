#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "glm_gguf_mtp_fragment.h"
#include "glm_gguf_named_parameters.h"
#include "glm_mlx_mtp_fragment.h"
#include "glm_original_target43_fragment.h"
#include "glm_original_target44_fragment.h"

// Converted Flash weights, explicit F32 runtime. Only named-constant binding
// and pure composition live here. The pooled MLA/history mathematics and its
// 16 entries are shared with the existing plain-cache path; the caller still
// owns Program, selected-expert Supply, K1, StateHead and accepted repair.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_gguf_flash_fragment.h"
