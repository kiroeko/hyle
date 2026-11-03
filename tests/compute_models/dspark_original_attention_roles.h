#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "dspark_target_binding.h"
#include "original_role_input.h"

// Exact twelve original Q8_0 attention banks, supplied once as ordinary
// readonly Graph operands. The output-a role's original8192 contiguous rows
// naturally describe8groups of1024 rows; no cross-group dense product or
// weight conversion is needed. This helper owns no executor or model cache.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_attention_roles.h"
