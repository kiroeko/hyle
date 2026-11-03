#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dspark_target_binding.h"
#include "original_role_input.h"
#include "hyle/mem/mapping.h"

// Test-only pinned compressed role supply. Nine original 256-expert files
// stay read-only mapped; selected 17B/32 ranges are never expanded to F32.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_expert_roles.h"
