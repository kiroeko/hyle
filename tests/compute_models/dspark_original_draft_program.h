#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "cpu_c4_model_branches.h"
#include "dspark_original_draft_fragment.h"
#include "dspark_original_formal_roles.h"

// Named formal parameter wiring into one ordinary Fragment. Source ownership
// and ready device leases are shared by target/draft/vocabulary; construction
// neither uploads weights nor executes model operations. The caller retains
// the existing four-phase router cut, native runner and StateHead.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_draft_program.h"
