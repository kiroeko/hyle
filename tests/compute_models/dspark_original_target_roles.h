#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "original_role_input.h"
#include "dspark_original_role_source.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/weight_import.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Original formal Flash0731 target40->41->42 and distinct main finalHC/norm.
// One authenticated bundle owns all raw roles; experts remain original packed
// FP4 plus UE8M0 group32, not GGUF MXFP4. This file does not execute a Graph.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_target_roles.h"
