#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "original_role_input.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/weight_import.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Private original-format fixture I/O. This shares byte authentication and
// lossless carrier preparation, not an executor, model registry or cache.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/detail/dspark_original_role_source.h"
