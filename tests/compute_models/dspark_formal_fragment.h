#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_deepseek_official_attention.h"
#include "dspark_target_geometry.h"
#include "hyle/compute/low_precision.h"

// Formal DeepSeek projection, HC-head and MoE mathematics shared by trained
// target layers and their original DSpark stages. All weights/caches are
// caller-owned Values. No loader, host coefficients, executor or provider
// decision participates in construction.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_formal_fragment.h"
