#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_shared_vocabulary.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Exact official Flash MLA attention supplier, separate from Graph math.
// The default pins identify NextN45; another actual layer supplies its own
// complete authenticated pin array and retains the same geometry contract.
// Nine original bodies become six immutable runtime banks: three original
// E4/F32 block128 pairs use the existing canonical importer; kv_b remains ONE
// original BF16 bank. K/V absorption uses views of that bank, never unrelated
// generated matrices or a full F32 weight expansion.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_attention_roles.h"
