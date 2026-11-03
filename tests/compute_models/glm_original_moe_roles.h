#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "glm_shared_vocabulary.h"
#include "glm_moe_geometry.h"
#include "glm_moe_storage.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Model-local original GLM MoE input. The default Flash catalogue pin is
// NextN layer45; another actual layer supplies its authenticated explicit pin.
// One read-only mapping retains the full domain without a floating shadow.
// The private binary catalogue is emitted beside its authoritative JSON by
// exact-range acquisition; its compile-time identity authenticates role order.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_original_moe_roles.h"
