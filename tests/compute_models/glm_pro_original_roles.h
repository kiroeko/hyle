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

#include "glm_shared_vocabulary.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/weight_import.h"

// Immutable original Pro NextN78 parameters. The 24 layer-owned files become
// 17 runtime banks; seven E4/SF pairs retain their original canonical bytes.
// Embedding, head and NextN shared-head norm use ONE shared-vocabulary source.
// This supplier owns mappings and banks, with no model execution or cache policy.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_pro_original_roles.h"
