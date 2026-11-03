#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../engine/inference/artifact_digest.h"

// Test input only. The caller supplies the fixed artifact identity and logical
// element count. There is no model registry, network access or product loader.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/detail/original_role_input.h"
