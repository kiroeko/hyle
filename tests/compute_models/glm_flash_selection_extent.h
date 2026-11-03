#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Cold geometry only. The physical selection carrier stays2051 entries;
// readers consume the actual selected pools immediately followed by live tail.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_flash_selection_extent.h"
