#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// The two pinned GA configs use the same target mathematics. Only these
// immutable cold dimensions/routing facts vary; codecs, state ownership and
// execution stay in their existing layers. This is not a model registry.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_target_geometry.h"
