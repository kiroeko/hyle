#pragma once

#include <cmath>
#include <stdexcept>

#include "hyle/compute/model_ops.h"

// Immutable graph-construction arithmetic, independent of source identity and
// provider choice. The existing official default remains byte-for-byte Divide.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_fp8_arithmetic.h"
