#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "dspark_stored_projection_fragment.h"

// A test-consumer boundary for the selected Flash0731 Q8_0 sidecar. This
// supplies ordinary Graph values and retained constant/input permissions;
// it does not load files, schedule work, identify providers or own an executor.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_target_binding.h"
