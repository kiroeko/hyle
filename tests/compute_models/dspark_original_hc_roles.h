#pragma once

#include <array>
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

#include "dspark_target_binding.h"
#include "original_role_input.h"

// The exact34 original F32 roles from the formal Flash0731 Q8 DSpark sidecar.
// This is test-only data supply plus ordinary readonly parameter binding. It
// has no network, Graph executor, mutable cache, hidden precision conversion
// or original-router-bias injection. Runtime roles retain their host source
// because cold model reference callbacks borrow the original coefficients.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_hc_roles.h"
