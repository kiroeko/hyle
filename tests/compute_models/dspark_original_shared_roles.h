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

// Test-only supply of the exact nine shared FFN Q8_0 banks. They are ordinary
// immutable MatMul operands, with no decoder/loader service or executor. The
// physical original[N,K] rows are described as logical MatMul RHS[K,N].
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_shared_roles.h"
