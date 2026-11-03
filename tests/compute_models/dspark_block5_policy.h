#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <span>

#include "hyle/error/result.h"
#include "../../engine/inference/mtp_acceptance.h"

// Consumer policy for the formal fixed five-row DSpark draft. This does not
// change its trained shape, execute a verifier, or manufacture a pending lease.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_block5_policy.h"
