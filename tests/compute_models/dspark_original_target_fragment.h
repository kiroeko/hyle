#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "d5_deepseek_official_attention.h"
#include "dspark_target_geometry.h"
#include "dspark_formal_fragment.h"
#include "hyle/compute/low_precision.h"

// Shared original Flash0731/Pro0813 target-tail mathematics. Every parameter is an actual
// caller-owned immutable Value; every cache is a typed ordinary Graph input.
// Decode plans are cold frontier facts, checked by the consumer before submit.
// No parameter loading, routing-provider choice, executor or acceptance policy.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_target_fragment.h"
