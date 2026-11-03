#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "cpu_c4_model_branches.h"
#include "dspark_original_target_fragment.h"
#include "dspark_original_target_roles.h"

// Cold wiring of the original three-layer target tail. The caller executes the cold
// OA preparation once, binds its three actual leases, then cuts/runs this
// ordinary Fragment at the three actual router frontiers. No execution,
// resource allocation, acceptance policy or host model math is hidden here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_target_program.h"
