#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/mapping.h"

// Test-local immutable GLM shared embedding/head/norm supplier. This header
// neither constructs nor executes a model Graph. Default identity is official
// Flash layer45; explicit geometry and pinned identities can describe other
// GLM artifacts without treating the Flash4096 state calculation as Pro6144.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/glm/glm_shared_vocabulary.h"
