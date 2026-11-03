#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "dspark_formal_fragment.h"
#include "dspark_vocabulary_fragment.h"

// The fixed formal DSpark forward_embed/three stages/forward_head expressed
// as ordinary Values. Accepted main histories are inputs; proposal KV remains
// episode-local. Loading, resident uploads, execution and StateHead policy
// belong to the existing consumer and are deliberately absent here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_original_draft_fragment.h"
