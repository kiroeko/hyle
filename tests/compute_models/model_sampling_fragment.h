#pragma once

#include <array>
#include <bit>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

#include "d5_qwen38_24t_semantic_fragments.h"
#include "../../engine/inference/sampling_graph.h"

// Only fixed-shape recipe declaration adaption. Local fragment identities stay
// local; no Graph generation is forged and no sampling mathematics is copied.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/detail/model_sampling_fragment.h"
