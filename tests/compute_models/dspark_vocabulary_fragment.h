#pragma once

#include "d5_qwen38_24t_semantic_fragments.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Model-local vocabulary math. Actual descriptors determine the storage
// operation; no provider, artifact name, weight decoding or execution is here.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_vocabulary_fragment.h"
