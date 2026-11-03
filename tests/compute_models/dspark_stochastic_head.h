#pragma once

#include <limits>

#include "dspark_original_draft_program.h"
#include "model_sampling_fragment.h"
#include "../../engine/inference/mtp_acceptance.h"

// Source-specific Markov head wiring only. All distribution, draws and
// speculative acceptance mathematics are shared with the inference engine.
// Model-private declarations; oracle/fixture code remains in tests.
#include "../../engine/models/deepseek/dspark_stochastic_head.h"
