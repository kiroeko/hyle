#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "glm_gguf_model_types.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Converted GGUF projection leaves only. Original packed bytes and native
// activation quantization remain provider operands. No weight decoding,
// BF16 activation/store insertion, loading, state or execution lives here.
// In particular, the per-head K/V banks are ALREADY split and transposed by
// the converter; do not pass them through the original FP8 KV_b preparation.
namespace glm_gguf_projection_fragment {
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
using Value=sem::ValueId;
using Projection=glm_gguf_model_types::Projection;
Projection linear(sem::Fragment &f,Value input,Value weight);

// Router/index-weight projections are original F32[N,K], not packed and
// not BF16 metadata. A View gives MatMul its logical [K,N] without a copy.
Projection f32_linear(sem::Fragment &f,Value input,Value weight);

// A[T,H,K] x original packed bank[H,N,K] -> F32[T,H,N]. Each slot h
// addresses bank h; there is no cross-head dense multiply or 64-Graph loop.
// Ordinary MatMul's current packed RHS admission is rank2 only. The existing
// GroupedMatMul is the natural public operation for these native Q8 head rows.
Projection per_head(sem::Fragment &f,Value input,Value weight);

Value rms(sem::Fragment &f,Value input,Value gamma,float epsilon);
struct PackedProjections final {
  Projection linear(sem::Fragment &f,Value input,Value weight) const {
    return f.description(weight).is_plain()?glm_gguf_projection_fragment::f32_linear(f,input,weight):
        glm_gguf_projection_fragment::linear(f,input,weight);
  }
  Projection per_head(sem::Fragment &f,Value input,Value weight) const {
    return glm_gguf_projection_fragment::per_head(f,input,weight);
  }
};
} // namespace glm_gguf_projection_fragment
