#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "../detail/component_definition.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/graph.h"

// Cold typed projection leaves only. Plain BF16 weights explicitly consume
// BF16 activations and return F32 accumulation through DenseLinearF32. This
// is a Hyle precision policy, not equality to every peer's F32-A dispatch.
namespace dspark_stored_projection_fragment {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace sem=hyle::inference::model_definition;

c::TensorDesc bfloat_matrix(std::int64_t outputs,std::int64_t width);
bool bfloat_matrix(const c::TensorDesc &weight,std::int64_t outputs,std::int64_t width);
sem::ValueId linear(sem::Fragment &f,sem::ValueId input,sem::ValueId weight);
hyle::Result<g::ValueId> linear(g::GraphBuilder &f,g::ValueId input,
    g::ValueId weight,const c::TensorDesc &description);
sem::ValueId dense_permute(sem::Fragment &f,sem::ValueId value);
// Once-only preparation from the original grouped [G,N,K] BF16 rows to an
// ordinary batch-MatMul RHS [G,K,N]. The caller shares the actual output lease.
sem::ValueId prepare_output_a(sem::Fragment &f,sem::ValueId original,
    std::int64_t groups,std::int64_t outputs,std::int64_t width);
// The result is a REAL BF16 BMM result [T,G,N]. There is no observable raw
// F32 accumulator. A later F32 Cast is only an exact stored-result widening.
sem::ValueId output_a(sem::Fragment &f,sem::ValueId input,sem::ValueId rhs,
    std::int64_t groups,std::int64_t outputs,std::int64_t width);
} // namespace dspark_stored_projection_fragment
