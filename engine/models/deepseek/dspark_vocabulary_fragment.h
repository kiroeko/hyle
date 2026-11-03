#pragma once

#include "../detail/component_definition.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

// Model-local vocabulary math. Actual descriptors determine the storage
// operation; no provider, artifact name, weight decoding or execution is here.
namespace dspark_vocabulary_fragment {
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
using Value=sem::ValueId;
bool bfloat_table(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden);
bool packed_head(const c::TensorDesc &head,std::int64_t vocabulary,std::int64_t hidden);
bool embedding_table(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden);
void require_embedding(const c::TensorDesc &table,std::int64_t vocabulary,std::int64_t hidden);
void require_head(const c::TensorDesc &head,std::int64_t vocabulary,std::int64_t hidden);
struct Embedding final {Value decoded,stored;};
Embedding append_embedding(sem::Fragment &f,Value table,Value ids,
    std::int64_t vocabulary,std::int64_t hidden);
Value append_head(sem::Fragment &f,Value actual_bfloat,Value weight,
    std::int64_t vocabulary,std::int64_t hidden);
} // namespace dspark_vocabulary_fragment
