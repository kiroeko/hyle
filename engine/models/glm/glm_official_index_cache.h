#pragma once
#include <array>
#include "../detail/component_definition.h"
#include "hyle/compute/model_ops.h"

namespace glm_official_index_cache {
using Value=hyle::inference::model_definition::ValueId;
inline constexpr std::int64_t kDepth = 128, kHeads = 32, kPool = 4;
inline constexpr std::int64_t kCapacity = 514, kTopPools = 512;
inline constexpr std::int64_t kLatent = 512, kMlaHeads = 64;
inline constexpr std::int64_t kSelected = kTopPools * kPool;
inline constexpr float kFloor = 1.0e-4F;

struct Quantized final { Value codes, scale, rotated, transformed, input; };
Value cast(hyle::inference::model_definition::Fragment &f, Value x, hyle::compute::DataType type);
Value bfloat_boundary(hyle::inference::model_definition::Fragment &f, Value x);
Quantized quantize(hyle::inference::model_definition::Fragment &f, Value x, std::int64_t rows);
Value score_encoded(hyle::inference::model_definition::Fragment &f, Value bank, Value scales,
                           const Quantized &query, Value raw_weights,
                           std::int64_t visible);
}
