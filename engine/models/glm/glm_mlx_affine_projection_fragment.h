#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "../detail/component_definition.h"
#include "hyle/compute/model_ops.h"

// Original affine planes, without a loader, executor or storage conversion.
// In the Q32 domain BF16/F32 A plus original F16 SF/bias promotes to F32.
// Keep code*scale+bias in F32; narrowing decoded W to BF16 changes the model.
namespace glm_mlx_affine_projection_fragment {
namespace sem=hyle::inference::model_definition;
namespace c=hyle::compute;
using Value=sem::ValueId;
struct Parameters final {Value codes,scales,biases;std::uint32_t group=64;};
struct Projection final {Value input,output;std::uint32_t bits,group;};

std::uint32_t bits(const sem::Fragment &f,Value input,const Parameters &p,bool grouped);

Projection linear(sem::Fragment &f,Value input,const Parameters &p);
Projection grouped(sem::Fragment &f,Value input,const Parameters &p,Value actual_local_ids);
} // namespace glm_mlx_affine_projection_fragment
