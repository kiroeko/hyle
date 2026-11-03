#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "glm_moe_geometry.h"
#include "glm_shared_vocabulary.h"
#include "hyle/compute/low_precision.h"

// Two actual source encodings, independent of repository identity. MXFP4
// carriers concatenate unchanged U8 codes then U8 UE8 scales; they are not
// GGUF blocks, an FP8-A128 carrier, or an expanded floating weight bank.
namespace glm_moe_storage {
namespace c=hyle::compute;
using Geometry=glm_moe_geometry::Geometry;
using Layout=glm_moe_geometry::Layout;
enum class Kind {kFloat8Block128,kMxfp4Group32};
void require(Kind kind,const Geometry &model);
std::size_t code_bytes(Kind kind,const Geometry &model);
std::size_t scale_bytes(Kind kind,const Geometry &model);
Layout layout(Kind kind,const Geometry &model);
c::TensorDesc packed_description(std::int64_t n,std::int64_t k,std::size_t slots,Kind kind,bool grouped);
} // namespace glm_moe_storage
