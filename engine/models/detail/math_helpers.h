#pragma once
#include <array>
#include "component_definition.h"
#include "hyle/compute/operation.h"

// Cold declarations of existing public operations. Materialized slice/permute
// retain their original Copy, and reduction retains its original dimensions.
namespace hyle::inference::model_definition::ops {
ValueId reshape(Fragment &f, ValueId input,
                       std::initializer_list<std::int64_t> shape);
ValueId unary(Fragment &f, c::UnaryKind kind, ValueId input);
ValueId binary(Fragment &f, c::BinaryKind kind, ValueId a, ValueId b);
ValueId filled(Fragment &f, const c::TensorDesc &desc, float value);
ValueId sum(Fragment &f, ValueId input, std::int32_t axis);
ValueId reduce(Fragment &f, ValueId value, std::int32_t axis,
               c::ReductionKind kind = c::ReductionKind::kSum);
ValueId slice(Fragment &f, ValueId input, std::int32_t axis,
                      std::int64_t first, std::int64_t count);
ValueId permute(Fragment &f, ValueId input,
                        std::initializer_list<std::int32_t> axes);
ValueId concat(Fragment &f, ValueId a, ValueId b, std::int32_t axis);
}
