#include "hyle/compute/apple_batch.h"

#include <array>
#include <algorithm>
#include <new>
#include <optional>

namespace hyle::compute::apple {
Result<Program> prepare_materialized_batch(const Context&context,
    const MaterializedBatchProgram&program,const WorkloadContext&workload)noexcept{
  if(!context.valid() || !program.valid())return std::unexpected(Status::InvalidArgument("Apple materialized batch context/program is empty"));
  try {
    std::array<InputSpec,5> inputs;const auto descriptions=program.descriptions();
    for(std::size_t i=0;i<inputs.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],input_spec(descriptions[i]));}
    // MLX retains the tracing callable. Copy its finite value records so it
    // does not borrow the common program's allocator/schema after prepare.
    // The retained native closure is separate, opaque native metadata.
    std::array<MaterializedBatchStep,19> recipe;const auto count=program.steps().size();
    std::copy(program.steps().begin(),program.steps().end(),recipe.begin());const auto output_value=program.output_value();
    return context.compile([recipe=std::move(recipe),count,output_value](const std::vector<mlx::core::array>&inputs,mlx::core::Stream stream)->Result<std::vector<mlx::core::array>>{
      if(inputs.size()!=5)return std::unexpected(Status::InvalidArgument("Apple materialized batch needs five inputs"));
      std::array<std::optional<mlx::core::array>,24> values;
      for(std::size_t i=0;i<inputs.size();++i)values[i]=inputs[i];
      for(const auto&step:std::span(recipe).first(count)){
        std::vector<mlx::core::array> arguments;arguments.reserve(step.input_count);
        for(std::size_t i=0;i<step.input_count;++i)arguments.push_back(*values[step.inputs[i]]);
        HYLE_ASSIGN_OR_RETURN(auto result,lower(step.operation,arguments,stream));
        if(result.size()!=1)return std::unexpected(Status::Internal("Apple materialized recipe output arity"));
        values[step.output]=std::move(result[0]);
      }
      return std::vector<mlx::core::array>{*values[output_value]};
    },inputs,workload);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Apple materialized batch cold recipe metadata"));}
}
} // namespace hyle::compute::apple
