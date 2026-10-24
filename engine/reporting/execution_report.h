#pragma once

#include <iosfwd>

namespace hyle::inference {
struct StepObservation;
struct PreparationObservation;
}

namespace hyle::engine::reporting {

void write_step_observation(std::ostream &output, const inference::StepObservation &step);
void write_preparation_observation(std::ostream &output, const inference::StepObservation &step,
                                   const inference::PreparationObservation &preparation);

} // namespace hyle::engine::reporting
