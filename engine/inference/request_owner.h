#pragma once

#include <cstddef>
#include <stdexcept>

#include "request_loop.h"

namespace hyle::inference {

struct RequestOwnerInputProgress final {
  bool arrived = false;
  bool waiting = false;
};

// Common host orchestration only. Callers retain model state, native work,
// actual retirement, input and output owners. Template binding keeps the
// existing direct calls and does not introduce a runtime execution interface.
template<class ServiceOutput, class Pump, class Observe, class AwaitRetirement,
         class PollInput, class WaitIdle>
void drive_request_owner(std::size_t &active, ServiceOutput &&service_output, Pump &&pump,
                         Observe &&observe, AwaitRetirement &&await_retirement,
                         PollInput &&poll_input, WaitIdle &&wait_idle) {
  while (active) {
    bool output_progress = service_output();
    const auto pumped = pump();
    observe(pumped);
    output_progress = service_output() || output_progress;
    await_retirement(pumped);
    const auto input = poll_input(pumped);
    if (!pumped.executed && !pumped.waiting_retirement && !input.arrived && !output_progress && active) {
      if (wait_idle(pumped, input)) continue;
      throw std::runtime_error("generation admission has no runnable resident request");
    }
  }
}

} // namespace hyle::inference
