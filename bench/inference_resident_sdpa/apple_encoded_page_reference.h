#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/batch_paged_native.h"

namespace hyle::bench::resident {
// Benchmark-only observation of the original command buffers. It never waits
// in place of the measured native invocation or weakens its completion check.
class OriginalMetalCompletion final {
public:
  OriginalMetalCompletion();
  ~OriginalMetalCompletion();
  Status check_completed();
  struct State;
  State &state() noexcept;
private:
  std::unique_ptr<State> state_;
};
struct AppleEncodedPeerContext final {
  const compute::apple::Context *context = nullptr;
  std::shared_ptr<OriginalMetalCompletion> completion;
};
Result<compute::apple::Program> encoded_page_reference_program(
    const AppleEncodedPeerContext &, const compute::NativePagedBatchProgram &,
    compute::BatchLayoutVersion, const compute::WorkloadContext &);
}
