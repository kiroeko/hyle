#pragma once

#include "evaluation_failure.h"

namespace hyle::inference {

enum class StepPosition { kCurrent, kSuccessor };
enum class StepSuccessor { kNone, kPrepared, kRefused, kUnavailable };
enum class StepCarry { kCurrent, kSuccessor, kDeferred, kUnavailable };
enum class StepDriveResult { kCompleted, kDeferred };

struct StepDriveOptions final {
  bool resume = false;
  bool defer_successor = false;
};

// Serialized control of original model steps. Adapter borrows their existing
// owner, calls, Sources and credits; no callback or execution state survives
// this function. The model supplies preparation facts, mathematics and Head
// publication. Graph/Compute remain the only execution and permission owners.
// A synchronous adapter needs no successor-specific methods.
template<class Adapter>
StepDriveResult drive_steps(Adapter &adapter, StepDriveOptions options = {}) {
  static_assert(noexcept(adapter.retire_failure(static_cast<EvaluationFailure *>(nullptr))));
  static_assert(noexcept(adapter.recover_records(false)));
  try {
    adapter.prepare(StepPosition::kCurrent);
    if (!options.resume) adapter.submit(StepPosition::kCurrent);
    if (adapter.cancelled()) {
      adapter.cancel(false);
      return StepDriveResult::kCompleted;
    }
    if constexpr (!Adapter::kSupportsSuccessor) {
      adapter.finish(StepPosition::kCurrent);
      return StepDriveResult::kCompleted;
    } else {
      const auto successor = adapter.prepare_successor();
      bool submitted = false;
      if (successor == StepSuccessor::kPrepared) {
        adapter.submit(StepPosition::kSuccessor);
        submitted = true;
      }
      if (adapter.cancelled()) {
        adapter.cancel(submitted);
        return StepDriveResult::kCompleted;
      }
      adapter.finish(StepPosition::kCurrent);
      if (successor == StepSuccessor::kRefused) {
        adapter.finish_refusal();
        return StepDriveResult::kCompleted;
      }
      if (successor == StepSuccessor::kUnavailable) {
        adapter.carry(StepCarry::kUnavailable);
        return StepDriveResult::kCompleted;
      }
      if (submitted) {
        if (!adapter.keep_successor()) {
          adapter.discard_successor();
          return StepDriveResult::kCompleted;
        }
        adapter.bind_successor();
        if (options.defer_successor) {
          adapter.carry(StepCarry::kDeferred);
          return StepDriveResult::kDeferred;
        }
        adapter.finish(StepPosition::kSuccessor);
        adapter.carry(StepCarry::kSuccessor);
      } else {
        adapter.carry(StepCarry::kCurrent);
      }
      adapter.finish_reuse();
      return StepDriveResult::kCompleted;
    }
  } catch (EvaluationFailure &failure) {
    const auto retired = adapter.retire_failure(&failure);
    failure.observe_retirement(retired);
    adapter.recover_records(retired.retired);
    throw;
  } catch (...) {
    const auto retired = adapter.retire_failure(nullptr);
    adapter.recover_records(retired.retired);
    throw;
  }
}

} // namespace hyle::inference
