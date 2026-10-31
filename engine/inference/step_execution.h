#pragma once

#include "evaluation_failure.h"

namespace hyle::inference {

enum class StepPosition { kCurrent, kSuccessor };
enum class StepSuccessor { kNone, kPrepared, kRefused, kUnavailable };
enum class StepCarry { kCurrent, kSuccessor, kDeferred, kUnavailable };
enum class StepDriveResult { kCompleted, kDeferred, kPending };

// Serialized host progress only. The original adapter's model keeps both
// submitted steps and every native completion/permission owner.
struct StepDriveCursor final {
  bool awaiting_current = false;
  StepSuccessor successor = StepSuccessor::kNone;
};

struct StepDriveOptions final {
  bool resume = false;
  bool defer_successor = false;
  StepDriveCursor *cursor = nullptr;
};

// Serialized control of original model steps. Adapter borrows their existing
// owner, calls, Sources and credits; only the optional host cursor survives
// this function, never a callback or native owner. The model supplies preparation facts, mathematics and Head
// publication. Graph/Compute remain the only execution and permission owners.
// A synchronous adapter needs no successor-specific methods.
template<class Adapter>
StepDriveResult drive_steps(Adapter &adapter, StepDriveOptions options = {}) {
  static_assert(noexcept(adapter.retire_failure(static_cast<EvaluationFailure *>(nullptr))));
  static_assert(noexcept(adapter.recover_records(false)));
  try {
    const bool observing = options.cursor && options.cursor->awaiting_current;
    if (!observing) {
      adapter.prepare(StepPosition::kCurrent);
      if (!options.resume) adapter.submit(StepPosition::kCurrent);
    }
    if (adapter.cancelled()) {
      adapter.cancel(observing && options.cursor->successor == StepSuccessor::kPrepared);
      if (options.cursor) *options.cursor = {};
      return StepDriveResult::kCompleted;
    }
    const auto ready = [&](StepSuccessor successor) {
      if constexpr (requires { adapter.current_ready(); adapter.defer_current(successor); }) {
        if (options.cursor && !adapter.current_ready()) {
          // Save the actual disposition only after preparation/submission.
          // Reobserving this cursor performs none of those actions again.
          if (!observing) adapter.defer_current(successor);
          *options.cursor = {true, successor};
          return false;
        }
      }
      return true;
    };
    if constexpr (!Adapter::kSupportsSuccessor) {
      if (!ready(StepSuccessor::kNone)) return StepDriveResult::kPending;
      adapter.finish(StepPosition::kCurrent);
      if (options.cursor) *options.cursor = {};
      return StepDriveResult::kCompleted;
    } else {
      const auto successor = observing ? options.cursor->successor : adapter.prepare_successor();
      const bool submitted = successor == StepSuccessor::kPrepared;
      if (submitted && !observing) {
        adapter.submit(StepPosition::kSuccessor);
      }
      if (adapter.cancelled()) {
        adapter.cancel(submitted);
        if (options.cursor) *options.cursor = {};
        return StepDriveResult::kCompleted;
      }
      if (!ready(successor)) return StepDriveResult::kPending;
      adapter.finish(StepPosition::kCurrent);
      if (options.cursor) *options.cursor = {};
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
    if (retired.retired && options.cursor) *options.cursor = {};
    throw;
  } catch (...) {
    const auto retired = adapter.retire_failure(nullptr);
    adapter.recover_records(retired.retired);
    if (retired.retired && options.cursor) *options.cursor = {};
    throw;
  }
}

} // namespace hyle::inference
