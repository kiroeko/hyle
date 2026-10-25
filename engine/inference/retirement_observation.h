#pragma once

#include "hyle/error/status.h"

namespace hyle::inference {

// Actual owner retirement and the observed outcome are independent facts.
// Neither an error nor a successful observation grants release permission.
// This value owns no execution, resource, cleanup action or persistent ledger.
struct [[nodiscard]] RetirementObservation final {
  bool retired = false;
  hyle::Status status{};
};

// A caller's expected cancellation must not hide a later real retirement
// failure. Other first errors preserve their original identity.
inline void record_retirement_status(hyle::Status &status,
                                     const hyle::Status &observed) noexcept {
  if (status.IsCancelled() && !status.has_native_error() && !observed.ok() &&
      (!observed.IsCancelled() || observed.has_native_error()))
    status = observed;
  else status.Update(observed);
}

} // namespace hyle::inference
