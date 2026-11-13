#pragma once

#include "route_reason.h"

namespace hyle::inference {

// Cold selection result only. A model/backend policy defines the applicable
// algorithm, constraints and preferences; this is not a universal capability.
struct RouteChoice final {
  bool enabled = false;
  RouteReason reason = RouteReason::kNone;
};

} // namespace hyle::inference
