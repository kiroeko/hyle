#include "hyle/sched/join.h"

#include <atomic>

#include "hyle/sched/fiber.h"

namespace hyle::sched {
bool is_cancel_requested() noexcept {
  return detail::current_cancel_requested();
}

}  // namespace hyle::sched
