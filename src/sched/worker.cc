#include "hyle/sched/worker.h"

#include "hyle/core/compiler.h"
#include "hyle/sched/fiber.h"

namespace hyle::sched {

std::size_t Worker::run() {
  std::size_t resumes = 0;
  for (;;) {
    Fiber* f = rq_.next();
    if (f == nullptr) break;  // LIFO slot empty AND FIFO drained -> done
    f->resume();
    ++resumes;
    if (!f->done()) {
      // Voluntary yield: back to the FIFO tail for fair round-robin. The ring is
      // sized for the live fiber set; a full ring here is a sizing bug, not an
      // operational case in sched-02 (global spill is sched-03) -> fail-fast.
      HYLE_FATAL_ASSERT(rq_.schedule_fifo(f));
    }
  }
  return resumes;
}

}  // namespace hyle::sched
