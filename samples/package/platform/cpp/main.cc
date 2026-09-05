#include <hyle/platform/cpu_convenience.h>
#include <hyle/platform/dynlib.h>
#include <hyle/platform/fs_convenience.h>
#include <hyle/platform/fs.h>
#include <hyle/platform/host_memory.h>
#include <hyle/platform/process.h>
#include <hyle/platform/random.h>
#include <hyle/platform/thread.h>
#include <hyle/platform/thread_convenience.h>
#include <hyle/platform/virtual_memory.h>
#include <hyle/platform/wait.h>

#include <array>
#include <atomic>
#include <cstddef>

int main() {
  using namespace hyle::platform;

  auto cpus = available_cpu_ids();
  auto temp = temp_directory();
  auto memory = host_memory_info();
  if (!cpus || cpus->empty() || !temp || temp->empty() || !memory ||
      memory->physical_total_bytes == 0) {
    return 1;
  }

  std::array<std::byte, 16> seed{};
  if (!secure_random(seed).ok()) return 2;

  auto region = VmRegion::allocate_rw(page_size());
  if (!region || !region->valid()) return 3;
  static_cast<std::byte*>(region->data())[0] = std::byte{0x5a};

  Parker parker;
  parker.unpark();
  if (!parker.park_for(hyle::time::Duration::zero())) return 4;

  std::atomic<int> ran{0};
  auto thread = Thread::spawn([&ran] { ran.store(42); });
  if (!thread) return 5;
  thread->join();
  if (ran.load() != 42) return 6;

  auto current = DynLib::current();
  if (!current || !current->valid()) return 7;

  // Exercise the compiled process-policy symbol without changing the host
  // application's scheduling policy in this delivery-contract sample.
  if (!set_process_priority(static_cast<ProcessPriority>(99))
           .IsInvalidArgument()) {
    return 8;
  }

  return 0;
}
