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
  auto cache_classes = available_cpu_cache_classes();
  auto temp = temp_directory();
  auto memory = host_memory_info();
  if (!cpus || cpus->empty() || !cache_classes || cache_classes->empty() ||
      !temp || temp->empty() || !memory ||
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

  char path[4096];
  auto created = create_temp_file(nullptr, "hyle-platform-", path,
                                  sizeof(path));
  if (!created) return 9;
  auto owner_copy = created->file.duplicate();
  if (!owner_copy || !owner_copy->info()) return 10;
  if (!created->file.close().ok() || !owner_copy->close().ok()) return 11;

  auto random = open_random_access_file(path, kOpenRead | kOpenWrite);
  if (!random || !random->write_all_at(0, "ok", 2).ok()) return 12;
  auto random_copy = random->duplicate();
  std::array<std::byte, 2> payload{};
  if (!random_copy || !random_copy->read_exact_at(0, payload).ok()) return 13;
  if (!random->close().ok() || !random_copy->close().ok() ||
      !remove_file(path).ok()) {
    return 14;
  }

  return 0;
}
