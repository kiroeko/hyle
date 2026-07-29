#include <hyle/platform/convenience.h>
#include <hyle/platform/dynlib.h>
#include <hyle/platform/thread.h>
#include <hyle/platform/virtual_memory.h>
#include <hyle/platform/wait.h>

#include <atomic>
#include <cstddef>

int main() {
  using namespace hyle::platform;

  auto cpus = available_cpu_ids();
  auto temp = temp_directory();
  if (!cpus || cpus->empty() || !temp || temp->empty()) return 1;

  auto region = VmRegion::allocate_rw(page_size());
  if (!region || !region->valid()) return 2;
  static_cast<std::byte*>(region->data())[0] = std::byte{0x5a};

  Parker parker;
  parker.unpark();
  if (!parker.park_timeout(0)) return 3;

  std::atomic<int> ran{0};
  auto thread = Thread::spawn([&ran] { ran.store(42); });
  if (!thread) return 4;
  thread->join();
  if (ran.load() != 42) return 5;

  auto current = DynLib::current();
  if (!current || !current->valid()) return 6;
  return 0;
}
