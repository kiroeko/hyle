#include <hyle/platform/convenience.h>
#include <hyle/platform/dynlib.h>
#include <hyle/platform/fs.h>
#include <hyle/platform/mapped_file.h>
#include <hyle/platform/thread.h>
#include <hyle/platform/virtual_memory.h>
#include <hyle/platform/wait.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

namespace {

struct TempPath {
  char value[1024]{};
  ~TempPath() { (void)hyle::platform::remove_file(value); }
};

}  // namespace

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
  if (!parker.park_for(hyle::time::Duration::zero())) return 3;

  std::atomic<int> ran{0};
  auto thread = Thread::spawn([&ran] { ran.store(42); });
  if (!thread) return 4;
  thread->join();
  if (ran.load() != 42) return 5;

  auto current = DynLib::current();
  if (!current || !current->valid()) return 6;

  // A model loader normally opens and validates once, then transfers the file
  // owner into a persistent read-only mapping. Mapping by owner avoids a second
  // path lookup and makes the lifetime explicit.
  TempPath model_path;
  auto model_file =
      create_temp_file(nullptr, "hyle-platform-", model_path.value,
                       sizeof(model_path.value));
  if (!model_file) return 7;
  std::array<std::byte, 4096> model_bytes{};
  model_bytes[17] = std::byte{0x6b};
  if (auto status = model_file->file.write_all(model_bytes); !status.ok())
    return 8;
  auto model = MappedFile::map(std::move(model_file->file));
  if (!model || model->bytes()[17] != std::byte{0x6b}) return 9;

  return 0;
}
