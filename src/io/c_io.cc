#include "hyle/c/io.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/io/proactor.h"
#include "hyle/io/shared_memory.h"
#include "c_io_internal.h"

namespace hyle::io::c_detail {

thread_local const char* g_last_error = "";

void clear_error() noexcept { g_last_error = ""; }
void set_error(const char* message) noexcept { g_last_error = message; }

bool empty_allocator(const hyle_allocator& a) noexcept {
  return a.state == nullptr && a.alloc == nullptr && a.free == nullptr &&
         a.realloc == nullptr;
}

bool valid_allocator(const hyle_allocator& a) noexcept {
  return a.alloc != nullptr && a.free != nullptr && a.realloc != nullptr;
}

bool normalize_allocator(hyle_allocator* a) noexcept {
  if (empty_allocator(*a)) *a = hyle_system_allocator();
  if (valid_allocator(*a)) return true;
  set_error("allocator must be zero-initialized or provide alloc/free/realloc");
  return false;
}

}  // namespace hyle::io::c_detail

namespace {

using hyle::io::c_detail::clear_error;
using hyle::io::c_detail::normalize_allocator;
using hyle::io::c_detail::set_error;

bool valid_token(std::uint64_t token) noexcept {
  if (token != HYLE_IO_WAKE_TOKEN) return true;
  set_error("UINT64_MAX is reserved for the proactor wake token");
  return false;
}

template <class Fn>
int submit(hyle_io_proactor* p, std::uint64_t token, Fn&& fn) noexcept;

}  // namespace

static_assert(std::is_standard_layout_v<hyle_io_proactor_options>);
static_assert(std::is_trivially_copyable_v<hyle_io_proactor_options>);
static_assert(offsetof(hyle_io_proactor_options, struct_size) == 0);
static_assert(std::is_standard_layout_v<hyle_io_completion>);
static_assert(std::is_trivially_copyable_v<hyle_io_completion>);
static_assert(sizeof(hyle_io_completion) == sizeof(hyle::io::Completion));
static_assert(alignof(hyle_io_completion) == alignof(hyle::io::Completion));
static_assert(offsetof(hyle_io_completion, user_data) ==
              offsetof(hyle::io::Completion, user_data));
static_assert(offsetof(hyle_io_completion, result) ==
              offsetof(hyle::io::Completion, result));
static_assert(offsetof(hyle_io_completion, flags) ==
              offsetof(hyle::io::Completion, flags));
static_assert(offsetof(hyle_io_completion, buffer_id) ==
              offsetof(hyle::io::Completion, buffer_id));
static_assert(static_cast<unsigned>(HYLE_IO_COMPLETION_MORE) ==
              static_cast<unsigned>(hyle::io::Completion::kMore));
static_assert(static_cast<unsigned>(HYLE_IO_COMPLETION_BUFFER) ==
              static_cast<unsigned>(hyle::io::Completion::kBuffer));
static_assert(static_cast<unsigned>(HYLE_IO_OPEN_READ) ==
              static_cast<unsigned>(hyle::io::Proactor::kRead));
static_assert(static_cast<unsigned>(HYLE_IO_OPEN_WRITE) ==
              static_cast<unsigned>(hyle::io::Proactor::kWrite));
static_assert(static_cast<unsigned>(HYLE_IO_OPEN_CREATE) ==
              static_cast<unsigned>(hyle::io::Proactor::kCreate));
static_assert(static_cast<unsigned>(HYLE_IO_OPEN_TRUNCATE) ==
              static_cast<unsigned>(hyle::io::Proactor::kTruncate));
static_assert(static_cast<unsigned>(HYLE_IO_OPEN_APPEND) ==
              static_cast<unsigned>(hyle::io::Proactor::kAppend));
static_assert(HYLE_IO_WAKE_TOKEN == hyle::io::Proactor::kWakeToken);
static_assert(HYLE_IO_DATAGRAM_SCRATCH_SIZE ==
              hyle::io::Proactor::kDatagramScratch);
static_assert(std::is_standard_layout_v<hyle_io_capabilities>);
static_assert(std::is_trivially_copyable_v<hyle_io_capabilities>);
static_assert(offsetof(hyle_io_capabilities, struct_size) == 0);
static_assert(std::is_standard_layout_v<hyle_io_uring_extension>);
static_assert(std::is_trivially_copyable_v<hyle_io_uring_extension>);
static_assert(sizeof(hyle_io_uring_extension) == sizeof(void*));
static_assert(std::is_same_v<hyle_io_file_stat,
                             hyle::io::Proactor::FileStat>);

struct hyle_io_proactor_o {
  hyle_allocator backing;
  hyle::io::Proactor value;
};

struct hyle_io_shared_memory_o {
  hyle_allocator backing;
  hyle::io::SharedMemory value;
};

namespace {

template <class Fn>
int submit(hyle_io_proactor* p, std::uint64_t token, Fn&& fn) noexcept {
  if (p == nullptr || !p->value.valid()) {
    set_error("proactor is null or invalid");
    return 0;
  }
  if (!valid_token(token)) return 0;
  if (!fn(p->value)) {
    set_error("operation could not be submitted");
    return 0;
  }
  clear_error();
  return 1;
}

hyle::io::IoUringExtension checked_uring(
    hyle_io_uring_extension extension) noexcept {
  auto* p = static_cast<hyle_io_proactor*>(extension.proactor);
  if (p == nullptr || !p->value.valid()) {
    set_error("io_uring extension is null or its proactor is invalid");
    return {};
  }
  auto uring = p->value.io_uring_extension();
  if (!uring) {
    set_error("proactor is not using the io_uring backend");
    return {};
  }
  return uring;
}

template <class Fn>
int submit_uring(hyle_io_uring_extension extension, std::uint64_t token,
                 Fn&& fn) noexcept {
  auto uring = checked_uring(extension);
  if (!uring || !valid_token(token)) return 0;
  if (!fn(uring)) {
    set_error("io_uring operation could not be submitted");
    return 0;
  }
  clear_error();
  return 1;
}

}  // namespace

extern "C" {

const char* hyle_io_last_error(void) {
  return hyle::io::c_detail::g_last_error;
}

hyle_io_proactor_options hyle_io_proactor_options_default(void) {
  hyle_io_proactor_options out = HYLE_IO_PROACTOR_OPTIONS_INIT;
  out.backing = hyle_system_allocator();
  return out;
}

hyle_io_proactor* hyle_io_proactor_create(
    const hyle_io_proactor_options* options) {
  hyle_io_proactor_options normalized = hyle_io_proactor_options_default();
  if (options != nullptr) {
    if (options->struct_size < offsetof(hyle_io_proactor_options, backing)) {
      set_error("proactor options struct_size is too small");
      return nullptr;
    }
    const std::size_t bytes =
        options->struct_size < sizeof(normalized) ? options->struct_size
                                                  : sizeof(normalized);
    std::memcpy(&normalized, options, bytes);
  }
  if (normalized.entries == 0) normalized.entries = 256;
  if (!normalize_allocator(&normalized.backing)) return nullptr;

  void* storage =
      normalized.backing.alloc(normalized.backing.state,
                               sizeof(hyle_io_proactor), alignof(hyle_io_proactor));
  if (storage == nullptr) {
    set_error("proactor handle allocation failed");
    return nullptr;
  }
  auto* out = new (storage) hyle_io_proactor{normalized.backing, {}};
  if (!out->value.init(normalized.entries, normalized.defer_taskrun != 0)) {
    out->~hyle_io_proactor();
    normalized.backing.free(normalized.backing.state, storage,
                            sizeof(hyle_io_proactor),
                            alignof(hyle_io_proactor));
    set_error("proactor backend initialization failed");
    return nullptr;
  }
  clear_error();
  return out;
}

void hyle_io_proactor_destroy(hyle_io_proactor* p) {
  if (p == nullptr) return;
  hyle_allocator backing = p->backing;
  p->~hyle_io_proactor();
  backing.free(backing.state, p, sizeof(hyle_io_proactor),
               alignof(hyle_io_proactor));
}

int hyle_io_proactor_valid(const hyle_io_proactor* p) {
  return p != nullptr && p->value.valid();
}

int hyle_io_proactor_in_flight(const hyle_io_proactor* p) {
  return p != nullptr && p->value.valid() && p->value.in_flight();
}

hyle_io_backend hyle_io_proactor_backend(const hyle_io_proactor* p) {
  return p != nullptr ? p->value.backend_kind() : HYLE_IO_BACKEND_NONE;
}

hyle_io_capabilities hyle_io_proactor_capabilities(
    const hyle_io_proactor* p) {
  if (p != nullptr) return p->value.capabilities();
  return hyle_io_capabilities{sizeof(hyle_io_capabilities),
                              HYLE_IO_BACKEND_NONE, 0};
}

const char* hyle_io_backend_name(hyle_io_backend backend) {
  switch (backend) {
    case HYLE_IO_BACKEND_IOCP: return "iocp";
    case HYLE_IO_BACKEND_IO_URING: return "io_uring";
    case HYLE_IO_BACKEND_EPOLL: return "epoll";
    case HYLE_IO_BACKEND_KQUEUE: return "kqueue";
    default: return "none";
  }
}

int hyle_io_has_async_file_data(const hyle_io_proactor* p) {
  return p != nullptr && p->value.has_async_file_data();
}

int hyle_io_has_async_file_metadata(const hyle_io_proactor* p) {
  return p != nullptr && p->value.has_async_file_metadata();
}

int hyle_io_submit_read(hyle_io_proactor* p, intptr_t fd, void* buffer,
                        uint32_t length, uint64_t offset, uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("read buffer is null");
    return 0;
  }
  if (offset > INT64_MAX) {
    set_error("read offset exceeds the portable signed offset range");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_read(fd, buffer, length, offset, token);
  });
}

int hyle_io_submit_write(hyle_io_proactor* p, intptr_t fd,
                         const void* buffer, uint32_t length, uint64_t offset,
                         uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("write buffer is null");
    return 0;
  }
  if (offset > INT64_MAX) {
    set_error("write offset exceeds the portable signed offset range");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_write(fd, buffer, length, offset, token);
  });
}

int hyle_io_submit_recv(hyle_io_proactor* p, intptr_t fd, void* buffer,
                        uint32_t length, uint32_t flags, uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("receive buffer is null");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_recv(fd, buffer, length, flags, token);
  });
}

int hyle_io_submit_send(hyle_io_proactor* p, intptr_t fd,
                        const void* buffer, uint32_t length, uint32_t flags,
                        uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("send buffer is null");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_send(fd, buffer, length, flags, token);
  });
}

int hyle_io_submit_accept(hyle_io_proactor* p, intptr_t fd, void* address,
                          void* address_length, uint64_t token) {
  if ((address == nullptr) != (address_length == nullptr)) {
    set_error("accept address and address_length must both be null or non-null");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_accept(fd, address, address_length, token);
  });
}

int hyle_io_submit_accept_multishot(hyle_io_proactor* p, intptr_t fd,
                                    uint64_t token) {
  return submit(p, token,
                [&](auto& x) { return x.submit_accept_multishot(fd, token); });
}

int hyle_io_submit_connect(hyle_io_proactor* p, intptr_t fd,
                           const void* address, uint32_t address_length,
                           uint64_t token) {
  if (address == nullptr || address_length == 0) {
    set_error("connect address is empty");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_connect(fd, address, address_length, token);
  });
}

int hyle_io_submit_recvfrom(hyle_io_proactor* p, intptr_t fd, void* buffer,
                            uint32_t length, void* address,
                            uint32_t address_capacity, void* scratch,
                            uint64_t token) {
  if ((length != 0 && buffer == nullptr) || address == nullptr ||
      address_capacity == 0 || scratch == nullptr) {
    set_error("recvfrom requires buffer, address and scratch storage");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_recvfrom(fd, buffer, length, address, address_capacity,
                             scratch, token);
  });
}

int hyle_io_submit_sendto(hyle_io_proactor* p, intptr_t fd,
                          const void* buffer, uint32_t length,
                          const void* address, uint32_t address_length,
                          void* scratch, uint64_t token) {
  if ((length != 0 && buffer == nullptr) || address == nullptr ||
      address_length == 0 || scratch == nullptr) {
    set_error("sendto requires buffer, address and scratch storage");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_sendto(fd, buffer, length, address, address_length, scratch,
                           token);
  });
}

int hyle_io_register_handle(hyle_io_proactor* p, intptr_t fd) {
  if (p == nullptr || !p->value.valid()) {
    set_error("proactor is null or invalid");
    return 0;
  }
  const bool ok = p->value.register_handle(fd);
  if (ok) clear_error();
  else set_error("handle registration failed");
  return ok;
}

int hyle_io_submit_open(hyle_io_proactor* p, const char* path, uint32_t flags,
                        uint32_t mode, uint64_t token) {
  if (path == nullptr) {
    set_error("open path is null");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_open(path, flags, mode, token);
  });
}

int hyle_io_submit_close(hyle_io_proactor* p, intptr_t fd, uint64_t token) {
  return submit(p, token,
                [&](auto& x) { return x.submit_close(fd, token); });
}

int hyle_io_submit_stat(hyle_io_proactor* p, intptr_t fd,
                        hyle_io_file_stat* output, uint64_t token) {
  if (output == nullptr) {
    set_error("stat output is null");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_statx(fd, output, token);
  });
}

int hyle_io_submit_sync(hyle_io_proactor* p, intptr_t fd, int data_only,
                        uint64_t token) {
  return submit(p, token, [&](auto& x) {
    return x.submit_fsync(fd, data_only != 0, token);
  });
}

int hyle_io_submit_allocate(hyle_io_proactor* p, intptr_t fd, uint32_t mode,
                            uint64_t offset, uint64_t length, uint64_t token) {
  if (length > INT64_MAX || offset > INT64_MAX - length) {
    set_error("file allocation range exceeds the portable signed offset range");
    return 0;
  }
  return submit(p, token, [&](auto& x) {
    return x.submit_fallocate(fd, mode, offset, length, token);
  });
}

int hyle_io_submit_cancel(hyle_io_proactor* p, uint64_t target,
                          uint64_t cancel_token) {
  if (!valid_token(target)) return 0;
  return submit(p, cancel_token, [&](auto& x) {
    return x.submit_cancel(target, cancel_token);
  });
}

hyle_status hyle_io_uring_extension_get(
    hyle_io_proactor* p, hyle_io_uring_extension* output) {
  if (output == nullptr) {
    return hyle_status_invalid_argument("io_uring extension output is null");
  }
  output->proactor = nullptr;
  if (p == nullptr || !p->value.valid()) {
    return hyle_status_invalid_argument("proactor is null or invalid");
  }
  if (!p->value.io_uring_extension()) {
    return hyle_status_unimplemented(
        "the selected I/O backend is not io_uring");
  }
  output->proactor = p;
  clear_error();
  return hyle_status_ok();
}

int hyle_io_uring_register_buffer_ring(hyle_io_uring_extension extension,
                                       int group_id, uint32_t entries,
                                       uint32_t buffer_size) {
  auto uring = checked_uring(extension);
  if (!uring) return 0;
  const bool ok =
      uring.register_buffer_ring(group_id, entries, buffer_size);
  if (ok) clear_error();
  else set_error("io_uring buffer-ring registration failed");
  return ok;
}

int hyle_io_uring_register_file_table(hyle_io_uring_extension extension,
                                      uint32_t entries) {
  auto uring = checked_uring(extension);
  if (!uring) return 0;
  const bool ok = uring.register_file_table(entries);
  if (ok) clear_error();
  else set_error("io_uring fixed-file table registration failed");
  return ok;
}

int hyle_io_uring_assign_fixed_file(hyle_io_uring_extension extension,
                                    intptr_t fd) {
  auto uring = checked_uring(extension);
  if (!uring) return -1;
  const int slot = uring.assign_fixed_file(fd);
  if (slot >= 0) clear_error();
  else set_error("io_uring fixed-file table is full or not registered");
  return slot;
}

int hyle_io_uring_release_fixed_file(hyle_io_uring_extension extension,
                                     int slot) {
  auto uring = checked_uring(extension);
  if (!uring) return 0;
  const bool ok = uring.release_fixed_file(slot);
  if (ok) clear_error();
  else set_error("io_uring fixed-file slot is invalid or release failed");
  return ok;
}

int hyle_io_uring_submit_recv_fixed(hyle_io_uring_extension extension,
                                    int slot, void* buffer, uint32_t length,
                                    uint32_t flags, uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("receive buffer is null");
    return 0;
  }
  return submit_uring(extension, token, [&](auto& x) {
    return x.submit_recv_fixed(slot, buffer, length, flags, token);
  });
}

int hyle_io_uring_submit_send_fixed(hyle_io_uring_extension extension,
                                    int slot, const void* buffer,
                                    uint32_t length, uint32_t flags,
                                    uint64_t token) {
  if (length != 0 && buffer == nullptr) {
    set_error("send buffer is null");
    return 0;
  }
  return submit_uring(extension, token, [&](auto& x) {
    return x.submit_send_fixed(slot, buffer, length, flags, token);
  });
}

int hyle_io_uring_submit_accept_direct(hyle_io_uring_extension extension,
                                       intptr_t fd, uint64_t token) {
  return submit_uring(extension, token,
                [&](auto& x) { return x.submit_accept_direct(fd, token); });
}

int hyle_io_uring_submit_close_direct(hyle_io_uring_extension extension,
                                      int slot, uint64_t token) {
  return submit_uring(extension, token,
                [&](auto& x) { return x.submit_close_direct(slot, token); });
}

int hyle_io_uring_submit_recv_multishot(hyle_io_uring_extension extension,
                                        intptr_t fd, int group_id,
                                        uint64_t token) {
  return submit_uring(extension, token, [&](auto& x) {
    return x.submit_recv_multishot(fd, group_id, token);
  });
}

void* hyle_io_uring_buffer_address(hyle_io_uring_extension extension,
                                   int group_id, uint32_t buffer_id) {
  auto uring = checked_uring(extension);
  if (!uring) return nullptr;
  void* address = uring.buffer_address(group_id, buffer_id);
  if (address != nullptr) clear_error();
  else set_error("buffer ring or buffer id is invalid");
  return address;
}

int hyle_io_uring_buffer_return(hyle_io_uring_extension extension,
                                int group_id, uint32_t buffer_id) {
  auto uring = checked_uring(extension);
  if (!uring) return 0;
  const bool ok = uring.return_buffer(group_id, buffer_id);
  if (ok) clear_error();
  else set_error("buffer ring or buffer id is invalid");
  return ok;
}

size_t hyle_io_reap(hyle_io_proactor* p, hyle_io_completion* output,
                    size_t capacity, uint64_t timeout_ns) {
  if (p == nullptr || !p->value.valid() ||
      (capacity != 0 && output == nullptr)) {
    set_error("reap requires a valid proactor and output for non-zero capacity");
    return 0;
  }
  hyle::io::Completion native[128];
  const size_t want = capacity < 128 ? capacity : 128;
  const size_t n = p->value.reap(native, want, timeout_ns);
  for (size_t i = 0; i < n; ++i) {
    output[i].user_data = native[i].user_data;
    output[i].result = native[i].result;
    output[i].flags = native[i].flags;
    output[i].buffer_id = native[i].buffer_id;
  }
  clear_error();
  return n;
}

int hyle_io_post(hyle_io_proactor* p, uint64_t token, int64_t result) {
  if (result < 0 || result > UINT32_MAX) {
    set_error("post result must fit the portable non-negative 32-bit range");
    return 0;
  }
  return submit(p, token, [&](auto& x) { return x.post(token, result); });
}

int hyle_io_wake(hyle_io_proactor* p) {
  if (p == nullptr || !p->value.valid()) {
    set_error("proactor is null or invalid");
    return 0;
  }
  const bool ok = p->value.wake();
  if (ok) clear_error();
  else set_error("proactor wake failed");
  return ok;
}

hyle_io_shared_memory* hyle_io_shared_memory_create(const char* name,
                                                     size_t size) {
  return hyle_io_shared_memory_create_with_allocator(
      name, size, hyle_system_allocator());
}

hyle_io_shared_memory* hyle_io_shared_memory_open(const char* name) {
  return hyle_io_shared_memory_open_with_allocator(name,
                                                    hyle_system_allocator());
}

hyle_io_shared_memory* hyle_io_shared_memory_create_with_allocator(
    const char* name, size_t size, hyle_allocator backing) {
  if (name == nullptr || *name == '\0' || size == 0) {
    set_error("shared memory requires a non-empty name and non-zero size");
    return nullptr;
  }
  if (!normalize_allocator(&backing)) return nullptr;
  auto value = hyle::io::SharedMemory::create(name, size);
  if (!value.valid()) {
    set_error("shared memory create failed (name exists or OS rejected it)");
    return nullptr;
  }
  void* storage = backing.alloc(backing.state, sizeof(hyle_io_shared_memory),
                                alignof(hyle_io_shared_memory));
  if (storage == nullptr) {
    set_error("shared memory wrapper allocation failed");
    return nullptr;
  }
  auto* out =
      new (storage) hyle_io_shared_memory{backing, std::move(value)};
  clear_error();
  return out;
}

hyle_io_shared_memory* hyle_io_shared_memory_open_with_allocator(
    const char* name, hyle_allocator backing) {
  if (name == nullptr || *name == '\0') {
    set_error("shared memory requires a non-empty name");
    return nullptr;
  }
  if (!normalize_allocator(&backing)) return nullptr;
  auto value = hyle::io::SharedMemory::open(name);
  if (!value.valid()) {
    set_error("shared memory open failed");
    return nullptr;
  }
  void* storage = backing.alloc(backing.state, sizeof(hyle_io_shared_memory),
                                alignof(hyle_io_shared_memory));
  if (storage == nullptr) {
    set_error("shared memory wrapper allocation failed");
    return nullptr;
  }
  auto* out =
      new (storage) hyle_io_shared_memory{backing, std::move(value)};
  clear_error();
  return out;
}

void hyle_io_shared_memory_destroy(hyle_io_shared_memory* memory) {
  if (memory == nullptr) return;
  hyle_allocator backing = memory->backing;
  memory->~hyle_io_shared_memory();
  backing.free(backing.state, memory, sizeof(hyle_io_shared_memory),
               alignof(hyle_io_shared_memory));
}

void* hyle_io_shared_memory_data(hyle_io_shared_memory* memory) {
  return memory != nullptr ? memory->value.data() : nullptr;
}

size_t hyle_io_shared_memory_size(const hyle_io_shared_memory* memory) {
  return memory != nullptr ? memory->value.size() : 0;
}

int hyle_io_shared_memory_flush(hyle_io_shared_memory* memory) {
  if (memory == nullptr || !memory->value.flush()) {
    set_error("shared memory flush failed");
    return 0;
  }
  clear_error();
  return 1;
}

hyle_status hyle_io_shared_memory_unlink(const char* name) {
  hyle::Status status = hyle::io::SharedMemory::unlink(name);
  if (status.ok())
    clear_error();
  else
    set_error("shared memory unlink failed");
  return hyle_status{status.release_rep()};
}

}  // extern "C"
