// C-ABI bridge for platform (include/hyle/c/platform.h).

#include "hyle/c/platform.h"

#include <atomic>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/dynlib.h"
#include "hyle/platform/fs.h"
#include "hyle/platform/mapped_file.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/platform/wait.h"
#include "../error/status_access.h"

namespace {

static_assert(std::is_standard_layout_v<hyle_thread_opts>);
static_assert(std::is_trivially_copyable_v<hyle_thread_opts>);
static_assert(offsetof(hyle_thread_opts, struct_size) == 0);
static_assert(std::is_standard_layout_v<hyle_file_info>);
static_assert(std::is_trivially_copyable_v<hyle_file_info>);
static_assert(std::is_standard_layout_v<hyle_cpu_features>);
static_assert(std::is_trivially_copyable_v<hyle_cpu_features>);
static_assert(std::is_standard_layout_v<hyle_cache_info>);
static_assert(std::is_trivially_copyable_v<hyle_cache_info>);
static_assert(std::is_standard_layout_v<hyle_cpu_topology>);
static_assert(std::is_trivially_copyable_v<hyle_cpu_topology>);
static_assert(HYLE_THREAD_PRIORITY_LOWEST ==
              static_cast<int>(hyle::platform::ThreadPriority::kLowest));
static_assert(HYLE_THREAD_PRIORITY_HIGHEST ==
              static_cast<int>(hyle::platform::ThreadPriority::kHighest));
static_assert(HYLE_PROTECT_NONE ==
              static_cast<int>(hyle::platform::Protect::kNone));
static_assert(HYLE_PROTECT_READ_WRITE ==
              static_cast<int>(hyle::platform::Protect::kReadWrite));
static_assert(HYLE_NUMA_PREFERRED ==
              static_cast<int>(hyle::platform::NumaPolicy::kPreferred));
static_assert(HYLE_NUMA_INTERLEAVE ==
              static_cast<int>(hyle::platform::NumaPolicy::kInterleave));
static_assert(HYLE_NUMA_PLACEMENT_APPLIED ==
              static_cast<int>(hyle::platform::NumaPlacement::kApplied));
static_assert(HYLE_NUMA_PLACEMENT_UNAVAILABLE ==
              static_cast<int>(hyle::platform::NumaPlacement::kUnavailable));
static_assert(HYLE_MAP_READ_ONLY ==
              static_cast<int>(hyle::platform::MappedFile::Mode::kReadOnly));
static_assert(HYLE_MAP_COPY_ON_WRITE ==
              static_cast<int>(
                  hyle::platform::MappedFile::Mode::kCopyOnWrite));
static_assert(HYLE_ISA_SCALAR ==
              static_cast<int>(hyle::platform::Isa::kScalar));
static_assert(HYLE_ISA_RVV == static_cast<int>(hyle::platform::Isa::kRvv));

thread_local char g_platform_error[256] = {0};

void set_platform_error(const char* message) noexcept {
  if (message == nullptr) {
    g_platform_error[0] = '\0';
    return;
  }
  std::size_t n = std::strlen(message);
  if (n >= sizeof(g_platform_error)) n = sizeof(g_platform_error) - 1;
  std::memcpy(g_platform_error, message, n);
  g_platform_error[n] = '\0';
}

int fs_result(bool ok, const char* failure) noexcept {
  set_platform_error(ok ? nullptr : failure);
  return ok ? 1 : 0;
}

hyle_status steal_status(hyle::Status&& s) noexcept {
  hyle_status out;
  out.rep = hyle::detail::StatusAccess::release(s);
  return out;
}

hyle_numa_placement to_c_numa_placement(
    hyle::platform::NumaPlacement placement) noexcept {
  return static_cast<hyle_numa_placement>(placement);
}

// C volatile uint32_t* is the public wait word; Parker uses atomic<uint32_t>.
// Same object representation for aligned uint32 storage.
const std::atomic<std::uint32_t>* as_atomic(const volatile uint32_t* addr) {
  return reinterpret_cast<const std::atomic<std::uint32_t>*>(
      const_cast<const uint32_t*>(addr));
}
std::atomic<std::uint32_t>* as_atomic_mut(volatile uint32_t* addr) {
  return reinterpret_cast<std::atomic<std::uint32_t>*>(
      const_cast<uint32_t*>(addr));
}

hyle::platform::Protect to_cpp_protect(hyle_protect p) noexcept {
  using hyle::platform::Protect;
  if (p == HYLE_PROTECT_READ) return Protect::kRead;
  if (p == HYLE_PROTECT_READ_WRITE) return Protect::kReadWrite;
  return Protect::kNone;
}

hyle::platform::NumaPolicy to_cpp_numa_policy(hyle_numa_policy p) noexcept {
  using hyle::platform::NumaPolicy;
  if (p == HYLE_NUMA_BIND) return NumaPolicy::kBind;
  if (p == HYLE_NUMA_INTERLEAVE) return NumaPolicy::kInterleave;
  return NumaPolicy::kPreferred;
}

hyle::platform::ThreadOpts to_cpp_opts(const hyle_thread_opts* opts) {
  hyle::platform::ThreadOpts o;
  if (opts == nullptr) return o;
  if (opts->name != nullptr && opts->name[0] != '\0') o.name = opts->name;
  if (opts->has_stack_size) o.stack_size = opts->stack_size;
  if (opts->has_affinity_cpu) o.affinity_cpu = opts->affinity_cpu;
  if (opts->has_priority) {
    o.priority = static_cast<hyle::platform::ThreadPriority>(
        static_cast<std::int8_t>(opts->priority));
  }
  return o;
}

bool valid_priority(int p) noexcept {
  return p >= HYLE_THREAD_PRIORITY_LOWEST &&
         p <= HYLE_THREAD_PRIORITY_HIGHEST;
}

bool valid_protect(hyle_protect p) noexcept {
  return p == HYLE_PROTECT_NONE || p == HYLE_PROTECT_READ ||
         p == HYLE_PROTECT_READ_WRITE;
}

bool valid_numa_policy(hyle_numa_policy p) noexcept {
  return p == HYLE_NUMA_PREFERRED || p == HYLE_NUMA_BIND ||
         p == HYLE_NUMA_INTERLEAVE;
}

struct DirVisitPack {
  hyle_fs_visit_fn fn;
  void* user;
};

bool dir_visit_trampoline(const char* name, bool is_dir, void* ctx) {
  auto* pack = static_cast<DirVisitPack*>(ctx);
  return pack->fn(name, is_dir ? 1 : 0, pack->user) != 0;
}

}  // namespace

struct hyle_parker_o {
  hyle::platform::Parker p;
};
struct hyle_thread_o {
  hyle::platform::Thread t;
};
struct hyle_dynlib_o {
  hyle::platform::DynLib lib;
};
struct hyle_vm_region_o {
  hyle::platform::VmRegion region;
};
struct hyle_mapped_file_o {
  hyle::platform::MappedFile mf;
};

static hyle_status wrap_vm_region(
    hyle_vm_region_o** out,
    hyle::Result<hyle::platform::VmRegion>&& region) noexcept {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  if (!region) return steal_status(std::move(region).error());
  auto* wrapper = new (std::nothrow) hyle_vm_region_o{};
  if (wrapper == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("VM region wrapper allocation failed"));
  }
  wrapper->region = *std::move(region);
  *out = wrapper;
  return steal_status(hyle::Status::Ok());
}

extern "C" {

const char* hyle_platform_last_error(void) { return g_platform_error; }

void hyle_wait_on_address(volatile uint32_t* addr, uint32_t expected) {
  if (addr == nullptr) return;
  hyle::platform::detail::wait_on_address(as_atomic(addr), expected);
}

int hyle_wait_on_address_for(volatile uint32_t* addr, uint32_t expected,
                             uint64_t timeout_ns) {
  if (addr == nullptr) return 1;
  return hyle::platform::detail::wait_on_address_for(as_atomic(addr), expected,
                                                     timeout_ns)
             ? 1
             : 0;
}

void hyle_wake_one(volatile uint32_t* addr) {
  if (addr == nullptr) return;
  hyle::platform::detail::wake_one(as_atomic_mut(addr));
}

void hyle_wake_all(volatile uint32_t* addr) {
  if (addr == nullptr) return;
  hyle::platform::detail::wake_all(as_atomic_mut(addr));
}

hyle_status hyle_parker_create(hyle_parker_o** out) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_parker_create: out is NULL"));
  }
  *out = nullptr;
  auto* p = new (std::nothrow) hyle_parker_o{};
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_parker_create: wrapper allocation failed"));
  }
  *out = p;
  return steal_status(hyle::Status::Ok());
}
void hyle_parker_destroy(hyle_parker_o* p) { delete p; }
void hyle_parker_park(hyle_parker_o* p) {
  if (p) p->p.park();
}
void hyle_parker_unpark(hyle_parker_o* p) {
  if (p) p->p.unpark();
}
int hyle_parker_park_timeout(hyle_parker_o* p, uint64_t timeout_ns) {
  return p && p->p.park_timeout(timeout_ns) ? 1 : 0;
}

hyle_status hyle_thread_spawn(hyle_thread_o** out, hyle_thread_entry_fn entry,
                              void* ctx, const hyle_thread_opts* opts) {
  if (out == nullptr || entry == nullptr) {
    return steal_status(hyle::Status::InvalidArgument("thread spawn args"));
  }
  *out = nullptr;
  if (opts != nullptr && opts->struct_size < sizeof(hyle_thread_opts)) {
    return steal_status(
        hyle::Status::InvalidArgument("thread opts struct is too small"));
  }
  if (opts != nullptr && opts->has_priority &&
      !valid_priority(opts->priority)) {
    return steal_status(
        hyle::Status::InvalidArgument("thread priority out of range"));
  }
  auto* o = new (std::nothrow) hyle_thread_o{};
  if (o == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted("thread object OOM"));
  }
  auto r = hyle::platform::Thread::spawn(entry, ctx, to_cpp_opts(opts));
  if (!r.has_value()) {
    delete o;
    return steal_status(
        std::move(r).error().WithContext("hyle_thread_spawn"));
  }
  o->t = std::move(*r);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_thread_spawn_opts(hyle_thread_o** out, hyle_thread_entry_fn entry,
                                   void* ctx, const char* name, size_t stack_size,
                                   int affinity_cpu, int priority) {
  // Translate sentinels into the has_* struct so an FFI caller never touches
  // the flags. name==NULL stays NULL (spawn treats it as "no name").
  hyle_thread_opts o = HYLE_THREAD_OPTS_INIT;
  o.name = name;
  if (stack_size != 0) {
    o.has_stack_size = 1;
    o.stack_size = stack_size;
  }
  if (affinity_cpu >= 0) {
    o.has_affinity_cpu = 1;
    o.affinity_cpu = static_cast<unsigned>(affinity_cpu);
  }
  o.has_priority = 1;  // always set; HYLE_THREAD_PRIORITY_NORMAL==0 = default
  o.priority = priority;
  return hyle_thread_spawn(out, entry, ctx, &o);
}

void hyle_thread_join(hyle_thread_o* t) {
  if (t == nullptr) return;
  if (t->t.joinable()) t->t.join();
  delete t;
}

int hyle_thread_joinable(const hyle_thread_o* t) {
  return t != nullptr && t->t.joinable() ? 1 : 0;
}

unsigned hyle_hardware_concurrency(void) {
  return hyle::platform::hardware_concurrency();
}
uint64_t hyle_this_thread_id(void) {
  return hyle::platform::this_thread::get_id();
}
void hyle_this_thread_yield(void) { hyle::platform::this_thread::yield(); }
void hyle_this_thread_set_name(const char* name) {
  hyle::platform::this_thread::set_name(name != nullptr ? name : "");
}

hyle_status hyle_dynlib_open(hyle_dynlib_o** out, const char* path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_open: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::open(path);
  if (!lib) return steal_status(std::move(lib).error());
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_open: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_dynlib_get(hyle_dynlib_o** out, const char* path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_get: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::get(path);
  if (!lib) return steal_status(std::move(lib).error());
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_get: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_dynlib_current(hyle_dynlib_o** out) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_current: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::current();
  if (!lib) return steal_status(std::move(lib).error());
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_current: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

void hyle_dynlib_close(hyle_dynlib_o* lib) { delete lib; }
int hyle_dynlib_valid(const hyle_dynlib_o* lib) {
  return lib != nullptr && lib->lib.valid() ? 1 : 0;
}
hyle_status hyle_dynlib_symbol(void** out, hyle_dynlib_o* lib,
                               const char* name) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_symbol: out is NULL"));
  }
  *out = nullptr;
  if (lib == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_dynlib_symbol: library is NULL"));
  }
  auto symbol = lib->lib.symbol(name);
  if (!symbol) return steal_status(std::move(symbol).error());
  *out = *symbol;
  return steal_status(hyle::Status());
}
const char* hyle_dynlib_extension(void) {
  return hyle::platform::dynlib_extension();
}

hyle_status hyle_dynlib_open_symbol(hyle_dynlib_o** out_lib, void** out_sym,
                                    const char* path, const char* name) {
  if (out_lib) *out_lib = nullptr;
  if (out_sym) *out_sym = nullptr;
  if (!path || !name || !out_lib || !out_sym) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_open_symbol: invalid argument"));
  }
  hyle_dynlib_o* lib = nullptr;
  hyle_status status = hyle_dynlib_open(&lib, path);
  if (!hyle_status_is_ok(status)) return status;
  void* sym = nullptr;
  status = hyle_dynlib_symbol(&sym, lib, name);
  if (!hyle_status_is_ok(status)) {
    hyle_dynlib_close(lib);
    return status;
  }
  *out_lib = lib;
  *out_sym = sym;
  return status;
}

size_t hyle_page_size(void) { return hyle::platform::page_size(); }
hyle_status hyle_vm_reserve(void** out, size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_reserve: out is NULL"));
  }
  *out = nullptr;
  auto result = hyle::platform::vm_reserve(size);
  if (!result) return steal_status(std::move(result).error());
  *out = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_vm_commit(void* ptr, size_t size) {
  return steal_status(hyle::platform::vm_commit(ptr, size));
}
hyle_status hyle_vm_decommit(void* ptr, size_t size) {
  return steal_status(hyle::platform::vm_decommit(ptr, size));
}
hyle_status hyle_vm_release(void* base, size_t size) {
  return steal_status(hyle::platform::vm_release(base, size));
}
hyle_status hyle_vm_protect(void* ptr, size_t size, hyle_protect prot) {
  if (!valid_protect(prot)) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_protect: invalid protection"));
  }
  return steal_status(
      hyle::platform::vm_protect(ptr, size, to_cpp_protect(prot)));
}

hyle_status hyle_vm_alloc_rw(void** out, size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_alloc_rw: out is NULL"));
  }
  *out = nullptr;
  auto allocated = hyle::platform::vm_allocate_rw(size);
  if (!allocated) return steal_status(std::move(allocated).error());
  *out = *allocated;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_free_rw(void* p, size_t size) {
  // Same page rounding as alloc_rw's reserve, so the released span matches.
  const size_t aligned = hyle_page_align_up(size);
  if (p != nullptr && aligned == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_free_rw: zero or overflowing size"));
  }
  return steal_status(hyle::platform::vm_release(p, aligned));
}

hyle_status hyle_vm_region_reserve(hyle_vm_region_o** out, size_t size) {
  return wrap_vm_region(out, hyle::platform::VmRegion::reserve(size));
}

hyle_status hyle_vm_region_reserve_guarded(hyle_vm_region_o** out,
                                           size_t usable) {
  return wrap_vm_region(out,
                        hyle::platform::VmRegion::reserve_guarded(usable));
}

hyle_status hyle_vm_region_reserve_numa(hyle_vm_region_o** out, size_t size,
                                        int node) {
  return wrap_vm_region(out,
                        hyle::platform::VmRegion::reserve_numa(size, node));
}

hyle_status hyle_vm_region_adopt(hyle_vm_region_o** out, void* base,
                                 size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_region_adopt: out is NULL"));
  }
  *out = nullptr;
  // Allocate the wrapper before taking ownership. On wrapper OOM the caller
  // still owns `base`; an adopt API must transfer ownership only on success.
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) {
    return steal_status(
        hyle::Status::ResourceExhausted("VM region wrapper allocation failed"));
  }
  auto adopted = hyle::platform::VmRegion::adopt(base, size);
  if (!adopted) {
    hyle::Status status = std::move(adopted).error();
    delete o;
    return steal_status(std::move(status));
  }
  o->region = *std::move(adopted);
  *out = o;
  return steal_status(hyle::Status::Ok());
}

void hyle_vm_region_destroy(hyle_vm_region_o* r) { delete r; }
int hyle_vm_region_valid(const hyle_vm_region_o* r) {
  return r && r->region.valid() ? 1 : 0;
}
void* hyle_vm_region_data(hyle_vm_region_o* r) {
  return r ? r->region.data() : nullptr;
}
size_t hyle_vm_region_size(const hyle_vm_region_o* r) {
  return r ? r->region.size() : 0;
}
void* hyle_vm_region_usable(hyle_vm_region_o* r) {
  return r ? r->region.usable() : nullptr;
}
size_t hyle_vm_region_usable_size(const hyle_vm_region_o* r) {
  return r ? r->region.usable_size() : 0;
}
int hyle_vm_region_has_guard(const hyle_vm_region_o* r) {
  return r && r->region.has_guard() ? 1 : 0;
}
hyle_numa_placement hyle_vm_region_numa_placement(
    const hyle_vm_region_o* r) {
  return r ? to_c_numa_placement(r->region.numa_placement())
           : HYLE_NUMA_PLACEMENT_NOT_NEEDED;
}
hyle_status hyle_vm_region_commit(hyle_vm_region_o* r, size_t off,
                                  size_t len) {
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_commit: region is NULL"));
  }
  return steal_status(r->region.commit(off, len));
}
hyle_status hyle_vm_region_commit_usable(hyle_vm_region_o* r) {
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_commit_usable: region is NULL"));
  }
  return steal_status(r->region.commit_usable());
}
hyle_status hyle_vm_region_decommit(hyle_vm_region_o* r, size_t off,
                                    size_t len) {
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_decommit: region is NULL"));
  }
  return steal_status(r->region.decommit(off, len));
}
hyle_status hyle_vm_region_protect(hyle_vm_region_o* r, size_t off,
                                   size_t len, hyle_protect prot) {
  if (!valid_protect(prot)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_protect: invalid protection"));
  }
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_protect: region is NULL"));
  }
  return steal_status(r->region.protect(off, len, to_cpp_protect(prot)));
}

hyle_file_info hyle_fs_stat(const char* path) {
  hyle_file_info out{};
  if (!path) {
    set_platform_error("hyle_fs_stat: path is NULL");
    return out;
  }
  set_platform_error(nullptr);
  auto i = hyle::platform::stat(path);
  out.exists = i.exists ? 1 : 0;
  out.is_directory = i.is_directory ? 1 : 0;
  out.size = i.size;
  out.modified_time = i.modified_time;
  out.is_symlink = i.is_symlink ? 1 : 0;
  return out;
}
int hyle_fs_exists(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_exists: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::exists(path) ? 1 : 0;
}
int hyle_fs_is_directory(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_is_directory: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::is_directory(path) ? 1 : 0;
}
uint64_t hyle_fs_file_size(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_file_size: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::file_size(path);
}
int hyle_fs_make_directory(const char* path) {
  return fs_result(path && hyle::platform::make_directory(path),
                   "hyle_fs_make_directory failed");
}
int hyle_fs_make_directory_recursive(const char* path) {
  return fs_result(path && hyle::platform::make_directory_recursive(path),
                   "hyle_fs_make_directory_recursive failed");
}
int hyle_fs_remove_directory(const char* path) {
  return fs_result(path && hyle::platform::remove_directory(path),
                   "hyle_fs_remove_directory failed");
}
int hyle_fs_remove_directory_recursive(const char* path) {
  return fs_result(path && hyle::platform::remove_directory_recursive(path),
                   "hyle_fs_remove_directory_recursive failed");
}
int hyle_fs_remove_file(const char* path) {
  return fs_result(path && hyle::platform::remove_file(path),
                   "hyle_fs_remove_file failed");
}
int hyle_fs_rename(const char* from, const char* to) {
  return fs_result(from && to && hyle::platform::rename_path(from, to),
                   "hyle_fs_rename failed");
}
int hyle_fs_copy_file(const char* from, const char* to) {
  return fs_result(from && to && hyle::platform::copy_file(from, to),
                   "hyle_fs_copy_file failed");
}
int hyle_fs_touch(const char* path) {
  return fs_result(path && hyle::platform::touch(path), "hyle_fs_touch failed");
}
size_t hyle_fs_current_directory(char* dst, size_t cap) {
  const size_t n = hyle::platform::current_directory(dst, cap);
  set_platform_error(n == 0 ? "hyle_fs_current_directory failed" : nullptr);
  return n;
}
int hyle_fs_change_directory(const char* path) {
  return fs_result(path && hyle::platform::change_directory(path),
                   "hyle_fs_change_directory failed");
}
size_t hyle_fs_temp_directory(char* dst, size_t cap) {
  const size_t n = hyle::platform::temp_directory(dst, cap);
  set_platform_error(n == 0 ? "hyle_fs_temp_directory failed" : nullptr);
  return n;
}
size_t hyle_fs_home_directory(char* dst, size_t cap) {
  const size_t n = hyle::platform::home_directory(dst, cap);
  set_platform_error(n == 0 ? "hyle_fs_home_directory failed" : nullptr);
  return n;
}
size_t hyle_env_get(const char* name, char* dst, size_t cap) {
  if (!name) {
    set_platform_error("hyle_env_get: name is NULL");
    return HYLE_ENV_NOT_SET;
  }
  set_platform_error(nullptr);
  return hyle::platform::get_env_var(name, dst, cap);
}

int hyle_fs_directory_iterate(const char* path, hyle_fs_visit_fn visit,
                              void* ctx) {
  if (!path || !visit) {
    set_platform_error("hyle_fs_directory_iterate: invalid argument");
    return 0;
  }
  DirVisitPack pack{visit, ctx};
  return fs_result(
      hyle::platform::directory_iterate(path, &dir_visit_trampoline, &pack),
      "hyle_fs_directory_iterate failed");
}

hyle_cpu_features hyle_cpu_features_get(void) {
  hyle_cpu_features out{};
  const auto& f = hyle::platform::cpu_features();
#define HYLE_B(x) out.x = f.x ? 1 : 0
  HYLE_B(sse2);
  HYLE_B(ssse3);
  HYLE_B(sse41);
  HYLE_B(sse42);
  HYLE_B(avx);
  HYLE_B(avx2);
  HYLE_B(fma);
  HYLE_B(f16c);
  HYLE_B(lzcnt);
  HYLE_B(bmi1);
  HYLE_B(bmi2);
  HYLE_B(avx512f);
  HYLE_B(avx512bw);
  HYLE_B(avx512vl);
  HYLE_B(avx512dq);
  HYLE_B(avx512cd);
  HYLE_B(avx512vnni);
  HYLE_B(avx_vnni);
  HYLE_B(avx512bf16);
  HYLE_B(avx512fp16);
  HYLE_B(avx10_2);
  HYLE_B(neon);
  HYLE_B(sve);
  HYLE_B(sve2);
  HYLE_B(sve_aes);
  HYLE_B(sve_pmull);
  HYLE_B(sve_i8mm);
  HYLE_B(sve_f32mm);
  HYLE_B(sve_f64mm);
  HYLE_B(sve_bf16);
  HYLE_B(arm_i8mm);
  HYLE_B(arm_bf16);
  HYLE_B(rvv);
  HYLE_B(rvv_zvbb);
  HYLE_B(rvv_zvbc);
  HYLE_B(rvv_zvkned);
  HYLE_B(rvv_zvfh);
  HYLE_B(rvv_zvfhmin);
#undef HYLE_B
  out.logical_cpus = hyle::platform::hardware_concurrency();
  out.best_isa = static_cast<uint32_t>(hyle::platform::best_isa());
  return out;
}

int hyle_cpu_supports_isa(uint32_t isa) {
  return hyle::platform::supports(static_cast<hyle::platform::Isa>(isa)) ? 1
                                                                         : 0;
}

const char* hyle_cpu_isa_name(uint32_t isa) {
  return hyle::platform::isa_name(static_cast<hyle::platform::Isa>(isa));
}

unsigned hyle_cpu_cache_line_size(void) {
  return hyle::platform::cache_line_size();
}

hyle_cache_info hyle_cpu_cache_info(void) {
  const auto& in = hyle::platform::cache_info();
  return {in.l1d, in.l1i, in.l2, in.l3};
}

hyle_cpu_topology hyle_cpu_topology_get(void) {
  const auto& in = hyle::platform::cpu_topology();
  return {in.logical_cpus, in.performance_cpus, in.efficiency_cpus,
          in.hybrid ? 1 : 0};
}

/* ---- v3: hugepage/THP + NUMA ------------------------------------------------ */

size_t hyle_large_page_size(void) { return hyle::platform::large_page_size(); }

hyle_status hyle_vm_advise_hugepage(void* ptr, size_t size) {
  return steal_status(hyle::platform::vm_advise_hugepage(ptr, size));
}

hyle_status hyle_vm_advise_no_hugepage(void* ptr, size_t size) {
  return steal_status(hyle::platform::vm_advise_no_hugepage(ptr, size));
}

unsigned hyle_numa_node_count(void) { return hyle::platform::numa_node_count(); }

unsigned hyle_numa_current_node(void) {
  return hyle::platform::numa_current_node();
}

unsigned hyle_numa_node_of_cpu(unsigned cpu) {
  return hyle::platform::numa_node_of_cpu(cpu);
}

hyle_status hyle_vm_bind_numa(hyle_numa_placement* out_placement, void* ptr,
                              size_t size, int node,
                              hyle_numa_policy policy) {
  if (!valid_numa_policy(policy)) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_bind_numa: invalid policy"));
  }
  auto result = hyle::platform::vm_bind_numa(
      ptr, size, node, to_cpp_numa_policy(policy));
  if (!result) return steal_status(std::move(result).error());
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(*result);
  }
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_bind_numa_interleave(
    hyle_numa_placement* out_placement, void* ptr, size_t size) {
  auto result = hyle::platform::vm_bind_numa_interleave(ptr, size);
  if (!result) return steal_status(std::move(result).error());
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(*result);
  }
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_reserve_numa(void** out,
                                 hyle_numa_placement* out_placement,
                                 size_t size, int node) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_reserve_numa: out is NULL"));
  }
  *out = nullptr;
  auto result = hyle::platform::vm_reserve_numa(size, node);
  if (!result) return steal_status(std::move(result).error());
  *out = result->data;
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(result->placement);
  }
  return steal_status(hyle::Status::Ok());
}

/* ---- v4: fs perms/symlink/tempfile/env-write + mapped file ----------------- */

uint32_t hyle_fs_get_permissions(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_get_permissions: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::get_permissions(path);
}
int hyle_fs_set_permissions(const char* path, uint32_t mode) {
  return fs_result(path && hyle::platform::set_permissions(path, mode),
                   "hyle_fs_set_permissions failed");
}
int hyle_fs_is_readonly(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_is_readonly: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::is_readonly(path) ? 1 : 0;
}
int hyle_fs_set_readonly(const char* path, int readonly) {
  return fs_result(path && hyle::platform::set_readonly(path, readonly != 0),
                   "hyle_fs_set_readonly failed");
}
int hyle_fs_create_symlink(const char* target, const char* link) {
  return fs_result(
      target && link && hyle::platform::create_symlink(target, link),
      "hyle_fs_create_symlink failed");
}
int hyle_fs_create_hard_link(const char* target, const char* link) {
  return fs_result(
      target && link && hyle::platform::create_hard_link(target, link),
      "hyle_fs_create_hard_link failed");
}
int hyle_fs_is_symlink(const char* path) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_is_symlink: path is NULL");
    return 0;
  }
  set_platform_error(nullptr);
  return hyle::platform::is_symlink(path) ? 1 : 0;
}
size_t hyle_fs_read_symlink(const char* path, char* dst, size_t cap) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_read_symlink: path is NULL");
    return 0;
  }
  const size_t n = hyle::platform::read_symlink(path, dst, cap);
  set_platform_error(n == 0 ? "hyle_fs_read_symlink failed" : nullptr);
  return n;
}
hyle_status hyle_fs_create_temp_file(hyle_file_handle* out, const char* dir,
                                     const char* prefix, char* dst, size_t cap) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_create_temp_file: out is NULL"));
  }
  *out = hyle::platform::FileHandle::kInvalid;
  auto result = hyle::platform::create_temp_file(dir, prefix, dst, cap);
  if (!result) return steal_status(std::move(result).error());
  *out = result->release();
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_close(hyle_file_handle file) {
  hyle::platform::FileHandle owned{file};
  return steal_status(owned.close());
}
hyle_status hyle_fs_open(hyle_file_handle* out, const char* path,
                         unsigned flags) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_open: out is NULL"));
  }
  *out = hyle::platform::FileHandle::kInvalid;
  auto result = hyle::platform::open_file(path, flags);
  if (!result) return steal_status(std::move(result).error());
  *out = result->release();
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_read(size_t* out_moved, hyle_file_handle file, void* buf,
                         size_t n) {
  if (out_moved == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_read: out_moved is NULL"));
  }
  *out_moved = 0;
  hyle::platform::FileHandle tmp{file};  // borrow; release before returning
  auto result = tmp.read(buf, n);
  tmp.release();
  if (!result) return steal_status(std::move(result).error());
  *out_moved = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_write(size_t* out_moved, hyle_file_handle file,
                          const void* buf,
                          size_t n) {
  if (out_moved == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_write: out_moved is NULL"));
  }
  *out_moved = 0;
  hyle::platform::FileHandle tmp{file};
  auto result = tmp.write(buf, n);
  tmp.release();
  if (!result) return steal_status(std::move(result).error());
  *out_moved = *result;
  return steal_status(hyle::Status::Ok());
}
int hyle_fs_set_env_var(const char* name, const char* value) {
  return fs_result(name && hyle::platform::set_env_var(name, value),
                   "hyle_fs_set_env_var failed");
}
int hyle_fs_unset_env_var(const char* name) {
  return fs_result(name && hyle::platform::unset_env_var(name),
                   "hyle_fs_unset_env_var failed");
}

hyle_status hyle_mapped_file_open(hyle_mapped_file_o** out, const char* path,
                                  hyle_map_mode mode, size_t offset,
                                  size_t length) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_mapped_file_open: out is NULL"));
  }
  *out = nullptr;
  auto m = hyle::platform::MappedFile::open(
      path, static_cast<hyle::platform::MappedFile::Mode>(mode), offset, length);
  if (!m) return steal_status(std::move(m).error());
  auto* o = new (std::nothrow) hyle_mapped_file_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_mapped_file_open: wrapper allocation failed"));
  }
  o->mf = std::move(*m);
  *out = o;
  return steal_status(hyle::Status());
}
void hyle_mapped_file_destroy(hyle_mapped_file_o* m) { delete m; }
int hyle_mapped_file_valid(const hyle_mapped_file_o* m) {
  return m && m->mf.valid() ? 1 : 0;
}
void* hyle_mapped_file_data(hyle_mapped_file_o* m) {
  return m ? m->mf.data() : nullptr;
}
size_t hyle_mapped_file_size(const hyle_mapped_file_o* m) {
  return m ? m->mf.size() : 0;
}
hyle_status hyle_mapped_file_flush(hyle_mapped_file_o* m) {
  if (m == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_mapped_file_flush: mapping is NULL"));
  }
  return steal_status(m->mf.flush());
}

}  // extern "C"
