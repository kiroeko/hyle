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
  out.rep = s.release_rep();
  return out;
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

hyle_parker_o* hyle_parker_create(void) {
  set_platform_error(nullptr);
  auto* p = new (std::nothrow) hyle_parker_o{};
  if (p == nullptr) {
    set_platform_error(
        "hyle_parker_create: wrapper allocation failed (out of memory)");
  }
  return p;
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
    return steal_status(std::move(r).error());
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

hyle_dynlib_o* hyle_dynlib_open(const char* path) {
  if (path == nullptr) {
    hyle::platform::dynlib_set_last_error("hyle_dynlib_open: path is NULL");
    return nullptr;
  }
  auto lib = hyle::platform::DynLib::open(path);
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    hyle::platform::dynlib_set_last_error(
        "hyle_dynlib_open: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->lib = std::move(lib);
  return o;
}

hyle_dynlib_o* hyle_dynlib_get(const char* path) {
  auto lib = hyle::platform::DynLib::get(path);
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    hyle::platform::dynlib_set_last_error(
        "hyle_dynlib_get: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->lib = std::move(lib);
  return o;
}

hyle_dynlib_o* hyle_dynlib_current(void) {
  auto lib = hyle::platform::DynLib::current();
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    hyle::platform::dynlib_set_last_error(
        "hyle_dynlib_current: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->lib = std::move(lib);
  return o;
}

void hyle_dynlib_close(hyle_dynlib_o* lib) { delete lib; }
int hyle_dynlib_valid(const hyle_dynlib_o* lib) {
  return lib != nullptr && lib->lib.valid() ? 1 : 0;
}
void* hyle_dynlib_symbol(hyle_dynlib_o* lib, const char* name) {
  if (!lib || !name) {
    hyle::platform::dynlib_set_last_error(
        "hyle_dynlib_symbol: library or name is NULL");
    return nullptr;
  }
  return lib->lib.symbol(name);
}
const char* hyle_dynlib_last_error(void) {
  return hyle::platform::dynlib_last_error();
}
const char* hyle_dynlib_extension(void) {
  return hyle::platform::dynlib_extension();
}

int hyle_dynlib_open_symbol(const char* path, const char* name,
                            hyle_dynlib_o** out_lib, void** out_sym) {
  if (out_lib) *out_lib = nullptr;
  if (out_sym) *out_sym = nullptr;
  if (!path || !name || !out_lib || !out_sym) {
    hyle::platform::dynlib_set_last_error(
        "hyle_dynlib_open_symbol: invalid argument");
    return 0;
  }
  hyle_dynlib_o* lib = hyle_dynlib_open(path);
  if (!hyle_dynlib_valid(lib)) {
    hyle_dynlib_close(lib);
    return 0;
  }
  void* sym = hyle_dynlib_symbol(lib, name);
  if (sym == nullptr) {
    hyle_dynlib_close(lib);
    return 0;
  }
  *out_lib = lib;
  *out_sym = sym;
  return 1;
}

size_t hyle_page_size(void) { return hyle::platform::page_size(); }
void* hyle_vm_reserve(size_t size) { return hyle::platform::vm_reserve(size); }
int hyle_vm_commit(void* ptr, size_t size) {
  return hyle::platform::vm_commit(ptr, size) ? 1 : 0;
}
int hyle_vm_decommit(void* ptr, size_t size) {
  return hyle::platform::vm_decommit(ptr, size) ? 1 : 0;
}
int hyle_vm_release(void* base, size_t size) {
  return hyle::platform::vm_release(base, size) ? 1 : 0;
}
int hyle_vm_protect(void* ptr, size_t size, hyle_protect prot) {
  if (!valid_protect(prot)) {
    hyle::platform::vm_set_last_error("hyle_vm_protect: invalid protection");
    return 0;
  }
  return hyle::platform::vm_protect(ptr, size, to_cpp_protect(prot)) ? 1 : 0;
}
const char* hyle_vm_last_error(void) { return hyle::platform::vm_last_error(); }

void* hyle_vm_alloc_rw(size_t size) {
  if (size == 0) {
    hyle::platform::vm_set_last_error("hyle_vm_alloc_rw: size is zero");
    return nullptr;
  }
  size_t aligned = hyle_page_align_up(size);
  if (aligned == 0) {
    hyle::platform::vm_set_last_error("hyle_vm_alloc_rw: size overflow");
    return nullptr;
  }
  void* base = hyle::platform::vm_reserve(aligned);
  if (base == nullptr) return nullptr;
  // vm_commit makes the range Protect::kReadWrite.
  if (!hyle::platform::vm_commit(base, aligned)) {
    (void)hyle::platform::vm_release(base, aligned);
    return nullptr;
  }
  return base;
}

int hyle_vm_free_rw(void* p, size_t size) {
  // Same page rounding as alloc_rw's reserve, so the released span matches.
  const size_t aligned = hyle_page_align_up(size);
  if (p != nullptr && aligned == 0) {
    hyle::platform::vm_set_last_error(
        "hyle_vm_free_rw: zero or overflowing size");
    return 0;
  }
  return hyle::platform::vm_release(p, aligned) ? 1 : 0;
}

hyle_vm_region_o* hyle_vm_region_reserve(size_t size) {
  auto r = hyle::platform::VmRegion::reserve(size);
  if (!r.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) {
    hyle::platform::vm_set_last_error("vm_region create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->region = std::move(r);
  return o;
}

hyle_vm_region_o* hyle_vm_region_reserve_guarded(size_t usable) {
  auto r = hyle::platform::VmRegion::reserve_guarded(usable);
  if (!r.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) {
    hyle::platform::vm_set_last_error("vm_region create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->region = std::move(r);
  return o;
}

hyle_vm_region_o* hyle_vm_region_reserve_numa(size_t size, int node) {
  auto r = hyle::platform::VmRegion::reserve_numa(size, node);
  if (!r.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) {
    hyle::platform::vm_set_last_error("vm_region create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->region = std::move(r);
  return o;
}

hyle_vm_region_o* hyle_vm_region_adopt(void* base, size_t size) {
  if (base == nullptr || size == 0) {
    hyle::platform::vm_set_last_error(
        "hyle_vm_region_adopt: null base or zero size");
    return nullptr;
  }
  // Allocate the wrapper before taking ownership. On wrapper OOM the caller
  // still owns `base`; an adopt API must transfer ownership only on success.
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) {
    hyle::platform::vm_set_last_error("vm_region create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->region = hyle::platform::VmRegion::adopt(base, size);
  if (!o->region.valid()) {
    delete o;
    return nullptr;
  }
  return o;
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
int hyle_vm_region_commit(hyle_vm_region_o* r, size_t off, size_t len) {
  if (r == nullptr) {
    hyle::platform::vm_set_last_error("hyle_vm_region_commit: region is NULL");
    return 0;
  }
  return r && r->region.commit(off, len) ? 1 : 0;
}
int hyle_vm_region_commit_usable(hyle_vm_region_o* r) {
  if (r == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_vm_region_commit_usable: region is NULL");
    return 0;
  }
  return r && r->region.commit_usable() ? 1 : 0;
}
int hyle_vm_region_decommit(hyle_vm_region_o* r, size_t off, size_t len) {
  if (r == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_vm_region_decommit: region is NULL");
    return 0;
  }
  return r && r->region.decommit(off, len) ? 1 : 0;
}
int hyle_vm_region_protect(hyle_vm_region_o* r, size_t off, size_t len,
                           hyle_protect prot) {
  if (!valid_protect(prot)) {
    hyle::platform::vm_set_last_error(
        "hyle_vm_region_protect: invalid protection");
    return 0;
  }
  if (r == nullptr) {
    hyle::platform::vm_set_last_error("hyle_vm_region_protect: region is NULL");
    return 0;
  }
  return r && r->region.protect(off, len, to_cpp_protect(prot)) ? 1 : 0;
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

void hyle_vm_set_last_error(const char* msg) {
  hyle::platform::vm_set_last_error(msg);
}

size_t hyle_large_page_size(void) { return hyle::platform::large_page_size(); }

int hyle_vm_advise_hugepage(void* ptr, size_t size) {
  return hyle::platform::vm_advise_hugepage(ptr, size) ? 1 : 0;
}

int hyle_vm_advise_no_hugepage(void* ptr, size_t size) {
  return hyle::platform::vm_advise_no_hugepage(ptr, size) ? 1 : 0;
}

unsigned hyle_numa_node_count(void) { return hyle::platform::numa_node_count(); }

unsigned hyle_numa_current_node(void) {
  return hyle::platform::numa_current_node();
}

unsigned hyle_numa_node_of_cpu(unsigned cpu) {
  return hyle::platform::numa_node_of_cpu(cpu);
}

int hyle_vm_bind_numa(void* ptr, size_t size, int node, hyle_numa_policy policy) {
  if (!valid_numa_policy(policy)) {
    hyle::platform::vm_set_last_error("hyle_vm_bind_numa: invalid policy");
    return 0;
  }
  return hyle::platform::vm_bind_numa(ptr, size, node,
                                      to_cpp_numa_policy(policy))
             ? 1
             : 0;
}

int hyle_vm_bind_numa_interleave(void* ptr, size_t size) {
  return hyle::platform::vm_bind_numa_interleave(ptr, size) ? 1 : 0;
}

void* hyle_vm_reserve_numa(size_t size, int node) {
  return hyle::platform::vm_reserve_numa(size, node);
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
intptr_t hyle_fs_create_temp_file(const char* dir, const char* prefix, char* dst,
                                  size_t cap) {
  hyle::platform::OwnedFd fd = hyle::platform::create_temp_file(dir, prefix, dst, cap);
  const intptr_t raw = fd.release();
  set_platform_error(raw == hyle::platform::OwnedFd::kInvalid
                         ? "hyle_fs_create_temp_file failed"
                         : nullptr);
  return raw;  // transfer ownership to the C caller (close via close_fd)
}
void hyle_fs_close_fd(intptr_t fd) {
  set_platform_error(nullptr);
  hyle::platform::OwnedFd owned{fd};  // adopts and closes on scope exit
  (void)owned;
}
intptr_t hyle_fs_open(const char* path, unsigned flags) {
  if (path == nullptr) {
    set_platform_error("hyle_fs_open: path is NULL");
    return hyle::platform::OwnedFd::kInvalid;
  }
  const unsigned known = HYLE_OPEN_READ | HYLE_OPEN_WRITE | HYLE_OPEN_CREATE |
                         HYLE_OPEN_TRUNCATE | HYLE_OPEN_APPEND;
  if ((flags & ~known) != 0 ||
      ((flags & (HYLE_OPEN_CREATE | HYLE_OPEN_TRUNCATE | HYLE_OPEN_APPEND)) != 0 &&
       (flags & HYLE_OPEN_WRITE) == 0)) {
    set_platform_error("hyle_fs_open: invalid flag combination");
    return hyle::platform::OwnedFd::kInvalid;
  }
  const intptr_t raw = hyle::platform::open_file(path, flags).release();
  set_platform_error(raw == hyle::platform::OwnedFd::kInvalid
                         ? "hyle_fs_open failed"
                         : nullptr);
  return raw;
}
int64_t hyle_fs_read(intptr_t fd, void* buf, size_t n) {
  if (n != 0 && buf == nullptr) {
    set_platform_error("hyle_fs_read: buffer is NULL for nonzero length");
    return -1;
  }
  hyle::platform::OwnedFd tmp{fd};       // borrow (don't close: caller owns fd)
  int64_t r = tmp.read(buf, n);
  tmp.release();
  set_platform_error(r < 0 ? "hyle_fs_read failed" : nullptr);
  return r;
}
int64_t hyle_fs_write(intptr_t fd, const void* buf, size_t n) {
  if (n != 0 && buf == nullptr) {
    set_platform_error("hyle_fs_write: buffer is NULL for nonzero length");
    return -1;
  }
  hyle::platform::OwnedFd tmp{fd};
  int64_t r = tmp.write(buf, n);
  tmp.release();
  set_platform_error(r < 0 ? "hyle_fs_write failed" : nullptr);
  return r;
}
int hyle_fs_set_env_var(const char* name, const char* value) {
  return fs_result(name && hyle::platform::set_env_var(name, value),
                   "hyle_fs_set_env_var failed");
}
int hyle_fs_unset_env_var(const char* name) {
  return fs_result(name && hyle::platform::unset_env_var(name),
                   "hyle_fs_unset_env_var failed");
}

hyle_mapped_file_o* hyle_mapped_file_open(const char* path, hyle_map_mode mode,
                                          size_t offset, size_t length) {
  if (!path || (mode != HYLE_MAP_READ_ONLY &&
                mode != HYLE_MAP_READ_WRITE &&
                mode != HYLE_MAP_COPY_ON_WRITE)) {
    hyle::platform::vm_set_last_error(
        "hyle_mapped_file_open: null path or invalid mode");
    return nullptr;
  }
  auto m = hyle::platform::MappedFile::open(
      path, static_cast<hyle::platform::MappedFile::Mode>(mode), offset, length);
  if (!m.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_mapped_file_o{};
  if (!o) {
    hyle::platform::vm_set_last_error(
        "hyle_mapped_file_open: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  o->mf = std::move(m);
  return o;
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
int hyle_mapped_file_flush(hyle_mapped_file_o* m) {
  return m && m->mf.flush() ? 1 : 0;
}

}  // extern "C"
