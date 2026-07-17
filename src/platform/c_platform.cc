// C-ABI bridge for platform (include/hyle/c/platform.h).

#include "hyle/c/platform.h"

#include <atomic>
#include <cstring>
#include <new>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/dynlib.h"
#include "hyle/platform/fs.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/platform/wait.h"

namespace {

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

hyle::platform::ThreadOpts to_cpp_opts(const hyle_thread_opts* opts) {
  hyle::platform::ThreadOpts o;
  if (opts == nullptr) return o;
  if (opts->name != nullptr && opts->name[0] != '\0') o.name = opts->name;
  if (opts->has_stack_size) o.stack_size = opts->stack_size;
  if (opts->has_affinity_cpu) o.affinity_cpu = opts->affinity_cpu;
  return o;
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

extern "C" {

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
  return new (std::nothrow) hyle_parker_o{};
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
  auto r = hyle::platform::Thread::spawn(entry, ctx, to_cpp_opts(opts));
  if (!r.has_value()) return steal_status(std::move(r).error());
  auto* o = new (std::nothrow) hyle_thread_o{};
  if (o == nullptr) {
    r->join();
    return steal_status(hyle::Status::ResourceExhausted("thread object OOM"));
  }
  o->t = std::move(*r);
  *out = o;
  return steal_status(hyle::Status());
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
  if (path == nullptr) return nullptr;
  auto lib = hyle::platform::DynLib::open(path);
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) return nullptr;
  o->lib = std::move(lib);
  return o;
}

hyle_dynlib_o* hyle_dynlib_get(const char* path) {
  auto lib = hyle::platform::DynLib::get(path);
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) return nullptr;
  o->lib = std::move(lib);
  return o;
}

hyle_dynlib_o* hyle_dynlib_current(void) {
  auto lib = hyle::platform::DynLib::current();
  if (!lib.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) return nullptr;
  o->lib = std::move(lib);
  return o;
}

void hyle_dynlib_close(hyle_dynlib_o* lib) { delete lib; }
int hyle_dynlib_valid(const hyle_dynlib_o* lib) {
  return lib != nullptr && lib->lib.valid() ? 1 : 0;
}
void* hyle_dynlib_symbol(hyle_dynlib_o* lib, const char* name) {
  if (!lib || !name) return nullptr;
  return lib->lib.symbol(name);
}
const char* hyle_dynlib_last_error(void) {
  return hyle::platform::dynlib_last_error();
}
const char* hyle_dynlib_extension(void) {
  return hyle::platform::dynlib_extension();
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
  return hyle::platform::vm_protect(ptr, size, to_cpp_protect(prot)) ? 1 : 0;
}
const char* hyle_vm_last_error(void) { return hyle::platform::vm_last_error(); }

hyle_vm_region_o* hyle_vm_region_reserve(size_t size) {
  auto r = hyle::platform::VmRegion::reserve(size);
  if (!r.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) return nullptr;
  o->region = std::move(r);
  return o;
}

hyle_vm_region_o* hyle_vm_region_reserve_guarded(size_t usable) {
  auto r = hyle::platform::VmRegion::reserve_guarded(usable);
  if (!r.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_vm_region_o{};
  if (!o) return nullptr;
  o->region = std::move(r);
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
  return r && r->region.commit(off, len) ? 1 : 0;
}
int hyle_vm_region_commit_usable(hyle_vm_region_o* r) {
  return r && r->region.commit_usable() ? 1 : 0;
}
int hyle_vm_region_decommit(hyle_vm_region_o* r, size_t off, size_t len) {
  return r && r->region.decommit(off, len) ? 1 : 0;
}
int hyle_vm_region_protect(hyle_vm_region_o* r, size_t off, size_t len,
                           hyle_protect prot) {
  return r && r->region.protect(off, len, to_cpp_protect(prot)) ? 1 : 0;
}

hyle_file_info hyle_fs_stat(const char* path) {
  hyle_file_info out{};
  if (!path) return out;
  auto i = hyle::platform::stat(path);
  out.exists = i.exists ? 1 : 0;
  out.is_directory = i.is_directory ? 1 : 0;
  out.size = i.size;
  out.modified_time = i.modified_time;
  return out;
}
int hyle_fs_exists(const char* path) {
  return path && hyle::platform::exists(path) ? 1 : 0;
}
int hyle_fs_is_directory(const char* path) {
  return path && hyle::platform::is_directory(path) ? 1 : 0;
}
uint64_t hyle_fs_file_size(const char* path) {
  return path ? hyle::platform::file_size(path) : 0;
}
int hyle_fs_make_directory(const char* path) {
  return path && hyle::platform::make_directory(path) ? 1 : 0;
}
int hyle_fs_make_directory_recursive(const char* path) {
  return path && hyle::platform::make_directory_recursive(path) ? 1 : 0;
}
int hyle_fs_remove_directory(const char* path) {
  return path && hyle::platform::remove_directory(path) ? 1 : 0;
}
int hyle_fs_remove_directory_recursive(const char* path) {
  return path && hyle::platform::remove_directory_recursive(path) ? 1 : 0;
}
int hyle_fs_remove_file(const char* path) {
  return path && hyle::platform::remove_file(path) ? 1 : 0;
}
int hyle_fs_rename(const char* from, const char* to) {
  return from && to && hyle::platform::rename_path(from, to) ? 1 : 0;
}
int hyle_fs_copy_file(const char* from, const char* to) {
  return from && to && hyle::platform::copy_file(from, to) ? 1 : 0;
}
int hyle_fs_touch(const char* path) {
  return path && hyle::platform::touch(path) ? 1 : 0;
}
size_t hyle_fs_current_directory(char* dst, size_t cap) {
  return hyle::platform::current_directory(dst, cap);
}
int hyle_fs_change_directory(const char* path) {
  return path && hyle::platform::change_directory(path) ? 1 : 0;
}
size_t hyle_fs_temp_directory(char* dst, size_t cap) {
  return hyle::platform::temp_directory(dst, cap);
}
size_t hyle_fs_home_directory(char* dst, size_t cap) {
  return hyle::platform::home_directory(dst, cap);
}
size_t hyle_env_get(const char* name, char* dst, size_t cap) {
  if (!name) return HYLE_ENV_NOT_SET;
  return hyle::platform::get_env_var(name, dst, cap);
}

int hyle_fs_directory_iterate(const char* path, hyle_fs_visit_fn visit,
                              void* ctx) {
  if (!path || !visit) return 0;
  DirVisitPack pack{visit, ctx};
  return hyle::platform::directory_iterate(path, &dir_visit_trampoline, &pack)
             ? 1
             : 0;
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

int hyle_platform_c_abi_version(void) { return HYLE_PLATFORM_C_ABI_VERSION; }

int hyle_platform_api_get(hyle_platform_api* out, int expected_abi) {
  if (!out || expected_abi != HYLE_PLATFORM_C_ABI_VERSION) return -1;
  out->abi_version = HYLE_PLATFORM_C_ABI_VERSION;
  out->wait_on_address = &hyle_wait_on_address;
  out->wait_on_address_for = &hyle_wait_on_address_for;
  out->wake_one = &hyle_wake_one;
  out->wake_all = &hyle_wake_all;
  out->parker_create = &hyle_parker_create;
  out->parker_destroy = &hyle_parker_destroy;
  out->parker_park = &hyle_parker_park;
  out->parker_unpark = &hyle_parker_unpark;
  out->parker_park_timeout = &hyle_parker_park_timeout;
  out->thread_spawn = &hyle_thread_spawn;
  out->thread_join = &hyle_thread_join;
  out->hardware_concurrency = &hyle_hardware_concurrency;
  out->dynlib_open = &hyle_dynlib_open;
  out->dynlib_get = &hyle_dynlib_get;
  out->dynlib_close = &hyle_dynlib_close;
  out->dynlib_symbol = &hyle_dynlib_symbol;
  out->page_size = &hyle_page_size;
  out->vm_reserve = &hyle_vm_reserve;
  out->vm_commit = &hyle_vm_commit;
  out->vm_release = &hyle_vm_release;
  out->vm_region_reserve_guarded = &hyle_vm_region_reserve_guarded;
  out->vm_region_destroy = &hyle_vm_region_destroy;
  out->fs_exists = &hyle_fs_exists;
  out->fs_make_directory_recursive = &hyle_fs_make_directory_recursive;
  out->env_get = &hyle_env_get;
  return 0;
}

}  // extern "C"
