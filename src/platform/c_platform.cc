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
#include "hyle/platform/mapped_file.h"
#include "hyle/platform/numa.h"
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

int hyle_dynlib_open_symbol(const char* path, const char* name,
                            hyle_dynlib_o** out_lib, void** out_sym) {
  if (out_lib) *out_lib = nullptr;
  if (out_sym) *out_sym = nullptr;
  if (!path || !name || !out_lib || !out_sym) return 0;
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
  return hyle::platform::vm_protect(ptr, size, to_cpp_protect(prot)) ? 1 : 0;
}
const char* hyle_vm_last_error(void) { return hyle::platform::vm_last_error(); }

void* hyle_vm_alloc_rw(size_t size) {
  if (size == 0) return nullptr;
  size_t aligned = hyle_page_align_up(size);
  if (aligned == 0) return nullptr;
  void* base = hyle::platform::vm_reserve(aligned);
  if (base == nullptr) return nullptr;
  // vm_commit makes the range Protect::kReadWrite.
  if (!hyle::platform::vm_commit(base, aligned)) {
    (void)hyle::platform::vm_release(base, aligned);
    return nullptr;
  }
  return base;
}

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
  out.is_symlink = i.is_symlink ? 1 : 0;
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

int hyle_vm_bind_numa(void* ptr, size_t size, int node, hyle_numa_policy policy) {
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
  return path ? hyle::platform::get_permissions(path) : 0;
}
int hyle_fs_set_permissions(const char* path, uint32_t mode) {
  return path && hyle::platform::set_permissions(path, mode) ? 1 : 0;
}
int hyle_fs_is_readonly(const char* path) {
  return path && hyle::platform::is_readonly(path) ? 1 : 0;
}
int hyle_fs_set_readonly(const char* path, int readonly) {
  return path && hyle::platform::set_readonly(path, readonly != 0) ? 1 : 0;
}
int hyle_fs_create_symlink(const char* target, const char* link) {
  return target && link && hyle::platform::create_symlink(target, link) ? 1 : 0;
}
int hyle_fs_create_hard_link(const char* target, const char* link) {
  return target && link && hyle::platform::create_hard_link(target, link) ? 1 : 0;
}
int hyle_fs_is_symlink(const char* path) {
  return path && hyle::platform::is_symlink(path) ? 1 : 0;
}
size_t hyle_fs_read_symlink(const char* path, char* dst, size_t cap) {
  return path ? hyle::platform::read_symlink(path, dst, cap) : 0;
}
intptr_t hyle_fs_create_temp_file(const char* dir, const char* prefix, char* dst,
                                  size_t cap) {
  hyle::platform::OwnedFd fd = hyle::platform::create_temp_file(dir, prefix, dst, cap);
  return fd.release();  // transfer ownership to the C caller (close via close_fd)
}
void hyle_fs_close_fd(intptr_t fd) {
  hyle::platform::OwnedFd owned{fd};  // adopts and closes on scope exit
  (void)owned;
}
int hyle_fs_set_env_var(const char* name, const char* value) {
  return name && hyle::platform::set_env_var(name, value) ? 1 : 0;
}
int hyle_fs_unset_env_var(const char* name) {
  return name && hyle::platform::unset_env_var(name) ? 1 : 0;
}

hyle_mapped_file_o* hyle_mapped_file_open(const char* path, hyle_map_mode mode,
                                          size_t offset, size_t length) {
  if (!path) return nullptr;
  auto m = hyle::platform::MappedFile::open(
      path, static_cast<hyle::platform::MappedFile::Mode>(mode), offset, length);
  if (!m.valid()) return nullptr;
  auto* o = new (std::nothrow) hyle_mapped_file_o{};
  if (!o) return nullptr;
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
  out->thread_joinable = &hyle_thread_joinable;
  out->hardware_concurrency = &hyle_hardware_concurrency;
  out->this_thread_id = &hyle_this_thread_id;
  out->this_thread_yield = &hyle_this_thread_yield;
  out->this_thread_set_name = &hyle_this_thread_set_name;

  out->dynlib_open = &hyle_dynlib_open;
  out->dynlib_get = &hyle_dynlib_get;
  out->dynlib_current = &hyle_dynlib_current;
  out->dynlib_close = &hyle_dynlib_close;
  out->dynlib_valid = &hyle_dynlib_valid;
  out->dynlib_symbol = &hyle_dynlib_symbol;
  out->dynlib_last_error = &hyle_dynlib_last_error;
  out->dynlib_extension = &hyle_dynlib_extension;
  out->dynlib_open_symbol = &hyle_dynlib_open_symbol;

  out->page_size = &hyle_page_size;
  out->vm_reserve = &hyle_vm_reserve;
  out->vm_commit = &hyle_vm_commit;
  out->vm_decommit = &hyle_vm_decommit;
  out->vm_release = &hyle_vm_release;
  out->vm_protect = &hyle_vm_protect;
  out->vm_last_error = &hyle_vm_last_error;
  out->vm_alloc_rw = &hyle_vm_alloc_rw;

  out->vm_region_reserve = &hyle_vm_region_reserve;
  out->vm_region_reserve_guarded = &hyle_vm_region_reserve_guarded;
  out->vm_region_destroy = &hyle_vm_region_destroy;
  out->vm_region_valid = &hyle_vm_region_valid;
  out->vm_region_data = &hyle_vm_region_data;
  out->vm_region_size = &hyle_vm_region_size;
  out->vm_region_usable = &hyle_vm_region_usable;
  out->vm_region_usable_size = &hyle_vm_region_usable_size;
  out->vm_region_has_guard = &hyle_vm_region_has_guard;
  out->vm_region_commit = &hyle_vm_region_commit;
  out->vm_region_commit_usable = &hyle_vm_region_commit_usable;
  out->vm_region_decommit = &hyle_vm_region_decommit;
  out->vm_region_protect = &hyle_vm_region_protect;

  out->fs_stat = &hyle_fs_stat;
  out->fs_exists = &hyle_fs_exists;
  out->fs_is_directory = &hyle_fs_is_directory;
  out->fs_file_size = &hyle_fs_file_size;
  out->fs_make_directory = &hyle_fs_make_directory;
  out->fs_make_directory_recursive = &hyle_fs_make_directory_recursive;
  out->fs_remove_directory = &hyle_fs_remove_directory;
  out->fs_remove_directory_recursive = &hyle_fs_remove_directory_recursive;
  out->fs_remove_file = &hyle_fs_remove_file;
  out->fs_rename = &hyle_fs_rename;
  out->fs_copy_file = &hyle_fs_copy_file;
  out->fs_touch = &hyle_fs_touch;
  out->fs_current_directory = &hyle_fs_current_directory;
  out->fs_change_directory = &hyle_fs_change_directory;
  out->fs_temp_directory = &hyle_fs_temp_directory;
  out->fs_home_directory = &hyle_fs_home_directory;
  out->env_get = &hyle_env_get;
  out->fs_directory_iterate = &hyle_fs_directory_iterate;

  out->cpu_features_get = &hyle_cpu_features_get;
  out->cpu_supports_isa = &hyle_cpu_supports_isa;
  out->cpu_isa_name = &hyle_cpu_isa_name;

  out->c_abi_version = &hyle_platform_c_abi_version;

  /* v3 appended */
  out->vm_set_last_error = &hyle_vm_set_last_error;
  out->large_page_size = &hyle_large_page_size;
  out->vm_advise_hugepage = &hyle_vm_advise_hugepage;
  out->vm_advise_no_hugepage = &hyle_vm_advise_no_hugepage;
  out->numa_node_count = &hyle_numa_node_count;
  out->numa_current_node = &hyle_numa_current_node;
  out->vm_bind_numa = &hyle_vm_bind_numa;
  out->vm_bind_numa_interleave = &hyle_vm_bind_numa_interleave;
  out->vm_reserve_numa = &hyle_vm_reserve_numa;

  /* v4 appended */
  out->fs_get_permissions = &hyle_fs_get_permissions;
  out->fs_set_permissions = &hyle_fs_set_permissions;
  out->fs_is_readonly = &hyle_fs_is_readonly;
  out->fs_set_readonly = &hyle_fs_set_readonly;
  out->fs_create_symlink = &hyle_fs_create_symlink;
  out->fs_create_hard_link = &hyle_fs_create_hard_link;
  out->fs_is_symlink = &hyle_fs_is_symlink;
  out->fs_read_symlink = &hyle_fs_read_symlink;
  out->fs_create_temp_file = &hyle_fs_create_temp_file;
  out->fs_close_fd = &hyle_fs_close_fd;
  out->fs_set_env_var = &hyle_fs_set_env_var;
  out->fs_unset_env_var = &hyle_fs_unset_env_var;
  out->mapped_file_open = &hyle_mapped_file_open;
  out->mapped_file_destroy = &hyle_mapped_file_destroy;
  out->mapped_file_valid = &hyle_mapped_file_valid;
  out->mapped_file_data = &hyle_mapped_file_data;
  out->mapped_file_size = &hyle_mapped_file_size;
  out->mapped_file_flush = &hyle_mapped_file_flush;
  return 0;
}

}  // extern "C"
