// C-ABI bridge for platform (include/hyle/c/platform.h).

#include "hyle/c/platform.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

#include "../error/status_access.h"
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

#define HYLE_PLATFORM_UNSIGNED_TAG_ABI(type)                                   \
  static_assert(std::is_same_v<type, std::uint32_t>)
#define HYLE_PLATFORM_SIGNED_TAG_ABI(type)                                     \
  static_assert(std::is_same_v<type, std::int32_t>)
HYLE_PLATFORM_SIGNED_TAG_ABI(hyle_thread_priority);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_thread_config_policy);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_protect);
HYLE_PLATFORM_SIGNED_TAG_ABI(hyle_numa_node);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_numa_policy);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_numa_placement);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_numa_capability);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_file_type);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_file_perms);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_symlink_kind);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_file_open_flags);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_map_mode);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_map_advice);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_isa);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_cpu_kind);
HYLE_PLATFORM_UNSIGNED_TAG_ABI(hyle_cpu_order);
#undef HYLE_PLATFORM_SIGNED_TAG_ABI
#undef HYLE_PLATFORM_UNSIGNED_TAG_ABI

static_assert(std::is_standard_layout_v<hyle_thread_opts>);
static_assert(std::is_trivially_copyable_v<hyle_thread_opts>);
static_assert(offsetof(hyle_thread_opts, struct_size) == 0);
static_assert(offsetof(hyle_thread_opts, reserved) == 4);
static_assert(offsetof(hyle_thread_opts, name) == 8);
static_assert(offsetof(hyle_thread_opts, has_priority) == 40);
static_assert(offsetof(hyle_thread_opts, priority) == 44);
static_assert(offsetof(hyle_thread_opts, config_policy) == 48);
static_assert(offsetof(hyle_thread_opts, affinity_cpus) == 56);
static_assert(offsetof(hyle_thread_opts, affinity_cpu_count) == 64);
static_assert(HYLE_THREAD_OPTS_REQUIRED_SIZE == 8);
static_assert(sizeof(hyle_thread_opts) == 72);
static_assert(
    std::is_same_v<decltype(hyle_thread_opts::priority), hyle_thread_priority>);
static_assert(std::is_same_v<decltype(hyle_thread_opts::config_policy),
                             hyle_thread_config_policy>);
static_assert(std::is_standard_layout_v<hyle_file_info>);
static_assert(std::is_trivially_copyable_v<hyle_file_info>);
static_assert(std::is_standard_layout_v<hyle_cpu_features>);
static_assert(std::is_trivially_copyable_v<hyle_cpu_features>);
static_assert(offsetof(hyle_cpu_features, struct_size) == 0);
static_assert(offsetof(hyle_cpu_features, reserved) == 4);
static_assert(offsetof(hyle_cpu_features, sse2) == 8);
static_assert(HYLE_CPU_FEATURES_REQUIRED_SIZE == 8);
static_assert(sizeof(hyle_cpu_features) == 160);
static_assert(std::is_standard_layout_v<hyle_cache_info>);
static_assert(std::is_trivially_copyable_v<hyle_cache_info>);
static_assert(std::is_standard_layout_v<hyle_cpu_topology>);
static_assert(std::is_trivially_copyable_v<hyle_cpu_topology>);
static_assert(std::is_standard_layout_v<hyle_cpu_info>);
static_assert(std::is_trivially_copyable_v<hyle_cpu_info>);
static_assert(std::is_same_v<decltype(hyle_cpu_info::kind), hyle_cpu_kind>);
static_assert(offsetof(hyle_cpu_info, kind) == 8);
static_assert(sizeof(hyle_cpu_info) == 12);

#define HYLE_PLATFORM_TAG_VALUE_ABI(c_value, cpp_value)                        \
  static_assert(static_cast<std::int64_t>(c_value) ==                          \
                static_cast<std::int64_t>(cpp_value))
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_PRIORITY_LOWEST,
                            hyle::platform::ThreadPriority::kLowest);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_PRIORITY_LOW,
                            hyle::platform::ThreadPriority::kLow);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_PRIORITY_NORMAL,
                            hyle::platform::ThreadPriority::kNormal);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_PRIORITY_HIGH,
                            hyle::platform::ThreadPriority::kHigh);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_PRIORITY_HIGHEST,
                            hyle::platform::ThreadPriority::kHighest);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_CONFIG_BEST_EFFORT,
                            hyle::platform::ThreadConfigPolicy::kBestEffort);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_THREAD_CONFIG_REQUIRED,
                            hyle::platform::ThreadConfigPolicy::kRequired);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_PROTECT_NONE, hyle::platform::Protect::kNone);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_PROTECT_READ, hyle::platform::Protect::kRead);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_PROTECT_READ_WRITE,
                            hyle::platform::Protect::kReadWrite);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_ANY, hyle::platform::kNumaAny);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_PREFERRED,
                            hyle::platform::NumaPolicy::kPreferred);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_BIND, hyle::platform::NumaPolicy::kBind);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_INTERLEAVE,
                            hyle::platform::NumaPolicy::kInterleave);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_PLACEMENT_APPLIED,
                            hyle::platform::NumaPlacement::kApplied);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_PLACEMENT_NOT_NEEDED,
                            hyle::platform::NumaPlacement::kNotNeeded);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_PLACEMENT_UNAVAILABLE,
                            hyle::platform::NumaPlacement::kUnavailable);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_CAP_NONE,
                            hyle::platform::NumaCapability::kNone);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_CAP_TOPOLOGY,
                            hyle::platform::NumaCapability::kTopology);
HYLE_PLATFORM_TAG_VALUE_ABI(
    HYLE_NUMA_CAP_PREFERRED_RESERVATION,
    hyle::platform::NumaCapability::kPreferredReservation);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_NUMA_CAP_BIND_EXISTING,
                            hyle::platform::NumaCapability::kBindExisting);
HYLE_PLATFORM_TAG_VALUE_ABI(
    HYLE_NUMA_CAP_INTERLEAVE_EXISTING,
    hyle::platform::NumaCapability::kInterleaveExisting);
HYLE_PLATFORM_TAG_VALUE_ABI(
    HYLE_NUMA_CAP_SELECTED_INTERLEAVE,
    hyle::platform::NumaCapability::kSelectedInterleave);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_UNKNOWN,
                            hyle::platform::FileType::kUnknown);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_REGULAR,
                            hyle::platform::FileType::kRegular);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_DIRECTORY,
                            hyle::platform::FileType::kDirectory);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_SYMLINK,
                            hyle::platform::FileType::kSymlink);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_BLOCK_DEVICE,
                            hyle::platform::FileType::kBlockDevice);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_CHARACTER_DEVICE,
                            hyle::platform::FileType::kCharacterDevice);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_FIFO,
                            hyle::platform::FileType::kFifo);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_FILE_TYPE_SOCKET,
                            hyle::platform::FileType::kSocket);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_SYMLINK_AUTO,
                            hyle::platform::SymlinkKind::kAuto);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_SYMLINK_FILE,
                            hyle::platform::SymlinkKind::kFile);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_SYMLINK_DIRECTORY,
                            hyle::platform::SymlinkKind::kDirectory);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_OPEN_READ, hyle::platform::kOpenRead);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_OPEN_WRITE, hyle::platform::kOpenWrite);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_OPEN_CREATE, hyle::platform::kOpenCreate);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_OPEN_TRUNCATE, hyle::platform::kOpenTruncate);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_OPEN_APPEND, hyle::platform::kOpenAppend);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_READ_ONLY,
                            hyle::platform::MappedFile::Mode::kReadOnly);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_READ_WRITE,
                            hyle::platform::MappedFile::Mode::kReadWrite);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_COPY_ON_WRITE,
                            hyle::platform::MappedFile::Mode::kCopyOnWrite);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_ADVICE_NORMAL,
                            hyle::platform::MappedFile::Advice::kNormal);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_ADVICE_RANDOM,
                            hyle::platform::MappedFile::Advice::kRandom);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_ADVICE_SEQUENTIAL,
                            hyle::platform::MappedFile::Advice::kSequential);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_ADVICE_WILL_NEED,
                            hyle::platform::MappedFile::Advice::kWillNeed);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_MAP_ADVICE_DONT_NEED,
                            hyle::platform::MappedFile::Advice::kDontNeed);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_SCALAR, hyle::platform::Isa::kScalar);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_SSE2, hyle::platform::Isa::kSse2);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_SSSE3, hyle::platform::Isa::kSsse3);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_SSE42, hyle::platform::Isa::kSse42);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_AVX, hyle::platform::Isa::kAvx);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_AVX2, hyle::platform::Isa::kAvx2);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_AVX512, hyle::platform::Isa::kAvx512);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_NEON, hyle::platform::Isa::kNeon);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_SVE, hyle::platform::Isa::kSve);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_ISA_RVV, hyle::platform::Isa::kRvv);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_KIND_UNKNOWN,
                            hyle::platform::CpuKind::kUnknown);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_KIND_EFFICIENCY,
                            hyle::platform::CpuKind::kEfficiency);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_KIND_PERFORMANCE,
                            hyle::platform::CpuKind::kPerformance);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_ORDER_OS_DEFAULT,
                            hyle::platform::CpuOrder::kOsDefault);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_ORDER_PERFORMANCE_FIRST,
                            hyle::platform::CpuOrder::kPerformanceFirst);
HYLE_PLATFORM_TAG_VALUE_ABI(HYLE_CPU_ORDER_EFFICIENCY_FIRST,
                            hyle::platform::CpuOrder::kEfficiencyFirst);
#undef HYLE_PLATFORM_TAG_VALUE_ABI

static_assert(HYLE_PERM_OWNER_READ == 0400);
static_assert(HYLE_PERM_OWNER_WRITE == 0200);
static_assert(HYLE_PERM_OWNER_EXEC == 0100);
static_assert(HYLE_PERM_GROUP_READ == 0040);
static_assert(HYLE_PERM_GROUP_WRITE == 0020);
static_assert(HYLE_PERM_GROUP_EXEC == 0010);
static_assert(HYLE_PERM_OTHER_READ == 0004);
static_assert(HYLE_PERM_OTHER_WRITE == 0002);
static_assert(HYLE_PERM_OTHER_EXEC == 0001);
static_assert(HYLE_PERM_ALL == 0777);

hyle_status steal_status(hyle::Status &&s) noexcept {
  return hyle::detail::status_to_ffi(std::move(s));
}

hyle_numa_placement
to_c_numa_placement(hyle::platform::NumaPlacement placement) noexcept {
  return static_cast<hyle_numa_placement>(placement);
}

// C volatile uint32_t* is the public wait word; Parker uses atomic<uint32_t>.
// Same object representation for aligned uint32 storage.
const std::atomic<std::uint32_t> *as_atomic(const volatile uint32_t *addr) {
  return reinterpret_cast<const std::atomic<std::uint32_t> *>(
      const_cast<const uint32_t *>(addr));
}
std::atomic<std::uint32_t> *as_atomic_mut(volatile uint32_t *addr) {
  return reinterpret_cast<std::atomic<std::uint32_t> *>(
      const_cast<uint32_t *>(addr));
}

hyle::platform::Protect to_cpp_protect(hyle_protect p) noexcept {
  using hyle::platform::Protect;
  if (p == HYLE_PROTECT_READ)
    return Protect::kRead;
  if (p == HYLE_PROTECT_READ_WRITE)
    return Protect::kReadWrite;
  return Protect::kNone;
}

hyle::platform::NumaPolicy to_cpp_numa_policy(hyle_numa_policy p) noexcept {
  using hyle::platform::NumaPolicy;
  if (p == HYLE_NUMA_BIND)
    return NumaPolicy::kBind;
  if (p == HYLE_NUMA_INTERLEAVE)
    return NumaPolicy::kInterleave;
  return NumaPolicy::kPreferred;
}

hyle::platform::ThreadOpts to_cpp_opts(const hyle_thread_opts *opts) {
  hyle::platform::ThreadOpts o;
  if (opts == nullptr)
    return o;
  if (opts->name != nullptr && opts->name[0] != '\0')
    o.name = opts->name;
  if (opts->has_stack_size)
    o.stack_size = opts->stack_size;
  if (opts->affinity_cpu_count != 0) {
    o.affinity = hyle::platform::CpuAffinity(std::span<const unsigned>(
        opts->affinity_cpus, opts->affinity_cpu_count));
  } else if (opts->has_affinity_cpu) {
    o.affinity = opts->affinity_cpu;
  }
  if (opts->has_priority) {
    o.priority = static_cast<hyle::platform::ThreadPriority>(
        static_cast<std::int8_t>(opts->priority));
  }
  o.config_policy = opts->config_policy == HYLE_THREAD_CONFIG_REQUIRED
                        ? hyle::platform::ThreadConfigPolicy::kRequired
                        : hyle::platform::ThreadConfigPolicy::kBestEffort;
  return o;
}

bool valid_priority(hyle_thread_priority p) noexcept {
  return p >= HYLE_THREAD_PRIORITY_LOWEST && p <= HYLE_THREAD_PRIORITY_HIGHEST;
}

bool valid_thread_config_policy(hyle_thread_config_policy policy) noexcept {
  return policy == HYLE_THREAD_CONFIG_BEST_EFFORT ||
         policy == HYLE_THREAD_CONFIG_REQUIRED;
}

constexpr std::size_t kThreadOptsAffinityPairEnd =
    offsetof(hyle_thread_opts, affinity_cpu_count) +
    sizeof(decltype(hyle_thread_opts::affinity_cpu_count));

constexpr bool thread_opts_has_member(std::uint32_t struct_size,
                                      std::size_t offset,
                                      std::size_t member_size) noexcept {
  const std::size_t readable = struct_size;
  return readable >= offset && readable - offset >= member_size;
}

#define HYLE_THREAD_OPTS_HAS_MEMBER(size, member)                              \
  thread_opts_has_member((size), offsetof(hyle_thread_opts, member),           \
                         sizeof(decltype(hyle_thread_opts::member)))

hyle::Status normalize_thread_opts(const hyle_thread_opts *input,
                                   hyle_thread_opts *normalized) noexcept {
  const hyle_thread_opts defaults = hyle_thread_opts_default();
  *normalized = defaults;
  if (input == nullptr)
    return hyle::Status::Ok();

  // struct_size itself is the one mandatory read. Once accepted, overlay at
  // most the byte prefix the caller declared and this library understands.
  // This is safe for an actual shorter allocation that follows the explicit
  // prefix contract and ignores a future caller's unknown tail without
  // assuming it has this library's sizeof(struct).
  std::uint32_t declared_size = 0;
  std::memcpy(&declared_size, input, sizeof(declared_size));
  if (declared_size < HYLE_THREAD_OPTS_REQUIRED_SIZE) {
    return hyle::Status::InvalidArgument("thread opts struct is too small");
  }
  const std::size_t readable = declared_size < sizeof(*normalized)
                                   ? declared_size
                                   : sizeof(*normalized);
  std::memcpy(normalized, input, readable);
  normalized->struct_size = static_cast<std::uint32_t>(sizeof(*normalized));
  if (normalized->reserved != 0) {
    return hyle::Status::InvalidArgument(
        "thread opts reserved field must be zero");
  }
  normalized->reserved = 0;

  // A partially readable scalar or pointer must never escape the raw overlay.
  // Flag/value options and pointer/count views are atomic extension units: both
  // fields must be complete before either can affect validation or dispatch.
  if (!HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, name))
    normalized->name = defaults.name;
  if (!HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, has_stack_size) ||
      !HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, stack_size)) {
    normalized->has_stack_size = defaults.has_stack_size;
    normalized->stack_size = defaults.stack_size;
  }
  if (!HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, has_affinity_cpu) ||
      !HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, affinity_cpu)) {
    normalized->has_affinity_cpu = defaults.has_affinity_cpu;
    normalized->affinity_cpu = defaults.affinity_cpu;
  }
  if (!HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, has_priority) ||
      !HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, priority)) {
    normalized->has_priority = defaults.has_priority;
    normalized->priority = defaults.priority;
  }
  if (!HYLE_THREAD_OPTS_HAS_MEMBER(declared_size, config_policy))
    normalized->config_policy = defaults.config_policy;
  if (declared_size < kThreadOptsAffinityPairEnd) {
    normalized->affinity_cpus = defaults.affinity_cpus;
    normalized->affinity_cpu_count = defaults.affinity_cpu_count;
  }

  if (normalized->has_priority && !valid_priority(normalized->priority)) {
    return hyle::Status::InvalidArgument("thread priority out of range");
  }
  if (normalized->affinity_cpu_count != 0 &&
      normalized->affinity_cpus == nullptr) {
    return hyle::Status::InvalidArgument(
        "thread affinity CPU array is null");
  }
  if (normalized->has_affinity_cpu &&
      normalized->affinity_cpu_count != 0) {
    return hyle::Status::InvalidArgument(
        "thread opts contains both single and set affinity");
  }
  if (!valid_thread_config_policy(normalized->config_policy)) {
    return hyle::Status::InvalidArgument(
        "thread config policy is invalid");
  }
  return hyle::Status::Ok();
}

#undef HYLE_THREAD_OPTS_HAS_MEMBER

bool valid_protect(hyle_protect p) noexcept {
  return p == HYLE_PROTECT_NONE || p == HYLE_PROTECT_READ ||
         p == HYLE_PROTECT_READ_WRITE;
}

bool valid_numa_policy(hyle_numa_policy p) noexcept {
  return p == HYLE_NUMA_PREFERRED || p == HYLE_NUMA_BIND ||
         p == HYLE_NUMA_INTERLEAVE;
}

class NumaNodeArray {
public:
  NumaNodeArray() noexcept = default;
  ~NumaNodeArray() { delete[] heap_; }
  NumaNodeArray(const NumaNodeArray &) = delete;
  NumaNodeArray &operator=(const NumaNodeArray &) = delete;

  hyle::Status assign(const hyle_numa_node *nodes, std::size_t count) noexcept {
    if (nodes == nullptr && count != 0) {
      return hyle::Status::InvalidArgument(
          "NUMA node array is NULL with nonzero count");
    }
    if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(unsigned)) {
      return hyle::Status::OutOfRange("NUMA node array is too large");
    }
    if (count > kStackCount) {
      heap_ = new (std::nothrow) unsigned[count];
      if (heap_ == nullptr) {
        return hyle::Status::ResourceExhausted(
            "NUMA node conversion allocation failed");
      }
      data_ = heap_;
    }
    for (std::size_t i = 0; i < count; ++i) {
      if (nodes[i] < 0) {
        return hyle::Status::InvalidArgument(
            "explicit NUMA node array contains a negative node");
      }
      data_[i] = static_cast<unsigned>(nodes[i]);
    }
    size_ = count;
    return hyle::Status::Ok();
  }

  std::span<const unsigned> span() const noexcept { return {data_, size_}; }

private:
  static constexpr std::size_t kStackCount = 32;
  unsigned stack_[kStackCount]{};
  unsigned *data_ = stack_;
  unsigned *heap_ = nullptr;
  std::size_t size_ = 0;
};

struct DirVisitPack {
  hyle_fs_visit_fn fn;
  void *user;
};

bool dir_visit_trampoline(const char *name, hyle::platform::FileType type,
                          void *ctx) {
  auto *pack = static_cast<DirVisitPack *>(ctx);
  return pack->fn(name, static_cast<hyle_file_type>(type), pack->user) != 0;
}

} // namespace

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

static hyle_status
wrap_vm_region(hyle_vm_region_o **out,
               hyle::Result<hyle::platform::VmRegion> &&region) noexcept {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  if (!region)
    return steal_status(std::move(region).error());
  auto *wrapper = new (std::nothrow) hyle_vm_region_o{};
  if (wrapper == nullptr) {
    return steal_status(
        hyle::Status::ResourceExhausted("VM region wrapper allocation failed"));
  }
  wrapper->region = *std::move(region);
  *out = wrapper;
  return steal_status(hyle::Status::Ok());
}

extern "C" {

void hyle_wait_on_address(volatile uint32_t *addr, uint32_t expected) {
  if (addr == nullptr)
    return;
  hyle::platform::detail::wait_on_address(as_atomic(addr), expected);
}

int hyle_wait_on_address_for(volatile uint32_t *addr, uint32_t expected,
                             uint64_t timeout_ns) {
  if (addr == nullptr)
    return 1;
  // The C contract is a uint32_t word accessed atomically by the caller, not a
  // C++ std::atomic object.  Do the zero-timeout poll through atomic_ref so the
  // backend never invokes std::atomic::load on storage whose C++ lifetime was
  // not started as std::atomic<uint32_t>.  Non-zero waits only hand the object
  // representation to the native futex/WaitOnAddress/os_sync primitive.
  if (timeout_ns == 0) {
    auto &word = *const_cast<uint32_t *>(addr);
    return std::atomic_ref<uint32_t>(word).load(std::memory_order_relaxed) !=
                   expected
               ? 1
               : 0;
  }
  return hyle::platform::detail::wait_on_address_for(as_atomic(addr), expected,
                                                     timeout_ns)
             ? 1
             : 0;
}

void hyle_wake_one(volatile uint32_t *addr) {
  if (addr == nullptr)
    return;
  hyle::platform::detail::wake_one(as_atomic_mut(addr));
}

void hyle_wake_all(volatile uint32_t *addr) {
  if (addr == nullptr)
    return;
  hyle::platform::detail::wake_all(as_atomic_mut(addr));
}

hyle_status hyle_parker_create(hyle_parker_o **out) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_parker_create: out is NULL"));
  }
  *out = nullptr;
  auto *p = new (std::nothrow) hyle_parker_o{};
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_parker_create: wrapper allocation failed"));
  }
  *out = p;
  return steal_status(hyle::Status::Ok());
}
void hyle_parker_destroy(hyle_parker_o *p) { delete p; }
void hyle_parker_park(hyle_parker_o *p) {
  if (p)
    p->p.park();
}
void hyle_parker_unpark(hyle_parker_o *p) {
  if (p)
    p->p.unpark();
}
int hyle_parker_park_timeout(hyle_parker_o *p, uint64_t timeout_ns) {
  return p && p->p.park_timeout(timeout_ns) ? 1 : 0;
}

hyle_status hyle_thread_spawn(hyle_thread_o **out, hyle_thread_entry_fn entry,
                              void *ctx, const hyle_thread_opts *opts) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument("thread spawn args"));
  }
  *out = nullptr;
  if (entry == nullptr) {
    return steal_status(hyle::Status::InvalidArgument("thread spawn args"));
  }
  hyle_thread_opts normalized = HYLE_THREAD_OPTS_INIT;
  hyle::Status normalized_status = normalize_thread_opts(opts, &normalized);
  if (!normalized_status.ok())
    return steal_status(std::move(normalized_status));
  auto *o = new (std::nothrow) hyle_thread_o{};
  if (o == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted("thread object OOM"));
  }
  auto r = hyle::platform::Thread::spawn(entry, ctx,
                                         to_cpp_opts(&normalized));
  if (!r.has_value()) {
    delete o;
    return steal_status(std::move(r).error().WithContext("hyle_thread_spawn"));
  }
  o->t = std::move(*r);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_thread_spawn_opts(hyle_thread_o **out,
                                   hyle_thread_entry_fn entry, void *ctx,
                                   const char *name, size_t stack_size,
                                   int affinity_cpu,
                                   hyle_thread_priority priority) {
  // Translate sentinels into the has_* struct so an FFI caller never touches
  // the flags. name==NULL stays NULL (spawn treats it as "no name").
  hyle_thread_opts o = HYLE_THREAD_OPTS_INIT;
  o.name = name;
  if (stack_size != 0) {
    o.has_stack_size = 1;
    o.stack_size = stack_size;
  }
  if (affinity_cpu >= 0) {
    hyle_thread_opts_set_affinity(&o, static_cast<unsigned>(affinity_cpu));
  }
  o.has_priority = 1; // always set; HYLE_THREAD_PRIORITY_NORMAL==0 = default
  o.priority = priority;
  return hyle_thread_spawn(out, entry, ctx, &o);
}

void hyle_thread_join(hyle_thread_o *t) {
  if (t == nullptr)
    return;
  if (t->t.joinable())
    t->t.join();
  delete t;
}

int hyle_thread_joinable(const hyle_thread_o *t) {
  return t != nullptr && t->t.joinable() ? 1 : 0;
}

unsigned hyle_hardware_concurrency(void) {
  return hyle::platform::hardware_concurrency();
}
uint64_t hyle_this_thread_id(void) {
  return hyle::platform::this_thread::get_id();
}
void hyle_this_thread_yield(void) { hyle::platform::this_thread::yield(); }
hyle_status hyle_this_thread_set_name(const char *name) {
  return steal_status(
      hyle::platform::this_thread::set_name(name != nullptr ? name : ""));
}
hyle_status hyle_this_thread_set_affinity(unsigned cpu) {
  return steal_status(hyle::platform::this_thread::set_affinity(cpu));
}
hyle_status hyle_this_thread_set_affinity_cpus(const unsigned *cpus,
                                               size_t count) {
  if (cpus == nullptr || count == 0) {
    return steal_status(
        hyle::Status::InvalidArgument("thread affinity CPU set is empty"));
  }
  return steal_status(hyle::platform::this_thread::set_affinity(
      hyle::platform::CpuAffinity(std::span<const unsigned>(cpus, count))));
}
hyle_status hyle_this_thread_set_priority(hyle_thread_priority priority) {
  if (!valid_priority(static_cast<int>(priority))) {
    return steal_status(
        hyle::Status::InvalidArgument("thread priority out of range"));
  }
  return steal_status(hyle::platform::this_thread::set_priority(
      static_cast<hyle::platform::ThreadPriority>(
          static_cast<std::int8_t>(priority))));
}

hyle_status hyle_dynlib_open(hyle_dynlib_o **out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_open: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::open(path);
  if (!lib)
    return steal_status(std::move(lib).error());
  auto *o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_open: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_dynlib_get(hyle_dynlib_o **out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_get: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::get(path);
  if (!lib)
    return steal_status(std::move(lib).error());
  auto *o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_get: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

hyle_status hyle_dynlib_current(hyle_dynlib_o **out) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_current: out is NULL"));
  }
  *out = nullptr;
  auto lib = hyle::platform::DynLib::current();
  if (!lib)
    return steal_status(std::move(lib).error());
  auto *o = new (std::nothrow) hyle_dynlib_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_dynlib_current: wrapper allocation failed"));
  }
  o->lib = std::move(*lib);
  *out = o;
  return steal_status(hyle::Status());
}

void hyle_dynlib_close(hyle_dynlib_o *lib) { delete lib; }
int hyle_dynlib_valid(const hyle_dynlib_o *lib) {
  return lib != nullptr && lib->lib.valid() ? 1 : 0;
}
hyle_status hyle_dynlib_symbol(void **out, hyle_dynlib_o *lib,
                               const char *name) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_symbol: out is NULL"));
  }
  *out = nullptr;
  if (lib == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_dynlib_symbol: library is NULL"));
  }
  auto symbol = lib->lib.symbol(name);
  if (!symbol)
    return steal_status(std::move(symbol).error());
  *out = *symbol;
  return steal_status(hyle::Status());
}
const char *hyle_dynlib_extension(void) {
  return hyle::platform::dynlib_extension();
}

hyle_status hyle_dynlib_open_symbol(hyle_dynlib_o **out_lib, void **out_sym,
                                    const char *path, const char *name) {
  if (out_lib)
    *out_lib = nullptr;
  if (out_sym)
    *out_sym = nullptr;
  if (!path || !name || !out_lib || !out_sym) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_dynlib_open_symbol: invalid argument"));
  }
  hyle_dynlib_o *lib = nullptr;
  hyle_status status = hyle_dynlib_open(&lib, path);
  if (status != nullptr)
    return status;
  void *sym = nullptr;
  status = hyle_dynlib_symbol(&sym, lib, name);
  if (status != nullptr) {
    hyle_dynlib_close(lib);
    return status;
  }
  *out_lib = lib;
  *out_sym = sym;
  return status;
}

size_t hyle_page_size(void) { return hyle::platform::page_size(); }
hyle_status hyle_vm_reserve(void **out, size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_reserve: out is NULL"));
  }
  *out = nullptr;
  auto result = hyle::platform::vm_reserve(size);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_vm_commit(void *ptr, size_t size) {
  return steal_status(hyle::platform::vm_commit(ptr, size));
}
hyle_status hyle_vm_decommit(void *ptr, size_t size) {
  return steal_status(hyle::platform::vm_decommit(ptr, size));
}
hyle_status hyle_vm_release(void *base, size_t size) {
  return steal_status(hyle::platform::vm_release(base, size));
}
hyle_status hyle_vm_protect(void *ptr, size_t size, hyle_protect prot) {
  if (!valid_protect(prot)) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_protect: invalid protection"));
  }
  return steal_status(
      hyle::platform::vm_protect(ptr, size, to_cpp_protect(prot)));
}

hyle_status hyle_vm_alloc_rw(void **out, size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_alloc_rw: out is NULL"));
  }
  *out = nullptr;
  auto allocated = hyle::platform::vm_allocate_rw(size);
  if (!allocated)
    return steal_status(std::move(allocated).error());
  *out = *allocated;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_free_rw(void *p, size_t size) {
  // Same page rounding as alloc_rw's reserve, so the released span matches.
  const size_t aligned = hyle_page_align_up(size);
  if (p != nullptr && aligned == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_free_rw: zero or overflowing size"));
  }
  return steal_status(hyle::platform::vm_release(p, aligned));
}

hyle_status hyle_vm_region_reserve(hyle_vm_region_o **out, size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  return wrap_vm_region(out, hyle::platform::VmRegion::reserve(size));
}

hyle_status hyle_vm_region_reserve_guarded(hyle_vm_region_o **out,
                                           size_t usable) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  return wrap_vm_region(out, hyle::platform::VmRegion::reserve_guarded(usable));
}

hyle_status hyle_vm_region_reserve_numa(hyle_vm_region_o **out, size_t size,
                                        hyle_numa_node node) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  return wrap_vm_region(out,
                        hyle::platform::VmRegion::reserve_numa(size, node));
}

hyle_status hyle_vm_region_alloc_numa_interleaved(hyle_vm_region_o **out,
                                                  size_t size,
                                                  const hyle_numa_node *nodes,
                                                  size_t node_count,
                                                  size_t bytes_per_chunk) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("VM region output pointer is NULL"));
  }
  *out = nullptr;
  NumaNodeArray converted;
  hyle::Status converted_status = converted.assign(nodes, node_count);
  if (!converted_status.ok())
    return steal_status(std::move(converted_status));
  return wrap_vm_region(out,
                        hyle::platform::VmRegion::allocate_numa_interleaved(
                            size, converted.span(), bytes_per_chunk));
}

hyle_status hyle_vm_region_adopt(hyle_vm_region_o **out, void *base,
                                 size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_region_adopt: out is NULL"));
  }
  *out = nullptr;
  // Allocate the wrapper before taking ownership. On wrapper OOM the caller
  // still owns `base`; an adopt API must transfer ownership only on success.
  auto *o = new (std::nothrow) hyle_vm_region_o{};
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

void hyle_vm_region_destroy(hyle_vm_region_o *r) { delete r; }
int hyle_vm_region_valid(const hyle_vm_region_o *r) {
  return r && r->region.valid() ? 1 : 0;
}
void *hyle_vm_region_data(hyle_vm_region_o *r) {
  return r ? r->region.data() : nullptr;
}
size_t hyle_vm_region_size(const hyle_vm_region_o *r) {
  return r ? r->region.size() : 0;
}
void *hyle_vm_region_usable(hyle_vm_region_o *r) {
  return r ? r->region.usable() : nullptr;
}
size_t hyle_vm_region_usable_size(const hyle_vm_region_o *r) {
  return r ? r->region.usable_size() : 0;
}
int hyle_vm_region_has_guard(const hyle_vm_region_o *r) {
  return r && r->region.has_guard() ? 1 : 0;
}
int hyle_vm_region_is_numa_interleaved(const hyle_vm_region_o *r) {
  return r && r->region.numa_interleaved() ? 1 : 0;
}
hyle_numa_placement hyle_vm_region_numa_placement(const hyle_vm_region_o *r) {
  return r ? to_c_numa_placement(r->region.numa_placement())
           : static_cast<hyle_numa_placement>(HYLE_NUMA_PLACEMENT_NOT_NEEDED);
}
hyle_status hyle_vm_region_commit(hyle_vm_region_o *r, size_t off, size_t len) {
  if (r == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_region_commit: region is NULL"));
  }
  return steal_status(r->region.commit(off, len));
}
hyle_status hyle_vm_region_commit_usable(hyle_vm_region_o *r) {
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_commit_usable: region is NULL"));
  }
  return steal_status(r->region.commit_usable());
}
hyle_status hyle_vm_region_decommit(hyle_vm_region_o *r, size_t off,
                                    size_t len) {
  if (r == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_region_decommit: region is NULL"));
  }
  return steal_status(r->region.decommit(off, len));
}
hyle_status hyle_vm_region_protect(hyle_vm_region_o *r, size_t off, size_t len,
                                   hyle_protect prot) {
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

hyle_status hyle_fs_stat(hyle_file_info *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_stat: out is NULL"));
  }
  *out = {};
  auto info = hyle::platform::stat(path);
  if (!info)
    return steal_status(std::move(info).error());
  out->type = static_cast<hyle_file_type>(info->type);
  out->size = info->size;
  out->modified_time = info->modified_time;
  out->is_symlink = info->is_symlink ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_exists(int *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_exists: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::exists(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_is_directory(int *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_is_directory: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::is_directory(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_file_size(uint64_t *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_file_size: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::file_size(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_make_directory(const char *path) {
  return steal_status(hyle::platform::make_directory(path));
}
hyle_status hyle_fs_make_directory_recursive(const char *path) {
  return steal_status(hyle::platform::make_directory_recursive(path));
}
hyle_status hyle_fs_remove_directory(const char *path) {
  return steal_status(hyle::platform::remove_directory(path));
}
hyle_status hyle_fs_remove_directory_recursive(const char *path) {
  return steal_status(hyle::platform::remove_directory_recursive(path));
}
hyle_status hyle_fs_remove_file(const char *path) {
  return steal_status(hyle::platform::remove_file(path));
}
hyle_status hyle_fs_rename(const char *from, const char *to) {
  return steal_status(hyle::platform::rename_path(from, to));
}
hyle_status hyle_fs_copy_file(const char *from, const char *to) {
  return steal_status(hyle::platform::copy_file(from, to));
}
hyle_status hyle_fs_touch(const char *path) {
  return steal_status(hyle::platform::touch(path));
}
hyle_status hyle_fs_current_directory(size_t *out_size, char *dst, size_t cap) {
  if (out_size == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_current_directory: out_size is NULL"));
  }
  *out_size = 0;
  auto result = hyle::platform::current_directory(dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_size = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_change_directory(const char *path) {
  return steal_status(hyle::platform::change_directory(path));
}
hyle_status hyle_fs_temp_directory(size_t *out_size, char *dst, size_t cap) {
  if (out_size == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_temp_directory: out_size is NULL"));
  }
  *out_size = 0;
  auto result = hyle::platform::temp_directory(dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_size = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_home_directory(size_t *out_size, char *dst, size_t cap) {
  if (out_size == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_home_directory: out_size is NULL"));
  }
  *out_size = 0;
  auto result = hyle::platform::home_directory(dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_size = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_env_get(size_t *out_size, const char *name, char *dst,
                         size_t cap) {
  if (out_size == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_env_get: out_size is NULL"));
  }
  *out_size = 0;
  auto result = hyle::platform::get_env_var(name, dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_size = *result;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_fs_directory_iterate(int *out_completed, const char *path,
                                      hyle_fs_visit_fn visit, void *ctx) {
  if (out_completed == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_directory_iterate: out_completed is NULL"));
  }
  *out_completed = 0;
  if (visit == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_directory_iterate: visit is NULL"));
  }
  DirVisitPack pack{visit, ctx};
  auto result =
      hyle::platform::directory_iterate(path, &dir_visit_trampoline, &pack);
  if (!result)
    return steal_status(std::move(result).error());
  *out_completed = *result ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_cpu_features_get(hyle_cpu_features *out) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("CPU features output pointer is NULL"));
  }

  std::uint32_t declared_size = 0;
  std::memcpy(&declared_size, out, sizeof(declared_size));
  std::uint32_t reserved = 0;
  if (declared_size >= HYLE_CPU_FEATURES_REQUIRED_SIZE) {
    std::memcpy(&reserved, reinterpret_cast<const unsigned char *>(out) +
                               offsetof(hyle_cpu_features, reserved),
                sizeof(reserved));
  }

  hyle_cpu_features snapshot = HYLE_CPU_FEATURES_INIT;
  snapshot.struct_size = declared_size;
  const std::size_t capped_writable =
      declared_size < sizeof(snapshot) ? declared_size : sizeof(snapshot);
  std::size_t writable = 0;
  if (capped_writable >= sizeof(snapshot.struct_size))
    writable = sizeof(snapshot.struct_size);
  if (capped_writable >= HYLE_CPU_FEATURES_REQUIRED_SIZE) {
    writable = HYLE_CPU_FEATURES_REQUIRED_SIZE +
               ((capped_writable - HYLE_CPU_FEATURES_REQUIRED_SIZE) /
                sizeof(int)) *
                   sizeof(int);
  }
  std::memcpy(out, &snapshot, writable);

  if (declared_size < HYLE_CPU_FEATURES_REQUIRED_SIZE) {
    return steal_status(hyle::Status::InvalidArgument(
        "CPU features struct_size is smaller than the required prefix"));
  }
  if (reserved != 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "CPU features reserved field must be zero"));
  }

  const auto &f = hyle::platform::cpu_features();
#define HYLE_B(x) snapshot.x = f.x ? 1 : 0
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
  std::memcpy(out, &snapshot, writable);
  return steal_status(hyle::Status::Ok());
}

hyle_isa hyle_cpu_best_isa(void) {
  return static_cast<hyle_isa>(hyle::platform::best_isa());
}

int hyle_cpu_supports_isa(hyle_isa isa) {
  return hyle::platform::supports(static_cast<hyle::platform::Isa>(isa)) ? 1
                                                                         : 0;
}

const char *hyle_cpu_isa_name(hyle_isa isa) {
  return hyle::platform::isa_name(static_cast<hyle::platform::Isa>(isa));
}

unsigned hyle_cpu_cache_line_size(void) {
  return hyle::platform::cache_line_size();
}

hyle_cache_info hyle_cpu_cache_info(void) {
  const auto &in = hyle::platform::cache_info();
  return {in.l1d, in.l1i, in.l2, in.l3};
}

hyle_cpu_topology hyle_cpu_topology_get(void) {
  const auto &in = hyle::platform::cpu_topology();
  return {in.system_logical_cpus, in.performance_cpus, in.efficiency_cpus,
          in.hybrid ? 1 : 0};
}

hyle_status hyle_cpu_available_count(size_t *out_count) {
  if (out_count == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_cpu_available_count: out_count is NULL"));
  }
  *out_count = 0;
  auto result = hyle::platform::available_cpu_count();
  if (!result)
    return steal_status(std::move(result).error());
  *out_count = *result;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_cpu_available_ids(size_t *out_count, unsigned *dst,
                                   size_t cap) {
  if (out_count == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_cpu_available_ids: out_count is NULL"));
  }
  *out_count = 0;
  auto result = hyle::platform::available_cpu_ids(dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_count = *result;
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_cpu_available_info(size_t *out_count, hyle_cpu_info *dst,
                                    size_t cap) {
  if (out_count == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_cpu_available_info: out_count is NULL"));
  }
  *out_count = 0;
  if (dst == nullptr) {
    auto result = hyle::platform::available_cpu_info(nullptr, 0);
    if (!result)
      return steal_status(std::move(result).error());
    *out_count = *result;
    return steal_status(hyle::Status::Ok());
  }
  for (unsigned attempt = 0; attempt < 4; ++attempt) {
    auto count = hyle::platform::available_cpu_info(nullptr, 0);
    if (!count)
      return steal_status(std::move(count).error());
    const size_t temp_cap = cap < *count ? cap : *count;
    hyle::platform::CpuInfo stack[128];
    auto *temp = temp_cap <= 128 ? stack
                                 : new (std::nothrow)
                                       hyle::platform::CpuInfo[temp_cap];
    if (temp == nullptr) {
      return steal_status(hyle::Status::ResourceExhausted(
          "hyle_cpu_available_info: temporary allocation failed"));
    }
    auto result = hyle::platform::available_cpu_info(temp, temp_cap);
    if (!result) {
      if (temp != stack)
        delete[] temp;
      return steal_status(std::move(result).error());
    }
    if (*result > *count && cap > *count) {
      if (temp != stack)
        delete[] temp;
      continue;
    }
    const size_t written = temp_cap < *result ? temp_cap : *result;
    for (size_t i = 0; i < written; ++i) {
      dst[i] = {temp[i].id, temp[i].performance_rank,
                static_cast<hyle_cpu_kind>(temp[i].kind)};
    }
    if (temp != stack)
      delete[] temp;
    *out_count = *result;
    return steal_status(hyle::Status::Ok());
  }
  return steal_status(hyle::Status::Unavailable(
      "hyle_cpu_available_info: CPU set changed repeatedly"));
}

hyle_status hyle_cpu_available_ids_ordered(size_t *out_count,
                                           hyle_cpu_order order, unsigned *dst,
                                           size_t cap) {
  if (out_count == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_cpu_available_ids_ordered: out_count is NULL"));
  }
  *out_count = 0;
  if (order < HYLE_CPU_ORDER_OS_DEFAULT ||
      order > HYLE_CPU_ORDER_EFFICIENCY_FIRST) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_cpu_available_ids_ordered: order is invalid"));
  }
  auto result = hyle::platform::available_cpu_ids(
      static_cast<hyle::platform::CpuOrder>(order), dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_count = *result;
  return steal_status(hyle::Status::Ok());
}

/* ---- v3: hugepage/THP + NUMA ------------------------------------------------
 */

size_t hyle_large_page_size(void) { return hyle::platform::large_page_size(); }

hyle_status hyle_vm_advise_hugepage(void *ptr, size_t size) {
  return steal_status(hyle::platform::vm_advise_hugepage(ptr, size));
}

hyle_status hyle_vm_advise_no_hugepage(void *ptr, size_t size) {
  return steal_status(hyle::platform::vm_advise_no_hugepage(ptr, size));
}

hyle_numa_capability hyle_numa_capabilities(void) {
  return static_cast<hyle_numa_capability>(hyle::platform::numa_capabilities());
}

size_t hyle_numa_nodes(hyle_numa_node *out_nodes, size_t capacity) {
  const auto nodes = hyle::platform::numa_nodes();
  if (out_nodes != nullptr) {
    const size_t copied = capacity < nodes.size() ? capacity : nodes.size();
    for (size_t i = 0; i < copied; ++i)
      out_nodes[i] = static_cast<hyle_numa_node>(nodes[i]);
  }
  return nodes.size();
}

unsigned hyle_numa_node_count(void) {
  return hyle::platform::numa_node_count();
}

hyle_numa_node hyle_numa_current_node(void) {
  return static_cast<hyle_numa_node>(hyle::platform::numa_current_node());
}

hyle_numa_node hyle_numa_node_of_cpu(unsigned cpu) {
  return static_cast<hyle_numa_node>(hyle::platform::numa_node_of_cpu(cpu));
}

hyle_status hyle_vm_bind_numa(hyle_numa_placement *out_placement, void *ptr,
                              size_t size, hyle_numa_node node,
                              hyle_numa_policy policy) {
  if (out_placement != nullptr) {
    *out_placement = HYLE_NUMA_PLACEMENT_UNAVAILABLE;
  }
  if (!valid_numa_policy(policy)) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_bind_numa: invalid policy"));
  }
  auto result =
      hyle::platform::vm_bind_numa(ptr, size, node, to_cpp_numa_policy(policy));
  if (!result)
    return steal_status(std::move(result).error());
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(*result);
  }
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_bind_numa_interleave(hyle_numa_placement *out_placement,
                                         void *ptr, size_t size) {
  if (out_placement != nullptr) {
    *out_placement = HYLE_NUMA_PLACEMENT_UNAVAILABLE;
  }
  auto result = hyle::platform::vm_bind_numa_interleave(ptr, size);
  if (!result)
    return steal_status(std::move(result).error());
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(*result);
  }
  return steal_status(hyle::Status::Ok());
}

hyle_status hyle_vm_reserve_numa(void **out, hyle_numa_placement *out_placement,
                                 size_t size, hyle_numa_node node) {
  if (out_placement != nullptr) {
    *out_placement = HYLE_NUMA_PLACEMENT_UNAVAILABLE;
  }
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_vm_reserve_numa: out is NULL"));
  }
  *out = nullptr;
  auto result = hyle::platform::vm_reserve_numa(size, node);
  if (!result)
    return steal_status(std::move(result).error());
  *out = result->data;
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(result->placement);
  }
  return steal_status(hyle::Status::Ok());
}

hyle_status
hyle_vm_alloc_numa_interleaved(void **out, hyle_numa_placement *out_placement,
                               size_t size, const hyle_numa_node *nodes,
                               size_t node_count, size_t bytes_per_chunk) {
  if (out_placement != nullptr) {
    *out_placement = HYLE_NUMA_PLACEMENT_UNAVAILABLE;
  }
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_vm_alloc_numa_interleaved: out is NULL"));
  }
  *out = nullptr;
  NumaNodeArray converted;
  hyle::Status converted_status = converted.assign(nodes, node_count);
  if (!converted_status.ok())
    return steal_status(std::move(converted_status));
  auto result = hyle::platform::vm_allocate_numa_interleaved(
      size, converted.span(), bytes_per_chunk);
  if (!result)
    return steal_status(std::move(result).error());
  *out = result->data;
  if (out_placement != nullptr) {
    *out_placement = to_c_numa_placement(result->placement);
  }
  return steal_status(hyle::Status::Ok());
}

/* ---- v4: fs perms/symlink/tempfile/env-write + mapped file -----------------
 */

hyle_status hyle_fs_get_permissions(hyle_file_perms *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_get_permissions: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::get_permissions(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_set_permissions(const char *path, hyle_file_perms mode) {
  return steal_status(hyle::platform::set_permissions(path, mode));
}
hyle_status hyle_fs_is_readonly(int *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_is_readonly: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::is_readonly(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_set_readonly(const char *path, int readonly) {
  return steal_status(hyle::platform::set_readonly(path, readonly != 0));
}
hyle_status hyle_fs_create_symlink(const char *target, const char *link,
                                   hyle_symlink_kind kind) {
  hyle::platform::SymlinkKind cpp_kind;
  switch (kind) {
  case HYLE_SYMLINK_AUTO:
    cpp_kind = hyle::platform::SymlinkKind::kAuto;
    break;
  case HYLE_SYMLINK_FILE:
    cpp_kind = hyle::platform::SymlinkKind::kFile;
    break;
  case HYLE_SYMLINK_DIRECTORY:
    cpp_kind = hyle::platform::SymlinkKind::kDirectory;
    break;
  default:
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_create_symlink: invalid kind"));
  }
  return steal_status(hyle::platform::create_symlink(target, link, cpp_kind));
}
hyle_status hyle_fs_create_hard_link(const char *target, const char *link) {
  return steal_status(hyle::platform::create_hard_link(target, link));
}
hyle_status hyle_fs_is_symlink(int *out, const char *path) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_is_symlink: out is NULL"));
  }
  *out = 0;
  auto result = hyle::platform::is_symlink(path);
  if (!result)
    return steal_status(std::move(result).error());
  *out = *result ? 1 : 0;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_read_symlink(size_t *out_size, const char *path, char *dst,
                                 size_t cap) {
  if (out_size == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_read_symlink: out_size is NULL"));
  }
  *out_size = 0;
  auto result = hyle::platform::read_symlink(path, dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_size = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_create_temp_file(hyle_file_handle *out,
                                     size_t *out_path_size, const char *dir,
                                     const char *prefix, char *dst,
                                     size_t cap) {
  if (out != nullptr)
    *out = hyle::platform::FileHandle::kInvalid;
  if (out_path_size != nullptr)
    *out_path_size = 0;
  if (out == nullptr || out_path_size == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fs_create_temp_file: output is NULL"));
  }
  auto result = hyle::platform::create_temp_file(dir, prefix, dst, cap);
  if (!result)
    return steal_status(std::move(result).error());
  *out_path_size = result->path_size;
  *out = result->file.release();
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_close(hyle_file_handle file) {
  hyle::platform::FileHandle owned{file};
  return steal_status(owned.close());
}
hyle_status hyle_fs_open(hyle_file_handle *out, const char *path,
                         hyle_file_open_flags flags) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_open: out is NULL"));
  }
  *out = hyle::platform::FileHandle::kInvalid;
  auto result = hyle::platform::open_file(path, flags);
  if (!result)
    return steal_status(std::move(result).error());
  *out = result->release();
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_read(size_t *out_moved, hyle_file_handle file, void *buf,
                         size_t n) {
  if (out_moved == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_read: out_moved is NULL"));
  }
  *out_moved = 0;
  hyle::platform::FileHandle tmp{file}; // borrow; release before returning
  auto result = tmp.read(buf, n);
  tmp.release();
  if (!result)
    return steal_status(std::move(result).error());
  *out_moved = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_fs_write(size_t *out_moved, hyle_file_handle file,
                          const void *buf, size_t n) {
  if (out_moved == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fs_write: out_moved is NULL"));
  }
  *out_moved = 0;
  hyle::platform::FileHandle tmp{file};
  auto result = tmp.write(buf, n);
  tmp.release();
  if (!result)
    return steal_status(std::move(result).error());
  *out_moved = *result;
  return steal_status(hyle::Status::Ok());
}
hyle_status hyle_env_set(const char *name, const char *value) {
  return steal_status(hyle::platform::set_env_var(name, value));
}
hyle_status hyle_env_unset(const char *name) {
  return steal_status(hyle::platform::unset_env_var(name));
}

hyle_status hyle_mapped_file_open(hyle_mapped_file_o **out, const char *path,
                                  hyle_map_mode mode, size_t offset,
                                  size_t length) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_mapped_file_open: out is NULL"));
  }
  *out = nullptr;
  auto m = hyle::platform::MappedFile::open(
      path, static_cast<hyle::platform::MappedFile::Mode>(mode), offset,
      length);
  if (!m)
    return steal_status(std::move(m).error());
  auto *o = new (std::nothrow) hyle_mapped_file_o{};
  if (!o) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_mapped_file_open: wrapper allocation failed"));
  }
  o->mf = std::move(*m);
  *out = o;
  return steal_status(hyle::Status());
}
void hyle_mapped_file_destroy(hyle_mapped_file_o *m) { delete m; }
int hyle_mapped_file_valid(const hyle_mapped_file_o *m) {
  return m && m->mf.valid() ? 1 : 0;
}
void *hyle_mapped_file_data(hyle_mapped_file_o *m) {
  return m ? m->mf.data() : nullptr;
}
size_t hyle_mapped_file_size(const hyle_mapped_file_o *m) {
  return m ? m->mf.size() : 0;
}
hyle_status hyle_mapped_file_flush(hyle_mapped_file_o *m) {
  if (m == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_mapped_file_flush: mapping is NULL"));
  }
  return steal_status(m->mf.flush());
}

hyle_status hyle_mapped_file_advise(hyle_mapped_file_o *m,
                                    hyle_map_advice advice, size_t offset,
                                    size_t length) {
  if (m == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_mapped_file_advise: mapping is NULL"));
  }
  return steal_status(m->mf.advise(
      static_cast<hyle::platform::MappedFile::Advice>(advice), offset, length));
}

hyle_status hyle_mapped_file_lock(hyle_mapped_file_o *m) {
  if (m == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_mapped_file_lock: mapping is NULL"));
  }
  return steal_status(m->mf.lock());
}

hyle_status hyle_mapped_file_unlock(hyle_mapped_file_o *m) {
  if (m == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_mapped_file_unlock: mapping is NULL"));
  }
  return steal_status(m->mf.unlock());
}

int hyle_mapped_file_locked(const hyle_mapped_file_o *m) {
  return m != nullptr && m->mf.locked() ? 1 : 0;
}

} // extern "C"
