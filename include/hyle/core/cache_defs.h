#ifndef HYLE_CORE_CACHE_DEFS_H_
#define HYLE_CORE_CACHE_DEFS_H_

/* Stable target-family layout recommendations, deliberately shared by C and
 * C++. They are not runtime observations: algorithms may query the
 * best-effort platform::cache_line_size()/hyle_cpu_cache_line_size(), but
 * safety-critical false-sharing spacing uses these compile-time constants.
 *
 * Dense data-plane arrays (queue cells, pool/map shards) use the target value
 * so x86 does not double its footprint merely to share one sizeof across
 * unrelated platform ABIs. Sparse control-plane atomics that are continuously
 * written by different threads may opt into the certified 128-byte maximum.
 * Unknown ARM/RISC-V targets fail closed with the 64/128 conservative pair
 * until a concrete distribution profile is certified. */
#define HYLE_MIN_CONSTRUCTIVE_INTERFERENCE_SIZE 64
#define HYLE_MAX_DESTRUCTIVE_INTERFERENCE_SIZE 128

#if defined(__APPLE__) &&                                             \
    (defined(__aarch64__) || defined(__arm64__))
#  define HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE 128
#  define HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE 128
#elif defined(__x86_64__) || defined(_M_X64)
#  define HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE 64
#  define HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE 64
#else
#  define HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE 64
#  define HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE 128
#endif

#endif /* HYLE_CORE_CACHE_DEFS_H_ */
