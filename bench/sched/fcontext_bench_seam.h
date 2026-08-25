#pragma once

#include <cstddef>

// Validation-only caller-owned storage for the direct fcontext consumer seam.
// Keeping this object on the benchmark caller's stack matches the public Fiber
// row and prevents an unrelated new/delete pair from biasing the comparison.
inline constexpr std::size_t kHyleBenchFcontextSeamStorageSize = 512;
inline constexpr std::size_t kHyleBenchFcontextSeamStorageAlignment =
    alignof(std::max_align_t);

#if defined(HYLE_BENCH_PEER_STATIC)
#  define HYLE_BENCH_PEER_API
#elif defined(_WIN32)
#  if defined(HYLE_BENCH_PEER_EXPORTS)
#    define HYLE_BENCH_PEER_API __declspec(dllexport)
#  else
#    define HYLE_BENCH_PEER_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_BENCH_PEER_API __attribute__((visibility("default")))
#else
#  define HYLE_BENCH_PEER_API
#endif

struct HyleBenchFcontextSeamContext;
extern "C" HYLE_BENCH_PEER_API void hyle_bench_raw_fcontext_round_trips(
    int rounds);
extern "C" HYLE_BENCH_PEER_API HyleBenchFcontextSeamContext*
hyle_bench_fcontext_seam_create(
    void* storage, std::size_t storage_size, void (*entry)(void*),
    void* argument, bool tracks_current);
extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_destroy(
    HyleBenchFcontextSeamContext* context);
extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_resume(
    HyleBenchFcontextSeamContext* context);
extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_yield_handle(
    HyleBenchFcontextSeamContext* context);
extern "C" HYLE_BENCH_PEER_API void hyle_bench_fcontext_seam_yield_current();
