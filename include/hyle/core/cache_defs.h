#ifndef HYLE_CORE_CACHE_DEFS_H_
#define HYLE_CORE_CACHE_DEFS_H_

/* Stable compile-time layout bounds, deliberately shared by C and C++.
 *
 * Destructive interference needs an upper bound: underestimating it permits
 * false sharing.  128 bytes matches oneTBB's max_nfs_size and covers the
 * currently certified 64/128-byte cache-line families (including Rosetta and
 * native Apple Silicon) without making public layouts depend on runtime
 * discovery. A target whose runtime coherency line lies outside [64, 128]
 * requires a new layout certification before release.
 *
 * Constructive interference needs a lower bound: data kept inside this span
 * is within one L1 line on every currently certified target. 64 bytes remains
 * honest for both Intel Macs and Rosetta (which cannot be distinguished at
 * compile time), as well as native 128-byte Apple Silicon. Runtime algorithms
 * that need the exact line size must query platform instead of using either
 * layout constant. */
#define HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE 128
#define HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE 64

#endif /* HYLE_CORE_CACHE_DEFS_H_ */
