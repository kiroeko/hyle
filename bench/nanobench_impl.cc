// Single translation unit that compiles the nanobench implementation. Every
// other bench file includes nanobench.h as a declarations-only header; the
// heavy impl (ANKERL_NANOBENCH_IMPLEMENT) lives here once.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
