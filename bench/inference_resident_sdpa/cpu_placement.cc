// Retained cold code changes text placement; it is outside the measured work.
#if defined(_WIN32)
#define G01_CPU_PLACEMENT_EXPORT __declspec(dllexport)
#else
#define G01_CPU_PLACEMENT_EXPORT
#endif
extern "C" G01_CPU_PLACEMENT_EXPORT int g01_cpu_resident_placement_prefix(int value) {
  return value * 17 + 3;
}
