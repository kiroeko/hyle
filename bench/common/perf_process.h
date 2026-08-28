#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace hyle::bench::perf {

struct ThreadQosState {
  const char *name;
  const char *ok;
};

struct ThreadAffinityState {
  const char *topology;
  const char *cpu;
  const char *ok;
};

// Multi-thread contracts let the launcher constrain the process to a reviewed
// CPU set, then ask each benchmark role to pin itself inside that set. Process
// affinity alone is insufficient: Windows may place both threads on one CPU,
// which changes a park/unpark experiment by an order of magnitude. The role
// assignment is setup-only and its exact values are emitted as child telemetry.
inline ThreadAffinityState configure_thread_affinity(
    const char *environment_name) {
  const char *requested = std::getenv(environment_name);
  const char *topology = std::getenv("HYLE_PERF_SAMPLING_TOPOLOGY");
  if (requested == nullptr || requested[0] == '\0')
    return {"external", "external", "not-applicable"};
#if defined(_WIN32)
  char *end = nullptr;
  const unsigned long cpu = std::strtoul(requested, &end, 10);
  if (end == requested || *end != '\0' || cpu >= sizeof(DWORD_PTR) * 8)
    std::abort();
  const DWORD_PTR previous = ::SetThreadAffinityMask(
      ::GetCurrentThread(), DWORD_PTR{1} << cpu);
  if (previous == 0) std::abort();
  return {topology != nullptr ? topology : "instrument", requested, "1"};
#else
  // The v1 cross-core launcher currently admits Windows only. Fail closed if a
  // future runner injects the role contract on a backend that cannot prove it.
  (void)topology;
  std::abort();
#endif
}

// Darwin has no supported hard P/E-core affinity API. Every benchmark using
// the common qualification runner therefore selects and reports an explicit
// QoS class in the child process. Other platforms report N/A and rely on the
// launcher affinity/priority policy recorded in the evidence manifest.
inline ThreadQosState configure_thread_qos(
    const char *environment_name = "HYLE_PERF_APPLE_QOS") {
#if defined(__APPLE__)
  const char *requested = std::getenv(environment_name);
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char *name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr, "%s must be interactive or utility\n",
                 environment_name);
    std::abort();
  }
  if (pthread_set_qos_class_self_np(qos, 0) != 0)
    std::abort();
  return {name, "1"};
#else
  (void)environment_name;
  return {"not-applicable", "not-applicable"};
#endif
}

// The qualification runner assigns a different opaque token to every child
// process.  Manual benchmark launches remain usable, but their telemetry is
// deliberately distinguishable from evidence collected by the runner.
inline const char *run_nonce(
    const char *environment_name = "HYLE_PERF_RUN_NONCE") noexcept {
  const char *value = std::getenv(environment_name);
  return value != nullptr && value[0] != '\0' ? value : "unmanaged";
}

} // namespace hyle::bench::perf
