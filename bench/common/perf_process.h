#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace hyle::bench::perf {

struct ThreadQosState {
  const char *name;
  const char *ok;
};

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
