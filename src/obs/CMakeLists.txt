# obs-01/02 closure: structured logging + stable C ABI. The trace macro seam is
# header-only and its optional Tracy backend is wired once by HyleTrace.cmake.
hyle_add_module(obs SOURCES log.cc c_obs.cc PUBLIC_DEPS hyle_error)
