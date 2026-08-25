# obs-01/02 historical closure: structured logging plus a pre-release FFI schema
# candidate. The trace macro seam is header-only and its optional Tracy backend
# is wired once by HyleTrace.cmake.
hyle_add_module(obs SOURCES log.cc c_obs.cc)
