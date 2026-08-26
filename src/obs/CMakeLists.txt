# Structured logging is the compiled native surface. The trace macro seam is
# header-only and its optional Tracy backend is wired once by HyleTrace.cmake.
hyle_add_module(obs TRACE SOURCES log.cc)
