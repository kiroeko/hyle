# util: foundational primitives (conventions §6/§8). Most are header-only inline
# templates/macros -- ref_counted (intrusive Ref<T>), scope_guard (HYLE_DEFER),
# type_name, registry, random, spin_lock. demangle needs a compiled unit (per-
# platform __cxa_demangle / UnDecorateSymbolName), so the module is now a real
# library rather than INTERFACE. On Windows demangle pulls in dbghelp (linked via
# #pragma comment(lib) in the .cc). c_util.cc is the stable C seam for runtime
# demangle + the complete type-erased util surface. Registry is a public native
# header and exposes container::string_ref/SwissTable-backed templates, so
# hyle_container is a PUBLIC dependency (important for static consumers).
hyle_add_module(util SOURCES demangle.cc c_util.cc)
