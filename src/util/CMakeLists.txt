# util: foundational primitives (conventions §6/§8). Most are header-only inline
# templates/macros -- ref_counted (intrusive Ref<T>), scope_guard (HYLE_DEFER),
# type_name, registry, random, spin_lock. demangle needs a compiled unit (per-
# platform __cxa_demangle / UnDecorateSymbolName), so the module is now a real
# library rather than INTERFACE. On Windows demangle pulls in dbghelp (linked via
# #pragma comment(lib) in the .cc).
hyle_add_module(util SOURCES demangle.cc)
