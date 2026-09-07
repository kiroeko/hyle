# util-runtime: demangle and Registry. The independent
# hyle_util_core INTERFACE target owns borrow/function_ref/ref_counted/
# scope_guard/type_name/random/spin_lock; Registry no longer consumes its old
# static-registration/default-new
# helpers. demangle's owning std::string convenience stays header-inline and
# reaches the runtime through a narrow internal sink, so STL ownership never
# crosses the DLL boundary. Registry publicly exposes error Status, mem
# allocator/region, and container string/hash/index templates, so those targets
# remain PUBLIC dependencies (important for static consumers).
hyle_add_module(util_runtime SOURCES demangle.cc)
