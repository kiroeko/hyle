# util-runtime: demangle, Registry and the type-erased C adapter. The independent
# hyle_util_core INTERFACE target owns ref_counted/scope_guard/type_name/random/
# spin_lock and is a PUBLIC dependency because Registry and the C adapter expose
# or consume those capabilities. Registry also publicly exposes
# error Status, mem allocator/region, and container string/hash/index templates,
# so those targets remain PUBLIC dependencies (important for static consumers).
hyle_add_module(util SOURCES demangle.cc c_util.cc)
