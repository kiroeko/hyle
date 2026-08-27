# util-runtime: demangle and Registry. The independent
# hyle_util_core INTERFACE target owns ref_counted/scope_guard/type_name/random/
# spin_lock; Registry no longer consumes its old static-registration/default-new
# helpers. Registry publicly exposes error Status, mem allocator/region, and
# container string/hash/index templates, so those targets remain PUBLIC
# dependencies (important for static consumers).
hyle_add_module(util SOURCES demangle.cc)
