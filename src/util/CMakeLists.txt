# util-runtime: demangle, Registry and the type-erased C adapter. The independent
# hyle_util_core INTERFACE target owns ref_counted/scope_guard/type_name/random/
# spin_lock and is a PUBLIC dependency because Registry and the C adapter expose
# or consume those capabilities. Registry also publicly exposes
# container::string_ref/SwissTable-backed templates, so hyle_container remains a
# PUBLIC dependency (important for static consumers).
hyle_add_module(util SOURCES demangle.cc c_util.cc)
