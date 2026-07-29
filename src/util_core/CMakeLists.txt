# util-core is the low-level, header-only half of util.  Keeping it as an
# independent INTERFACE target makes the real public-header edges from
# mem/sched (and the private graph edge) visible without pulling the compiled
# util runtime and its container dependency back into the foundation.
hyle_add_module(util_core)
