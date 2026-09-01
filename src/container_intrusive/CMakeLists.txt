# Allocation-free intrusive containers form an internal header-only factor.
# Low-level HYLE factors such as time can consume it without pulling the
# compiled Container product and its mem/platform dependency closure; ordinary
# users select and link only hyle::container.
hyle_add_module(container_intrusive)
