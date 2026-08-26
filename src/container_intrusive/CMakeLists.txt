# Allocation-free intrusive containers form a standalone header-only product
# target.  Low-level users such as time can consume this capability without
# pulling the compiled container module and its mem/platform dependency closure.
hyle_add_module(container_intrusive)
