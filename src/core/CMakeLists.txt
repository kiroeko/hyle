# Core contains only compile-time source/layout primitives. A version-only
# runtime would diagnose itself rather than the actual HYLE artifacts, so the
# component is intentionally header-only.
hyle_add_module(core)
