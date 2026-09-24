# The implementation was retired by M3-D4.  Keep only an internal absent
# target so the append-only component/status identity remains auditable; no
# product facade or route depends on it and it is never exported.
hyle_add_module(compute_provider_marlin OPTIONAL_ABSENT)
set_property(TARGET hyle_compute_provider_marlin PROPERTY
  HYLE_INTERNAL_ONLY_ABSENT TRUE)
