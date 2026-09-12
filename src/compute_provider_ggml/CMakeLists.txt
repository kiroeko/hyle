# SCOPED audit-only factor.  The empty target makes the component identity and
# selection contract executable without pretending that a ggml provider or a
# dependency has already been admitted.  Product code is forbidden until the
# registry advances to AUDITED.
hyle_add_module(compute_provider_ggml)
