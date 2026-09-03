# Util façade package probe correction

> Record: `COMPLETED BUILD FIX`, 2026-09-03. This task changes a package
> conformance fixture, not a product factor, public API, dependency edge or
> module closure state.

## Finding

The selected-component smoke for `util` configured the correct factor closure:
`util_core`, `util`, and the transitive factors required by the compiled Util
runtime. The installed consumer fixture nevertheless required the unrelated
`container_intrusive` detail target. That factor belongs to the Container
product façade and is not a dependency of the ordinary `container` factor used
by Util. The stale assertion therefore rejected the intended minimal package.

## Resolution

- Keep checking that neither internal factor is advertised as a user product.
- Require the `util_core` detail target because it is a direct Util façade root.
- Require `container_intrusive` to remain absent from a Util-only package, so a
  future accidental over-selection fails mechanically.

The source, relocated shared and relocated static Util consumers are replayed
after this correction. Product-module state remains unchanged.
