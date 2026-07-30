# util installed-package probes

These focused projects verify that an installed shared or static HYLE package
can resolve `COMPONENTS util`, propagate its public dependencies, compile both
the C++ Registry/demangle surface and the pure-C adapter, and stage runtime
libraries without relying on the HYLE source tree.
