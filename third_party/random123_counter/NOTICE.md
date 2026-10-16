# Random123 Philox counter core

Origin: D. E. Shaw Research Random123 v1.14.0,
revision `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`,
`include/Random123/philox.h`. The original BSD-3-Clause license is in LICENSE.

HYLE ports the fixed Philox4x32-10 equations/constants into a small C++/CUDA
core, and the same integer equations into its native Metal/GLSL sampling
programs. It does not ship the upstream generator classes, portability
framework, distributions, or stateful seed/counter policy.

The versioned HYLE numerical schema defines key64/draw64/word addressing.
Official known-answer vectors and independent public-path corpus comparisons
are required for every qualified carrier. Categorical numerical transforms,
checks, native completion and storage are HYLE's adapter responsibilities.
