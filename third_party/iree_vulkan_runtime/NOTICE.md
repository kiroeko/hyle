# IREE Vulkan resource integration

Original runtime: https://github.com/iree-org/iree at
`6ef606b6fa3e12bd2c12d36a07463069eb71185f` (Apache-2.0 WITH LLVM-exception).

The exact patch manifest applies only to authenticated build-local source.
It adds synchronous external queue ownership and actual VkDeviceMemory
allocation/free accounting hooks, retaining original allocators, pools, queue
progress and numerical programs. The bytecode build generator uses CMake's
selected Python interpreter instead of a process-wide `python3` alias.
No original source cache or compiler mathematics is changed.

Build integration privately renames the original Vulkan header targets to allow
composition with the separately locked HYLE/ggml headers. Offline parameterized
programs preserve the qualified original scan, conversion, reduction and MatMul
recipes. Their byte identities are in programs/manifest.json; no compiler or
Python is required by an installed C++ consumer.

External queue lookup preserves the actual VkDeviceQueueCreateFlags supplied
by its owner, including internally synchronized queues. This changes resource
wrapping only; the original scheduling and math remain unchanged.

The external wrapper (including distinct logical lanes on one native queue) exposes original stop/drain completion through
a retained HAL semaphore. No new worker, scheduler or math is introduced.

The POSIX semaphore completion path clears its operation tracker reference
before the final callback, which may free the operation under the original
async ownership contract. This fixes a reproduced host use-after-free without
changing numerical programs or adding a new execution mechanism.
