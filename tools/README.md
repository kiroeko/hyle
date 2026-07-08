# HYLE shader tooling

`hyle_shader.py` turns shaders into something HYLE's Vulkan backend can load. Vulkan does not
run GLSL directly — it consumes **SPIR-V** (a portable IR). Turning your GLSL into SPIR-V, and
optionally into a C header you compile into your program, is what this tool is for.

> The final GPU machine code (device ISA) is produced by the **driver at runtime**
> (`vkCreateComputePipelines`), not by any offline tool — that is how one SPIR-V module runs on
> any vendor's GPU. To avoid re-compiling ISA every cold start, persist the driver's pipeline
> cache (see `HYLE_VK_PIPELINE_CACHE` in the compute-06 docs). SPIR-V is as far as offline tools go.

## Two subcommands

```
hyle_shader.py compile <in.comp> -o <out.spv> [--stage comp] [--glslang PATH]
hyle_shader.py embed   <in.spv>  -o <out.h> --name NAME [--namespace NS]
```

- **compile** — GLSL → SPIR-V. Shells out to a `glslang`/`glslc` executable (it does **not**
  reimplement the compiler). The compiler is located via `--glslang <path>`, else the
  `HYLE_GLSLANG` env var, else `glslc`/`glslangValidator` on `PATH` (e.g. from your Vulkan SDK).
- **embed** — SPIR-V → a C header defining an `inline constexpr hyle::compute::SpirvBlob`. Pure
  Python, no dependencies. `--name` sets the symbol; `--namespace` wraps it so many shaders
  don't clash. `inline constexpr` means the data is folded to **one copy** across every
  translation unit that includes the header (no per-TU bloat).

`compile` can chain into `embed` in one call with `--embed <out.h> --name NAME`.

## Three ways to get a shader into HYLE

### 1. You have a `.spv` already → embed it (compile-time, no runtime file IO)
```
hyle_shader.py embed my_kernel.spv -o my_kernel.spv.h --name my_kernel --namespace myapp
```
```cpp
#include "my_kernel.spv.h"
auto k = hyle::compute::Kernel::load_spirv(dev, myapp::my_kernel);  // one line
```

### 2. You have GLSL source → compile then embed (full offline pipeline)
```
hyle_shader.py compile my_kernel.comp -o my_kernel.spv --embed my_kernel.spv.h --name my_kernel
```
Then `#include` and `load_spirv` as in (1). Use this when you keep shaders as `.comp` in your repo.

### 3. You want runtime loading (hot-swap, user-supplied shaders) → no tool, no header
```cpp
std::vector<char> bytes = read_file("my_kernel.spv");   // your own IO
hyle::compute::SpirvBlob blob{reinterpret_cast<const unsigned*>(bytes.data()),
                              static_cast<unsigned>(bytes.size())};
auto k = hyle::compute::Kernel::load_spirv(dev, blob);  // same call as (1)/(2)
```
`SpirvBlob` is just `{const unsigned* words; unsigned byte_size;}` — point it at any SPIR-V
bytes you have, embedded or read at runtime. The loading strategy is **your** choice; HYLE does
not force a fixed "there must be a file here" flow.

## Specialization constants

`load_spirv` has an overload taking `SpecConstant` entries + a value blob (checklist #2/#3):
shader constants baked into the pipeline at build time (workgroup size, loop bounds, tile
sizes). See `include/hyle/compute/kernel.h` and the `SpecializationConstants` test.

## For HYLE's own build

The `hyle_embed_shader()` CMake function (cmake/HyleModule.cmake) runs this tool at build time to
regenerate the test-shader headers from `tools/shaders/*.comp`, using the glslang built via
FetchContent. Requires Python3 (interpreter only). This is how the checked-in-style headers stay
reproducible from source rather than hand-pasted SPIR-V words.
