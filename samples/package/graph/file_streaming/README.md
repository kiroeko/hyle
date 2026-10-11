# File/resource streaming consumer

This independent C++ project uses the ordinary Graph product and selected Compute routes.
The small consumer streams two BF16/encoded-weight groups through fixed staging, calls existing
Direct and Graph mathematics as each group becomes ready, writes the resources to a new temporary
file, reloads, and checks the mathematics again. CPU/CUDA/Vulkan use existing GGUF Q8 bytes;
Apple uses its original affine Q4 representation. These fixtures do not claim full model inference.

Use `HYLE_SAMPLE_SOURCE` for `add_subdirectory` consumption, or the installed Hyle package through
`CMAKE_PREFIX_PATH`. Select routes with `HYLE_COMPUTE_ROUTES`; do not select internal factors/providers.
The program owns its temporary files and leaves the source model unchanged.

The explicit local-weight program takes an official-tool checked range manifest:

```text
python prepare_qwen.py <local-model.safetensors> <output-directory>
hyle_file_streaming_qwen <output-directory>/ranges.tsv
```

The mature GGUF adapter creates a deterministic mixed BF16/F32/Q8 fixture, checks the
original offsets/encodings/bytes with `gguf.GGUFReader`, and feeds the same small consumer:

```text
python prepare_gguf.py <fixture-directory>
hyle_file_streaming_sample <fixture-directory>/gguf-ranges.tsv
```

Use a second path argument to `prepare_gguf.py` to check an existing matching fixture.
This adapter requires the mature `gguf` Python package. It is a bounded format-consumer
example, not a model loader. Apple continues to use its original affine Q4 fixture.

The Python adapter requires official `safetensors`, validates metadata and reads only the three
layer-0 BF16 projections and F32 norm (about 72 MiB). It does not download weights or hash the
whole checkpoint. The C++ consumer reads those original ranges using the pipeline, compares
existing Direct/Graph mathematics against independent references, writes a new temporary file,
compares every stored original byte in bounded chunks, then reloads and repeats the mathematics. The consumer uses each selected
route's original Direct/Graph interfaces. Windows CPU/CUDA/Vulkan and Air CPU/Apple/Vulkan source
roundtrips have passed. Apple retains dense original Metal input and forms the transposed view
inside the original native Direct region or Graph view node. These programs are under
F3 qualification, so this README does not claim source/static/shared or all-route acceptance.

Cancellation and deadlines stop later chunks; already accepted file/native work must retire.
They cannot safely interrupt a blocking file operation. Native contexts, queues, resources and
injected allocators outlive every retained result/use. Stage bytes and old result capacity are
bounded separately; driver-private residency is not claimed zero or a process-wide memory bound.
