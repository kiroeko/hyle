# Original narrow IQ1 Metal carrier

Mathematical origin: Anemll/anemll-flash-llama.cpp
`a809594fd2035979ad2b7ef23398127c1d07e891`, MIT.
`ggml-metal.metal` provides the three IQ1_XS/XXS/XXXS matrix-vector bodies
and original grouped-expert template instances. `ggml-common.h` supplies
their1024/512/256 GPU grids; entries match the qualified original Unsloth
CUDA grids exactly. `origins.json` seals borrowed and adapted fragments.

The existing ggml456172ec split source receives these exact bodies/tables.
Only the original extra-buffer-offset macro is removed: this encoder already
binds base offsets through Metal buffer arguments. Host tune constants and
dispatch are the original ones. New types select their existing MV carrier
instead of the pin's absent MM carrier. No mathematical instruction, grid,
HYLE IQ1 algorithm, Flash-MoE engine or slot/file loader is added.
Core enums/blocks/host codec remain the already registered original Unsloth
port. Apple-only builds do not enable the CPU backend. Source/runtime,
numeric/resource and all three delivery qualifications are separate gates.
