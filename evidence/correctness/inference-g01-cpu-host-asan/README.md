# G01 CPU 驻留 attention 宿主 ASan/LSan

candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，Windows canonical输入只单向
同步WSL验证副本，2013项必要输入核对、42项差分复制，保原树/cache。[receipt](receipt.json)
和[raw.zip](raw.zip)记录真实commands、输入、原FAIL及Ninja compdb插桩证明。

原CPU ggml与layout TU的地址插桩实际覆盖两种layout、独立value depth、F32长KV/D256、
F32/F16/BF16容量cache padding/源只读/重放、非法depth stride/overlap/空view。四个实际
attention tests通过，ASAN_OPTIONS=halt_on_error=1:detect_leaks=1，实际exit0。
不代签全部CPU算子/全模型、GPU/driver或其他san模式。

旧树47个实际dependency/改动TU步骤4路重编，没有全库/D5模型回放。私有CMake hook
最初在DEFER add_subdirectory、随后project启用CXX阶段递归失败，改在语言启用后的
project hook一次引入GTest；原FAIL保留，不改公共工具/候选产品、不重置原预算。

UBSan/TSan、普通CPU installed SDK、P1与终审仍待完成。Vulkan的三交付和宿主风险
另有对应证据，不由本CPU记录代签。整个G01/G02和goal仍未收口。
