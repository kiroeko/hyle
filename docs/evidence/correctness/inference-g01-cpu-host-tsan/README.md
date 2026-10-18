# G01 CPU 驻留 attention 宿主 TSan

不可变 candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`。Windows canonical 输入单向同步 WSL2 验证副本，复用
原 CPU-only GCC14.2 构建树。[receipt](receipt.json) 与 [raw.zip](raw.zip) 保留实际命令、
日志、输入同步收据和两个变更 TU 的 Ninja compdb 插桩证明。

四个实际 attention tests 通过：两 layout 与独立 value depth、F32 长 KV/D256、
F32/F16/BF16 容量 padding/源只读/重放，以及非法 depth stride/overlap/空 view。
`TSAN_OPTIONS=halt_on_error=1；setarch x86_64 -R`，实际退出码 0。47 个 dependency/变更 TU 步骤以 4 jobs 完成，
build 22.01s、run 2.53s。

本记录只签上述 CPU attention 风险域，不代签全部 CPU 算子、Graph 生命周期、GPU/driver、
全模型或其他 sanitizer。CPU 普通 SDK、同合同 P1 与终审仍待；整个 G01/G02 和 goal 未收口。
