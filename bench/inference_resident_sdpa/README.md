# Resident SDPA 性能适配器

这里提供有限 attention 消费者和独立参考。原始 Vulkan 实验对应产品 revision
`d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`；历史 JSON 和结果保留原身份，不代表当前实现性能。

Vulkan 参考从 ggml `456172ec733a135778adcd32d00e576a58232e45` 独立编译 F32 flash、mask、
conversion、split-K 和 mask-opt shaders，按原 recipe 的 strides/tuning 组装。双方使用相同
Program/Invocation admission、checked Completion 与输出归还；参考不调用产品 typed lowering。
私有生命周期适配要求 static source build。

预检形状是 F32 BSHD GQA8/2：Q1/KV1024/D256 causal decode、Q128/KV1024/D256 causal prefill、
Q17/KV257/D64 noncausal 留出。容量多 13 token、padding=128，double oracle 容差
`6e-6*abs(expected)+2e-6`。F16 helper 尚未验证；参考 tuning 只覆盖实际 NVIDIA 与 Apple 主机，
其他 vendor 明确拒绝。

在本目录的独立 CMake 工程配置静态 Vulkan 路线，提供 `HYLE_RESIDENT_REFERENCE_SOURCE` 和
`HYLE_RESIDENT_GLSLANG`；直接构建 `g01_resident_bench` 及按需 `_placement` 变体并运行。
这会配置相关产品，优先使用已有对应 build/cache，避免无关重编。

- `HYLE_BENCH_VERIFY_ONLY=1` 做数值与稳定重放检查。
- `HYLE_BENCH_AA_ROUTE=native|selected` 将两侧替换成同一路线；保留执行和分配顺序参数。
- 计时包括 ready-input bind、submit/wait、warm mask/conversion/split-K 和输出 owner 归还；
  host download 与独立 oracle 不在热计时内。冷 setup、驻留与峰值另外报告。
- VkDeviceMemory quota 不包含 driver/pipeline 私有分配，不能据此推断总显存。

保存原始输出，用[性能比较方法](../../docs/developer/performance/README.md)检查相同工作量、A/A、
顺序、分辨率与完整样本。历史 contract 描述当时的实验，重测时记录本次配置与分析方法。

CPU 入口见 [cpu_p1](cpu_p1/README.md)。各 backend 的 encoded/page/ragged 子目录保留其具体输入
和参考实现。该目录的有限片段不能推断完整引擎 token/s、其他精度或所有形状的结果。
