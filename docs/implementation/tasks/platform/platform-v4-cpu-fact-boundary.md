# Platform v4：CPU 事实与执行 target 权威分离

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由已关闭 Platform 的新产品事实触发。当前仅登记 `REOPENED` 证据；完成有限能力审计和
> closure card 以前，不授权修改产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / Platform /
  `platform-v4-cpu-fact-boundary` / `REOPEN`；
- baseline：`66dc97620d34dd05b566fdbc359f2a28ad981447`；前轮：
  `platform-v3-capability-factorization`；
- trigger：`PLATFORM-CPU-FACT-TARGET-AUTHORITY-2026-09-05`；
- `PLAT4-R1`：公开注释把 `platform::Isa/best_isa()` 定义为主机 CPU 能运行的最高 ISA，但 x86
  实现实际复用了 Highway target group：所谓 AVX2 还要求 FMA、F16C、LZCNT、BMI1/2 与 SSE4.2，
  AVX-512 又要求 Highway AVX3 的完整 group。它是 provider/codegen policy，不是单纯 CPU/OS 事实；
- `PLAT4-R2`：SIMD 通过 `using Isa = platform::Isa` 同时把该类型用于实际 compiled family target，
  而 `simd::provider::Target` 又描述 implementation TU target。一个公开类型因而同时承担 host fact、
  product selection 与 provider body 三种语义，已经击穿前轮的唯一 authority 和自然 API 结论；
- `PLAT4-R3`：当前 llama.cpp `427291b5b34cd914a31b3fd3b61a68f6184f4b9f` 的随仓 ggml tree
  `9bf0d932917ba3ccfe7e3d9fd1723390dc870ec0` 用 ARM DotProd/FP16/SVE/SVE2/I8MM/SME/SME2
  与 x86 AVX512-VBMI 等运行时事实选择真实推理 backend；Platform 现有公开事实缺 DotProd、ARM
  FP16、SSE3、AVX512-VBMI，且 macOS/Windows ARM 对已经公开的 SVE/SVE2/I8MM 等字段未探测；
- `PLAT4-R4`：当前锁定 Highway 参考 `63f4b16cda024e2adff8a18dee81e067f8a87d61` 也明确把
  raw CPUID feature bits 与 target groups 分开。该证据支持分离事实与策略，不授权把 Highway 的完整
  API 并集复制进 Platform；
- 初步受影响面：`include/hyle/platform/cpu.h`、隔离 probe TU、CPU tests、Platform 用户/架构文档，
  以及直接使用旧 `Isa/best_isa/supports` 的 SIMD 与 Container；`platform_thread/platform_fs` 的产品
  源码不受影响，只回放 Platform façade 交付；
- 初步性能级别：`P0`。探测仍只允许一次有界初始化，随后 feature query/membership 为 O(1)；本轮
  不改变 Parker、filesystem、topology、memory、VM 或 NUMA 算法，也不重跑与 diff 无关的 P1 数字合同；
- 初步非目标：新增 kernel 或 SIMD provider、修改现有算法 winner、device inventory、scheduler
  policy、异步 I/O、Native FFI/C API、稳定 C++ ABI、公共性能仪器和 Formal 认证。

只读 replay verdict 为 `PRODUCT_TRIGGER`。下一步必须冻结 CPU feature-set 的自然公开形状、有限特征
清单、三平台探测合同、直接 consumer 迁移和验证预算，再进入 `SCOPED`；当前没有产品修改授权。

