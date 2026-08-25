# 调试案例与方法

这些记录来自已经处理过的棘手问题，用于快速识别相同症状、选择观测点并避免重复走弯路。
它们是维护经验，不是当前产品状态或性能证书。

- [io_uring DEFER_TASKRUN](defer-taskrun-reenable-hunt.md)
- [ggml 零拷贝空 buffer](ggml-zerocopy-null-buffer-hunt.md)
- [Highway Pack16 roundtrip](highway-pack16-roundtrip-hunt.md)
- [I/O pipelined hang](io-pipelined-hang-hunt.md)
- [WSL/9p 崩溃调查](ki-1-crash-hunt.md)
- [multishot recv awaitable](multishot-recv-awaitable-hunt.md)
- [SIMD benchmark 仪器伪影](op-parity-harness-artifacts-hunt.md)
- [when_any loser UAF](when-any-loser-uaf-hunt.md)
- [xsimd ctanh 极大虚部](xsimd-ctanh-extreme-imaginary-hunt.md)
- [xsimd ctanh 近极点](xsimd-ctanh-near-pole-hunt.md)
- [xsimd lgamma lane mixing](xsimd-lgamma-lane-mixing-hunt.md)

新案例应在开头写清症状、影响范围、最终根因、最短复现、诊断工具、修复和以后如何提前发现。
纯时间线或一次性命令输出放 implementation/evidence，而不是继续扩写这里。
