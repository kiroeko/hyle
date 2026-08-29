# SIMD-10：Bitpack 与 random

- ticket state：`QUEUED`
- kind：product capability/correctness
- depends on：SIMD-09
- budget：1 工程日；性能预算 60 分钟

把当前“同 tier 才可解码且没有 format tag”的普通 bitpack 修成 canonical little-endian wire layout：
u8/u16/u32、1..lane bits、跨 x86/NEON/SVE/RVV/scalar 字节一致、VL 独立、明确 overlap/size。native quant
layout 与协议 framing/压缩格式仍由 Compute/格式 owner 持有。

同时完成 caller-owned non-crypto RNG state 的 seed/copy/advance/split/fill 生命周期，明确复制会复制流、
同一 state 非并发安全、worker 应持有独立 split state。entropy、密码 RNG 和分布采样不属于本票。P1 只测
u16 11/13-bit 与代表性 f32 fill；优先检查 bytes moved、额外 pass 和隐藏状态。
