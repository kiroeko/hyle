# SIMD-10：Bitpack 与 random

- ticket state：`QUEUED`
- kind：product capability/correctness
- depends on：SIMD-09
- budget：1 工程日；性能预算 60 分钟

审查固定 wire layout 的 u8/u16/u32 bitpack、1..lane bits、VL 独立性、overlap/size，以及 caller-owned
non-crypto RNG state、determinism、jump/split/fill。协议 framing、压缩格式、entropy 和密码 RNG 不属于
本票。P1 只测 u16 11/13-bit 与代表性 f32 fill；优先检查 bytes moved、额外 pass 和隐藏状态。
