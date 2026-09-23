# M2-B5 成熟 Provider 稀疏注意力与压缩状态批次

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- round / owner：M2-B5；`/root`。
- baseline：`d8b330bb420e4d5439783e79eabd8aaa3d508d44`；[B4 已关闭](llm-moe-m2-b4-terminal.md)，
  Windows canonical 工作树在本卡登记前干净。
- trigger：F15.1--F17.3 的 MLA、DSA learned index、跨层 selection、pool/tail 与压缩状态消费。
  B2 已提供 attention/KV/RoPE，B4 已提供模型专用 region seam；公共层尚不能表达锁定 ggml 四后端
  已实现的 `LIGHTNING_INDEXER`。
- stop：只把审计后准入的成熟 sparse/index region 接入普通 Direct/Graph，并证明其资源、完成与两次
  replay；MLA、pool/tail 与 compression 若可由已有成熟 MatMul/Attention/Gather/TopK/Pool/状态端口
  组成，就登记复用边界，不复制算法。新增 HYLE 数学 kernel/shader/SIMD 算子为 0。

## SCOPED owner 与有限边界

| owner | 本批允许审计/修改 | 本批禁止 |
|---|---|---|
| `compute_ops` | 增加 DSA Lightning Indexer typed descriptor、稳定 identity、自然 shape/dtype 验证 | 模型 loader、稀疏策略、压缩算法 |
| CPU | ggml CPU 原版 indexer retained recipe；已有 MatMul/Attention/Gather/TopK/Pool 与 Graph state ports 只作组合证明 | host sparse loop、自研 compressor |
| CUDA | ggml CUDA 原版 indexer 与 native stream/device buffer；已接 attention Provider 继续按冷规则选择 | 新 `.cu`、运行时竞速、CPU fallback |
| Apple | 复用 B4 有限泛化的 ggml Metal whole-region seam；普通 MLA/状态组合继续由 MLX | 新 Metal shader、拆成熟 region |
| Vulkan | ggml Vulkan managed indexer region，原版 backend capability 作最终资格 | 新 SPIR-V、相似 shape 代签 |

共享路由 matcher、Mem、公共构建/性能仪器和冻结自研算子不在预期 FIX。模型 identity、平台、硬件、
驱动/SDK、路线、量化档位与 Provider 版本仍在 prepare 冷绑定；热执行不搜索、不计时。

## SCOPED 验收

1. Lightning Indexer 固定自然布局：query `[S,T,H,D]`、key `[S,K,1,D]`、prescaled weights
   `[S,1,T,H]`、F16 mask `[Sm,1,T,K]`，输出 `[S,1,T,K]`；第一批跨四路线公共域固定
   `D=128,H=64`、dense F32 query/key/weights、dense F16 mask。
2. 同一独立 oracle 覆盖非零 mask、多 token/key、正负 dot 的 ReLU、stream mask broadcast；非法
   dtype/shape 在准备前拒绝，两个 invocation/replay 不共享可变结果。
3. F15 的 absorbed MLA/gated NoPE 复用已有 MatMul/Attention/View；F16.2 复用显式 graph ports，
   F16.3 复用 Pool/TopK/Gather；F17 复用 MatMul/Conv/Attention 与显式 state ports。只有完整成熟
   region 的现成入口才允许加入本批，不为表中每行造 adapter。
4. 四路线各一次增量构建和一组 Direct/Graph native acceptance；Apple shared/static 各一次。P0 核对
   packing、device residency、同步、retained program 与重复执行。没有冻结完整 sparse/compression
   fragment P1 card 时保持 `NOT_APPLICABLE`。
5. 完整 checkpoint 的 MLA、sparse selection schedule、unpooled tail 与增量 compressor 属于 M3-05
   真实模型链验收；未运行保持 `UNPROVEN`，不能用基础算子存在写 PASS。

完成上游四后端入口、固定限制与已有组合能力审计后，本卡须写入有限 `KEEP/FIX/BORROW/DEFER/REJECT`
表并转为 `AUDITED`，之后才能修改产品代码。
