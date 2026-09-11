# HYLE 性能定位与可证明价值

> 决策状态：`ACCEPTED`，2026-09-11。本文回答 HYLE 以什么性能价值与更成熟、功能更多的库竞争。
> 怎样测见[性能证据标准](../developer/performance/benchmark-evidence-standard.md)，一次测量结果属于
> [implementation evidence](../implementation/evidence/performance/README.md)；当前全库结论见
> [性能总账](../implementation/evidence/performance/full-library-scorecard.md)。

## 为什么不能只说“全库 SOTA”

基础能力没有脱离语义和机器的单一速度。allocator、队列、HTTP、SIMD 和文件映射分别存在不同的
ownership、失败、进展、规模、ISA、冷热与资源合同；一个实现可以在小对象延迟上赢，在大批吞吐或
峰值内存上输。成熟 peer 还常已接近 syscall、内存带宽、原子往返或最佳渐进复杂度的下界。

因此，HYLE 不以“每个微基准都赢”或一个跨模块平均分为产品承诺。可证明的性能主语必须限定为：

```text
(capability, semantic contract, workload distribution,
 hardware/OS/toolchain, delivery shape, candidate)
```

这不降低目标。HYLE 仍要求所有热能力采用同档最佳复杂度，不隐藏稳定劣势，并在真实组合路径上形成
成熟单库难以取得的系统收益。

## 用户为什么选择 HYLE

HYLE 的差异化来自三层，可分别验证：

1. **广度复用**：成熟 provider 已经优秀时，使用其锁定、可升级实现，例如 Highway/xsimd、mimalloc
   或协议/密码学 provider；HYLE 不为品牌重写一份较差版本。
2. **窄点胜出**：HYLE 自实现或静态路由只保留在具名平台、ISA、规模和合同上有可靠优势的路线；
   无法分辨时选择成熟 provider，不为保留自研代码制造运行时分派税。
3. **跨模块组合**：统一 allocator、buffer ownership、scheduler、cancel/error、IO completion 和
   topology authority，减少一次真实任务中的复制、分配、唤醒、syscall、状态机与线程池。模型加载、
   数据准备、请求调度和流式服务等纵切才是这层价值的最终证据。

只有第一层，HYLE 通常只是“同等性能的统一交付”；只有第二层，用户仍可逐个选库。HYLE 真正可持续
的竞争理由是：以第一层取得能力广度，以第二层保留明确局部胜项，再由第三层把这些机制组成更少税的
自然 C++ 路线。若当前证据还没有证明第三层，应如实写“尚未证明”，不能拿原子 benchmark 代替。

## 方向、实用差距和改动是三件事

每个同合同、已资格化 workload 同时保留：

- **方向**：置信区间相对 ratio `1.0` 的位置；稳定在下方就是 HYLE `LOSS`，即使只有 1%；
- **实用等价/非劣**：是否仍在预声明的消费者容忍带内；它回答差距是否重要，不改写方向；
- **产品行动**：只有找到实现机制，并能证明它传播成真实消费者的 material 影响，才修改 API、layout、
  常态 metadata 或依赖。

所以“稳定慢 1%，但通过 5% non-inferiority”是合法且诚实的双重事实；不能缩写成 `PARITY`，也不自动
授权花一天追逐它。方向翻转、区间跨 1 或 A/A 同量级偏移写 `INCONCLUSIVE`。只有没有固定方向且通过
双边 practical-equivalence，或生成代码/执行路径已证明等价时，用户级总账才写 `PARITY`。

现有统一仪器 v1 的 `exact_verdict` 历史字段是**相对预声明 practical band 的工程分类**，名称不可当作
ratio `1.0` 的方向结论。当前总账必须从其逐 stratum ratio/CI 另行导出方向，不重写旧 evidence。

## 全库总账与 SOTA 声明

全库性能页面必须覆盖每个 `CLOSED` factor，但计数按独立 `(workload, peer, contract)` 去重。每项至少
记录 comparison class、真实来源、规模、平台/ISA、交付、绝对值、方向、practical gate、P0 成本结构、
根因或未知、消费权重和证据身份。P0、无合理 peer、未覆盖 API、过期 artifact 与平台缺口不能补成平局。

允许的公开表述分三档：

- **SOTA route**：在限定合同和范围内，对当前最强合理 peer 有合格、可复核的固定优势；
- **competitive route**：没有 material regression，但方向可能平、未知或存在已披露的小幅 loss；
- **unproven**：只有 P0、证据过期、平台/规模不足或没有同合同 peer。

模块名、代码量、胜项数量或非劣门通过都不能单独产生 SOTA。每个 SOTA route 还要有正交 holdout 或
真实消费者相关性，避免只对开发集优化。产品首页可以总结已证明的路线，但必须同时链接固定 loss 和
unknown map。

## 优化顺序与停止规则

1. 先确认功能和自然 API，不让 benchmark 设计产品；
2. 先看复杂度、allocation/copy/syscall/wake/contention/cache/vectorization 和源码完成边界；
3. 只对仍然重要且不可静态裁决的差距运行统一仪器；
4. 按消费者热度 × 可复现差距 × 证据强度 ÷ 修复成本排序；
5. 高权重固定 loss 必须修复、说明必要语义税，或登记具名未知；低权重小 loss 可以停止，但不能消失；
6. candidate 只运行一次适用 P1/P2；文档、状态和格式变化不重跑。

提升来自更好的算法/复杂度、按 ISA/规模的编译期或一次性路由、batch/fusion、更少搬运和跨模块组合，
而不是反复改变函数地址、测试顺序或只对固定输入展开。若同行已在硬件下界，达到同路径同 codegen 的
parity 就是正确结果；HYLE 应把研发预算移到仍能传播到真实工作负载的层。
