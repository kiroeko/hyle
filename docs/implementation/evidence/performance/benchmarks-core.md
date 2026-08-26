# Core 性能证据

> **状态：`HISTORICAL / UNQUALIFIED`。** Core 正处于 v2 `OPEN` 复审，没有当前性能证书，也不
> 宣称模块已达到 SOTA。当前施工记录见
> [`core-rebuild`](../../tasks/core/core-rebuild.md)。

## 能力与证据形状

Core 只保留三个非 owning、非领域化的基础能力族，它们不能共用一个纳秒表来签证：

| 能力族 | 合适的证据 | 当前资格 |
|---|---|---|
| 产品身份 | complexity、code size、冷路径正确性与 package journey；不制造 getter 竞速 | 源码上 O(1)，ownership 与 linked API 仍待裁决 |
| compiler/binary 合同 | 各受支持 toolchain 的 codegen/negative compile checks，加真实消费 kernel | 有旧 object inspection，未绑定 v2 candidate 与完整消费者账本 |
| cache layout 与 prefetch | footprint/contention 分析，加拥有访问模式的 container/mem/sched/SIMD workload | 单独计时 hint 无资格；真实消费者尚未重签 |

intrusive list 已迁到独立的 `container_intrusive` 组件；它的历史数据与后续资格计划见
[`benchmarks-container-intrusive`](benchmarks-container-intrusive.md)，不得再用来证明整个 Core。

## 为什么旧绿色证书无效

- 尚未冻结 v2 candidate commit/tree；
- 三个能力族与根级文件的 ownership、语法消费者和语义重复账本尚未完成；
- compiler/cache 行为需要绑定当前工具链和真实 owning consumer；
- list microbenchmark 从来不能证明 version、visibility、compiler 或 cache 合同；
- 当前 Core 的 Native FFI 产品要求为 `N/A`，旧证据却混入 provisional exports 与 managed smoke。

这表示旧证据不能支持当前 claim，不表示所有旧测量都错误。Git 历史仍保留原始运行，供定位回归。

## V2 资格计划

1. 先冻结能力 ownership、API 与 candidate identity，再开始测量；
2. compiler/macro 以 codegen、negative compile contract 和真实热 kernel 为主，不给单条 intrinsic
   制造“更快”排行榜；
3. cache layout/prefetch 只在拥有该访问模式的真实 container/mem/sched/SIMD workload 中判断收益、
   footprint 与回退；
4. Windows、Linux、macOS 都绑定 compiler、flags、source、binary 与 raw evidence；对亚纳秒结果做
   顺序、placement/code-shape 与跨进程扰动；
5. `LOSE`、`UNRESOLVED` 或 shape-sensitive 结果先调查机制，不为 benchmark 修改公开 API 或常态
   metadata；
6. source/static/shared 的正常 C++ 消费 journey 与正确性证据必须绑定同一 candidate。

完成前唯一有效结论是：**Core 性能尚未取得 v2 资格。**
