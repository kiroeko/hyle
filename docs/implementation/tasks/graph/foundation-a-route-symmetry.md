# Foundation A：四路线 Graph 语义对称性

## Closure card

- component / owner：`graph`、`graph_route_apple`、`graph_route_vulkan`、
  `graph_route_cpu_apple`、`graph_route_cpu_vulkan`；共同静态语义由 Graph 拥有，
  原生 artifact、端口、完成与资源由各路线拥有。
- baseline commit：c0792501319ad1eab482c41b054e1e63a3ca7b8a。
- trigger：用户明确要求实施四路线对称性；
  [固定源码研究](../project/foundation-f45-reference-research-2026-10.md)与
  [设计 A-01..04](../../../architecture/compute-state-paged-design.md)证明
  CPU+CUDA 复用共同 compiler，而 Apple/Vulkan 及 hybrid 拒绝全部 symbolic/control 图。
- 正常 consumer：同一有界变长张量管线，冷准备有限精确形状，独立 invocation 选择固定形状，
  显式控制前驱虽没有数据消费者也必须执行；不支持的 effect/alias 明确拒绝。
- 主纵切：F4/F5 显式状态消费者的 pure initial/final 与有限 batch shape。
  正交纵切：渲染资产变长矩阵管线的独立分支排序与旧结果保活。
- public journey：GraphBuilder 的已有 shape/control/effect 校验 → 冷 native artifact 表 →
  原路线 invocation/bind/prepare/submit → 原 Completion/ResultLease；不引入共同 native runtime。
- scope：仅 A-01、A-02、A-03、A-04。实际可支持的 effect 需由端口和 permission 证明；
  拒绝不能包装成支持，pure control 不得只删除检查。
- non-goals：F4/F5 产品代码、全引擎、自动 placer、运行时 JIT、padding/general 伪覆盖、
  新数学/copy kernel、万能 Queue、公共验证/构建/性能工具重写、重开 F1–F3。
- 依赖：既有 Graph/Compute/Mem/Sched 与锁定 MLX/ggml/IREE；不升级依赖。
  直接消费者：原 native 与 CPU+Apple/Vulkan 的普通 Graph consumers，CPU+CUDA 仅作合同比较。
- 性能：P0，检查冷表与 warm 路径、复制/等待/私有成本；本卡不运行 P1 或全算子目录。
- 机器：Windows CPU/CUDA/Vulkan，WSL CPU/CUDA 与 sanitizer，Air Apple/Vulkan/CPU；
  Apple 运行前核对真实身份，仅 MacBook Air。WSL 软件 Vulkan 仅构建与包证据。
- 预算：静态审计两轮；有限四 FIX；每个 gate configure 300 秒、build 1800 秒、
  test 120 秒、consumer 180 秒。相同 gate 的失败/超时不随目录或提交重置。
  重型 gate 串行；库 build 按可用内存 j4≥12GiB / j2≥8GiB / j1≥4GiB，test1。
- 预声明验证：selector 选择 actual owners/直接消费者；共同 shape/control/effect 边界与 OOM；
  exact coverage 缺口/外来 ID/重复 binding/derived guard/overflow、独立 slots、旧结果与 close；
  control-only 活前驱与跨 fragment 排序，禁止 host wait 的拒绝；实际支持 effect 的权限/alias/失败。
  受影响 native/hybrid 普通 source 与 installed static/shared、公开头、缺路线、搬迁、runtime；
  共同静态代码适用 ASan/UBSan，新增并发共享元数据适用 TSan；无新增并发机制不重复全库 TSan。
  每轮 docs/closure/diff，registry 明确 previous-ref，生成 status，不手改生成表。
- stop：预算到达或 native 无法满足所需语义时如实 BLOCKED；新系统性 API/owner 事实退回
  SCOPED/AUDITED；原本不准入的 effect 保留明确拒绝与限制，不冒充 A 完成。

## 当前阶段

SCOPED。尚未写产品代码；下一步审计原有 specialization、liveness 与 native ordering，
固定有限 FIX 的 API、ownership、成本与验证后才进入 AUDITED。

## 有限改动 ledger

| ID | 工作 | 当前结果 |
|---|---|---|
| A-01 | 共用既有静态合法性与身份，保留 native artifact/owner | UNPROVEN |
| A-02 | 冷有限 exact artifact 表与创建时 coverage 拒绝、独立 slots | UNPROVEN |
| A-03 | 真实 pure control liveness/顺序与禁止等待裁决 | UNPROVEN |
| A-04 | 真实支持 effect 的端口/权限/alias；其余明确拒绝与旧结果合同 | UNPROVEN |

## Gate

scope、architecture、correctness、delivery、usability_docs、red_team、engineering_performance
均未裁决；本卡建立不代表产品能力或 native 验证 PASS。
