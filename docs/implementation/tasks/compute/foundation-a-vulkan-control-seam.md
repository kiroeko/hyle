# Foundation A：Vulkan 冷控制边接缝

本轮有限接缝已在[最终收口](../graph/foundation-a-final-closure.md)CLOSED；以下为阶段事实。

## Closure card

- component / owner：`compute_route_vulkan`，仅冷 PlanStep/prepare_plan 的受控分片。
- baseline：4d0d892b903013a3dbc0e9f477db71d3c118e42a。
- trigger：用户授权的 [A-03](../graph/foundation-a-route-symmetry.md) pure control consumer。
  当前 Graph 只传 operation/input/output，prepare_plan 会把相邻 ggml nodes 融成 region，
  control-only 前驱可能没有 retained output；不能以原 ordinal 顺序冒充真实执行完成顺序。
- consumer / 主纵切：无数据依赖的前驱成功完成后才执行后继；正交资产管线相同 pure control 操作。
- scope：为冷 native PlanStep 传递 control boundary / forced materialized outputs；
  split 原 ggml regions，复用原 command barriers、managed completion、storage planning。
  不改变普通无控制图分片，不改变 F3 已签 file streaming 范围。
- non-goals：effects 新端口、数学 shader、Queue/runtime、公共构建/验证工具与全算子回放。
- 直接依赖/消费者：既有 ggml Vulkan/IREE、graph_route_vulkan；原锁不升级。
- P0：只在冷准备保存两个 bool；最终 artifact 不热遍历控制图。额外物化/等待按原报告呈现。
- 验证：与 A 卡同一实际 native/hybrid control/math/old result/cancel/close fixture；
  Win/Air source/static/shared、受影响公开 detail 头、缺路线/搬迁；WSL 软件 Vulkan 仅构建/包。
  原派发边界及 consumer-order 用真实 native evidence，不能让软件 driver 代签硬件格。
- 预算：一轮定向源码审计，有限一个 FIX；configure300/build1800/test120/consumer180 秒，
  失败不重置，重型 gate 串行，库按实际可用内存 j4/j2/j1，test1。
- stop：原 native 机制无法完成显式控制顺序则 BLOCKED/INCONCLUSIVE；系统合同改变退回 AUDITED。

当前 AUDITED；原 `foundation-f3-final-closure.md` 仅为原已签事实。

定向审计：vulkan_program.cc 相邻 managed_operation 合并；vulkan_region.cc 只把 last_uses≥end
或 published 输出交给原 ggml recipe。Graph 所需控制前驱需冷保留输出并在端点截断合并。
vulkan_pipeline.cc 的原 native record 在 dispatch 间建立 compute execution/memory barrier；
vulkan_run.cc 的原 managed/native stages 与 IREE continuation 负责完成/失败/实际退休。
本轮 BORROW 原机制，不更改队列、provider 或 GPU completion 模型；native 故障资格仍待测试。

冻结：PlanStep 末尾增加 control_boundary/preserve_outputs 两个冷 bool；默认 false 保留旧
aggregate 初始化与无控制图路径。原 prepare_plan 在 boundary 两侧截断 managed region，
preserve_outputs 只抬高原 last_uses，最终 commands/managed recipes 复用原执行路径。
无新 public Compute 用户对象/线程；普通 Graph 自身必须完成合法性与 liveness。
全图 typed ports/alias、取消/close 与 unknown storage 保持原 owner，额外 materialized
control outputs 如实计入现 storage facts，不声称零成本。

七问：原锁/许可证 KEEP，无 fallback；本有限复合控制旅程 SELF 冷 glue/BORROW 原 native；
不复制 barrier/runtime；P0 冷 O(N+E)、warm 原 commands；无新 FFI；现公开 detail 头/三交付
适用；Graph guide 解释 controlled materialization 与 managed waits。资格仍为 UNPROVEN。

## 有限 FIX

| ID | 工作 | 验收 |
|---|---|---|
| A-VK-01 | 冷控制端点分片与 forced outputs，原 barriers/完成排序 | control-only 前驱真实执行、失败阻止后继；普通图原分片与 native owners 不变 |

七门均 UNPROVEN，SCOPED 不代表资格。

Candidate：546b0d2468e5de29031c34668f4afa4bc31b9a17。A-VK-01 实现固定，Win native/hybrid
source 控制前驱/error 数学已签首批事实；Air Vulkan/installed/detail header 的剩余格待验。
产品修改必须先退回 AUDITED，不以文档/日志提交重置原 budget。
