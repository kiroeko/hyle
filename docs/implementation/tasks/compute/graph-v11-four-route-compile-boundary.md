# R2-G：既定规则的编译边界与单路线计算

<!-- PROJECT UNIT REVIEW: AUDITED -->

## Closure card

- owners：`graph` 与 `compute`；baseline `b3abb2f5`，前者当前仍指向 R1 terminal，须重新绑定。
- trigger：四路线长期合同排除用户 timing profile 竞赛及模型多设备计算放置；现有公开 profile 及
  Graph incumbent 比赛与该合同冲突，属于已批准迁移，不重审整体架构。
- consumer：普通单路线 Graph compile/replay、同设备多 Provider/Context/lane、host storage/transfer；
  正交消费为资源副作用、常量/shape/alias/预算约束下的非模型计算。
- scope：G31–G35；继承 [R2-D](compute-v16-direct-owned-results.md) 的新接口验证尾项，其实现与
  Windows 9 门已通过；不因换卡把未完成的 Graph/oneDNN、三交付、Unix/sanitizer 门写 PASS。
- non-goals：真实路线 kernel/默认赢家资格、统一 native compiler/executor、模型 CPU+GPU/多 GPU
  分担、全库工具改造、算子扩充。各路线自己的规则与 P1 仍归 R3–R6。
- 性能：P0 先核对编译的资源/搜索费用与原热路径；不以删除 profile 接口签路线最优。
- 预算：定向审计 30 分钟；固定范围后的实现/普通验证最多四小时、一次候选纠正，不追加架构探索。
  复用已通过 Q4 的精确记录，稳定 R2 候选一次性补齐当前公共边界的适用验证。

## 已固定的有限 FIX

- G31：移除 Compute 公共 timing profile API 和 Graph profile context/identity 接收面及其私有编码。
  保留 ProviderRouteKey 的实现身份用途，不把身份误作性能资格。
- G32：移除 Graph 为用户 profile 保留/比较多个已准备计划的 incumbent 路径。保留有限的 region
  覆盖、资源可达性、workspace/alias、依赖、预算及无副作用拒绝后的事务化重试；首个满足固定规则与
  硬约束的完整计划即可绑定，不计时、不执行用户输入、不在 replay 搜索。
- G33：把原 profile carrier 中确有消费者的并发执行要求放回直接 cold compile 选项；不再携带测量
  fingerprint、经济阈值或对未知 replay 次数的摊销排名。
  固定为 `CompileOptions::require_concurrent_invocations = false`；true 是制品并发能力硬约束，
  不承诺预留 N 个 invocation，也不推测吞吐收益。
- G34：共同 Compute-backed 编译只接受同一 Device 的计算 targets；允许同设备不同 Context/
  Provider/lane 和不同 host/device storage。普通 `compile(graph, context, options)` 路径保留。
  Apple/Vulkan 的原生 compiler 可直接消费同一逻辑图，本卡不强迫其转换成这套执行制品。
- G35：迁移实际 sample/docs，删除 profile 比赛专属测试，保留并迁移其中真实预算/资源/region/
  错误回收反例；给不同 Device 拒绝和同设备组合补具名消费。只动受影响消费者。

## 审计锚点与接入边界

`graph/executable.cc` 的 prepare_placed_steps、build_report 和 compile environment 校验；
`graph/executable.h` 的 CompileProfileContext/CompileOptions/SearchReport；Compute 与 Graph
各自 route_profile 文件，以及 package custom-kernel sample。当前冷搜索在没有 profile 时已选择
第一个可行计划，后续比较分支只由用户资料打开；这不是保留该分支的需求。

路线 owner 在 R3–R6 准备时提供研发已确定的适用实现/片段规则，普通用户选择 native route；本卡
建立无用户 profile 的编译边界，不虚构新的性能赢家。专家约束 Provider 是显式约束，不是普通用户
必须提供的优先级或性能资料。必要的 Graph feasibility planning 仍受原累计预算约束。

## 预声明验证

API/算法稳定前只回放改动 owner 与真实直接消费者；完成后对不可变 R2 候选运行 selector，保留
skipped reasons。Windows 运行 Compute、CPU/CUDA/Native/oneDNN、Graph/Graph-oneDNN 对应门；
WSL/两 Mac 回放共同代码和原生适用消费者，原 R2-D 的寿命/并发采用适用 sanitizer。
source/static/shared/relocation 使用同一普通 C++ 消费并包含 owning result 的 producer DSO。
没有独立 binary 的接口不制造额外交付 target，未运行的适用门仍为 UNPROVEN。

红队检查：冷编译不执行工作、未知/失败不被吞掉、明确拒绝后的资源回滚、同设备多 Provider、host
storage 和多 lane 不被误禁、异设备模型计算确实拒绝、workspace/effect/alias/shape/retained result
不因 profile 清理退化。R2-M 的 Mem v10 定向尾项继续由当前 Mem 卡处理，不扩大底层审计。

## 定向审计结论

2026-09-16 对 baseline 的上述函数、公开类型及直接消费者完成定界。`swap_route_state` 仅服务
profile incumbent；去掉该分支即可同时去掉快照及 metadata-moved 重建，不改真正拒绝后的回滚。
并发布尔约束迁移保留现有硬约束语义；Device 使用既有值身份相等，不猜测 backend 名字或序号。

受影响测试限于 Compute profile 单测、Graph profile/资源/packing 用例及 oneDNN report 消费；
原预算、OOM、常量共享、packing 拒绝释放、并发和真实数值回放继续保留。oneDNN 的产品代码无需
改动；其 profile report 断言随公共 API 迁移。sample 是现有 package custom-kernel 消费。
P0 确认本卡删除冷测量资料和多个制品保留；不增加 replay 分配、计时、搜索或逐 kernel 间接层。

Compute 与 Graph 均绑定本卡 AUDITED 后实施 G31–G35；D21–D24 已推送于 `b3abb2f5`，本卡承接
其尚未运行的跨平台/三交付/Graph 集成门，终局一起裁决，不把换卡当作结案。

## 2026-09-16 实现检查点

G31–G35 已实现：删除公共 profile/matcher、Graph profile identity 编码和 incumbent 快照；冷编译
绑定首个完整可行计划。并发要求成为直接布尔硬约束；所有计算 targets 在 Provider 查询前核对同一
Device，含同 backend 的不同 Device 拒绝。同设备不同 Context/Provider/lane、host storage、常量与
packing 共享及事务化回滚保留。package custom-kernel 样例与 user 文档已迁移。

Windows Ninja / Release / WERROR、实际并行 2：Compute/逐头和 Graph/逐头共 5 个针对性门已
通过；Graph executable 内 281 个具名用例通过，包含真实 CUDA 存储/capture 及 CPU+CUDA 计算放置
拒绝。同设备的多 Context 聚合、晚到失败、放弃后回收和 copy 不可行反例继续通过。该协议 fixture
不计作完整 CUDA 路线；真实 CUDA kernel/默认规则仍归 R4。

首轮失败均保留在外部原始日志：测试 helper 参数写错；旧 fixture 把 Context probe 当 Device 身份；
纯 View 错用 Queue submit；两个 lane 的拒绝计数仍沿旧 profile 比较写成一次。已按真实接口/物理
身份迁移，不放宽产品的同 Device 校验。旧 CPU+CUDA 混合计算正例已迁为明确拒绝；其历史回放
不再算当前能力。日志根 `C:/hyle-scratch/r2-q4-20260916` 的 `g-*` 文件保留各轮原始结果。

以上只签此次 Windows 针对性检查；不可变 R2 候选的完整 selected gates、oneDNN、三交付、WSL/
两 Mac 与适用 sanitizer 尚待执行，当前保持 AUDITED。API/算法冻结后只补这些已声明门，不再重跑
已结案 R2-Q 的独立 schema 验收；不启动 P1 或增加性能测量框架。
