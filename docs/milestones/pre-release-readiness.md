# HYLE 预发布就绪度与模块严格收口队列（2026-07-25）

## 结论

HYLE 的“实现进度”与“当前标准的收口证书”必须分开读：除 compute/graph 仍有产品能力尾，
其余模块的主体实现基本存在；但只有 **error、platform、mem、time、core** 已按本轮实际使用导向
完成严格复审。早期标成 closure/frozen 的任务仍是有效施工证据，不自动等于新标准认证。

| 层次 | 当前判断 | 下一动作 |
|---|---|---|
| 严格认证基线 | error、platform、mem、time、core | 只因明确触发器或依赖受影响项局部回验 |
| 已实现、待新标准复审 | obs、math、util、container、sched、io、simd | 按本文队列逐模块收口，不按旧 benchmark 形状默认通过 |
| 仍有能力尾 | compute、graph | 先用真实北极星补能力，再执行同一严格合同 |
| 整库发布面 | 部分完成 | 模块内 C ABI 随模块收；最后只做 install/export/FFI/bundle 总验收 |

当前审查单元是 **build/package system**，完成后进入 util-core：构建
系统必须把模块 target 图、可组合子集、迁移/安装和三平台环境变成可执行契约，不能继续让
关键前置散落在 CMake、preset、脚本与文档之间。当前队列见“施工顺序”。

## 严格收口合同

模块同时满足以下八项，才标成“当前标准严格收口”：

1. **真实功能面**：列出外部用户与 HYLE 内部消费者的常用场景；同本地成熟库能力对账，缺项实现或写清负空间和理由。
2. **架构与类型形状**：ownership、lifetime、concurrency、failure、组合方式贴正常调用；删除只为 benchmark 方便而存在的 API/元数据/类型。
3. **复杂度与性能**：先争取同类 SOTA 算法复杂度，再在同合同正常 workload 对拍强 peer；微基准只能解释机制，不能决定公开形状。
4. **正确性与平台**：所有权、OOM/overflow、并发、失效、平台差异和 feature dispatch 红队；Windows/Linux/macOS，适用时 ASan/UBSan/TSan/特殊 ISA。
5. **C++ 易用性**：首选 source API 有清晰的默认路径、RAII/typed 入口、错误传播和组合 recipe；HYLE C++ 内部不得绕经 C ABI。
6. **C ABI 完整性**：覆盖全部语言无关能力；共享 `hyle_status`/error 语义，纯 C 头、动态导出、所有权、callback/visitor 生命周期与 FFI 组合机械验证。
7. **依赖收束**：向下确认正确使用已冻结 error/platform/mem/time 等能力；向上扫描直接消费者是否自造重复机制或错误调用。只重测实际受影响边。
8. **文档归位**：architecture 管稳定设计，performance 管方法和点值，`guide/<module>.md` 管“什么情况选什么”的 C++/语言无关入口，`c-<module>.md` 只管精确 C 映射与坑，milestone 保留施工记录。

参考库不是 feature 数量竞赛。每个借鉴项必须能回答“哪个真实场景需要它”；每个性能结论
必须同合同、同所有权、同并发度、同错误语义。否则宁可登记未知，也不能用一个容易赢的
benchmark 宣称 SOTA。

## 施工顺序

排序依据是**接口爆炸半径 + 消费关系**，不是目录顺序，也不是谁的 benchmark 更多：

1. **core**：冻结 version、公开头与 intrusive 基础形状；对已收口 time 做受影响项回验。
2. **build/package system**：让 CMake target 图准确描述模块组织；审 options→宏→target、依赖获取、三平台统一入口、static/shared、模块子集、source embed、install/find_package、交叉编译与生成目录治理。
3. **util-core seam**：只审低层 header-only 能力及 include/link 边；对 mem 的 SpinLock 消费做局部回验，不提前冻结 Registry。
4. **obs**：分开审 log runtime 与 trace gate；统一 C error 形状，再核 mem/sched/io/compute hooks 是否正确。
5. **container**：在 mem/platform/util-core 稳定后审容器能力、复杂度、争用、allocator/error 和 C/C++ 形状。
6. **util-runtime / 完整 util**：container 稳定后再冻结 Registry、type-erasure 与完整 C ABI，合并 util-core 的证书。
7. **sched**：汇合 platform/mem/time/container；按真实任务、同步、取消、关闭和 blocking workload 审算法及 API。
8. **io**：在 sched 合同稳定后审 proactor、网络/文件/TLS/HTTP 的普通路径、错误、deadline/cancel 与跨后端性能。
9. **math↔simd 边界审查，再收 simd**：先裁决 scalar/std/SIMD/消费域归属，再冻结 SIMD dispatch、dtype/math/sort/search。
10. **compute**：补完真实 CPU/GPU/ggml/Vulkan 产品能力后收口；复用已冻结 simd/sched/mem/platform。
11. **graph**：最后汇合 sched/compute/simd，完成 DAG 与数据并行北极星能力后收口。
12. **发布总验收**：install/export、跨编译器 ABI、纯 C/TS/Python 最小 FFI 和发布 bundle。

util 的两步是其真实产品分层：前一步只冻结低层自包含头，后一步才冻结依赖 container 的
`hyle_util` runtime。两步必须使用同一任务页累计证据，禁止演变为两轮全模块复审。

math 的处理相反：它当前不是经过独立需求设计的产品，而是迁移残余。严格复审必须与 simd
的标量/向量边界一起做；可能结论包括保留清晰 scalar helper、迁移到 std/其他模块或撤销
独立 target。先决定边界，再决定 guide/C ABI，不能用文档和导出把历史偶然固化下来。

## 当前模块判定

| 模块 | 实现状态 | 当前严格证书 |
|---|---|---|
| error | 功能主体完成 | ✅ 严格收口；全库 adoption 随各消费者复审继续 |
| platform | 功能主体完成 | ✅ 严格收口 |
| mem | 功能主体完成 | ✅ 严格收口；已在 error/platform 变化后回审 |
| time | 功能主体完成 | ✅ 严格收口；core 变化时只做依赖回验 |
| core | 功能主体存在 | ✅ 严格收口；最终实现 `71dabf7`，三平台同合同无 LOSE |
| util、obs、math、container、sched、io、simd | 功能主体基本完成 | 待按新合同认证；旧 closure 是输入证据，不是自动通行证 |
| compute、graph | 主干存在、不是 stub | 仍有能力尾；完成能力后按同合同认证 |

## 停止与重开规则

严格收口不是声称“永远没有更快写法”，而是停止无证据的重复全审。通过后只有以下事件可重开：

- correctness/security/ABI 缺陷；
- 新平台、ISA、后端或编译器使原合同不成立；
- 同合同强 peer 证明复杂度档落后或可复现的实质性能差距；
- 真实消费者暴露缺失能力、难用接口或错误依赖；
- 发布工程要求改变模块边界。

亚纳秒点值波动、没有 workload 的猜测 API、新库出现、命名偏好、为了增加 op/LOC，都不能
单独触发重开。下游模块改变只回验它真正触碰的依赖边。

## 证据信源

- 构建拓扑和慢变顺序：[`../architecture/roadmap.md`](../architecture/roadmap.md)；参考库角色：[`../architecture/references.md`](../architecture/references.md)。
- 当前任务与 LOC：[`README.md`](./README.md)；每个旧 closure 文件只是该轮施工证据。
- 稳定设计：[`../architecture/`](../architecture/README.md)；性能：[`../performance/`](../performance/README.md)；用户入口：[`../guide/`](../guide/README.md)。
- 已严格认证模块：[`error-module-reclosure.md`](./error-module-reclosure.md)、[`platform-remaining.md`](./platform-remaining.md)、[`mem-close-plan.md`](./mem-close-plan.md)、[`time-module-closure.md`](./time-module-closure.md)。
