# Mem v9：生命周期计划器的 metadata allocator

## Closure card

- component / owner：`mem` / Mem；baseline `03975179`；前一 terminal 为
  [v8](mem-v8-resource-type-visibility.md)，其它 Mem 能力沿用既有结论。
- trigger：Graph G106 的冷编译/运行期预算需要约束完整 metadata；`src/mem/lifetime_plan.cc` 的
  RawArray、PlanImpl 与 PlannedImpl 全部硬编码 `system_allocator()`，因此 Context allocator 的 quota
  和逐分配故障注入无法覆盖这一真实直接依赖。不是 benchmark 或重复 review 触发。
- 主旅程：Graph 用自己的 metadata allocator 规划并创建 workspace，再以 Mem Buffer 保有设备 backing。
  正交旅程：直接 Mem 用户将编译计划与运行实例分配给不同的 caller-owned quota/arena。
- scope：只补 LifetimePlanner/LifetimePlan 的显式 metadata allocator 与完整 sized-free；Graph 传入
  既有 allocator。不会让 payload 改走 host allocator，也不扩展 tensor/graph/runtime 职责。
- non-goals：best-fit/alias/lifetime 算法更改、常态 allocation 次数增加、device resource 虚表更改、
  全局 allocator setter、其它 Mem 能力、Native FFI、公共仪器或 CMake 改造。
- 依赖不变；真实产品调用者为 Graph，Mem-vm API surface 测试也消费该公开头；公开交付通过 Mem 及
  Graph source/static/shared 验证。
- 性能 P0：算法/分配次数/布局查询保持，检查新增 allocator identity 的冷 metadata 成本；不为冷绑定
  跑一轮纳秒 P1。既有 allocator/资源内核未改，不重新签其它能力的性能胜负。
- 预算：一次表示/实现及一轮纠正；错误注入覆盖每次 scratch/result/instance metadata 分配，验证
  payload OOM、move/reset、plan 释放后 instance、继承与覆盖 allocator、空 plan/非法 allocator。
- 验证：Mem correctness/逐头、Graph/oneDNN/CUDA 定向回归、source/static/shared；Windows、WSL GCC14、
  两台 Apple 的 Mem 定向正确性及适用 ASan/UBSan；独立于实现步骤的一遍源码反证；不跑全库/P1/CI。
- stop：必须改变规划算法或 payload ownership、allocator 生命周期无法自然表达，或新的常态工作；
  适用验证未通过不关闭。本轮成功只补依赖缺口，不代签 Graph 的完整内存 gate。

## SCOPED

现状缺口为 `PARTIAL/FIX`：payload 的 MemoryResource 已可选，但 planner scratch/result 与 instantiated
metadata 没有可注入路径。职责仍归 Mem；Graph 不复制 planner 或在 Mem 外猜测内部 allocation 数量。
下一阶段先固定公开调用与生命周期，再修改产品。

## AUDITED：固定接口与有限改动

| ID | FIX | 验收 |
|---|---|---|
| M90 | `LifetimePlanner::build(requests, resources, allocator = system_allocator())`；所有 scratch/result 经同一 allocator，保留 sized-free identity | 逐分配拒绝完全回滚；无 payload 分配；无新算法或额外 heap 次数 |
| M91 | `plan.instantiate(resources)` 继承 plan allocator；`plan.instantiate(resources, allocator)` 显式选择另一 metadata owner | instance 可超过 plan 生命周期；override 不借旧 allocator；payload 仍由 resources 分配/释放 |
| M92 | Graph 向 LifetimePlanner 传 `plan.allocator`，运行实例自然继承；guide/交付与故障测试同迁移 | Context fault injection 不再漏过 Mem scratch；三交付同义 |

所有 allocator 均为 borrowed `AllocatorRef`，必须活到其所分配的 plan/instance 最后释放。instantiate 的
默认继承与显式覆盖是两种真实 owner 需求，不是旧 ABI 兼容分支；无参数改变的 view/backing 查询保持
原样。显式 invalid allocator 拒绝，空计划也需要可用 metadata allocator。临时 RawArray 保有实际
allocator，转交结果后由 PlanImpl/PlannedImpl 独立保有 identity；用户可直接组合已有 `QuotaAllocator`，
不新增另一个 quota、全局 setter 或资源种类。

普通 C++ 形状：`LifetimePlanner::build(requests, resources, compile_quota.get())`，然后
`plan.instantiate(resources, run_quota.get())`。metadata quota 不约束 resource payload 或 allocator 内部
rounding；这一边界必须在头和用户指南同时说明。公开 owner 仍为单指针，新增 identity 只在 cold impl/
临时数组；分配数量、best-fit/alias、`O((N+E) log N)` 和 view 的 O(1) 路径不变。

至此 M90--M92 为有限 FIX。需要改变 payload/算法或增加其它公共功能时重新裁决，不顺手扩大本轮。

## 2026-09-15 实现检查点

M90--M92 已实现：全部 RawArray scratch/result 和 PlanImpl/PlannedImpl 使用显式 allocator；每个 owner
保存实际释放身份。默认实例继承，显式实例覆盖，payload 仍使用原 MemoryResource。Graph 唯一的
LifetimePlanner build 调用现传入 `plan.allocator`，没有复制 planner 或修改 placement 算法。

新增六项测试逐次拒绝 planner/instance metadata 分配，检查同 owner/size/alignment 释放、失败重试、
payload 第二次分配失败回滚、零配额/非法 allocator、计划先释放、实例 move/reset 和并发实例化。Mem-vm
API surface 与 Mem package sample 同步验证两个公开实例化形状；guide 区分 metadata 与 payload 预算。

Windows Release 增量：`compute-route-v8-static-20260914` 的 Mem/Graph/逐头 5/5，
`compute-route-onednn-v8-20260914` 的 Mem/Mem-vm/Graph/oneDNN/逐头 10/10；后者 Mem 237、Mem-vm 65、
Graph 基础 25、Graph 执行 227、oneDNN direct 1、region 17 个用例通过。最后仅整理新增声明的局部格式与
并发限制注释，不据此运行 P1。这些是增量正确性，不代签尚待候选验证的三交付、跨平台、sanitizer 和红队。

影响边界：公开头的保守 selector 会展开 Mem 的传递消费者；本卡实际代表集为直接使用 LifetimePlanner
的 Graph、Mem-vm API surface 与 Mem 三交付。其余 Container/IO/Process/Sched/Util 不调用本次变更的
planner API，AllocatorRef/MemoryResource/Buffer 的合同与实现没有变化，不额外运行那些模块的产品测试
或 package；该裁决遵循 correctness profile 的代表性直接消费者规则，不把未运行的自动选择项写成 PASS。

## Candidate

产品候选为 `8e4f096e801fda16aa3b8ab507556188e7883159`。M90--M92 的 API/算法/正常成本与直接消费者
迁移已固定，后续只验证并记录证据；Mem 进入 `CANDIDATE`，不是 `CLOSED`。Graph 保持原活动阶段，
完整内存预算仍待此依赖验证后继续。

## 候选后源码反证 / P0

2026-09-15，主 agent 在候选固定后的独立复核步骤检查 `RawArray`、两类 impl 的创建/销毁、全部失败
出口、Graph 调用和公开 sample；不是另一个 agent 的独立签名。有限检查结果：

- **分配/释放配对**：每个成功数组由相同 allocator、原始 count/sizeof/alignment 释放；成功转交后
  RawArray 清空，不 double-free。空数组不调用无效 allocator。两类 impl 销毁前先复制 allocator identity，
  不从已释放 impl 读回它；逐分配失败测试实际核对这些参数。
- **回滚**：plan metadata 转交只发生在 impl 成功分配以后；instance payload 中途失败时，RAII 释放前面
  已成功的 Buffer/数组。错误仍为 Error 的 InvalidArgument/FailedPrecondition/ResourceExhausted，不吞
  provider 原始失败，也不将 quota OOM 当作成功空计划。
- **独立生命周期**：instance 保有 placements/resource records 的复制，不保有 Plan 指针；override 不访问
  旧 allocator。继承则借用同一个外部 owner，并由头/guide 说明其寿命和并发前提。Graph Context owner
  与 planner 的关系不改变，submit 没有增加分配、计数或路由。
- **成本**：PlanImpl 与 PlannedImpl 各新增一份四字 AllocatorRef，临时数组各保存同一身份；两种 public
  owner 仍为一指针。build 仍为 `O((N+E) log N)` / `O(N)` scratch，instantiate 为线性 metadata 复制及
  每 backing 一次 Buffer 创建；没有新增 heap 次数、锁、原子、syscall 或热查询分支。分配转调使用已有
  AllocatorRef，不新增封装层。未改变规划/查询算法，不用 P1 给冷 owner 注入签纳秒胜负。
- **接口/依赖**：两个实例化形状分别对应继承与独立 runtime quota 的真实任务，不是 benchmark 专用或
  旧 ABI 兼容分支。C++ 导出由现有 `HYLE_MEM_API` 承载，无 C/FFI；没有新第三方、组件边或 CMake option。

没有发现候选产品必须纠正项；这里只签源码反证和 P0 结构成本，尚待完成的验证不能由此代签。

## 跨平台与交付证据

同一候选通过完整 `git archive` 送到 WSL/两台 Apple；压缩包 SHA256 为
`3e84a46935267c37b1583540f9291e388cbe06629b75b06bb01be1632922e233`。远端只作执行副本，没有向
Windows 回写产品。WSL 在 `/root/hyle-mem-v9-8e4f096e`，两台 Mac 在
`/Users/kiro/hyle-mem-v9-8e4f096e`；构建分别在同路径后缀 `-validation/{off,asan,ubsan}`。

| 环境 | 普通正确性/逐头 | ASan | UBSan | Mem source / shared / static |
|---|---|---|---|---|
| Windows MSVC 19.50.35728 | 上述 10/10 定向回放 | 2/2 PASS | MSVC 不支持，另以 GCC/Apple 验证 | 3/3 PASS |
| WSL GCC 14.2.0 | 4/4 PASS | 2/2 PASS | 2/2 PASS | 3/3 PASS |
| Air Mac16,12 / AppleClang 21.0.0.21000334 | 4/4 PASS | 2/2 PASS | 2/2 PASS | 3/3 PASS |
| Mini Mac16,10 / AppleClang 21.0.0.21000101 | 4/4 PASS | 2/2 PASS | 2/2 PASS | 3/3 PASS |

普通/UBSan：Mem 237、Mem-vm 65 用例；ASan 的 Mem 额外编译一个既有 sanitizer 用例，合计 238，
Mem-vm 仍 65。Unix 构建采用 RelWithDebInfo/static、shipping layout、build 并行 2、CTest 串行；
ASan/UBSan 为不同目录，未将未插桩测试冒充 sanitizer。

复现命令为 `scripts/hyle-unix.sh cmake -S <candidate> -B <build> -G Ninja
-DCMAKE_BUILD_TYPE=RelWithDebInfo -DHYLE_COMPONENTS=mem -DHYLE_BUILD_TESTS=ON -DHYLE_BUILD_BENCH=OFF
-DHYLE_STATIC=ON -DHYLE_INSTALL=OFF -DHYLE_SANITIZER=<off|asan|ubsan>`；构建
`--target hyle_mem_test hyle_mem_vm_test --parallel 2`，CTest `--parallel 1 --timeout 120`，普通 regex
`^hyle_mem(_vm)?(_header_self_contained)?_test$`，sanitizer regex `^hyle_mem(_vm)?_test$`。
Windows sanitizer 使用相同开关和 VS 环境入口，目录 `C:/hyle-scratch/mem-v9-8e4f096e-asan`；
MSVC 没有 UBSan。

四个环境的 Mem 三交付均运行 `HyleCiSelectedComponentSmoke.cmake`，参数为
`HYLE_SMOKE_COMPONENT=mem`、`HYLE_SMOKE_GENERATOR=Ninja`、绝对 candidate/source 和独占 smoke root；
Windows root 是 `C:/hyle-scratch/mem-v9-8e4f096e-delivery`，Unix 是上述 source 路径后缀 `-delivery`。
它验证真实 package sample、共享导出卫生和 relocated SDK，不安装到 Program Files。Mac 初次交付配置
因非登录 shell 的 PATH 缺 Ninja 失败；在工具目录显式进入 PATH 后修正，未改产品或公共脚本。

Graph Windows 依赖回放：`compute-graph-v7-cuda` 3/3 PASS（Graph 25、执行 229，含真实 CUDA），
`custom-execution-v10-source-consumer`、`custom-execution-v10-shared-cuda-consumer`、
`custom-execution-v10-static-consumer` 各 2/2 PASS。SDK 从同名 `*-build` 重建并显式安装至 `*-sdk`；
每个 CPU/CUDA consumer 的 Mem/Graph DLL 与 SDK 校验一致，避免旧 stage 产物假通过。设备为 RTX 4070，
driver 610.62；本轮不是 CUDA 性能资格。shared Mem DLL SHA256 为
`49965CEE9114B2F1052DA50571C384E6C271713D4F2FC3A8AB55AC9695E1925B`，Graph DLL 为
`E6712793E49E4A102CA07F776B42D9980724EAE19356BFC35315C412D36984FB`。

所有验证只针对本轮 allocator seam 及代表性消费者；未运行全库/CI/P1。Graph foundation 的 Linux/Apple
真实计算探针、终局红队和性能门仍属于其活动卡，不能由 Mem 验证代签。

## 七问、语义覆盖与裁决

1. **第三方**：没有升级或新增依赖；默认 backing 仍为中央 lock 的 mimalloc v3.5.1，MIT 及三交付许可证
   由既有 Mem 合同承载。新增接口只接受 HYLE AllocatorRef，不泄露第三方类型。
2. **功能/自然形状**：主旅程 Graph workspace metadata 与正交的独立 compile/run quota 均已可用；M90--M92
   没有未完成项。现有其它 Mem 能力沿用 v8，不以本轮的小改动重新声称全域功能超集或性能。
3. **内部复用**：直接使用现有 AllocatorRef、QuotaAllocator、MemoryResource、Buffer 和 Error；Graph
   复用 LifetimePlanner。未引入第二个 allocator/planner/quota 或向更高层依赖。
4. **性能**：上述 P0 通过；算法、heap 次数、热路径与 public owner 大小不变，冷 impl 增加明确 owner
   identity。没有新计时结果，不声称新 SOTA 或恢复签发未受影响能力的 P1。
5. **交付边界**：仅 C++，无 Native FFI；现有导出卫生在四环境共享 SDK 中通过。
6. **构建**：中央图和 CMake option 未改；四环境 Mem source/static/shared 和 Windows Graph 三交付通过。
7. **易用性/文档**：头、Mem guide 与 package recipe 同时说明默认继承、显式覆盖、owner 寿命、线程安全
   前提和 metadata 不等于 payload/物理内存上限，未要求用户模拟测试内部生命周期。

| 本轮 public journey | 正常/边界与 failure | ownership/teardown | 并发/取消 | consumer / 交付 |
|---|---|---|---|---|
| build 使用 caller metadata owner | PASS：默认/显式、空/无效/配额、每次分配拒绝与重试、既有 overflow/property | PASS：scratch 回滚、同 size/alignment/owner 释放 | PASS：无共享可变 plan state；同步函数无 cancel/deadline | PASS：Graph 与独立 quota，四环境 |
| instance 继承或覆盖 owner | PASS：metadata OOM 在 payload 前、payload 部分失败回滚 | PASS：plan 先销毁、move/reset、override 不访问旧 owner | PASS：4 线程 × 16 独立实例；同步函数无 cancel/deadline | PASS：Mem sample 三交付、Graph CPU/CUDA |

| Gate | 本轮证据 | Verdict |
|---|---|---|
| scope | 具体 Graph 预算/故障注入缺口；仅 M90--M92 | PASS |
| architecture | metadata/payload、plan/instance owner 分离，原规划算法与组件图保持 | PASS |
| correctness | 逐分配拒绝、生命周期、并发、四环境与适用 sanitizer | PASS |
| delivery | 四环境 Mem 三交付/导出卫生，Windows Graph 三交付和真实 CUDA | PASS |
| usability_docs | public 注释、Mem guide、architecture row、真实 sample | PASS |
| red_team | 候选后有限独立源码复核；无未处理产品 finding | PASS |
| engineering_performance | P0 成本结构通过；未运行且不要求本轮 P1 | PASS |

最终 **CLOSED**，只关闭 Mem v9 的显式 planner metadata allocator 缺口。保留限制：AllocatorRef 是借用，
caller 必须保持 owner 生命周期与所需线程安全；metadata quota 不代表 payload/实际物理占用。新的
allocator lifetime/OOM/跨交付缺陷或真实新能力需求可触发定向重开；本轮不豁免 Graph G106 剩余预算。

统一 LOC（候选 `8e4f096e`，self-check PASS）：全库 **679,165**；自实现不含测试 **124,091**；Mem
实现 **8,285**（统一脚本的 `include/hyle/mem` + `src/mem` 目录口径，和前轮一致）。
