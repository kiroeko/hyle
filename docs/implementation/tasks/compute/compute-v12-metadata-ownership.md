# Compute v12：编译与运行 metadata 的分配归属

## Closure card

- component / owner：`compute`；baseline `6caa79f9`，承接未 terminal 的
  [v11](compute-v11-bounded-instantiation.md)。C93 真实 profile 与完整 foundation 验证仍保留。
- trigger：Graph G106 的独立 Executable/Invocation 预算需要指定 metadata owner；当前私有 prepare、
  domain instantiate 和 invocation create 固定从 Context/root 分配。Mem v9 已修复 planner 的相同缺口，
  但不能覆盖 Compute 控制对象和 Provider 通过传入 allocator 进行的分配。
- 主旅程：Graph 编译有限形状 family，编译制品和运行实例使用不同配额；失败完整回滚，多个实例保活
  同一域根。正交旅程：普通 direct exact 与 typed custom kernel 保持原 Context owner 和提交路径。
- 范围 C120--C122；只在私有 factor seam 传递 Mem `AllocatorRef`，不新增公开 allocator/policy 产品。
  Graph 在 G106 范围传递其已存在的编译/实例 owner；完整预算、opaque packing/driver 内存仍归 G106。
- 依赖：已有 Core/Error/Mem/Util/ComputeOps；直接消费者 Graph、Native、oneDNN、typed custom；
  不修改 CLOSED backend、第三方源码、组件图、CMake option 或公共治理/性能仪器。
- 预算：一轮有限表示/实现及一轮产品纠正；正常 fixture/编译错误就地修正，不用计时裁决 owner。
- 验证：逐分配 OOM/精确 owner-size-alignment、源 root 先释放、独立并发实例、迟到 Completion；
  Windows static/shared Compute/CPU/Native/Graph/oneDNN、CUDA runtime/Graph、逐头和 CPU/CUDA 三交付。
  实现阶段 P0；本轮不重复 P1/全库/CI。Linux/两台 Apple、适用 sanitizer、独立终局红队与一次 foundation
  P1 仍在共同候选完成后验证，当前不代签。
- stop：owner 选择进入 submit、产生另一 Context/Queue/runtime、Provider 必须 JIT 或忽略传入 owner，
  或必须修改 CLOSED 依赖；出现这些新事实先回到审计。

## AUDITED：有限合同

| ID | 当前缺口与动作 | 验收 |
|---|---|---|
| C120 | FIX 私有 propose/prepare/domain instantiate 接受可选 allocator；未指定继承 Context/root，显式无效拒绝 | scratch、Provider callbacks、PreparedState 和 shape copy 全部使用选定 owner；失败不泄漏且能重试 |
| C121 | FIX 私有 invocation create 可选独立 owner；公共 direct 入口仍默认继承 prepared | Provider run-state 与 Compute wrapper 同一 owner；销毁 prepared handle/迟到 Completion 不提前释放实例 |
| C122 | FIX 部分 invocation 回滚及 parent 保活顺序 | create 返回错误且有可销毁 partial 时回收；释放 child 自身存储后再释放所保有 root/Context；不改变工作接纳或 warm completion |

准入为 `SELF` 的组合 ownership seam，复用 Mem 分配机制；不是新增 allocator 算法或通用 quota 实现。
公开 `prepare`、`PreparedOperation::create_invocation`、typed kernel 概念及三交付用法均不增加参数；
普通用户仍配置 Context metadata allocator。内部可选值未指定和显式无效分开，后者返回 InvalidArgument。
allocator 是借用，必须活到相应对象及其 Completion 完全释放，且支持实际分配/释放线程；Compute 不把
借用自动变成 owning arena。root 与实例可以有不同 owner，但实例仍保活其借用的域制品。

PreparedState 已有 AllocatorRef，直接记录所选值而不增加字段。InvocationState 用冷标记选择继承或
同一 allocation 的 allocator 尾部；只有不同 owner 才多存一份 AllocatorRef，不增加 heap 次数。
标记放在既有 bool 尾部区域，测试核对普通实例大小与分配账单；不通过 pointer tagging 或新的虚表改变
热提交。所有 owner 比较、初始化与选择都在 create/destroy；submit 的算法、callback 层级与锁/原子不变。

Provider create-invocation 失败应返回空指针或一个可由 destroy-invocation 销毁的 partial；Native、oneDNN
和 typed custom 现有失败路径均置空且自己清理。Compute 补齐非空 partial 的通用回滚，与 Graph region
已有合同一致。不把 borrowed payload、Context 自身/Queue、Provider 私有 allocator 或驱动占用算进
metadata owner；本卡不能签完整内存上限。

Graph v10 的实际调用同时传递 plan/instance allocator，防止多个 target 的 Context allocator 绕过
Graph owner。Region prepare 已接收 Graph allocator，本卡不改变它的 Provider 合同。没有新公开 C/FFI
入口；增加的私有 C++ exported seam 由相同版本 source/static/shared Graph 消费。

本卡绑定时 Compute 已处于 `AUDITED`，这里在产品修改前重新固定有限 FIX；保持 `AUDITED`，不以这些
增量内容替代未完成的 foundation 范围或 candidate/terminal。

## 2026-09-15：metadata owner 检查点

C120--C122 已实现。Graph 的跨 target leaf 准备现在使用 Graph allocator；domain instance 和 operation
invocation 也明确传递 Graph 实例 owner。普通 direct 使用原默认，显式无效 override 不会退回系统堆。
部分 invocation 错误先调用同 allocator 的 destroy，再返回原 Status。PreparedState 在释放自己的
allocation 后才释放 domain root/Context；迟到完成仍保活完整实例，未改变工作接纳和完成状态机。

新增 10 个定向测试：Native 的 exact/domain 多 owner、逐分配失败/重试、无效 override、默认大小/
分配次数、并发独立 owner；CPU 的多输出 inference scratch、部分失败与迟到 Completion；typed custom
的 Invocation/Compute wrapper 两处分配失败和自然数值执行；Graph 的不同 Context owner、有限 exact
变体/general fallback 与重复运行。Native 的 tracking allocator 逐项校验 pointer/size/alignment 和
最终零 live，避免只看一个总分配计数。Graph public journey 明确禁止被选 target Context 新增 metadata
分配，仍能编译/创建；Queue 完成槽的冷预留仍属于 Queue Context，不错误记成 Graph 分配。

| Windows Release 验证 | `C:/hyle-scratch/` 下缓存 | 结果 |
|---|---|---|
| static Compute/CPU/Native/Graph 与逐头 | `compute-route-v8-static-20260914` | 9/9；后补 fixture 后 CPU/Graph 2/2 |
| shared + oneDNN direct/composition/Graph/region 与逐头 | `compute-route-onednn-v8-20260914` | 15/15；最后 wide scratch fixture 后 CPU 1/1 |
| CUDA runtime、late failure、Graph 与逐头 | `compute-graph-v7-cuda` | 8/8；Graph execution 230 用例含 2 个真实 CUDA 用例 |
| source CPU/CUDA custom consumer | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA custom consumer | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA custom consumer | `custom-execution-v10-static-consumer` | 2/2 |

最终 CPU runtime/custom 47、Native 84、普通 Graph execution 228 用例通过。首次 build target 名误写为
`hyle_graph_execution_test`，在 Ninja 阶段停止，改为已登记的 `hyle_graph_executable_test`；新增 Graph
fixture 最初给 inactive lane 也传了 Queue，被原合同正确拒绝，改为只传 active lane 后通过。未放宽产品。

命令沿用 `scripts/hyle-windows.ps1 -Action Build/Test -BinaryDir <cache>`，build `--parallel 2`，CTest
`--parallel 1 --timeout 120 --output-on-failure`，只选表内 target/regex，不启动全库/CI/P1。source consumer
显式构建 `hyle_custom_kernel_sample hyle_custom_cuda_sample`；两个 SDK 从同名 `*-build` 构建
`hyle_graph hyle_compute_backend_cuda hyle_compute_provider_native` 后显式 `cmake --install ... --prefix ...-sdk`，
不写 Program Files。Mem/Compute/Graph DLL 从真实 producer/SDK stage 到 CPU/CUDA 目录并比较 hash。
shared producer/SDK/两个加载目录的 Compute DLL SHA-256 均为
`F7BA0BACFB5ACDDDED2EB83685B76D83D0884C2FCB83D44D083833EE6B066040`。

P0：已有 shape/inference/descriptor 算法和 allocation 次数不变；PreparedState/公开 handle 不增字段。
测试用只在测试定义的旧字段布局确认普通 InvocationState 大小不变，默认和显式同 owner 都不分配尾部；
不同 owner 才在原 allocation 中多 4-word AllocatorRef。create 有 `O(1)` 校验/owner 比较，destroy 有
`O(1)` owner 选择；包括最后引用由 Completion 释放时的 teardown，不宣称它免费。正常保留 Invocation
的重复 submit 不读取标记/尾部，不新增查找、分配、锁、原子或 callback 层次。不是计时胜负结论。

独立于实现的一次本轮源码对账覆盖所有 allocator 入口/回滚/最后引用、tail alignment/lifetime 和 Graph
直接消费者，没有遗留产品 finding；这是主 agent 的增量源码复核，不冒充独立终局红队或跨平台资格。
末次产品文件仅修正上述 teardown 注释，未改变已验证的代码/token/layout。selector 基线 `f2b6d6a7`
READY，83 项有据跳过；closure（含相对 `6caa79f9` 的历史检查）、docs/diff 通过。

统一 LOC self-check：全库 **679,867**，自实现不含测试 **124,154**；Compute **16,731**、Graph **12,275**，
合计 **29,006**，按两个产品全部实现因子、不含测试和第三方的相同口径。

保持 **AUDITED**，不是新的 terminal。C93 真实 profile、G101 动态属性、G103 联合选择、G106 完整
resident/冷峰值/并发/capture/packing、真实 CPU/CUDA/Apple 探针和共同 foundation 的 Linux/两台 Apple/
sanitizer/独立终局红队/P1 继续保留；本轮 allocator seam 不代签，也不授权横向扩充算子目录。
