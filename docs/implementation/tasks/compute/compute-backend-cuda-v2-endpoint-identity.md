# Compute CUDA backend v2：完成端点不能把 Context 当 Queue

## Closure card / AUDITED

- owner：`compute_backend_cuda`；baseline `a4dee153`；前轮为
  [CUDA v1](compute-backend-cuda-v1-async-runtime.md)。
- trigger：Graph 的真实 pinned → device kernel → pinned 双 lane consumer 在 `a4dee153` 失败。
  `cuda_backend.cc::completion_endpoint_impl` 返回 Context 指针；Graph 按既定 exact endpoint equality
  判为 ordered，于是不同 CUDA stream 的数据依赖没有得到 event wait。这是 correctness trigger，不是性能点值。
- 主旅程：同 Context 两个 CUDA Queue 在 Graph 中上传/计算/下载；两个独立 Invocation 重复 commands/
  capture。正交旅程为 direct Queue event import 和 borrowed native stream，不改变模型或算子语义。
- scope：只纠正私有 endpoint 身份及同 Context 的可导入判断；复用稳定 QueueState、现有 event wait、
  Queue/Completion 保活。公开 C++ API、layout、CMake、版本/许可证和 CUDA 内核不变。
- non-goals：Q4/provider 目录、跨 Context 互通、全设备同步、Graph workaround、热态 probe 或 endpoint registry。
- 性能：P0；endpoint/query 只在冷准备，O(1)、零新 allocation/lock/syscall/metadata；跨 stream 的真实
  event wait 是正确性必要成本，不能沿用错误省略同步的路线作性能基线。不为此冷修复单开 P1。
- 预声明验证：同 Queue 重复 query 相等、同 Context 不同 Queue 不等但可互相导入、另一 Context/invalid
  endpoint 拒绝、Queue move 身份稳定；现有真实 delayed producer/event/import、CUDA runtime/late error/
  逐头，Graph CUDA 定向回放，source/static/shared 的新双 lane kernel consumer；docs/closure/diff。
  CUDA 平台共享实现的 Linux 编译/运行与适用 sanitizer 按完整候选实际环境取证；Apple 不存在 CUDA，
  且本轮未改 AUTO/OFF/CMake，旧平台不存在路线不因私有函数纠正重测。
- 预算：一个表示/实现，一次纠正，候选后独立复核；无新指针 owner、逃逸生命周期或额外字段。
  如需改共用 endpoint 布局/生命周期或扩大 CUDA 上下文互通，停止并重新审计该 owner。

## 有限审计与修复集

`CompletionEndpoint` 的 exact ordering identity 与 compatibility 域是两层：公共内部头明确身份借到
owning Queue 存活期间；Graph equality 表达同一有序端点，可导入只是能建立依赖。CUDA 的
[`cudaStreamWaitEvent`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
才建立另一个 stream 的完成依赖。现有 `QueueSequence::dependency_route` 已按真实 Queue 区分，只有
backend endpoint 错用了 Context。这不是 Graph 应猜 CUDA 类型来补救的问题。

| ID | 允许修复 | 验收 |
|---|---|---|
| CU20 | endpoint identity 改为已存在、稳定的私有 QueueState 指针；先验证 authority/non-null，再通过该 QueueState 的 Context 判断可导入 | 不同 stream 不相等；同 Context 仍可 event import；move 稳定；invalid/其它 Context 拒绝 |
| CU21 | 实际 direct/Graph/三交付反例和指南说明 | 真实 copy/compute/copy 报告 device waits，无 host wait；commands/capture 两个独立 Invocation 数值正确 |

两处私有函数是完整产品 FIX；不新增字段、owner、引用计数或虚构通用 registry。相同 native stream 的
不同 borrowed Queue wrapper 可保守插入一次合法 event wait，不根据 stream 指针猜额外跨 Context 互通。
warm enqueue/copy/event/Completion 实现不改；Graph 自然消费正确身份，不添加 workaround。

能力族是“同执行域内跨队列依赖”；当前为 `PARTIAL/FIX`。正常 C++ 调用、错误与 async lifetime 沿用
v1，所有其它 v1 公开入口 `KEEP`。直接/间接依赖、CUDA Toolkit 13.3 与许可证、FFI=N/A、三个交付形状
均不变；复用 Compute endpoint/QueueSequence 与既有 backend wait。无新模块、开关或其它依赖重开。

`1def523a` 的最小反例已在 RTX 4070 上实际失败：不同 Queue 的 16-byte endpoint 完全相同；其余
compatibility/move/foreign/invalid 检查通过。对应安装源码 consumer 已在 `a4dee153` 因异步依赖报告
失败。两者与源码指向同一根因，接下来只实施 CU20，再按本卡回归；未达到 Candidate/关闭。

## 候选与 Windows 消费回放（2026-09-15）

产品候选 `1644bdc23a6a94a98ac65ccc8bb94fbf64897756` 只改 CU20 的两个私有函数；公开 API、对象布局、
warm enqueue 与 Graph 产品代码不变。CU21 的端点反例和公开双 lane consumer 已提交，指南补清“同
Context 可互通，不等于同一有序 Queue”。有限产品 FIX 完成，进入 CANDIDATE；尚不签 CLOSED。

- Windows RTX 4070、CUDA 13.3.73、MSVC Release：`compute-graph-v7-cuda` 精确构建 CUDA runtime/
  late-failure/Graph executable，串行 CTest 的 CUDA 13/13、late failure 1/1、逐头 1/1、Graph 284/284
  全通过；旧反例失败、新候选通过，不用成功编译代替跨 stream 数值验证。
- `custom-execution-v10-source-consumer`、`custom-execution-v10-shared-cuda-consumer` 和
  `custom-execution-v10-static-consumer` 各自精确构建 `hyle_custom_cuda_sample`，串行 CTest
  `^hyle_custom_cuda_sample$` 全通过。安装 producer 只增量构建 CUDA backend，再显式安装到各自
  `C:/hyle-scratch/custom-execution-v10-{shared,static}-sdk`，不写系统安装目录。
- 每种交付都实际执行 pinned input → H2D → 两个 kernel → D2H；两个独立 Invocation/Queue 对、commands
  和 require-capture、每种模式三次变化输入，检查两条 device-wait、零 host-wait 与精确数值。
- 源码 consumer 首次增量仍失败，已定位为样例目录的旧 DLL：producer SHA256 `133d8c83…`，staged
  `fe600192…`。更新到同一 producer 后通过；installed shared 的 SDK/staged 均为 `ff91df9e…`。
  这是 [增量 staging 维护问题](../build/build-runtime-staging-incremental-v1.md)，不是 CU20 修复失败；
  本卡没有顺手修改公共构建脚本。

完整候选还需 Linux CUDA/适用 sanitizer 与候选后独立复核。没有新增 P1；必要 event wait 是正确性成本，
不能拿旧错误路线当性能 peer。当前 Compute/Graph 实现行数分别 16,996/14,135，合计 31,131；全库
686,973、自实现不含测试 126,316，统一 LOC 工具在同一候选 self-check 通过。

### Linux 候选纠正（实现前固定）

同一 `1644bdc2` archive 在 WSL native ext4、GCC 14.2、CUDA/nvcc 13.3.73 上启用真实 CUDA 后，
`-Werror` 暴露两处旧 factor 代码：`CompletionReaper::start` 的部分指定 ThreadOpts 初始化被 GCC 判为
missing-field-initializers；`cuda_memory.cc` 的 const `host_address` overload 没有调用者。两者均不改变
CU20 的证明；不是关闭 WERROR 的理由。退回 AUDITED，追加唯一 CU22 纠正：ThreadOpts 先值初始化再设置
name，删除未使用的私有 overload。无字段、公共 API、线程策略或热路径变化，复用既定验证矩阵。

配置先遇到显式空 `HYLE_COMPONENTS` 与 `HYLE_DEV_FACTORS` 互斥，随后未在 PATH 中的 nvcc 无法自动
识别；分别删除冲突的 cache 输入并显式传 Toolkit/compiler/host compiler 后，已经完成 configure。
这两项是验证命令纠正，不是产品 CUDA discovery 缺陷。

测试夹具的 `if (graph) ASSERT_EQ(...)` 另补显式大括号，消除 GCC 对 GoogleTest 宏的 dangling-else
诊断。随后 nvcc 拒绝公共 helper 传入的 host `-Werror`，由独立的
[CUDA host warning 维护卡](../build/build-cuda-host-warning-v1.md)处理，不在 CUDA 模块放宽 warnings。
