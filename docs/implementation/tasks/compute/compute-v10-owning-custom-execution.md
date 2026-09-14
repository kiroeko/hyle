# Compute v10：owning custom execution

## Closure card

- component / owner：`compute`；baseline `a82ae7dc3facc4f8e1170d9b4c64a489cd48cbf0`。
- 承接未 terminal 的 [v9](compute-v9-shape-replay-substrate.md)：C90--C92/C94 的已实现结构保留，
  **C93 完整 qualified profile identity/匹配仍在本轮范围内，未被关闭或删掉**。
- trigger：[整体架构复核 A3](compute-graph-route-unification-audit.md#整体架构复核与合同修正)已经接受：
  只有 `Operation::custom` schema、私有 ProviderAccess 和一次性 native submit，安装用户不能把自己的
  owning kernel 放入同一 PreparedOperation/Graph 生命周期。
- 主纵切：推理用户提供未进 catalog 的 CPU/CUDA kernel，与成熟 Provider 共用 HYLE Graph；正交纵切：
  外部缓冲区变换和具名副作用资源，不把模型/KV/batch policy 放入 Compute。
- 范围：C100--C103 及继承的 C93；不增加算子、不新增 public Plan/factor/FFI/global registry，不改厂商内核。
- 依赖：复用 ComputeOps schema、Mem allocator/resource、Error、私有 Util-core RefCounted，以及已有
  Context→PreparedOperation→Invocation→Completion 保活链。Graph/Native/oneDNN 是直接回归消费者。
- 验证：Windows source-tree shared/static、逐头、direct/Graph 和 oneDNN 迁移回归；当前有限实现完成后
  用仅安装头/库的 CPU/CUDA 独立 consumer 验证生命周期、失败、并发与三交付。跨平台/sanitizer/基础 P1
  与最终 foundation candidate 一起执行，不能从 Windows 增量通过推导它们已通过。
- 性能：本次 P0，记录 owning 配置和 prepared/invocation 冷分配；内置 Provider 热回调、Queue admission、
  Completion 不增加分支/查询/分配。统一 P1 仍由基础卡在 immutable candidate 后一次启动。
- 预算：一次表示审计、一次实现纠正和终局独立红队；不修改公共仪器。改变 runtime owner、接纳后重跑、
  新全局注册或热态分配是停止条件；真实 backend seam 缺口先绑定其 owner，不能借本卡越权改 CLOSED 产品。

## 能力与形状审计

| 能力族 | 当前覆盖 / 动作 | 实现合同与有限 FIX |
|---|---|---|
| 描述计算语义 | COMPLETE / KEEP | OperationId/traits/infer/resource access 由 ComputeOps 唯一拥有 |
| 持有用户执行配置 | MISSING / C100 | `make_custom_provider(identity, backend, factory, allocator)` 持有可 move-only 的 factory，复制 identity 文本；失败不移动具名 factory |
| 冷查询与选中后编译 | MISSING / C101 | const factory 的 `propose` 返回结构事实与 route；`prepare` 返回 owning kernel 或明确 outcome，复用已选 resources，不在 propose 编译 |
| 不可变 kernel 与每次运行状态 | MISSING / C102 | `make_custom_kernel` 擦除 typed kernel；`Kernel::Invocation` 由冷 create 创建；hot enqueue 是一跳 typed trampoline，不新增虚函数层 |
| 普通用户完整旅程 | PARTIAL / C103 | 公开头、指南、direct/Graph、source/installed static/shared、CPU/CUDA；没有 src include 或手写 provider 函数表 |
| 完整 route profile identity | PARTIAL / C93 | 原 v9 的 build/device/domain/boundary 资格与匹配待实现，不用本次 owning identity 替代 |

这些是复合执行机制，HYLE 拥有生命周期适配，用户拥有算法；不把 kernel 算术移入 catalog。
用户只定义 C++ factory/kernel 类型，内部擦除表是模板实现细节，不公开 vendor context/region SPI。

### 表示与成本（实现前固定）

1. `Provider` 保持现有 inline payload 大小。增加冷 copy/move/destroy 行为，私有表可选 retain/release
   configuration hooks；内置 inline 配置不分配、不引用计数。custom factory 一块分配同时拥有配置、
   identity 文本与引用计数，Context 的 ProviderSlot 复制该 owner；销毁 Context 不早于全部执行持有者。
2. `KernelRequest/KernelPrepareRequest` 明确 Device、Operation、描述、精确资源和 opaque route；span
   仅在调用期借用。`ResourceRequirement` 公开既有冷约束的同一表示，内部使用它，不复制 matcher。
   本轮 custom factory 先声明 exact point；完整 bounded 资格仍由 G101/C93 的后续实现承担。
3. factory 的 propose/prepare 为 const、noexcept，可在多个 Context 并发调用；只允许冷准备，无执行
   tensor payload 或接纳外部工作。`KernelPreparation` 保留 prepared/unsupported/resource-decline/error
   四态，不从 Status code 猜安全回退。返回的部分 owning kernel 由 RAII 回收。
4. `CustomKernel` 是 move-only 的准备结果；调用方提供 workspace/runtime/persistent 事实，复用共同
   validator。typed kernel 及其独占 Invocation 按各自 allocator 创建/销毁，不用跨 DLL delete。
   prepare 以后 kernel 只读；可变 run state 在 Invocation，借用资源仍由消费者保活。
5. 本 factory 是 **shared HYLE Queue** 的 custom-kernel 路线；delegated runtime/region 的 vendor
   adapter 仍走现有 factor SPI，不把两种 owner 伪装成一个回调。`KernelCommandResult` 明确是否已接纳
   工作。enqueue 在已 admission 的 Queue 内记录，不再调用一次性 `submit_host/cuda::submit` 产生
   第二次 admission/Completion；使用 typed native handle 时遵守对应 backend 的 context/stream 合同。
6. 把已有 prepared callback 表移入安装的 detail erasure header，模板直接产出同一 callback 表；不是
   暴露整个 Provider SPI，也不加 warm adapter→virtual 的第二跳。不保证代码 DSO 自动保活：包含用户
   factory/kernel 类型代码的 image 必须存活至最后一个 Provider/执行持有者销毁。

自然旅程：定义 schema → owning factory → `Context::create` 配置 providers → 原有 `prepare` 或 Graph
compile → 每 lane 一个 Invocation → 重复 submit/Completion。没有第二套 Graph 或 custom 调度入口。

## 预声明反例与交付

- factory move-only、Provider copy/move/self-assignment、调用方原配置销毁、identity 字符串销毁；
- 无效 identity/backend、wrong device、配置/Context/kernel/Invocation 每次分配失败及安全恢复；
- exact 描述/resource/alias 错误，unsupported 与相同 Status code 的 fatal Error 不混淆；
- proposal/artifact runtime/workspace 承诺矛盾，partial kernel 销毁，成功以后不再 propose/prepare；
- kernel/Invocation 所有权、不同 Invocation 并发、接纳前拒绝与接纳后错误 Completion、调用方先销毁
  Provider/Context/PreparedOperation；CPU 实际计算和 Graph repeated replay，不只测计数；
- 仅 public 头的独立 consumer；shared factory/type 跨 image 构造并执行；CUDA native context/stream、
  多 lane、异步错误与保活必须真机验证，未运行保持未证明；
- docs/closure/previous-ref/diff 与 change-impact；不运行全库/P1，不升级外部库。

本卡已完成 scope 与上述表示审计，以继承的 `AUDITED` 活动阶段绑定新有限集合；不是 CLOSED 的重新
审查，也不重置/伪造 v9 的未完成项。七门与 foundation gate 均未签发。

## 2026-09-14 C100--C103：公开 owning execution 检查点

C100--C102 的上述表示已实现。`custom_kernel.h` 提供 typed factory/kernel，`custom_kernel.cc` 只负责冷
owner/结果适配；原 prepared callback 表移到安装的 detail 头，内置 Provider 与 custom 使用同一个
回调层。`ResourceRequirement` 公开既有约束表示，内部 matcher 复用它，不增加平行资源政策。

源码复核确认 Context 先销毁 provider context 再释放配置，PreparedOperation/Invocation/Completion
沿既有链保活；factory name/version 在移动 factory 前复制。Kernel 和 Invocation 各自保留或使用原
allocator 的释放边界。具名 factory/kernel 在无效参数或 allocation 失败时不被移动，部分结果按 RAII
清理；unsupported/resource-decline 与同码 fatal error 不混淆。

- 11 个 custom 单元用例覆盖 move-only、const copyable、over-alignment、复制/移动/自赋值、配置文本
  生命周期、全部冷分配失败与恢复、partial cleanup、错误分类、串行声明和独立 Invocation 并发。
  direct 与两节点 Graph 检查实际输出和重复重放不再 propose/prepare，不只检查回调次数。
- 公开 [独立消费工程](../../../../samples/package/graph/custom_kernel/README.md) 不包含 `src/`。CPU
  factory/type 位于单独 shared library；CUDA native kernel 位于独立 CUDA library，普通 C++23 factory
  使用 HYLE Queue 的原生 stream。CUDA 覆盖 257 元素尾块、双 Queue、两节点 exact capture/replay、配置
  保活，以及真实 launch 后注入错误的 Completion 和实际输出。
- 示例的 scale/bias 属于 Operation 数值语义，factory 只持有不改变结果的分块配置；CPU `std::fma` 与
  CUDA `__fmaf_rn` 同为一次舍入。静态 CPU consumer 曾因 factory 要求另一 image 的精确
  `system_resource()` 而失败，已按算法实际需求改为 host-accessible/alignment；未放宽产品 matcher。
- 串行 artifact 的 direct 用法在公开声明/guide 明确：同一 PreparedOperation 的全部 Invocation 须
  串行到 Completion 终态；direct 不额外增加共享锁。Graph 约束同一 Executable 内的并发，跨 Executable
  的外部借用资源仍由调用方同步。

本次实际 Windows Release 验证（MSVC 19.50；CUDA 13.3，RTX 4070）：

| 范围 | 缓存/命令定位 | 结果 |
|---|---|---|
| Compute/CPU/Graph source-tree shared/static 与逐头 | `C:/hyle-scratch/compute-route-v8-20260914`、`compute-route-v8-static-20260914`；`hyle-windows.ps1` Build/Test，`--parallel 2` | 各 7/7；补齐 11 个 custom 用例后 shared CPU/Native 定向 4/4、static CPU/Graph 定向 2/2 和 Compute 逐头 1/1 通过 |
| oneDNN direct/composition/Graph region 与逐头 | `C:/hyle-scratch/compute-route-onednn-v8-20260914` | 9/9 通过 |
| CUDA backend、late-failure 独立进程、Graph 与 CUDA 逐头 | `C:/hyle-scratch/compute-graph-v7-cuda`；CTest `--parallel 1` | 4/4 通过 |
| CPU/CUDA 普通源码消费 | `C:/hyle-scratch/custom-execution-v10-source-consumer`；`HYLE_SAMPLE_SOURCE` + `HYLE_SAMPLE_CUDA=ON` | 2/2 通过 |
| CPU/CUDA installed shared 消费 | `custom-execution-v10-shared-build` → `-shared-sdk` → `-shared-cuda-consumer`，均在 `C:/hyle-scratch/` | 2/2 通过 |
| CPU/CUDA installed static 消费 | `custom-execution-v10-static-build` → `-static-sdk` → `-static-consumer`；真实 `HYLE_STATIC=ON` | 2/2 通过 |

源码/安装样例的两种 kernel 在统一 fma 语义后均重新通过。CUDA 示例保持一个 HYLE runtime owner，
native kernel 可以跨 image；不承诺不同 DLL 各自静态链接 HYLE 后可互换 backend 私有资源身份。
这一限制、code image/allocator 保活、exact-only 和非 delegated 适用范围均写入 guide。

P0：Provider 仍为 table pointer + 6 pointers inline payload（本机 56 bytes）；内置配置不分配、不引用
计数。custom 配置一次冷分配，Context、typed kernel、typed Invocation 各有冷存储；复制 custom handle
有冷 retain/release。`operation.cc`、Graph warm execution 和 CPU/CUDA backend 产品代码本次未改，
custom warm enqueue 直接进入既有 prepared callback 的 typed trampoline，无额外虚函数层或 HYLE 热
分配/search/JIT。未运行 P1，不据源码结构签发最终性能胜负。

此检查点完成 Windows 的 C103 旅程，不签整个 foundation gate。Linux/macOS、适用 sanitizer、终局
独立红队和基础 P1 随完整 candidate 验证；C93 完整 profile、Graph bounded/variants、统一完整成本与
constant ownership 仍在既定范围内待实现。组件保持 `AUDITED`，不重开已 CLOSED 的 backend 或其它依赖。

### C93/G103：复用资源要求校验（实现前固定）

Graph 将 region 资源查询前移后，需要在尚未选择具体 storage 时拒绝 malformed proposal。给已有私有
`OperationPreparationAccess` 增加 `validate_requirement`，只转调 Compute 的同一 validator（alignment、
矛盾 flags、exact domain）；不新增公开 API，不使用一个伪造 resource 来绕过匹配，不在 Graph 复制规则。
由 Graph 的坏提案/fatal/不提前 prepare 反例和 Compute/CPU/Native/oneDNN 直接回归验证；warm callback、
Context/Queue/Prepared layout 和 backend 不变，P0 为常数次冷字段检查，仍不签 C93 完整 profile 完成。

已实现并通过：Windows static Compute/Graph 4/4，shared Compute/CPU/Native/oneDNN/Graph 与逐头
13/13，CUDA backend/Graph 4/4，CPU/CUDA source/installed static/installed shared 各 2/2。Graph 对
三类端口的坏 alignment/矛盾 flags/无效 exact domain 均在零 prepare 时拒绝；完整记录与剩余范围见
[Graph v10 本段结果](graph-v10-reviewed-foundation-contracts.md#g103region-冷资源-skeleton实现前固定)。
