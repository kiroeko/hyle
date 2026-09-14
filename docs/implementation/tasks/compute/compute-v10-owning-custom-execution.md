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
