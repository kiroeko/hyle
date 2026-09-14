# ComputeOps v3：资源访问 schema

## Closure card 与已完成审计

- component / owner：`compute_ops` / ComputeOps；当前未 terminal 的 v2 在 `AUDITED` 阶段绑定本后继有限卡，
  保留 [v2 identity](compute-ops-v2-stable-custom-identity.md) 的实现和未完成交付验证，不伪造一次 CLOSED；
- baseline：`925aaf27c40c5ba05562ffcc0b5a6dadd71eb036`；
- trigger：已接受的 [Graph §4.4](../../../architecture/graph-design.md#44-按资源描述副作用) / G105
  需要按资源区分 read/write；当前 `OperationTraits` 只有 pure/未知外部副作用，不能表达独立资源；
- 主纵切：推理 custom state read/update 的执行与 shape 推导分离。SGLang
  `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0` 的 `srt/utils/custom_op.py:57` 显式声明 `mutates_args`，
  本次已确认本地 clean revision；不是声称该 revision 是最新版本；
- 反偏置：两个非模型的独立外部资源、只读与写入；参考
  [MLIR side effects](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/) 的按资源访问分类。
  该文只讨论 CFG region，不把其尚未定义的 graph-region 语义当作 HYLE DAG 调度证明；
- scope：O30--O32 的 schema 与冷验证。Graph ID/物理 alias/顺序属于现有 Graph G105；不让 ComputeOps
  依赖 Graph/Mem，也不把 schema 补齐当作 installed custom execution 已完成；
- non-goals：算子目录扩充、进程全局 resource registry、模型/KV 策略、隐式锁、RTTI、Native FFI；
- 依赖 / 消费者：Core/Error；Compute、Graph、各 Provider。普通 pure builtin 没有资源端口；
- P0：每端口 O(1) 查询，完整 schema O(R)，无成功路径分配；不增加 `Operation` payload/热执行状态，
  不在 submit 查询 schema。新增 cold traits 字段、producer vtable 槽的实际布局在候选检查；
- 预算：一轮表示实现、一轮定向纠正/红队；预声明 Windows source-tree shared/static、ComputeOps/Graph/
  Compute/Native/oneDNN 直接消费者及逐头；完整候选再做 Linux/两台 Apple、sanitizer 和独立 installed SDK；
- stop：需要全局 registry/热态 schema 扫描/把 Graph 身份放入 ComputeOps 时停回审计。基础 gate 前不跑
  新 P1，不以定向 Windows 回归冒充整个 foundation 完成。

## 有限改动集与自然形状

| ID | 当前能力 / 动作 | 实现与验收 |
|---|---|---|
| O30 | MISSING / FIX 资源端口读写 schema | `EffectKind::kResourceAccess`、`ResourceAccess::{kRead,kWrite,kReadWrite}`、`OperationTraits::resource_count`；descriptor 用 `resource_access(index) const noexcept` 按值声明，端口序号不是 Graph ID 或 tensor input 序号 |
| O31 | PARTIAL / FIX 统一冷验证 | `operation_resource_access()` 返回按值 Result；`validate_operation_effects()` 拒绝未知 enum、pure/未知外部副作用带端口、资源模式无端口、缺少 callback、非法访问值；infer 与 symbolic Graph 均不能绕过 |
| O32 | PARTIAL / FIX 消费与交付说明 | 原 producer callback 跨 shared 使用；最大端口边界、wrong ID/size、descriptor copy/lifetime、Graph 保守行为、原 pure/未知效果回归；更新 guide |

resource count 最多 255，与已有小型 inline operation schema 的有限 arity 同属冷描述合同。声明的访问必须
完整且确定，不读取 tensor payload、不执行工作、不分配。只读资源也不是 pure：它会观察外部可变状态。
未知副作用继续用 `kExternalSideEffect`、零具名端口，不能用空列表谎称已证明独立。用户不需要为普通
算子实现新 callback；拥有资源端口的 descriptor 则必须提供精确的 `noexcept`、按值返回接口。

Graph G105 逐步接入期间，尚无域绑定的资源访问必须保守处理/明确拒绝，不能静默按 pure 编译。
Compute direct 路线由调用方 Queue/Completion 管理外部资源生命周期与先后；schema 自身不是资源 owner。

## 七问与当前裁决

无新第三方/FFI/组件边；复用 Error 和已有 inline custom callback，不创造新的 dispatch 系统。功能缺口
来自已接受的 consumer 架构，不由 benchmark 产生。公开头由 ComputeOps owning factor 导出，三种 C++
交付保持同形状。当前 `AUDITED`，尚无新的 Candidate/终局正确性、性能或交付 PASS。

## 2026-09-14 O30/O31 实现检查点

资源 enum/count、按值 producer callback、查询和统一验证已实现。Graph 的 symbolic-output 路径与 Compute
bounded proposal 都在可能跳过 concrete inference 的位置显式复用同一个 checker；没有新增 kernel 或
submit 中的资源扫描。Graph 暂把新 resource effect 与未知副作用一样串行检查，不能静默当 pure。
G105 的 graph-local identity、物理 alias 和独立域并行仍须接着完成。

反例覆盖三个访问值、复制后的原 descriptor 改写、最大 255 端口、越界 callback 不执行、坏 effect/count、
任意端口坏 enum、缺方法/可能抛异常的方法、producer DLL callback、exact/bounded 在 Provider 前拒绝、
symbolic Graph 不漏检以及两种副作用的双向混合链。源代码第二遍反证特别确认：callback 结果是 enum 值，
不会返回解码栈对象的 span；新增方法必须 `const noexcept`；无 Graph/Mem identity 泄漏至 ComputeOps。
这不是完整 foundation 的终局红队/异步资源执行证明。

Windows MSVC 19.50、Release、每次最多 2 个编译任务，使用现有独占 cache：

| 配置 / 验证 | 本次实际结果 |
|---|---|
| `compute-route-v8-20260914` shared：ComputeOps/Compute/CPU/Native/Graph + 逐头 | 11/11；293 单元用例通过 |
| `compute-route-v8-static-20260914` static：同范围 | 11/11；291 单元用例通过，两个跨 DLL fixture 只适用 shared |
| `compute-route-onednn-v8-20260914` shared：direct/composition/Graph oneDNN + Graph executable/逐头 | 8/8；149 单元用例通过 |
| 同 shared cache 的 Graph-only configure，CUDA/oneDNN OFF | 通过；无新增 option/依赖，OFF 不获取依赖/探测 SDK |

可复现入口为 `scripts/hyle-windows.ps1 -Action Build/Test -BinaryDir C:/hyle-scratch/<上述目录>`，Build
targets 为 `hyle_compute_ops_test,hyle_compute_test,hyle_compute_backend_cpu_test,hyle_compute_provider_native_test,
hyle_graph_test,hyle_graph_executable_test`；Test filter 为
`^hyle_(compute|compute_ops|compute_backend_cpu|compute_provider_native|graph)(_header_self_contained|_executable)?_test$`。
oneDNN targets 为 `hyle_compute_provider_onednn_test,hyle_compute_provider_onednn_composition_test,
hyle_compute_provider_onednn_graph_test,hyle_graph_provider_onednn_test,hyle_graph_executable_test`，filter 为
`^hyle_((compute_provider_onednn|graph_provider_onednn)(_[a-z_]+)?|graph_(executable|header_self_contained))_test$`。

P0 结构检查：新增一字节 traits 字段（包含其自然 padding）和每个 custom descriptor 的静态 vtable 函数
指针；`Operation` 本体、PreparedState/Invocation 未加字段。查询每端口 O(1)，完整验证 O(R)、成功无分配，
新 checker 不进入 submit。旧 direct alias 检查原本就会取得 `OperationTraits`，现在返回值包含新字段，
因此不宣称完整 direct 机器码逐字相同；完整 foundation 的公平性能门仍按总卡执行。

impact selector READY（82 项 skipped/reused），closure（含 `--previous-ref 925aaf27`）、docs 和 diff 检查
通过。没有运行 Linux/macOS/sanitizer/真实设备/独立 installed SDK/P1，不把 source-tree shared 测试当成
安装包资格；O32 的完整交付门仍未裁决。保持 `AUDITED`，不是 ComputeOps 或 foundation CLOSED。
