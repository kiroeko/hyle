# CPU ATen：存储与索引批量接线

<!-- MODULE REVIEW: CANDIDATE -->

- owner：`compute_provider_native`、`compute_route_cpu`、`graph_route_cpu`；baseline
  `79ded44e1307b5161f3068b65b22b9feb894d029`，Windows canonical。
- trigger：[全族审计](../../../../evidence/correctness/cpu-ggml-mature-batch/adapter-audit.md)的
  Select 宽标量、I64 Gather、宽 Fill、odd-byte stride 与 duplicate ScatterAssign 是现存合法公共域。
  现有 Torch SDK 仅接 lowp，不能把泛型 adapter 未接当作原库没有实现。Graph 的新成熟候选默认组装
  是本次具名新消费，重开其有限接缝，不重审已关闭规划器。
- 主纵切：模型专家选择/embedding Gather 与最后逻辑更新获胜的状态索引；正交纵切：普通 tensor
  全宽标量存储、广播 Select、任意字节 strides 与 exact Fill，不特化模型名。
- scope：已有 15 种 plain scalar 的 Select/Gather/ScatterAssign/Fill/Copy/Concat；精确 shape、
  public rank<=8、稠密输出、原 API byte carrier。不新增算子语义或 payload 数学。
  数值 Cast、ScatterAdd、激活/归约、bounded shape 与其它路线继续按剩余 ledger 处理。
- owner/依赖：已有 Native binary 承载已锁 LibTorch 2.8.0 CPU，新增独立 `aten_cpu::provider()`
  identity，在该 factor 已有 expert header `providers/native.h` 声明；不覆盖 Native 自研 factory，
  供后续同域裁决分别绑定。不新建 factor、不改中央 component 声明、pin/fetch、工具、SDK 选型或
  package dependency。
- 普通 API：用户继续 `compute::cpu::create_context()` / `graph::cpu::create_context()`；编译可用的
  ATen 存储程序固定排在现有 oneDNN/ggml 后、Native 前，准备时匹配，执行无搜索或计时。
  SDK 关闭时 expert factory 返回 invalid，普通 assembler 跳过；不把它当执行失败或空能力完成。
- 生命周期：原 ATen 同步 CPU 程序，HYLE checked host mappings/Completion/close；共享 immutable
  program、每 invocation 私有 mapping/Tensor slots；借用端口的 native Tensor 不跨 mapping close。
  并发用不同 invocation。Torch 私有 allocator/线程与执行时 Tensor metadata 显式为 Managed replay，
  不冒充 Prepared/no-allocation；不改全局线程设置或系统电源/亲和性。
- 数值与失败：Byte 是存储载体，保留宽整数/FP8/NaN payload/负零，不做 numeric reinterpret。
  Bool condition 按非零判真；I32/I64 索引先 checked 验界再写输出。Assign 的重复索引用原稳定 sort/
  mask/index_select 去重，再原 index_copy；不运行原库重复写 UB。exact input0 alias 只按原 copy 与
  Scatter 的实际可行域资格化；Copy 公共 trait 禁止 alias，其它 alias 明确拒绝。
  失败终态 invocation 必须重建，重建后可恢复；OOM/原库异常与 flush/close 保留真实失败。
- P0：Select/Fill/Copy/Concat O(payload)；Gather O(indices+selected payload)，必要 index packing；
  Assign O(N log N+selected payload+base copy)，包含 stable sort、winning update materialization。
  第三方内部 scratch/metadata 未给出完整上界时成本保持 unknown；不签 Native WIN、不跑自研 P1。
- 预算：一批原 API/源码诊断，有限 SI1–4；所有接线后统一 Windows/Linux 受影响测试、普通
  Direct/Graph 变化输入及 source/static/shared；最多一次具体产品失败纠正，不逐算子重做 SDK。
- 验证：全 15 scalar exact bits，rank0/rank8、广播条件、odd-byte strides、任意轴、多维与 scalar
  indices、I32/I64、duplicate last-write、empty、checked 越界输出不变及恢复；普通默认 identity；
  close/OOM/alias/独立 invocation，SDK关闭 stub，新增 expert header 独立消费；selector/closure/docs/
  diff、三交付与统一 Compute+Graph LOC。
- stop：原程序不能承载同一合同的域记录具名限制，不添自研 payload 或签 UNIQUE。此卡完成不是
  全部 CPU 收口，剩余数值候选和冻结自研裁决仍需完成。

## 有限审计与 FIX

| ID | 审计对象与允许改动 | 必需验收 |
|---|---|---|
| SI1 | 独立 ATen factory 在既有 Native header 声明，existing SDK/component 内的原存储 kernel 程序；不改 Native 数学 | 成熟 identity 可单独绑定、关闭 SDK stub、自含头与交付 |
| SI2 | Select/Fill/Copy/Concat Byte carrier，I32/I64 Gather 与 duplicate-safe Assign 原 API 组合 | exact bits、stride/rank/axis/broadcast、原逻辑更新顺序与检查失败 |
| SI3 | checked mappings、冷 slots、借用 Tensor 清理、真实完成/错误、有限 exact alias | replay、资源平衡、OOM/close、两个独立 invocation |
| SI4 | 普通 Direct/Graph 默认候选、成本/限制文档与一批三交付 | 用户不注册或排序 Provider；变化输入与下游消费、所有适用门真实运行 |

原版程序 witness 已完成：[454个具名原程序断言与边界](../../../../evidence/correctness/cpu-aten-storage-index/README.md)。
包括 I32/I64、odd-byte stride、rank9 Byte、重复读与原稳定排序选择最后逻辑写入。
所需原 API 已提供且可运行，SI1–4 固定 BORROW/FIX；其它数学差异明确 DEFER。
owner AUDITED，产品未改、产品 gate 均 UNPROVEN；不会继承原程序 witness 当普通路线通过。

### 接缝定界补充

产品草稿新增 split header 时，selector 把中央 `HyleComponents.cmake` 变更选成全库组件图风险。
本次没有新 factor 或依赖，仅为现有 Native CPU binary 的第二个 factory；在第一次编译前固定为
该 factor 既有 `providers/native.h` 内声明，factory/identity/原程序/资源/普通默认行为和 SI1–4
范围不变。中央 component 文件恢复原字节，不修改 selector。此为 AUDITED 内明确接口落点更新，
仍须验证既有公开头与 source/static/shared；没有减少算子覆盖或删除适用门。

## 七问

1. 已锁 Torch CPU 2.8.0 与既有 license/package shim，不增加下载或 upgrade policy；SDK关闭有明确
   invalid factory 与编译期程序集缺席。
2. 模型索引和普通宽存储两纵切，所有现存公开 op 回映；完整模型加载/编排不是此基础库能力。
3. 复用 Compute/Graph/CPU backend/Mem；不重开成熟运行时、SIMD 或公共治理器。
4. 原 kernel 数学和 payload copy，边界 checking/重复胜者控制；包含真实 packing/sort/metadata成本，
   成熟准入不等于性能领先。
5. public C++ factory 不暴露 ATen/C 类型；私有 pinned SDK 程序承担 native Tensor 生命周期。
6. existing Native binary、source/static/shared export 与依赖复用；新增 header 必须同一消费者验证。
7. 普通用户仅选 CPU；expert factory 用于资格与后续有限比较，禁止要求用户提供 timing profile。

## 一次有限产品纠正

候选 `685010865ec8fd43f6f2990cb5b72919515ca8eb` Windows 编译通过、Provider112/114通过。
失败原始 XML/日志保留在 evidence：SI3 的 Copy 回调误声明 exact alias，违反既有 public
`AliasPolicy::kNone`，即使原 ATen copy 可自拷贝也不能改变公共语义。仅收紧该回调为 Scatter
exact input0，测试改成 Copy alias 明确拒绝。另一个失败是测试连续复用 failed invocation：
普通 Runtime 既有合同在失败终态要求冷重建，测试验证旧槽拒绝并重建恢复，不改公共运行状态。
两者不涉及原库 payload 数学或追加算子；这是本批唯一产品纠正，SI1–4范围不变。

## 验证状态

所有产品 correctness/delivery/usability/red-team gate 尚未运行；scope/architecture 已固定，P0 成本已声明。
原 API diagnostic 不计成新普通路线接线完成或性能胜负。

## 不可变候选

`91ad443bbaa20d51535d0bfcae145ba46b07b4a4`：六组原存储/索引程序与默认组装、真实 completion、
冷 private slots、原 stable duplicate winner、checked indices与限定 alias 已固定。
Windows Provider114/114（其中新增8项批量资格）、普通CPU12通过/1旧fixture跳过、Graph15/15；
同一普通消费者变化输入的 Direct 和 op→Copy Graph 已通过。两个 invocation 用独立 CPU queue
实际并发执行12轮，各自结果与资源保持独立，tracked冷分配故障逐点恢复与 teardown平衡通过。

普通消费者最初错误要求所有六组都必须选择ATen；现有oneDNN/ggml已具备某些Copy/Select/Concat域。
仅修正该qualification假设：所有结果仍检查精确bits且必须是具名成熟identity，I64宽Gather、宽Assign、
宽Fill本批具名消费仍必须绑定ATen。不给Native fallback签新接线通过；不修改产品排序。
初始返回3保留收据与日志，此次仅消费者断言调整，不改变产品候选数学或常态成本。

Linux、公开头、SDK关闭stub、installed static/shared尚未运行，均UNPROVEN；适用验证继续串行。
