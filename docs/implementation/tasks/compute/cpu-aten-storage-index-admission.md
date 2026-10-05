# CPU ATen：存储与索引批量接线

<!-- MODULE REVIEW: CLOSED -->

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
审计时owner AUDITED、产品尚未改；原程序 witness 不代签普通路线。产品资格见下方最终裁决。

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

产品最终资格见下方；原 API diagnostic 不计成新普通路线接线完成或性能胜负。

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

Windows/Linux公开头、相同consumer的源码与installed static/shared全部通过；Windows关闭feature adapter/普通assembler资格通过，具体边界见证据。


## 逐能力语义覆盖与最终七门

[完整收据/命令/XML/失败与边界](../../../../evidence/correctness/cpu-aten-storage-index/README.md)。
本表PASS仅本卡exact/plain/rank<=8/dense-output域，不代签任何未运行模型、GPU或数值math。

| 能力/public journey | 正常与边界 | failure/OOM/overflow | ownership/teardown | concurrency/取消 | 主/正交消费 | 平台/交付 |
|---|---|---|---|---|---|---|
| Select/Fill/Copy/Concat，普通Direct→Graph下游 | PASS：全15scalar/odd stride/broadcast/rank0/8/empty/33parts | PASS：checked容量源码与tracked冷OOM；Copy非法alias拒绝 | PASS：borrowed Tensor先于maps关闭、资源平衡与close | PASS：私有槽/独立queue；中途原kernel取消NOT_APPLICABLE，无该承诺 | PASS：普通存储变化输入、UInt64 exact bits；无模型名特化 | PASS：Win/Linux测试/头及source/static/shared |
| Gather/ScatterAssign，索引状态与下游 | PASS：all scalar/I32/I64/3axes/scalar及多维indices/odd stride/duplicate最后逻辑写入 | PASS：晚越界输出不变、旧failed槽拒绝、新槽恢复；tracked冷OOM，容量源码检查 | PASS：cold私有index slots，exact Scatter input0 alias，close/归零 | PASS：两个独立invocation/queue两线程12轮；同步完成，不承诺kernel抢占 | PASS：expert精确oracle+普通UInt64/I64 Direct与Graph消费 | PASS：Win/Linux及同一consumer三交付 |
| factory/默认程序集与代价 | PASS：独立identity、旧成熟域顺序、关闭feature无ATen绑定 | PASS：明确unsupported/unknown预算与managed属性；native物理OOM只源码审计 | PASS：既有CPU Context/Completion不变 | PASS：原私有线程明确，ggml线程预算不冒充ATen局部控制 | PASS：用户仅选CPU，不给Provider排序/timing | PASS：existing header/package seam；stub不是完整SDK-OFF重建 |

| Gate | 证据与限制 | Verdict |
|---|---|---|
| scope | SI1–4固定，六族全部public入口回映；math/模型loader/其它路线不扩张 | PASS |
| architecture | existing Native binary/SDK与独立factory；Byte原程序/checked边界；CPU/Graph默认排序；无万能新executor | PASS |
| correctness | 两平台Provider114/114、CPU12通过/1旧fixture跳过、Graph15/15、公开头1/1；逐语义/故障/生命周期上述矩阵 | PASS |
| delivery | 两平台同一ordinary consumer source shared/static、installed shared/static；1332输入hash一致，Win DLL邻exe | PASS |
| usability_docs | compute/graph说明普通入口、范围、managed/private线程/未知内存与三交付；docs/closure工具通过 | PASS |
| red_team | root于2026-10-06独立一遍固定SI范围复核；Copy产品纠正、两消费者假设与failed槽纠正、原始失败全保留，无新未决产品发现 | PASS |
| engineering_performance | P0包含packing/sort/base copy/native allocation与热metadata，属性明确；不是SOTA/WIN资格，Native P1冻结 | PASS |

sanitizer/codegen测量不在本卡预声明验证；未改intrinsic/layout ABI/source hint，不补虚构测量。
native物理OOM与非coherent mapping fault未注入，只源码检查传播，不写注入PASS；受供allocator冷OOM逐点实测。
旧Qwen fixture跳过是本轮未改画像的既有缺失，本卡主/正交普通旅程独立已实跑。

最终 **CLOSED**：候选 `91ad443bbaa20d51535d0bfcae145ba46b07b4a4`，全部适用SI工程门完成。相关code没有候选后修改；
收口只证据/文档/状态。重开须具名数值/资源/生命周期/组合/交付反例或相关原依赖升级。
Compute + Graph统一实现LOC **61,764/249文件**，由不可变候选统一脚本统计。
CPU整体未完成，剩余数值成熟接线与冻结自研逐域裁决仍需后续独立有限批次。
