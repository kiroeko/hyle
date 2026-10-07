# F5 CPU自研源码清退

<!-- MODULE REVIEW: SCOPED -->

当前owner `compute_provider_native`；`compute`仅绑定两份私有RoPE生成文件的有限删除，不重审runtime。
baseline `0d96c635040e70b1ee561e51e4cb1a74169870f0`；总合同[CPU最终收口](cpu-final-closure-plan.md)。
F4已完成当前软件的有限全候选优选／工程默认与三平台CPU-only／四mixed三交付，
由[35种源码交接](cpu-final-f4-retirement-handoff.json)驱动本轮真实冗余清退，不继承旧187/17。
自研胜出只保留有资格的必要族/子域；无完整优势且有合格成熟接替的自研实现整项源级删除。
共享kernel若有胜出域保留必要分支，劣势路由不进入默认；不能用后备1000或旧证书证明价值。
真实独有动态/资源消费先成熟替代或明确用户能力取舍，不能删后才发现Mac或通用Graph失能。
SIMD/Mem独立基础保留；仅为已退役算子增的私有SIMD扩展可按真实需求一起裁决，借用原库组合不误称自研。
清理实现/CMake/私有plan/无需求入口及旧测试，新增明确拒绝和普通成熟消费者，不新增数学优化。
稳定后只受影响correctness/两入口/三交付/P0，未改变赢家数学不重复P1。

## SCOPED：有限审计、消费者和预算

主consumer为固定104 GGUF＋官方／非GGUF／MTP／DSpark的普通CPU Direct/Graph；
正交consumer为无模型身份的tensor管线、strided/empty/alias/checked-index与具名动态／严格资源消费。
MatMul/SDPA及其它已退役能力不复活；图语义、public descriptors、runtime／完成模型和Provider接缝不重构。

| 审计项 | 固定边界／输出 |
|---|---|
| A1 冗余14种 | Abs/Sigmoid/Softplus/Multiply/Maximum/Select、Sum/Mean、LayerNorm、ScatterAssign/ScatterAdd/Concat、RoPE/LogSoftmax逐语义／精度／alias／bounds／资源核对成熟接替 |
| A2 20有优势种 | 从F4实际41精确WIN点映射必要dtype／variant和共享源码分支；不以部分WIN或未资格点扩保留，不改赢者数学 |
| A3 独有与借入 | 具名declared-domain F32 RMSNorm和Copy／严格资源；View别名、Mem/SIMD基础、TorchAO/ATen/MLX/DS4原程序与codec分清 |
| A4 私有生成与owner | `src/compute/rope_kernel_impl.h`／`rope_kernels.h`仅被Native引用；有限Compute owner覆盖其删除，不删除SIMD通用机制 |
| A5 验收迁移 | 旧Native-only tests按真实当前消费者迁成熟普通入口／明确Native拒绝；现规则43点/普通19/20、失败/OOM/alias/state/declared-domain及三平台三交付差异批量执行 |

调查预算只一批上述源码／现有资格对账；不新开性能调研／AA／P1、不扩算子／依赖，不优化self。
实现前依法AUDITED固定有限FIX；真实缺成熟语义／资源或不能合法撤回的具名能力必须说明，不用删除后skip冒充完成。
需要用户取舍时先给具体源级方案和保留／撤回能力，不把所有旧benchmark严格域自动当新的必需consumer。
stop：有限FIX与适用差异验收全部结束裁决CLOSED；真正缺实现／资源取舍按协议明确，不无限研究。
