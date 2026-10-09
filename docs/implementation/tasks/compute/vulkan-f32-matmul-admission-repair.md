# Vulkan F32 MatMul 精度接入定向返修

<!-- MODULE REVIEW: CANDIDATE -->

- 当前 owner：`compute_provider_ggml_vulkan`、`compute_ggml_runtime`；baseline `2614676ee64bceddadcf791c218d6eea1f051c64`。
- 合法 reopen trigger：C3 普通 Direct/Graph FFN 的实际 F32 correctness 反例，见
  [失败收据](../../../../evidence/correctness/compute-unified-c3/windows-vulkan-failure/receipt.json)。
  不回写旧 CLOSED terminal，不恢复历史算子/性能队列。
- 当前直接依赖 `compute_ggml_runtime` 仍绑定 C3 候选；若审计确认须改其原 Vulkan 接入，
  实现前按协议将该 actual owner 返回 AUDITED并重新绑定有限范围，不能直接改候选。

## 正常旅程与事实

主纵切：普通 F32 gate/up→SiLU乘法→down FFN，再接 application 的 native affine；
正交纵切：F32 dense/转置 RHS MatMul 保留输入精度、F32累加与原生资源。
现数据 gate/up 常数为可精确表示的1/64；rows1通过，rows16首输出不通过原数值门。
只对 down 激活做 F16舍入的独立模拟与GPU首输出完全一致。
源码已确认原 ggml 既有 F32 storage/F32 accumulation 的半精度内部计算路径，
也含原 `matmul_f32_f32_fp32` 程序；不能把F32 accumulation等同于F32输入求值。

Graph 另有旧样例断言：实际5节点/2native/1managed，而样例仍固定4native。
成熟 region 应保留，不能为旧计数拆坏融合。此项属于消费者断言迁移，不签数值修复。

## 有限审计和允许方向

一遍检查原 ggml precision API、原全精度 shader/pipeline接缝及现 RegionRecipe/桥；
固定一个成熟完整F32路径的接法后进入AUDITED，才改产品。优先接现有原版程序，
不写新MatMul，不放宽误差，不要求用户设置GGML环境开关，不全局撤回合法F16/BF16/量化能力。
min-source precision/原生全精度 pipeline 的选择必须在冷编译/固定程序内表达；
运行期不重新搜索Provider。若原路径不满足完整语义/资源，先具名记录而不假称可用。
其它算术、模型目录、CPU/CUDA/Apple算法、公共治理和性能仪器不扩审。

## 验收、成本与停止

- 原误差门 `1e-4 + 4e-5*abs(expected)` 保留，真实Direct和Graph rows1/16、三次改变输入均须通过。
- 增补使隐式F16舍入可见的实际F32数据/范围；核对RHS转置、完整region、资源复用/保活及错误边界。
- 对实际改动 owner 跑定向 correctness；从已完成source objects恢复，不重启会删root的整批driver。
- Windows/Air真实native Vulkan三交付，WSL记录实际device/适用性，不能冒充GPU；其余原门继续C3矩阵。
- P0列原全精度程序的成本、转换、同步和融合边界；无自研/Provider赛跑，不比较半精度与全精度赢家。
  API/路径稳定后若新事实需要P1，必须另冻结同语义完整成本和有限预算，不能循环计时寻找PASS。
- 审计预算一遍接缝和一次具体反例资格；最多一个有限产品修正候选。
  不可承载则写精确缺口/恢复条件，不以“继续研究”无限追加姿势。
- 本卡不代签C3全部交付、H1–H3或全模型容量；完成相关验收后归档实际结果和统一LOC。

## AUDITED：固定有限FIX

原ggml的`ggml_prec_set_acc(F32)`只限制累加，不限制输入缩窄；原Vulkan subgroup F32/F32
map在Float16设备上选择半精度内部程序，原完整`matmul_f32_f32_fp32`程序已生成但只在
FP32-only fallback登记。精度API还有`ggml_prec_set_src(...,F32,1)`；HYLE应表达其普通F32输入合同。
原库既有通用subgroup tile、pipeline创建/资源检查和dispatch继续承载，不能新增乘法算法。

| FIX | 有限产品动作 | 验收 |
|---|---|---|
| F32-01 | 原Vulkan接入登记完整F32/F32原shader及其原subgroup tile，避免Float16设备覆盖该完整路径；其它dtype/量化map不变 | 原FFN rows1/16；非F16可表达的F32输入；已有F16/BF16/量化定向回归 |
| F32-02 | 普通plain F32 MatMul冷构造表达source精度最低F32，累加仍F32；不改外部API、Provider搜索或replay | Direct/Graph相同精度、转置RHS和重复运行 |
| F32-03 | Graph样例按成熟region实际覆盖核对报告，撤掉旧自研dispatch数量假设；数值门不变 | 5个live节点、实际native/managed边界和输出正确 |

`compute_ggml_runtime`因拥有build-local原Vulkan port进入AUDITED并绑定本卡；当前CPU/CUDA源码
和规则不修改，未启用Vulkan的recipe不包含该backend manifest。返修后冻结同一有限候选再验适用门。

## 固定实现

F32-01接入原FP32 normal/ID shader，借原通用subgroup三个tile并按真实FP32 shared bytes与
workgroup limits筛掉不合法tile；原F32/F32半精度map不再覆盖它，其它dtype/量化map未改。
F32-02普通plain F32乘法请求原precision API的source最低F32，仍保留F32累加。
F32-03只要求实际native/custom和managed边界存在，不再假设旧自研4dispatch；原数值门未改。
依赖仍为原锁pin，没有升级或新shader/乘法算法。原port result/tree/NOTICE身份已更新；
原上下文匹配唯一性检查通过，selector产品授权和docs/closure/diff通过，数学尚待候选实跑。

产品candidate：`18593789ad90a9823d37d0b11b470db28bce5cb9`。只恢复当前source-build的真实
Direct/Graph目标，再验原数值门；原失败收据保留，不能改标成PASS。

该候选真实编译通过，Direct/Graph数值仍FAIL，同rows16首输出未改善；不重置或抹掉该失败。
定位到同F32-01遗漏：原`ggml_vk_mul_mat_q_f16`在coopmat2时先对F32 src0/src1做F16转换，
使完整F32/F32 map无法命中。唯一额外纠正是：对明确请求F32 operands的plain F32/F32操作
撤去这项强制缩窄，真实non-contiguous处理仍保留；Grouped plain F32同时表达source最低F32。
原“一个修正候选”记录为已消耗，当前明确追加一次同FIX的限定完成，不换库/模型/误差/仪器。
此完成候选若仍不满足原数学门，停止追加姿势并按具体事实重新裁决。无P1，CPU/CUDA不重测。

同FIX限定完成candidate：`c96c7c6eefb41f9aca71c92b4304814a731fe088`。
显式F32 operands时不做coop的F16预转换/decode-vector暂存；真实非连续条件、其它dtype及量化
路径仍保持，Grouped plain F32同样使用原source精度API。实际数值验收仍待运行。
