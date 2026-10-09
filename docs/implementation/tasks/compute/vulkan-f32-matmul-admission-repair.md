# Vulkan F32 MatMul 精度接入定向返修

<!-- MODULE REVIEW: SCOPED -->

- 当前 owner：`compute_provider_ggml_vulkan`；baseline `2614676ee64bceddadcf791c218d6eea1f051c64`。
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
