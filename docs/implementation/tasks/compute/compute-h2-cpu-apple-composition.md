# H2：CPU+MLX/Metal粗片段与offload基础机制

<!-- MODULE REVIEW: SCOPED -->

- baseline：`7f428d7a9`（H1 Windows/WSL完整合同CLOSED）；新factor身份登记`b842269f`。
- canonical：Windows；唯一Apple验收机MacBook Air `Mac16,12`，2026-10-10实读M4/16GiB、
  macOS27.0.1 build26A434、AppleClang21.0.0、CMake4.3.4、Ninja1.13.2。
- 当前owner：compute_route_apple、graph_route_apple、新graph_route_cpu_apple。
- reopen trigger：用户准入CPU+Apple联合数学、异步/重叠/专家片段及H1补齐的普通主存/预算/诊断。
  现单Apple执行只有原native数组/region，不能自然承担这些组合旅程；不以旧CLOSED拒绝新需求。
- 只借用现CPU Compute/Graph、ResourceBudget、Mem、Results和必要Sched；它们无具体缺陷不重开。
  需要修改额外owner时先记录具名源码反例并依法绑定，不在本卡顺手扩审算子或公共仪器。

## 已核实的缺口与承载

1. [Apple入口](../../../../include/hyle/compute/apple.h)的bind只接受成功ready的MLX数组；
   Context已有自己的提交线程及GPU stream，compile会追踪/预热MLX函数，不能在该追踪函数中
   执行CPU数学后假装CPU片段会在每次replay运行。
2. [原生计划](../../../../include/hyle/compute/detail/apple_plan.h)仅有MLX/ggml region；
   [执行器](../../../../src/compute_route_apple/apple_context.cc)在region边界检查同步。借用该原生
   owner/成熟编译region，不复制CUDA Queue、开启第二后台runtime或加入逐kernel调度。
3. [现Metal桥](../../../../src/compute_route_apple/apple_recurrent.cc)使用真正MLX allocator Buffer、
   shared views和MTLBuffer import；普通CPU resource不是MLX array，统一物理内存不取消owner
   或可见性合同。复用这条原生边界，必要copy/等待明确报告。
4. 当前[Apple Graph报告](../../../../include/hyle/graph/apple.h)只给logical constant bytes及未知
   私有存储，尚不能把CPU/全部槽/迁移和实际alias计入组合预算；复用H1 ResourceBudget和报告原则，
   不把MLX全局allocator/driver成本记零或当可预留物理内存。

## 有限审计与拟定FIX

实施前在本卡完成AUDITED，固定实际native承载/API及成本；当前仅SCOPED，不授权产品实现。
调查上限：上述已具名接缝一遍、锁定MLX的buffer/stream/Metal completion源码一遍，至多一个
有界native buffer/copy/可见性实验。不能无限重选整体架构或重审模型算子目录。

| ID | 本包需要完成的机制 | 验收 |
|---|---|---|
| H2-01 | 自然CPU+Apple typed Graph入口、显式CPU/Apple粗片段及端口描述，沿现冷规则并保留完整MLX/ggml region；单Apple与CPU入口保持 | 两条真实多数学纵切、普通入口及原guard，不用mock/copy代签 |
| H2-02 | 普通CPU主存与有owner的MLX/Metal边界，必要有界copy/转换/可见性；CPU大权重不全量复制进GPU暂存 | CPU权重大于boundary容量仍真实计算；dtype/encoding/layout/alias/范围/OOM |
| H2-03 | 引擎给IDs/范围的固定容量两槽迁移、原始编码保留、ready/依赖及旧引用阻止覆盖 | IDs/目录/容量/overflow、两槽更新、独立迁移与计算及native实证 |
| H2-04 | 复用已有Apple原生owner及必要Sched进展，异步结果/错误/取消/close/abandonment保活 | 接纳前拒绝、接纳后排空、旧lease、失败不重算CPU，无第二runtime |
| H2-05 | 共享CPU/native/boundary requested-byte预算与冷诊断，alias去重、未知私有成本严格拒绝或显式partial；bind/reset准备失效 | 正确账单、共享/所有槽/旧lease占用、回滚、host-wait策略与实际执行一致 |
| H2-06 | 普通C++ Direct迁移与Graph示例/指南，source及installed static/shared集中验收 | 真机数学/资源、changed headers、native资产/搬迁、缺路线诊断和普通CPU/Apple回归 |

## 纵切与边界

主纵切：CPU-required normalization/projection等现成熟数学→完整Apple FFN/合法native region→
CPU消费，明确放置、CPU-resident权重及固定boundary容量。正交纵切：CPU分块变换→Apple矩阵/
归一化→CPU结果，双槽/旧结果/取消。具体算子仅从已有合格域挑选，不新增数学实现。
专家选择/缓存/预测、模型loader、KV、offload比例、GPU层数及请求循环属于引擎；多GPU不做。

## 预声明验收与停止

- API/算法稳定后一次冻结候选，批量native正确性/失败/生命周期/资源与三交付；只修实际失败半径。
- 正常/空/坏ID/shape/encoding/layout/范围/overflow、有限metadata与payload OOM；至少两槽、
  更新expert/payload不重选Provider、旧lease、accepted错误、取消、close、丢弃owner与晚到完成。
- 原生copy/compute进展与统一内存可见性实际取证；不以两个pending或host后台线程推断物理重叠。
- MacBook Air重型run串行、build1/test1；16GiB机器启动前读内存余量、监控自己的验证进程，
  使用已锁定依赖/对象，传完整可追溯候选并保持未变文件timestamp，不反向覆盖Windows。
- 本包P0，成熟数学/规则不比赛；若有具体完整成本反例才另定适用有预算P1，不制造性能仪器课题。
- 当前新能力UNPROVEN，H1证据只证明已完成CPU+CUDA及可复用下层，不代签Apple组合。
- docs/state轮仅docs/closure/diff；Queued身份config11项已通过，selector中央manifest给宽superset，
  该identity未添加target/dependency/public header，未重编/重跑全库或P1。不能把scope登记算新能力完成。

完成固定FIX及适用验收才CLOSED；超范围或native承载实质缺失要列明确问题与恢复条件，不能无限调查。
