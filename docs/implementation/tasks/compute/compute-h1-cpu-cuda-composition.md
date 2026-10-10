# H1：CPU+CUDA 粗片段、异步传输与流式权重

<!-- MODULE REVIEW: SCOPED -->

- baseline：`061b46700`（C3按用户接受边界CLOSED）；Windows canonical。
- 当前owner：graph、compute、graph_route_cpu、graph_route_cuda、compute_route_cpu、
  compute_route_cuda、compute_backend_cpu、compute_backend_cuda、compute_ggml_runtime、
  compute_provider_ggml_cuda、graph_provider_ggml_cuda。
- trigger：用户新准入的真正CPU/GPU片段执行、有界异步传输/依赖、真实重叠和引擎驱动expert片段迁移。
- 不重开Apple/Vulkan、Mem/Sched或无具体缺陷的算子owner；是否需要定向底层修复以有限反例为准。
- C3十二项WSL Vulkan实GPU运行保留UNQUALIFIED，用户明确不block；不代签GPU数学，不阻止本H1。

## 正常任务与两条纵切

主纵切：静态typed推理子图含明确CPU-required片段与CUDA数学片段；引擎声明权重存放/允许执行
位置和本次expert IDs/范围。编译绑定真实dtype/physical encoding/layout、model/artifact、hardware/
driver/toolkit/recipe；重复运行只更新实际payload/绑定，程序固定。另验纯host权重/GPU算术的
固定容量分批路径，不能把它当CPU数学参与。

正交纵切：CPU分块数值变换→CUDA矩阵/归一化完整片段→CPU消费结果的管线；两槽在途，
保留旧结果、取消未接纳后继、独立plan关闭。无需模型loader/token/KV/专家缓存策略。

复用Graph既有IR/placement/region/resource/replay、CPU成熟Provider与CUDA原生程序、Mem backing、
CUDA streams/events/Completion、现有结果lease与Sched。Provider成熟完整region优先，不拆成逐kernel
调度；不建万能Queue、另一套后台runtime或第二套路由IR，不写数学算子或扩大模型目录。

## 需在AUDITED前固定的合同

1. Graph拥有跨片段逻辑边界、真实转换/资源/依赖、compile/runtime分离和组合终态；原普通同Device
   compile保护保持，新组合入口显式CPU+一个CUDA端点，不隐式GPU失败回CPU。
2. CPU/设备/staging/Provider-private成本分别预算；未知不当零。pageable host先显式有界staging，
   同CUDA context的pinned backing才可作为实际async源；physical encoding不同必须合格转换或拒绝。
3. 提供阻塞完成的自然基线和有界异步进展；真实独立copy/compute可重叠，而非后台线程阻塞整图。
   CPU不能消费CUDA pending事件时显式进展/等待并报告，借既有Sched而不建新线程池。
4. 至少两个独立slot；旧lease/在途DMA/native region结束前不复用、改绑或释放。输出只成功ready后发布。
   borrow资源/allocator/executor寿命可发现；取消只阻止未接纳后继，close只排空本plan工作。
5. expert迁移只提供IDs/范围验证、实际选择/搬运、固定容量暂存、绑定更新和在途保活；引擎管理
   cache/predict/prefetch/evict/模型放置/循环。变payload或expert不重选Provider；变格式/shape域则重新准备。

## 已核实的有限输入

[原生接缝预审](../../../../evidence/architecture/compute-entry-design/h1-native-preflight.md)记录具体源码：
同Device guard；CPU Queue同步/不接受async completion；CUDA transfer仅接同context原生storage；
原CPU程序ABI没有ggml backend factory，通用mixed ggml CPU backend15实际commands未启AVX2；
普通Graph输出是borrowed而非可覆盖旧lease。它们是有限集成约束，不是H1已完成或性能LOSS。

本轮先审计成熟ggml scheduler的完整region承载与现CPU优选/程序资格是否可保留；不盲目初始化
通用CPU backend。若不合格，复用现合格CPU/CUDA完整片段和原生边界，不为形式复用牺牲语义/优化。
调查上限：一遍已具名源码接缝、一次自然调用草图反证与最多一个必要的有界native实验；
固定有限FIX后实施。不无限重选架构，不以benchmark行设计API。

## 预声明验收与停止

- 一批接线后冻结不可变候选；Windows/WSL实际CPU+CUDA主/正交完整数学、三交付，build2/test1。
- 正常/空/错shape/错encoding/坏ID/范围/溢出、资源/OOM、重复payload/绑定、两槽busy、旧lease、
  accepted错误终态、request_cancel、plan close、丢弃completion后owner保活与销毁。
- 原生stream/event依赖与copy/compute进展证据，host wait/转换/融合边界完整成本；不凭async命名签重叠。
- 公开头/target/运行资产与普通C++guide；CPU-only/CUDA-only原入口与明确缺组合能力诊断定向回归。
- P0为常态账单。只在API/算法/完整费用固定且源码有未决消费者预算机制时做一次有预算P1；
  一家成熟承载不做无限Provider赛跑，复用既有单路线证据，不改公共仪器。
- 每轮docs/closure/diff；按selector真实owner/消费者选择。文档/状态不重建/P1。
- 具体阻塞说清，不能用两个Context、mixed SDK、copy/mock或空target冒充本包。

当前仅SCOPED，尚无产品FIX授权或实现。AUDITED须给出有限coverage/FIX、自然public形状、
owner/lifetime/concurrency/error与完成账单；H2/H3按其native域后续独立施工，增量图继续DEFER。
