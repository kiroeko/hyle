# H3：CPU+Vulkan粗片段与offload基础机制

<!-- MODULE REVIEW: SCOPED -->

- baseline：84e8bc28724df5759cd7d28202dbe78a06e714fe（H2 terminal、Windows唯一产品真相，工作树原干净）。
- owner：compute_route_vulkan、graph_route_vulkan、新graph_route_cpu_vulkan；其他已关闭owner无具名缺陷不重开。
- trigger：用户准入CPU+Vulkan联合数学、有界异步传输/专家片段以及普通CPU存储、共享预算/准备诊断。
  原native route只有单GPU fixed Program/Tensor与同步upload/download，不能自然承担新组合。
- 不撤销C3 accepted boundary；旧WSL Vulkan十二项仍UNQUALIFIED、不代签PASS。H3是新能力新资格。

## 已核实的接缝与当前边界

1. Vulkan Context的upload/download在vulkan_transfer.cc临时allocate staging/create_run并同步wait；
   原Buffer、native容量/队列族、fence、pending owner与异常观测保活可借用，固定两槽与重用缺失。
2. 原Vulkan Program/Invocation拥有命令、managed ggml/IREE region、native输出/结果slot；
   用原native程序承载GPU片段，不复制CUDA Queue或用逐kernel万能vtable重写成熟资源管理。
3. 原Tensor retains Buffer，native_owner与borrow_ready要求same VkDevice/queue-family且已完成交接；
   外部pending semaphore不能混成ready。普通CPU权重大于staging时仍必须直接CPU计算。
4. 原Context memory_quota计actual VkDeviceMemory；不能冒充覆盖CPU、metadata、所有槽/旧lease、
   private packing的共享预算。复用现ResourceBudget并单列native容量/实际分配观察与私有unknown。
5. 原native Runtime已在fence wait时释放host mutex，独立producer可提交；单VkQueue命令没有隐含
   全部串行保证，但fence代表有序prefix，实际overlap只由原生时间线签，不从pending数量推断。

## 固定任务/候选FIX（本节不是产品实现授权）

| ID | 本包机制 | 验收 |
|---|---|---|
| H3-01 | 自然graph::cpu_vulkan入口、typed DAG冷partition/value-ID目录、required CPU/Vulkan位置；CPU/原Vulkan编译器保留完整regions/workload/全部端口合同 | CPU→完整GPU FFN→CPU与正交数据管线，原单路线guard保持 |
| H3-02 | 原native device/staging资源适配与普通CPU主存边界；精确编码/布局、同Device/queue-family、flush/invalidate/visibility | 大CPU权重不全量staging；dtype/alias/范围/empty/overflow/OOM |
| H3-03 | Vulkan Compute固定两槽WeightSlots/WeightBatch，ID/描述/容量有界选择、原始编码、原VkBuffer copy/barrier/fence与Completion保活 | 两槽/更新/旧引用不能覆盖、坏ID/format/容量、close/abandon与独立transfer/compute原生取证 |
| H3-04 | 组合每Invocation独立buffer/程序/结果，借现Sched CPU进展与原Vulkan Completion/retirement，失败排空不CPU重算 | 真数学、错误/取消/close/晚到、旧lease、未接纳和接纳后的区分 |
| H3-05 | ResourceBudget共享CPU/设备/staging/metadata/packing/所有槽及lease；现原生未知private明确strict reject或partial，冷transfer/replay/resource报告 | 去重/已知与unknown不相加、配额admission/rollback、host-wait拒绝及bind/reset失效 |
| H3-06 | 唯一cpu;vulkan集合自动选组合factor，Direct compute/Graph入口、三系统三交付、头/资产/搬迁/缺路线/指南与终审 | Windows、WSL和Air独立source/static/shared；Windows DLL同exe目录，旧入口定向回归 |

主纵切：CPU norm/projection与CPU-resident普通大权重→现成熟Vulkan完整FFN/合格量化片段→CPU消费。
正交：CPU分块变换→Vulkan矩阵/归一化→CPU结果；动态expert ID只决定有限payload，不重挑Provider。
引擎管理模型loader、KV/offload比例/层/专家cache/predict/evict；多GPU、推理引擎、算子扩张和增量图不做。

## 有界审计、复用与验收预算

- 先完成上述具名seam的一遍源码审计，以及锁定ggml Vulkan原队列/设备/资源/退休入口一遍；
  至多一项必要原生copy/fence/owner小probe。不重新翻模型或历史算子任务，不改变总体架构无具体反例。
- 公共shape/错误/ResourceBudget/Mem/Results/Sched和既有CPU Graph全部BORROW；独立counterexample
  才增补owner。技术细节可有界查原标准/锁定源码；不在本卡重写公共治理/性能仪器。
- 审计必须固定native承载、API、冷/热边界、ready/borrow、错误/弃置和资源账单后AUDITED再写代码。
- 批量接线/consumer迁移后API固定一次candidate，三系统三交付集中验收，复用相同身份不变域；
  private/source小修只补其因果半径，不重跑旧全集。build1/test1，重型run串行，Windows余量/commit
  及owned RSS guard，WSL限制12GiB unit，Air实际身份/余量，缓存/对象复用且不反向覆盖Windows。
- 正常/empty/坏ID/shape/encoding/range/overflow、payload/metadata OOM、same device/family、flush与
  visibility、两槽更新/绑定、scope total/unknown/rollback、cancel/close/abandon/accepted error、旧lease。
- 真实typed数学与原规则、源/安装头、runtime/native shader资产、真正SDK rename与missing route。
- 本包P0：cold固定程序/目录/容量，hot沿粗片段及必要copy/barrier/等待，managed private成本如实列出。
  native copy/compute actual GPU timeline证明适用机制；不因新math竞争为空制造P1或速度WIN。
  硬件/Provider无合法重叠域时列真实限制，不用mock/copy/枚举签混合能力。未运行一律不PASS。
- docs/state-only只跑closure/docs/diff；到有限预算/真实阻塞按协议裁决，不无限研究或假关闭。

## 当前状态

新身份QUEUED登记后，本卡将三个owner绑定SCOPED。尚未AUDITED、尚无产品实现或H3资格。
下一动作为固定有限native承载与FIX，然后批量实施。H2 CLOSED只作可复用机制证据，不授权改其产品。
