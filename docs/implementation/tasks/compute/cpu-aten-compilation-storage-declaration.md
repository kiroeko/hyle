# CPU ATen：冷编译存储声明

<!-- MODULE REVIEW: CANDIDATE -->

- owner `compute_provider_native`；baseline `e6a34210731f7d0e957f5470eab8e0a3988029c8`，Windows canonical。
- trigger：[实际资源审计](../../../../evidence/correctness/cpu-resource-domain-audit/README.md)的严格
  compilation预算选择Native；源码进一步证明ATen `Program`无native Tensor/代码/packing，
  `propose/prepare/destroy_program`只用supplied allocator及值描述。未填private bounds是声明遗漏，
  不是原库缺能力或Native唯一。
- scope：该独立factory所有已接exact域的额外compilation resident/peak声明；包含prepare失败与
  teardown。Program+TensorDesc仍由supplied allocator计费，retained bytes已知；native Tensor
  仅在invocation/执行阶段出现，其unknown上界与managed/private threads保持原合同。
- 主消费：普通CPU Graph在有限compilation内存预算内运行已接成熟数学；正交消费：宽storage/
  Cast/各数学族的冷propose、metadata预算不足、invocation未知预算、prepared-only仍正确拒绝。
- 有限CP1–2：CP1按源码声明额外private resident/peak=0（不是总编译零内存），更新版本身份；
  CP2一批冷事实/同普通预算consumer与三交付验证、资源限制和默认路由说明。无新SDK/公共API/
  数学/布局/执行/owner结构/Graph planner/factor/治理工具；Native/SIMD比较、清退和GPU继续冻结。
- P0：常态执行、算法、对象布局、source/dependency/flags不变；新增冷成本事实使Graph在正确的
  预算条件下绑定成熟程序。没有hot dispatch/search或性能WIN。
- 预算：一次有限源码与基线counterexample；两个FIX后不可变candidate，一批低并行Win/Linux
  owner及受影响CPU/Graph验证、同source/static/shared。最多一次具名产品纠正，不重测不相关模型。
- 预声明验收：各已接族cold bounds与managed/unknown invocation；严格compilation消费者实际
  prepare/submit/wait/结果/变化输入/close，有限metadata预算真实失败；prepared-only/invocation
  约束保持真实拒绝；两平台三个C++交付。Native factor和CPU默认suite执行，Graph执行普通
  `CpuGraph.*:CpuDefaultGraph.*-CpuDefaultGraph.D5*`；AS的D5/math/lifecycle/关闭分支资格按源码/合同/
  machine/toolchain匹配复用，不因成本声明重跑完整模型、GPU/P1。公开头未变，沿既有身份复用。
  selector/closure/docs/diff、统一LOC和本批消费目录清理。
- stop：源码若存在未计费native cold分配，不能声明0，先按有限反例纠正或明确阻塞；适用未运行
  不签PASS。只修这项声明，之后回到冻结自研的有限精确域裁决，不扩展算子或研究队列。

## 审计与有限 FIX

| FIX | 精确依据与允许改动 | 验收 |
|---|---|---|
| CP1 | Program仅Operation/描述/固定数组与span；prepare单次supplied allocate；destroy由同allocator释放，没有原Tensor/JIT | proposal额外resident/peak已知0、retained非零；失败与销毁源码；managed/unknown invocation不变 |
| CP2 | 当前ordinary Graph因adapter unknown拒绝已接成熟域 | strict compilation Graph选ATen并实际正确完成；不足预算失败；其它政策不虚报；两平台普通三交付 |

审计固定CP1–2，owner AUDITED。原Program、select/infer、prepare单次supplied allocate与值初始化、
destroy闭包已核对，无native Tensor/JIT/packing或额外private scratch；诊断Status分配按既有quota
合同排除，Context/Queue/driver控制与invocation另计。实际普通Graph baseline counterexample返回
预期5（未绑定ATen），不是PASS或原库数学失败。实现不得改变数学、布局、typed lane或invocation。

## 不可变候选与首批事实

candidate `a527dd73dcc21a5b4b89979d43339d67becabe86`：只修改propose的额外cold bounds与版本身份。
源码三个完整受保护片段（Program/全部数学与执行/invocation/mapping/lifecycle、prepare、SDK-off）
逐字节相同；1331个既有非fixture输入SHA256相同。AS未变数学/lifecycle/D5/关闭分支资格可复用，
新cold facts、预算规划与更新binary三交付仍需要本卡实际验收，不用复用声明替代它们。

Windows Provider141/141（新增一case覆盖18族cold事实）、CPU12通过/1旧skip、Graph17/17、header1/1，
strict compilation普通Direct/Graph结果/变化输入/close通过；零metadata预算、限制ATen后的strict
invocation未知预算和prepared-only未被放宽。Linux与两平台installed/static尚未签PASS。
