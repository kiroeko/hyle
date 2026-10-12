# F5 CUDA公开Provider普通产品链接修复

## Closure card

- 状态：AUDITED；owner compute_route_cuda；baseline82caf4942。由CANDIDATE合法退回，
  原scope与失败见[反证](../../../../evidence/correctness/foundation-f5-cuda-shared-link-failure/README.md)。
- trigger：Linux shared安装包普通Compute/Graph的公开batch消费者未解析
  flashinfer::provider()；对应头与HYLE_COMPUTE_FLASHINFER能力宏确实供应，static已实际运行。
- consumer：公开batch schema→原FI Provider→BatchProgram与Graph batch opt-in→真实Direct/Graph。
  正交为已准入CUDA原Provider factory的普通C++资源批处理，不涉及模型/页调度策略。
- scope：只修compute_route_cuda的FlashInfer直接边可见性及未准入平台的安装投影。没有新API、
  kernel、Provider顺序、普通warm字段、runtime或公共构建工具改写；中央graph仍唯一声明。
- 准入裁决：公开factory的自然source/static/shared消费是既有F5-06 MUST；
  SELF只声明边，原Provider/数学/寿命BORROW，不把内部target选择负担推给消费者。
- 依赖/消费者：既有compute_provider_flashinfer→compute_route_cuda→compute/graph产品。
  不重开CUDA kernels/ggml/torch或A/F4。其他19 owner继续固定77df候选；本边修复完成后
  统一新candidate并按实际影响恢复证据资格，原FAIL/elapsed绝不重置。
- 预算：一次有限源/target审计，一个FIX；两Linux cpu,cuda SDK同root、tree、身份、configure300/
  build1800/consumer180的原累计继续。非Linux缺席/安装与新头资格按原18格实际复核；
  CPU san不依赖CUDA边，继续固定77df原预算。P0只link可见性，不启动P1。
- 验证：Linux普通source/static/shared不补内部target，D256四域Direct/Graph oracle；
  Linux package interface实际包含供应factory的原Provider，Win CUDA-off/无FI准入仍缺席且
  SDK不依赖未导出placeholder。selector selected package/component gates、docs/closure/diff。
  公开/API数学及caller/native风险已过证据按输入匹配复用，不回放旧算子全集。
- stop：原预算/停止协议；本次实际产品反证先退AUDITED，不伪装fixture错误或BLOCKED。

## 能力与形状审计

公开factory header供应、能力宏和符号export成立，PARTIAL在普通shared link closure：
central边PRIVATE使shared消费者没有显式FI库，而static的LINK_ONLY传递闭包偶然能解析。
明确修正自然C++产品发现的public usage requirements；不把算法修好写成包已修好。

七问：锁定FI0.6.18/69ff不升级；真实batch及普通资源批处理驱动；复用原Provider/Queue；
无数据layout/拷贝/atomic/lock/常态调用增量；仍无C/FFI产品；三交付须实际普通产品验证；
只本owner一条直接边与缺席投影，不重写公共instrumentation。

API/usability冻结：已有Context::create(device, flashinfer::provider())与graph::batch::provider
草图不变，消费者只链接hyle::compute/hyle::graph。借用context/resources、old result、
取消/失败/实际drain合同全部保持原F5；不新增封装API或捕获承诺。

| FIX | 固定改动 | 验收 |
|---|---|---|
| CUDA-PUBLIC-LINK | 中央compute_route_cuda的既有FI边改为PUBLIC；owner目录对非Linux未准入placeholder只保BUILD_INTERFACE，静态/共享安装均不要求未导出factor | 同root失败shared自然消费恢复，static数学保持；Windows无FI宏/符号与SDK缺席保持 |

候选卡原20owner范围、F5六FIX、其余SDK/san/P0/LOC/guide/红队保留，单个link PASS不签整个goal。
