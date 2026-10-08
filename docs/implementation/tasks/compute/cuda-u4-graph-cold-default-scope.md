# CUDA Graph：冷默认与完整 region

<!-- MODULE REVIEW: AUDITED -->

- baseline：`71904172`；实际owner graph_route_cuda 的旧CLOSED不授权本修改。
- trigger：U4 ordinary Graph51项中5个region结构断言FAIL，46项通过；原native region8项
  和Direct等62项通过。新的leaf工程优先级100小于既有region200，拆散成熟region。
- 有限FIX：复用CPU默认已有的Direct/Graph冷scope，Graph leaf优先级加200，原ggml
  完整region200保持；private CUDA context factory带scope，只有Graph入口传Graph。
- 资源/数学/执行表/公共API/其它路线/公共Graph规划器不改，不把数值PASS注销结构FAIL。
- 验收：5个原失败、原ggml region及Graph资源/replay/普通SGLang组合；Direct证据不重跑。
  新scope输入改变只增量编译相关factors，原51项日志保留，P1仍未启动。
- 预算：一次源码定位、一次范围修复、一次具名纠正验收；不得更换输入/数值门或重跑全集。

AUDITED：现CPU冷decorator的Graph scope已以leaf偏移200保留region，CUDA原region的
准入/contractibility/资源仍适用。冻结只在Compute route私有factory和Graph route调用传scope，
其余Context/provider ownership和callback不变，Unknown模型仍使用descriptor工程默认。
