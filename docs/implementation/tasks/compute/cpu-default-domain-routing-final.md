# CPU 最终适用域裁决与默认规则

<!-- MODULE REVIEW: SCOPED -->

- owners：compute_provider_native、compute_route_cpu、graph_route_cpu；baseline `f6470870d3de7f2475e245b6ee2ca76905f82249`。
- user trigger：完成当前CPU goal；成熟接线及MatMul退役已实际关闭，不重开GPU/旧大队列。
- 当前consumer：本地Qwen3.5-2B hidden2048 token/small batch数学与真实普通C++数值/索引管线；
  原ND3完整操作P1已QUALIFIED，187实用优势、余域成熟优先；不宣称所有peer/所有shape全局SOTA。
- 现状：默认CPU仍只按整Provider优先级，未利用这些已证明完整收益；逐dtype/kind/shape保留
  共享数学代码不能由1000后备本身决定。13剩余LOSS场景的源码均与小shape/其它mask的WIN共用。
- 纠正：softmax variant=mode+2*BoolMask，6项LOSS是三dtype Bool Softmax/LogSoftmax rows32，
  不是none-mask log。原观测/估计不改，只纠正人读解释。
- scope：一张完整domain ledger，冷匹配Native优势，13场景在同已测HW/SW/串行合同下Native
  exact propose/prepare明确拒绝；whole-domain RMSNorm/Copy与原借用数学保留；无算法优化。
- default：ordinary用户不排序provider/输入timing。Windows/MSVC已测CPU/单线程、精确dense端口
  与kind/attrs条件匹配才给Native优选；其它域保持成熟优先，不从tensor payload读值或猜赢家。
  Direct优选在oneDNN前；Graph叶优选在原成熟完整region后，禁止叶胜负拆成熟融合。
- HW事实：Platform补最小cached CpuIdentity（x86 vendor/brand/family/model/stepping；其它未知），
  只扩已有baseline/LTO-off探测，不在Provider写CPUID/syscall。当前Native依赖已含Platform。
- private helper/factory属于Native；不新增普通用户开关/公共路由对象、不改万能queue/vtable、
  Graph planner/治理/性能统计核心。Compute/Platform其余已关闭能力不重审。
- 预算：一次批量施工/候选/适用验证；失败先查源码，最多一批定向纠正；优化0、不重跑ND3 P1。
  保留原输入、算法、workspace、Invocation layout和热submit；准备/Context只增加有限匹配。
- 验收：Native factor、普通Direct/Graph最终绑定、13拒绝、dtype/attrs/geometry/版本错配成熟fallback；
  图region保持、changed replay/并发/close/OOM复用未变原程序加实际默认入口；两平台三交付；
  platform事实/header与CPU身份实际一致。冻结SDK成本证据按相同算法/请求与已测scope作规则依据，
  不把新default candidate伪写成刚做过P1或跨平台WIN。
- docs/closure/selector/统一LOC/垃圾清理；所有适用门实际完成才关闭CPU goal，GPU继续暂停。

产品代码尚未授权；下一步AUDITED必须把有限FIX、匹配条件及保留理由固定，不无限研究。
