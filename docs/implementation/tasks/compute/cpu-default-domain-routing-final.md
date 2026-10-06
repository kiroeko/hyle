# CPU 最终适用域裁决与默认规则

<!-- MODULE REVIEW: CANDIDATE -->

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

本卡已 AUDITED；产品施工只限下列 F1–F5，不增加算子或重开性能比较。

## 有限 FIX（源码与成本已审计）

| ID | 允许修改 | 通过条件 |
|---|---|---|
| F1 | Native新增私有默认规则factory与semantic domain matcher；真实品牌/ISA/MSVC195035728/串行条件cached于Context，Win表只含187个固定case对应op/kind/dtype/ports/attrs | 构造时硬件匹配；prepare阶段只匹配/绑定，submit不变化；无Timing表/搜索/计时，公共Options与NativePlan/Invocation layout不变 |
| F2 | 同一13个LOSScase在已测硬件/软件/串行exact请求的Native propose/prepare明确Unsupported；shared kernel保留用于对应其它WIN与已验whole-domain | 6项Bool Softmax/LogSoftmax、RMS1、Assign3、SDPA1、Softplus2；未测不是LOSS，whole-domain Copy/RMS不被exact拒绝误伤 |
| F3 | compute_route_cpu与graph_route_cpu仅换具名私有Native默认factory；pref Direct50/Graph175，非匹配沿已有mature顺序 | 原GGML region100/oneDNN region150先于175叶，公共用户无需选provider，无热选路 |
| F4 | 显式domain disposition ledger与统一普通consumer/测试 | 各族源码是否仍被WIN/真实resource使用可复核；不能以1000优先级本身保留；3小幅Cast和71非MatMul AA失格成熟优先，不签胜负 |
| F5 | 文档/状态/三交付/源码身份/cleanup终审 | Win/Linux owning与真实普通入口逐格实跑；完整Graph成熟region资格复用未变源码，并实际核对不被抢走 |

非泛化原则：只匹配当前协议的dense端口、相同arity/shape/params与资源精度，不因hidden=2048
单轴相同就误绑定broadcast、extra bias、另一axis/scale/mask/rotary layout；bounded roots不签exact WIN。
CUDA/Apple/Vulkan与第三方数学不改。187比较只证明相对本轮默认mature完整调用的收益，条件不满足
不做全局CPU架构外推。Native expert算法共享保留与默认cold规则分开，MatMul退役保持。

## 不可变候选

候选 `7090222ac6da73edab35c0f4d8d1bd098748a793`；API/算法/有限规则已固定，适用 P0 与批量 correctness/交付验收继续，
不重跑已结束 ND3 P1，不把新候选称为新计时比较。
