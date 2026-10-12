# F5 paged/ragged 与显式批次规划

## Closure card

- 状态：SCOPED，仅本轮审计/有限native原型；尚无产品实现/API资格。新compute_batch与
  graph_batch已在0cc357684以QUEUED登记；本卡绑定同一F5六FIX，不缩成单次read。
- baseline：db8861394（F4已CLOSED，candidate29d3c7a3e）；A/F1–F4与R1–R7原范围不重开。
- trigger：用户活动goal、[固定peer研究](../project/foundation-f45-reference-research-2026-10.md)、
  [F5设计](../../../architecture/compute-state-paged-design.md)，原dense FlashInfer没有真实batch
  page端口、原普通Graph Invocation没有per-layout preparation接缝。
- consumer：单机单/多请求的packed Q、显式paged/ragged KV→host-authoritative layout版本→
  原成熟native prepare/update/run→独立slot/输出保活；append/读取→F4 branch/成功发布。
  正交纵切为不含token/模型策略的分段矩阵/资源批次，相同capacity/有效区/metadata lifetime。
- owners：compute_batch负责typed容量/有效区/schema、owned metadata版本和批次生命周期；
  graph_batch负责opt-in规划协调，绑定原Graph region的同一invocation/workspace/token。
  Compute不得反依赖Graph；普通19版region/Invocation无新增无条件字段或warm规划。
  原Provider若确需增BatchPlanning extension、Graph若需窄private访问桥，必须独立把其
  实际owner按本trigger先SCOPED/AUDITED，不凭本卡改CLOSED产品。路线资源/执行仍原owner。
- 承载：Linux CUDA先借锁定FlashInfer0.6.18/69ff11fc原C++页结构/scheduler/kernel，不带
  Python/Torch runtime或SGLang私有猴补。CPU/Apple/Vulkan显式选择原native gather/materialize
  参考档，单独计copy/KV/mask/workspace/host wait，不叫同吞吐native paged，不偷跑CPU。
  Windows CUDA native paged当前未准入，保持具名pending/unsupported，不拿WSL代签。
- 固定语义：Paged与Ragged为独立typed schema；F16/BF16、F32 accumulation、checked GQA/
  head/page/layout/causal域。物理capacity与active counts分离，host快照/device mirror/plan
  token同generation，接纳前校验indptr/page ID/tail/shape/资源/overflow。空域明确原数学定义
  或拒绝；padding不是KV。layout更新不能凭shape/地址复用旧plan。
- 冷/热：容量/算法域与有限slots冷准备，每layout显式prepare_batch核对/复制metadata及原
  native planning/H2D，然后run只执行已准备批次。原native host临时vectors/occupancy/private
  bytes如实计账，未知不写0；无submit隐式plan/JIT/fallback/无限bucket缓存。
- 生命周期：PreparedBatch不可伪造且绑定原slot/generation；in-flight/旧输出阻止覆写。已
  接受H2D/native prefix失败或取消不发布旧token；原F1/F2/native证明退休后才能reset/reuse。
  shared page写前由F4显式clone/私有化，原append/scatter需实际writer/effect权限。
- 非目标：模型格式/token/RNG/scheduler/admission/page/LRU/eviction策略、MLA/压缩稀疏/
  所有KV量化、任意rollback、通用增量图、万能Queue/runtime、新math/copy kernel、公共
  build/selector/perf工具重写、自动P1、完整引擎/F6。
- 预算：两轮有限owner/API审计；六FIX。先一个固定原C++ paged/ragged lowering原型，
  native build1800/run120秒；各产品gate configure300/build1800/test120/consumer180，
  sanitizer build1800/test120。每gate原失败/时间累计，不因目录/候选/提交重置；到达预算
  BLOCKED/INCONCLUSIVE，不追加新kernel或无限外库research。owner/生命周期/常态成本
  新事实改变则退SCOPED/AUDITED，AUDITED冻结接口后才写产品。
- 机器/范围：Windows CPU/CUDA/Vulkan、WSL CPU/CUDA与san、实际Air CPU/Apple/Vulkan；
  Mac16,12先核对，Mini不可用；WSL软件Vk只包。只新batch/真实直接消费者/必要新package
  seams与实际需改的旧owner，不回放原算子全集。重型跨机器串行，库j4≥12GiB/j2≥8GiB/
  j1≥4GiB、test1，复用身份匹配source/cache/独立配置binary tree。
- 性能：P0先核capacity/metadata/plan/gather费用和常态路径；仅不可变候选+真实hotspot+
  同合同才另定有限P1。稳定地址/固定batch size不能签capture，topology需原native独立证明。
- 预声明风险：乱序physical页/非整尾/非uniform Q/GQA/causal右对齐；同shape不同IDs/layout
  generation/错indptr/坏页/尾/overflow/empty拒绝；host snapshot/device mirror同版本，payload/
  metadata/plan native use保活、两slots/旧输出、pending producer、OOM/partial upload/native
  error/cancel/abandon/unknown/suffix/close/drain；实际append/read/F4branch/发布/旧版本。
  Direct与Graph同schema/同slot真正运行，对独立dense oracle；完整计入显式参考成本。
  三交付/新头/缺路线/迁址/loader、适用san/并发、guide、P0/LOC/分离红队。每轮docs/closure/
  diff，registry明确previous-ref；仅适用风险可NA，未完成MUST不以NA隐藏。
- stop：必须按活动预算终局或退明确阶段；整个goal完成后再做用户要求的剩余基础库/引擎
  自实现代码量评估，scope不扩完整engine。F4/CLOSED证据不代签F5数学或整引擎吞吐。

## 本轮有限FIX

| ID | 完成定义 |
|---|---|
| F5-01 | typed paged/ragged数学端口、物理容量/有效区、owned host layout快照/同版本device mirror |
| F5-02 | 原FlashInfer C++真正native plan/run、独立slots/workspaces，实际平台/资源资格 |
| F5-03 | prepare/update失效/错误/实际prefix/metadata H2D/退休，旧token不能误run，capture单独资格 |
| F5-04 | opt-in版本化BatchPlanning与Graph原region同slot协调，普通19版region/Invocation不改warm合同 |
| F5-05 | 明确native/显式materialize/pending，actual append→read→F4branch/成功发布及独立oracle/成本 |
| F5-06 | 四自然route Direct/Graph、source/static/shared/新头/缺席/迁址/运行库，风险/guide/P0/红队 |

当前先核原C++参数/计划owner和公开C++自然草图，固定API/依赖/权限/slot失效后才进入AUDITED。
existing Operation::custom的64-byte typed schema可避免改旧variant；原ProviderExtensionRef可借
原Context/Provider保活与admission，但它们的存在不能代签batch完整journey或Graph规划资格。
直接修改旧owner前单独登记其合法scope；不为方便把新batch和所有旧route造成反向环。

QUEUED登记只加两identity，没有源码/头/target/产品root/依赖边。selector保守选全图的原plan
在.scratch/foundation-f4/f5-queued-plan.json；本轮只执行中央身份/状态/docs，不构建未影响
模块/全算子，不修改公共selector缩表。后续真实product seam扩展按本卡再选适用门。

## SCOPED native原型事实

[锁定C++ lowering原型](../../../../evidence/architecture/foundation-f5-native-probe/README.md)
实际PASS：直接原FlashInfer PrefillPlanWorkspaceSize/PrefillPlan与paged/ragged原kernel，
F16/BF16四场景对独立double dense oracle。head64/GQA2，两request Q长1/2、KV长19/10，
page16、物理页2/0/3、尾3/10，padding填64，causal右对齐；最大误差F16约1.15e-4、
BF16约1.01e-3。原planner选CTA16/no split，float workspace0/int64bytes只适用于这个域。
WSL RTX4070/GCC14.2/NVCC13.3，build16.807/run0.365秒计预算，不是性能计时资格。
原代码/锁定输入hash/精确身份/命令与日志封存；没有Python/Torch runtime或新kernel。

原型只证明承载，不签HYLE lifecycle、Graph同slot、其他head/layout/capture/私有allocation。
Graph现存v19 PreparedRegion/RegionInvocation确实拥有opaque slot和workspace，外部规划一个
独立Direct slot不能冒充它；opt-in桥必须访问这同一原slot。Apple/Vulkan的原typed Program
同样保留自身native owner，不能为统一签名转成共同Queue/CPU数学。原native原型后继续
有限owner/API审计，状态仍SCOPED，六个FIX产品未开始。
