# CPU混合FFN：撤回错误自研gate资格

<!-- MODULE REVIEW: AUDITED -->

- owner `graph_provider_onednn`；baseline `9489dd2af0e698e13ff5951c346b9a06e4215799`，Windows canonical。
- trigger：普通四节点F32 FFN完整输出在−100/value1e38误差约1.7%；该region内部是HYLE SIMD SwiGLU，
  不能以oneDNN身份代签成熟gate。见[实际诊断](../../../../evidence/correctness/cpu-onednn-hybrid-ffn/README.md)。
- scope：退役旧通用四节点FFN提案、artifact、执行和仅此路径的HYLE SIMD回调；保留有效oneDNN norm
  region与原MatMul路径。共享Brgemm成熟helper不在本卡内重写，SIMD本体和其冻结算子函数不改。
  不是优化自研gate、增加数学核、重排provider或重开全模型/全库。
- consumer：普通constant-weight FFN的完整数值、变化输入、两个独立invocation与真实完成；正交Norm
  加scale/residual的有效region回归。用户不注册/排序，准备匹配已资格mature leaf/其他有效region。
- finite NX1–3：确认所有旧FFN代码与仅此callback的live引用；退役该错误通用region；迁移显式旧
  whole-provider/step-count假设并验证真正成熟gate与最终输出。没有Native任意fallback代签。
- tier P0：必要正确边界/融合损失和ATen managed转换/临时数据全部计入，不以错误旧fragment做公平
  性能基线；Native/SIMD优化、比较、清退继续冻结，SIMD原公共能力保留。
- 验证：graph_provider_onednn/CPU Graph直接受影响测试；旧oneDNN norm/mapping/OOM/close/并发回归；
  默认whole FFN在−80/−90/−100、large/Inf、普通变化输入/独立槽的真实结果；Win/Linux三交付同一
  consumer，声明prepared/managed、资源与SDK限制；selector/closure/docs/diff与统一实现LOC。
- 预算：只使用已完成一批actual probe与已有原程序资格；冻结NX1–3后一次产品批量修改/编译；最多
  一次具体纠正，j2重型串行。测试维护只重跑受影响/实际失败，旧一致输入证据可复用；不重跑TC
  四组unit、旧模型、P1或扩建仪器。源码/动态/静态交付在本卡完成。
- stop：不合格generic route不能保留为默认/SOTA；新的范围事实另绑owner，缺验不能PASS。此卡
  完成也不是CPU全部完工。未审计固定FIX前不改产品。


## 固定审计/FIX

源码live引用已核对：MlpPattern/match_pattern、WorkspaceLayout/RegionPlan/RegionInvocation、
contains_constant、旧create/destroy_invocation与submit/apply_swiglu、kPreparedVTable及
compile_region_checked均只服务旧四节点混合FFN。UseTable/build_use_table和NormPattern/
match_norm/propose_norms及NormRegionPlan/compile_norm保留，不能误删有效norm融合。
共用Brgemm source/helper与SIMD公共函数不改；并非宣称这些函数性能落后。

| FIX | 固定动作 | 验收 |
|---|---|---|
| NX1 | RETIRE_INVALID_QUAL：删除仅此FFN的提案、私有artifact、workspace/执行及HYLE SwiGLU调用；不留可误选旧路线 | expert region-only不再提案FFN；普通Graph实际mature gate与完整−100/large输出正确 |
| NX2 | KEEP/BORROW：保留原oneDNN norm region与已资格MatMul；自然Graph经有效region/qualified leaf组合，不新建跨provider逐kernel执行器 | scale/residual norm region/alias/close/OOM/两个invocation回归与Direct同合同 |
| NX3 | FIX：迁移旧whole-oneDNN/step-count属性，新增公共实际完整FFN witness、变化输入与独立槽，批量三交付及docs | 不接纳任意Native数学为成熟；旧性能历史不当新WIN；跳过/未验不签PASS |

原oneDNN JIT exp截断源码只是最初有界疑点；实际旧完整FFN没有调用其swish，而调用HYLE SIMD。
确认的机制是Sigmoid先F32舍入再乘x，亚正常负尾部丢失SiLU有效位后被value放大；已有ATen原
Double SiLU→F32 activation→F32 value的完整阶段已由MN/TC资格化。不能为保留旧fusion放宽规范，
也不在此优化SIMD或引入新的HYLE数学。AUDITED后只实施NX1–3。


## 实施状态

产品`80c9ab0075c667bd16255d43d46a21d46df234ac`已退役旧generic FFN route/artifact/执行与HYLE SIMD
callback，保留norm全部数学/生命周期、共享oneDNN MatMul与SIMD本体。仅Graph owner产品改变。
旧FFN专属测试随退役迁移到真实norm region、旧FFN资格拒绝与普通完整FFN：两个独立槽/变化输入、
−80/−90/−100大值/Inf、最终输出、真实wait，norm并发/close/冷compile与invocation OOM恢复。
Windows初编译失败为测试Status/ResultError三元式歧义；fixture commit
`1b50371c94f2a8f5560b5f7efb5c59c6844ccdee`只改两处测试表达式，产品/API/成本不变，原日志保留。
首批Windows provider 3/3和selected Graph已通过，Linux与三交付未运行；状态仍AUDITED，不签CLOSED。
统一Compute+Graph实现LOC **61,318/249文件**（产品80c9ab00，不含tests/vendor），比TC减少692实现行。
