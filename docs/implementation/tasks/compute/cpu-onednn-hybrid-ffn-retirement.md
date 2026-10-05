# CPU混合FFN：撤回错误自研gate资格

<!-- MODULE REVIEW: SCOPED -->

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
