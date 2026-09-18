# C1-AA-Q1：归一化同实现对照资格恢复

## SCOPED

- owner：`/root`；模块专属 `bench/graph_provider_onednn/normalization_preflight.py` 的验证任务。
  不新增产品 factor，不重开 SIMD、Sched、Platform 或公共性能统计核心。
- baseline：`64e1e169`，F1 产品为 `ae270af9`；Windows 是唯一产品真相。
- 当前授权来源：[oneDNN 活动卡](llm-moe-c1-onednn-normalization.md)的 AA04 资格阻塞。
- trigger：F1 的采样前 workload A/A 通过，n=15 的120个产品进程及30个公共控制进程完成，
  五项公共控制均通过；采样后 workload A/A 在256个比值中有一个1.21435，资格 FAIL。
- 证据：[完整失败与限定诊断](../../../../evidence/performance/onednn-normalization-f1-64e1e169-w1-win/README.md)。
  失败项是 oneDNN 对自身的 Direct LayerNorm 1×2048、primary、执行与分配均 peer-first；
  两个角色虽然保留候选/peer 标签，但此 A/A 模式实际都调用 oneDNN，不能误称 Native 输了。
  该进程内部配对区间也排除1，不能把它直接解释为一个 timer tick。

## 已固定事实

1. source、artifact、host 身份未漂移；Release `/O2`、同一 P 核和自然完成边界保持。
2. 所有 A/A 聚合及 variant/order/allocation 中位数仍在原5%带内；失败来自单个进程比值的
   10% gross guard。不能据此擅自删点、放宽门或把整轮改写成 PASS。
3. 唯一追加的有限诊断已经结束：同一 primary binary、oneDNN/oneDNN 的四种顺序/分配组合，
   受影响项依次为1.00731、0.99646、0.95003、1.01484。没有复现21%偏差；这不证明具体 OS、
   JIT 或代码/数据地址原因，也不替代原失败。无未使用的原样完整重跑额度。
4. F1 只是冷融合准入修复，没有 SIMD、Sched 或热状态改动。该资格异常不能授权修改那些产品。

## 有限调查与 FIX 准入

追加静态调查最多30分钟，只允许两条具名路径：

- 对照合同：核对本模块 gross guard、独立进程/固定因素中位数及公共工程门分别声明什么；
  解释单实例差异与进程中位数结论的关系，不能因已看到失败就修改统计目标。
- 实例差异：核对两个真实 oneDNN plan 的 JIT、资源/代码地址、预热和完成责任是否同义；
  若缺少失败实例的必要观测，就明确证据缺失，不通过反复运行等待再次失败。

先把具体反例、拟修的有限验证代码和验收写成 AUDITED ledger，再改模块验证代码。没有具体缺陷时
不得改公共仪器、性能阈值、产品 API、常态 metadata 或源算法。原始失败和全部观测始终保留。
本卡不是额外产品优化轮；oneDNN 主卡仍只剩原定第二轮性能纠正预算。

若确有可修的模块验证缺陷，验收包括：同义对照、故意固定偏差必须被拒绝、原异常必须可追溯、
制品/合同/机器身份与失败传播、全部过程有界；使用原统一性能仪器。任何恢复后的完整比较必须
先冻结新候选和合同，最多一次，不能追加原样重试。若方案要求改变公共仪器或放松原工程门，
先形成独立、可审查的具体方案，不能在这里顺手实施。

## 停止与恢复

当前仅 SCOPED，尚无验证代码 FIX，也没有性能 PASS。到调查预算后必须给出原因/证据缺口及
明确修复或 blocker；不能继续无界研究。主卡两个 owner 暂为 BLOCKED；Q1 形成受审资格恢复
方案后，主卡按 BLOCKED→SCOPED→AUDITED 重新绑定剩余 AA04，再冻结候选验证。
默认规则、Native 取舍和 C1 整包仍未完成；其他已授权且独立的 C1 工作不由此自动变成 BLOCKED。
