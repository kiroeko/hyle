# CPU 自研 MatMul 源级退役：MM / K1 完成

用户明确批准删除并撤回具名严格能力。产品删除 `b4c8899a`，实际成熟消费候选
`7157655201df3b63619a4e5b11dfe272bcf35273`；后续只清理旧bench、补GCC测试fixture和文档。
Windows是产品真相，Linux是1336个输入逐SHA前向匹配的验证副本，没有反向覆盖。

## 已删除与保留

- 删除 Native row-major、transposed-weight、batch/broadcast、K0 MatMul执行/准备/helpers，
  两份历史私有kernel文件、七种fat生成路线；不是降低优先级或留下普通备用实现。
- Native propose/prepare对该数学域明确Unimplemented，不形成制品、不偷偷调用managed库。
  原Native-only、0额外private invocation、具名高rank prepared-only矩阵能力同时撤回。
- 普通CPU MatMul由原oneDNN/ggml/ATen接替；schema、SIMD通用dot/FMA/codec、原Torch/ds4数学
  及已验RMSNorm/Copy whole-domain能力保留。104个SIMD/成熟/借用源码输入hash与RN相同。
- 三组只比较旧Native矩阵/MLP的benchmark targets和源移除；原合同/证据通过Git追溯。
  固定775b2bed SDK的本轮ND3资产、RMSNorm和CUDA pipeline比较仍保留。
- CPU executor样例迁ordinary mature：保留独立double oracle、并发、nested completion、资源平衡；
  不再强制已退场Native矩阵/FFN region或prepared-only。

## 实际验证

| 验证 | Windows | Linux |
|---|---|---|
| Native factor，含原ATen与迁移矩阵fixture | 144/144，无skip/fail | 144/144，无skip/fail |
| Core runtime/路由，未变产品代码 | 63/63，含测试fixture纠正后实跑 | 63/63 |
| 普通CPU受影响入口 | 13/13，无skip/fail | 13/13，无skip/fail |
| 受影响Graph入口，未变D5不全量重跑 | 19/19，无skip/fail | 19/19，无skip/fail |
| Native公开头 | 1/1 | 1/1 |
| 普通Direct/MatMul→Copy Graph consumer | source shared/static、installed shared/static全通过 | 同四种旅程全通过 |
| CPU executor实际样例 | 单/双worker、两个并发invocation、三次changed replay、nested、close/owner平衡通过 | 同项通过 |

同一普通consumer覆盖row-major、物理转置RHS、batch broadcast、rank5、K0与empty；每个shape
两次变化输入，独立分数值参考，真实Completion wait，关闭后拒绝再次提交。记录实际成熟identity；
Native-only Graph及严格unknown-reject invocation预算都明确拒绝，高rank prepared-only也拒绝。
拒绝不是执行PASS，不把mature未知上界改为0。

## 有限失败纠正与停止

Linux第一次Core测试未运行是驱动漏build target，原NOT RUN保留。补target后GCC发现两处已有
test-only告警：显式空selector字段、range-loop改const reference；没有修改公共路由/算法。
Windows追加Core检查曾在tests-off树中请求不存在target，原失败保留；恢复测试配置后63/63通过。
这些日志不改写成PASS。没有重跑已结束P1、优化自研或重写公共仪器。

selector由于两份历史private文件的目录归属及Native CMake seam选出大闭包；实际依赖扫描仅Native
include/生成代码使用它们。Compute runtime/公共头/组件DAG/成熟provider/GPU源码与其编译include
闭包未变，适用工程风险为上述CPU与真实直接consumer。GPU/其它D5不执行、不假报本轮PASS；
其未变资格复用原关闭卡。源码/静态/动态交付与改动的Native/普通入口均实际运行。

红队核对源码引用0、无备用矩阵、借用数学与SIMD保持、冷拒绝taxonomy、未知预算不放宽、
成熟完整region未拆散、无假skip/PASS。MM/K1 scope、architecture、correctness、delivery、
usability/docs、red-team、P0成本门通过；不是剩余全部CPU去留/default优选或整个goal完成。

## 体积、归档与清理

统一脚本29个Compute+Graph实现路径，不含tests/vendor/bench：**60,695行 / 247文件**，
相对61,386/249减少**691行、2文件**；统计commit `ada73615fc76cccb36b5f9418392a2a441d9f731`。
四个本批consumer目录安全清理**624,947,376 logical bytes**（不是保证的物理释放量）。
新SDK、producer、依赖cache和固定P1的775b2bed SDK均保留；旧P1 52个安装输入另核对未变。

[receipt.json](receipt.json)与[qualification.zip](qualification.zip)保留全部成功/失败命令、XML、
source identities、driver、普通consumer、selector plan、LOC和cleanup；zip SHA256为
`e0119b5e93181f96c336531b311a6c2e404d031a9069b1051cd1fb0da59946fe`。
新真实数值/生命周期/交付缺陷、相关依赖升级或新的确切Native优势/独有consumer才重开本退役决定。
