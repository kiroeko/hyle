# CPU ggml 布局、广播与宽标量接线

候选 `395418f62cfb1226a76c8a5e1d4d319fe5543364`；Windows产品真相，Linux单向验证副本。
[LB1–5当前卡](../../../docs/implementation/tasks/compute/cpu-ggml-layout-broadcast-admission.md)。

## 已实现与实际消费

- F32 MatMul：rank2..4 dense row-major或contiguous-K转置RHS；两leading轴双向NumPy广播。
  LHS需要扩展时用原repeat，RHS必要transpose/cont；K0非空输出用原零identity/repeat。
- 三floating dtype Softmax任意轴：原reshape/permute/cont后原Softmax再逆布局；scale、同类型
  dense additive广播、安全特殊行及恢复复用已验边界，无新指数或归约数学。
- SDPA：BSHD/BHSD、独立Dk/Dv、GQA、causal/noncausal。F16 causal保留原flash，其他原F32
  scores→安全Softmax→value完整程序；仅causal建mask，sequence轴随layout绑定。
- 全15种plain scalar的Copy/Concat借原byte/row-copy，包括strided输入及空Concat部件。
  I8/I16/I32/I64/F64只是位载体，unsigned/FP8不获得未提供的数学；宽Gather/numeric Cast明确拒绝。
- 普通CPU选择、typed Direct和Graph无需provider排序。每项Graph接原operation→Copy完整region，
  准备一次、变化输入、真实Completion和重放；K0与宽Concat亦已实跑。

## 适用验收

| 环境 | 测试 | C++交付 |
|---|---|---|
| windows | Provider 58/58；CPU 12通过/1跳过；Graph 15/15 | 同一普通 consumer 源码、installed static、installed shared通过 |
| linux | Provider 58/58；CPU 12通过/1跳过；Graph 15/15 | 同一普通 consumer 源码、installed static、installed shared通过 |

Provider独立double oracle覆盖row-major/转置rank2..4、双向leading广播，三dtypeaxis0/1/−2
mask/NaN/+Inf/all−Inf与输入恢复、两attention布局/独立value depth/GQA/两因果模式；位比较覆盖
15种carrier的宽整数/NaN位、负零、element-aligned stride和三个Concat轴。原OOM/slot/close/alias
测试同批通过；普通consumer仅链接hyle::compute/graph，Windows DLL按既有helper邻置。
CPU跳过的D5RealQwenProjectionUsesExactProfileAndReplays缺旧模型fixture，不计PASS。
selector另选的model-input工具13/13通过；GPU、完整模型和Native性能裁决未运行、不签资格。

先行失败均保留：测试Attention字段初始化顺序、旧MLA冷拒绝断言与普通K0整region失败。
前两项迁移测试，数值门不变；Graph原空输入总拒绝是实际接线缺陷，经owner审计后一次定向
产品纠正。最终候选所有适用命令通过，不抹掉失败日志。

## 成本、终审与剩余

P0包含RHS必要repack、LHS全展开与checked字节累加；K0不用输入repack。Softmax非末轴增加
两次原布局materialization，安全score缓冲及线性特殊行扫描、scale/add节点成本保持显式。
SDPA按实际原flash/score程序与必要cast/layout计成本；新math为零。原arena/payload/pool/worker
owner不变，冷规划、热复用，无热搜索/计时赢家或新的系统控制。七门及红队在LB有限范围通过。

rank>4 MatMul、Bool/log-softmax、attention mask、宽index/duplicate Scatter、宽Fill、特殊alias
或bounded预算继续具名待资格化，拒绝不等于Native唯一。activation/gate、Min/Max/Select/reduce、
RoPE和I64 index的成熟接线仍待后续批次；Native优化与性能淘汰冻结，SIMD基础保留。
Compute + Graph统一实现LOC **61,214 / 248文件**，不含测试与vendor。

[收据](receipt.json)与[工件](qualification.zip)保存命令/XML/先行失败、同一consumer、产品输入
hash、selected/skipped与驱动/LOC，不复制SDK或build大目录。关闭仅文档状态，不再重建或P1。
重开只由本批数值/布局/资源/交付具体反例；CPU全收口与goal仍未完成。

本批临时安装consumer已清理，共619,893,081字节；producer、SDK与依赖缓存保留以复用。
[Windows清理](cleanup-windows.json)、[Linux清理](cleanup-linux.json)记录精确路径；产品源未删除。
