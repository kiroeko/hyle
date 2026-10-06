# CPU Native 一次批量 P1：资格通过，14 项劣势需要处理

产品数学 `775b2bedace019de423edc5d8a6477772b8ef3ec`；benchmark 固定
`481fc3ea2bd2bc9ab50e6b4185c7daa64111ddb3`，实际采样 source revision
`e8fec259ec6e687acce81affd09deec39c0e302b` 只增加状态说明。安装 SDK 与同源码 peer 都冻结。
W1-WIN / engineering / MSVC 19.50 / static / 单P核CPU2 / above-normal / 每个成熟程序1线程。
这不是跨机器 Formal 或全模块 SOTA。

## 实际结果

此前 [278项前置](../cpu-native-domain-preflight/README.md) 的整体 FAIL 保留；其中206项在正式
A/B 前固定为有资格子批，72项失格不参赛。一次统一 runner 用15个独立样本×4执行/分配 strata，
另30个 controls 进程；1422.625秒。统一资格 **PASS / QUALIFIED**，controls、全部validators、
输入/host/artifacts稳定性通过。原始 manifest 已用统一 verify 复核通过；没有追加 n31 或优化。

| 口径 | 结果 |
|---|---|
| 相对1.0的严格方向，所有strata的既有CI同侧 | 190 WIN、14 LOSS、2 INCONCLUSIVE |
| 冻结5%实用门的原summary | 187 WIN、14 LOSE、5 INCONCLUSIVE |
| non-inferiority | 192 PASS、14 FAIL；聚合performance FAIL |

3个同dtype Cast的严格方向为小幅WIN，但其区间未全部证明5%完整优势，不能把它们改成PARITY，
也不因这几个点保留重复的普通入口。方向由统一仪器原CI直接相对1.0列示，没有另写估计器。
187个场景证明的是相对**当前已资格化默认成熟程序**的完整操作收益，包含实际热转换、packing、
原库分配、检查、物化和真实wait；不是相对所有候选库的全局冠军，更不是整个模型加速比。

## 明确劣势，算法优化预算0

| 场景 | Native吞吐/成熟吞吐，4 strata范围 | 成本结构 / 动作 |
|---|---|---|
| F32 weighted RMSNorm，rows32/hidden2048 | 0.592–0.607 | 大批次成熟oneDNN更优；小rows和其它dtype有独立WIN，不能整族删除 |
| F32/I32、F16/I64、BF16/I32 Assign，32条2048元素更新 | 0.126–0.226 | Native逐元素索引/offset/写入；成熟原index-copy整段处理。退役对应普通域，保留已赢小更新/确认独有域所需共享代码 |
| 三dtype log-softmax，rows32/hidden2048，none/Bool | 0.242–0.261 | Native标量逐元素exp/log；成熟原Tensor程序摊薄成本。6项退役普通域，不优化自研 |
| F32 equal-head SDPA，query rows32 | 0.098–0.099 | Native简单dot/softmax/value loops落后原ggml完整attention约10倍；小query有独立结果，不能据此拆成熟region |
| F16/BF16 Softplus，rows32/hidden2048 | 0.736–0.789 | 标量稳定函数/低精度codec成本；普通大批次选择成熟，已赢小批次与资源域分开 |
| F32物理转置RHS MatMul，M32/K=N1024 | 0.522–0.535 | Native逐row/column dot，成熟ggml更优约1.9倍；两个M1布局无可靠方向，row-major M32前置失格，整个本批MatMul没有完整优势证据 |

不能把这些14个**场景**写成14个互不共享的算子代码文件。尤其MatMul：普通CPU已有合格成熟
替代；但Native还提供当前已声明的caller workspace/0额外private与更高rank组合。是否撤回无真实
消费者要求的这些严格资源域，必须在实际源级清退FIX前明确，不能借未知library bounds假签数学
UNIQUE，也不能悄悄把未知上界改成0。RMSNorm/Copy已验whole-domain能力仍保留。

## 证据与当前边界

[receipt.json](receipt.json)列206项方向、CI和ratio；[qualified.zip](qualified.zip)保留统一原始
manifest、observations、controls、validators、raw logs、source/host/toolchain/SDK/artifact身份、
launcher与统一LOC。原校验路径仍在固定Windows SDK、peer和fresh consumer中，未删除这些输入。

此批是一次有效比较，**尚未完成新增源级删除或默认规则接线**。后续须固定有限FIX并依法绑定
相关owner后实施；单算子WIN规则不能压过未比较的成熟完整region。SIMD基础不删除，GPU暂停。
Compute+Graph实现仍为 **61,386行 / 249文件**，29个统一模块路径、不含tests/vendor/本批工具。
