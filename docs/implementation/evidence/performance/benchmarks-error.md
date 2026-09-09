# error 性能证据


> **error-v5 current engineering evidence：`PASS`。** candidate `d61f0710` / tree `305b2cb5` 在
> W1-WIN 以 `error-source-consumer-v2` 完成 fresh n=15 qualification；资格门、工程性能门、全部
> artifact/build/source/codegen validators、A/A、5%/10% known-delta、timer-floor 与 placement controls
> 均 `PASS`。五个同合同 consumer workload 的所有四种 order/placement strata 均通过 5% non-inferiority；
> success 为 `PARITY`，三个 owning failure handoff 与新增 registry borrow 为 `WIN`。这是冻结消费者范围的
> P1 工程结论，不外推为 Error 所有操作、所有平台或 Formal SOTA。
> 产品状态只看
> [`module status`](../../status.md)。

本页是 [performance 总账](README.md) 的 error 模块详节。稳定设计见
[error-design](../../../architecture/error-design.md)，用户选型见
[error guide](../../../user/error.md)。只有绑定 consumer provenance/materiality 的 benchmark 才能裁决产品；
benchmark 本身不定义 API；v4 合同通过受检 source provenance、ownership topology 和 material
contribution 取得产品门资格，旧 v3 合同没有。

## error-v5 current qualification

v5 不建立 microbenchmark 大全。合同 `error-source-consumer-v2` 只运行两部分：

1. 在新 product/Abseil identity 上重放 v4 的四个 source-primary owning Status/Result workload，确认
   新增 public surface 没有改变既有正常路径；
2. 增加一个 fixed-table registry lookup consumer：命中返回 guaranteed non-null borrow，缺失返回
   code-only error；HYLE `Result<T&>` 与 Abseil `StatusOr<T&>` 共享 lookup、命中分布、referent lifetime、
   charged phases 和 observable，按 `SAME_CONTRACT` / 5% non-inferiority 裁决。

fixed-table 分布由 `borrow_consumer_shapes_v1.json` 明确标为 admitted north-star journey，而不是捕获的
生产频率；benchmark 不能反向定义 API。v2 peer 固定为 Abseil
`5e12b490a2009cb223dd75a70fd0213847b82f9c`，contract SHA-256 当前为
`61993b4c19750f6d19b9906c06216f4453de10335e8788e13397a23683a526e0`。首次 n=1 probe 只证明 semantic、
call topology、source hash 与 placement validators 工作，按协议保持 `DIAGNOSTIC / INCONCLUSIVE`；最终
资格轮从 clean candidate 和 clean peer checkout 运行，没有复用 probe build/output，也没有触发 n=31。

| consumer workload | combined exact verdict | 四 strata 中位 advantage ratio | 最低 99.26% CI lower | 工程解释 |
|---|---|---:|---:|---|
| TaskGraph 64-node success | `PARITY` | 0.9923–0.9986 | 0.9673 | 5% NI `PASS`；亚百分比摆动不改产品 |
| TaskGraph fresh owning last failure | `WIN` | 1.2118–2.5097 | 1.1814 | 5% NI `PASS` |
| parallel first-error owning handoff | `WIN` | 1.4103–3.9143 | 1.3534 | 5% NI `PASS` |
| Join failed owning handoff | `WIN` | 1.5182–4.7840 | 1.4904 | 5% NI `PASS` |
| fixed-table `Result<T&>` borrow | `WIN` | 1.7282–1.7515 | 1.6850 | 5% NI `PASS` |

ratio 大于 1 有利于 HYLE，5% NI 下限为 `1/1.05 = 0.95238`。部分 Abseil owning-failure strata 存在很大的
execution-order 摆动，所以本轮只采用四 strata 全通过和最保守 CI 下限，不把最大倍数宣传成稳定产品
加速比；placement、A/A 和 known-delta 全通过也不授权这种外推。

2026-09-05 的跨冻结固定方向复核专门检查了 success 四个都低于 1 的点值：精确 ratio 为
`0.9986028 / 0.9923150 / 0.9985339 / 0.9929537`，名义落后 `0.14%–0.77%`，但全部 CI 跨过 1；A/A 为
`0.9999113 / 1.0009608`。retained MSVC object 中 HYLE/Abseil 的 noinline success node 与 64-node
evaluate 热路逐指令相同，OK 都是同一 one-word 编码；外层成功清理只剩未执行的析构目标和代码布局
差异。上一版同一 source-consumer success 的四 strata 又跨过 1（`0.9984–1.0018`）。因此没有找到
复杂度、资源或热路 codegen 的产品机制钥匙，本行保持 `PARITY`，分类为“代码布局/运行分辨率内无
机制”，而不是把固定点值方向藏起来。完整横向记录见
[固定方向审计](../../tasks/performance/frozen-module-fixed-direction-audit-2026-09-05.md)。

retained engineering bundle 位于 `C:/hyle-scratch/error-v5/qualify-v2-output`；manifest、summary 与
contract-lock SHA-256 分别为
`12685d8f398d4f55c9f05c71d54a3f5054ee3b99f21147778dfe91644f8b7eff`、
`3303032a19b5cf036733ac2627626631f8460147b26432453cb60e5a6421cd45` 与
`04b41774b914124c3189083dee962a46819e8312256bcd5ad51cb2fb5ca10612`。独立 verifier 返回
`qualification=QUALIFIED`、`paths_verified=true`、`provenance_compatibility=EXACT` 和
`current_certification_policy_eligible=true`。该本机 bundle 是可深验工程证据，不是发布 artifact 或
Formal certificate。

WinSock classifier、bounded formatting 与 `std::error_code` adapter 是错误冷边界，使用 O(1)/bounded
结构、allocation count、异常遏制和 correctness 证明，不为罕见失败率编造 P1 计时。旧 fixed-backing
shared-message copy 仍只作 mechanism diagnostic；只有正常 owning handoff 显示 material loss 才能触发
一次机制检查。

## 被替代证据

v1--v4 的微基准、旧 source-consumer 表和证据纠正过程已由 v5 qualification 取代，不在当前页继续累积。原始事实与当时 verifier 可从 Git 恢复；不得与当前 candidate 合并或把历史点值当作重开依据。
