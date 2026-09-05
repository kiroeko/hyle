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

## error-v4 source-consumer qualification（历史基线）

合同 `error-source-consumer-v1` 的 SHA-256 为
`dde5b9a2e8c85d11014d2bb8312a948da422af1b1d339f09ef1046f9e1ec6f97`，benchmark source SHA-256 为
`ebb8c919a1d5236138cbd865d442e87d8f7f9167b02946b795df463259c96466`。HYLE/Abseil 使用同一模板
consumer algorithm，并锁定 source location、值类别、ownership/lifetime、最终 observable、双执行顺序
和 primary/placement。peer 是 Abseil commit `81c993af4fb88b3a51829a796df02251fa27102f`。

W1-WIN fresh MSVC x64 Release/static、portable ISA、LTO off 的 n=15 qualification 未触发 n=31：
artifact/build/source-consumer validators、A/A、known delta、timer floor 和 placement controls 全部
`PASS`；4 workload × 4 strata 全部 comparable 且 5% non-inferiority `PASS`。

| source consumer workload | combined exact verdict | 四 strata advantage ratio | 最低 99.26% CI lower | 产品解释 |
|---|---|---:|---:|---|
| `TaskGraph::evaluate` 64-node success | `PARITY` | 0.9984–1.0018 | 0.9945 | 成功热路同档，NI PASS |
| 64-node fresh owning last failure | `WIN` | 1.1950–1.2160 | 1.1627 | HYLE 约 1.20x，NI PASS |
| parallel evaluator first-error owning handoff | `WIN` | 1.4083–1.4181 | 1.3933 | HYLE 约 1.41x，NI PASS |
| `JoinState` failed owning handoff | `WIN` | 1.5030–1.5221 | 1.4550 | HYLE 约 1.50x，NI PASS |

ratio 大于 1 有利于 HYLE；5% NI 下限是 `1/1.05 = 0.95238`。成功路径的结论是统计等价，不把亚百分比
地址/顺序摆动解释成产品缺陷；三个失败路径的优势来自真实 owning transfer 形状，不外推到未测能力。

retained bundle 位于
`C:/hyle-scratch/error-source-consumer-v1-d9644548-win-rerun02-evidence`；manifest/summary/
observations SHA-256 分别为
`0afe6d46813cab7c6b949b1f4478544c99b7f040054d579f6839bc19307b7c13` /
`97873ce6a55d96792a3a269984488f95afa4233f5b2d1976a6259bd9b566d39f` /
`6cc1921740edf57feecda19f4c0e15c96533cfa724ad6c133a1b63414611c408`。v4 资格轮及当时的独立重放返回
`qualification=QUALIFIED`、`paths_verified=true`、`current_certification_policy_eligible=true`。
这是绑定当时 receipt 的本机工程证据，不是发布 artifact 或 Formal certificate。

> **2026-08-29 回放兼容性：**后续提交 `2995286c` 为 Tracy 加入了全局
> `TRACY_ONLY_LOCALHOST` sanitizer 项，曾因原地扩展 v1 使 verifier 停止。本合同明确
> `HYLE_TRACE=OFF`，且 strata/controlled environment 未使用 Tracy 控制。环境策略版本化修复后，当前
> CLI 对原 manifest 返回 `environment_policy_compatibility=LEGACY_COMPATIBLE_TRACE_DISABLED`、
> `paths_verified=true`、`qualification=QUALIFIED`；旧 manifest 与 hash 均未修改。通用规则见
> [统一性能仪器](../../../developer/performance/instrument-v1.md)。

## error-v3 证据纠正

原 v3 合同当时把工程裁决机固定为 `W1-WIN`，仪器为 `closure-perf-v1`，并声明 required comparable
workload 对固定
Abseil commit `81c993af4fb88b3a51829a796df02251fa27102f`，使用 5% material/non-inferiority
margin。Folly commit `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于 Expected/API/实现查漏；
C++23 `std::expected` 是标准合同和较弱成本下界，不冒充第三个同合同性能 peer。

| workload family | 原合同计费边界 | post-terminal verdict |
|---|---|---|
| graph replay | 64 个 stable node 的一次 `TaskGraph::evaluate`；全成功与末节点 code-only 失败 | raw samples/statistics retained；原合同 computed `PARITY`/`INCONCLUSIVE` 不足以签首席 source consumer |
| fixed-backing copy micro | 同一个预建 29B owning message backing 的 copy+destroy | raw samples/statistics retained；原合同 computed `LOSE` 只作机制诊断，comparison/product verdict rejected |
| synthetic diagnostic | message construct、context 1/3、两 frame Result、format、Update | raw samples/statistics retained；`result-message-error` 的原合同 comparison/product verdict 因 ownership 不对称 rejected |
| boundary diagnostics | 7/256B construct、16 层 context、512B format、runtime-code primitive、native/errno | `NOT_P1_GATING`；只记录 correctness、复杂度或绝对成本 |
| native detail / payload | HYLE 固定 `{domain,int64}`，Abseil 可携带任意多 payload | `NOT_COMPARABLE`；不得写成同能力胜负 |

原 v3 资格流程至多一次 probe、一次 fresh n=15 `qualify-platform`；只有统一仪器停止规则允许时才自动进入
n=31。candidate/peer/source/build/artifact identity、真实 batch timer-floor、双执行顺序、placement、
A/A 与 semantic setup oracle 必须同时完整；任一 required 门缺失即保持 `INCONCLUSIVE`/`BLOCKED`，
不能换 benchmark 姿势或放宽 margin。shared 交付只做绝对回归与 package/correctness 验证，不与
静态 peer 结果混成算法胜负。

exact candidate 已完成 fresh clean n=15。全部 A/A、delta、timer-floor、placement、semantic、
artifact/build identity 与 error-codegen controls 通过，统一 verifier 返回 `QUALIFIED`；这证明 raw
样本忠实执行了原合同，不证明原合同具备 consumer/materiality 资格。下表必须按“原合同的统计观察”
阅读：

| 原合同 workload | HYLE / Abseil 中位 ns | original-contract computed verdict | 当前产品解释 |
|---|---:|---|---|
| `taskgraph-active-success-64` | 79.119 / 76.388 | `INCONCLUSIVE` | 4 strata 中 2 PASS、2 UNRESOLVED；需要 source-consumer 贡献合同 |
| `taskgraph-active-last-error-64` | 78.644 / 77.716 | `PARITY` | 可保留的 component observation，不能单独签首席消费者 |
| `status-message-29` | 23.821 / 48.054 | `WIN` | diagnostic observation |
| `status-message-copy` | 11.841 / 10.923 | `LOSE` | raw samples/statistics retained（4 strata ratio 0.920057/0.925041/0.920034/0.920214）；mechanism diagnostic；comparison/product verdict rejected |
| `status-context-1` | 41.891 / 265.869 | `WIN` | diagnostic observation |
| `status-context-3` | 74.750 / 879.920 | `WIN` | diagnostic observation |
| `result-message-error` | 10.892 / 21.971 | `WIN` | raw samples/statistics retained；comparison/product verdict rejected：HYLE move，peer lvalue copy |
| `status-format-54` | 34.599 / 43.811 | `WIN` | diagnostic observation |
| `status-update-prvalue` | 48.148 / 54.826 | `WIN` | diagnostic observation |

最小 timer-floor multiple 为 53.96x，所以 fixed fixture 的 copy 差距不是随机噪声。但该行反复复制同一
cache-hot backing，HYLE/peer allocation 顺序固定，合同没有记录 heap address/cache-line/page offset；
它只能提出 heap size-class/data-placement 或产品机制的归因问题。candidate 的 mature low-bit tag 和
真实 TaskGraph ownership transfer 仍是合理产品修复，不因证据分类错误而回滚。

固定方向复核还检查了 retained object，而不只看源码：HYLE 与 Abseil 的 copy lambda 在 retain、sink、
heap-tag branch 上逐指令相同；唯一不同是析构调用目标。两侧 release 都是 acquire load 后在非末次引用上
做 acq_rel `lock xadd`，而 HYLE 非末次路径甚至没有 Abseil `Unref` 的 stack prologue/epilogue。旧结果
因而缺少能解释 7.5%–8.0% 落后的实现机制；A/A 又只有约 0.1%–0.2%，所以也不能把它简单称作随机噪声。
结合未分层的固定 backing/address 与后续三平台自然 owning-copy 已反向得到 HYLE
`1.038x / 1.045x / 1.314x`，该行保留为“未归因 data/code-placement diagnostic”，不形成当前产品
`LOSE` 或重开。只有地址分层的同合同诊断或真实 copy-heavy consumer 同时给出机制和 material 影响时
才恢复调查。

更严重的是 `result-message-error`：HYLE 从临时 `Result` move error，peer 却从命名 lvalue 调用
`result.status()`，即使固定 Abseil 已有 `status() &&`。absolute/raw samples retained；原合同 computed
约 2x `WIN` 及其 comparison/product verdict rejected。v3 的
统计仪器正确执行了错误合同；当前不得用任何一侧点值宣布产品胜负，也不得以 isolated copy 改产品。
恢复条件是新的 source-primary consumer 合同、自然且对称的 ownership、保真 workload/trace 分布和
material contribution budget；microbench 只作机制诊断。

## 历史证据（2026-07-29，三平台 n=15）

基础产品能力基于 `befaad0`，非法 Result 诊断冷热分离基于 `7d22eb8`，渐进 context
表示固定为 `8dace9d`；benchmark 消费形状仍是 `6d42add`，没有为该实现新增测量专用 API。
`f31e8d0` 新增通用 Win32 adapter，但没有改变既有 Status representation/热路；Windows 在该
commit 上复跑两套 n=15，同能力结论仍为 1 项 TIE、其余全部 WIN。Linux/M4 旧数字继续绑定
未改变的算法实现，新入口以复杂度与真实 platform 去重证明，不制造高错误率 workload。
Abseil 固定到本地参考库 commit `35b9f25b`。环境为 Windows MSVC 19.50、WSL GCC 14.2、
Apple M4 AppleClang 21，Release、`HYLE_TEST_INSTRUMENTATION=OFF`（发布形状），使用
`scripts/bench-stat.py -n 15` 的同进程配对判定。表中绝对值统一按
**Windows / Linux / M4** 排列，不能跨机器横比；倍数是配对样本的中位比值，可能与两列
独立中位数之比略有差异。

该轮历史证据严格分成两种消费者信息合同：

1. **不携带额外诊断**：只有成功值或 canonical code。主对手是 Abseil
   `Status/StatusOr` 的同能力路径；`std::error_code/expected` 是不能携带任意消息和 native
   detail 的较弱下界，只单独登记，不用于宣称同能力 SOTA。
2. **携带诊断**：相同 canonical code、owning message、语义 context 或同一
   machine-readable native value。对手是 Abseil `Status/StatusOr/StatusBuilder/payload`。

没有第三类“为了 benchmark 好测”的公开类型。benchmark 也不允许给产品对象增加常驻字段、
缓存或特殊入口。

## 合同一：不携带额外诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| OK 构造并检查 | 0.378 / 0.057 / 0.226 | 0.376 / 0.054 / 0.226 | 三平台 TIE |
| OK Status 拷贝并检查 | 0.507 / 0.280 / 0.236 | 0.547 / 0.276 / 0.257 | Win **WIN 1.085x**；Lin TIE；M4 **WIN 1.093x** |
| code-only Status 构造 | 0.377 / 0.060 / 0.224 | 1.511 / 0.916 / 0.958 | **WIN 3.94 / 15.58 / 4.28x** |
| `Result<int>` 经过四个真实调用帧，caller 解包并消费 | 2.565 / 2.478 / 3.058 | 3.129 / 2.915 / 3.415 | **WIN 1.215 / 1.191 / 1.113x** |
| code-only Result 错误经过四个真实调用帧 | 2.372 / 1.103 / 1.443 | 4.009 / 2.258 / 2.455 | **WIN 1.579 / 2.006 / 1.681x** |

四层传播不是在同一个函数里连续创建三个临时 Result，而是 leaf → layer1 → layer2 → caller
的四个真实 `noinline` 调用帧，三家边界完全对称，最后消费成功值。旧形状曾在 M4 同时保活
多个临时对象，制造约 8% 假 LOSE；旧 GCC 形状还只内联了 std/Abseil 的外层包装，制造约
18% 假 LOSE。修复的是测量合同，不是为跑分修改产品 API。

### 能力更弱的 std 下界

| 操作 | HYLE ns（Win/Lin/M4） | `std::error_code/expected` ns | 结论 |
|---|---:|---:|---|
| code-only error 构造 | 0.377 / 0.060 / 0.224 | 0.364 / 0.062 / 0.224 | Windows std 约 **1.04x** 快但亚纳秒 TIE；Linux/M4 TIE |
| 四帧 Result OK + consume | 2.565 / 2.478 / 3.058 | 2.209 / 2.624 / 3.328 | Windows std **1.16x** 快；Linux/M4 HYLE 约 **1.07/1.09x** 快 |
| 四帧 code-only Result error | 2.372 / 1.103 / 1.443 | 4.737 / 4.413 / 2.166 | HYLE **WIN 2.00 / 4.00 / 1.50x** |

Windows 的四帧 Result 成功下界是真实、保留的 LOSE；code-only 构造本轮落入亚纳秒 TIE。
`std::error_code` 是平凡析构的两个 word，能力比
owning `Status` 小。它适合只需 category/value 的局部代码，但不能代替 HYLE 跨模块需要的
canonical message/native/context carrier。故它是成本下界和选型提示，不是同能力对手。

## 合同二：携带消费者需要的诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| owning message Status 构造 | 27.300 / 9.876 / 17.412 | 45.982 / 20.017 / 31.054 | **WIN 1.683 / 2.037 / 1.877x** |
| owning message Status 拷贝 | 10.044 / 8.850 / 3.181 | 10.445 / 9.262 / 4.180 | **WIN 1.038 / 1.045 / 1.314x** |
| 前置一层语义 context | 44.016 / 21.686 / 17.401 | 258.235 / 91.417 / 120.739 | **WIN 5.857 / 4.203 / 6.440x** |
| 前置三层语义 context | 78.318 / 43.255 / 42.714 | 761.229 / 277.394 / 339.224 | **WIN 9.704 / 6.480 / 7.721x** |
| message Result 错误跨两个调用边界 | 9.517 / 9.221 / 3.814 | 21.193 / 21.248 / 7.202 | **WIN 2.209 / 2.292 / 1.860x** |
| owning 人读格式化 | 30.843 / 19.741 / 23.787 | 39.864 / 22.874 / 30.729 | **WIN 1.284 / 1.165 / 1.232x** |
| 首错聚合 `Update` | 9.156 / 8.794 / 3.181 | 10.417 / 9.188 / 4.184 | **WIN 1.138 / 1.045 / 1.315x** |
| errno → canonical + operation + native value | 40.062 / 51.330 / 26.238 | 167.139 / 74.727 / 116.480 | **WIN 4.194 / 1.451 / 4.608x** |
| message + Vulkan native value 构造 | 34.507 / 12.122 / 16.477 | 86.585 / 37.709 / 53.482 | **WIN 2.513 / 3.110 / 3.373x** |
| Vulkan native value 查询 | 2.266 / 0.642 / 0.962 | 5.681 / 4.636 / 4.422 | **WIN 2.519 / 7.151 / 4.616x** |

Abseil payload 是可携带多项任意数据的通用系统；HYLE native detail 只表达一个固定
`{domain, int64 code}`。上表的 native 行只说明**同样存取一个原生数值**时 HYLE 更快，
不声称 HYLE 具备或击败任意 payload 系统。这个窄形状来自 Vulkan/CUDA/OS 恢复逻辑的真实
需求：此前只有 canonical code + message，消费者不得不解析 message。

同能力 Abseil 项在三平台没有确认 LOSE。这个结论不外推到未上架的 payload、序列化、
backtrace 或异常系统；参考库有而 HYLE 用户没有需求的能力，不会只为扩大 benchmark 表而
进入 error。

## 空间与复杂度

- 64 位平台：`Status=8B`，`Result<int>/Result<void>=16B`；`absl::Status=8B`，
  `absl::StatusOr<int>` 在 Windows 为 24B、Linux/M4 为 16B。
- OK 与 code-only Status 是 O(1)、零分配；native query 是 O(1)。
- message 构造是 O(message bytes)，heap Status copy 是 O(1) 原子 retain。
- 第一层 `WithContext` 精确生成扁平消息；第二层 rvalue 才进入 context 专属几何 prefix
  storage，唯一所有权且容量足够时只写新前缀。连续 rvalue 链的总复制为 O(最终消息字节)，
  扩容 O(log final bytes)；lvalue/共享值保持 COW，单次仍是 O(context + old message)。普通
  message/native Status 不携带 capacity/offset。
- errno canonical 分类是 O(1)；native description 的取得由 OS/CRT 主导，最终只构造一个
  owning Status。
- Win32 canonical 分类是跨 host 的固定 O(1) switch；Windows 仅在错误路径调用
  `FormatMessage`，最终同样只构造一个 owning Status。这里不增加“高错误率”微基准；其真实
  验收是 platform 删除重复 translator 后保持相同复杂度、统一分类与诊断合同。
- WinSock canonical 分类同样是固定 O(1) switch；raw integer 直接保留为 native detail，错误文本只在
  factory 冷路径获取或使用 numeric fallback。
- `Result<T&>` 使用 pointer 与普通 `ResultError` discriminator，成功/错误观察均 O(1)，不分配、无锁、
  无 registry；referent lifetime 由 caller 所有，const carrier 只暴露 deep-const borrow。
- `format_status` 的格式化工作集固定为 1 KiB caller stack，输出 O(min(formatted bytes, 1023))，随后
  Status 至多一次 exact-size backing allocation；截断、formatter exception 与 OOM 均不抛出。
- `std::error_code` 的 generic/system 分类在 caller image O(1) 完成；custom category 的 `message()` 成本由
  category 实现主导，但 pointer 只在调用期间借用，不跨 DSO 保存。
- 非法 Result error arm 的 fail-fast 诊断放在冷 `noinline` helper；正常传播帧不复制 abort
  文本和控制流。M4 栈帧从 112B 降到与 Abseil 相同的 80B，这是冷热路径架构修正，不增加
  产品状态。

## error-v3 运行记录

terminal candidate 绑定 HYLE `9595724a0a284cbc3502f1c9e9e3ec9c9cf9e2b9` / tree
`a828fb4cdf1cd7f72aaea098b44fd42e0d966d2d`、Abseil
`81c993af4fb88b3a51829a796df02251fa27102f` / tree
`b8e8bdc41d9f13b8894b4540b910ed8dc2ced66c`，contract hash 为
`82cf325480a85d7ea76e17fcf526d12c36619b7089ec81d4c947fa56aea215d7`。W1-WIN retained evidence 位于
`C:/hyle-scratch/error-v3-final-win-perf-evidence`；manifest/summary SHA-256 分别为
`4cd3627d134be6d44835c7ac976554a3f65d963532af155fd28b21044a728b85` /
`68dc98addc69a00007a1be0f88a605a495d3c4fa120a666646fa1c5b817740c6`；统一 verifier 为
`QUALIFIED`、`paths_verified=true`，manifest 的 original-contract computed gate 为 `FAIL`。post-terminal
consumer/ownership 审计保留 raw samples/statistics，拒绝 derived comparison/product verdict，并把当前
engineering state 更正为 `INCONCLUSIVE`；
原 manifest 和 hash 不倒写。该本机路径不是发布 artifact；task receipt 记录三平台 correctness/
delivery、二进制 hash、审计 finding 与恢复条件。

被驳回 replacement 历史绑定 HYLE `a4ba5a14cdd46a7b9c6e33c06211ce2d6c7a760f` / tree
`4a2ca7977241080a41b5eb5ab79a6fd37d71db04`、Abseil
`81c993af4fb88b3a51829a796df02251fa27102f` / tree
`b8e8bdc41d9f13b8894b4540b910ed8dc2ced66c`，contract hash 为
`82cf325480a85d7ea76e17fcf526d12c36619b7089ec81d4c947fa56aea215d7`。W1-WIN retained evidence 位于
`C:/hyle-scratch/error-v3-replacement-win-perf-evidence`；统一 verifier 的 original-contract computed
fields 为 `qualification_gate=PASS`、`performance_gate=FAIL`、`QUALIFIED`。这个路径是本机可追溯证据，
不是当前产品 qualification 或可发布 artifact；
task manifest 另记二进制 hash 和跨平台结果。

原 v3 round 当时只接受由冻结 `bench/contracts/error-p1-v1.json`、`closure-perf-v1` runner 与 candidate
manifest 共同生成的证据，并归档 machine/affinity/power、source/peer/build/binary hash、双 order/
placement、timer-floor、A/A 与 raw samples。该 contract 现为 immutable historical/raw diagnostic，只能
重放既有 bundle 或运行明确不具资格的 mechanism diagnostic，禁止产生新的 current qualification；未来
error qualification 必须使用新 contract ID，
并增加受检的 consumer provenance/materiality 与 ownership topology，不能再由 workload 自报
`tier=replay` 获得产品门。

修复 v3 证据问题的 replacement 是 `bench/contracts/error-source-consumer-v1.json`。它不复用上述历史结论，而是把
`TaskGraph::evaluate`、parallel evaluator first-error 和 `JoinState` failed handoff 的当前源码 hash、
ownership topology 与逐组件 material contribution 冻结进新契约；HYLE/Abseil 由同一模板算法实例化，
source location、值类别、生命周期、执行顺序和 placement strata 对称。它已经产生上文 v4 的 current
qualification；本段保留在 v3 历史区，只用于说明旧 blocker 如何被满足。
v5 的 current replacement 已另起 `error-source-consumer-v2`，不改写 v1 的历史合同或 bundle。
