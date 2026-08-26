# error 性能证据

> **error-v3：`CANDIDATE / PENDING REQUALIFICATION`。** replacement candidate 已通过证据资格门、
> 但未通过 P1 工程性能门；唯一机制修复形成 `9595724a` / `a828fb4c` 新 candidate，尚无新 verdict。
> 本页下方 2026-07 点值只保留为历史输入，不是当前模块证书；现行状态只看
> [`module status`](../../status.md)。

本页是 [performance 总账](README.md) 的 error 模块详节。稳定设计见
[error-design](../../../architecture/error-design.md)，用户选型见
[error guide](../../../user/error.md)。benchmark 只验证消费者会使用的产品形状，不定义 API。

## error-v3 待资格合同

工程裁决机固定为 `W1-WIN`，仪器为 `closure-perf-v1`。required comparable workload 对固定
Abseil commit `81c993af4fb88b3a51829a796df02251fa27102f`，使用 5% material/non-inferiority
margin。Folly commit `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于 Expected/API/实现查漏；
C++23 `std::expected` 是标准合同和较弱成本下界，不冒充第三个同合同性能 peer。

| workload family | 正常消费者任务与计费边界 | 当前资格状态 |
|---|---|---|
| graph replay | 64 个 stable node 的一次 `TaskGraph::evaluate`；全成功与末节点 code-only 失败，对称消费最终结果 | replacement：末错误非劣；全成功四 strata 约慢 6%，候选已驳回 |
| normal diagnostic | 29B owning message construct/shared copy、1/3 层 context、两 frame message Result、54B owning format、fresh-prvalue Update | replacement：除 shared copy 外 6 项均 WIN；copy 四 strata 约慢 8%，候选已驳回 |
| boundary diagnostics | 7/256B construct、16 层 context、512B format、runtime-code primitive、native/errno | `NOT_P1_GATING`；只记录 correctness、复杂度或绝对成本 |
| native detail / payload | HYLE 固定 `{domain,int64}`，Abseil 可携带任意多 payload | `NOT_COMPARABLE`；不得写成同能力胜负 |

资格流程至多一次 probe、一次 fresh n=15 `qualify-platform`；只有统一仪器停止规则允许时才自动进入
n=31。candidate/peer/source/build/artifact identity、真实 batch timer-floor、双执行顺序、placement、
A/A 与 semantic setup oracle 必须同时完整；任一 required 门缺失即保持 `INCONCLUSIVE`/`BLOCKED`，
不能换 benchmark 姿势或放宽 margin。shared 交付只做绝对回归与 package/correctness 验证，不与
静态 peer 结果混成算法胜负。

replacement 已完成一次 clean n=15，但其结论是有效的产品失败，不是当前性能证书。9 项中 message
construct、context 1/3、两 frame Result、format 和 Update 共 7 项 WIN，末节点 code-only error 通过
5% non-inferiority；全成功 graph replay 的候选优势中位比为 0.940--0.942，shared message copy 为
0.917--0.923。全部 A/A、delta、timer-floor、placement、semantic 和 error-codegen controls 通过。

唯一 codegen 诊断把两个失败分别归到真实产品形状：当前 range-based heap 判别令 shared-copy 循环比
固定 Abseil low-bit tag 多两个更宽的 compare；TaskGraph 把失败即返回的局部 `Status` 当 lvalue 交给
传播宏，保留了不需要的复制语义。临时编译采用 mature low-bit inline tag 后复制循环逐指令同形；局部
显式 move 后 MSVC 函数由 133B 收敛到与直接 `if/return` 相同的 104B。修复不新增公开 API、常驻
metadata 或 benchmark-only 分支，且真实 `TaskGraph` 与 replay 必须同步。新 candidate 仍使用完全相同
的 9 项合同、peer、5% margin 与统一仪器，只允许一次 fresh qualification。

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
- 非法 Result error arm 的 fail-fast 诊断放在冷 `noinline` helper；正常传播帧不复制 abort
  文本和控制流。M4 栈帧从 112B 降到与 Abseil 相同的 80B，这是冷热路径架构修正，不增加
  产品状态。

## error-v3 运行记录

被驳回 replacement 绑定 HYLE `a4ba5a14cdd46a7b9c6e33c06211ce2d6c7a760f` / tree
`4a2ca7977241080a41b5eb5ab79a6fd37d71db04`、Abseil
`81c993af4fb88b3a51829a796df02251fa27102f` / tree
`b8e8bdc41d9f13b8894b4540b910ed8dc2ced66c`，contract hash 为
`82cf325480a85d7ea76e17fcf526d12c36619b7089ec81d4c947fa56aea215d7`。W1-WIN retained evidence 位于
`C:/hyle-scratch/error-v3-replacement-win-perf-evidence`；统一 verifier 为 `QUALIFIED`、
`qualification_gate=PASS`、`performance_gate=FAIL`。这个路径是本机可追溯证据，不是可发布 artifact；
task manifest 另记二进制 hash 和跨平台结果。

新 candidate 的正式运行仍必须由冻结的 `bench/contracts/error-p1-v1.json`、`closure-perf-v1` runner
和 candidate verification manifest 共同生成，并把 machine/affinity/power、source/peer/build/binary
hash、双 order/placement、timer-floor、A/A 与 raw samples 一起归档。旧专用 runner 命令不是 v3
入口，不能从历史参数手工拼出当前证书。
