# G04/G05 Apple 编码页与有限数学域终审

状态：**CLOSED，2026-10-15**；owner `compute_route_apple`，不可变产品 candidate
`3b51575320c3f823ad1a31d18e99b8c55523a117`。授权、有限 FIX 与历史见
[原施工卡](inference-g04-g05-cache-attention-domain.md)。本记录只关闭 Apple 已准入域，
不关闭整个 G04/G05。[聚合证据](../../../../evidence/correctness/inference-g04-g05-apple-ce14-acceptance/receipt.json)
绑定普通工程门、独立 peer 数值、合格 P1 和统一 LOC；候选后没有产品修改。

## 七门与公共旅程

| Gate | Verdict | 实际证据与边界 |
|---|---|---|
| scope | PASS | 原 CE01–07 及 CE14 有限域；F16/BF16 query/output、Q8_0 K/V、page16、有限 D64/128/256、独立 Dv、causal window/full noncausal；单/多请求和 native/required CPU+Apple |
| architecture | PASS | 原 MLX 数学、Program/Invocation、F1/F2/G03 页权限与实际完成复用；私有 Query 类型进入 primitive state/equivalence/JIT key，无公共 schema/权限/退休变化 |
| correctness | PASS | [当前 source 与风险](../../../../evidence/correctness/inference-g04-g05-apple-ce14-ordinary/receipt.json)：13 数学、4 public consumer、52 cold allocation 失败点与 6 native 生命周期模式；原独立 byte/double oracle，subnormal Q×large K、小概率×large V、65532 消去与长 prefill；3 host sanitizer 与单独 shader validation |
| delivery | PASS | 同证据中 source、relocated static/shared；两 SDK 各 native13/public4、CPU14/public3、12 独立公共头及 installed Metal lifetime；shipping 无 HYLE_TESTING，原组件图/原依赖不扩张 |
| usability_docs | PASS | [batch 指南](../../../user/batch-attention.md)给 CacheStorage/spec、dirty encoder→原 F2→native/Graph、旧 reader/旧结果、失败、CPU 前后数学与成本；未宣称 capture 或全模型吞吐 |
| red_team | PASS | 候选后独立于实现的一遍逐 hunk 反查：34-byte 块边界、partial subgroup 实际 code 数、inactive 行不读、scale bit 类型、Query identity/JIT cache/default plain 分支、精度与完整生命周期成本；无未决本域 MUST |
| engineering_performance | PASS | [actual Air P1 attempt5](../../../../evidence/performance/inference-g04-g05-apple-encoded-m1-p1-attempt-5/receipt.json)：QUALIFIED/PASS、strict verify0、16 A/A、原 n15 的120产品/30controls；三组原5%非劣门 PASS，方向均 WIN |

公共消费者覆盖 paired/independent 原 Graph encoder→同一个 Update→native/required joint
最终结果，实际六次 dirty 更新、只改新行、旧 frame/shared page 数值和 readonly 字节、
accepted NaN 失败及所有 writer 拒绝发布。BF16 的原源阶段补充见
[BF16 记录](../../../../evidence/correctness/inference-g04-g05-apple-bf16-encoded-page-source/result.json)；
当前四项 public tests 已在 source/static/shared 与三 host sanitizer 中复验。
未变 G03 状态/来源/退休仍按原范围复用，不制造第二执行 owner 或权限体系。

第三方保持原 MLX/Metal 锁、license 与派生源码认证；private patch 属原 Apple owner。
对齐完整/部分 Q8 code group 只加载一次 scale，解码到 F32；原 F16 Query 保留为 F16
staging/fragment，D256 使用 BQ32/BK8/WM4。K/V、P、scores、O 与累加仍 F32，原屏障
全部保留；BF16/default plain 路径不窄化。未采用 packed_char4、范围 guard 或删屏障试改。
无新增 FFI、公开 vendor 类型、兼容转发、planner 或全历史 KV 解码。

## 工程性能与限制

实际 MacBook Air Mac16,12/M4 arm64、macOS27.0.1/AppleClang21，M1-MAC engineering，
performance source `4f2666b54a765330b9a35d88a0b5f06f5241aeab`，contract
`64bc53254b3a33f92e312afed568c867a5d9fb7fc0490c2f0880932e384b4992`。
三形状同为两请求、F16 query/output、Q8 KV；不同执行/分配顺序与 placement 均保留：

| workload | ratio 范围（大于1有利HYLE） | 方向 | 5%非劣 |
|---|---|---|---|
| window D256 | 1.2279–1.2381 | WIN | PASS |
| prefill D256 | 1.1443–1.1513 | WIN | PASS |
| noncausal D64/Dv128 | 1.0765–1.0833 | WIN | PASS |

原 general-Q8/pad/blk peer 明确选定、额外64/128实例披露，不是原 host best-of。
prepare/native 对象、F2、精确 guards、gather/mask/normalization、拷贝、完成与 output
release 全部计费；初始库/pipeline 编译在双方 untimed 数值/warmup 阶段。713.801秒远程
成本保留，bundle/deploy另列，外层观察重叠不重复相加。方向写入
[性能总账](../../evidence/performance/inference-encoded-g04-g05-2026-10.md)。

原 qualified prefill0.325 LOSS、否决的 NAX/tile/guard/sync 尝试、attempt3磁盘设置失败
和 attempt4文档hash资格失败全部保留。attempt4的有利样本仍为 DIAGNOSTIC/UNPROVEN，
没有手工提升旧 manifest；attempt5在采样前核对整条 provenance 链。

ASan detect_leaks=0；TSan两项原显式 refcount/callback 模型不签 driver/shader。
shader validation另有实际13+4测试与独立8pipeline enabled-state见证；该见证不冒充
读取了每条产品 pipeline 的 property。native private-pool retained bytes/hard upper仍
UNKNOWN；metadata准备/等待、native对象更替等成本保留，G06/G10独立处理。
本 P1 不签 BF16 吞吐、独立硬件 Formal、capture、producer/growth 性能或 token/s。

[统一 LOC self-check](../../../../evidence/correctness/inference-g04-g05-apple-ce14-acceptance/loc.json)
绑定3b5157532：全库3,587,223，自实现不含测试179,661，Apple模块src4,781。
third_party patch不计作本模块自实现；不以LOC代表完成度。

本有限域适用七门完成，停止重复普通验证和 P1。reopen trigger仅为本已验域可复现的
数学/编码/范围、权限/poison/旧reader/实际退休、OOM/交付缺陷、合格material成本回归，
或具名真实消费者要求的新常用机制/相关锁定依赖升级；新轮先scope/audit。
原G02 Windows尾项、CUDA资格、整个G04/G05与G06–G10继续，不由本记录代签。
