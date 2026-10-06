# CPU 资源域审计与最后的索引接线反例

2026-10-06，Windows canonical HEAD `214cb3d48cb39416e8540f7188249105cde58b10`。
这是一批冷准入诊断，不是算子执行、性能或 UNIQUE/WIN 资格；产品未改、未重建、未运行 P1。
复用 TM 的现有 producer，1337 个前向产品/测试/构建输入 SHA256 全部一致；完整源码、原始输出、
编译命令、一次 fixture compile failure、输入身份在 [diagnostic.zip](diagnostic.zip)，计数和摘要见
[receipt.json](receipt.json)。MSVC14.50，i9-13900K，已锁 Torch2.8 CPU、优化 ggml 与 oneDNN。
子进程隐藏启动、DLL 位于 exe 相邻目录；没有更改机器电源、线程亲和性或用户服务。

## 实际冷结果

- 191 个合法 public descriptor 案例 × 4 Provider = **764 条 prepare 记录**，无 schema/其它错误。
  ggml 准入115/拒绝76，oneDNN15/176，ATen141/50，Native167/24。
  这些是冷能力计数，不是执行通过率；不同 Provider 不需要覆盖同一子集。
- 其中 Native 准入且三个成熟 factory 都拒绝的普通样例只有 **3 行**：三种浮点 storage 的
  I64 ScatterAdd。不是原库缺失：同版原 ATen `Tensor::index_add_` 对 I32/I64、F32/F16/BF16、
  两轴、重复索引与 empty 的 **156 个断言**实际通过。需有限 adapter 接线和产品资格，不能签 UNIQUE。
- 单节点普通 CPU Graph 七种实际消费 × 五种资源政策 = **35 次 compile/invocation 创建**。
  最终批全部成功；没有 submit，不能据此签数值、热分配或故障注入 PASS。
- 三个整域 proposal × 四 Provider = **12 条记录**。Native RMSNorm/Copy 准入完整 domain；
  三个成熟 factory 均明确 exact-only；动态 Exp 连 Native 也拒绝。不能把 Native 的整个目录
  宣称为 bounded-shape 支持。

## 资源政策真实影响

固定 shape：matrix F32 `[2,8]×[8,5]`；stable packed gate `[2,16]`；Bool Softmax `[2,8]`；
RMSNorm/Copy `[2,8]`；I64 Assign base `[4,8]`、indices `[3]`、updates `[3,8]`；E4M3FN→F32 `[2,8]`。

| Graph 冷政策 | matrix | stable gate / Bool Softmax / I64 Assign / FP8 widen | RMSNorm / Copy |
|---|---|---|---|
| 普通默认 | oneDNN | ATen | oneDNN |
| `kRequirePrepared` | oneDNN | Native | ggml |
| compiled retained ≤1MiB，unknown reject | ATen | ATen | ATen |
| compilation resident/peak ≤64MiB，unknown reject | Native | Native | Native |
| invocation ≤1MiB，unknown reject | Native | Native | Native |

首次 compilation ≤1MiB 被 Graph metadata 自身耗尽，不能归因 Provider；仅把这项诊断容量提高到
64MiB，保留第一次输出。没有用这一改变调整性能阈值或产品内存合同。

关键区别：ATen 的 **prepared artifact retained bytes 已知**（Program+描述），但 compilation 额外
私有峰值和 invocation 私有存储 unknown；不能把它们统称“所有内存 unknown”。oneDNN MatMul
prepared，norm/reorder 当前 managed；ggml普通程序 prepared，但 native 私有内存 unknown；标准库
Scan/TopK 是独立 borrowing 接缝，已有精确声明，不能并入所有 ggml unknown。未知不等于零。

整域样例：F32 derived-dense `[rows,8]`，同一 rows 表达式范围1..4，固定 hidden width；
RMSNorm 与 Copy 的 whole-domain 接缝可按现有已资格化实现保留，不能外推动态其它数学。

## 有限后续

1. [AS 接线卡](../../../docs/implementation/tasks/compute/cpu-aten-scatter-add-admission.md)：
   原 ATen ScatterAdd，先产品数值/布局/alias/失败与两平台普通三交付，仍冻结自研比较。
2. 之后固定同合同普通域的有限自研去留批；严格资源/整域边界单列，不以 fallback 优先级或
   本次 prepare 成功签 WIN，不删 SIMD 基础。冷诊断没有证明所有成熟库的潜在能力缺失。

源码锚点：`aten_cpu_provider.cc::propose/prepare/create_invocation`、
`native_provider.cc::validate_bounded/propose`、`ggml_cpu_provider.cc::propose/runtime`、
`standard_index.cc::costs`、`onednn_provider.cc::propose` 与公开 Graph 三种 storage limits。
