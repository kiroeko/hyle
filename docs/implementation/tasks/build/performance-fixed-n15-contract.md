# 统一性能工具：允许预声明的单阶段 n=15

<!-- MODULE REVIEW: CLOSED -->

本卡是独立 tooling 修复，不属于 CPU Native 产品 FIX，不重开模块或改统计核心。
baseline `0bb1e82b`。具体反例：CPU 官方完整片段合同固定 `sample_schedule: [15]`，
通用 contract loader 接受，runner extension 却硬要求 `[15,31]`，在配置/采样前拒绝。
实际产品样本为零。[仪器合同](../../../developer/performance/instrument-v1.md)规定
“升级 n=31 只允许合同预声明”，没有要求每轮必须授权升级。

有限 FIX 只有 runner validation 允许 `[15]` 或原 `[15,31]`；保留相同 n=15 最低样本、
四顺序、控制、5%门、tail alpha、holdout=false。运行器既有 `len(schedule)==2` 分支已能
跳过单阶段升级，无需改测量/统计/重放。其它样本数继续拒绝。

验收为 runner canaries 和 evidence runner canaries；添加单阶段接受、低于最低样本或任意
第二阶段拒绝的针对测试。docs/closure/diff 检查；不构建产品、不计入产品性能样本。
一次修复及测试；通过即 CLOSED，不扩展到公共仪器重构。

结果：允许两种既定 schedule，既有运行循环及统计/重放未改。runner 38项、evidence runner
9项 canaries 通过；其中新用例证明 `[15]` 保持原值且 `[14]`、`[31]`、`[15,29]`、
`[15,31,63]` 均拒绝。docs/closure/diff 检查通过后本工具任务 CLOSED。
