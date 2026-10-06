# C5 当前源代码完整消费者批次

候选 `182f5d036ae214d13beba23d65646a3064545b0b` 相对
`46a55325a465a94da3d79d2e5e92b0cb6b2cfb62` 只增文档/证据，产品与测试源未变。
已经资格化的每入口594项 C5 数值矩阵通过 filter 排除，复用对应候选收据，不再重跑。

## Windows

[批次收据](windows/receipt.json)：ggml factor通过；剩余普通 Direct172项、Graph190项。
首次 Direct165通过、7skip；Graph183通过、7skip，XML保留原状态。
原因是 `HYLE_D5_QWEN35_FIXTURE` 未设置，现有本地提取 fixture 仍完整存在。

[补跑收据](windows/fixture-followup/receipt.json)对这14项原样补跑：全部通过、没有 skip。
使用本地原官方 Qwen3.5-2B 提取参数，文件 SHA 与原 receipt 对账，精确 fixture 身份随收据保存。
覆盖实参数 projection、FFN、GQA、卷积/GDN、MTP 边界以及九画像完整 recurrent 状态反馈。
没有重跑已经通过的其余348项，也没有把首次 skip 改成 PASS。

## Linux

当前批次尚在执行，未签 PASS；完成后附其实际收据。

以上是已有完整消费者的当前软件回归资格，尚不替代新增制品全部参数角色/required-region对账。
