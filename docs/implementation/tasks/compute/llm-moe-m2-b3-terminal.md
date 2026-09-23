# M2-B3 成熟 Provider 递归混合状态终局

<!-- MODULE REVIEW: CLOSED -->

2026-09-24，B3 在产品候选
`92314d71150510813eb5165e098989b74b965a81` 上关闭。实现只接入锁定 ggml/MLX 的原版能力：公共
`CausalConv1d` descriptor 描述 `[B,T+K-1,C] × [C,K] → [B,T,C]`；CPU、CUDA、Vulkan lower 到
`ggml_ssm_conv`，Apple lower 到 MLX depthwise `conv1d`；公共 GatedDeltaNet 接受 scalar gate 与
per-channel KDA gate，CPU/CUDA recipe 支持原版 packed 结果的两个独立 owning output，Apple/Vulkan
复用已有 whole region。没有新增 HYLE 数学 kernel、CUDA `.cu`、Metal shader 或 SPIR-V。

## 实际资格与验证

| 路线 | 已绑定域 | 实际验证 |
|---|---|---|
| CPU | dense F32 causal conv；dense F32 equal-depth scalar/KDA GDN，显式初态与 scores/state 两输出 | Windows shared：公共推导、ggml Direct、CPU/Graph route 全过；KDA 的零更新独立 oracle 证明两输出边界 |
| CUDA | SM89 dense F32 causal conv，`C % 128 == 0`；dense F32 equal-depth scalar/KDA GDN，native stream/device buffers | Windows RTX 4070：CUDA route/Graph/provider 与新增 conv/KDA 实跑全过；unsupported shape 仍由原版 `supports_op` 拒绝 |
| Apple | matching floating MLX depthwise conv；dense F32 equal-depth scalar/KDA 使用原版 ggml Metal region | MacBook Air shared 5/5；Mac mini static 5/5；两机均执行新增 conv/KDA、重复提交、资源与完成路径 |
| Vulkan | dense F32 causal conv 与 scalar/KDA GDN 的 ggml managed region | Windows RTX 4070：完整 Vulkan route 122/122；相关 provider/Graph gates 通过；原版 backend capability 仍作最终裁决 |

Windows 候选还完成低并行增量构建以及 Compute/Graph 相关 CPU/CUDA/Vulkan selected gates。首次 KDA
oracle 把 per-channel gate 错套到 value 维，原版源码证明 gate 作用于 state 的 key 维后修正；首次
CUDA conv 用 `C=2` 被原版 `C % 128 == 0` 资格正确拒绝，改用真实 `C=128` 消费形状后通过。这两项
都是测试合同订正，不是放宽产品实现。P1 对本批 `NOT_APPLICABLE`：没有冻结的完整递归模型片段
候选；P0 已确认 prepare 后保留原生图/编译程序，replay 不搜索 Provider、不走 host payload fallback。

## 有限残差

- F16/BF16 recurrent、更多 state snapshot、conv+bias+SiLU 完整融合和非 equal-depth GDN 没有在本批
  获得四路线共同资格，保持 `UNPROVEN`，归 M3-03 汇总，必要时才进入 M4 缺口审计。
- 本地 2B 的 24 层 B16/Q4/Q8、257-token chunk 与 8 步 teacher-forced decode 属于 M3 模型链验收；
  本批没有模型 loader，不能把单 fragment 结果写成完整模型 PASS。
- CPU/CUDA 的普通 Direct 已发布 GDN 两输出；Graph 能通过 Compute leaf 消费该能力，但完整多输出
  ggml region 合并未在本批证明。Apple/Vulkan 已证明 whole region。该差异进入 M3-03，不用自研
  recurrent 算法补齐。
- CUDA causal conv 的锁定 upstream 限制 `C % 128 == 0`；其他通道数不签支持。CPU/Apple/Vulkan
  各自继续按原版 backend 的实际 capability 准备期裁决。

这些残差均有具名成熟 Provider 边界，不影响 B3 的退出条件“已准入且适用的成熟入口完成接线”。
持续承载 M2 的组件 owner 保持 `AUDITED`；下一批须另行 SCOPED→AUDITED 后才可修改产品代码。

统一 `loc-report.py --self-check` 在不可变候选上报告 Compute+Graph 实现 **59,253 行 / 252 文件**，
不含测试、文档、样例与第三方源码。
