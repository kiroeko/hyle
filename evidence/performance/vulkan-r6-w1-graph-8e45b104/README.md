# R6 Vulkan Windows Graph P1

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`，测量快照
`3894b72e3b4fbbda9457e2c31730e48bd0d43c49`。按
[R6 性能合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)，
比较普通 typed Graph 完整 strict-F32 FFN 与独立锁定 ggml shader recipe；两侧承担相同
Invocation、bind/submit、fence、生成结果 lease 和 owner 回收。

## 实际结果

- A/A 16 个进程通过；执行/分配正反序及两种 placement 均覆盖。
- K4096/H11008/O4096，M1 与 M16：16 项原 5% 非劣门全部 PASS，n=15 停止，未追加样本。
  实际完成 120 个产品进程和 30 个公共控制进程；全部控制项通过。
- 输入、制品、host 身份稳定；公共 `verify` 回放通过，资格为 EXACT/CURRENT/QUALIFIED。
- M1 的 candidate/peer 吞吐比中位数范围为 0.99407–1.00492；M16 为 0.99199–1.00486。
  全部 16 个区间包含 1，严格方向均为 INCONCLUSIVE；不声称 HYLE 算法胜出。
- 链接 map 证实两种 placement 的 `submit_locked` 与 `observe` 均移动 48 bytes。

[summary.json](summary.json) 是公共仪器原始裁决，[identity.json](identity.json) 保存候选、合同、
汇总和归档 hash。[raw.zip](raw.zip) 保存 202 个原始证据文件及逐项 manifest，包含所有样本、
controls、A/A、机器/制品身份、selector receipts/跳过原因和链接 map，逐项复核通过。
[reference-shaders.zip](reference-shaders.zip) 另保留独立参考的实际 SPIR-V、生成头和完整来源/
编译命令 manifest；其身份与已通过的 A/A 完全匹配。

本记录只签 Windows Graph；结合 [Windows Direct](../vulkan-r6-w1-direct-8e45b104/README.md)，
Windows 两个入口的正式 P1 均完成。Air 两入口、原生边界诊断、最终三交付和中央 selector/终审
仍须各自完成，未将这些剩余门写成 PASS。
