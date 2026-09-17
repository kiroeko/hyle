# R6 Vulkan Windows Direct P1

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`；测量快照
`3894b72e3b4fbbda9457e2c31730e48bd0d43c49` 只另含 benchmark Windows 路径规范化。
按 [R6 性能合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)，
在 W1-WIN / RTX 4070 上比较普通 Direct 完整 strict-F32 FFN 与独立锁定 ggml shader recipe，
两侧都承担相同 Recorder、提交、fence 和完成 owner 生命周期。

## 实际结果

- A/A 资格矩阵 16 个进程通过；执行正反序、分配正反序、两种 placement 均覆盖。
- K4096/H11008/O4096 的 M1、M16：共 16 个验收项全部通过原 5% 非劣门，n=15 停止。
  120 个产品进程、30 个公共控制进程；没有扩到 n=31。
- 完整控制项通过，输入、制品与 host 身份稳定；公共 `verify` 回放为 EXACT/CURRENT/QUALIFIED。
- M1 各 strata 的 candidate/peer 吞吐比中位数为 0.99049–1.00422；M16 为
  0.99188–1.00547。全部 16 个置信区间包含 1，严格方向均为 INCONCLUSIVE。
  这是成熟实现接入的工程非劣证据，不是 HYLE 算法优胜或整个 ggml runtime 的排名。
- 实际 caller 为 CPU 2，进程 mask 为 0x14，覆盖两个不同物理 P 核；affinity_ok=1。
  两个 placement 的 `submit_locked` 与 `observe` 地址相差 48 bytes，保留链接 map 复核。

首次尝试在 CMake 构建阶段因独立 peer 路径反斜杠产生无效转义停止；尚未进入 A/A 或性能采样。
修正仅限 benchmark CMake 路径处理，失败日志一并保留，不混作性能失败或删除旧记录。

## 制品与限制

[identity.json](identity.json) 记录候选、合同、汇总、归档 SHA256；[summary.json](summary.json)
保留公共仪器的原始裁决。[raw.zip](raw.zip) 含 205 个证据文件及逐项 manifest：全部原始样本、
A/A、控制、机器/制品身份、合同锁、selector receipts/跳过原因、链接 map、controller 与失败记录。
归档全部 entry 已重新校验；原 scratch 中制品保留供公共 verify 检查绝对路径。

[reference-shaders.zip](reference-shaders.zip) 另保留独立参考的实际 SPIR-V、生成头及完整来源/
编译命令 manifest；其身份与已通过的 A/A 匹配，hash 记在 identity.json，原 raw.zip 不变。

本记录只签 Windows Direct；Windows Graph、Air Direct/Graph、单列的原生边界成本、最终交付与
中央 selector/终审仍待完成。未知设备、其它数值格式和整个 Vulkan 路线不由这一次 P1 自动签证。
