# R6 Vulkan Air Direct P1

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`；Apple 测量快照
`a41d43d531365a1fb8617d60113b37924e9a9864`，包含 benchmark 冷 loader 路径绑定。
按 [R6 性能合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)，
比较公开 Direct 完整 strict-F32 FFN 与独立锁定 ggml shader recipe；两侧承担相同 Recorder、
checked submit/fence、完成与资源 owner 生命周期。

## 实际结果

- Air Mac16,12 / Apple M4，M1-MAC engineering profile，实际 interactive QoS；不声称硬锁 P 核
  或 Formal 机器身份。MoltenVK 1.4.1、loader 1.4.350、device API 1.2.334、headers 357、
  glslang 16.3.0；subgroup 32、shared-memory 32 KiB，ggml pin `456172ec`。
- 16 个 A/A 资格进程、120 个产品进程、30 个公共 control 进程完成；全部 controls 和身份检查
  通过。K4096/H11008/O4096 的 M1、M16 共 16 项原 5% 非劣门全部 PASS，n=15 停止，未扩到 31。
- M1 八分层 median 吞吐比为 1.000268–1.001713，严格方向为 7 WIN / 1 INCONCLUSIVE；
  M16 为 0.988563–1.021558，方向为 4 WIN / 4 INCONCLUSIVE。局部正向、负向点值及跨 1 的
  CI 全部保留；两个 workload 的整体严格方向均为 INCONCLUSIVE，不签算法优胜或严格等速。
- 原始公开 `verify` 返回 EXACT/CURRENT/QUALIFIED、`paths_verified=true`。原始 summary 的
  非劣裁决与 identity 验证分别保存；没有把制品验证本身当作统计门重放。
- 两种 placement 的 `submit_locked` / `observe` 地址均移动 68 bytes，函数大小不变；保留
  [链接 map 复核](placement-review.json)。使用既有冷 prefix，不加热路径 padding 或 LTO。

这是成熟原样 shader 接入和固定规则的工程资格。没有比较整个 ggml runtime，也没有拿缺少取消、
quarantine 和生成结果 lease 的裸 Vulkan 路径签同合同输赢。Graph、原生边界诊断、最终三交付/
selector 与 R6 终审不由本记录签证；Windows 的独立结果继续保留。资格不外推 Linux GPU、
AMD/Intel、其它数值格式、机器或版本。

## 原始数据与完整性

[raw.zip](raw.zip) 包含原始 P1 manifest/summary/observations/controls、A/A 及各进程日志、selector
receipts、构建 cache、链接 map、任务启动器和独立参考 SPIR-V/生成头/来源 manifest，共 210 个
具名文件，另含逐项 archive-manifest。Windows 接收后重新校验全部条目 SHA-256，通过；参考
manifest 与 compiled header 的 hash 与合格 A/A 身份一致。外置 [summary.json](summary.json)
保留原始字节；[identity.json](identity.json) 保存归档与测量身份。

- raw.zip SHA-256：`49691b5b4144792064ae000f401528f6765a071b3fa95fa527db8fcb81da7f76`
- summary SHA-256：`5ec80c1b3898545beb814d19f1d0462b249e3823719de5d252423502ef2c1f8f`
- contract SHA-256：`8bd424c0fc95b3a9cfeb09718f410f19cc6a08554418ca32f8b00b71c81305f4`

两次正式采样前的 loader 启动失败仍由
[独立原始记录](../../correctness/vulkan-r6-air-loader-setup/README.md)保留，没有覆盖或改写为 PASS。
