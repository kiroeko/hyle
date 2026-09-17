# R6 Vulkan Air Graph P1

产品候选 `8e45b10423686a0753aaa9d461c4ef3b117412c5`；Apple 测量快照
`a41d43d531365a1fb8617d60113b37924e9a9864`。按
[R6 性能合同](../../../docs/implementation/tasks/compute/vulkan-r6-performance-contract.md)，
普通 typed Graph 冷编译完整 strict-F32 FFN；参考独立描述锁定 ggml shader 的四个 dispatch。
两侧使用相同 HYLE native 运行槽、bind、checked submit/fence、生成结果 lease 与 owner 回收边界。

## 实际结果

- Air Mac16,12 / Apple M4，M1-MAC engineering profile，实际 interactive QoS；没有硬 P-core pin
  或 Formal 机器身份声明。MoltenVK 1.4.1、loader 1.4.350、device API 1.2.334、headers 357、
  glslang 16.3.0，subgroup 32、shared-memory 32 KiB，ggml pin `456172ec`。
- 16 个 A/A 资格进程、120 个产品进程与 30 个公共 control 进程全部完成；controls、资格与
  输入/制品/host 身份检查通过。M1/M16 共 16 项原 5% 非劣门全部 PASS，n=15 停止，未扩样。
- M1 八分层 median 吞吐比为 0.999569–1.001189，严格方向 2 WIN / 6 INCONCLUSIVE。
  M16 为 0.985050–1.011573，4 LOSS / 4 INCONCLUSIVE；两个 workload 整体方向均未决。
- M16 四个 LOSS 都在 peer-allocation-first 分层；执行正反序和两种 placement 均保留该负向。
  median 吞吐比为 0.985050–0.988134，对应约 1.2%–1.5% 的额外延迟，全部通过原 5% 工程门。
  其余分层 CI 跨 1。这里记录实际分配顺序与方向关系，不由相关性推断具体 driver/cache 根因，
  不把稳定小输改名为误差，也不据此另开性能纠正或追加样本。
- 公共 `verify` 返回 EXACT/CURRENT/QUALIFIED、`paths_verified=true`；它验证身份和资格，
  原始 summary 单独保存统计裁决。原始链接 map 显示 `submit_locked` / `observe` 均移动
  68 bytes、函数大小不变，见 [placement-review.json](placement-review.json)。

这些结果验证成熟原样 shader 接入、固定规则和冷 Graph 编译的工程非劣性；不证明 HYLE 算法优胜、
严格等速或整个 ggml runtime 的排名。裸 Vulkan 路径的取消/quarantine/结果 lease 责任不同，
按 NATURAL_ROUTE 单独诊断。原生边界成本、最终 ASan/三交付/selector 及 R6 终审不由本记录签证。
资格不外推 Linux GPU、AMD/Intel、其它数值格式、机器或版本。

## 原始数据与完整性

[raw.zip](raw.zip) 保留 P1 原始 manifest/summary/observations/controls、A/A 及各进程日志、
selector receipts、build cache、链接 map、启动器和独立参考 SPIR-V/生成头/来源 manifest；共
210 个具名文件，另含逐项 archive-manifest。Windows 接收后重算全部条目 SHA-256，通过；参考
manifest/header 与合格 A/A 身份一致。[summary.json](summary.json) 保留原始字节，
[identity.json](identity.json) 记录全部分层方向及归档身份。

- raw.zip SHA-256：`42ca72fa0c23b55f03fec9c6ba6244cf13bf94e6e92a21c712a6c9da57e0620f`
- summary SHA-256：`19e62cac5e58b3b0ff4c9e065b1d8b0381c86a33c0ee050247df8fcf1dbbd6a5`
- contract SHA-256：`ab0299a1092524006502ae0869198ab1cd2c645d83cd0fed76652ca64e50cf75`

此前的 [Air loader 启动失败](../../correctness/vulkan-r6-air-loader-setup/README.md)继续保留；
那些尝试没有有效性能样本，不能被本次 PASS 覆盖。
