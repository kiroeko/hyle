# Qwen3.8-2.4T 混合权重 MoE 批次

本收据只签本批权重角色、grouped expert 和完整 R10/R11 整链。四路线 Direct/Graph 的
**32/32 项批次测试通过**，不代签整份 profile、三交付或 P1；176 格 ledger 仍为 48 PASS、128 未运行。
原始日志、XML、候选身份和逐用例结果见 [receipt.json](receipt.json)。

## 真实形状与边界

以已固定的 `unsloth/Qwen3.8-2.4T-A95B-GGUF@567d3e6ac26c5474b18311e619c04350fb9a5556`
header 为准，所有权重内容是按真实格式生成的合成数据，不声称下载了 2.4T checkpoint。

| 角色 | UD-IQ4_XS | Q8_0 |
|---|---|---|
| 512-way router、shared gate | F32 | F32 |
| routed gate/up/down | IQ4_XS | Q8_0 |
| shared gate/up | Q5_K | Q8_0 |
| shared down | Q6_K | Q8_0 |

保留 H=8192、expert width=2048、512-way/top-10。仅把常驻 expert 容量缩成 11，包含近并列
边界的第 11 个 expert；三次输入重放改变路由 ID。官方 BF16 和两个 GGUF 画像共用同一套
Direct recipe/Graph 结构。独立 oracle 解码原生 blocks，再以 F64 累加计算 expert、shared、
概率加权和最终结果。量化链复用既有误差门，并按参考最大值归一化，防止微小结果被绝对容差掩盖。

## 实现与裁决

- `ea5cb859`：批量接通四路线八入口，撤销 `b20c95bd` 不必要的公共 schema 改动。原有 schema
  已允许 F32 logical blocked weights；新增公共测试与实际 grouped 执行证明该能力原本存在。
- `ceef59d8`：CPU 实测暴露 F32 router 被 oneDNN/ggml adapter 双重拒绝，落到 native。只给已有
  ggml MatMul 增加 plain F32 rank-2、连续左输入、output-major 转置右权重的准入条件；继续调用
  上游 `ggml_mul_mat`，没有新增数学、公共 API 或热路径搜索。GPU 三路线无需产品修改。
- `2e3d4c8d`：CPU 旧测试要求所有转置 F32 MatMul 都选 native，按新路由合同更新该断言。
  数值、变化输入和生命周期断言保留。

## 验证

| 范围 | 结果 |
|---|---|
| CPU Direct / Graph 本批 | 各 4/4，包含在完整 owner 运行中 |
| CUDA Direct / Graph 本批 | 各 4/4，Windows RTX 4070 |
| Vulkan Direct / Graph 本批 | 各 4/4，Windows RTX 4070 |
| Apple Direct / Graph 本批 | 各 4/4，MacBook Air M4 |
| 公共 Compute ops、ggml CPU owner | 51/51、41/41 |
| CPU Direct / Graph 完整 owner | 初跑 77/78、92/93；各自唯一失败为旧路由断言，迁移后该测试复跑通过 |
| 共享模型输入工具 | 13/13 |

不把完整 owner 的初跑原始 XML 改成全 PASS：两个失败及其定向复测日志均保留，产品候选未变。
GPU 只运行本批定向测试；完整 profile 的全套 selector/owner suite 留到其它角色接线完成后的批末。
各构建树低并行复用，重型验证串行；Windows CUDA 运行依赖放在 EXE 同目录。

统一脚本统计：Compute + Graph **58,047 实现行 / 228 文件**，不含测试。相对本批开始净增加
4 行实现，仅为 CPU ggml 的准入条件。未完成项仍包括其它精确画像 region 的组合、量化 grouped
错误 ID 的独立失败收据、完整资源/负例门、D5-A5 三交付和 D5-A6 P1。
