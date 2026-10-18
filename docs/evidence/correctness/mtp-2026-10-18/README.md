# MTP 底座审查与补全

本轮针对多系统上的单机 CPU、CPU＋一张 GPU、单 GPU，检查标准线性 MTP/speculative 的
“草稿→多 token 目标验证→接受前缀→继续计算”。不实现 loader、采样/接受策略或完整模型生成循环。
用户入口见[状态版本](../../../user/state-versions.md#mtp-接受前缀并继续计算)。

## 发现及修改

| 实际缺口 | 补全 |
|---|---|
| GDN 只导出最终状态，部分接受后无法直接恢复中间状态 | 新增 `snapshot_count`，CPU/CUDA/Metal/Vulkan 接通锁定 ggml 已有快照；设备内整理成正序 `[B,K,Hv,Dv,Dk]` |
| 融合 raw-gate recurrent 只允许 T=1 | 放宽 T 轴，raw a/b 随 token 变化，negative_a/dt_bias 广播；透传快照数，自定义 schema ID 升版 |
| 失败/取消且 writer 被 producer 消费后，保留的页 reservation 无法重写 | `PagedPageWriter::discard()` 在原物理范围实际重新可分配后恢复新 generation；旧数据有效行数归零，仍有权限/native use 时拒绝并可重试 |
| CUDA GDN 256 字节输入对齐阻止直接使用逐 token gate 子范围 | 锁定 F32 内核使用标量读取；GDN 改为自然对齐，真实前缀接续验证 offset 4/8/12 |

另纠正原公开注释的 head mapping：现有内核和既有数值 fixture 都是 `h % Hk`，原注释误写为
连续分组。未改变内核公式；连续分组的模型仍显式展开 Q/K 或重排权重与状态。
默认 K=1 保持原四维最终状态与原 payload 路径，不生成快照索引或增加快照整理。

卷积无须新增状态算子：接受 p 个后取验证输入的 `[:,p:p+K-1,:]`。
已有 StateStore 的共享、dirty clone、finalize 和 expected-base publish 可一次提交各层
main/draft 的 recurrent/conv/KV/位置等成员；不新增 MTP 事务或采样 API。
cached attention 的右对齐 causal、显式 mask、逐页 CopyRange 与有效尾长能够表达线性验证和前缀。
不同 batch 接受长度的非连续状态可 gather 或按 batch 复制，并不总是零复制。

## 验证范围

- Schema：默认/尾部快照、无效计数、输出尺寸溢出、多 token raw-gate 广播。
- CPU：两个锁定 ggml pin、baseline 与现有 AVX2 路径；独立 FP64 参考检查 B=2、T=5、
  Hk=2/Hv=6、K=1/3/5、不同输入重绑，检查所有快照、所有输出及输入状态不变。
- CUDA：RTX 4070；相同非对称批次与快照测试，以及 B1/T3/H16/D128 的融合 raw-gate 多 token 数值回归。
- Metal：MacBook Air Mac16,12 / Apple M4；独立原生 probe 覆盖相同非对称批次和 K1/3/5，
  另测融合 raw-gate B1/T5/H16/D128；36 次输出比较最大绝对误差 `5.71e-9`（预算 `3e-6`）。
  未运行 canonical Apple gtest，实际还运行了下面的公开消费者。
- Vulkan：RTX 4070；非对称批次快照与融合 raw-gate 多 token 两项定向回归通过。
- 页恢复：21 项定向测试通过；覆盖真实 CPU accepted DataLoss/Cancelled、权限/native use 保活、
  重试、新 generation、旧 pending frame 失效及关闭边界。未制造实机 GPU 故障。
- [公共消费者](../../../../tests/compute_state/public_state_consumer.cc)：CPU/CUDA/Apple/Vulkan 接受 0..4 个 token，
  同时提交 GDN、conv history、KV 和有效长度，随后用库自己的输出继续 GDN/conv/attention，覆盖 Direct/Graph。
  CPU→CUDA 与 CUDA→CPU 搬运实际 accepted snapshot 子范围和 history 后，在目标侧继续计算。

这是相关主机 TU 的增量编译与功能验证，复用原生缓存，不进行完整 SDK、全库或 CI 构建。
CUDA 使用显式 ggml component Context 与当前 Graph adapter，未重新交付全默认 provider 组装；
旧 CUDA 原生缓存缺少三项无关 IQ1 codec，验证副本对它们明确 decline，未修改产品编码支持。
CPU＋CUDA 同进程统一使用 mixed ggml pin，未混用两套 ggml ABI。
测试会冷准备多种形状，不证明动态热路径无重编、零拷贝、capture 或整引擎 tokens/s。
结构化输入与独立公式也不替代第六项的真实权重、完整 Qwen3.5-2B 引擎及后续 MTP 生成验证。

命令、源身份和小型日志分别见 [CPU/Vulkan](cpu-vulkan-results.zip)、
[CUDA 与跨设备接续](windows-cuda-state-results.zip)、[Metal](metal-results.zip)及
[页恢复](page-recovery-results.zip)。页恢复包中的 hybrid 源码记录属于该子任务的编测前快照，
跨设备已执行结果以 CUDA 包为准。二进制及大型依赖保留在可复用缓存，不重复收进 docs。
