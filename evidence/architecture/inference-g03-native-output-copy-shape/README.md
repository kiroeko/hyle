# G03 Apple 原生输出到 F2 的公开消费形状反证

2026-10-14；固定产品 b428、活动 SCOPED cfb883，无产品修改。
[公共探针](public-probe.cc)、[实际结果](final.log)、[累计记录](receipt.json)、
[源码身份](source-identity.json)与[全部尝试](raw.zip)。验证机为 MacBook Air
Mac16,12 / arm64 / macOS 27.0.1 / Apple Clang 21，公共 target 仅链接 hyle::compute。

原 Context/Unary Abs Program/Invocation 重用两次，输入和输出为 F16 {1,2,256}，
每次 1024 bytes。输入依原公共规则完成 async_eval/synchronize/wait 后才绑定；
producer 完成后 import_readonly 保留原 MLX owner，前一次导出也保留，排除地址回收混淆。
两次导出的 MemoryResource 身份都与设备 ref 不同，彼此也不同。固定以设备 ref 声明
源的 CopyBatchProgram 每次 bind_source 都返回 InvalidArgument。

按每次导出的实际 resource 重建 CopyBatchProgram/Invocation 后，原 F2 1024 bytes
device-to-device copy 成功，完整 F16 bits 分别为 0x3000 / 0x3400。这证明当前自然
公开路线可完成 ready producer 到 copy，却要求每次导出的 cold program/slot 重建。
源码解释是 ImportedArrayStorage 的逐导出 resource 身份，与 F2 严格 resource 相等检查。
仍需冻结同设备 readonly 来源资格与 owner 的有限修正，不能从本探针直接推导具体 API。

此证据没有 pending 输出、完整页更新到 attention、warm allocation 计数或性能结论，
不签整个 G03。输入准备的 host 等待、producer 结果等待及最后 copy 等待均保留。
普通数值 Abs 与下一阶段 QKV/页旅程共用这个接缝；不能只把每步重建写成引擎适配后结案。

前三个输入保留：未求值绑定失败；const array 调用非 const eval 的编译失败；
eval 后仍未达到 available 的绑定失败。第四个输入采用既有 ready 用法通过；累计
42.421 秒，所有 configure/build/run 与输入 SHA 均保留。旧前缀记录的 candidate/scope
字段由复用脚本带入，不能作为资格；实际固定源码身份与本说明界定此次有限探针。
