# CUDA U3 R1：当前官方与三量化 sidecar 的原生资格

候选 `da67204ecae4d3a347ab6c52992ed06c76529517` 在 Windows 与 WSL Linux 实际运行通过。
每平台普通 Direct/Graph 各 7 项当前模型测试，无 skip、无零过滤；原 ggml Provider 资源与编码
Windows 33+8 项、Linux 34+8 项通过。Linux 额外包含原 capture 失败注入与 invocation 恢复。
精确命令、候选、结果、源码 SHA 与原失败见 [receipt](receipt.json) 和 [native batch](native-batch.zip)。

新增通过的三个 sidecar 是 Qwen3.8-27B Q3_K/Q4_K/Q6_K 完整 MTP、Flash Q8_0/MXFP4/BF16
完整 DSpark、Flash BF16/MXFP4 完整 DSpark。保留三 stage、真实内宽、实际 state/cache 反馈、
confidence/Markov 与全词表；head 分片降低驻留，不删 vocabulary。不声称全权重常驻或模型 engine。
当前官方 Pro/Flash 的五项完整程序也在同一候选两入口重新通过。

## 具体修复与成本

- Qwen 大型 quantized embedding 仅放宽原选中 row 解码的逻辑元素限制，physical bytes、stride、
  checked IDs、输出及其它 operation 原限制保留。
- Qwen MTP 的 noncausal attention 采用原 ggml null-mask/无 diag 程序。新增独立源方程验证
  F16/BF16、Q 长度大于 KV、正/零/负 scale、两独立 slot、变化地址和重复执行。
- 原固定 VMM 池每 backend/stream 预留 32 GiB，完整 Flash 接线触发 Windows driver code 2；
  Linux live stack 也停在 `cuMemAddressReserve`。冷 accessor 改选原 ggml cudaMalloc 缓存池，
  保留独立 stream/backend/slot 和原数学/融合。不添加用户开关、不禁用全局 CUDA VMM。
  原 private 缓存/分配仍属 managed 未知上界；不宣称无分配或性能胜利。

量化差分参考独立拥有原 ggml CUDA backend/graph/buffer，使用同原 GPU activation codec，
证明 adapter 接线。它不是另一个性能 peer；独立物理 codec/方程资格仍由基础格式测试承担。
CPU 默认 fixture 数学保持，未将 CPU vec-dot 直接当作 CUDA 量化参考。临时 abort/terminate
诊断钩子和日志输出已删除；诊断失败和栈仍归档，不改写为 PASS。

## 尚未完成

本证据只覆盖上述原生程序及实际原 Provider 检查。全部固定 GGUF 数值/shape 合同、其余
49 family-region 关联、最终联合 CUDA-only/mixed 的 source/static/shared 交付、U4 冷默认/
best-of 和 U5 清退仍待完成；本卡仍 AUDITED，不签 CUDA 收口。

统一脚本 [LOC](loc.json)：Compute + Graph 全部实现 **60,647 行／253 文件**，不含测试、
bench、第三方数学或构建脚本，`--self-check` 通过。
