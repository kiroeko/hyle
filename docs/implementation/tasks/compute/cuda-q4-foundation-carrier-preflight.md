# CUDA Q4 foundation：真实 carrier 前置验证

## 研究卡 / AUDITED

- owner：Graph v10/G107 与总路由审计工作包 5；baseline `4fe60889`。这是基础架构的具名实验，
  不新增 shipping Provider、不为研究 sample 创建假 component 或 CLOSED 证书。
- 需求：同一逻辑 Q4 MatMul 必须有明确的格式、CUDA Queue、权重表示和完整执行成本，才能决定真实
  HYLE/vendor/第三方路线。普通 packed int4、ggml Q4_0/Q4_K 和 NVIDIA FP4 不是可互换的 dtype。
- 主纵切：量化线性层的 decode/prefill；正交面是 caller-owned device buffer 与 native library 互操作。
  不引入模型 loader、量化校准、调度策略或横向算子目录。
- 实现前冻结：一个独立 source/static/shared-compatible C++ sample，只用公开 HYLE Compute/CUDA
  接口；native CUDA TU 不包含 HYLE 私有 SPI。先在现有 Windows shared SDK 上执行 carrier preflight，
  后续完整 Provider/Graph candidate 再执行预声明三平台/三交付门，不因 sample 首次成功提前签发。
- 预算：一个 Q4_0 表示、一个 direct/native 与 vendor 验证程序、一次 fixture 纠正；不做 P1/调参，不
  构建完整 ggml CUDA 算子全集。发现 API/ownership 反例先记录 owner，不用私有 ggml metadata 绕过。

## 版本化来源与裁决

2026-09-15 从五个 clean reference 的远端 HEAD fast-forward 更新后检查；只更新参考仓库，不改 HYLE
产品依赖锁或沿用它们的新版本资格：

| reference | clean commit |
|---|---|
| ggml | `456172ec733a135778adcd32d00e576a58232e45` |
| llama.cpp | `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` |
| vLLM | `51918e252af4dfff381001f42b7ebe6a0454501b` |
| SGLang | `06992c92ad4494fe9f9636e75be3e35040512447` |
| ds4 | `9139e2ae58a41503968a500f36f75895c1ba63fc` |

本实验的直接 strong peers 是 ggml 的 Q4_0 编码/公开 CUDA runtime 与 NVIDIA cuBLASLt 13.3；另外三者
用于核对冷权重准备、decode/prefill 与显式 stream 的正常消费，不宣称已重新审查它们全部能力。

- ggml `src/ggml-common.h` / `src/ggml-quants.c`：Q4_0 每 32 个逻辑值使用 18 字节，FP16 scale 加
  16 个 low/high nibbles，值为 `(q - 8) * scale`；与 `CUDA_R_4F_E2M1` 不相同。
- [cuBLAS 13.3](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html) §3.1.4/3.4.17：FP4 为
  Blackwell 10.x+ 的 E2M1/block-scaled 路线，不能直接吃 Q4_0，RTX 4070 也不具备该资格。
  可用 FP32 cuBLASLt + **显式冷解码**作为功能 carrier，但必须计入更大常驻权重，不能冒称 native Q4。
- 最新 ggml `ggml_init` 仍以 aborting `GGML_MALLOC` 创建 context；CUDA `event_record/wait/synchronize`
  使用 fatal `CUDA_CHECK`，CUDA public header 无通用任意 device-pointer/stream import，graph-plan
  callbacks 为空。它不是现有 CPU blocker 已恢复，也不是一个能直接签发 fallible shared adapter 的 API。
  后续 ggml CUDA factor 必须单独解决这些具体 error/ownership 边界；本 sample 不调用私有 kernel、
  不伪造 buffer，也不把它的未交付能力算成已完成。

## 有限实验

| ID | 行为 | 成功只证明什么 |
|---|---|---|
| CQ1 | caller-owned Q4_0 device bytes 上执行 warp-reduction MatMul；与 host decode/double accumulation 比对 | 格式、尾部、正常 F32 数值与 HYLE Queue/Completion 合同；不是 SOTA kernel |
| CQ2 | 冷期把同一权重显式转为 F32 并上传；只查询一次 cuBLASLt algorithm，caller-owned workspace 与 HYLE Queue 重放 | vendor 的真实 device/stream/storage seam；不隐藏 32/4.5 倍权重占用，不声称自动路由/零分配 |
| CQ3 | M=1/7/33、N=65、K=96，三次变化输入，非法 K 拒绝；最终 Completion 后读输出 | decode/prefill、partial output tile、输入变化与提交完成边界；不代替完整模型测试 |

HYLE 原生样例复杂度 O(MNK)，额外 device workspace 为零；vendor preflight 显式保留 NK*4 字节冷转换
权重及算法要求的 workspace。两者不在此次做计时比较。未来 Provider 应只在 cold region compile 的
显式 constant/materialization 合同下打包；不能把这个 sample 的初始化阶段悄悄搬进 warm submit。

尚未完成：真实 HYLE/vendor/third-party Provider proposal/region、两种 residence、完整成本裁决、
bounded/exact/capture 组合与最终 P1。本卡通过也不关闭这些门，更不改变 Compute/Graph 的未验收状态。

## 首次运行与有界诊断

样例候选 `e8e0c95e` 在 Windows installed shared SDK 上编译成功，首次运行却在 M=1/replay=2 的
cuBLASLt 输出位置 0 得到预填充值 12345，而 double oracle 为 6.3671875。未运行 P1，也未放宽数值
容差。CUDA `MemoryResource::write/fill/copy` 的实现分别直接返回 `cuMemcpyHtoD/cuMemsetD8/cuMemcpy`，
但 Mem 的公开合同是同步 range operation。
[NVIDIA 同步规则](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html)明确指出 pageable
H2D 可能在最终 DMA 完成前返回，D2D 与设备 memset 也不保证 host 完成；因此有具体的同步合同反例假设，
不能先归咎于 cuBLASLt 算法或 Completion 缓存。

唯一诊断变体增加显式 `--synchronize-upload-control`：在同步 write 后等 legacy stream，其他数值、
kernel、算法、HYLE Queue/Completion 不变。默认路线保留原行为，control 成功也不算产品已修复；若得到
支持，先为 CUDA backend 建立独立同步资源修复卡，再更改产品。此诊断不引入主执行路径的全设备同步。
