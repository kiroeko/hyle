# 四路线架构定案：研究卡与裁决

## 本轮范围

- baseline：`fac7c9c5`；2026-09-15 用户要求开始剩余架构调研。
- 只研究 CPU、CUDA、MLX/Metal、Vulkan 四条独立路线。模型 CPU+GPU 联合推理明确不在范围内。
- 输出：四条路线的资源/准备/执行/完成方案、共同合同、旧代码复用/改写/删除清单，以及有限未验证项。
- 不实现正式 Provider、不填算子目录、不更改 registry、依赖锁或公共治理工具。旧 goal 不自动恢复。
- 主纵切：真实 FFN 的第三方计算与自定义 kernel 组合；反偏置：两次 GPU compute pass 的资源依赖和重复执行。
- 参考：沿用主审计记录的具名 ggml/llama.cpp、vLLM、SGLang、ds4 pins；MLX v0.32.2，厂商公开 API。
- 预算：源码与必要实验最多 90 分钟；最多两个新独立探针，每个最多两次编译/夹具纠正，不扩大矩阵。
- Apple 探针：MLX GPU MatMul → C++ `fast::metal_kernel` 自定义 SwiGLU → MLX GPU MatMul；变化输入、
  六次重复调用、显式最终完成、double oracle，沿用绝对误差 `1e-4 * max(1, abs(expected))`。
  检查不需要中间 host read/wait，编译缓存 trace 与资源 owner；不据此签零分配、无 JIT 或性能胜负。
- Vulkan 探针：冷建 pipeline/buffers/descriptor/command buffer；两个 storage-buffer compute pass，
  中间明确 compute-write→compute-read barrier，最终 fence 后读取；三次变化输入复用录制命令，
  先等待上轮完成再修改 host input。小整数 F32 运算用精确 oracle；不用删除 barrier 后偶然正确证明无须同步。
- 验证只运行上述原型、docs/closure/diff。无 P1、无产品构建、无故意 GPU hang/device-loss 注入。
- 停止：报告架构能否定案；仍影响共同合同的未知量逐条点名。未完成的正式交付/性能不冒充架构问题，
  也不签产品通过。Windows 是原型来源，远程只运行对应完整 Git archive。

## 研究结果

本节在完成预声明研究后填写。当前不代表新架构已定案或产品通过。
