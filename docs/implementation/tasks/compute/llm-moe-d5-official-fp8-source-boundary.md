# D5 官方 FP8：现有拒绝与成熟路径的边界

本页是[当前 D5](llm-moe-d5-real-fragment-acceptance.md)对四个官方画像剩余域的有界源码复核。
baseline `6c27e592`，没有新增产品 FIX、运行 PASS 或完整画像裁决。只读固定版本，不下载权重，
不重新审查已通过的 GGUF 画像。模型角色仍由现有官方 header 事实约束。

## 不能把本机 cuBLASLt 拒绝扩大为上游不存在

固定 SGLang `06992c92ad4494fe9f9636e75be3e35040512447` 有直接反例：

| 源码 | 已证明的事实 | 尚未证明 |
|---|---|---|
| `python/sglang/srt/layers/quantization/fp8_utils.py::_dispatch_auto_backend` | 非新架构专用路径可回退到 `triton_w8a8_block_fp8_linear` | HYLE Windows C++ 三交付可直接消费该回退 |
| 同文件 `triton_w8a8_block_fp8_linear` | 浮点输入先经成熟 per-token group128 量化，再接 block FP8 GEMM；也支持已有 FP8 输入及 scale | 当前 HYLE 已绑定该量化/GEMM 链 |
| `python/sglang/kernels/ops/quantization/fp8_kernel.py::_w8a8_block_fp8_matmul` | 原版 Triton 点积按 K block 应用激活/权重 scale，F32累加 | SM89 上该具体程序的编译、数值、资源和完整成本资格 |
| 同文件 `prepare_block_fp8_matmul_inputs` | scale 使用 `ceil(N/128)`、`ceil(K/128)`；不是一概要求完整 N block | 当前 HYLE `float8_weight_description` 的 N%128 限制覆盖 GLM 的 N576角色 |
| `python/sglang/kernels/ops/gemm/fp8_blockwise_gemm.py` | 独立 CUDA/CUTLASS JIT wrapper 明确限定 SM120 | 可把这个 wrapper 当 SM89实现；该结论为否 |
| `python/sglang/kernels/ops/quantization/per_token_group_quant.py` | 原版量化 wrapper支持 BF16/F16/F32 输入、group128及F32 scale | 它已具有HYLE独立C++绑定；当前没有 |

上述文件位于 Windows 已登记 SGLang checkout；检查时 HEAD 与固定 pin 相同。
源文件内容 SHA256 见[复核收据](../../../../evidence/research/llm-moe-d5-official-fp8/source-boundary.json)。
Triton 源码存在不等于可直接连接当前原生 CUDA target；不能靠 Python 模型引擎执行冒充普通
C++ Direct/Graph。当前未新增 Triton 编译/装载接缝，也未把缺少接缝写成数学缺失。

## 四路线当前边界

| 路线 | 可复用的证据 | 本轮不能签的内容 |
|---|---|---|
| CPU | oneDNN 3.13.2 在13900K拒绝具名FP8 MatMul；普通入口16个签名拒绝 | 所有成熟CPU组合都不可能；reorder不是完整MatMul |
| CUDA | CUDA13.3/SM89 cuBLASLt block128拒绝；普通入口16个签名拒绝 | SGLang原版Triton候选不存在或已执行失败 |
| Apple | MLX0.32.2既有压缩入口的group32模式与block128不同；普通入口16签名拒绝 | MLX没有FP8转换原语：其`ops.h`实际公开`from_fp8/to_fp8`，但完整block128链尚未绑定/验收 |
| Vulkan | 原版IREE预编码FP8 dense八签名×M1/M3两入口通过 | 动态激活量化、grouped专家、partial block及完整官方MTP已通过 |

旧运行证据见[FP8 dense批次](../../../../evidence/correctness/llm-moe-d5-real-fragments/official-fp8-dense-batch/README.md)。
DeepSeek的UE8M0权重scale可无损扩为F32，但不改变128×128分组或替代激活量化。
当前IREE程序自身有片段级F32临时权重/激活成本；不能把它标成无解码、无workspace执行。

## 后续裁决必须保持的范围

剩余工作是成熟程序的接缝和资格，不授权HYLE新写FP8数学。完整官方画像暂不增加通过格，
也不以BF16/GGUF的通过或某一家backend拒绝为全部Provider的终局结论。
若需新编译/runtime Provider，遵守D5的停止条件，先形成具名后续任务；没有资格证据时
保留未验状态。已有CPU/CUDA/Apple/Vulkan GGUF批次、A5交付和A6性能继续独立推进。
