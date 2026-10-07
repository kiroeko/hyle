# CPU 热门量化与MTP计算覆盖

普通入口为 `compute::cpu::create_context()` 和 `graph::cpu::create_context()`。
本页说明Windows/Linux x86_64和macOS ARM64 **CPU-only SDK** 的固定计算合同覆盖，不提供模型loader、tokenizer或采样器。
资格由原格式端口、参数及相同边界的完整semantic/state程序共同证明，
`LIBRARY_CONTRACT_PASS`不能读作104次整模型实权重推理通过。

当前三平台成熟承载和source/static/shared计算/状态/精度资格见
[CPU最终F3证据](../../evidence/correctness/cpu-final-f3/README.md)。原Windows/Linux联合合同、当前升级候选和
Mac首次资格分别按真实输入/软件身份对账；默认规则仍有工程默认与已测优选的区别。
当前版本的有限全候选优选与工程默认已完成F4验收；F5已物理清退14种冗余自研数学。
删除后的三平台／mixed差异验收与[CPU最终终审](../implementation/tasks/compute/cpu-final-f6-final-qualification.md)
已完成；固定目标集合的CPU计算合同可依赖。默认选择仍区分具名工程默认与已测条件下的统计优选。

## 固定制品与范围

| 模型族 | 固定GGUF仓库 | 档位 | CPU计算合同 |
|---|---|---:|---|
| Qwen3.5-2B | [f6d5376](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | 22 | 全实际matrix/Gather与Qwen norm/GQA/GDN/state |
| Qwen3.8-27B | [4ca7207](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | 25 | 真实内宽、原编码、GQA/GDN及MTP边界 |
| Qwen3.8-2.4T-A95B | [567d3e6](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | 9 | GQA/GDN、router/selected experts/shared、完整NextN数学 |
| Kimi K3 | [a083636](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | 9 | KDA、MLA、latent MoE、SiTU及AttnRes |
| GLM-5.3 | [346b359](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | 12 | MLA/DSA/index state、MoE及NextN |
| GLM-5.3-Flash | [621d456](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | 12 | KDA、pooled DSA/ragged tail、mHC、MoE与NextN |
| DeepSeek V4 Pro-0813 | [固定C1仓库/档位表](../../evidence/correctness/cpu-quant-mtp-c1-demand/actual-storage-matrix.json) | 2 | 新ratio0/4/128压缩/attention/indexer、普通/hash MoE、mHC |
| DeepSeek V4 Flash-0731 | [固定C1仓库/档位表](../../evidence/correctness/cpu-quant-mtp-c1-demand/actual-storage-matrix.json) | 13 | 同模型实际原编码、layout/state与完整片段 |

另覆盖两份新版DeepSeek官方混合FP4/FP8/BF16制品及完整三stage DSpark数学，
Qwen27B命名Q4_0的实际Q3/Q4/Q6 MTP sidecar、Flash Q8/BF16两份实际MXFP4混合DSpark sidecar。
四非GGUF制品为nested NF4、AWQ GEMM int4、compressed-tensors symmetric int4、MLX affine4/MX4/MX8，
准确revision/真实端口/数学与限制见[联合资格](../../evidence/correctness/cpu-quant-mtp-c5-required/joint-verdict.md)。
文件整体Q标签不是tensor dtype，也不保证任意其它GPTQ/g_idx/packing版本可以继承。

## 用户怎样接入

1. 选择 `compute-cpu` preset，或 `HYLE_COMPONENTS="compute;graph"`，其它路线保持OFF。
   CPU依赖由HYLE默认AUTO组装；只选择路线，无需排序Provider或传timing profile。
2. 消费方核对实际文件并解析tensor/scale/codebook/zero、原dtype/block/group和shape。
   按[Compute指南](compute.md#ggufggml-原生压缩权重)与[非GGUF接口](compute.md#cpu-非-gguf-原格式与压缩专家)
   提供checked TensorDesc。HYLE不猜“4-bit”布局或自动读取模型文件。
3. 已知部署身份时提供不可变ModelArtifactProfile/WorkloadContext；它只辅助冷规则匹配，
   每个端口的真实存储与精度仍由本次descriptor证明。未知身份仅能用合法通用路径。
4. Direct准备并复用Invocation；Graph编译并准备独立执行槽。重复调用更新activation/cache/ids，
   等真实Completion完成后读输出；借用权重活到最后一次完成。
5. source/static/shared均为C++交付，安装消费用find_package；调用 `hyle_stage_runtime_dependencies`。
   Windows将依赖DLL邻置exe，Linux/macOS保留实际loader路径。HYLE static仍含外部shared CPU runtime资产。

## 需要保留的成本与限制

- CPU-only锁定ggml0.26＋原Unsloth IQ1扩展；mixed SDK保留原GPU core，并在同core接入原CPU格式端口。
  当前Windows CPU+CUDA／Vulkan、Linux CPU+CUDA和Mac CPU+Apple组合的CPU三交付已完成F1／F4验收；
  F5清退后的CPU差异也已通过。这不增加GPU格式支持。
  具体边界见[CPU最终F1证据](../../evidence/correctness/cpu-final-f1/README.md)。同进程不要混用未隔离的SDK原库对象。
- 原quantized dot的activation转换、累加及输出storage都属于合同。NF4/AWQ/CT显式cold解码后缓存，
  F16原权重必要cold转换真实计费；MLX只current K group/selected experts，不能每轮全库展开。
- managed原库有临时、copy、可能私有线程，私有上界unknown，capture/prepared-only/严格内存域可明确拒绝。
  新F16/BF16卷积原ATen接缝与原F32 ggml路径分开，见[原text资格](../../evidence/correctness/cpu-quant-mtp-c5-plain/README.md)。
- 失败slot需重建；后续节点失败不回滚此前输出，cache提交/回滚及draft接受策略属于消费方。
- 非本地payload是合法非零数据，保留真实K/N/state与完整V，可减token/context/驻留expert；
  全head以驻留分片验证，text MRoPE三轴position相同。没有整模型常驻或吞吐/SOTA承诺。
- 这份CPU资格不修改[旧四路线账本](model-fragment-coverage.md)中的GPU拒绝/FAIL/P1阻塞，
  不授予新版GPU、视觉或任意新release资格。
