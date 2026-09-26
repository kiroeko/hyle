# D5：DeepSeek 剩余真实片段审计

这是当前 [D5 A2/A3](llm-moe-d5-real-fragment-acceptance.md) 的输入审计，**没有新增执行通过格**，
也不授权自研数学。baseline `7af6eae4`；固定官方 Pro/Flash 和 Flash 两个 GGUF，
不把另外的 Pro-0813 制品加入本轮。下一批接线须复用此处事实，不能再从档位名称推断整模型 dtype。

## 精确权重边界

[逐角色事实](../../../../tests/compute_models/d5_deepseek_remaining_facts.json)从已经保存在 Windows
的完整官方/GGUF header 提取，记录原文件 SHA256、模型 revision、shape、dtype、层集合和数量。
官方各归并为101种角色签名，Flash GGUF 各44种；本次没有下载大权重，也没有重做全模型研究。

| 固定制品 | 主干形状 | 主要存储 | MTP 事实 |
|---|---|---|---|
| DeepSeek V4 Pro 官方 | H7168；128×512 heads；Q-rank1536；16个 output groups×1024；E384/top6 | routed FP4/group32；dense/shared FP8+UE8M0 block128×128；norm/router BF16；mHC/sink F32 | 独立 `mtp.0`，含 FP8 `e_proj/h_proj`、attention/shared 与 FP4 experts，以及自身 mHC |
| DeepSeek V4 Flash 官方 | H4096；64×512 heads；Q-rank1024；8个 output groups×1024；E256/top6 | 同上，按各 tensor 角色取格式 | 同上，不能用普通 GLM NextN 的双 norm/Concat/EH 程序代签 |
| Flash UD-Q4_K_XL | 同 Flash 主干；43层；词表129280 | attention/compressor/shared Q8_0；routed MXFP4；router BF16；norm/mHC/sink F32 | 全部 header 只有 `blk.0..42`，没有 NextN tensor |
| Flash UD-Q8_K_XL | 同 Flash 主干 | attention/compressor/shared BF16；routed 仍为 MXFP4；norm/mHC/sink F32 | 同上 |

两份 GGUF 的前三层都有 I32 `[129280,6]` token→expert 表；其余层使用 score+bias 选择。
压缩倍率4的21层采用双窗口投影，倍率128的20层采用单窗口投影；另两层没有压缩器。
这些角色不能由此前仅有 score TopK 的 MoE 或无 sink 的普通 attention 代签。

固定首分片的GGUF metadata现已补入同一事实文件，两个量化画像均为tail64、raw RoPE
base10000、compressed base160000、YaRN factor16/original65536/beta32/1、窗口128，
不是从文件名推测。此次只请求header，总响应10,513,216字节。固定成熟cache planner对
压缩结果使用`source_start = pos + 1 - ratio`作为RoPE位置，APE则逐token取`pos % ratio`；
共享consumer保留这两个不同的索引。metadata读取不是正确性验收，不增加运行通过格。

GGUF 缺少 NextN 是该固定制品的适用性边界，不是库缺少 MTP 的证据。后续纠正对应两个画像的
R13 适用性时必须引用完整 header；官方两画像的 MTP 要求仍然保留。当前 ledger 尚未因此修改。

## 已核对的成熟程序与接缝

- llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` 的 `src/models/deepseek4.cpp`：
  原版压缩器、按 token 查表的 MoE、mHC、sink attention 和 NextN 程序。GGUF loader 对缺少
  NextN EH 权重的制品把 NextN 层数置零，与实际 header 一致。
- SGLang `06992c92ad4494fe9f9636e75be3e35040512447` 的 `hash_topk.cuh`：已有
  `moe_hash_topk_fused`，采用稳定 `sqrt(softplus)`、I64 token ID、I32 查表和
  `sum+1e-20` 归一化。文件身份及 Python consumer 保存于上述事实文件。
- 当前 HYLE `model_ops::SqrtSoftplusRouting` 只暴露 score TopK，不接收 token ID/查表。
  CUDA 的普通 Softplus 又不在本轮成熟准入内；因此不能通过复用旧 score TopK 宣称 hash 层已支持。
  下一步应审计并接入登记版本的原版 hash kernel，明确原生归一化、scale 和索引检查；不能自写替代核。
- 当前 `ScaledDotProductAttention` 没有 sink 端口。原版 ggml 提供
  `ggml_flash_attn_ext_add_sinks` / `ggml_soft_max_add_sinks`；应先确定能保留成熟执行程序的
  语义与资源接缝，再绑定四路线，不能为了测试通过忽略 learned sink。
- attention 还需要尾部64维 RoPE/YaRN、压缩器与 indexer 的实际历史状态、CSA/HCA 选择，
  以及 grouped output projection。它们是下一共用批次的真实组合需求，不是新算子目录扩充。

本页只固定事实和下一批边界。产品改动前仍须把具名接缝加入当前 owner 的有限 AUDITED FIX；
接齐四路线共享消费者后再冻结候选、串行验证。官方 FP8 动态激活量化与 grouped storage 的已有
未验边界继续保留，不能以 BF16/GGUF 改写官方画像。三交付和性能仍归 D5-A5/A6。
