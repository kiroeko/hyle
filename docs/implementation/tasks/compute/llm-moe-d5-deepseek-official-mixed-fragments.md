# D5-F43：DeepSeek 官方完整混合片段

<!-- MODULE REVIEW: AUDITED -->

- baseline：`a33823074f56b3f1bbc84bff1e173df0487ff262`；owner 为 `compute_ops` 和
  `compute_provider_cuda`。继承 D5 的144/176格及F38/F41/F42已验能力，其他owner不重开。
- trigger：原子FP4投影、FP8投影和group64/32 roundtrip均不能证明官方DeepSeek完整片段。
  固定原版 `linear` 在FP4专家前执行group128 FP8激活量化；`Expert` 在down投影前乘路由权重。
  官方attention/compressor/MTP还有BF16、F32与冷解码的明确边界，GGUF消费者不能直接代签。
- 来源：F42所固定的DeepSeek V4 Pro官方revision
  `b5968e9190ef611bbf34a7229255be88a0e937c1` 的 `inference/model.py` 和 `kernel.py`；
  逐权重事实为 `tests/compute_models/d5_deepseek_remaining_facts.json`。两模型独立形状保留。
- 主纵切：官方BF16输入→真实FP8/BF16/FP4混合投影与激活边界→最终输出/实际状态反馈。
  正交纵切：score/hash路由、变化输入/选择专家、压缩比例4/128、cache环绕/reset与完整MTP/head。

## 有限 FIX（实现前已冻结）

1. 原版 `act_quant_kernel` 再固定一次group128 AOT；复用F42的公开roundtrip语义和已有
   DeepSeek Provider，只增加明确格式，不新增数学实现、provider种类或动态编译。
   group64/32程序保持身份；group128的caller workspace同为4×组数，shared为8192字节。
2. 共享官方FP8投影测试消费者增加DeepSeek幂次scale模式，既有GLM普通scale默认不变。
   FP4专家采用已有原版ggml packed执行，保留FP8激活量化、BF16投影输出、F32激活/路由权重、
   BF16 down输入和最终BF16累加输出；共享专家使用原版动态FP8。参考独立解码官方权重字节。
3. 官方attention/compressor消费者按原版位置保留BF16边界、group64 KV与group32 indexer
   量化还原，wo_a由F41原版程序冷解码并真实驻留；接回实际状态，禁止上传expected状态。
4. mHC/MTP接原版双投影相加、完整attention/MoE、head折叠与129280词表，不能套用GLM concat。
   所有必要region齐备且普通Direct/Graph通过才更新官方CUDA画像格；局部通过不增加格数。

## 成本、资源与验证预算

组量化一次launch，O(elements)，无私有allocator或host同步；官方FP4组合的额外ggml激活
量化及F32运输成本必须如实保留，不能因为权重格式无损就宣称整链精度/性能无损。
group128使用原F42 caller workspace/Completion生命周期，不更改共享执行器或热期路由。
FP8权重解码只在冷期，prepared常量实际进入热期；完整片段的转换/融合损失一并计入A6。

预算：按上述有限消费者批量接线后一次CUDA build/test批次，两个普通入口使用同一组case；
共享语义/Provider资源因新增格式执行适用检查。原版AOT来源和重建可独立执行，不扩充性能实验。
出现具体失败才定向修正；不反复换容差或自研替代核。完整官方CUDA通过后进入其余路线，
全局三交付/剩余selected gates及P1仍按D5-A5/A6，D6汇总；未运行项不写PASS。

## 当前状态

`4b300218`固定SCOPED文档，当前两个AUDITED owner重新绑定本卡；不制造非法状态回退。
原版kernel支持block_size128，模型明确在FP4 linear前调用；F42的独立组flatten与workspace ABI
可复用，源函数不改。GGUF FP4权重lossless repack及投影已验，但其激活近似必须由整链原门裁决。
上述FIX已冻结，可批量实施；group128及完整链路尚未运行，未新增完整画像资格。
