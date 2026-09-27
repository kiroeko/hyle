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

### 第一组实现：两模型完整MoE与激活边界

原版group128已在SM89离线生成，SHA256
`e1f7c7314e43676fbfca104feedfd5b541d95c7039615f1316a46a700484b970`；独立native的12案例
覆盖全部有限BF16编码、midpoint、tiny/zero、尾组和guards，输出与scales精确通过。
原版有限极值的量化还原可溢出，保留F42已声明边界。该结果仍不签普通入口。

Pro/Flash的score/hash四种完整MoE消费者与group128共享语义/资源测试已批量接线，统一执行
一次CUDA增量build/test；shared FP8使用源UE8M0无损扩宽scale，FP4参考不借用ggml中间量。
attention/compressor/MTP消费者继续按FIX3/4接合后统一下一组验证；不逐原子重建。
当前完整画像仍144/176，尚未验收MoE、三交付或P1。

### 当前裁决：原子通过，ggml完整MoE候选FAIL，返回AUDITED

`a4b70bc2`的group128普通入口/资源、三份AOT重建、12个新增native案例和4项独立头通过；
两入口的完整MoE均未通过既有数值门，最终输出归一化误差最高约7.13%，原门4%。
不提高容差或换输入；初次诊断View绑定问题修成Copy后数值失败仍在，完整ledger仍144/176。
见[完整检查点](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-official-moe-checkpoint/README.md)。

本批ggml MoE候选停止重复试验，owner继续AUDITED。其CUDA MXFP4点积还有Q8_1激活转换；
独立有界检查已确认官方原版 `fp4_gemm_kernel` 可在SM89执行，N4096/K2048的6个native案例
逐BF16精确一致。下一轮先固定该原版程序的动态专家接线、格式与成本，再改产品；不扩成自研。
原子资格不受消费者失败撤销；其它shape、普通专家入口和完整画像未验。A5/A6/D6继续未决。

原版专家接线按[F44有限范围](llm-moe-d5-original-fp4-expert-binding.md)推进，数学主体保持官方，
只增加设备端索引寻址/验证与已有资源接合。

F44在`bc72243d`完成原版接线：两种官方模型的score/hash完整MoE在CUDA Direct/Graph
通过原门，资源与四形状专家投影同时通过，见
[F44收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-original-fp4-entry/README.md)。
FIX2的ggml执行腿由该具名原版替代；完整attention/compressor/MTP按FIX3/4继续批量接合。
完整ledger仍144/176，A5/A6/D6未完成，不再把局部推进隐藏在固定百分比中。

### 官方CUDA完整画像已验

`9e846d87`将FIX3/4一次接入两入口：官方F32 compressor、BF16/FP8/FP4激活边界、真实TopK
驱动的完整attention、原版冷解码wo_a、ratio0 MTP的双投影/mHC/attention/MoE，以及完整
129280行F32 logits。两入口六组均通过，实际状态连续回送并reset。见
[完整收据](../../../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-official-cuda-complete/README.md)。
初次构建ADL歧义已限定命名空间；初次attention继承compressor时丢失整链精度分类，恢复预先
固定quantized门，独立compressor门不变。官方RoPE改为源程序power/reciprocal/YaRN计算。
未更改输入以过测，旧失败保留。只转换fixture非零BF16权重项，保持上传字节与完整维度，
避免对海量零值做重复host转换；最终构建13.70秒、测试42.59秒。

本卡消费者范围已验，结合F44及不变的资源/R01 Gather资格，新增4格到148/176；CUDA22画像
44格全部完成正确性。其它路线28格、三交付/P1和D6仍未完成，不把本卡结果当全goal完成。
统一Compute+Graph实现60,150行/240文件，不含测试和第三方程序。
