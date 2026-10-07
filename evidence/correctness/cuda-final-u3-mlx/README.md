# CUDA U3 N2：原 MLX packed GPU 资格

任务：[N2当前卡](../../../docs/implementation/tasks/compute/cuda-final-u3-mlx-carrier.md)。
Windows canonical；WSL Ubuntu仅前向验证副本。实际GPU为RTX4070，Toolkit13.3.73；
原locked Torch2.8.0 CPU common＋Torch2.8.0+cu129 leaf，未替换CPU实现或增加SDK。

## 已实现

原MLX `2b2bb497` 的affine／MX4／MX8公式由原CUDA tensor库执行，保留F32 accumulation、
source scale/bias dtype、MX group-dot后缩放顺序与原E8M0。只展开当前K group的选中experts，
embedding只展开选中rows；没有全float expert缓存、新HYLE数学或CPU推理fallback。
shifts／FP4 LUT在cold prepare上传一次；execution使用固定shape／program和原external stream。
checked ids的原GPU reduction／D2H同步完整计费；失败output不可消费，result活至Completion。
NoTF32Guard保留F32合同并还原调用方线程状态，不篡改用户global TF32设置。

原CUDA BF16 QMM会把dequant权重先舍入为BF16，当前完整公式没有这个舍入。
[源审计及反例](source-audit.json)只证明源码方程不等价，不是GPU fusion资格或性能输赢；
原CUDA融合候选没有被假报为已接入。当前使用原precision-preserving tensor公式。

## 实际验证

- 固定16个真实contract的Direct／Graph各16项通过；完整N／K、packed字节与原mixed identity保留，
  grouped驻留最多8个实际expert，重复输入／selected ids／失败恢复由共同fixture验证。
- Windows产品候选`2f9ec129`上述32项及N1生命周期／metadata OOM／FFN回归通过。
  `f4c95ca0`最终precision测试9项、原OOM／恢复2项通过；变更只在test oracle，原数学源相同。
- Linux `1749e731`：57 GTests、6 CTests（含三个公开头）全部通过，无skip／零GTest过滤。
  组成是16 Direct＋16 Graph＋12 N1生命周期＋metadata OOM1＋FFN1＋precision9＋原OOM2。
- precision覆盖F16／BF16／F32、BF16 cancellation反例、外部TF32 policy及MX scale codes
  0／143／255；18个source dtype边界case包含subnormal、F16 overflow与Inf，未改产品误差合同。
- Linux回归揭示FlashInfer宏止于route PRIVATE边，修复actual capability的literal转发；
  Windows实际生成的两个consumer compile definitions验证只有Torch CUDA、没有FlashInfer，
  默认FFN回归通过。absent factor不造能力，源码／SDK最终一致性仍随联合delivery验收。
- 全Compute+Graph非测试实现 **60,612行／253文件**，统一脚本自检通过。

## 原失败与修正

[原始归档](native-batch.zip)与[候选／范围摘要](native-receipt.json)保留首次configure／build及失败。
新vendor首次用了archive展开hash而非in-tree文本归一化hash，已按原项目算法修正；
schema的`custom_as`返回optional，绑定接口已纠正；没有改治理工具或CPU源码。

测试初版把F32消去后的结果要求逐字节相同，原合同允许F32 reduction association。
现仅F32按两个源码term的8×epsilon误差界核对，BF16关键反例仍严格；MX负零参考也按原正零
accumulator初始化修正。这些是test oracle修正，不改变公共误差门或产品数值。

## 尚未完成

N2/source-static-shared与CUDA-only／mixed最终SDK交付，和N1／R1联合U3候选一次集中验收；
必须在U2／U3关闭前完成。所有其它模型region、全部GGUF、unknown默认／best-of／自研清退及U6
仍未由本32项代签。本批P0／单成熟carrier，不跑P1或宣称SOTA；owner保持AUDITED，goal active。
