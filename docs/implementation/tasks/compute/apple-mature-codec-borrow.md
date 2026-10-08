# Apple 成熟 codec：IQ1 三类型与精确 Q4_K row decode

<!-- MODULE REVIEW: SCOPED -->

baseline `ce7f52f6`，Windows canonical。父轮是
[Apple parity](apple-cpu-cuda-parity-closure.md)。旧runtime
[CUDA terminal](cuda-final-u6-audit.md)保留，不解除CPU/CUDA/Vulkan资格或旧失败。

## 具名 reopen 与 owner

`compute_ggml_runtime` CLOSED→SCOPED的触发事实：当前共同目录8个真实IQ1_XS/XXS/XXXS
expert合同已有Anemll原Metal程序，但Apple-only旧core没有三个enum/block/host codec，无法
连接原程序。不是重审全库。已有Apple/Graph/ggmlMetal三个AUDITED owner继续父轮有限FIX。
此卡只为runtime的Apple条件源码选择、exact manifests与原3类型准备重新绑定。

成熟来源固定：Anemll `a809594fd2035979ad2b7ef23398127c1d07e891`（MIT）包含三个
matrix-vector数学body、grouped wrappers和GPU grids；原CPU/core类型沿已经登记的Unsloth
`46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f`。接入前核对原grids/block offsets/nibbles/
delta/scales与CPU/CUDA原codec，不凭README或类型枚举签资格。只移植成熟代码，不写HYLE
IQ1 kernel，不移植Flash-MoE engine、文件加载、slot cache或模型调度。

另一真实缺陷是父轮五个Q4_K embedding精确FAIL：Metal先做half高nibble除16导致scale
精度损失，原CPU先widen至F32。优先借用原ggml CPU `dequantize_row_q4_K`的完整运算序列，
由MLX公开GPU tensor ops对**所选行**承载，保留原bytes和精确scale/offset，不新增算法/shader，
不放宽输出门；成熟source/API充分审计后才写该接线。整个table不在热路径解成浮点副本。

## 有限审计/实现边界

原3类型/代码表/算法不改；只允许旧core ABI大小trait适配、原buffer-offset已由encoder绑定的
args适配、original template实例/分派、route-only source条件和版本/license/source/static/shared
登记。不更改CPU/CUDA数学或原选择条件，不制造第二套ggml runtime。公共HYLE schema不变。
IQ1优先原MV/grouped完整程序；缺原MM者保持合法MV而不去调用不存在的MM。准备期绑定，执行
无搜索、测时或新增provider仲裁。MLX所选行准备由MLX保活，既有completion/checked status保留。

source审计每具名问题一次follow-up；固定AUDITED FIX后再改产品。批量接线集中一次原body/
编码golden/8实际专家合同/5 Q4_K失败项/变输入与ID/资源/两入口验证。具名新失败仅一次纠正。
其它通过域不重跑全集/P1；API/source稳定后在父轮A3/A4做相应成本和交付。运行Air build2/test1，
本机不并发编译，复用缓存、完整Git候选正向同步。无成熟同合同资格则保留具体缺口并停止微调。
