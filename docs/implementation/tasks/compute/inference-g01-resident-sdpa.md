# G01 驻留 SDPA：视图消费与原生 fused 路径

状态：AUDITED，2026-10-13。owner：`compute_provider_ggml`、
`compute_provider_ggml_vulkan`；baseline：`070609442806ebfe6a0dc8d6594b7388c494f20b`。
授权与完整目标见[缺口执行总卡](../project/inference-gaps-goal-execution-2026-10.md)。
本卡先修 G01 中已有 SDPA 的实际消费和成本缺陷，不签整个 G01，也不替代多请求分页及
Apple/Vulkan 联合分页的后续 owner 卡。

## 真实触发与消费域

单请求 cache 按容量驻留，decode 只改变有效长度；公开 TensorDesc 已能表示字节 strides，
CPU ggml 的 tensor 导入也设置 native nb，但 SDPA qualification 逐个拒绝非 dense Q/K/V。
因此 head-first 的容量 cache 截短 seq_len 后仍须重排/复制；不能由“存在 view 类型”判断
该旅程完整。主消费者为 Qwen3.5-2B full-attention 片段，GQA8/2 D256，prefill 和单 token
decode；留出为两个 batch 的非均匀 padding、两种 attention layout、不同 Q/KV 长度和
独立 slot/旧 output。权限仍用原 ready ReadLease，跨页与 arbitrary page-table 另验。

Vulkan 当前这条 SDPA lower 显式生成 F32 scores/softmax/MatMul，即使锁定 ggml Vulkan 已
有 FLASH_ATTN_EXT。这是同数学同 owner 的承载差距，不由 benchmark 排名触发。原生核
要求 Q/output F32、mask F16、K/V 支持编码、D 为8倍数及实际 subgroup/cooperative 资格；
不能直接用“上游支持”代签本机资格。CPU 已有部分 F32/F16 fused 档，Apple 已借 MLX fast
SDPA，均保其已有实现和限制；本卡不新增第三方库或自写 attention 数学。

固定参考仍为总卡的 SGLang/vLLM/llama.cpp SHA。实际 vLLM CPU 核从 block_table 直接
读取，显式 query/cache strides、ISA dispatch 和 OpenMP workers，说明输入形状和执行预算
必须同时核对；不把整个 vLLM/Torch 运行时作为本轮借用对象。当前锁定 ggml CPU 原核
要求 depth 连续、接受外层 nb；CPU qualifier 必须与实际 lowering/allocator 一起反证。

## 本轮边界与审计输出

职责不变：原 Provider 拥有 native 支持域、lower、workspace/ports、错误与 actual retirement；
Graph/Compute 原 executor、lease 和 Queue 不改。公开 ScaledDotProductAttention/布局不新增
模型参数或兼容签名。SCOPED提交`f65616d2b`保留；下表固定后才允许产品改动。

有限审计：CPU contiguous-depth、非重叠正 strides 的 rank4 view 能否沿 native 导入和 fused
分支消费；BF16 是否原核直接支持、原 causal mask 与 query cast 的费用；Vulkan fused
替换时 mask 常量/上传、GQA、输出顺序、dtype 和设备 capability 的全链资格。若需改变
公开 shape、资源身份或依赖方向，先回本卡定界，不在 qualifier 中偷偷接受未证输入。
有限 FIX 已按实际源码固定如下；其他能力不得顺手扩张。

| FIX | 最终形状与承载 | 验收与负空间 |
|---|---|---|
| RS-01 | CPU原SDPA接受rank4 plain、depth连续、外层正的row-major padding strides；原native输入已设置nb，不再强制全dense。原同dtype F32/F16/BF16、两layout、GQA和有效长度语义不变 | 容量cache截短，padding哨兵不参与数学；拒绝depth跳跃/负或重叠strides，不公开whole paged arena |
| RS-02 | 借锁定CPU ggml flash原核直接读K/V，F16/BF16域覆盖causal/noncausal，F32保现有锁资格。query转换允许O(QHD)，不转换/展开完整KV；causal F16 mask按Q×K显式计费 | 原核depth连续、BF16原vec-dot/to-float资格已在实际锁源码核对；两layout、不同Dqk/Dv和保留原小shape域用oracle反证；不得仅靠if条件称 fused 成功 |
| RS-03 | Vulkan仅对同F32/F16、BSHD且Dqk/Dv均8倍数的SDPA接原FLASH_ATTN_EXT；Q转F32、K/V保原编码，返回原output顺序/类型。causal mask以原native Fill→DiagMask→Cast F16生成，非causal无mask；实际device supports_op仍冷拒绝无原生资格设备 | Qwen GQA8/2 D256与留出D64/128；保原非8倍数的显式传统lower域，不能CPUfallback或将不同dtype性能相加。mask和query cast计费，避免Hq×Q×K的scores及按Q/H展开KV |
| RS-04 | Vulkan原typed region在SDPA有实际需求的rank4正row-major padding域接native nb；imports/端口范围仍按ggml_nbytes校验并保原owner。其他operation原qualifier不放宽，非dense输入若被未资格节点读取必须拒绝 | TensorDesc required_bytes/原VkBuffer范围、非零offset、padding、源只读与旧输出寿命；native graph库支持检查不绕过 |

source/static/shared仍原compute/graph产品选择，没有新factor/第三方/FFI/public类型。普通调用
仍`ScaledDotProductAttention`加TensorDesc；新view支持只为原合法数学提供自然消费，descriptor
本身不授权页capacity。冷graph/slot身份不改，只有原Provider定向资格和lower；warm native
submit仍原Queue和lease use，OOM/未支持/失败由原Result/Status路径回滚，借用和关闭合同不变。

具体审计锚点：CPU `create_tensor_input`原nb导入、`supports`1494、lower821、cold mask1646；
实际锁CPU `ggml-cpu/ops.cpp` flash的depth连续断言及K vec-dot/V to-float、BF16 traits；
Vulkan `tensor_domain`/`native_tensor`目前dense-only且未设置nb、SDPA lower857、
`native_storage_impl.h`原device supports_op/owner/range拒绝；锁Vulkan flash19420实际dtype/
D/device检查。Vulkan新增native stride导入属于RS-04，不能误称已有端到端能力。

不纳入任意负/重叠 strides、任意 attention 特例、quant KV、新 scheduler、量化格式扩展、
原生分页或全引擎吞吐。这些当期 G 项仍保总卡工作，不因本卡有限收口删除。
直接依赖/消费者沿中央 component DAG 和 change-impact selector，不新增重复名单。

## 预声明验证、成本与停止

先精确复现原 view 拒绝，再以同 oracle 比较有效 KV 与 padding；包含短 decode、长 prefill、
两布局、GQA、尾部/无效区、独立 slots、保旧输出、失败/OOM及原 cancel/close/retirement。
模块 tests/逐头与真实直接消费者按 selector；CPU 实际 Windows/WSL，Vulkan 实际 Windows，
Apple 只在影响闭包适用时用 MacBook Air，不用 CPU fallback 填 GPU 格。公开消费者要跑
source、relocated static/shared，复用现有树/锁/缓存；适用宿主 ASan/UBSan/TSan，GPU/driver
私有内存与既有外库限制仍明确。文档/closure checker 每轮照常，不重写公共工具。

P0 先核 input copy、Q/KV cast、mask、scores、scratch、native private UNKNOWN 与成本阶。
冻结 candidate 后针对新增 fused/view 档做一次同合同定向 P1：自然 native ggml flash 为 peer，
实际 admission→completion→output owner return，短 decode/长 prefill 和一组留出；A/A 与双
顺序先验噪声，预声明非劣 margin 10%，不以阈值把稳定小幅 LOSS 改成 parity。无法证明
同合同的格如实 INCONCLUSIVE，不借不同 dtype/位置结果签性能。原参考档不无故重测。

调查检查点2小时、实现检查点4小时、候选验证检查点2小时；按用户常设授权正常推进时
自动延长并保原日志、FAIL/TIMEOUT和累计 elapsed，普通编译12 jobs按内存调整，Air10。
有限缺陷已裁决且全部适用门通过才 CLOSED；真实外部阻断按协议记录精确条件，不靠换
日志目录重置，不把“继续调研”当终局。七问/固定 FIX/最终资格集中追加本卡与一份证据。
