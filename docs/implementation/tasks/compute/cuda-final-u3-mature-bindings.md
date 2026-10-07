# CUDA U3：共同需求的成熟承载批量接线

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- baseline：`ffeb224ed4ff56337f9aae958c5f4dd54daf1803`；Windows canonical，保留用户／后续修改。
- 当前 scope owner：compute_provider_ggml_cuda；Graph region 的既有 Recipe/同源 supports/lowering
  随真实产品变化再绑定 graph_provider_ggml_cuda。runtime／其它成熟载体按相应子批合法定界。
- 总要求：[共同算子／量化目录](../../../architecture/model-compute-catalogue.md)与 [CUDA goal](cuda-final-closure-plan.md)。
  U1 原源已检出；本包不重新推导需求或调查全部模型。
- trigger：124 个当前原 GGUF 计算合同有成熟 CUDA 类型／GET_ROWS 但 HYLE 未接；另8个原 IQ1 扩展、
  34个 non-GGUF 与当前新版完整 branch 尚无本路线资格。不能把它们当成124/34个需新写算子。
- 普通 consumer：原 format/role/shape 的 projection／head／embedding／expert，通过普通 CUDA
  Direct／Graph 冷绑定并重复运行；正交 consumer：typed tensor 批处理、重复／坏 IDs 与失败输出边界。
- scope：原成熟程序／dtype／codec／scale／rounding／资源接线，完整必要 region／state、两入口、三交付。
  每一子批批量修改，稳定后一次集中构建／correctness，三交付复用同候选；不逐 op 重配。
- non-goals：自研 CUDA 数学冻结、无优化挽留、无新数学／模型 loader／CPU fallback；
  不变闭合 CPU 算法、原验证仪器、其它 GPU 路线。原 mature fusion/resource 不拆成逐kernel万能执行器。
- tier：单一成熟 carrier 接线正确性／资源／P0；合格重叠域与完整 region 的有限 P1 交 U4，
  不给没有资格的格式造赢家。三交付不拖到 U6 首次验收。
- 预算：每 carrier 一次原来源／精度／资源接缝审核，一次批量实施／资格化，至多一批具名纠正；
  没有真实成熟承载时具名裁决，不新写数学或无限调查。
- predeclare：普通两入口、原物理格式／precision／shape、bounds／OOM／失败／replay／寿命、
  selected factors/headers／消费者、source/static/shared/mixed/offline/staging、docs／closure／LOC。
  未运行 UNVERIFIED，静态 hook 不签 RUN_PASS。

## 有限承载集

| ID | 共同合同／原成熟来源 | 允许的接缝 | 完成证据 |
|---|---|---|---|
| G1 | 原 stock ggml `456172ec` 已有 Q4_0/1、Q2_K、IQ2_XXS/XS/S、IQ1_S/M，原 GET_ROWS 已有目标8种 quant embedding | 同 core 的 HYLE type mapping、描述符/准入及两入口；不改原算法、CPU ABI或数学/driver资格 | 原 codec 与实际格式矩阵/embedding/专家、变化输入/IDs、错误/三交付 |
| G2 | Unsloth `46cbf0e9` 原 IQ1_XS/XXS/XXXS CUDA convert/MMVQ/MMQ | 借入原三个成熟扩展与必要 traits/codebook/dispatch；先审计与当前同进程 core 及 CPU port 的一致性 | 原编码／A8/F32边界，真实 expert shapes，普通两入口与生命周期／mixed／三交付 |
| N1 | bitsandbytes/AutoAWQ/CT 原 NF4/nested、AWQ、CT GPU准备序列 | 按既有 ArtifactWeightDecode 原端口，GPU-only original carrier，冷准备常量复用；runtime／owner 方案先冻结 | 原 group/zero/scale/codebook/offset、精度、实际模型消费者、两入口／资源／交付 |
| N2 | MLX 原 CUDA affine/MX4/MX8、QQMM／GatherQMM／dequant | 既有 MlxQuantized 端口；原 selected groups/experts/rows 与 F32 accum/原A8要求；不直接拿不同精度 qmm 冒充同合同 | 16原合同、原packed/scale/bias/E8M0、full consumers／资源／三交付 |
| R1 | 八族完整 region、官方混合dtype、新版 DSpark／三sidecar/Qwen本地原版 | 复用共同程序，仅改变执行环境；真实 admission缺口定向修复、原 state/carry/head 保留 | 全branch/状态反馈/完整head/官方精度，普通两入口，不以一条简单MatMul代签 |

U2 新冷事实产品已单独落盘，沿 [U2](cuda-final-u2-build-contract.md) 继续验收；G1开始时将
ggml CUDA 的当前 owner 重定界至本卡，U2字段消费／资格阈值保持并纳入同一稳定候选验证。
G2/N1/N2/R1 在实际方案 AUDITED 前不从 G1 获得实现授权。公共 schema 当前已足够描述需求；
新来源若提出必须改变公开精度／生命周期，先定界，不局部伪兼容。

## 普通 API、资源与冷／热边界

N1 独立范围与资格记录见 [成熟 GPU 权重准备载体](cuda-final-u3-artifact-carrier.md)。

普通用户只选路线，原 checked descriptor、ModelArtifactProfile／WorkloadContext／resource 不变。
成熟图 region 的原 metadata／pool／stream 由 Provider 持有；调用方权重／resource 覆盖最后 Completion。
独立 invocation/scratch；坏 IDs、alias/overflow/unsupported 精度在正常边界处理，失败输出不可消费。
MatMul 保留原量化 activation 与累加/输出；必要 cast/contiguous／upload/download/sync/融合完整计费。
Direct 与 Graph 共用真实 operation supports／lowering；prepare匹配一次，execute 不再搜／计时。

## 七问

版本／许可证取原唯一锁与 U1 逐文件 SHA；真实 tensor/region 共同目录决定能力；资源/完成复用下层；
P0先查cost，U4只同合同竞争；无新FFI；三交付/依赖/staging 同候选；用户自然两入口、无需Provider排序。
全部门在各子批实际验收后签，不因本施工卡存在签 CLOSED。

## G1 子批 AUDITED

SCOPED 卡已独立提交。原 ggml CUDA type/support/convert/mmvq/mmq/GET_ROWS 的已锁源码
由 U1 逐文件检出并固定。本子批只接八个已有 stock 类型、目标八种 quant embedding 格式；
IQ1_XS/XXS/XXXS 仍属 G2，不因 source 存在授予编译／运行资格。
有限 FIX：mapping＋embedding准入；扩展同原 codec 测试格式并迁移旧 blanket decline 为只拒绝 G2；
普通 Direct／Graph 原格式、正／负 IDs、失败输出、变化激活／重放及 full region 对应资格。
精度、rank、storage/strides、SM/runtime/driver数值资格、资源拒绝和原 math/source/core 不改。
Graph Provider 当前 CANDIDATE 依法退回 AUDITED，因为同源 operation 支持集／正常 workload 改变，
不是重新修改原 Graph API。两个 owner 绑定本卡仅 G1，保持 U2 冷事实迁移并在联合稳定候选运行验证。
N1/N2/G2/R1 仍待各自接缝 AUDITED；没有自研扩张授权。

G1 局部接线批次 `45a1d4a8` 的 stock格式／embedding／grouped／Graph replay及既有普通消费者
已通过 Windows 验证，见[U2/U3 批次](../../../../evidence/correctness/cuda-final-u23/README.md)。
这不是当前104制品全部精确数值合同、Linux或三交付的完成结论，G1仍需随完整路线补齐这些适用项。

## G2 子批 SCOPED：原 narrow IQ1 CUDA port

U1/C1固定原Unsloth `46cbf0e9` 的源码已在本地，三个物理格式与当前CPU端口完全相同。
源diff只取对应新增traits、convert、vecdot、MMVQ、MMQ／config／分发、三个原模板实例，以及GPU表和QI/QR定义。
不复制全部0.22 GPU core、不升级当前0.24 GPU pin，不引入AMD/HIP数学。
真实ABI差异只见新版type traits需要 `bs=sizeof(block)`，不是新算法。数学/表/原模板正文保持原SHA。
source audit已生成11个文件的有界context补丁＋3个原实例，无产品修改或运行资格。

scope owner：compute_ggml_runtime（源manifest组合／原程序／license/build identity）、
compute_provider_ggml_cuda（mapping／原prepare资格）与 graph_provider_ggml_cuda（同源region）。
原CPU port和CUDA resource edits需由 owning helper归并为单一旧core manifest，满足现prepare_source的
唯一路径／before/after hash，不扩公共治理。CUDA-only也须取得原物理types／codec，但CPU执行target仍OFF。
三个模板原文件可作为owning target sources加到已有ggml-cuda，唯一include接缝改为同一build-local mmq.cuh，
不为新文件扩prepare_source通用协议，不维护第二源码缓存。

预算一次原源码／ABI账单、一次原port组合和差分自检、一个稳定candidate的Windows/Linux两入口／
原8个expert合同与model内宽／state／三交付／mixed CPU差异。至多一批具名接缝纠正；不扩自研或反复换peer。
未来产品修改前G2独立AUDITED；暂不把本scope当实现授权。

### G2 AUDITED：保持单核 ABI 的原版增量

SCOPED 提交 `e1ed0253`；原diff为新增IQ1函数／分发，两个替换只涉及新增traits及slow-type集合扩容。
冻结有限 FIX：

1. `third_party/ggml_cuda_runtime` 增原IQ1有界context manifest／来源SHA、三个原template实例及NOTICE；
   只借 CUDA/通用NVIDIA配置，AMD/HIP文件不引入。template唯一include路径接缝适配，数学正文不改。
2. owning runtime 的私有 manifest composer归并原mixed CPU types/codec、原CUDA resource与IQ1补丁，
   一文件一次before/after校验；现 `prepare_source.py` 与公共治理不改。新增template通过target_sources编入原ggml-cuda。
3. CUDA-only也借已资格化的原physical types/codecs（CPU target仍OFF）；CPU-only0.26与Apple/Vulkan-only
   源选择/patch/身份不变。mixed CPU原数学不改，GPU修订后的SDK/Provider身份如实更新，不继承未测性能证书。
4. 原21个GGUF格式的CUDA mapping全接，Graph同源，三narrow类型只用于共同原8个expert合同；
   不扩embedding到无需求格式、不改精度/rank/资源/SM资格。新Provider版本含原IQ1来源及实际source id。
5. 原format字节、每类型解码/MMVQ/MMQ/Grouped与变化输入、坏IDs、两入口/原模型内宽，
   Windows/Linux/三交付/mixed CPU资源、source差分与P0按同一稳定候选资格化；未运行不签PASS。

runtime、Compute/Graph ggml CUDA owner依法重定界至本卡G1/G2；U2已实现冷事实与架构接缝保留并随最终
候选交付验证。没有新公共数学API或自研CUDA算法，没有全库依赖升级或新的Provider family。

G2已实现并在Windows完成原carrier/MMVQ/MMQ与8个共同真实expert形状的普通Direct/Graph局部资格，
见 [IQ1证据](../../../../evidence/correctness/cuda-final-u23/IQ1.md)。原F32解码点积与CUDA Q8激活codec的
初版参考不等价已按源码纠正；产品数学/精度/共同门不变，原失败保留。
全104制品、其它payload/完整region/state、Linux、三交付/mixed差异仍待后续适用验证，不能据此CLOSED。
