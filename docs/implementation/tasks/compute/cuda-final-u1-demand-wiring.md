# CUDA U1：当前需求与接线审计

## Closure card

- 类型／owner：CUDA goal 只读协调审计；当前实际产品 owner 不变，见 [registry](../../module-status.json)。
- baseline：`f5f137f1b3e48bdd56acd42f7471d3c412c7b6e3`；Windows canonical。
- trigger：用户要求覆盖当前算子表。具体反例：CPU 扩充的 GGUF 类型在 CUDA
  `native_quantized_type` 明确 default decline；旧 GPU 验收不覆盖当前 104 制品／新版 sidecar。
- consumer：普通 CUDA Direct／Graph 完成固定模型计算片段与状态反馈。
  主纵切为真实 GGUF projection／embedding／expert／head；正交纵切为官方混合 BF16／FP8／FP4、
  四 non-GGUF 制品和必要 MTP／DSpark，避免把“单一量化矩阵能乘”当完整覆盖。
- scope：当前固定需求→来源／真实 Provider 程序→接线／资格限制→冷／热成本→有限后续 FIX。
  只读审计与 evidence／任务文档；不改产品、公共 API、依赖 lock、registry 或旧 terminal 结论。
- 依赖／消费者：CUDA backend、cuBLAS、ggml CUDA、SGLang／DeepSeek／CCCL、Linux FlashInfer、
  Compute／Graph CUDA façade 与 Graph ggml region；CPU 状态只作需求出处，不提供 CUDA 资格。
- tier：P0 源码／合同／成本审核；本包不跑性能，不新增 runtime PASS。
- 预算：一批需求归并，一批已登记成熟源码能力检出，一批接线／成本／构建审核，一次反证复核；
  每个问题最多一次定向 follow-up。未知当前来源即保留 UNVERIFIED 并具名移交 U2/U3，不无界调研。
- 预声明：输入 SHA／需求与映射完备性自检、selector（docs/evidence）、docs／closure／diff、统一 LOC。
  无产品改动不 build，不启动 P1。
- stop：产出逐合同静态 ledger、region/official/sidecar 待验清单与有证据有限 FIX 后，本审计结束；
  静态映射不签正确性／资源／交付／性能，无当前运行证据统一 UNVERIFIED。

## 有限审计集

| ID | 范围 | 产物／后续 owner |
|---|---|---|
| A1 | 559 GGUF 数值合同、80 参数类、104 制品，核对具体原编码与 projection／head／expert／embedding | 逐合同映射及已登记 ggml CUDA 原版源码支持；缺接线移交 compute_provider_ggml_cuda／graph_provider_ggml_cuda |
| A2 | 34 non-GGUF packed、45 plain，四制品及其原量化状态、比例／零点／codebook／bias／嵌套状态 | 原版成熟程序来源与现有 GPU 接线，缺 carrier 具名，不推导自研需求 |
| A3 | 八族 region、官方更新及三项 MTP／DSpark、Qwen3.5 原版 | 完整分支／状态及当前资格缺口；需要公共 schema 新能力时另定 owner scope |
| A4 | 两 façade、硬件／版本／模型／端口条件、准备与热成本、资源／失败／capture 边界、原版融合 | wiring/cost ledger；后续 route/provider/card 的有限 FIX，不给所有 owner blanket reopen |
| A5 | 标准构建、必要 flags、架构／SDK／Provider 投影、runtime DLL 闭包、三交付／混合 SDK | 源码合同与实际既有证据边界；U2 仅凭具体缺陷／新交付需求定界 |

用户澄清：CPU 扩覆盖阶段已经指定新增模型／MTP 并推导新算子需求；A1–A3 直接继承完整冻结目录，
不另开模型画像、版本或算子需求研究。能力目标跨路线一致，运行资格按各路线实际承载证明。
只有具体目录／Provider 合同矛盾才恢复该项来源查询；先前“0816”身份查询不作为本包 blocker。

## 七问与证据边界

1. 成熟来源按锁定源码与原版程序事实检出，版本／license／fallback 不按 Provider 名称推测。
2. 逐 tensor／完整 region 与普通两入口审核；原版模型数据／量化状态不写成同质权重。
3. 复用既有 CUDA resource／Queue／Completion、Graph region 与 runtime staging；不复制通用机制。
4. P0 检查转换、同步、融合、工作区／寿命与准备／重放；没有本包性能胜负。
5. 本包不增加 C/FFI／万能执行器；成熟库的单算子／融合／region 边界保持。
6. 构建投影与现有 SDK/DLL 事实分开，当前未跑三交付统一 UNVERIFIED。
7. 普通用户只选路线，内部 Provider 顺序不是当前全域最佳的证明；公开 guide 待 U2–U6 验收后更新。

## 当前结果

审计输出见 [U1 evidence](../../../../evidence/correctness/cuda-final-u1/README.md)。
本卡结束前补齐 A1–A5 结果、有限后续 FIX 和只读检查收据；不得提前写产品 CLOSED。

共同需求与首轮 hook 对账已实际生成：104 profiles／559 GGUF／34 non-GGUF packed／45 plain／八族，
六份固定输入 SHA 匹配。119 格式映射＋5 Gather 接缝可在现有成熟 core 检出，三种 IQ1 扩展另有 8 项，
34 non-GGUF 公共操作在 CUDA 没有 hook。全部保留 UNVERIFIED，没有增加运行资格。
region／原版精度／资源审计及固定成熟 GPU 程序检出尚未完成，U1 未签结束；具体 W1–W6／B1–B3
见上述 evidence。无另一个“0816”画像调研或待用户确认阻塞。

## U1 只读审计结束与有限交接

A1–A5 的源码／合同审计已在有限预算内完成；对应 [共同人读目录](../../../architecture/model-compute-catalogue.md)
与 evidence 的原 GPU 来源身份／原承载结论、W1–W6／B1–B3 构成有限交接。
Unsloth 原 IQ1 CUDA 的 18 个相关文件与 C1 固定 SHA 完全一致，已有 convert/MMVQ/MMQ；
其它四固定成熟源共取 71 个必要文件，archive SHA 校验，没有再调查模型或下载大权重。
NF4／AWQ／CT 原准备序列已有 GPU 承载，MLX 原 CUDA 有 affine／MX4／MX8、gather-QMM；
与原 FP8/FP4／新版 DSpark 同样都须 U3 冻结实际接缝并运行资格，不以原源码存在签支持。

新 DSpark／三 sidecar 与八族 49 个 family-region 关联复用共同已定义完整程序，不增加公共数学语义。
接线完整度、原 dtype／rounding／selected group／生命周期／失败、Windows／Linux 交付和性能仍 UNVERIFIED。
不能把源库可以做与 HYLE 当前真正做到混为一谈；U1 完成的是审计，产品验收未完成。
U2 从 B1 preset 子项开始，其他 owner 只按具名实际 FIX 绑定；U3 先成熟接线，自研冻结。
