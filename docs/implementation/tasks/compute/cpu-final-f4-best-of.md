# F4 CPU全重复域批量best-of

<!-- MODULE REVIEW: AUDITED -->

当前包SCOPED，baseline `83240ab43ba80cd549dcf03933e3d69e8f94c2b8`（登记完整revision），F1/F2/F3已完成。
owner compute_provider_native、compute_route_cpu、graph_route_cpu；总合同[CPU最终收口](cpu-final-closure-plan.md)。输入F2重叠域与F3当前合格承载。
只比较实际可执行且同语义/精度/资源/完成边界的候选，覆盖oneDNN/ggml/ATen及剩余Native，
完整region与等价完整程序比较，保留融合。单承载只正确性/资源/P0，能力差异不能算loss。
按真实模型角色、原格式、dtype、布局、attrs、shape、线程与软件/硬件身份归并，normal/holdout预冻结。
API/算法稳定后每批一个不可变候选，原仪器四strata/AA/controls/固定n15与5%实用门，重型串行。
一轮有限preflight/适用P1与至多一批具名接线纠正；不自研优化、不n31寻找PASS，未知不得解释平局。
按用户最新效率取舍，接近或预算内无法裁决时快速选择oneDNN/ggml等完整成熟路线，记录
`ENGINEERING_DEFAULT`及格式/融合/转换/资源/交付依据。无需重复长资格采样，不假写赢家；
该项以正确接入/资源/P0和明确性能限制验收，未裁决测量保持原状态。只有具名material风险仍需原适用门。
稳定方向与实用门分别记录；赢家按低成本冷规则绑定，submit不搜、不计时。旧187/17不自动取得全候选最优。
先完成固定新版oneDNN，不能测完旧版再升级；自研同合同落后且有完整成熟接替即交F5删除，优化挽留预算0。
规则显式覆盖已验平台/版本/硬件/线程/真实模型端口；未知身份走合法成熟基线，不承诺所有机器绝对SOTA。
通用默认必须逐合同归属，而非沿oneDNN→ggml→ATen全局顺序；原压缩矩阵/完整融合/稳定函数等分别确定
合格成熟承载与依据，复用库内ISA，精度/错误/资源不匹配明确排除。验证特化命中、未知条件合理回退及feature裁剪。
数值/资源/交付缺口的MUST仍是阻塞；仅性能无法区分的成熟候选按用户授权工程默认裁决。
不得宣称全部规则均为统计性能WIN；各批规则正确性/资源/三交付同时完成。

## F1等待期间的有界只读输入对账

基线 `36123059f23e409978499681ee2df748e74d386f`；本节不启动F4产品实施、不改变registry owner，
也没有运行P1。F1 Linux混合loader仍待，完成后再按协议绑定SCOPED/AUDITED与有限FIX。

已读 `bench/compute_provider_native/installed_p1/native_domain_bench.cc` 和其两份contract，
与当前三个成熟factory、F2端口矩阵及Native源码对账，确认以下有限迁移需求：

| 旧验收入口的具体限制 | F4必须固定的迁移 |
|---|---|
| Session只以bool选择Native或普通CPU默认栈 | 比较具名oneDNN/ggml/ATen与实际self候选，不把默认栈替代全部候选 |
| catalog仍含Native MatMul和SDPA，当前两项已物理退役且原plan明确拒绝 | 这两族继续比较实际成熟交集；从self裁决集删除已退役候选，不复活源码 |
| contract固定旧oneDNN3.13.2/旧ggml456与历史206/72预资格结果 | 按当前不可变SDK/实际软件和新的预资格批绑定；不继承历史WIN或AA结果 |
| 通用宽度1536与rows1/32不能代表F2全部真实角色 | 从固定需求选有限normal/holdout；完整内宽/编码/precision/资源不偷换，未测域成熟工程默认 |
| Native identity同时交付自研与借入原程序 | 逐源码族区分self与成熟codec/程序，不能按Provider名字整项删借入能力 |

普通用户入口继续只选CPU路线；上述显式候选控制仅属于开发验收。新比较保留完整准备/重放、
转换/packing/临时与完成成本，Graph完整region单独同合同对账，不把拆叶子结果外推融合。
复用现Data/Session、统一仪器/controls/四strata和固定n15，只迁移具体benchmark consumer与contract，
不重写公共性能核心；具名新接线缺陷最多一批纠正，近似或噪声按既定工程默认快速裁决。

## 本轮SCOPED：固定owner与有限审计

真实trigger：F1固定oneDNN3.13.4与单/混合CPU core交付后，旧187/17证书不适用；普通模型CPU两入口
需要当前真实同合同候选裁决、未知条件合理默认，并向F5提供源码族去留依据。旧harness的具名限制见上节。
Native与CPU route原F3 CLOSED、Graph CPU route原F1 CLOSED按此新版本/完整候选消费要求重开；
成熟oneDNN/ggml承载owner保留原CLOSED，只有具体接线反例才另按协议定向绑定，不能泛化重审。

| 有限审计 | 固定输出/允许后续FIX边界 |
|---|---|
| A1 实际重复合同 | F2→当前源码的dtype/precision/layout/shape/attrs/资源/状态交集与完整region；单承载不跑P1，self与借入程序分清 |
| A2 批量consumer | 复用原Data/Session与仪器，改具名候选/当前SDK身份与有限normal/holdout，不复活退役MatMul/SDPA，不写新数学 |
| A3 准备与执行账单 | 准备、转换、packing、临时、线程、checks、同步/完成与融合损失同边界；资格差异排除，不计LOSS |
| A4 有预算裁决 | 一批preflight，固定四strata/AA/controls/n15/5%；明确差距与方向分别记录，接近/噪声快速成熟工程默认，不追加长循环 |
| A5 普通规则 | 既有冷规则/private表按真实格式/模型/硬件/软件/线程/recipe绑定；完整region优先，未知合法默认、无在线计时、submit不搜索 |
| A6 退役交接 | 35种/15源码族逐项映射；没有完整优势且成熟完整替代的self交F5物理删除，独有资源域明确，SIMD/Mem基础不删 |

主consumer：固定目标模型的projection/norm/index/state/完整片段普通Direct与Graph；正交consumer为
无模型身份的tensor批处理及相同操作的不同真实role/布局/holdout，不由旧benchmark宽度定义API。
公开API/layout、Provider资源与原执行程序不变；SCOPED只授权有界审计，AUDITED有限FIX冻结前不改产品。
current source/static/shared与三平台资格沿F1/F3对账，规则新命中/失配及新默认在本包完成两入口/资源/三交付验证。
无新依赖升级、无自研优化、无模型loader或GPU资格队列；F4结束前不启动F5删除。

## AUDITED：真实候选交集与有限FIX冻结

源码对账：`ggml_cpu_provider.cc:1278`的真实supports/standard_index、
`aten_cpu_provider.cc:126`的select、oneDNN的BRGEMM/normalization/tensor准入与
Native的build_plan/propose/borrowed torch分支。F2需求保留；候选表不能仅沿旧F2暂定名字推断。

| 当前交集/独有需求 | 具体结论 |
|---|---|
| oneDNN与ggml | 实际plain F32 dense rank2 MatMul、RMSNorm、LayerNorm、Copy；Select不在ggml普通承载，旧暂定表此项撤回 |
| oneDNN与ATen | 上述实际合格交集、Select和F32稳定Softplus；Mac oneDNN不存在不制造失败竞争 |
| ggml与ATen | Cast/Fill/Concat/Gather/Scatter/Reduce/Softmax/RoPE/SDPA/MatMul/norm的具体precision/layout/index交集，真实prepare还须确认 |
| stable/checked差异 | ggml普通gate/稳定Sigmoid/SiLU/GELU/Softplus、Bool或LogSoftmax不是原融合模型程序的数值合同，不强行视为同候选 |
| ARM F16 MatMul | ggml原half dot已由F3拒绝F32 accumulation；ATen承载，不能用低精度dot签性能胜利 |
| self | 11 Unary、6 Binary、4 Reduce、2 norm、Softmax/LogSoftmax、Gate、Copy/Cast/Fill/Gather/Scatter assign/add/Concat/RoPE/Select共35种/15族；Gelu/MatMul/SDPA不是现self候选 |
| borrowed | Native前置is_torch_cpu_operation及ds4原程序不是self；433原GGUF端口等单承载不逐格式竞速 |
| 真实资源差异 | Native bounded RMSNorm/Copy、caller-owned prepared与严格private边界需独立保留/替代裁决；managed exact/private unknown不能仅凭数学接近算同合同替代 |

成本账单：Native是原caller-owned程序与plan/workspace；ggml固定原graph/pool及必要cast/codec；oneDNN原
threadpool/BRGEMM/primitive资源；ATen managed含每次numeric_input byte copy、dtype转换、临时与可能私有线程。
计费统一从Invocation.submit至真实Completion.wait，cold准备/常量转换另记录。Graph必须保留完整region和已发布边界。

| 有限FIX | 允许产品/consumer动作 |
|---|---|
| V1 | 新有限benchmark consumer复用旧Data/Session，参数化具名Provider与F2实际normal/holdout；逐候选明确decline/error及资源事实，原retired种从self集排除；不改统一统计/runner |
| V2 | 更新本包contract/SDK/provenance，冻结真正可比pair和完整region；一轮预资格与适用P1，不继承旧206/72或187/17，不扩大采样预算 |
| R1 | 仅按已资格证据更新Native/private CPU冷规则和逐合同成熟工程默认；保留原create/prepare/submit/销毁，不改API/对象/数学 |
| R2 | 普通两入口的命中/失配/未知回退、资源/状态/三交付差异集中验证；历史规则/未测域不假签WIN |
| L1 | 去留ledger区分self、borrowed、unique资源与成熟接替；F5才物理删除，SIMD/Mem基础保护 |

normal/holdout固定真实内宽（Qwen3.5-2B2048与F2 norm4096）、token1/小prefill32及原端口K/N与原布局；
MatMul保留同一实际weight布局，oneDNN拒绝原转置域就记录非候选，不用免费转换制造竞速。
非覆盖点采用已资格成熟工程默认，不把样本外推成通用统计冠军。默认caller预算1，记录原库实际线程/私有资源。
三owner从SCOPED进入AUDITED，仅V1/V2/R1/R2/L1，原成熟Provider owner保持CLOSED；具体接线反例另依法定界。

### V1/V2具名批次入口纠正与最终预算

767准入pair的初批A/A固定8轮完成：334短资格、433未裁决，原聚合INCONCLUSIVE保留。
统一runner不传CLI参数；尝试的逐pair sealed子进程传输一次99.281秒，源码证明它改变原batch进程生命周期，
不能凭child文件SHA继承原A/A。因此该传输未签性能资格，source/CMake全部删除，不带进产品。
纠正限定为benchmark冷入口读取固定pair清单，同进程保持原Data/Session/prepare/replay/完成与执行代码。
只对原334集合做一次固定8轮的纠正入口检查；不重测原433、不增加n、不移植公共runner/统计代码。
此次为V1/V2唯一具名性能consumer拓扑纠正；后续仍无法裁决按成熟工程默认/具名独有资源边界处理，
不得继续搭建仪器或追加A/A循环。最终P1只固定适用资格清单，n15/四strata，不升级n31。

同进程纠正8轮完成后最终128个pair资格，206未裁决；原两份INCONCLUSIVE均保留。
P1第一次启动在首样本被可选telemetry-prefix拒绝：现有marker仅有caller/affinity，不报告order。
不新增或伪造字段、不改测量image/公共runner；合同去掉该不适配的可选要求，order以已绑定source SHA、
runner受控双顺序/分配环境与原nanobench调用分支证明，实际process affinity/priority仍由公共canary验证。
原启动失败/raw保留；P1固定128/n15/四strata，正常/holdout数据不因v1的experiment.holdout=false改动。
