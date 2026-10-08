# Compute + Graph CUDA 最终收口

2026-10-07 用户授权新的 goal：按当前算子表审计 CUDA 覆盖与接线，处理普通构建／DLL／三交付，
裁决成熟 Provider 重叠域、条件路由和冗余自研。启动 baseline
`f5f137f1b3e48bdd56acd42f7471d3c412c7b6e3`；Windows 工作树干净，保留后续与用户修改。
当前状态只看 [registry](../../module-status.json)，不恢复旧 goal、百分比或施工队列。

## 本 goal 完成

后续：用户明确要求补齐与CPU同级的实测优选和对应清退，现活动计划为
[CUDA性能补齐](cuda-best-of-completion.md)。下述100%是上一轮工程默认范围历史，不签新性能补齐完成。

2026-10-08，U1--U6本次CUDA基础库冻结范围完成，整体100%，
见[逐项终审与限制](cuda-final-u6-audit.md)。当前owner经独立CANDIDATE后按本goal预声明
工程默认/P0裁决关闭；原U4 AA失败/未取得P1冠军保持INCONCLUSIVE，没有改成PASS/SOTA。
两OS固定原计算/完整branch与source/static/shared、CUDA-only/mixed已验；六重复公式已
物理删除借入成熟算术，具名独有数学/资源保留。全Compute+Graph实现60,774行/255文件。
旧U4临时产物Windows25.408GiB/WSL0.420GiB已清理；不扩算子、模型engine或其它GPU。
后面的百分比/阶段日志保留为历史进展，最终只以本段/registry/terminal为准。
完成后停止，等用户讨论下一阶段。

## 固定需求与边界

- 要求来自 [固定制品与 tensor 需求](cpu-final-demand-carriers.json)，其中 CPU 的候选／状态不签 CUDA。
  人读总表见 [模型／算子／实际量化格式共同目录](../../../architecture/model-compute-catalogue.md)。
  保留 104 GGUF、559 数值合同、80 参数角色／形状类、45 plain 与 34 non-GGUF packed 合同，
  八模型族、官方混合 dtype、Pro-0813／Flash-0731、三项 MTP／DSpark 和本地 Qwen3.5-2B 原版。
- 用户明确：CPU 那轮已根据新增模型、MTP／DSpark 与真实混合权重推导新算子需求；本轮 CUDA
  必须继承这份完整目录，能力目标与 CPU 对称。模型／制品身份以已固定目录为准，不因口头日期重新开画像调研。
  上一条“0816”外部身份查询只作历史诊断，不形成另一个待用户确认的需求／阻塞。
  只在当前目录与具体 Provider 程序发生可证矛盾时，定向核对该合同来源；不重复调研全部模型。
- 每项以 tensor 角色、原物理编码／量化状态、精度、真实内宽、阶段、状态及必要完整 region 判覆盖；
  文件 Q 标签、设备枚举、copy、mock、旧 CPU PASS 或旧 CUDA 代表样例不能代签。
- 普通用户选择 CUDA 路线；Direct／Graph 自然入口冷阶段匹配／绑定固定规则，热阶段不搜索、计时或重选。
  规则可表达 OS、GPU 型号／架构／SM／能力、真实驱动发行版、Driver API、Toolkit／runtime、Provider
  版本、模型家族／制品 revision、端口编码／精度／shape／资源条件。未知身份有具名合法默认或明确资格拒绝。
- 用户追加明确：缺已测best-of证书只撤性能优选，不禁用仍合法的成熟能力；
  默认按具体算子／原格式／精度／资源／融合成本决定，标ENGINEERING_DEFAULT，用户无需排序或timing。
  真实程序／设备／精度／资源不支持才拒绝，旧单机性能证据不能成为全Provider永久硬门。
- 先检出并接入成熟库；本轮所有自研 CUDA 数学先冻结，不扩张、不优化挽留。
  SGLang／DeepSeek／CCCL 原版程序与 HYLE 资源胶水不按目录误判成自研。
  若必需合同确无成熟承载，具名提交有限方案，不静默删需求或自行写新数学。
- 不重开已完成 CPU，不改 Apple／Vulkan，公共 Graph 的旧 Vulkan 性能 BLOCKED 不自动解除。
  不做 loader／tokenizer／采样／推理服务、模型级 CPU+GPU 联合推理、HIP、分布式或算子大全。
  大模型允许少量 token／context／驻留 experts 与分片，但保留实际内宽、原编码与全部必要分支；
  不声称整模型全权重常驻推理。

## 六包

当前U1审计、U2构建交付、U3成熟接线的冻结范围已完成，见[联合资格](cuda-final-u23-qualified.md)。
恢复后的四组合运行用76分09秒完成。U4 cold默认/能力分离与六种小交付已完成；
三组重复域AA FAIL无新WIN，按授权成熟ENGINEERING_DEFAULT取舍，不追加采样。性能资格
仍INCONCLUSIVE而非PASS，U5清退/U6终审待，整体工程估算约70%，不把工具/文档/旧CLOSED
计入产品完成。

| 包 | 权重 | 完成条件 |
|---|---:|---|
| [U1 覆盖与接线审计](cuda-final-u1-demand-wiring.md) | 10% | 全部固定需求与当前真实候选／硬限制／接线／缺口／成本逐项映射，形成有限 owner FIX；静态检出与运行资格分开 |
| [U2 构建与交付合同](cuda-final-u2-build-contract.md) | 15% | 普通 compute-cuda 配置必要依赖默认齐备，CUDA-only／CPU+CUDA、架构与软件身份清楚；source/static/shared、离线／搬移／DLL 邻置实际验收 |
| [U3 成熟承载批量接线](cuda-final-u3-mature-bindings.md) | 30% | 先接齐原版成熟承载，修正格式／布局／精度／融合／资源错误；普通两入口、生命周期、三交付随本包资格化 |
| [U4 重叠域比较与规则](cuda-final-u4-default-and-best-of.md) | 20% | 合格重复域有同合同完整成本裁决与冷规则；单 Provider 正确性／资源／P0；未知身份默认合理、缺性能证书不禁成熟能力、热阶段不裁判 |
| [U5 自研清退](cuda-final-u5-native-retirement.md) | 15% | 无完整优势且有完整成熟替代的自研 CUDA 算子物理删除；必要资源控制、通用基础和具名独有需求保留 |
| [U6 终审与停止](cuda-final-u6-audit.md) | 10% | Windows／WSL Linux 所有固定计算与必要 region／状态／失败／重放、三交付、规则负空间、七门、文档／LOC／清理完成 |

权重只用于新 goal 工程估算；文档、旧 CLOSED、研究探针不算新产品完成。
U6 汇总复核，不能首次接线／首次验证前五包承诺。每包产品施工前绑定实际 owner，按协议
SCOPED→AUDITED 固定有限 FIX；旧 AUDITED／CANDIDATE 范围不同先合法重定界，CLOSED 只凭具体新事实重开。
U1 先做只读协调审计，不批量改 registry 来假装所有 owner 已经重开。

## 执行与性能预算

2026-10-08 用户授权的执行纠正已经接到本 goal 的 U2/U3：
[本批续跑入口与固定清单](../../../../evidence/correctness/cuda-final-u3-joint-delivery/delivery.py)
及[有限工具卡](../build/cuda-delivery-execution-control.md)。用户随后明确恢复 U2/U3，
本轮从2026-10-08 09:59:04 +08:00计时，清单`run_enabled=true`；不创建另一个CUDA goal。
Linux日志参数的具体配方缺陷已由[有限路径修复](../build/cuda-delivery-linux-output-path.md)
纠正并通过真实WSL轻量写入检查，Windows身份及已通过记录不失效。
Linux首次configure的shell分号重解析由[原样参数修复](../build/cuda-delivery-wsl-argv.md)
定向处理，原FAIL和同ID纠正次数保持；配置通过前不启动其build。
Linux shared SDK完成后，任务内输入检查错误假设`.so`邻exe，实际40个HYLE库在搬移SDK的
`lib`目录、由原RPATH加载。只纠正任务内hash的路径（static对应archive），不复制/改库，
不改公共仪器或数学；已通过build/configure复用，未启动的运行项保持待验。
Windows的WSL UNC视图不能直接stat Linux `.so` symlink；本批Hyle0.1.0指纹只取原
`.so.0.1.0`实体payload，不把symlink误报为缺库。原symlink/RPATH/loader由实际Linux运行验收。
Linux shared的34个packed合同合并进程在大词表Affine用例触及显存11979/12282MiB，
原300秒timeout已FAIL并保留；进程退出，无后台遗留。唯一纠正按原34个完整case分新进程，
同ID累计两次预算/原300秒组预算、shape/精度/误差门/必需分支不变；并集逐名核对后才PASS。
未重建产品、不放宽数学、不把原timeout记作性能输赢；纠正仍FAIL则停止该项，不追加采样。
仅用此入口执行剩余交付，旧本批启动脚本退出执行路径。它保留原数学全集，四个组合共享
同一配方，命令通过后复用、失败后停止、记录/预算固定在 canonical task ID；换输出目录和
文档提交不重置预算。不得用新脚本/ID 绕开失败停止。进度读 `delivery.py status` 与当前
owner 卡，不能把工具任务完成算为 CUDA 产品完成；U4/U5/U6 的范围和权重保持。

U2/U3恢复后的本轮收尾目标约90分钟，总执行预算120分钟（从实际恢复开始，包含构建、
运行、诊断和记录时间）。只完成清单中的四个交付组合并封存已有Linux static结果：正常
交付60--80分钟、结果核对/收口10--15分钟、具名纠正最多20分钟。每30分钟报告已验收组合、
剩余组合和累计时间；实际阻塞立即说明。到120分钟停止启动/追加工作，已有有界进程保留
记录并安全结束，按当前owner协议裁决未完成项；不改ID、换目录或自行增加预算。时间预算
不是降低验收门或承诺全部PASS，未验证项仍未完成。本轮预算终点为2026-10-08 11:59:04 +08:00。

2026-10-08 用户追加 [Compute + Graph 宏规范专题](compute-graph-macro-normalization.md)：
全部 Compute/Graph 的 OS／arch／compiler／feature／visibility 和 GPU 条件分层核对、批量规范，
不改借入原数学、不自动解除其它路线 blocker；N1 当前批次验证后独立扫描并按真实 owner 定界。
这是额外横切任务，保留原 U1–U6 和全部验收范围，不能以规范工作代签 CUDA 覆盖／性能完成。

批量接线后集中构建／验证；重型串行，build j2、test j1，复用当前 producer、锁定依赖 cache 与身份相符证据。
Windows 是唯一产品真相，WSL 只前向同步不可变候选。工具隐藏启动，无 UAC／Terminal／缺 DLL 弹窗；
普通 Windows consumer 调用既有 runtime staging，让 DLL 邻 exe。

先源码 P0：每次 allocation／copy／同步／布局／量化状态／融合边界与冷／热成本。
API／算法稳定后，对不可变候选做一次适用批量 P1；比较包含必要转换、上传下载、同步、融合损失与准备／重放。
每批一次 preflight、一次适用完整采样，至多一批具名接线纠正；不升级 n、不无界换姿势重测。
资格不成立不伪 WIN／LOSS。近似或预算内无法判断时，按原格式直算、融合、资源与交付作具名成熟
`ENGINEERING_DEFAULT`，不声称全硬件 SOTA；仍无正确完整承载则具名 BLOCKED。

按 change-impact selector 执行当前卡预声明门；docs／状态不 build、不 P1，不顺手改公共治理或统计核心。
每轮 docs／closure／diff 检查，registry 变化传 previous-ref；提交推送带 `[skip ci]`，不用 GitHub Actions。
临时输出具名、及时归档校验清理，不删当前 producer／依赖 cache／模型／sealed 性能证据。

## 停止

全部固定需求及适用验收完成后明确通知“Compute + Graph CUDA 最终收口完成”，列限制并停止。
确需用户决策时说明具体合同／缺失成熟来源／有限选择；不以研究或历史记录冒充完成，不无限调查。
启动实现 LOC 由统一脚本复算；每包完成、重大范围变化或用户询问时报告实现／验证／未完成／阻塞与全 Compute+Graph 非测试 LOC。

启动统一脚本实测：**60,161 行／249 文件**，不含测试、bench 或第三方实现；`--self-check` 通过。
