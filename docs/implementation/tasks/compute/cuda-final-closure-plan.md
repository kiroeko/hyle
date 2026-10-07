# Compute + Graph CUDA 最终收口

2026-10-07 用户授权新的 goal：按当前算子表审计 CUDA 覆盖与接线，处理普通构建／DLL／三交付，
裁决成熟 Provider 重叠域、条件路由和冗余自研。启动 baseline
`f5f137f1b3e48bdd56acd42f7471d3c412c7b6e3`；Windows 工作树干净，保留后续与用户修改。
当前状态只看 [registry](../../module-status.json)，不恢复旧 goal、百分比或施工队列。

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
- 先检出并接入成熟库；本轮所有自研 CUDA 数学先冻结，不扩张、不优化挽留。
  SGLang／DeepSeek／CCCL 原版程序与 HYLE 资源胶水不按目录误判成自研。
  若必需合同确无成熟承载，具名提交有限方案，不静默删需求或自行写新数学。
- 不重开已完成 CPU，不改 Apple／Vulkan，公共 Graph 的旧 Vulkan 性能 BLOCKED 不自动解除。
  不做 loader／tokenizer／采样／推理服务、模型级 CPU+GPU 联合推理、HIP、分布式或算子大全。
  大模型允许少量 token／context／驻留 experts 与分片，但保留实际内宽、原编码与全部必要分支；
  不声称整模型全权重常驻推理。

## 六包

| 包 | 权重 | 完成条件 |
|---|---:|---|
| [U1 覆盖与接线审计](cuda-final-u1-demand-wiring.md) | 10% | 全部固定需求与当前真实候选／硬限制／接线／缺口／成本逐项映射，形成有限 owner FIX；静态检出与运行资格分开 |
| [U2 构建与交付合同](cuda-final-u2-build-contract.md) | 15% | 普通 compute-cuda 配置必要依赖默认齐备，CUDA-only／CPU+CUDA、架构与软件身份清楚；source/static/shared、离线／搬移／DLL 邻置实际验收 |
| [U3 成熟承载批量接线](cuda-final-u3-mature-bindings.md) | 30% | 先接齐原版成熟承载，修正格式／布局／精度／融合／资源错误；普通两入口、生命周期、三交付随本包资格化 |
| U4 重叠域比较与规则 | 20% | 合格重复域有同合同完整成本裁决与冷规则；单 Provider 正确性／资源／P0；未知身份默认合理、热阶段不裁判 |
| U5 自研清退 | 15% | 无完整优势且有完整成熟替代的自研 CUDA 算子物理删除；必要资源控制、通用基础和具名独有需求保留 |
| U6 终审与停止 | 10% | Windows／WSL Linux 所有固定计算与必要 region／状态／失败／重放、三交付、规则负空间、七门、文档／LOC／清理完成 |

权重只用于新 goal 工程估算；文档、旧 CLOSED、研究探针不算新产品完成。
U6 汇总复核，不能首次接线／首次验证前五包承诺。每包产品施工前绑定实际 owner，按协议
SCOPED→AUDITED 固定有限 FIX；旧 AUDITED／CANDIDATE 范围不同先合法重定界，CLOSED 只凭具体新事实重开。
U1 先做只读协调审计，不批量改 registry 来假装所有 owner 已经重开。

## 执行与性能预算

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
