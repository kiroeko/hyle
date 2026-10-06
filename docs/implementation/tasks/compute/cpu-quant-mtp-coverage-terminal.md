# CPU 热门量化/MTP/DSpark扩覆盖：终审

<!-- MODULE REVIEW: CLOSED -->

## 身份与范围

- owner：compute_ops、compute_provider_ggml、compute_provider_native、compute_ggml_runtime。
- 总计划：[CPU扩覆盖](cpu-quant-mtp-coverage-expansion-plan.md)；C5 baseline
  `400927ff7af0b150432b830ce47b3aa7ac5fd3c9`，scope/audit状态按C1–C5卡保留。
- 不可变产品/测试候选：`c740abc74a5a62ffacf60b26798228ca484317df`；之后仅docs/evidence/status。
- Windows canonical。已保留架构决策后的所有提交；无回退、无无关修改覆盖。
- 资格域：Windows/Linux x86_64 CPU-only新版SDK，全部固定104 GGUF档位、两新官方DeepSeek、
  三sidecar、四准入非GGUF制品的原语言计算合同；模型解析/权重加载/采样/接受策略在消费方。
- 完整权重模型推理和104次整模型run不属本记录的声明；没有以此删减任何已准入operator/region。
  原GPU状态/资格/拒绝/FAIL与Graph Vulkan P1 BLOCKED不改变。

## 完成输出与FIX

| 包 | 完成输出与证据 |
|---|---|
| C1 | 全104档位、826 shard/sidecar、152 Safetensors shard真实header/角色/量化/shape；成熟BORROW准入，无自研数学授权 |
| C2 | CPU-only锁定ggml0.26+原Unsloth IQ1扩展及资源patch、同源baseline/AVX2私有程序；一SDK一core；旧GPU envelope保留原版；真实software label和Native旧peer约束 |
| C3 | 11新增GGUF原block/type/MatMul/Grouped/Gather；NF4/nested、AWQ、CT与MLX affine/MX4/MX8原typed接缝，cold decode/selected group/原精度与资源 |
| C4 | 新Pro/Flash主干及四DSpark片段、完整三stage/cache/Markov/confidence、三个实际sidecar完整decoder/stages与全head |
| C5 | 559 numeric GGUF、34 packed非GGUF、45 raw text、80 GGUF role/shape；104身份及完整family semantic/state联合合同；新大Q8表准入、新原ATen lowdtype conv，准确Qwen RoPE/norm/完整NextN补验，更新两系统SDK |

全部对象的准确证明链、逐制品关联、边界dtype/state与复用条件见
[联合合同裁决](../../../../evidence/correctness/cpu-quant-mtp-c5-required/joint-verdict.md)及其JSON。
实际数值/生命周期/安装运行通过与metadata完整度分开记录；没有把旧CLOSED、研究或doc提交当新数学完成。

## 七问

1. 第三方：stock ggml ffa4e8b8、Unsloth原IQ1推理port46cb、LibTorch2.8 CPU和固定
   bitsandbytes/AutoAWQ/compressed-tensors/MLX原计算组合。锁、archive SHA、patch及许可证/notices
   随C1/C2/source cache与SDK资产保留；不追master、不同SDK不共用未隔离core对象。
2. 功能：全部C1准入typed contract及必需family/branch已有普通Direct/Graph完成路线，
   正交原压缩矩阵/Gather/重复expert及非模型名绑定合同亦通过；应用编排/loader不是HYLE数学kernel。
3. 内部复用：CPU资源、typed TensorDesc、准备/Invocation/Completion/Graph规划与Mem/Sched原边界复用，
   不复制线程/队列/allocator；本轮未重开下层关闭模块或扩大Mem尾项。
4. 性能：P0复杂度/布局/codec/cold转换/原库热临时/线程/同步真实记录。仅有一个具名合格承载的新域
   不造peer；没有新winner/SOTA、没有适用新P1采样。普通成熟顺序与旧native软件资格取舍显式保留。
5. 交付边界：仅C++源码/static/shared，无新增C/FFI或模型引擎；私有原库tensor/plan只属于对应SDK。
6. 安装：普通find_package产品链，两系统source/static/shared、DLL邻exe、搬移prefix、固定资产和
   现有锁定cache offline资格；static HYLE仍依赖外部shared LibTorch和私有CPU program。
7. 易用：用户CPU preset/组件选择即可，默认依赖AUTO，无Provider排序/timing profile；
   storage/scale/zero/codebook/group由实际TensorDesc提供，profile只提供已核对的冷部署身份。

## 七门裁决

| Gate | 实际证明 | Verdict |
|---|---|---|
| scope | C1原集合无删减，104逐typed/parameter/region身份关联；原数据与branch准确需求，明确LIBRARY_CONTRACT_PASS含义 | PASS |
| architecture | 四路线envelope/一core、CPU原库资源/程序/managed边界、prepare匹配而submit不重搜；相同合同组合复用 | PASS |
| correctness | 原codec与packing/precision、F16/BF16/F32/I32端口、完整semantic/state和branch；实际输出、changed replay、cache反馈/reset/wrap、index失败 | PASS |
| delivery | C3三交付/搬移/公共头，C4完整安装消费者；新大表64 SDK、新conv/raw text360 SDK和source180，软件/配置差异精确匹配 | PASS |
| usability_docs | CPU自然入口/preset/AUTO依赖、原dtype/模型身份/managed/strict/capture/失败限制、当前CPU覆盖页与旧四路线账本区分 | PASS |
| red_team | 冻结后独立核对dtype、完整NextN、旧fixture误判、codec、公平成本、真实OOM、旧binary/peer版本与scope误导；联合裁决无未决产品缺陷 | PASS |
| engineering_performance | [C5 P0](../../../../evidence/correctness/cpu-quant-mtp-c5-plain/cost-audit.md)与[C4 P0](../../../../evidence/correctness/cpu-quant-mtp-c4-installed-sdk/cost-audit.md)，真实必需转换及managed成本；无新同合同winner/P1门 | PASS（P0） |

新模块/公开头/ABI/构建图不存在最终变化，已匹配证据不因doc/status重建或重跑P1。
两个actual新adapter格均有自己的正确性/资源/更新SDK，不以旧binary代签。
每轮closure/docs检查实际通过；registry合法AUDITED→CANDIDATE→CLOSED，旧GPU及其它状态不改变。

## 限制与恢复条件

- CPU-only0.26/原IQ1扩展、固定LibTorch2.8与已验x86_64平台，不外推Mac CPU、任意ISA或新版依赖。
- 已验模型Math boundary和合成容量：真实K/N/格式/state/完整V；tokens/context/驻留expert可减，
  head驻留分片，text MRoPE三轴相同。没有整模型常驻、全实权重推理或性能吞吐承诺。
- NF4/AWQ/CT显式cold decode由caller持有dense结果；MLX只current group/selected expert。
  F16 GGUF某些输入需显式cold转换，成本保留；不称零转换原packed路线。
- managed原库可分配/使用私有线程，private bound unknown、capture不承诺；失败slot要重建，
  多节点不全链事务回滚，caller负责cache提交/回滚策略和借用权重保活。
- 原GPU发布拒绝与数值FAIL、Vulkan性能BLOCKED保留；不自动扩GPU或模型级联合推理。
- reopen仅按真实新制品/packing/g_idx/state/数学差异、依赖升级、可复现correctness/lifetime/UB缺陷，
  或合格同合同material性能反例；不是重看旧任务、固定日志秒数或未知硬件推测。

## LOC、清理与停止

统一29路径：Compute+Graph实现 **61,499行 / 250文件**；全库1,813,075行，
自实现不含测试156,726行，见[候选LOC](../../../../evidence/correctness/cpu-quant-mtp-c5-plain/loc.json)。
后续doc/status不改变实现规模。
临时consumer目录按owner/绝对路径/活动进程验证清理，C4约1.79GiB、C5两批SDK各约1.24/1.25GiB；
原大失败log/XML无损压缩校验后清理重复325MB。producer、SDK、source/dependency cache及原始证据保留。

最终裁决：本轮四CPU owner在此准确合同范围CLOSED。通知“CPU量化与MTP扩覆盖完成”，
停止本goal，等待用户讨论下一阶段；不自动填充算子大全或恢复旧GPU队列。
