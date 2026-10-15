# 独立工具缺口：共享 benchmark 目录归属

状态：待工具轮定界，2026-10-13；本记录不授权修改公共工具或产品。
baseline `702b6fed97ba1fe7dd2c4d80e992aeb7423f92d3`。

`validation-select.py` 对既有 `bench/inference_resident_sdpa/cpu*` 与新增
`cpu_p1` 返回 `unknown_paths`，原因是共享目录不是 component 名；实际仅修改 CPU
比较适配器，产品 owner 仍为原 CANDIDATE，product_authorization 没有违规项。
原计划 `.scratch/inference-gaps/g01/cpu-p1-adapter-impact-plan.json` 的 BLOCKED 保留。

后续独立工具轮应裁决共享 benchmark 的唯一 owner 与明确验证入口；不得把任意 bench
文件默认当产品或无条件跳过。当前模块不改 selector/policy，按 G01 已预声明的独立
CPU adapter 编译、oracle、A/A 和正式原仪器合同取证，docs/closure照常。此手工适用性
记录不改写 selector 为 PASS，也不授权扩展产品实现或重跑全库。

2026-10-13追加已准入工厂的manifest消费形状反例：CPU provider owner在2202fdde2
AUDITED后，973d3ffbf只把同一已构建ggml factory登记到compute/graph公开产品roots与
CPU未选择过滤集，未改其它factor源码/API/组件依赖。selector对产品root集合差异却将
原集合18个owner全部判为product input changed，要求无改动的CLOSED owner也AUDITED。
原BLOCKED计划保留于CPU候选raw（`cpu-ragged-factory-impact-plan.json`）；它不能用于
启动全库重开/构建，模块不修改公共selector。独立工具轮应区分root增加者与其余原root，
在组件图实际变化与逐头/真实source/static/shared消费上给出有限归属。
本轮CPU工厂实际先复现shared link FAIL，再验证公开facade链接PASS；未运行的其它
路线不标PASS。该metadata/tool归属待裁决，与真实CPU模块施工/验收证据分开保存。

Vulkan RF cold admission补充反例：baseline15dff0644已提交route CANDIDATE→AUDITED，
ddbf598b7仅在managed段尾普通分支增加!View，再以8ec986c38加source-only外部alias
负空间test。selector当前owner确为AUDITED、路径归属明确，仍报`net product change
could not be attributed to an introducing commit`；原计划
`.scratch/validation/g01-vulkan-rf-cold-admission-plan.json`保BLOCKED。Git父子记录和
当前registry明确证明实现前授权，没有其它产品owner改动；工具轮应定位净diff到新增/
替换行的introducing commit边界。模块不改公共checker、不重开无改动owner；本卡仅
保存工具事实，原RF精确source/SDK/宿主风险/真实直接消费者门仍按活动卡执行并保
实际结果，未把工具BLOCKED改名PASS。

2026-10-14 CPU准确页轮补充：baseline8fe62399f已将compute_provider_ggml与
compute_ggml_runtime合法AUDITED，产品3cf78c88b仅原CPU batch/flash私有页地址适配及
两CPU锁的已登记patch身份。原计划`.scratch/validation/g01-cpu-paged-implementation-plan.json`
没有unknown path，仍报同一introducing-commit归属失败；共同dependency lock摘要变化
另使所有facade保守入选。该锁未改其它依赖版本、API、组件图或GPU数学；不能由此
启动全库构建或重开无改动模块。活动卡的CPU source、原ragged/resident直接消费者、
真实relocated static/shared、页读原TU三san与候选定向P1继续；mixed456原资产组合身份
及实际CPU运行须分别保存，未运行域不代签。独立工具条件与产品验收分账。


2026-10-14 NP-01紧凑native页表实际产品owner只有AUDITED compute_batch；中央图只追加
同一owner的batch_paged_native.h split-header映射，未增删component/product root或依赖边。
原计划`.scratch/validation/g01-native-paged-metadata-plan.json`为READY、product_authorization
只compute_batch且无issues，但中央文件任何变化使selector保守选择全库factor/header/delivery。
保原selected/skipped reasons；不由此重开无改动owners或构建全CPU/CUDA/Apple/Vulkan。
本轮执行真实compute_batch/新头/其native直接消费者及原component-contract gate，未运行
无影响域不代签PASS；模块不改公共selector、统计或build core。


2026-10-14 native-page宿主证据/benchmark预校验追加：四新增比较适配器路径仍为
unknown_paths，product_authorization无issues；原plan
`.scratch/inference-gaps/g01/native-paged/validation-san-bench-preflight.json`为BLOCKED，
selected docs/closure与146 skipped reasons保留。产品520027a57没有改动，五owner仍
CANDIDATE。按活动卡已预声明的同数学/lifecycle oracle、原AA/controls/5%合同验证，
不修改公共selector、不因此重开任何产品或运行全库。新增page合同归同一有限适配器
范围；正式采样前继续保存对应新plan及unknown路径，工具归属问题仍独立未裁决。

2026-10-15 G04/G05共同geometry迁移：原10个shared bench C++输入仅迁移BatchCapacity
和PagedArena显式role，原shape/数学/P1合同未改；selector仍unknown同一路径族。
`.scratch/inference-gaps/g04-g05/geometry-plan.json`保原BLOCKED，七个实际产品修改owner
均AUDITED且product_authorization issues为空。共同批仅按其活动卡执行所声明的factor/
public consumer/头门，不改工具core、不启动旧benchmark采样，未运行格不写PASS。


2026-10-15 G04/G05 candidate 成本/CPU Q8 peer 预检：六个新/改 benchmark、CMake、合同与
A/A adapter 输入仍属同一 unknown 路径族，product_authorization 无 issues；原计划
`.scratch/inference-gaps/g04-g05/candidate-cost-plan.json` 保留 BLOCKED、selected docs/closure
与146 skipped reasons。十一产品 owner 在982676 immutable 产品输入上进入 CANDIDATE；
本轮无产品源码修改，仅独立原 DLL Q8 数学/交付片段预检通过。按活动卡继续有限原仪器
合同，不以此修改 selector/性能核心、重建全库或将未运行风险/P1记PASS。

同批 Vulkan Q8 原始 shader 数值预检扩展 reference.cc/.h 与一个独立预检 consumer；
`.scratch/inference-gaps/g04-g05/vulkan-peer-preflight-validation-plan.json` 仍因同一路径族
unknown 而 BLOCKED，product_authorization 无产品修改，docs/closure 与146 skipped reasons
保留。仅按活动卡执行14编码/2plain数值、重放与readonly验证，不改公共工具或启动旧 P1。

同批 Vulkan 多请求编码页 adapter 的三个新文件仍属该 unknown 路径族；计划
`.scratch/inference-gaps/g04-g05/vulkan-encoded-page-validation-plan.json` 保留 BLOCKED，
product_authorization owners/issues 均为空，selected docs/closure 与146 skipped reasons 保留。
仅构造独立比较消费链，下一步按活动卡预声明的三组 shape、两槽重放/readonly、数值和
完整 prepare→output 返回预检取证；产品 candidate 不变，不借工具状态扩大构建范围。

采样前把上述 adapter 的逐页 Concat 改为已有 GetRows 位复制 Gather 后，计划
`.scratch/inference-gaps/g04-g05/vulkan-encoded-page-gather-validation-plan.json` 仍保留同类
unknown 与146 skipped reasons，product_authorization 无 owner/issue；未修改产品或公共工具。
