# F3候选红队与收口审查

审查者：Codex `/root`，2026-10-12。采用实现停止后的独立审查阶段；没有第二agent/reviewer，
不伪称外部批准。范围固定为[活动卡](foundation-f3-file-resource-streaming.md)原八FIX与六owner，
只读共同/四route生命周期、普通consumer、已运行证据及交付边界，不重审无关模块。
初始候选3b7b9b9b8；当前不可变产品97858fbc8。最终verdict PASS，所有原适用运行/交付门已齐。

## 具名发现及处置

| 发现 | 严重性 | 处置与证据 |
|---|---|---|
| 首组错误撤销未启动后组时，后组已接纳producer保活环 | P1 correctness/lifetime | 原F3-06具体缺陷，独立WSL CUDA子进程真实active use残留1、metadata不balanced/abort6；共同owner退回AUDITED，锁内移出Completion、锁外交原native退休。只一次有限纠正。原FAIL与当前Win/WSL七项、三sanitizer各十三项见[suffix收据](../../../../evidence/correctness/foundation-f3-suffix-retirement/receipt.json)、[Windows复验](../../../../evidence/correctness/foundation-f3-windows-suffix-retirement/receipt.json) |
| 过期deadline及重复stop/close原来只有间接证据 | P2 evidence | 原F3-06补公开API组合，Win十二/Air十三/WSL三sanitizer各十三项实际运行；零IO/native/ready且原用户资源可用。见[当前风险](../../../../evidence/correctness/foundation-f3-current-risks/receipt.json)，该旧产品收据仍保留其精确身份；新产品回归见suffix收据 |
| getter误导缩进及测试empty optional初始化warning | P3 engineering | pre-candidate只换行/显式空初始化，完整TU非空白token等价；当前实际owner/route/sanitizer回归通过，不把旧SHA写成新候选 |
| 原Windows source与CPU test共用oneDNN binary目录 | P2 verification cost | 原SDK一次再次构建oneDNN日志保留；后续不再切两配置。新Win CUDA fixture用普通shipping source artifacts，最终static仅44项增量。没有改公共构建工具或把依赖扩成测试根 |
| GPU完成、逻辑owner归零被误读为driver内存归零 | P2 claim risk | 指南/P0/收据明确UNKNOWN，不外推vendor私有metadata/物理驻留；CUDA sticky terminal按原F2进程重启边界，不reset共享primary context |
| shared库依赖名字被误读为实际迁址加载 | P2 evidence | WSL shared `ldd`所有HYLE来自relocated/lib；Air CPU/Apple 25条、Vulkan两个consumer实际dyld全部来自迁址目录，原otool日志仅作依赖表；Windows用最小PATH运行及同目录DLL/imports |
| 软件Vulkan冒充WSL GPU资格 | P1 qualification | 重新通过原loader枚举，仅llvmpipe设备type4，无qualified GPU；GPU数学NOT_APPLICABLE，当前source与installed适用构建/头/缺席/运行库解析已实际通过 |
| explicit direct IO只有原IO证据，F3接缝缺直接旅程 | P2 evidence | 当前Win真实13项、WSL三sanitizer各14项、Air14项；Win/Linux用原kDirect文件与4KiB stage完成8KiB读写、错误tail不consume，Mac实际拒绝且direct数据旅程NA，见[direct收据](../../../../evidence/correctness/foundation-f3-direct-io/receipt.json) |
| 私有增量runner残留旧product label | P2 evidence identity | 原raw job字段保留；已核对完整SOURCE-MANIFEST/实际修正候选/changed TU SHA并在suffix收据显式说明。新收据按真实source身份签，不用粗标签换掉执行 |

## 当前反证结论

共同控制的admit先完成bounds/physical/alias/bytes/metadata/cancel准入才移动权限；失败保持输入，
固定目录与KS stage不随文件大小增长。旧结果保留request与backing；native未知保持quarantine，
complete与retirement分开。真正EOF/readonly/正数short-write与native suffix/unknown/late均有实际终端证据。
当前纠正覆盖worker失败、接纳spawn失败与collect后spawn失败的原producer交接，所有销毁回调在锁外。
每次仅移一个Completion；原retention在实际native退休前继续保留state/stage/uses，不增加native wait。

四route自然工厂仅消费原IO/Sched/Mem/F1/F2/各native owner；没有新复制/数学kernel、万能Queue、
第二runtime或模型parser。普通独立样例只选择Graph与路线，新内部compute_stream不成为用户开关。
资源/allocator/Queue/Context借用寿命见公开头和用户指南；close不关闭用户的整套runtime。
P0新增Failed目录扫描O(QGR)已计账，没有timed赢家、零热分配或driver硬预算承诺。

当前实际数学仍只签Qwen原始四范围约72MiB与已有编码小fixture，四路线Direct+Graph独立参考和
逐byte回写/reload；不是完整FFN、tokenizer、生成loop、F4/F5或完整推理引擎。
完整交付格与具名源证据见[矩阵](foundation-f3-verification-matrix.md)。旧PASS/原FAIL/新candidate
必须分开归因，尚未运行的交付与最终aggregate不能通过。

## 最终裁决

当前产品全部适用source旅程及十二个installed profile/linkage格取证：十格运行真实数学，
WSL软件Vulkan两格只签适用构建/头/缺席/运行库解析，普通source消费者也单独构建并签
[来源证据](../../../../evidence/correctness/foundation-f3-wsl-vulkan-source/receipt.json)，不签GPU数学。
Windows CPU/CUDA/Vulkan、WSL CPU/CUDA、Air CPU/Apple/Vulkan当前source真实Qwen往返均通过。

原suffix P1已复验关闭，证据缺格已补齐，未发现未决的原八FIX产品缺陷。原FAIL/TIMEOUT、
用户一次ASan追加预算、GCC warning与TSan启动布局恢复保留；没有换目录重置或P1/全库回放。
统一LOC从不可变a97a59ff5（产品与97858fbc8相同）复算：全库2,456,943，自实现不含测试170,170，
六owner中央归属实现并集26,706/155文件，self-check PASS；不是新增行数或进度百分比。
最终证据审核检查37份F3源收据与归档SHA/ZIP完整性，语义范围按本矩阵逐格裁决。

支持原F3七门PASS与CLOSED。剩余限制仅为未测硬件/版本、vendor私有物理驻留、原cooperative
取消边界以及未实现的完整模型/F4/F5；它们没有被伪写为通过。临时验证副本删除被自动审批
策略拒绝，目录保留，不改变产品资格；不得绕过拒绝清理。
