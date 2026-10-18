# G03 来源冷预留与原控制回接检查点

2026-10-14；baseline `f307dbc5026fdcdffe51b35b64b62dd21545905b`；同一
[AUDITED 活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g03-owned-outcome-observation.md)，
不是 G03 terminal 或七路线证书。前一[聚合 completion 检查点](../inference-g03-owned-outcome-observation/README.md)
保留原始失败/限制。

生产 `detail::CopyOutcomeSource<Update, Access>` 冷建状态、原 Update payload、completion
指针表和 retention 节点，四次注入 allocator/ResourceBudget 分配。接纳前验证原 dependency、
实际 accepted/used 数量、全部已知 completion 错误与容量，失败不消费原 Update，也不 poll。
类型 shell 仅操作原两 controls；节点订阅/观察/回收在原 retained_completion.cc。没有新 executor、
queue、结果槽或 F2/native 默认字段。全部实际退休后的正常完成保原控制供回接；pending abandon
才移交原 native owner，保原来源/后继 native 借用直到真实退休。

## 实际证据

| 验证 | 实际范围 | 原始记录 |
|---|---|---|
| Windows/MSVC | 36 core/result tests；四次 cold allocation 逐一 OOM 完整回滚、预算回到自身 control baseline、容量/known prefix不消费、producer unknown保活、全部回调退休、两 source/observer slots64次复用 | [receipt](windows-receipt.json)、windows-raw.zip |
| Air source | Apple3/Vulkan6原 native案例＋实际 CPU1KiB copy/suffix；Vulkan三 prepared phases仅两 accepted，第一真实 native prefix接纳、第二 native submit OOM拒绝，cold take不观察/不消费、不接纳suffix，最终原wait保OOM/guard0 | [receipt](source-receipt.json)、apple/vulkan-native logs、air-raw.zip |
| 实际 warm source | 两原 CPU F2 invocations 和两 production source slots、64步独立byte oracle；使用同一注入 allocator，分配/重分配调用次数和live metadata bytes均固定；回接原 invocation后reset继续，非每步重建 | [CPU log](apple-copy_cpu-2.log)、air-raw.zip |
| 宿主风险 | Apple/Vulkan各 ASan/UBSan/TSan；native3/6、core35、CPU8（含上述实际warm64）；只用已有两份逐SHA核验的TSan边界模型 | [receipt](host-receipt.json)、air-raw.zip |
| SDK | 两路线static/shared四 relocated prefixes；每组5独立头，共20；实际原CPU source/F2/suffix/原invocation64 replay＋原OwnedResult64成功/取消/原payload/原slot | [receipt](sdk-receipt.json)、air-raw.zip |
| CUDA | 原 producer/F2/suffix 同队列1KiB，四success/Unimplemented cancel/unknown/unknown abandon；production source冷建于接纳前，原guard/reaper/结果保留 | [receipt](cuda-receipt.json)、cuda-raw.zip |

Air身份沿同一 source receipt 实查 Mac16,12/arm64/macOS27.0.1/26A434/AppleClang21；
CUDA实际硬件/编译器见receipt。所有heavy串行、最多四jobs；CUDA执行前dry-run拒绝
.cu.o/nvcc，实际只 common retained_completion.cc 与 consumer 编译/link。没有全模型或旧数学P1。

## 失败、费用与剩余范围

Windows初次source单元夹具把native completions存进被移动的句柄，破坏原F2真实稳定地址
模型，订阅fatal FAIL保留；改为移动原native control identity。第二次预算测试把budget自身
control字节误当泄漏，并配置小于control本身的limit，FAIL保留；修正测试baseline后通过。
正常完成原控制可回接的产品纠正也保留前后source1/source2记录。最新代码/header SHA见
[canonical inputs](product-inputs.json)。一份未被当前测试引用的旧shared-source夹具恢复其
原include，封存脚本逐字证明只该include恢复，并确认当前所有测试不引用它，不用于当前PASS。

累计 source51.898s、host143.036s、SDK42.175s、CUDA52.514s；Windows累计506.594s，
本次增量265.000s（新头触发原tree重配置）。SDK source示例冷metadata总20248B包含原
F2 program/invocation、source和observer，不是source槽单体。Source元数据为固定O(capacity)，
四次冷分配；take/订阅/观察有界扫描、0 warm metadata allocation。OO02原扫描成本和普通
Completion40B/OwnedResult<int>48B沿前证据保留。

仍无实际 Graph 集成或原独立 reduction 消费证书；主G03 typed pending page/init/layout/全页
guards、late native output、七路线64步两request/two slots/oracle 未完成。旧Graph Windows gates
与G04–G10保留。宿主模型不代签driver/shader/private pool或LSan；不把cold utility的PASS
冒充完整pending receiver。详见[有限裁决](validation-result.json)。
