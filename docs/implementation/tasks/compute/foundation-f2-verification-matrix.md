# F2 固定候选与剩余资格矩阵

本页由[活动收口卡](foundation-f2-batched-transfers.md)纳入，只执行既定七FIX，不扩大F3或数学能力。
产品候选：`03162add05d3e26fe3314c6bcdaafa5a277a3a76`。四owner为compute、compute_backend_cuda、
compute_route_apple、compute_route_vulkan；其余owner保持原状态。候选后的文档/状态/证据不改变产品。

CUDA例外已定界修复：真实CudaCopyLateFailure反证owned free的fatal边界，backend按
[固定终态释放delta](foundation-f2-cuda-terminal-cleanup.md)退回AUDITED后重新CANDIDATE于
`5b3efe2cce3fcab2bc55cf749af2d862a786ef25`。原其他三owner候选/source scope不变。
Windows/WSL该旅程及正常回归已有实证；后续SDK按包含delta的5b3efe2cc全源码执行。

## 实施与已取证范围

| FIX | 固定实现/消费方 | 当前证据与限制 |
|---|---|---|
| F2-01 | common batch值、固定program/invocation/submission；两native namespace同形入口 | CPU/CUDA/Apple/Vulkan首轮slot/replay/OOM/旧句柄取证；完整平台和SDK待验 |
| F2-02 | canonical physical spans、排序alias scan、显式phase闭包、自复制/空批 | common 500组独立oracle及四native来源/对齐/alias用例；其他编译器和新头待验 |
| F2-03 | CUDA固定13.0原batch ABI，冷缺席沿原same-stream memcpy | Win真实batch及decorated absence 1次探测/12次fallback copy；WSL与旧driver实机域不外推 |
| F2-04 | Apple原Job/MLX stream、一次phase blit encoder、实际MCB同步 | Air原native/public/两槽数学取证；vendor成本unknown，worker每native phase实际host wait |
| F2-05 | Vulkan原Run/pair regions/fence、flush/invalidate与已有handoff | Win实际GPU、owned-view、unknown external拒绝及首轮数学取证；Air与WSL GPU资格待验 |
| F2-06 | root去重准入、guard交接、实际退休后ready、producer及partial prefix | 四路线首轮生命周期；Win CUDA cold fallback/suffix/unknown/unsupported cancel实物门；全平台故障待验 |
| F2-07 | ordinary graph façade样例、原Direct/Graph数学、guide与有限成本账 | Win CPU/CUDA/Vulkan、Air CPU/Apple首轮BF16与原编码量化通过；SDK/最终P0与红队待验 |

首轮证据：[共同控制](../../../../evidence/correctness/foundation-f2-common-controls/receipt.json)、
[CPU](../../../../evidence/correctness/foundation-f2-cpu-program/receipt.json)、
[CUDA公共](../../../../evidence/correctness/foundation-f2-public-program/receipt.json)、
[Vulkan公共](../../../../evidence/correctness/foundation-f2-vulkan-public/receipt.json)、
[Apple公共](../../../../evidence/correctness/foundation-f2-apple-public/receipt.json)、
[四路线数学](../../../../evidence/correctness/foundation-f2-math-consumers/receipt.json)、
[CUDA补齐故障](../../../../evidence/correctness/foundation-f2-cuda-faults/receipt.json)。
早期证据只按source/contract/toolchain/machine匹配复用，候选新文件或不同平台不由旧PASS代签。

## P0静态账：当前代码，不是计时证书

N为已准备endpoint数，R为有效copy数，G为phase数（硬上限64），E为显式依赖边数。
固定容量包含N×G角色表、最多2R root uses与producer索引、2R alias intervals、R写覆盖/ranges及G完成槽。
metadata allocator统计实际requested bytes；原native/vendor内部成本明确unknown，严格预算不得猜成0。

| 路径 | 实际结构与成本 | 来源 |
|---|---|---|
| 冷prepare/create | endpoints及所有slot数组一次分配；O(NG+R+G) metadata初始化；原native pool/event准备在冷路径 | `batch_transfer.cc`、`*_copy_program.cc`、CUDA create_slot |
| bind | 单端点验证/物理来源/原mapping；F2不每次重新发现Provider或driver能力 | bind_physical与两native MemoryAccess::copy_span |
| prepare_replay | O(N log N + R log R + NG + RG + G²)；排序physical intervals与roots，最多64phase scan；不做R² range-pair校验 | `copy_ranges.cc`、`copy_control.cc`；earlier-writer scan只对已使用root执行 |
| submit | O(R+E+G)控制/实际依赖；每phase一个原Completion；E≤G(G−1)/2，依赖密集成本不藏在单copy账中 | copy_control admission、original QueueSequence/Job/Run |
| CPU | 原同步byte copy，零伪异步；actual copies按有效range计 | original copy_async |
| CUDA | 有batch时每非空phase一次driver call；冷缺席时每range一次原memcpy；STREAM ordering；一次cold proc探测 | cuda_batch_transfer.cc及XML实数 |
| Apple | 每phase复用一个原Job/recording，单blit encoder内每range一次copy entry；原worker同步actual MCB；vendor MCB/descriptor unknown | apple_copy_recording.mm |
| Vulkan | 按实际buffer pair合并regions，每pair一次CmdCopyBuffer；复用原Run/pool/fence，资格允许时原transfer→compute handoff | vulkan_copy_recording.cc |
| output/reset/放弃 | output扫描逻辑覆盖与实际last-write退休；旧reader/后继readonly保活；drop不等待，原owner退休回调释放 | copy_output_coverage、各submission.output、原retirement |

program/slot准备后，首轮两槽三次copy replay的HYLE metadata allocation不增长；这是本批实际
allocator/控制账，不是全部vendor热分配为零。数学prepare/Graph compile在样例中独立计为cold消费
准备，不能混入copy热路径或作为性能赢家证据。F2不自分配payload staging，调用方LeasePool的
host/device backing及预算独立；report.staging_bytes不能被当作整个消费方的总RAM。
GPU stream drain和原CUDA abandoned-event reaper回调完成不同；借用allocator/resource须持续到
实际退休，取消/未知错误均不提前发布或回收。shipping archive实际确认两个test hook setter缺席。

## 固定资格门

仍按活动卡build1/test1、4GiB、configure300/build1800、test120或sample180，全部重型串行。
复用原source/cache/SDK树；只启动这里及活动卡已声明的门，失败与原日志继续保留，不重置预算。

| 平台/路线 | 当前ordinary source数学 | 必须完成的candidate资格 |
|---|---|---|
| Windows CPU/CUDA | PASS（5b3efe2cc source/static实际复验） | relocated static/shared数学、七头/CUDA ON、两native头与package路线缺席、DLL同目录、shipping hook缺席PASS；既定fault已取证，扩展/终审待验 |
| Windows CPU/Vulkan | PASS（5b3efe2cc全源码下scope匹配实际source复验） | relocated static/shared数学、七头/Vulkan ON、CUDA/Apple缺席、DLL同目录及真实header deps PASS；补齐unknown观察/实际退休10 native PASS，扩展/终审待验 |
| Air CPU/Apple | PASS（scope匹配source；新全源码installed复验） | relocated static/shared数学、七头/Apple ON、CUDA/Vulkan缺席、MLX assets PASS；补晚失败/旧ready/回收8 copy native PASS，扩展/终审待验 |
| Air CPU/Vulkan | PASS（5b3efe2cc全源码下source/shared） | source与relocated static/shared数学、七头/Vulkan ON、CUDA/Apple缺席、实际dyld迁址加载PASS；补齐unknown观察/实际退休10 native PASS，扩展/终审待验 |
| WSL CPU/CUDA | PASS（scope匹配source/static；新delta安装消费已复验） | 新delta真实producer late＋15 CUDA/34 common PASS；relocated static/shared数学及七头/CUDA ON、Apple/Vulkan头与package路线请求缺席PASS；扩展/终审仍待验 |
| WSL Vulkan | 实际NO_QUALIFIED_GPU（仅llvmpipe） | source普通consumer及relocated static/shared构建/七安装头/Vulkan ON、Apple/CUDA缺席、实际ldd迁址解析PASS；GPU数学未运行；扩展/终审待验 |

共同控制/CPU copy sanitizer使用原WSL ASan+UBSan与TSan树，固定34 common、3 CpuCopyProgram；
TSan启动限制只按真实事实处理，不以ASan代签并发。header sole-include涵盖batch_transfer、
copy_control/copy_backend/copy_ranges/lease_access/retained_completion及两native memory头的ON/OFF状态。
普通SDKconsumer仍链接hyle::graph与HYLE_COMPUTE_ROUTES，无新内部factor/provider用户开关。
中央private util边导致的selector component-contract/package扩展门在本candidate执行，保留selected/
skipped理由，不修改公共工具。已有F1消费者只作直接回归，不重开F1或全部成熟算子。

最后检查：完整selected适用门、closure（registry变化显式previous-ref）、docs、diff、P0运行账与
源代码红队/七问；所有适用格有实证才CLOSED。当前聚合资格UNPROVEN，不把表内首轮PASS扩大。

## WSL固定候选本批结果

[WSL收据](../../../../evidence/correctness/foundation-f2-wsl-qualification/receipt.json)绑定上述精确candidate/tree与
完整Git archive SHA。实际GCC14.2/CUDA13.3/RTX4070/driver617.14；8 CUDA、34 common与原backend
隔离late-failure 1项PASS。普通CPU/CUDA source/static同sample两组/两槽/三轮原BF16/Q8 bytes及
Direct/Graph scalar oracle PASS。原Torch CUDA12 leaf/native CUDA13链接警告保留，只签所运行旅程。
ASan、独立UBSan各34 common＋3 CpuCopyProgram PASS，真实compile flags保留；TSan同37项
在一次进程级setarch -R恢复后PASS，初始unexpected mapping/exit66发生于任何用例运行前，原日志
保留，无全局配置或suppression。六共同头及Apple/Vulkan OFF两头GCC sole-include PASS；native ON
及installed SDK当时仍UNPROVEN。原backend late-failure不是F2 producer late-failure证书。

## WSL installed static/shared（含CUDA释放delta）

[SDK收据](../../../../evidence/correctness/foundation-f2-wsl-sdk/receipt.json)绑定5b3efe2cc，原完整canonical archive
与CUDA delta scope见[终态收据](../../../../evidence/correctness/foundation-f2-cuda-terminal-cleanup/receipt.json)。
两生产树更新/安装后真实rename迁址，原prefix消失，Hyle CMake metadata无source/build/staging路径。
普通hyle::graph消费者从安装头编译、两组两槽三轮原BF16/Q8搬运及Direct/Graph oracle PASS。
各七头独立编译PASS（六common＋CUDA ON）；该CPU/CUDA SKU按合同不安装Apple/Vulkan头，
两个negative include明确header not found，未把缺席当native OFF头编译。static首fixture误把这两个
未选择头列为positive，原FAIL保留；同candidate/prefix只修scratch选择并补absence，未改产品/预算。
ldd无missing dependency，shared HYLE库实际解析到relocated/lib。未构建Apple/Vulkan路线的package
请求在两个真实迁址prefix各明确拒绝并列出cpu;cuda，见[补齐收据](../../../../evidence/correctness/foundation-f2-wsl-route-qualification/receipt.json)。
selector扩展/终审继续UNPROVEN；当前四owner仍CANDIDATE。

## Windows CPU/CUDA source/static/shared（含释放delta）

[Windows SDK收据](../../../../evidence/correctness/foundation-f2-windows-cuda-sdk/receipt.json)绑定5b3efe2cc产品scope，
validation commit与实际MSVC19.50/CUDA13.3/RTX4070/driver617.14、逐stage/log/DLL SHA均保存。
两个生产树真实安装/rename迁址，原prefix消失且Hyle metadata无source/build/staging路径；普通
hyle::graph消费者各七个独立安装头、原BF16/Q8搬运/Direct/Graph oracle通过。两native头按SKU
缺席，Apple/Vulkan package路线请求均明确拒绝并列出cpu;cuda。static初scratch文案匹配遗漏
实际does-not-contain诊断，原负向command已正确拒绝；同prefix/log只继续未运行门，事实保留。
两consumer在仅consumer/System32/Windows的PATH下数学PASS，DLL同目录及hash保存。更新后的
原source/static消费者也实际PASS；backend archive/共享DLL确认两个test hook setter缺席，真实
copy implementation/public资源符号存在。其他平台/路线SDK、selector扩展/终审仍待验。

## Windows CPU/Vulkan source/static/shared

[Vulkan SDK收据](../../../../evidence/correctness/foundation-f2-windows-vulkan-sdk/receipt.json)按5b3efe2cc全源码/未改变的
Vulkan product scope取证。实际RTX4070/NVIDIA617.14、software=no；source复验和真实迁址后的
static/shared ordinary hyle::graph消费者均通过两组/两槽/三轮原BF16/Q8搬运与Direct/Graph oracle。
两SDK各七头独立编译（六common＋Vulkan ON）、CUDA/Apple头缺席和未构建路线的package请求
明确拒绝；consumer/System32/Windows的限定PATH下成功运行，DLL同目录及hash保留。
两树实际Ninja #deps22包含batch_transfer.h。静态树初始C/C++中文前缀与英文cl输出不匹配，
主动停止验证，原日志保留；只修生成cache并清理核验的build/src对象（含封装ggml生成对象），
外部依赖缓存与产品源码未改。C语言覆盖全局prefix的第二个cache事实也保留，恢复后才签上述门。
源码/头/SDK资格不能代签未运行native unknown-fence或其他平台，当前四owner仍CANDIDATE。

## MacBook Air CPU/Apple installed static/shared

[Air SDK收据](../../../../evidence/correctness/foundation-f2-air-apple-sdk/receipt.json)绑定5b3efe2cc完整canonical Git archive/
tree/SHA，实际Mac16,12/M4/16GiB/macOS27.0.1/AppleClang21。两原production树单线程更新，
install后真实rename迁址，旧prefix消失，Hyle metadata无source/build/staging路径。
ordinary hyle::graph消费者实际原BF16/native affine Q4两组两槽三次搬运及Direct/Graph scalar
oracle PASS；两SDK各七安装头独立编译（六common＋Apple ON），CUDA/Vulkan头及package请求
均明确拒绝，available=cpu;apple。consumer MLX metallib与安装资产SHA匹配，并参与实际GPU
数学运行。此格不代签Air Vulkan或尚未运行fault/selector扩展/终审，四owner仍CANDIDATE。

## MacBook Air CPU/Vulkan native/source/static/shared

[Air Vulkan收据](../../../../evidence/correctness/foundation-f2-air-vulkan-qualification/receipt.json)绑定同一5b3efe2cc完整archive/
tree/SHA与Mac16,12/AppleClang21实机身份。9项VulkanCopyRecording/Program和34 common PASS，
包括原native producer、已接受前缀的后缀失败、旧ready/放弃回收及actual copy边界。普通CPU/
Vulkan source/shared与真实迁址后的installed static/shared消费者均两组两槽三轮BF16/原GGUF Q8
Direct/Graph scalar oracle PASS；实机Apple M4/MoltenVK1.4.1、software=no。
两SDK各七安装头/Vulkan ON，CUDA/Apple头与package请求明确缺席；旧prefix消失，Hyle metadata
无source/build/staging路径。共享consumer不要求Windows式DLL落地，实际DYLD_PRINT_LIBRARIES
在移除DYLD_LIBRARY_PATH/FALLBACK override后证明24条HYLE库全部来自relocated/lib。
CPU/Apple共享包也补同样实际加载取证，23条HYLE库来自其relocated/lib，见[Apple SDK收据](../../../../evidence/correctness/foundation-f2-air-apple-sdk/receipt.json)。
上述首轮故障/交付不扩大成未运行unknown-fence或最终七门；WSL Vulkan及扩展/终审仍待验。

## WSL当前Vulkan资格与适用installed交付

[当前资格收据](../../../../evidence/correctness/foundation-f2-wsl-route-qualification/receipt.json)实际调用系统Vulkan loader枚举physical devices，
仅llvmpipe/LLVM20.1.2、CPU device type4；没有VK/MESA路线override。NVIDIA CUDA设备可用并不
证明同一WSL的Vulkan GPU可用。本批GPU数学/native运行记NOT_RUN_NO_QUALIFIED_GPU，不签PASS。
Windows RTX4070和Air M4/MoltenVK的真实GPU证据保持独立；不以软件设备替代它们。

[适用SDK收据](../../../../evidence/correctness/foundation-f2-wsl-vulkan-sdk/receipt.json)绑定5b3efe2cc完整canonical源码/tree：复用原CPU/Vulkan
static/shared production树，单线程构建后install并真实rename，旧prefix消失，Hyle metadata无
source/build/staging路径。两普通hyle::graph消费者实际编译链接PASS；各七installed头独立编译
（六common＋Vulkan ON），Apple/CUDA头明确缺席，两个未构建路线请求各明确拒绝并列出cpu;vulkan。
ldd无missing，shared的HYLE库全部实际来自relocated/lib。该收据只签适用交付，不签GPU数学；
source embedding普通consumer构建也已补PASS，同收据保存完整configure/build/ldd；没有运行
软件设备数学。selector扩展与P0终审仍待验，四owner仍CANDIDATE。

## 固定产品后的原生故障补齐

[故障收据](../../../../evidence/correctness/foundation-f2-native-risk/receipt.json)绑定5b3efe2cc产品与8c27d8501测试点，产品scope不变。
Windows RTX4070及Air M4/MoltenVK各10项Vulkan copy PASS：原Device实际提交后持续装饰fence/
queue drain观察OOM，output保持pending、所有uses/metadata保活。普通持有分支实际等待线程
在unknown期间未返回，恢复原观察后才退休且保留首错；放弃分支原Context.close失败时pending
仍在，恢复原观察后才归零。两个分支都未发布失败ready或偷偷重放。
de11365a9首fixture误读blocking wait/pending合同，Windows120秒TIMEOUT保留；只修fixture，
没有产品修改、重置预算或放宽测试。Air同一测试文件SHA和实际主机/工具链身份已保存。

Air另8项Apple copy PASS：原MLX square/Checkpoint在真实GPU prefix后抛原生边界错误，原FIFO
中已排队copy归Internal且零native copy/不发布，已有成功ready仍按实际bytes可读；借用allocator/
resource保持到实际Jobs退休，所有uses/metadata/blocks最终归零。这是晚失败传播取证，不宣称
测试了每个Metal vendor硬故障。当前仍需selector扩展、P0七问和独立源码红队后才可CLOSED。
