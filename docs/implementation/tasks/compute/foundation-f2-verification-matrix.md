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
| Windows CPU/CUDA | PASS（scope匹配；shipping更新后复验） | source/relocated static/shared、ON/OFF头、absence、DLL同目录；8 CUDA/34 common与适用实际late failure |
| Windows CPU/Vulkan | PASS（scope匹配） | source/relocated static/shared、新头/absence/DLL；9 native/34 common及实际unknown-fence风险 |
| Air CPU/Apple | PASS（scope匹配） | source/relocated static/shared、新头/absence/MLX assets；10 native/34 common及实际故障/关闭 |
| Air CPU/Vulkan | UNPROVEN | 实际设备身份；对应native/common与ordinary数学、三交付/头/absence/迁址 |
| WSL CPU/CUDA | PASS（固定候选source/static） | 8 CUDA/34 common及原backend late failure PASS；installed static/shared、native ON头/absence/迁址、F2 producer actual late-failure仍待验 |
| WSL Vulkan | GPU资格UNPROVEN | 先实际枚举loader/device；只有合格GPU才签数学。若仍仅llvmpipe，则记录本候选条件结果，完成适用构建/安装/头/absence格 |

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
及installed SDK仍UNPROVEN。原backend late-failure不是F2 producer late-failure证书。
