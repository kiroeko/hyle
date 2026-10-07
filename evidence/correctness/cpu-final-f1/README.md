# CPU最终F1：部分证据

当前任务为[F1](../../../docs/implementation/tasks/compute/cpu-final-f1-build-contract.md)，
总目标为[三平台CPU最终收口](../../../docs/implementation/tasks/compute/cpu-final-closure-plan.md)。
本页不是terminal，不能签F1或整个goal完成。

## B1构建资格与可用性

- 产品`adc794aa3384499d23b6c9b054570b017b236fae`：flags/IPO/linkage等不匹配撤旧WIN，成熟数学保留。
- 私有构建metadata合同20格实际通过；后续`688d28f15361ed33de1c528fd865377c1742749c`修正CMake编译器尾`.0`写法，
  同20格重新通过。该metadata检查不冒充真正compiler/CPU运行。
- Windows shared source/public-facade的24格Direct/Graph、24成熟region、两次变输入replay、9错配、
  独立并发槽/close及真实动态view物化实际通过，Native WIN标签0。
- [receipt](b1-receipt.json)、[原始configure/build/run与普通consumer](b1-build-default.zip)。
  没有新数学或新赢家P1，未签新的shared性能优势。

## B5 oneDNN升级

采用[官方稳定版v3.13.4](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13.4)，
固定commit`c531fe1a5cd631b7f60277cdf6675c588cadbc28`，source/archive/Git tree身份已核对并进入中央锁。
11个具名seam中C API/BRGEMM/threadpool/stream/primitive运行时及LICENSE相同，
C++ header增加verbose查询、options增加默认OFF的SYCL profiling，原资源overlay仍精确匹配。
两adapter的版本由真实header/同一CPU版本owner投影；旧Native17/187不自动沿用新版oneDNN。

产品`688d28f15361ed33de1c528fd865377c1742749c`编译成功，Windows shared owning四组最终通过。
初次四组中组合后备测试仍要求已退役Native MatMul，原失败保留；测试候选
`d2ffda17793cfef84eb9be2a29652aee8eb198f5`迁移为ATen真实转置矩阵两次变输入执行，
显式裁剪ATen时检查明确拒绝，不复活自研。修正后四组、24格普通Direct/Graph与动态/并发/close通过。
[升级receipt](b5-receipt.json)、[原失败/纠正/新源码身份及完整日志](b5-onednn-upgrade.zip)。
最终三平台三交付未完成，不签整包PASS，没有运行赢家P1。
其它库锁与GPU产品没有改变；中央lock路径产生的selector全库superset原计划保留，
无oneDNN依赖的目标有语义不适用记录，不把未运行全库suite写PASS。

Mac Air已读取当前真实身份且可达，尚未运行新版CPU产品；macOS覆盖仍未资格。
当前统一Compute+Graph实现**61,084行/249文件**，不含测试/vendor/bench；新增版本身份胶水，新增自研数学0。

[清理记录](cleanup.json)：已校验归档与锁定cache后，删除本轮重复oneDNN源码/下载约87.8 MiB；
保留当前producer、统一锁定source/archive cache与全部原始资格记录。

## B4逐合同工程默认

候选`6216e6d0651ad2684950abec036d8749ccc3c3a1`，Windows Direct16/Graph15无skip、无fail；
原24格/变输入重放、完整region、错配/并发/close与动态物化通过。
[receipt](b4-receipt.json)、[原失败/纠正与完整日志](b4-engineering-defaults.zip)。
原库能力/配置/生命周期及PreparedVTable保留，静态cold表不增加Context allocation；expert不带普通默认policy。
这是工程默认，不是新性能WIN；其它平台和最终交付未验，本包仍未关闭。

## B2同core原CPU端口

[原port探针receipt](b2-probe-receipt.json)、[11文件/19接缝manifest、布局与MatMul/Gather日志](b2-original-port-probe.zip)。
旧core保持既有enum数值及tensor336/plan56，只添加CPU原IQ1 traits/编码；三格式不同row/scale实际输出通过。
无自研数学、无双core、无新GPU格式资格。产品生成器与实际探针的派生manifest逐项一致。
CPU+Vulkan标准产品组合仍在configure/build验收中，原失败保留，本页不把探针写成该组合PASS。

归档同时含官方Mac ARM64 LibTorch2.8候选库存：原包71,787,762字节，SHA
`be0642231d2008c7b90071fc0898d82a0c002d47f38350ebf3ed779fc01972d7`；
版本标签`2.8.0`与Win/Linux的`2.8.0+cpu`不同。只有获取/身份盘点，尚无HYLE Mac CPU资格。
当前统一实现LOC为**61,242行/251文件**（新增cold默认与身份胶水，数学新增0）；本节覆盖前面的旧子批计数。

## B2 Windows实际混合SDK子批

候选`afba74122c26d49489bee895c8b14dc5381426d2`的CPU+Vulkan标准shared源码组合已完成：
configure52.297秒、build611.562秒；CPU原格式owning、Direct默认16项、24格两入口/成熟region/两次变输入重放、
动态物化/并发/close通过。原GPU九项回归通过；十一种新增CPU格式没有取得GPU资格。
[混合子批receipt](b2-mixed-receipt.json)、[原配置错误/中断及成功日志](b2-mixed-windows.zip)。
具名GPU新格式拒绝测试已加入但在此receipt时尚未运行；不将该未验项写PASS。
这只补签Windows源码shared组合，不代表CPU+CUDA/Apple、安装static/shared或其它平台已完成。
Mac当前资格见F3，前文候选库存记录保持其原时间边界。

### B2负空间实际纠正

新增测试最初FAIL：Vulkan原格式映射拒绝CPU新格式，但布局domain绕过映射并错误准入。
候选`b2ad8084b4fcc996301e45426413817e62c27b8f`只在普通/分组矩阵domain复用原映射门；
十一CPU新格式两种矩阵拒绝通过。全fixture25项中24通过、0失败、1个显式validation-layer实例测试skip，
该skip未签PASS、不影响本次不依赖validation-layer的格式拒绝结论。原GPU数学未扩张、无P1。
[负空间receipt](b2-gpu-format-negative-receipt.json)、[原失败与纠正完整日志/XML](b2-gpu-format-negative.zip)。

## B2/B3 Windows CPU+CUDA source

Source `e0388fbda7fec39e98bee0757f5ab4f91f1cc625`, corrected test
`b041e792194ec1d1fd3367855d58d14562e6b11e`: CPU owning fixture, Direct20/Graph19,
40 ordinary numerical checks, four original GPU/rejection checks, six native
CPU-program/ISA/resource checks and eight finite Graph variant/cold-rule checks
passed without skips. The original GPU math remains unchanged; eleven new CPU
formats are rejected for both GPU matrix operations.

The first build failed in the new test because descriptor preparation lacked its
explicit resource argument. The original 1253.719-second log is retained; the
resource correction and remaining 29 steps took 18.703 seconds, reusing all
compiled dependencies. [Receipt](b2-mixed-cuda-source-receipt.json) and
[original failure/corrected logs, XML and drivers](b2-mixed-cuda-source.zip)
bind both candidates. Mixed installed static/shared and Apple still await
qualification; no F1 closure or performance WIN is claimed.

### CPU+CUDA installed source/static/shared completed

The same producer completed all 15 delivery steps. Both shared/static source
consumers and both relocated installed consumers passed the same 40 numerical
checks. Installed execution used only Windows system directories in PATH;
Torch/c10 and the exact `hyle_ggml_cpu_avx2_456172ec_...` program were beside
the executable. Actual shared export hygiene passed. The ggml GPU and oneDNN
dependencies were reused; shared/static increments took 124.594/192.500 seconds.
[Delivery receipt and runtime layout](b2-mixed-cuda-delivery-receipt.json),
[logs, consumer/driver and final build/install identity](b2-mixed-cuda-delivery.zip).
Apple and Vulkan installed mixtures remain pending; this does not close F1.

## B2/B3 Mac CPU+Apple three deliveries

Candidate `e0388fbda7fec39e98bee0757f5ab4f91f1cc625` completed 27 steps on
MacBook Air ARM64: CPU owning fixture, Direct16/Graph15, five original
Metal/rejection checks, six native CPU-program/resource/ISA checks and eight
Graph variant/cold-rule checks passed without skips. All five source/installed
consumer runs passed their 40 numerical checks; relocated shared/static loader
inspection and shared export hygiene passed.
[Receipt and immutable source identity](b2-mixed-apple-three-shape-receipt.json),
[logs/XML, consumer and exact drivers](b2-mixed-apple-three-shape.zip).
Initial/shared/static builds took 130.598/13.927/73.534 seconds; no P1 or new GPU
format qualification was run. Vulkan installation and the Linux mixed old-core
loader delta remain pending, so F1 is still open.

[Transfer cleanup](cleanup-mac-mixed-transfer.json) records removal of the two
SHA-verified duplicate Apple source tar files (496,189,440 bytes). Their exact
Git candidate and SHA remain archived; validation sources, SDK prefixes,
producers/cache and raw results remain. Reproduction can regenerate the archive
from the recorded canonical Git revision rather than retaining duplicate tars.

## B2 Windows CPU+Vulkan installed delivery completed

Candidate `7060cf257e1f8e90a1cd0a79e837f40fb9eaa034` completed 15 steps;
shared/static source and relocated installed consumers each passed 40 numerical
checks, with system-only PATH and Torch/c10 plus the exact old-core CPU-program
DLL beside the executable. Shared export inspection passed. Original ggml/IREE
dependencies were reused; shared/static increments took 144.266/187.297 seconds.
[Delivery receipt and layout](b2-mixed-vulkan-delivery-receipt.json),
[logs, exact consumer/driver and final static build identity](b2-mixed-vulkan-delivery.zip).
The earlier Vulkan validation-layer skip is unchanged; this CPU SDK batch does
not qualify that test, all GPU formats or P1. The Linux mixed loader batch is the
remaining F1 platform delta, not yet PASS.

## Linux old-core mixed loader delta completed

The corrected input `a3b621ca85853164ce48d133b2fde06c1fca1c0b` passed 21
continuation steps: six original CPU-program/ISA/resource checks, four original
GPU/rejection checks and all five source/installed consumer runs with 40
numerical checks each. Relocated shared/static loader checks found no missing
dependency; shared export inspection passed. Builds reused the dependencies:
93.859 seconds to finish the corrected source batch, then 50.586/135.019 seconds
for shared/static increments.
[Receipt and current immutable source identity](b2-mixed-linux-three-shape-receipt.json),
[original failure, correction, driver errors, numerical XML and delivery logs](b2-mixed-linux-three-shape.zip).
The original GCC warning failure is retained; the only product delta is equivalent
parentheses in the existing CUDA integer-role guard. Windows summary encoding
and continuation-variable errors are also retained; passing source tests were
not repeated for those driver repairs.

F1 is CLOSED within its build/precompiled/P0 scope. Its CPU product candidate
remains `560eb1d2332bb69ee5148e899b000fae85d80074`, with the separately audited
equivalent CUDA syntax delta bound to Linux evidence. F4/F5/F6 and old GPU/Graph
states remain outside this closure. Current implementation LOC: 61,246/251 files,
excluding tests/vendor/bench; no new self-written math or performance WIN.
