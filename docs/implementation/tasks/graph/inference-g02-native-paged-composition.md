# G02 CPU 与 Apple/Vulkan 精确页联合消费

状态：SCOPED，2026-10-14。当前 owner 为 graph_route_cpu_apple、
graph_route_cpu_vulkan；baseline `66af56852d0602ce14cdfde9aba3de373a7472c6`。
授权来自[十项 goal](../project/inference-gaps-goal-execution-2026-10.md)。原
[Foundation A](foundation-a-final-closure.md)有限域与失败证据保留；本卡不重开旧数学、
effects、普通 native Graph、scheduler 或资源桥。

## 触发、消费与边界

真实触发是已准入 G02 的 CPU→原生 page16 attention→CPU 旅程。G01 新 native
PagedBatchInvocation 已实现精确页读取并完成 actual Air 数值、原生风险和三交付；
其产品候选为 `520027a57`，性能仍在独立固定副本验收。本轮不修改该候选。共同
LeaseSubmission::bind_pages 已在[原 G02 卡](inference-g02-multi-queue-batch.md)验证。

当前两个联合 Invocation 公开面只有 ConstTensorView、单 ReadLease、native tensor
和 weight binding，不能传入 PagedArenaRead。编译器仅将普通节点切成原 CPU/native
fragments，原 native fragment 不接 owned layout/页表。把整 arena 投影为普通 readonly
tensor 会扩大权限，且不能证明数学只读取获准页。这是新消费需要与现有形状之间的
可复核缺口，不是对已关闭普通域重看一遍。

主消费者为 Qwen3.5 GQA8/2 D256 attention 片段：CPU query 数值预处理→实际 Apple/
Vulkan 两请求 page16 attention→CPU Cast/Neg，含单请求、混合 prefill/decode、乱序/
尾页/共享页和旧结果。正交消费者为非模型的两个只读分页数值域、独立 slots 与真实
CPU 前后段，防止以单一路线、raw pointer 或仅 GPU 运算冒充联合。

完整 public journey 必须让用户沿原 GraphBuilder、compile、create_invocation、绑定
确切 frames、prepare_replay、submit/wait、结果 owner 与 close 完成。图节点只表达
数学/固定形状，冷 binding 拥有 schema/recipe/layout；用户不生成 raw metadata tensor、
不写第二套提交循环、权限退休或 CPU/GPU scheduler。请求、cache 调度与模型政策不下沉。

范围是 owned 精确 layout 的 native fragment、公开 page binding、实际 Q/输出跨 CPU/GPU
边界、guard/generation/失败/取消/旧结果和实际退休。至少沿 G01 当前 Apple F16/BF16、
Vulkan F16、D64/128/256/page16 数学域验收。Vulkan BF16 全链继续由原 G05/G07 处理；
CUDA/CPU 与原 G02 未完成门保留，不能由本卡缩减全 goal。一般 shape、capture、producer、
delta planning、量化 KV 与重叠流水仍对应 G03–G10，不冒称本轮已经兑现。

## 承载审计与下一步

BORROW 原 CPU/native fragment、Context/Queue/Copy、G01 单 table/两 bank 共同 controller、
原 ResultSlot 和 MQ-LP guard retirement。SELF 只冷 graph node/owned layout 映射、准确
frame 资格与原 fragment 接纳，禁止新增数学、controller、executor、queue 或 whole-arena
permission。常态普通 nonpaged 路线的 binding、native output topology 与 alias/effect
预检保持其原合同；页专用承载应 opt-in，不为每个 ordinary value 分配最大页表。

原 CPU+Vulkan 在 prepare_replay 绑定并 seal 原 native 输出 backing，drive 检查结果
backing 不变；接入页 controller 必须核清该输出与后继的所有权，不能删掉原检查来
取得 PASS。Apple 原 drive 在 CPU prefix 实际完成后再绑定原 native inputs；两者保
不同的合法原生行为。失败或取消只有实际 prefix 退休后才释放所有 referenced guards；
未引用页始终独立可写，页输入不得发布或流向普通 dense/CPU 操作。

图内页节点/冷绑定的共同 owner 预计为 graph_batch。其原 MQ-CPU candidate `a0c047ea`
当前未重开；必须先以新事实合法回到 AUDITED、明确该共同 API/依赖和有限 FIX，才可
修改产品。两个本卡 owner 先进入 SCOPED；公共形状、每段 owner 与有限 FIX 还没有
最终冻结，不写产品代码。下一步只完成该接缝的有限源审计并进入 AUDITED。

## 预声明验证、预算与停止

P0 核实际 CPU→GPU→CPU 原 math/copy/完成事件，精确页集合和 host wait/table/native
private UNKNOWN；不重跑 G01 已关闭 CPU/ragged P1。若 API/算法固定后发现独立的常态
成本问题，须明确同层同合同片段再冻结定向 P1，不能从 benchmark 行改变公共形状。

源/relocated static/shared、逐公开头、actual Air 两路线联合数学与对应宿主 ASan/UBSan/
TSan；沿已声明 Metal TSan 同步边界模型与 ASan 无 LSan 限制，不把宿主插桩当 shader/
driver 资格。Windows Vulkan 定向编译/链接；本机重型 GPU run 仍不恢复。适用 unit 与
原 cpu_apple/cpu_vulkan contracts 直接负空间，selector 保 selected/skipped reasons。

必须覆盖 foreign schema/resource/generation、重复/共享 pages、空域、允许的 topology
变化、两 slots/旧 output、每个引用页防写、未引用页可写、冷预算/逐分配 OOM、CPU/native
失败、accepted prefix/未知完成、取消/放弃与 close。所有未运行门 UNPROVEN；不能
用共同 helper 或 G01 direct 输出代签新联合消费。CPU prefix 与 GPU fragment 都须实际
运算并由独立 oracle 核输出，必须检查原图其它消费者拒绝扩大页权限。

源审计/实现/验证各2小时检查点，按用户常设授权有限延长并保所有 FAIL/elapsed/累计；
重型串行、四 jobs 上限、不运行本机全模型。停止按固定 FIX 与适用门终态，或真实
外部阻断/协议回退；不因短观察超时重启工作。docs/closure 每轮适用检查，无产品修改
的本 SCOPED 登记不构建、不跑性能。
