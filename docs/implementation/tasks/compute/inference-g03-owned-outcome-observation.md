# G03 原结果槽的来源聚合观察接缝

状态：AUDITED，2026-10-14；owner compute、compute_route_apple、compute_route_vulkan；
baseline `16d677b38db855482ada49ed260a3611f5c17147`。这是[主卡](inference-g03-pending-page-consumption.md)
的实际来源/后继结果合同，不修改已闭数学、原取消资格检查、公共构建或工具。

触发：原 attention 的成功 completion 不包含 source 的取消/错误。
[SCOPED 实际原型](../../../../evidence/architecture/inference-g03-owned-outcome-owner/README.md)已在
Apple/Vulkan 实测原 source owner + 原 OwnedResult/ResultSlot 的 move-only 组合：
成功保同一 payload/旧结果禁止 reuse；后继接纳之后取消 source，聚合拒绝发布结果；
abandon 原 completion 仍保 source 至实际退休。Vulkan 还实测 GetFenceStatus OOM
时 guard/source 保留，实际最终 wait 保第一 OOM，并验证该未知状态下 abandon。
没有产品修改、没有第二 executor/queue/result slot。原型只是 shape/lifetime 反证，
没有独立数学 oracle/64 步/完整七路线/公共 pending frame 证书。

真实消费形状另有硬约束：原 CPU+Apple/Vulkan graph invocation 的 native fragment
先 await_completion(submitted.completion()) 再 submitted.wait()。只在新 façade 的
result()/wait() 上合并来源，仍让 completion() 报成功，会丢失该真实消费者的合同。
因此产品须把聚合接到原 Completion/OwnedResult 观察入口，保原 storage/payload，
而不是让 consumer 再写一个 wrapper 或复制 Graph controller。

普通数值 producer→range copy→readonly reduction 是正交任务；与 KV/page/request
政策无关。内部接缝准入为 SELF，复用原 CompletionRetention/RetainedCompletion/
ResultSlot/F2 accepted phase/native actual-retirement callbacks。原 update controls 仍
唯一观察；borrowed prefix 不得在 suffix 实际退休前由新 observer 失效。

范围：一个 opt-in 冷观察槽；原 OwnedResult 的 trusted completion 替换/retention
接缝；原 F2 已接纳各 phase 的实际退休订阅。普通对象 layout/队列/allocator/锁/原子
写和数学不变。所有来源错误与 suffix 错误须保第一原状态；unknown 不伪造退休。
实现前固定有限 FIX/API/冷热费用，SCOPED 不授权产品修改。

预声明：四原 source family、实际 source→suffix 的成功/接纳后失败/重复观察/old result/
两 slots、unknown+actual wait、取消/abandon/close、原 prefix 寿命；无 warm 新分配；
逐 cold allocation OOM；相同 completion()/result()/wait() verdict。适用 source/static/
shared、相关独立头、宿主 sanitizer 与实际直接 Graph 消费入口。CPU/CUDA 只用 tiny
有限载荷，不重建 nvcc/kernel/全模型；Apple only 实查 Air；Windows 定向编译。

P0 先记已有 native 生命周期、冷槽、retention fan-out 与 phase scan 的真实费用；
没有新的数学/normal algorithm，不重复旧 P1。检查点调查/实现/验证各 2h，沿用户
常设授权有限追加，原 FAIL/TIMEOUT/累计不重置；heavy 串行/最多四 jobs。所有门
按实际影响裁决，未运行不写 PASS。本接缝完成也不代签主卡 pending page/late native
output 或整个 G03/G01–G10。

## 冻结接口与有限 FIX

| ID | 固定改动 | 验收 |
|---|---|---|
| OO01 | detail::ResultAccess 提供原 OwnedResult 的 mutable completion 与 move replacement，仅 trusted adapters 可用；不动 storage/payload/layout | 同一 result identity/旧结果阻止原 slot reuse，completion()/wait()/result() 同 verdict，无 const_cast |
| OO02 | 三 CopyStreamAccess::retirement_sources(Update&, span<Completion*>)：空 span 查数量，非空填原已接纳 phase 与实际被这些 phase 使用的 producer completions，去重；不观察、不等待、不替换原对象 | 四 source families；未使用 producer 不加入；inline/partial-prefix/invalid capacity/原 identity 与借用寿命 |
| OO03 | 一个 opt-in DependentCompletion 冷槽，合并原 suffix completion 与唯一来源 owner 的 poll/wait/cancel/actual-retirement 协议，输出仍是原 Completion | unknown 不提前裁决/释放；source与suffix第一错误、后继接纳后取消、abandon、重复观察、冷槽复用及 OOM |
| OO04 | 原 OwnedResult completion 接入该冷槽，原结果 payload/slot保留；原 Graph await_completion 消费同一聚合状态 | Native Direct/原 Graph 形状、四 source/交付/宿主风险，不能把原型 façade 当产品 |

OO02 签名为 `Result<std::size_t> retirement_sources(CopyBatchUpdate&,
std::span<Completion*>) noexcept`；只借用原对象，调用方保原 update控制，不能另作观察。
数量是实际 accepted phases＋这些 phases 使用的不同 producer bindings，不用 report
中的 prepared phase 数代替 accepted 数，不收无关 bound producers。节点存储在冷
source owner，容量不足在填表之前拒绝；源 known error/unknown 必须原样保留。

OO03 的 private `DependentCompletion::create(AllocatorRef)`、`available()`、
`publish(Completion&&, OutcomeSources)`、`reset()` 固定。OutcomeSources 是同构建
trusted C++协议：state 与 poll/wait/request_cancel/abandon/subscribe_retirement/release
函数；成功 publish 消费一个源 owner reference。来源负责原 controls与有限节点，
subscribe_retirement 在全部已接纳原 source work实际退休后交回一个 CompletionLifetime。
poll/wait 必须推进原 source观察，错误不等于退休；source观察只能在 suffix实际退休
后进行。abandon 不等待、仅移交原 native owner保活，释放/订阅均由原 actual callback
完成。不能在 native callback 中 blocking wait 另一个 callback。

冷槽及来源节点先由注入 allocator/ResourceBudget 建立并逐分配 OOM；publish/admission
不分配。普通 Completion/OwnedResult/F2/native controller layout不变。只增加 opt-in
槽的 refcount/retention/actual flags，复用原 CompletionRetention 与原后端进展，不
创建 executor/queue/计时器、shadow输出或默认 native callback 字段。native借用与
状态观察外部串行，actual callbacks只发布对应退休事实并释放所有权。

允许 result.h/generated_result.h、retained_completion.cc 与新增 detail观察头、三原
CopyStreamAccess/copy实现、直接tests/guide/evidence；不改公共构建/closure工具/数学。
现有 source 元数据、原取消资格检查和 original ready binding保持。pending page/
late producer/public receiver 仍主卡；本有限FIX处理它们需要的真实观察合同，非另增
全库研究项。源／suffix已接纳之后的新能力不顺手扩入。

七问裁决：无需新依赖/FFI；两个具名纵切和真实 Graph completion消费；借原 native/
F1/F2/结果槽；P0 有界 phase/producer扫描与冷固定元数据；三交付原签名/头成立；
guide须说明Actual retirement与唯一观察。验证门未完成，不把 AUDITED 写成 PASS。

## 2026-10-14 实现检查点

OO01–OO04 的原 completion 接缝已写入产品，并通过
[实际检查点证据](../../../../evidence/correctness/inference-g03-owned-outcome-observation/README.md)：
Windows29 core/result；Air Apple3/Vulkan5 原 native lifecycle＋CPU byte oracle；
六组 host sanitizer；四组 relocated static/shared SDK/16 独立头/每组64 episodes；
实际 CUDA1 KiB 原 producer/F2/suffix 四案例，无 nvcc/kernel rebuild。
没有第二 executor/结果槽，没有普通对象字段改动；两个 cold observer slots 的64次复用
没有额外 metadata allocation。原 Graph 的 completion→wait 消费形状使用同一个原 OwnedResult。

本卡保持 AUDITED：原 partial accepted-prefix/used-producer failure 对抗 fan-out、生产
source owner 冷预留/逐节点 OOM及实际 Graph 集成尚未完成；测试 source owner 的每次
make_shared 不代签生产 warm 成本。主卡 pending page/late native backing/七路线64-step
仍未完成。窗口内 Windows 首编译测试 nodiscard FAIL 保留；各构建累计及原限制见证据。
下个实现动作只补这些固定验收及主卡准入形状，不重跑旧数学/P1或重开全库研究。

### 来源冷预留的固定内部形状

OO03 的来源节点使用 `detail::CopyOutcomeSource<Update, Access>` 冷槽，四 source family
仍移动原两 controls。`create(capacity, allocator/ResourceBudget)` 在 native 接纳前建立
状态、原 Update payload、Completion* 表和 retention 节点；有限四次冷分配逐一 OOM 回滚。
`take(Update&&)` 在 suffix 接纳前校验原 dependency、原数量与容量；失败不消费 Update，
成功返回一个 OutcomeSources reference。trusted receiver 通过冷槽中的原 Update 取得
原 input/prefix/guards；其后用原 DependentCompletion.publish 交接该 reference。
后继拒绝时 release 未订阅 reference，移交原 source controls；无另一个执行器。

`available()/update()/reset()` 是同构建内部 ownership 接缝；available 只有旧 protocol
reference 和所有原 actual-retirement 节点都释放后才成立。正常 publish/admission 不分配。
source 检测 known tail/dependency 失败在接纳前拒绝，未知或接纳后 producer 错误仍由
原完成与实际退休 fan-out 合并；不增加 F2/native 常态字段。类型 shell 只负责原 Update
的构造/移动与回调，节点扫描、订阅、观察和回收放入原 retained_completion.cc。
本形状是 OO03 已固定来源协议的冷 ownership 实现，owner/FIX/产品边界不扩张。

正常完成保原 retired Update 在冷槽，调用方经 update().release_invocation()/take_mutable()
回接原冷 controller/writers；不能每步销毁原 invocation 再重建来声称 warm 零分配。
DependentCompletion 只在来源尚未实际退休的 abandon 路径调用原 native handoff；全部
实际退休后的 completion settle 保留原控制的成功/失败状态，protocol reference 释放
后可回接。cold source handle 销毁仍按原 native 寿命清理，不额外等待。

## 2026-10-14 来源冷预留检查点

[实际新证据](../../../../evidence/correctness/inference-g03-cold-outcome-source/README.md)已补上
production source冷预留/四次OOM回滚、ResourceBudget基线、used producer unknown fan-out、
原CPU两invocation/two source slots64步固定分配次数及byte oracle；Apple3/Vulkan6原
native lifecycle（含实际partial accepted-prefix/tail拒绝不消费）与CUDA1KiB四案例；
六host risks、四relocated SDK/20头/每配置source64＋result64、Windows36均通过。
正常完成保原控制回接，pending abandon才交原native owner，移除当前消费者每episode
shared source分配。原窗口失败/累计和前后修正完整保留，不重置预算。

仍 AUDITED：实际 Graph 集成和原独立reduction消费者尚未验收；主卡pending frame/
late output/七路线64-step不由这组冷槽证据代签。下一步推进这两个真实组合入口与
主卡的有限产品形状，不再重复已通过的冷槽/数学门。
