# F1：外部资源租约与最后使用后回收

> 状态：AUDITED（2026-10-11），当前先实施 F1-04 的有限 completion 资格修复；其他 F1 接缝继续逐 owner 绑定。
> component / owner：拟由 compute 负责共同控制；compute_backend_cpu/cuda 与
> compute_route_cpu/cuda/apple/vulkan 负责实际 native 适配；Graph 及其四路线和 CPU+GPU 适配只改 leased 接缝。
> 具体改到的 owner 才在启动时绑定；不重开全部 Provider/算子模块。

## 当前启动绑定

- 实际启动 HEAD / previous registry ref：`833e6844b8e0160ec3eadd3b1945aed577cdad84`，工作树干净。
  其产品代码与设计基线 edb271438 相同，保留后续文档/证据提交，不回退。
- 当前绑定 owner：`compute`、`compute_backend_cuda`，均由 CLOSED 合法进入 SCOPED。
  trigger 为用户新增的外部 lease 完整旅程与 `poll_slot→Completion::settle/destroy` 的具体源码适配疑点。
- 先做 F1-04 源协议/自然接口 AUDIT 并冻结有限实现；其他 native/Graph owner 当前继续 CLOSED，
  真正需要修改时先绑定本卡、SCOPED/AUDITED，不从其历史卡恢复旧施工。
- 未启动构建、故障注入、数学运行或性能；结果目前只有源码疑点，不写 FAIL/PASS。

## 共同执行合同

- 基线：`edb2714384037b12a3d1e69839aa895fea17cec1`；设计见
  [资源与流送设计](../../../architecture/compute-resource-streaming-design.md)。
- 本卡已作为 compute/compute_backend_cuda 的当前 AUDITED record；其他 owner 仍按 registry。
  本轮产品验证未运行。已绑定实际 HEAD/工作树，保留后续/用户修改；完成 AUDITED 后才改产品。
- 主纵切：量化推理的外部状态/权重范围及真实成熟数学消费。
  正交纵切：渲染资产流送和分块矩阵管线；不加入 token/layer/KV policy。
- 复用 Mem/Util/Sched/IO/Compute/native 完成，不增加复制 kernel、数学算子、模型 loader 或第二套 runtime。
- 全部产品接线集中后冻结候选，批量执行适用检查；小修只运行影响门，同身份证据复用。
- 四路线资源/程序/完成独立。Windows 为唯一产品真相；CPU Win/WSL/Air，CUDA Win/WSL，
  Apple Air，Vulkan Win/Air；WSL Vulkan 按实际 GPU 资格，不以旧例外代签。
- 三交付：source/installed static/shared 普通 Direct+Graph 消费、SDK 迁址、ON/OFF 新头、路线缺席诊断、
  Windows DLL 同目录。不让普通用户添加内部 factor/provider/stream 开关。
- 默认 P0：有界资源、热分配/提交/完成计数与成本结构；没有 timed 性能赢家承诺。
  出现需性能裁决的具体反例才在稳定候选上按原统一仪器固定一次同合同有预算 P1，
  未资格/未运行如实记，不能无限换姿势或升级仪器。
- build1/test1、重型串行、启动前确认内存余量；复用源码缓存/SDK 构建树，完成后只清本批临时产物。
- 每轮 selected gates、`check-module-closure.py`、`check-docs.py`；
  registry 变化带 `--previous-ref`。文档/状态阶段不构建/P1。提交推送用 `[skip ci]`。
- 调研纠正预算：至多一个具体反例的一次补读、两次有限修复/验收；新公共语义或边界变化必须重新定界。
  所有 FIX/适用门完成则 terminal；不能把研究/文档/旧 CLOSED 计作本轮产品完成。

## 需求与已审计边界

实际反例：vLLM scheduler 2758–2805 在请求结束/CoW copy 后按 fence 延迟释放块；
SGLang l2_transfer 103–129 对 layout/index 保活；当前 HYLE BufferLease 在析构时归还，
TensorView 不拥有外部 payload，Graph Results 只保护其原声明 owner。
准入的是“外部范围引用及 native use 归零后回收”，不是 cache/sequence manager。

正常任务：取得块→共享读或独占写→绑定真实 Direct/Graph→提交→释放业务 handle→
完成后回池；旧结果仍持有时不能覆盖；取消/错误仍有正确的实际退役路线。

## API / usability freeze

- 共同 lease 值只表达实际 backing/range/generation/读写权及资源 owner。
  typed native factory 接受 CPU/CUDA Buffer、MLX array、Vulkan Tensor 的受控拥有/导入路径；
  不把 MLX/VkBuffer 塞进共同万能资源或复制其 allocator。
- 读租约可共享，写租约 move-only；发布 ready 读结果显式且以已成功完成为前提。
  非重叠块可并行，重叠 root/别名不得通过不同 wrapper 绕过检查。
- 新 lease-aware Direct/Graph binding owner/wrapper 预分配有限 guard 槽，代理原 Invocation。
  bind 自动保留 guard，submit 自动交接完成；plain borrowed API 的对象布局/常態路径不变。
  不采用“给每个原 Invocation 都塞一张资源表”的实现。
- 外部 owner 根/Context/resource/allocator 寿命明确；受控导入不能证明未跟踪别名不并发写，
  不满足独占合同的导入只读或拒绝，不能用空 deleter 假装拥有。
- 默认容量有限，满时明确 ResourceExhausted/Unavailable；retain/release 成本与统计冷查询明确。
  close 停接纳并退役本 owner，不同步/关闭用户所有 Queue 或 device。
- 首轮原生资格先裁决 CUDA unknown observation 的保活疑点；不靠非 OK Status 推断已完成。

## F1-04 AUDIT：固定实现与资格范围

- 基线：`165a67d8e1d8067fcd752b88c735166304e5c550`，两 owner 已 SCOPED，工作树干净。
- 源码反例已固定：共同 Completion::settle 以空 attachment 调 destroy 后立即 release；
  若 CUDA 观察返回非terminal错误，destroy 再观察为 Pending，原 attachment 会被过早释放。
  CUDA poll_slot 当前把全部非Success/NotReady当终态，reaper也照此回收。
  这是源协议缺陷；原生故障注入资格尚未运行，不能宣称原生产GPU已触发过它。
- 只改共同 settle 的 attachment 交接与 CUDA poll/wait/destroy 的退役分类；
  复用 EventSlot 已有 terminal_hint、CompletionRetention、原 CompletionReaper，不增公共对象字段或第二线程。
- CUDA Success 可退役；NotReady 待定；有官方依据的设备致命错误单列；其他观察错误保存原码并保活。
  wait 异常只尝试一次原 stream drain；若仍不能证明退役，错误返回但 native owner 保留 attachment，
  不因 public handle 终结释放。普通成功路径不新增context probe或额外同步。
- 测试复用现有 HostStreamGate、OwnedCudaBuffer、ResultSlot 与 increment helper，装饰实际 native event query/wait。
  故障 hook 仅 HYLE_TESTING 的线程局部内部 seam；shipping 无 hook 分支/额外状态。
  验证 observer failure、wait failure/drain、abandon、首错保留、回收恰一次与实际GPU数学结果。
- 适用：compute common correctness与CUDA backend correctness/late-failure直接消费者；
  完整F1数学和三交付仍按F1-05/06/08集中资格，不因本修复就写整包PASS。
- 用户再次要求已有高性能模块复用及完整开发流程：本批先源码审计/固定改动，实施后分别做review、
  正确性/成本性能验证与红队；API/算法稳定后才启动适用的有界P1，未运行不写PASS。

### 内部复用映射（F1–F3 必须核对）

| 任务 | 现有承载 | 使用条件/成本 |
|---|---|---|
| block/offset分配与有限metadata | Mem RangeAllocator、Fixed/SharedPool、AllocatorRef | 复用范围与容量，不能重复实现树/空闲链；线程同步由实际owner负责 |
| backing/guard保活 | Util RefCounted、CompletionRetention | 独立业务ref和native use；不把ObjectPool的destroy当warm实例复用 |
| bytes与quota | ResourceBudget、QuotaAllocator、BufferPool | 去重同backing、old lease仍计入；Mem pool本身不等GPU completion |
| 冷catalog/IDs与空槽 | Container flat_map/flat_set/dynamic_bitset或已排序固定数组 | 按lookup/内存/批量条件选已有自然路线，不为了形式引入热hash或metadata |
| 排序校验与host copy | 已有容器算法/成熟byte copy和原native copy | 不自写SIMD复制kernel；需要数值转换则复用已接Provider/typed operation |
| 进展、背压、取消/关闭 | Sched Semaphore/Channel/Waker/TaskScope/blocking handoff，IO read_exact_at/write_all_at | 无第二event loop/threadpool；原平台capability和生命周期不放宽 |
| 冷设备/时间/诊断 | Platform、Time、Obs及原route facts/reports | 查询冷路径；普通submit不新增探测、全局registry或计时选路 |

### F1-04 当前产品进展

- 定向修复子候选 `392d4503675fa095c3a89cc0a90641b4d48651cf` 已接线；
  [Windows资格收据](../../../../evidence/correctness/foundation-f1-q0-windows/README.md)为插桩95、shipping93项通过。
- 完成共同attachment/native退役交接、CUDA未知观察保活/一次异常drain、首错保存、实际迟到失败与hook交付隔离。
  未增加公共对象/EventSlot字段，不新增reaper，复用原Mem/Util与native提交。
- 已做这份有限修复的源码review/P0与局部红队；整包API、三交付、其他平台/直接消费者与计时性能未验。
  F1 owner继续AUDITED，不标记整包CANDIDATE/CLOSED；下一步是F1租约与typed Direct/Graph接缝。
- 统一脚本self-check：Compute+Graph非测试68,284行/301文件；全库非测试163,511行/652文件。
  Graph aggregate子错误/实际退役组合在F1-06接线前必须定向验证，本子批不代签它。

## F1-01/02/03：共同 lease 核接口冻结

- 当前baseline `711afd13f9a11b3534a880b8ea9a3f626bda852d`；compute owner仍AUDITED。
- 新增opt-in `resource_leases.h`：LeasePool冷创建一块原MemoryResource backing，
  allocate返回move-only WriteLease；publish成功转换为可复制ReadLease；ReadLease::try_write
  仅在唯一reader、无native use、非只读import时转回写。subview只共享原block，不分配新控制节点。
- 元数据复用ConcurrentBlockPool，offset复用RangeAllocator；RefCounted持有block/domain，
  每block有generation、明确ready/failed与在途写计数。复制read/保活不增加热堆分配。
- 独立LeaseUsePool冷分配固定episode和每episode资源槽；begin→read/write加入→attach真实Completion，
  全部资源、索引及控制owner由CompletionRetention持到实际退休；一次batch可跨多个LeasePool。
  未接受work时ticket析构撤销，接受work后必须attach正确Completion，不能用裸Status丢掉prefix。
- 不同blocks独立，重复同block使用在batch中去重；活跃write不能重入，pending/失败write不得publish。
  lease-aware Direct/Graph适配再负责自动绑定/提交，不把plain TensorView变成拥有者。
- create在ResourceBudget按backing一次计账，元数据走同scope allocator；原资源/Context/allocator仍有
  明确借用寿命。typed readonly导入/native array/tensor桥在原route owner绑定后实现，不能伪造空deleter。
- stop/close不隐式全设备wait；显式host drain等待已接纳uses，旧readlease保留backing；
  callback只更新use/失败并退引用，继续沿原runtime，不新建线程或timer。
- 本子批只写共同核和有限所有权/容量/跨pool回归测试；四路线、数学消费、包交付/P1仍按后续FIX兑现。

### 共同核初验进展

`db93f2e069ba5b22098d2fba21bc0604ece01a57` 已新增bounded lease/use核，
[Windows初验](../../../../evidence/correctness/foundation-f1-lease-core-windows/README.md) common69/CUDA29项通过，
新头单独编译通过。只读导入/native typed桥/自动Direct+Graph绑定及全部适用交付仍未完成。
原borrow view保持原合同，不把当前手动use ticket样例当作F1-05/06自然入口已兑现。

## F1-05 共同Direct适配冻结

- baseline `78686e9fe3907df648496563db4db13038dc0200`；compute owner AUDITED。
- `LeasedOperationInvocation::create(prepared, WriteLease workspace, options)` 冷创建原OperationInvocation、
  按原input/output数量预置view数组和一次use episode容量；不改变旧Invocation布局或Provider程序。
- 输入绑定拥有ReadLease，或显式在现有独占WriteLease下读取；输出借用WriteLease仅至submit返回。
  submit先取全部views/校验端口，再加入use guards，调用原submit并自动attach真实Completion。
  用户不再手写begin/add/attach，也不把错误输出publish为有效readlease。
- workspace由适配持有并加入write use。accepted prefix的有效Completion是原接口不变量，
  attach预留好metadata，无热分配；普通submit拒绝时ticket自动回退，不毒化未接受的write。
- 同block输入/输出采用独占writer use去重，原Provider alias准入照旧；不为简单示例撤销in-place能力。
- CPU/CUDA共有Direct路径先集中接线；Apple/Vulkan各自原生绑定与owned结果不经此Queue适配，另绑定原owner。

## 有限 FIX 集

### Direct / 拥有导入初验与 Graph 启动绑定

- 共同产品子候选 `5a38cc09b`：Direct 自动保活、拥有 Buffer 接管/永久只读导入已实现；
  [Windows 初验](../../../../evidence/correctness/foundation-f1-direct-import-windows/README.md)
  common71、新真实 cuBLAS 数学用例及两个新头隔离通过；八次 replay 租约 metadata 不分配。
  Graph、其他路线/平台、三交付及整包成本/红队仍未完成，不把子收据算作 F1 已关闭。
- Graph 当前新绑定 baseline `5a38cc09b`；只因本轮新增 leased 正常旅程与具名 aggregate 适配疑点进入 SCOPED，
  不恢复 H1 旧卡施工。compute / CUDA 已 AUDITED 的产品保留，其他 Graph/native route owner 仍 CLOSED。
- 初步源审计发现 aggregate 只在 destroy 时给仍 pending 的 children 加 lifetime；
  public poll/wait 若先把 unknown child 交给 native retirement，已丢掉随后连接 outer 的机会。
  leased Graph 必须在 child 任何观察之前保活，不能以 child 非 OK 推断可回收。
- Graph 本包范围是固定容量绑定/自动 native 保活及 owned/source-alias 结果；复用原 plan/Invocation、
  CompletionRetention 与内部 SubmissionCursor。普通借用 TensorView 和原 Direct 布局不变；
  定向 Graph aggregate 保护允许在既有冷 child 槽保存嵌套 retention，不增加第二 reaper/万能 Queue。
- 后续 AUDITED 前冻结结果 owner、旧结果阻止覆盖和 accepted-prefix 保护接口；
  未完成这份冻结前不写 Graph 产品代码，不以成熟库数学既有证据代签新的 leased 接缝。

### F1-01 共同拥有导入冻结

- baseline `65accb90c`，compute owner 继续 AUDITED；沿本卡已有拥有/只读导入范围。
- `LeasePool::adopt(mem::Buffer&&, options)` 接管已有唯一 Buffer，复用它的实际资源、容量和对齐；
  不再分配/复制 payload。所有有限 metadata 和 budget 准备成功后才移动 backing，失败保持输入所有权。
- `ReadLease::adopt(mem::Buffer&&, options)` 接管一个已成功 ready 的拥有 Buffer 并发布永久只读 lease；
  复制/subview 共享根，try_write 明确拒绝。单块 metadata 有界，原资源/Context/allocator 借用寿命不变。
- adoption 要求既有裸 views/第三方别名不再并发写；不能从一个 borrowed view 猜拥有权或传空 deleter。
  具体 native array/tensor 和受控外部 owner 导入仍由对应 native owner 的后续 scoped 桥承载。
- 新分配的 create 保持先预算准入后实际 backing 分配；adopt 按同 backing 一次记账。
  backing_alignment 只控制新分配，接管时以原 Buffer 对齐为准；不伪造更强对齐。
- 定向验证：拥有权拒绝不消费、readonly 的多 reader/subview、不允许重入写、预算失败及 native 完成保活。

### F1-06 Graph aggregate 定向 AUDIT

- scope baseline `a04c01083`；本次先将 graph owner 置 AUDITED，只授权以下 aggregate 子 FIX；
  lease-aware Graph 结果/绑定 façade 尚未冻结，不在这个子 FIX 中提前写它的产品代码。
- 源码证据：poll_aggregate/wait_aggregate 会先观察子 Completion；unknown wait 可以把子状态交给原 native
  retirement 并把公共 handle 变为缓存错误。旧 destroy_aggregate 在这之后才 attach_lifetime，不能补回保活。
  `Completion::settle` 已修好真实 attachment 交接，因此 Graph 必须提前提供 child 保护。
- make_aggregate_completion 在任何 public 观察之前，给全部 selected children 预挂 CompletionRetention；
  复用既有冷 child 槽并保留 inherited attachment，不以 attach_lifetime 覆盖 lease 或其他已有 owner。
- 每个 child guard 保持其原 Invocation 引用，unknown 观察后的 native callback 仍能安全访问 aggregate。
  sentinel 保持至外层 destroy；destroy 只交接 outer、保存首错、放弃各 child handle，不先观察再猜终态。
- 已 completed 的 child 在 attach 时可 inline 退休；unknown 不能释放 outer；真正 native 全部退役才释放 outer。
  不增加线程、timer/reaper或公共对象字段，不重写计划/Provider 选择。
- 验证是两独立真实 cuBLAS Graph 分支，给已接受的 native Completion 观察装饰一次未知错误并控制真实 stream prefix；
  外层等待错误不代表两个 native 分支结束，业务/Invocation/Completion 释放后仍安全，最终数学正确且回收一次。
  插桩隔离、空/inline child 和既有 Graph 核测试按实际影响执行；不是完整 Graph lease 资格。
- 内部 host-dependency 在 aggregate 组装前的观察，仍须由后续 opt-in leased SubmissionCursor 接缝在接受处保护，
  本子 FIX 不把那个尚未实现路径写 PASS。预算仍沿本卡两次有限纠正，不重启历史 H1 验证队列。

#### Graph aggregate 子 FIX 初验

`7907afec8` 已实现观察前的嵌套 child 保活和实际退休 sentinel 交接；
[Windows 初验](../../../../evidence/correctness/foundation-f1-graph-aggregate-windows/README.md)
Graph core31、新真实双分支 cuBLAS 用例通过。未知观察是装饰真实 native Completion 的确定性测试错误，
不声称已复现驱动错误；原 GPU 命令/退休照旧，已有 child read attachment 也保留。
目前整包 owner 继续 AUDITED；后续还要 Graph 绑定/结果与接受处保护、native 其他路线、全部适用交付/平台和整包红队。
统一脚本 self-check：Compute+Graph 非测试69,063行/305文件；全库非测试164,290行/656文件。

### F1-06 共同 Graph 绑定 / 结果接口冻结

- baseline `f8b93a264`，compute/graph owners AUDITED；只扩展本卡既有 Graph binding/result 与 accepted-prefix FIX。
- 新 opt-in `LeasedInvocation::create(executable, options, shape)` 冷创建原 Invocation、固定 input/caller-output
  绑定槽、一个 LeaseUseBatch 和每 segment 加一个 final 的 CompletionRetention；不重编计划或选择 Provider。
  `bind(value, description, ReadLease)` 共享输入；WriteLease 的重载接管输入/输出的独占权限。
  prepare_replay/submit 仍使用原 QueueBinding/原计划；零 Queue/单 Queue/多 Queue 都有原入口。
- 输出 `LeasedGraphOutput` 是可复制的只读结果，保留整个原 Invocation 与所有绑定 backing；内部 workspace、
  Executable 常量、source-alias 与 caller 输出均覆盖。tensor() 返回受此结果寿命保护的借用视图；
  storage_lease() 对 caller/source-alias 返回对应 ReadLease，内部 workspace/Executable 输出以原 Graph result
  owner 保活，不伪造可写 Buffer 拥有权。旧结果阻止本 Invocation 重放；单独导出的 ReadLease 也阻止对应写块复用。
- 输出只在成功且真实全退休后可取；由输出借来的输入别名也保持源 lease。重新绑定/reset 不得破坏旧结果。
  只读绑定不能当 mutable input 使用；独占输入由 adapter 拥有 WriteLease，读写 guards 与源/输出 alias 准入照原计划。
- 内部 SubmissionCursor 的可选 accepted 回调在每个已接受 Completion 的任何后续观察前安装 guard；
  opt-in finish 延迟原 Invocation terminal publication，由所有 child 与 final 的实际退休 latch 统一发布。
  这也保持非 concurrent Provider 的原 plan exclusion，不能因缓存观察错误提早允许另一 Invocation。
  原 public/borrowed submit 保持默认行为/对象布局，不增加资源表/万能 executor。
- 拒绝且无 native acceptance 时回退 guard、不毒化 write；已接受 prefix 出错保存首错并保护到实际退役。
  元数据冷分配，热 submit 不扩容。close 停接纳，drain 明确 host 等待本 adapter 已接纳工作；结果仍有效，
  Completion 负责原 cancel 请求，取消不证明 GPU 结束。原资源/Context/allocator借用寿命保持明确。
- 有限验收：真实 cuBLAS Direct+Graph 数学、两个 native 分支、旧结果/来源别名、重放计数、拒绝回退、
  accepted-prefix 观察故障/丢弃与关闭；后续 CPU/native routes 和三交付仍集中做，不以本入口编译代签。
- accepted-prefix 故障资格用 graph owner 的私有 HYLE_TESTING TLS 观察装饰 seam：
  仅在 opt-in accepted hook 已保活之后装饰真实 Completion，制造跨 lane 依赖的观察/交接失败。
  原数学/设备/事件/retirement 不替换；shipping 没有 hook/额外分支，不重写公共仪器。

### F1-01 CUDA owned external Buffer 接缝冻结

- baseline `aae604926`；compute / compute_backend_cuda owners 仍 AUDITED。沿已有 native 导入 FIX，
  不重开 Provider/算子模块，也不修改原 CUDA 指针检查/分配/释放 API。
- `ExternalBuffer::into_read_lease(options) &&` / `into_lease_pool(options) &&` 只接管确实 adopted 的 native Buffer；
  borrowed buffer 没有实际 payload owner，拒绝转成拥有 lease。准备失败保持原 ExternalBuffer 不变。
- 原 ExternalBufferState 本身持有 Context/resource；成功时移动其 mem::Buffer，并由 common lease domain
  保留这个实际 native resource owner，直到 backing/预算退休之后才销毁。内部 LeaseAccess 只衔接 owner release，
  不增加资源/执行 vtable，不用空 deleter。普通 Buffer 的既有 resource-borrow 合同不变。
- 复用原 cudaFree/driver retirement、Common Buffer adopt 和有限metadata；无新 copy/kernel。
  原生消费者定向验 owned/borrow拒绝、OOM拒绝不消费、真实数学与 native completion 保活。

### 当前共同 Graph / CUDA native owned 初验

`64720f782` 已实现上述 Graph 自动 binding/result、每 accepted prefix 保活/延迟真实终态、
CUDA owned external Buffer→ReadLease/LeasePool。拥有与拒绝、源码review/P0/局部红队及
[Windows 初验](../../../../evidence/correctness/foundation-f1-leased-graph-native-owner-windows/README.md)
完成：shipping13定向用例、插桩Graph4通过，三个头隔离编译，shipping两个hook导出不存在。
八次 Direct/Graph replay metadata 无额外分配；真实 cuBLAS 与原 driver allocation、旧结果、private workspace、
源 alias、pending close/drain、accepted-prefix failure 已验。夹具曾误断言两个 Context 必然 device wait，
按实际显式 host-wait 纠正，原失败记录保留，不写成 GPU 产品性能失败。
未完成：其余 native owner/普通 CPU 数学消费者、CPU+GPU组合绑定、适用平台与三交付及全包红队；
F1 继续 AUDITED，F2/F3 未启动，不把这些局部结果算作整包关闭。
统一统计 clean candidate self-check：Compute+Graph非测试69,557行/308文件；全库164,784行/659文件。

| FIX | 必须实现/定向修复 | 验收 |
|---|---|---|
| F1-01 | 冷创建 owner、有限块/范围、generation、大小/对齐/metadata budget、拥有与只读导入 | 零/边界/溢出/OOM；旧 generation 不释放新块；真实 native backing |
| F1-02 | shared read/exclusive write、非重叠并行、别名/权限、成功发布 | 多读阻止写/复用，写发布前不能读；共享根只计一次 |
| F1-03 | 使用 token/guard 经 CompletionRetention 与各 native 退役衔接 | business handle/Completion 丢弃仍保活；index/metadata 一同保护 |
| F1-04 | 对 CUDA poll/wait/destroy 的终态分类作定向资格；有实际缺陷才修该 owner/直接消费者 | native query 观察失败而真实 work 未结束的故障注入；不能提前释放；原错误保留 |
| F1-05 | lease-aware Direct 原生提交，真实成功/partial failure/unknown use 退役 | CPU/CUDA/Apple/Vulkan 成熟 MatMul/FFN 等实际数学，不以 copy/mock 代签 |
| F1-06 | lease-aware Graph binding/result 适配，外部输出/输入 alias 保活 | 原 plan 不重编/换 Provider；旧 result 阻止覆盖；四路线及组合边界 |
| F1-07 | 非阻塞 collect、cancel、stop/close/drain、late completion、失效 slot | 取消不等于 GPU 结束；重复关闭/最后 guard/合法线程销毁；容量恢复 |
| F1-08 | 普通入口、指南、P0、三交付与新头/缺席边界 | 自动 route 闭包、DLL、迁址；无热 metadata 增长/无第二 reaper |

## 预声明验证与停止

### Apple / Vulkan 原生 binding 子轮 SCOPED

- 实际 baseline / previous registry ref `73d9c847a`，Windows工作树干净；新增绑定 owner
  `compute_route_apple`、`compute_route_vulkan`，因 F1 外部 range lease→原生 math/Graph 的新旅程重开。
  两个 graph_route owner仍CLOSED：它们已生成各自原生 Program，本子轮只验证其直接消费，不改 lowerer。
- 原 Apple Invocation::bind 保存 ready array owners，Vulkan Invocation::bind 保存原 Buffer owners，
  原 Results/source-alias 已有对应 native retention。缺口是 common ReadLease backing 尚无无复制 native binding，
  不能把现有 native owning handles当作已经保护了新 lease root。
- 本子轮仅实现 MemoryResource 的 cold readonly binding；反向 native array/Tensor 只读导入与组合 owner
  继续保留在 F1 后续，不因这一子入口而裁决全部 native FIX 完成。
- MacBook Air已实际连接：Mac16,12、16 GiB、macOS27.0.1/26A434、CMake4.3.4、Ninja1.13.2。
  未编译，后续先读取实际 compiler/当前缓存；只用Air，Windows是唯一产品真相。

### Apple / Vulkan 原生 binding 子轮 AUDIT

- baseline `bca2fb233`；两 compute route owner 进入 AUDITED，有限产品只改各自 memory.h / memory implementation。
- `MemoryResource::binding(ReadLease source, TensorDesc description, LeaseImportOptions options)` 冷返回各自的
  原 MLX array / Vulkan Tensor。它保留实际 source ReadLease，native bind/Program/Graph/Results 原强 owner 链
  自动延长 root 寿命，原 native complete/unknown retirement 不变；用户不手写 use token，也不增加共同 Queue。
- 全部 native wrapping/descriptor metadata 在 binding 冷路径；repeat submit 不构造新 wrapper或复制payload。
  options 只计 wrapper metadata；既有 source backing 的预算保留一次，Apple opaque MLX descriptors仍按
  原明示 unattributed 合同，不把不可见 native 成本写成零。
- Apple只接同一MemoryResource的 ready shared Metal backing；校验实际范围、dtype carrier、item-origin对齐和
  carrier布局。使用MLX原copy_shared_buffer保留offset；捕获真ReadLease owner，不是空deleter假拥有。
- Vulkan只接同一device MemoryResource（staging缺storage usage，拒绝作为数学binding），descriptor bounds
  沿原native规则；四字节copy/storage尾容量必须已包含在source lease，不跨邻块偷用padding。
  借原Buffer view/owner机制持有root及ReadLease，保留原device/family/transfer资格，无新shader/kernel。
- 普通 native数组/Tensor本身可被更高级interop修改，返回binding是只读准入，所有未跟踪writable aliases
  必须停止；只用tracked Writer只能在native aliases消失后取得写。裸TensorView/resource寿命合同不变化。
- 新测：实际成熟MatMul和Graph/source-alias、两reader、old结果/native handles阻止writer、业务lease放弃与native
  最后退休、wrong resource/desc/padding/strict budget拒绝。无空target/mock provider/copy-only代签。

扩展现有 compute completion、graph cpu_cuda/apple/vulkan contracts 及 native route 用户样例；
保留原数学 reference，新增实际 lease 风险断言。只装饰 native observation/fault call，
不能将 mock 结果当路线资格。至少两独立在途块、两个读者、一写者、旧结果、取消、OOM、部分失败、
未知观察、自然 Direct/Graph 和完整 acquire/use/free；不全量重跑算子目录/P1。

P0 静态账单须列新增 metadata/引用/锁，及旧普通路径不变的证据；
热路径以计数验无 allocation/provider search，不为纳秒差改布局。

完成标准：F1-01至08及全部适用门真实通过，明确原借用 precondition 和未验硬件；
精确报告所有 applicable 未验，不自动继承 WSL Vulkan 例外；按协议形成 terminal。
F2 才提供 bulk copy，F4 才讨论版本历史/自动 CoW，本卡不扩成泛用事务系统。
