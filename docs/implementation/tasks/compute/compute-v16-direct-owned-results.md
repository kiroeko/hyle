# Compute R2-D：独立提交与原生结果 owner

<!-- PROJECT UNIT REVIEW: AUDITED -->

## Closure card

- component / owner：`compute` / Compute；baseline `50753347`。
- trigger：[R2](four-route-common-semantics-v1.md) 的 Direct delegated 与生成结果合同。当前 prepare 排除
  delegated，submit 需要 Queue；Completion 只容纳一个附件，无法同时保有 invocation 和独立结果 owner。
- normal consumer：有稳定 Mem 输入/输出的独立 runtime 直接提交；原生 managed runtime 生成 own array。
  正交 consumer 是普通非模型计算结果及保留旧结果后使用第二个槽。
- scope：D21–D24；原有 shape/resource/alias/native backend 机制继续由其真实 owner 负责。
- non-goals：新 backend/provider、算子扩充、原生 runtime 统一为 Context/Queue、用户计时或 Provider 排序。
- dependency：复用 Error/Mem 与 private util::RefCounted；无新公共组件边，无全局 registry。
- 既有 Compute AUDITED 重新定界到本卡，R1 terminal 不授权新 runtime 修改；Graph 产品由 R2-G 单独绑定。
- 性能：P0 先裁决新增可选机制的结构成本与旧 shared 路径；具体路线默认入口的同实现 direct 税和 P1
  仍须 R3–R6 兑现，不从本卡推出完整路线性能或零成本。
- 定向源码读取已完成；以下为有限固定动作。绑定 AUDITED 后实施，实施/普通验证最多四小时、一次纠正。
  公共 API/算法变化期间只做相应正确性；R2 最终候选统一完成适用平台/三交付，未运行门保持 UNPROVEN。

## 已固定的动作

- D21：独立的 prepare_delegated / PreparedDelegatedOperation / DelegatedInvocation。原 prepare 与
  Queue submit 保持 shared 模式不变量；不把 delegated 状态交给含 shared HYLE_ASSUME 的公共类型。
  接收稳定 Mem-backed 输入/调用方目的 buffer 的既有 Provider 可以直接提交，不依赖 Graph。
  MLX array 原生 Program 不必采用这套 Context/Buffer 形状，使用自己的 native 入口及 D23 owning result。
- D22：Completion 可选的复合 lifetime attachment。新 retention 节点内存由具体 owner 预留，保存
  原有 attachment 和新增 owner；真实 terminal 后分别通知/释放，abandonment 交回原 native completion。
  不改变 Completion/现有 backend event slot 大小，不增加默认路径分配或按 kernel 的分派。
- D23：typed OwnedResult<T> / ResultLease<T>，T 是原生 owning handle，不能把 T 当作 Mem allocation。
  producer 使用 detail::ResultSlot<T> 的冷预留 metadata；begin 独占，publish 移入真实 Completion。
  同 slot 在 in-flight 或有外部结果 lease 时拒绝覆盖；另一个 slot 可继续执行。未成功不发布 ready lease。
- D24：CPU 真计算 + 真 CUDA terminal/abandonment 的生命周期消费、已有 MLX C++ 机制映射；协议
  故障测试不宣称正式 MLX/CUDA Provider 完成。完整路线的默认入口和 P1 仍属于 R3–R6。

## Direct freeze

prepare_delegated(context, Operation, descriptions, resources) 与 views convenience 返回专用 Prepared
类型。create_invocation(DelegatedInvocationOptions{workspace, dependency_capacity}) 冷创建状态；
capacity=0 默认 input_count+1。依赖可由调用方以 span<const Completion* const> 提供；指针与 source
Completion 保活到消费完成，禁止同时移走/销毁/settle 其 native state。为空表示输入已经 ready。
submit(inputs, outputs) / submit_after(dependencies, inputs, outputs) 没有 Queue 参数。

每个 delegated slot 冷保留一次 dependency scratch，submit 转换有限依赖并校验 bindings，不分配。
公开包装保护 scratch 的同时提交；已有 Invocation in-flight gate 仍保留到真实终态。Provider 在
接纳前检查依赖的真实 native 类型/终态，不能隐式 host wait 或在失败后隐藏换 Provider。
依赖超 capacity、错误 binding、关闭 Context 等在接纳前返回错误；已接纳 prefix 必须返回 Completion
并保留晚到错误。原 shared Direct 没有新增模式判断、metadata 或锁。

## Native result freeze

OwnedResult<T> 为 move-only submission，拥有 Completion 及 typed 原生结果的存储 owner。
wait() 成功返回可保留的 ResultLease<T>；result() 只 poll，不做隐式等待，pending 返回明确拒绝。
lease 可复制，持有真实 T；即使原 slot/submission 析构，保留结果仍有效。producer 可在准备的存储中
构造 T，T 要求 noexcept 移动/析构；payload allocation 属 native runtime，metadata allocation 在冷创建。

detail::ResultSlot<T> / ResultWrite<T> 只供 route/provider 实现。slot 先 begin 获得写入权，producer
放入 native owning T，再把接纳后的真实 Completion publish；失败/取消不提前回收。未提交的 write
析构回滚；已提交后 producer 必须 publish Completion，包括晚到失败，不以普通 Status 丢弃工作。

隐藏 ResultStorage 使用 Compute 既有 private util::RefCounted 和 AllocatorRef；公共 typed wrapper
只持有 opaque pointer，不另加公共 util 依赖、不复制 refcount 算法。Composite retention 节点是可选
owner metadata，Completion 原 hot/cold ABI 不扩；跨 DSO 使用相同工具链和真实析构回调，无地址 identity。

## 验证与有限预算

范围固定并绑定 owner 后才改 canonical 产品。D21–D24 实施与 ordinary correctness 最多
四小时、一次有证据纠正；不追加整体架构研究。R2 的最终 API 候选统一完成适用三交付，避免每个小 FIX 新建同一 SDK 矩阵；
期间单测回放只覆盖发生变化的 owner/直接消费者，未做的适用验收仍标 UNPROVEN。
P0 先列冷分配、结果保留的 refcount、D21 scratch admission、完成回调费用；不宣称零成本。
完整 native default 入口的同实现 direct 税与 P1 由各路线在 API/算法固定后一次性验收。

## 预声明验收与七问

正确性：empty/moved handles、冷 OOM、错误模式/依赖/资源/已关闭 Context、同槽并发与接纳后 failure、
真实 terminal 后回收、Completion/slot 提前析构、保留结果与另一槽、旧 shared Direct 回归。协议故障
用具名 fixture；CPU 真运算和 CUDA 真异步资源消费单独取证，不能把 fixture 当作正式 native Provider。
WSL/两台 macOS 的共同代码正确性及适用 sanitizer 随 R2 候选验收；重型验证串行、最多两 jobs。
源码、static、shared、relocation 用同一真实消费；producer 位于单独 DSO 的 owning T 析构须有对应检查。
R2-Q 的已完成门复用精确候选证据，变化影响的门才回放，不把文档/状态当作重建理由。

1. 无新第三方版本/许可；MLX 只约束 owning result 必须保留原 handle，R5 可使用独立 native Program/Completion。
2. 新公共 owner 来自跨运行保留生成结果的真实需求；callee-owned 与 caller-owned 输出均可表达。
3. 复用 RefCounted、AllocatorRef 和 native Completion；不复制 quota、event、worker pool 或资源算法。
4. 冷 slot 分配一次；热态固定边界的引用计数、有限依赖转换和 admission 明确计费，不宣称零开销。
5. 没有 C ABI/Native FFI；同工具链 C++ 析构回调不是身份，不比较跨 DSO 静态地址。
6. Compute owning factor 导出新头/实现；native 库头不进入共同接口；三交付同合同。
7. 普通调用使用 checked result/lease；producer 的写槽属于 detail seam，由各路线的正常 Program/run-state 包装。

SCOPED 已于 `5bafe026` 提交；现完成 AUDITED 绑定，D21–D24 与上述 API/所有权边界固定。
源码锚点：`operation.cc` 的公开 prepare、shared submit、delegated retained/borrowed submit；
`completion.cc` 的 settle/reset/attach_lifetime；`prepared_state.h` 的 invocation 引用与 in-flight；
现有 Graph aggregate 证明 terminal/release 必须分开、先退役 native state 再释放其 owner。
新复合附件只用于需要多个 owner 的结果，不改现有 backend CompletionVTable 和 event slot 布局。
共享入口与独立入口在冷类型上分开；结果 slot 先取得写入权再接纳工作，publish 不申请内存。
失败的 publish 保留调用方传入的 Completion，不允许以普通错误丢掉已接纳工作。
candidate、七门实际结果、限制及统一 LOC 在实施后记录；当前不签 native 路线完成。

## 首版实现及 Windows 检查点

D21–D23 已实现：独立 typed prepare/submit、冷 dependency scratch、可选复合 Completion retention、
typed OwnedResult/ResultLease 与 private ResultSlot 存储。原 shared Invocation、Completion 和 backend
event slot 未增字段；新可选 Direct state 复用 private RefCounted，payload 仍归真实 native T。
emplace 接收显式移动引用 T&&，MSVC 的 128-byte aligned owner 编译反例证明不能用按值形参。

Windows MSVC 19.50.35728 / Ninja / Release、两任务普通回放：Compute、CPU backend、CUDA backend、
Native CPU 四个 factor 与对应逐头共 9 项门通过。包括 composite 嵌套 owner、旧结果/另一个槽、冷 OOM/
回滚、128-byte alignment、同槽并发 begin/submit、依赖失败、关闭 Context、晚到错误、取消、跨 DLL
producer 析构、真实 Native RMSNorm 结果保留和真实 CUDA increment kernel 的终态/abandonment。

CUDA 新 fixture 首次放置 stream gate 后再做首次 kernel 调用导致超时；把首次准备移到 gate 前后解除。
随后发现 fixture 把 device drain 当作后台 owner 已回收，已按源码的 CompletionReaper 边界加入 owner
回收信号；信号本身保有 owner，测试退出前等待实际 buffer 释放。没有修改 CUDA backend，也未以固定
sleep 或计时阈值证明完成。初始编译/超时/错误断言日志保留，只回放受影响 CUDA 门；其余 8 门复用。

P0：新 ResultSlot 冷分配一次 metadata+对齐 T；native payload allocation/destruction 由 T 负责。
Direct 另有可选冷 state/scratch，热态为既有 bindings 校验、有限依赖转换和 admission/refcount。
retention 的 completion 引用还保护 terminal 与 release 之间的节点，不能提前复用同一附件。不宣称零成本。

当前仍 AUDITED。Windows 记录在 `C:/hyle-scratch/r2-q4-20260916/direct-tests.log` 与
`direct-cuda-retest.log`；完整源/静态/动态、Graph/oneDNN 直接消费、WSL/两 Mac 及适用 sanitizer 随稳定
R2 API 候选统一验收，尚未运行项不签 PASS。本检查点不签完整路线、默认优选规则或 P1。
