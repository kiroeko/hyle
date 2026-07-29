# HYLE 用户文档（docs/guide）

> 这个文件夹是**面向库使用者**的文档——把 HYLE 当依赖集成进自己工程的人看这里。
> 想读"为什么这么设计"的内幕（面向贡献者）去 [`../architecture/`](../architecture/)。
> 想看可运行的示例工程去仓库最外层 [`../../samples/`](../../samples/)。

## 本目录文档

| 文档 | 给谁 / 讲什么 |
|---|---|
| [`usage-guide.md`](./usage-guide.md) | **怎么用**：先按 shared/static/source 选择消费入口，再查模块、构建选项、SIMD 变体、最小用法和 C-ABI 边界。**集成第一站。** |
| [`distribution.md`](./distribution.md) | **拿到什么就怎么用**：headers+shared、headers+static、完整源码三种用户接入方式；另含交付方制作命令、授权边界和离线依赖注入。 |
| [`core.md`](./core.md) | **core 选型入口 + C++ 自然写法**：compile/runtime version、静态/运行时 cache line、三种 prefetch、compiler contract 与 intrusive list 的适用场景和生命周期坑。 |
| [`error.md`](./error.md) | **error 语言无关选型入口 + C++ 自然写法**：按运营失败/空值合同/程序错误/诊断边界决定 Status、Result、值合同或 fail-fast；C 用户再去 c-error 查映射。 |
| [`mem.md`](./mem.md) | **mem 语言无关选型入口 + C++ 自然写法**：按 ownership/concurrency/capacity/lifetime 决定什么情况用什么；C 用户再去 c-mem 查映射。 |
| [`platform.md`](./platform.md) | **platform 语言无关选型入口 + C++ 自然写法**：按资源/生命周期选择 VM、文件、mapping、线程、等待、DynLib、CPU/NUMA；明确 OS 差异与当前拓扑边界。 |
| [`time.md`](./time.md) | **time 语言无关选型入口 + C++ 自然写法**：按 elapsed/墙钟/sleep/任意 deadline/同 duration timer 选择入口，并解释 single-owner、re-arm 与周期策略边界。 |
| [`util.md`](./util.md) | **util 选型入口 + C++ 自然写法**：ScopeGuard/Ref/Rng/SpinLock/type-name/demangle 什么情况选什么，明确 core/runtime 分层、复杂度与常见坑。 |
| [`obs.md`](./obs.md) | **obs 选型入口 + C++ 自然写法**：error/log/trace 怎么选，format/record/sink 生命周期、递归与异步桥接，以及 native trace 的零成本边界。 |
| [`container.md`](./container.md) | **container 选型入口 + C++ 自然写法**：小序列/POD vector/flat map/字符串驻留/五类并发队列分别何时选择，明确失效、顺序和当前性能边界。 |
| [`c-error.md`](./c-error.md) | **error C/FFI 精确映射**：选型沿用 error.md，只说明 `hyle_status` 的 owning 生命周期、Status/out、上下文、格式化与传播。 |
| [`c-platform.md`](./c-platform.md) | **platform 纯 C ABI**：Parker/Thread/DynLib/VM/mapped_file/fs/cpu/numa 不透明句柄与自由函数、named enum、所有权与易用示例。 |
| [`c-mem.md`](./c-mem.md) | **mem C/FFI 精确映射**：选型沿用 mem.md，只说明 Status/out、opaque handle、typed 宏、所有权和各入口。 |
| [`c-time.md`](./c-time.md) | **time C/FFI 精确映射**：选型沿用 time.md，只说明 Status、opaque timer/queue、同步 visitor、所有权和 shutdown。 |
| [`c-obs.md`](./c-obs.md) | **obs C/FFI 精确映射**：选型沿用 obs.md；共享 Status/out、结构化 record、opaque sink、flush 和 runtime trace。 |
| [`c-util.md`](./c-util.md) | **util 纯 C ABI 精确映射**：选型沿用 util.md；RNG POD、Ref/ScopeGuard/Registry/SpinLock 类型擦除句柄、demangle buffer-fill、所有权和并发合同。 |
| [`c-container.md`](./c-container.md) | **container C/FFI 精确映射**：选型沿用 container.md；共享 Status/out、字符串/路径、运行时擦除 flat map/vector、并发 opaque handle 与失效合同。 |
| [`c-simd.md`](./c-simd.md) | **SIMD 纯 C ABI**：247 个 whole-buffer 入口、低精度/排序 POD、RNG opaque handle、运行时 ISA 诊断、调用合同和 FFI 绑定方式。 |
| [`c-sched.md`](./c-sched.md) | **sched 纯 C ABI**：worker-stack/fiber 双腿、Waker、优雅关闭、同步/消息/取消/blocking pool 的类型擦除入口、生命周期和外语绑定合同。 |
| [`c-io.md`](./c-io.md) | **io 纯 C ABI**：proactor/completion、网络/文件提交、后端诊断、multishot/fixed-file、共享内存，以及 C 与 sched 的组合合同。 |
| [`io.md`](./io.md) | **IO C++ 使用指南**：per-worker proactor、TCP/UDP deadline/cancel、自动文件卸载、HTTP/HTTPS transport、DNS/signal 与专属快腿。 |
| [`c-core.md`](./c-core.md) | **core 纯 C/FFI 映射**：version 动态符号、纯 C compiler contract、静态 cache layout，以及零分配 intrusive list 的 header-inline 能力与显式生命周期。 |
| [`simd-usage.md`](./simd-usage.md) | **SIMD 怎么选和怎么调**：普通动态派发、整库固定 ISA、固定 kernel、常用 batch 能力与调用合同，以及已经落地的 whole-buffer C ABI 边界。 |
| [`usage-contract.md`](./usage-contract.md) | **怎么调才不出 UB**：为 SOTA 性能保留的调用安全红线（fiber 勿跨 `yield()` 持有 thread_local、分配器跨线程 free、CPUID 探测的 LTO 隔离）。每条带机制+安全模式+后果。**热路径集成前必读。** |

各 `c-*.md` 不只列导出符号，也必须解释原生 C++ 的模板/RAII/协程形状在 C/FFI
里怎样组合出来。找不到稳定组合通路就是 C ABI 缺口，不能用“C 没有模板”略过；
`c-sched` §9 是非平凡组合 recipe 的样板。

严格收口时，每个用户可见模块都必须有两条可发现入口：`<module>.md` 先回答“什么场景
选什么、C++ 怎样自然组合、哪里会踩坑”，`c-<module>.md` 再回答相同能力怎样映射到纯 C/FFI。
只有 C 符号表而没有前一条选型指南，仍算文档缺口；纯 header 或极小模块必须先证明它是
合理的独立产品，再决定 C 映射。若能力因模板/编译期语义不跨语言，主 guide 必须明说；
不能靠缺文件表达，也不能为了补齐文件反过来固化尚未审查的历史 surface。

## 用户文档的分工

- **guide（这里）**：正常怎么用 + 怎么调才安全 —— 使用者视角。
- **[`architecture/`](../architecture/)**：为什么这么造、各支柱选型、clay 资产负债 —— 贡献者视角。
- **[`performance/`](../performance/README.md)**：绑定 commit 的 benchmark 方法、点值与 win/loss；guide 不复制。
- **[`milestones/`](../milestones/README.md)**：施工任务和当前待办；guide 不维护就绪度镜像。
- **[`samples/`](../../samples/)**：照着抄就能跑的示例工程 —— 上手最快。
