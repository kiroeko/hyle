# 排障：从症状找到边界

先保留实际失败的版本、构建配置、输入、错误和完成条件，再缩小问题。
最小复现应保留触发因素：形状是否变化、lane 是否不同、是否跨线程、是否超过队列深度、
是否真的产生内存或 CPU 压力。删掉这些因素的“小程序”可能已经不再复现原问题。

| 症状 | 先观察什么 |
|---|---|
| 初始化、编译或加载失败 | 工具实际路径/版本、所选配置、第一条原始错误 |
| 挂起或 CPU 空转 | 提交/终结完成数量、在途数、队列、阻塞位置与系统调用错误 |
| 偶发 UAF 或迁移后崩溃 | 最后一个访问者、已复制回调、线程局部状态、真实迁移时序 |
| GPU 空 buffer 或跨形状崩溃 | allocation owner、backend context、offset、大小和外部导入合同 |
| SIMD 个别值错误 | 满值域、跨分支 lane、异常值及派生中间量 |
| 稳定但反常的速度差 | 实际工作量、最终指令、地址放置、顺序、完成与采样层次 |

以下故障模式来自项目调查，保留的是定位方法与设计原因，不是当前缺陷清单或历史测试通过证明。
接口已变动时按当前公开头和实现核对；原始源码、日志和数据仍在 Git 与 [evidence](../../evidence/README.md)。

## 构建、依赖与运行库

先运行相同参数的 `python scripts/init.py --dry-run`，核对实际编译器、后端、build 与共享 cache。
初始化的原始输出保存在 build 目录 `hyle-configure.log`，后续动作分别有 build/test/install 日志。
缺编译器、Python、CUDA 或 Metal 工具链时，按[平台说明](../setup/README.md)正常安装基础工具；
缺项目资产时检查[依赖缓存](../setup/dependencies.md)和[离线闭包](../setup/offline.md)。

不要在已有 CMake cache 中换编译器，也不要把另一系统的对象、cache 或 SDK 搬来续用。
SDK 消费失败检查安装 prefix、所选 `find_package` 组件、runtime 资源与 C++ ABI，见[集成说明](../../user/distribution.md)。

Windows 头改了却不重编时，检查真实编译日志、Ninja 依赖前缀与 `/showIncludes` 输出语言是否一致，
不要只看 exe 时间戳。WSL 的重构建和复现使用原生文件系统，避免 `/mnt/c` 引入额外 IO 干扰。

## I/O 挂起：先看完成守恒

正常单次请求能跑，不代表高扇出能正确收尾。曾有 256 项 ring 上 260/512 条独立流挂起，
而少量 ping-pong 正常。关键观测是提交数不再变化、终结完成数少于提交数、reap 次数持续增长，
且 pending SQE 与 CQ overflow 都为零：问题在完成推进，不是“队列满所以丢了请求”。

在 Linux io_uring 路径核对：

- worker 进入其他 parker 前，是否仍有在途 I/O；相关完成 task-work 是否需要带 `GETEVENTS` 的 enter 才会推进。
- `SINGLE_ISSUER` 的 issuer 是否与实际 enter 线程一致。历史 `EEXIST(17)` 来自创建线程和 owner worker 不同；
  `R_DISABLED` 方案还涉及由正确线程 enable、之后再武装 wake-read 的顺序。
- cancel 请求本身也是一次提交，它的完成应正常减少在途计数；不要用永久 wake-token 充当 cancel-token。
- multishot 的 `MORE` 中间完成与终结完成是否分开计数；注册 buffer-ring 时 ring 是否已经允许该操作。

当前相关实现：[Proactor](../../../src/io_core/proactor.cc)、[IoContext](../../../src/io_core/io_context.cc)、
[Scheduler](../../../src/sched/scheduler.cc)。历史修复线索：`e95cdd1`、`285c68a`。
不同后端推进机制不同，不能把某次 io_uring 现象外推到 IOCP 或 kqueue。

### 一次提交，多次完成

multishot 是完成流，不能套用 one-shot 的单结果槽。历史上同一次 reap 批先交付数据、再交付 EOF，
EOF 覆盖未消费数据，消费者误以为流已结束。检查完成是否排队、empty→non-empty 时是否只唤醒一次，
以及取消后 slot 是否活到 kernel 的终结完成。

吞吐突然增长几十倍时也先核对成功提交和完成数量：初始化失败后的空转不是加速，
STREAM 合并后的 chunk 数不能直接与保消息边界的请求数相除。buffer-ring 的分块 owner 也不天然适合
要求整个 body 连续的 HTTP API；为强接消费者而新增复制，会改变原来的成本问题。

## 并发与 fiber：取消引用不等于没有在途访问

`when_any` 曾出现以下时序：loser 在锁内读走栈上 GroupSignal 指针，解锁后被延迟；
winner 唤醒等待方，等待方清空指针并返回，栈对象销毁；loser 恢复后再调用旧指针，形成 UAF。
把共享字段置空无法追回已经复制出去的指针。

当前实现用 state 锁覆盖 group 回调，清理也取得同一把锁，并保持锁序无环。
检查类似代码时要证明对象活到最后一个回调结束，而不只是“析构前注销”。
回归应让 loser 也真正完成，永远 parked 的 loser 会避开整个窗口。
实现与测试见 [join.h](../../../include/hyle/sched/join.h)、[join_test.cc](../../../tests/sched/join_test.cc)。
TSan 没报告只说明这次运行未观察到相应冲突，不能替代时序论证。

### 跨 worker 迁移与 TLS

有栈 fiber 换栈不等于更换线程 TLS 基址。直接在可内联的 fiber 函数中跨 `yield()` 使用
`thread_local` 派生指针，可能遇到编译器缓存旧线程地址的问题；普通 memory barrier 或在同函数里
“重新取一次”不一定消除编译器的公共子表达式。不要跨迁移保存线程局部状态；需要线程亲和语义时
使用能保证亲和的调用方式，参见[调度器指南](../../user/sched.md)。

旧 KI-1 调查没有确定原始崩溃根因。原生文件系统、分配器检查和部分压力试验未复现，
不能据此宣布库代码已被排除，也不能仅凭 SIGILL 就认定是文件系统代码页损坏。
可复用的经验是保留原始 workload 的线程 churn/迁移，确认压力确实存在，并检查 allocator、
fiber 切换和 sanitizer 的可观察边界；static/LTO 还可能改变原本不透明的 TLS 调用边界。

## GPU buffer：数据指针不是完整资源

旧 ggml 集成曾用 size-0 dummy buffer 元数据包装外部 device 指针。单节点和同形状批次能过，
不同形状触发另一条 kernel 路径后却解引用了空 `buffer->context`。
定位依赖保留 varied-shape 输入、脱离 Hyle 的 ggml 最小复现，以及带符号的真实调试栈。
只搜索 assert 不能证明某字段不会被解引用。

不要伪造 backend buffer/context，也不要为获得合法元数据而偷偷分配整份用不到的显存。
先确认 provider 拥有 allocation，还是后端明确支持的外部导入，再遵守设备、范围、同步与寿命合同。
这个旧失败不证明所有后端、所有版本都不支持外部内存或零拷贝；当前边界见
[Compute 设计](../../architecture/compute-design.md)。
旧源码可用 `git show 1621b284:src/compute/ggml_bridge.cc` 追溯，不能当作现行 API。

### Metal 的 32 KiB 限制到底是什么

G09 旧日志中的错误是某个 attention pipeline 的 `threadgroupMemoryLength(58368)`
超过该设备的 `32768` 字节限制。这是**每个 threadgroup 的片上共享内存**，不是 GPU buffer、
显存驻留或整机内存的 32 KiB 上限；分割大 tensor/buffer 不会自动解决它。

同一 pipeline 当时在普通/API-only 验证下查询为 29184 字节，加入 Shader Validation 后为
58368 字节。因此要查询实际创建的 pipeline，不能只按 tile 公式预算，也不能据一例断言
所有 kernel 的验证开销都固定翻倍。当前局部修正见
[MLX patch](../../../third_party/mlx_cast/patches.json)：对 Hyle page-info Steel attention 检查实际
`staticThreadgroupMemoryLength`，超限时尝试 compact tile 后再次检查；仍超限则正常报错并排空已接纳工作。
这条修正没有改变 D256/dtype，也不意味着自动 CPU 回退或整条设备流程已经验证通过。

原日志保存在
[ordinary-checkpoint-5.zip](../../evidence/correctness/inference-g09-completion-wake/ordinary-checkpoint-5.zip)。

## SIMD 数值：输入有限不保证中间量安全

先对同一输入比较 Hyle 路径、依赖库 direct 路径和明确的 scalar oracle。
改变 ISA、编译器或使用可用的标量模拟能帮助区分算法与 codegen，但单个跨平台结果本身不是证明。

| 历史触发模式 | 应保留的测试或诊断 |
|---|---|
| Pack16 的 11/13-bit 满位宽 round-trip 失败 | 覆盖全位宽；`value & bits` 不是 `value & ((1<<bits)-1)` |
| `lgamma(1.29875)` 仅与另一分支 lane 混放时错误 | distinct-per-lane/ramp 与跨归约分支组合；同值填满会漏测 |
| `ctanh` 小实部、虚部接近 `π/2` | 小 denominator 放大误差，依赖库不能自动成为病态输入 oracle |
| `ctanh` 有限虚部大于 `FLT_MAX/2` | `2*imag` 会溢出；全-block 快路径不能绕开逐 lane 异常处理 |
| 轴、signed-zero、subnormal、NaN/Inf、tail、原位 alias | 分别检查 regular-lane 分类、fallback 和写回顺序 |

当前 `lgamma` 已由锁定 Highway 路线提供，不能照旧调查恢复过时的“全 scalar”结论。
相关代码：[bitpack](../../../src/simd/bitpack/kernel_impl.h)、[math kernel](../../../src/simd/math/highway_math_kernel-inl.h)、
[complex kernel](../../../src/simd/math/complex_hyle-inl.h)；回归见[数学测试](../../../tests/simd/math_test.cc)。
依赖升级时重跑实际受影响输入，不沿用过去版本的“已修复/仍有 bug”判断。

## 性能反常：稳定也可能有偏差

曾有最终指令相同的短循环稳定相差两位数。调查定位到 P/E 核迁移、代码实例放置，
以及闭包捕获链重载与 store 流的 4K 别名；同一个 exe 换启动器也会改变栈地址与结果。
同侧多个相同步长的副本还可能落在同一相位，不能把它们当独立样本来缩小区间。

先确认语义和逐条最终指令，包含标量尾巴、寄存器宽度和 sink；“向量指令签名相同”不代表整个循环相同。
随后用 A/A、顺序交换、指针外提和等价布局检查偏差，并保留完整分布。
发现某个仪器问题也不能据此解释所有慢项：额外 mov、spill、广播和依赖链仍可能是真实损失。
具体测量方法只在[性能指南](../performance/README.md)维护，不在每份故障记录中另定阈值。

## 查原始调查

本页提炼前的调查文档可从提交 `d12f2204d` 的 `docs/developer/debugging/` 读取，例如：

```text
git show d12f2204d:docs/developer/debugging/when-any-loser-uaf-hunt.md
```

需要复现时优先使用当前回归测试及仍存在的原始数据；旧命令中的路径、版本、测试数量和通过结论
只是当时事实，不自动成为本次验证结果。
