# error 设计：值语义失败、分层上下文与诊断边界

> **信源**：本文定义 HYLE error 模块的长期产品形状。用户用法放在
> `docs/guide/`，性能数字只放在 `docs/performance/`，施工和迁移状态只放在
> `docs/milestones/`。本文不以已有测试或 benchmark 的形状反推 API。

## 0. 定位

HYLE 是 C++ 基础加速库。error 模块只解决一种问题：**把可恢复的运营失败作为值，
可靠地跨函数、模块、线程和 C ABI 传递**。

它不承担日志系统、异常框架、崩溃报告器或任意元数据容器的职责。设计从下面的真实
调用场景出发：

1. OS、CUDA、Vulkan、文件和网络调用失败，叶子适配层把原生失败转换为统一状态；
2. compute/io/platform 等中间层增加“正在做什么”的语义上下文；
3. 上层按稳定错误码决定重试、降级、返回用户或结束一次任务；
4. C/FFI 调用方以明确所有权传递同一种失败；
5. 调用方违反合同或 HYLE 内部不变量损坏时立即 fail-fast，而不是伪装成 Status；
6. 应用需要日志时显式使用 obs，而不是 Status 在传播途中偷偷输出。

## 1. 三条互不重叠的失败通道

| 情况 | 通道 | 例子 |
|---|---|---|
| 可恢复的运营失败 | `Status` / `Result<T>` | OOM、设备暂不可用、文件不存在、超时 |
| 人读的运行诊断 | `obs/log` | 应用主动记录一次重试、降级或任务失败 |
| 程序错误/合同违反 | `HYLE_FATAL_ASSERT` | 越界、错误线程调用、内部状态不变量损坏 |

因此 error 不提供第二套全局日志 handler，不提供 soft assert，也不在返回 Status 时自动
记录日志。具体边界如下：

- `Status` 是数据，不产生 I/O；构造、复制、传播和销毁都不调用 obs。
- 宿主日志接入、sink 生命周期、结构化字段和 level gate 全部属于 obs。
- `HYLE_FATAL_ASSERT` 保持在 core `macros.h`，不得令 error 反向依赖 obs。
- 可恢复的非法输入用普通分支返回 `InvalidArgument`；内部不变量用 fatal assert；不存在
  “报告一下然后带着可能损坏的状态继续执行”的中间语义。
- 编译期弃用使用 `[[deprecated]]`，不在运行期打印警告。

原有 `error_handler`、`ErrorRecord`、`HYLE_WARN/ERROR/FATAL/ASSERT` 及对应 C ABI 与
上述边界重叠，而且没有 error 模块外的产品消费者，已在未发布阶段删除。需要日志的
用户使用 obs；测试日志 sink 也使用 obs 的可摘除 sink。

## 2. C++ 产品面

### 2.1 `Status`

`Status` 保持一个 machine word 的值语义对象：

```cpp
Status ok;                                      // OK
Status missing = Status::NotFound();            // code-only，零分配
Status failed = Status::Unavailable("device lost");
```

长期合同：

- `sizeof(Status) == sizeof(uintptr_t)`；
- OK 编码为零，构造、检查、复制均不分配；
- 无消息错误把 canonical code 内联在 tagged word 中，不分配；
- 只有带消息的错误使用 DLL 内部单块分配、原子引用计数 representation；
- owning STL 类型不跨共享库 ABI；消息以借用 `string_view` 暴露，格式化使用调用方缓冲区；
- 所有 Status 自身操作均为 `noexcept`；消息分配失败时至少保留错误码；
- `StatusCode::Ok` 不能携带消息；越界的枚举值在公共/C 边界归一为 `Unknown`；
- 相等比较 code 与完整消息；程序分支通常只判断 code。

批量/清理场景只提供两个零额外状态的组合工具：

- `overall.Update(next)`：`overall` 仍为 OK 时接收第一个错误，之后不覆盖；
- `status.IgnoreError()`：明确标注调用方有意忽略一次失败，替代含义不清的 `(void)`。

`status.h` 不引入 owning string 或 iostream。需要 C++ 人读输出时显式包含
`hyle/error/format.h`，使用 caller-owned `to_string(status)` 或 `operator<<`；C/FFI/低层
日志继续使用 `ToString(char*, size_t)`，避免为便利接口加重每个热路径 TU。

canonical code 使用 Abseil/gRPC 的稳定集合，不增设每个后端一套公共枚举。常用边界：

- `InvalidArgument`：调用者给出的值本身无效；
- `FailedPrecondition`：值可能有效，但对象/系统当前状态不允许操作；
- `ResourceExhausted`：内存、句柄、队列或设备资源耗尽；
- `Unavailable`：外部服务、驱动或设备暂不可用，可由上层决定是否重试；
- `Unimplemented`：当前构建/后端不支持用户明确请求的能力；
- `Internal`：外部依赖返回无法进一步分类的失败，不用来掩盖 HYLE 自身不变量损坏；
- `Unknown`：来自 C/FFI 的非法枚举或确实无法分类的外部错误。

### 2.2 分层上下文

真实加速库错误必须同时回答“哪一层操作失败”和“底层为什么失败”。`Status` 增加唯一
一个组合能力：

```cpp
[[nodiscard]] Status WithContext(std::string_view context) const& noexcept;
[[nodiscard]] Status WithContext(std::string_view context) && noexcept;
```

语义固定为：

- OK 或空 context 原样返回，不分配；
- context 前置，格式为 `outer context: existing message`；
- 原 Status 的 canonical code 不变；
- 多层上下文按最外层到最内层排列；
- 每次组合生成一个扁平消息，不保留链式节点或隐藏 source-location 栈；
- 分配失败时返回原 Status，不能为了增加上下文反而丢掉已有消息；
- rvalue 重载消费原对象，const lvalue 重载产生独立值。

单次组合必须读取并生成完整文本，因此时间/空间是最优的
`O(context.size + message.size)`。连续 k 层会重复复制已扁平文本，等长 context 的整个构造
序列最坏为 O(k²)；真实产品只应添加少量语义层。三层场景列入冻结 benchmark，若将来出现
深层机器生成 context，再以真实调用决定是否引入链式 representation，不能提前让所有
Status 承担节点和最终格式化复杂度。

示例最终消息：

```text
creating device buffer: vulkan vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
```

叶子适配层必须生成自足信息：后端/子系统名、失败操作、原生错误名称与数值。中间层只在
增加了用户可理解的新语义时调用 `WithContext`，不为每一层函数名机械加壳。

本阶段不加入以下能力：

- 任意 key/value payload；
- 公共 native error category 注册表；
- 自动 source location、栈回溯或日志；
- `StatusBuilder` 和流式 `operator<<`；
- 异常层级或嵌套 `exception_ptr`。

这些能力只有在出现消息无法满足的真实程序化消费者后才重新评估。后端原生类型仍不能
泄漏进公共头；若将来确有按原生码分支的需求，再设计独立、定长的 detail，而不是提前
引入通用 payload 系统。

### 2.3 `Result<T>` 与传播

```cpp
template <class T>
using Result = std::expected<T, Status>;
```

不包装、不复制 `std::expected` 的成员 API，不发明 HYLE 自有 monadic 容器。
只保留 C++ 缺少 `?` 运算符所需的两个传播宏：

```cpp
HYLE_RETURN_IF_ERROR(expr);
HYLE_ASSIGN_OR_RETURN(lhs, expr);
```

合同：

- `expr` 只求值一次；
- 第一项接受 `Status` 或无值成功态的 `Result<void>`，第二项接受 `Result<T>`；
- 同时支持外层函数返回 `Status` 或 `Result<U>`；
- 失败时移动 Status，成功路径不分配；
- `ASSIGN_OR_RETURN` 可以声明新变量或给已有 lvalue 赋值；
- 不隐藏日志、source location 或自动文本拼接；需要语义上下文的边界显式使用
  `WithContext`。

构造 `Result<T>` 的失败分支使用 `make_unexpected(Status)`，避免每个调用点重复
`std::unexpected<Status>`，但不包装 `std::expected`、不复制其容器 API。声明型
`HYLE_ASSIGN_OR_RETURN` 与成熟库同样会展开为多条语句，作为 `if/else/for/while` 裸语句时
必须加花括号。

HYLE 当前 `sched::Task<T>` 是独立的协程值模型，并不返回 `Result<T>`；因此 error 不添加
无法独立成立的 `CO_*` 宏。让异步任务携带 `Status/Result` 需要先审查 Task/JoinHandle 的
所有权、取消和错误槽，是 sched×error 的后续架构议题，不伪装成 error 内两个宏即可解决。

重构前的 `RETURN_NOT_OK` / `ASSIGN_OR_RAISE` 在产品代码中没有调用点，且 `RAISE` 暗示
异常。HYLE 尚未发布，不保留兼容别名。新宏必须迁入 compute、graph、platform 等真实
调用链后才能算被产品接受；仅有宏单测和微基准不构成验收。

## 3. C/FFI 产品面

C ABI 保持一个 machine word 的 owning handle：

```c
typedef struct hyle_status { uintptr_t rep; } hyle_status;
```

公开 C++ `Status` 不暴露 raw representation 的 adopt/release。HYLE 各模块内部只传 C++
`Status` / `Result<T>`，不通过 `hyle_status` 互调；raw representation 的零拷贝 ownership
转移只允许出现在 `src/*/c_*.cc` 对外 C ABI shim（共同使用 `src/error/status_access.h`），
不是模块间 ABI。C handle 的零值 initializer 是
`HYLE_STATUS_INIT`，所有会释放/替换旧值的 pointer API 都要求目标已初始化。

能力必须覆盖：

- OK/错误创建；
- clone、move、free；
- code、message、相等比较和 buffer-fill 格式化；
- 借用 NUL 结尾消息，生命周期绑定 owning handle；
- 原地增加上下文；
- 返回 `hyle_status` 的函数使用 `HYLE_C_RETURN_IF_ERROR` 传播。

上下文接口采用原地所有权形状：

```c
void hyle_status_add_context(hyle_status* status, const char* context);
```

`status == NULL`、OK、NULL/空 context 均为 no-op。成功时替换 handle 中的
representation；分配失败时保持原状态及原消息不变。这样 C 用户不需要同时管理“旧状态
和新状态”两个 owning handle。

C enum 属于不可信 FFI 输入。非法 `hyle_status_code` 归一为 `HYLE_STATUS_UNKNOWN`，不能
让越界值进入内部 tagged representation。C++ 与 C code 表仍由编译期逐项断言保持一致。

C error 头只负责 Status。日志与 sink 使用 `hyle/c/obs.h`；C 调用方自身的程序错误由
宿主语言/runtime 的 fail-fast 机制处理，不在 `hyle/c/error.h` 保留第二套
warning/error/fatal API。

## 4. 外部错误映射规范

每个外部边界提供局部 translator，但遵守同一合同：

```cpp
Status from_vk(VkResult result, std::string_view operation) noexcept;
Status from_cuda(cudaError_t error, std::string_view operation) noexcept;
Status from_errno(int error, std::string_view operation) noexcept;
```

translator 必须：

1. 成功值返回零表示的 OK；
2. 映射成 canonical code；
3. 在错误路径保留 operation；
4. 保留原生错误名称和数值，必要时再加供应商错误字符串；
5. 使用固定栈缓冲或一次 Status 消息分配，不在成功路径格式化；
6. 不把 CUDA/Vulkan/OS 类型放进公共 API；
7. 未知原生值仍返回 `Unknown`/`Internal` 和完整数值，不能只写“failed”。

推荐文本：

```text
vulkan vkQueueSubmit failed: VK_ERROR_DEVICE_LOST (-4)
cuda cudaMalloc failed: cudaErrorMemoryAllocation (2): out of memory
posix mmap failed: ENOMEM (12): Cannot allocate memory
```

## 5. 聚合、取消与并发

- 多任务聚合默认返回首个非 OK 状态；确定性模式按稳定任务顺序选择，不能由线程竞争
  偶然决定。若调用方需要全部错误，那是独立的结果集合 API，不塞进单个 Status。
- `Cancelled` 与 `DeadlineExceeded` 保持不同：前者来自显式取消，后者来自时间预算耗尽。
- Status 是不可变值语义；复制只增引用计数，可在线程间传递。
- error 模块不持全局 handler 锁，不创建线程，不拥有 executor，也不进行异步 I/O。

## 6. 性能合同

性能验收分两层，不能再由“专门为 benchmark 写的 API”代替用户路径：

### 微观不变量

- `sizeof(Status) == 8`（64 位平台）；
- OK construct/check/copy 和 code-only error 均零分配；
- `Result<int>` 正常 unwrap 与 `std::expected<int, Status>` 无额外层；
- code-only 和 message error 对标 Abseil Status；
- `WithContext(OK, ...)` 零分配且接近普通分支成本；
- 一层/三层 context 与 Abseil StatusBuilder 的等价文本场景对标。

### 真实产品路径

- platform：Thread/DynLib/MappedFile 的原生 OS 失败 → Status → C ABI；VM/NUMA
  已迁入同一 translator/context 合同；
- compute：CUDA/Vulkan 叶子错误 → buffer/kernel 高层上下文；
- graph：`Result<T>` 提取与跨层传播；
- sched：`Result<void>` 的 join/channel 正常返回与传播；
- diagnostics：caller-buffer 与 opt-in owning string/stream 格式化；
- 成功路径和错误路径分开报告，禁止用高错误率混合 workload 掩盖成功路径回退。

任何新增上下文能力不得改变 OK/code-only representation。性能结果、平台、编译器、样本
数和绑定 commit 只写入 `docs/performance/benchmarks-error.md`。

## 7. 收口门槛与重开条件

error 只有同时满足以下条件才能冻结：

1. C++ API：Status/Result/context/传播覆盖三条真实产品调用链；
2. C ABI：所有适合 FFI 的 Status 能力可达，无诊断 handler 残留；
3. 边界：error、obs、fatal 没有重复职责和依赖环；
4. 正确性：OOM 保级、复制/移动、并发引用计数、非法 C 输入和格式化截断均验证；
5. 可诊断性：CUDA/Vulkan/OS 错误保留 operation、原生名称和数值；
6. 性能：Windows、Linux、macOS 的核心与真实路径均无未解释回退；
7. 文档：architecture 只保留本文的稳定设计，guide 是 recipe，performance 是数字，
   milestone 是任务记录。

冻结后只有以下真实需求允许重开功能面：

- 用户必须按后端原生码做程序化分支，而文本不足；
- 一个 Status 必须携带多个独立机器可读 detail；
- 跨进程/网络序列化成为正式产品场景；
- 真实调用证明 `std::expected` 或当前传播方式无法表达所需所有权。

“某个 benchmark 可以更快”“参考库还有更多 API”“测试容易写”都不是重开理由。
