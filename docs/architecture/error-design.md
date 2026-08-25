# error 设计：值语义失败、分层上下文与诊断边界

> **Native FFI 状态校正（2026-08-25）**：HYLE 没有 native-C 客户，也没有已发布的 managed
> binding。本文关于 C/C++、C/FFI 或 C 用户的旧表述只记录待复审的 ABI 候选，不是当前跨语言
> 证书或兼容承诺；首发前可破坏性调整。先从真实 foreign-runtime 任务裁决 FFI 是否为 `N/A`；
> 只有保留 direct FFI 时，error 才必须通过 Java FFM + .NET P/Invoke、真实导出、所有权/缓冲区
> 与 batch/error 旅程后冻结该边界。

> **信源**：本文定义 HYLE error 模块的长期产品形状。用户用法放在
> `docs/guide/`，性能数字只放在 `docs/performance/`，施工和迁移状态只放在
> `docs/milestones/`。本文不以已有测试或 benchmark 的形状反推 API。

## 0. 定位

HYLE 是 C++ 基础加速库。error 模块只解决一种问题：**把可恢复的运营失败作为值，
可靠地跨函数、模块和线程传递；若真实 foreign-runtime 任务成立，再穿过 Native FFI 边界**。

它不承担日志系统、异常框架、崩溃报告器或任意元数据容器的职责。设计从下面的真实
调用场景出发：

1. OS、CUDA、Vulkan、文件和网络调用失败，叶子适配层把原生失败转换为统一状态；
2. compute/io/platform 等中间层增加“正在做什么”的语义上下文；
3. 上层按稳定错误码决定重试、降级、返回用户或结束一次任务；
4. 被选中的 foreign-runtime 任务以明确所有权传递同一种失败；没有任务时该边界可为 `N/A`；
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
- 带消息或 native detail 的错误使用 DLL 内部单块分配、原子引用计数 representation；
  detail 的 domain 复用 representation 原有 padding，只有实际携带 detail 时才在消息尾部
  增加一个 8B code，不扩大普通 message Status；
- owning STL 类型不跨共享库 ABI；消息以借用 `string_view` 暴露，格式化使用调用方缓冲区；
- 所有 Status 自身操作均为 `noexcept`；消息分配失败时至少保留错误码；
- `StatusCode::Ok` 不能携带消息；越界的枚举值在公共/C 边界归一为 `Unknown`；
- 相等比较 canonical code、native detail 与完整消息；程序分支通常只判断 canonical code。

批量/清理场景只提供两个零额外状态的组合工具：

- `overall.Update(next)`：`overall` 仍为 OK 时接收第一个错误，之后不覆盖；
- `status.IgnoreError()`：明确标注调用方有意忽略一次失败，替代含义不清的 `(void)`。

`status.h` 不引入 owning string 或 iostream。需要 C++ owning 人读输出时显式包含
`hyle/error/format.h`，使用 caller-owned `to_string(status)`；caller-buffer/低层
日志继续使用 `ToString(char*, size_t)`，避免为便利接口加重每个热路径 TU。

canonical code 使用 Abseil/gRPC 的稳定集合，不增设每个后端一套公共枚举。常用边界：

- `InvalidArgument`：调用者给出的值本身无效；
- `FailedPrecondition`：值可能有效，但对象/系统当前状态不允许操作；
- `ResourceExhausted`：内存、句柄、队列或设备资源耗尽；
- `Unavailable`：外部服务、驱动或设备暂不可用，可由上层决定是否重试；
- `Unimplemented`：当前构建/后端不支持用户明确请求的能力；
- `Internal`：外部依赖返回无法进一步分类的失败，不用来掩盖 HYLE 自身不变量损坏；
- `Unknown`：来自不可信 ABI/外部输入的非法枚举或确实无法分类的外部错误。

### 2.2 canonical code 与原生 detail

canonical code 回答“跨平台调用方该怎么处理”；少数后端恢复逻辑还需要回答“原生系统
到底返回了哪个数值”。因此 `Status` 提供一个窄而定长的 detail：

```cpp
Status::FromNative(StatusCode::Unavailable,
                   {NativeErrorDomain::Vulkan, -4}, "device lost");
NativeError detail = status.native_error();
```

domain 只覆盖 HYLE 的真实系统边界：POSIX errno、Win32、HRESULT、Vulkan、CUDA runtime、
CUDA driver 与 NVRTC。它不是可注册的字符串 category，也不是任意 payload map。普通调用方只按
canonical code 分支；确实要执行 backend-specific recovery 或输出结构化遥测时，才检查
`native_error()`。message 永远是人读文本，禁止解析文本取原生码。

这项能力由真实缺陷触发：Vulkan 的 initialization failed、incompatible driver 与 device
lost 都正确归到 `Unavailable`，但调用方无法在不解析 message 的情况下选择重建 device 或
直接降级。Abseil 用通用 payload 表达此类数据；HYLE 只需要一组 domain + 一个 `int64_t`，
故不把 map、Cord、type URL 和多 payload 生命周期强加给所有错误。`FromNative` 只在错误
路径分配；OK、code-only 与普通 message 形状不变，OOM 时保留 canonical code、丢弃可选
message/detail。

### 2.3 分层上下文

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
- 对外始终是一个扁平连续消息，不保留链式节点或隐藏 source-location 栈；
- 需要分配且分配失败时返回原 Status，不能为了增加上下文反而丢掉已有消息；已有
  prefix 容量足够时不分配，内存紧张也能完成原地 prepend；
- rvalue 重载消费原对象，const lvalue 重载产生独立值。

表示按实际使用渐进升级，而不是让每个错误预付成本：

- 普通 message/native Status 仍使用 12B 内部头和精确单块分配；公开 `Status` 仍为 8B；
- 第一层 context 仍精确分配，只用内部 flag 记录“已经有语义层”，因此最常见的一层不预留
  猜测性的未来空间；
- 第二个 rvalue context 证明这是渐进传播链，此时才切到 context 专属
  `{capacity, offset, prefix storage}`；容量按 1.5 倍增长（小容量至少增长 32B），消息放在
  storage 尾端；
- 唯一所有权且 prefix 足够时只写新 context 和分隔符，O(context bytes)、零分配；容量不足
  时搬迁一次；共享 representation 与 lvalue 保持 copy-on-write，不修改其他 Status 的值；
- 几何增长时历次搬迁字节是几何级数，因此连续 rvalue 链的总复制为
  `O(final message bytes)`，空间也是 O(final bytes)，不再是 O(k²)。单次 lvalue/COW 仍须复制
  当前完整消息；若机械地反复对 lvalue 赋值，调用方仍会主动选择 O(k²) 路径。

这项实现来自成熟库形状对照，而不是由新增 benchmark API 驱动。Abseil `StatusBuilder` 在每次
产出 Status 时仍生成扁平消息；Rust `anyhow`/`error-stack` 以链式 frame 做 O(1) 包装，把遍历
与格式化留到消费端；Boost.LEAF 以动态作用域的 typed error slot 做与调用深度无关的传递。
后两类都不能同时保留 HYLE 的自包含 owning value、跨线程传递、`string_view`/C
`message_cstr` 连续借用和查询零分配合同。context-only prefix storage 保留这些合同，同时不让
普通 Status 承担 node、缓存或容量字段。builder 仍只适合“多个片段同时已知”的一次组合，不能
代替跨函数逐层传播。

示例最终消息：

```text
creating device buffer: vulkan vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY (-2)
```

叶子适配层必须生成自足信息：后端/子系统名、失败操作、原生错误名称与数值。中间层只在
增加了用户可理解的新语义时调用 `WithContext`，不为每一层函数名机械加壳。

本阶段仍不加入以下能力：

- 任意 key/value payload；
- 可动态注册的 native error category；
- 自动 source location、栈回溯或日志；
- `StatusBuilder` 和流式 `operator<<`；
- 异常层级或嵌套 `exception_ptr`。

后端原生类型仍不能泄漏进公共头；公开面只保留定长 domain/code。只有出现一个 Status
必须携带多个独立机器可读 detail 的真实消费者时，才重新评估 payload 系统。

### 2.4 原生 OS 适配

POSIX/CRT 与 Windows 叶子边界不应在每个模块重复发明分类。error 提供三组公共基础能力：

```cpp
StatusCode status_code_from_errno(int error_number) noexcept;
Status status_from_errno(int error_number,
                         std::string_view operation = {}) noexcept;
StatusCode status_code_from_win32(std::uint32_t error_number) noexcept;
Status status_from_win32(std::uint32_t error_number,
                         std::string_view operation = {}) noexcept;
StatusCode status_code_from_hresult(std::int32_t error_number) noexcept;
Status status_from_hresult(std::int32_t error_number,
                           std::string_view operation = {}) noexcept;
```

errno classifier 按 Abseil `ErrnoToStatusCode` 的成熟 canonical taxonomy 做 O(1) 分类；
对应 factory 同时取得
线程安全的 native description，附加 `PosixErrno` detail，并生成 `operation: description`。
`errno == 0` 返回零值 OK；
失败路径只有最终 Status 的一次分配，OOM 只丢 message、不丢 code。一般调用方优先用工厂，
不再手写 `strerror`、message 拼接和 code switch。

Win32 入口同样做 O(1) canonical 分类，并保留稳定 32-bit code。Windows 构建用
`FormatMessage` 生成 description；其它 host 使用确定性数字文本，因此 crash/offline tooling
仍可分类远端 Win32 错误，且公共头不引入 `windows.h`。未知值映射 `Unknown`，不能因为某个
platform 调用者偏好重试就笼统映射 `Unavailable`。成功路径不格式化、不分配。

COM/WinRT 的 signed 32-bit HRESULT 由独立入口处理：非负值是成功，
`HRESULT_FROM_WIN32` 复用 Win32 taxonomy，但 detail 保留 `HResult` domain 与完整原值；Windows
同样优先 `FormatMessage`，其它 host 用确定性十六进制文本。它修复了 platform thread-name
失败曾只保留手写文本、无法机器恢复/遥测的消费者缺口。

通用映射不是操作语义的替代品。某个系统调用对同一 errno 有更具体的合同，可以局部覆盖；
例如 `pthread_create(EAGAIN)` 是资源耗尽，而通用 I/O 映射把 EAGAIN 视为暂不可用。需要在
诊断中保留原生符号名和数值的 platform/backend translator 仍可自己格式化 message，但 code
必须复用公共 taxonomy，避免 fs、VM、NUMA 和 mmap 对同一 errno 给出不同 code。

### 2.5 `Result<T>` 与传播

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
- rvalue 失败/值被移动，lvalue 失败/值被复制；传播命名 Result 不会暗中消费它；
- `ASSIGN_OR_RETURN` 可以声明新变量或给已有 lvalue 赋值；
- 不隐藏日志、source location 或自动文本拼接；需要语义上下文的边界显式使用
  `WithContext`。

构造 `Result<T>` 的失败分支使用 `make_unexpected(Status)`，避免每个调用点重复
`std::unexpected<Status>`，并在受控入口 fail-fast 拒绝 `Status::Ok()`。`Result<T>` 仍是标准
alias，所以直接写 `std::unexpected(Status::Ok())` 在类型系统中无法禁止；这是明确的合同
违反，传播时也会 fail-fast。Abseil `StatusOr` 自有容器会归一/拒绝 OK error，Folly
`Expected` 与 `std::expected` 则让 error 类型自行定义语义。为这一条不变量包装 expected
会复制整套构造、monadic 和引用规则，收益小于维护与兼容成本，因此不做。

声明型
`HYLE_ASSIGN_OR_RETURN` 与成熟库同样会展开为多条语句，作为 `if/else/for/while` 裸语句时
必须加花括号。

HYLE 当前 `sched::Task<T>` 是独立的协程值模型，并不返回 `Result<T>`；因此 error 不添加
无法独立成立的 `CO_*` 宏。让异步任务携带 `Status/Result` 需要先审查 Task/JoinHandle 的
所有权、取消和错误槽，是 sched×error 的后续架构议题，不伪装成 error 内两个宏即可解决。

重构前的 `RETURN_NOT_OK` / `ASSIGN_OR_RAISE` 在产品代码中没有调用点，且 `RAISE` 暗示
异常。HYLE 尚未发布，不保留兼容别名。新宏必须迁入 compute、graph、platform 等真实
调用链后才能算被产品接受；仅有宏单测和微基准不构成验收。

## 3. Native FFI 候选面

现有 `hyle/c/error.h` 与 bridge 使用一个 machine word 的 owning handle：

```c
typedef struct hyle_status { uintptr_t rep; } hyle_status;
```

它只是 C-syntax ABI 候选，不是 native-C 产品或已发布 managed contract。先由真实 Java/.NET 等
任务决定是否保留；没有任务时可整体 `N/A`。公开 C++ `Status` 不暴露 raw representation 的
adopt/release。HYLE 各模块内部只传 C++
`Status` / `Result<T>`，不通过 `hyle_status` 互调；raw representation 的零拷贝 ownership
转移只允许出现在 `src/*/c_*.cc` 对外 ABI 候选 bridge（共同使用 `src/error/status_access.h`），
不是模块间 ABI。当前候选 handle 的零值 initializer 是
`HYLE_STATUS_INIT`，所有会释放/替换旧值的 pointer API 都要求目标已初始化。

若任务裁决保留 owning Status direct FFI，至少复审以下任务能力，而不是按 C++ 名称机械 parity：

- OK/错误创建；
- clone、move、free；
- code、message、相等比较和 buffer-fill 格式化；
- 借用 NUL 结尾消息，生命周期绑定 owning handle；
- length-aware 创建/context，允许嵌入 NUL 且避免无意义 `strlen`；
- 原地增加上下文；
- 消费式首错聚合；
- errno 到 canonical code/Status 的直接适配；
- Win32 code 到 canonical code/Status 的直接适配；
- native domain/code 的创建与查询；
- 返回 `hyle_status` 的函数使用 `HYLE_C_RETURN_IF_ERROR` 传播。

上下文接口采用原地所有权形状：

```c
void hyle_status_add_context(hyle_status* status, const char* context);
```

`status == NULL`、OK、NULL/空 context 均为 no-op。成功时替换 handle 中的
representation；分配失败时保持原状态及原消息不变。这样 foreign host 不需要同时管理“旧状态
和新状态”两个 owning handle。

ABI 调用方传入的 fixed-width code 属于不可信输入。非法 `hyle_status_code` 归一为
`HYLE_STATUS_UNKNOWN`，不能
让越界值进入内部 tagged representation。C++ 与候选 schema 的 code 表仍由编译期逐项断言保持一致。

若保留该候选，error 头只负责 Status。日志与 sink 属于 obs；foreign host 自身的程序错误由
宿主语言/runtime 的 fail-fast 机制处理，不在 `hyle/c/error.h` 保留第二套
warning/error/fatal API。

## 4. 外部错误映射规范

每个外部边界提供局部 translator，但遵守同一合同：

```cpp
Status from_vk(VkResult result, std::string_view operation) noexcept;
Status from_cuda(cudaError_t error, std::string_view operation) noexcept;
Status status_from_errno(int error, std::string_view operation) noexcept;
Status status_from_win32(std::uint32_t error,
                         std::string_view operation) noexcept;
Status status_from_hresult(std::int32_t error,
                           std::string_view operation) noexcept;
```

translator 必须：

1. 成功值返回零表示的 OK；
2. 映射成 canonical code；
3. 在错误路径保留 operation；
4. 诊断整体保留 operation、人读描述和机器可读数值：message 不要求重复 detail 已保存的
   数字；SDK 有稳定 symbolic name 时应保留，`NativeError` 始终保留 domain/code；
5. 使用固定栈缓冲或一次 Status 消息分配，不在成功路径格式化；
6. 不把 CUDA/Vulkan/OS 类型放进公共 API，只转为稳定 domain + `int64_t`；
7. 未知原生值仍返回 `Unknown`/`Internal` 和完整数值，不能只写“failed”。

`status_from_errno`、`status_from_win32` 与 `status_from_hresult` 是通用 OS 入口；platform
的 syscall-specific translator 只有操作合同更具体时才覆盖 canonical code。CUDA、Vulkan 等依赖 SDK 类型和
SDK-specific recovery 的映射仍由所属叶子模块完成，再用 `Status::FromNative` 附加稳定
detail；不把 vendor 类型或动态 category 注册表塞进 error 公共头。

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

性能验收只允许两种由消费者需要定义的信息合同，不能再由“专门为 benchmark 写的 API”
代替用户路径：

1. **不携带额外诊断**：成功值或 canonical code；与成熟库同形 carrier/传播对拍。
   `std::error_code` 这类能力更弱、可平凡析构的类型只作成本下界，不能混入同能力输赢。
2. **携带诊断**：相同 canonical code、owning message、语义 context 或同一个可程序化
   native value；与成熟库提供同等信息的入口对拍。

benchmark 必须从正常调用链反推：函数边界、所有权、错误率和最终消费动作都要对称。不得
为分数给 Status/Result 增加常驻 metadata、特殊公开类型、缓存或 benchmark-only 入口；
参考库拥有但 HYLE 没有真实消费者的能力也不为扩表上架。

### 微观不变量

- `sizeof(Status) == 8`（64 位平台）；
- OK construct/check/copy 和 code-only error 均零分配；
- `Result<int>` 正常 unwrap 与 `std::expected<int, Status>` 无额外层；
- code-only 和 message error 分别在上述两个信息合同中对标 Abseil Status；
- `WithContext(OK, ...)` 零分配且接近普通分支成本；
- 一层/三层 context 与 Abseil StatusBuilder 的等价文本场景对标；一层不得因未来猜测回退，
  三层必须命中渐进表示；128 层正确性测试机械验证扩容次数保持 O(log final bytes)；
- native detail 构造/查询与 Abseil payload 的同能力场景对标；普通 Status 不因 benchmark
  常驻 payload metadata；
- errno 分类是 O(1)，`status_from_errno` 与 Abseil `ErrnoToStatus` 做同形 operation+native
  description 对照；不得为跑分引入全量常驻错误文本表或改变首次失败合同。
- Win32 分类也是固定 switch 的 O(1)，只有错误路径调用 `FormatMessage` 并做一次最终 Status
  分配；它的价值由 platform 消除重复 translator 与跨模块分类一致性证明，不造高错误率跑分。

### 真实产品路径

- platform：Thread/DynLib/MappedFile 的原生 OS 失败 → Status；若 platform 保留 direct FFI，
  再由其候选 adapter 转换。VM/NUMA 已迁入同一 translator/context 合同；
- compute：CUDA/Vulkan 叶子错误 → buffer/kernel 高层上下文；
- graph：`Result<T>` 提取与跨层传播；
- sched：`Result<void>` 的 join/channel 正常返回与传播；
- diagnostics：caller-buffer 与 opt-in owning string 格式化；
- 成功路径和错误路径分开报告，禁止用高错误率混合 workload 掩盖成功路径回退。

任何新增上下文能力不得改变 OK/code-only representation。性能结果、平台、编译器、样本
数和绑定 commit 只写入 `docs/performance/benchmarks-error.md`。

## 7. 收口门槛与重开条件

error 只有同时满足以下条件才能冻结：

1. C++ API：Status/Result/context/传播覆盖三条真实产品调用链；
2. Native FFI：先以真实 foreign-runtime 任务裁决；无任务可明确 `N/A`。保留时只覆盖任务所需
   Status 能力，Java FFM + .NET P/Invoke 从精确 shared library 验证 ABI bootstrap、owning handle、
   buffer 和错误旅程；纯 C 编译/manifest 只算 hygiene；
3. 边界：error、obs、fatal 没有重复职责和依赖环；
4. 正确性：OOM 保级、复制/移动、并发引用计数、非法 ABI 候选输入和格式化截断均验证；
5. 可诊断性：CUDA/Vulkan/OS 错误保留 operation、原生名称、数值和可查询 native detail；
6. 性能：Windows、Linux、macOS 的核心与真实路径均无未解释回退；
7. 文档：architecture 只保留本文的稳定设计，guide 是 recipe，performance 是数字，
   milestone 是任务记录。

冻结后只有以下真实需求允许重开功能面：

- 一个 Status 必须携带多个独立机器可读 detail；
- 跨进程/网络序列化成为正式产品场景；
- 真实调用证明 `std::expected` 或当前传播方式无法表达所需所有权。

“某个 benchmark 可以更快”“参考库还有更多 API”“测试容易写”都不是重开理由。
