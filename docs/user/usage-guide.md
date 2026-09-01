# HYLE 使用指南

> 适用于预发布的 C++23 surface。模块可用性和稳定度见
> [当前施工状态](../implementation/status.md)。

## 选择交付形式

HYLE 支持三种 C++ 消费方式：

| 方式 | 适合场景 | 用户入口 |
|---|---|---|
| 完整源码 | 需要审计、定制 ISA/backend、LTO 或整体组装 | `add_subdirectory` 后链接 `hyle::<product>` |
| headers + static | 固定工具链/配置，部署时不携带 HYLE 动态库 | `find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS ...)` |
| headers + shared | 多个 C++ 程序共享二进制、独立更新 artifact | `find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS ...)` |

完整构建、安装和离线准备见 [distribution](distribution.md)。这三种都是 C++ 发布形状，不等于
跨语言 FFI。

## 源码集成

```cmake
set(HYLE_BUILD_TESTS OFF)
set(HYLE_BUILD_BENCH OFF)
set(HYLE_COMPONENTS "mem;container")
add_subdirectory(path/to/hyle hyle-build)

target_link_libraries(my_app PRIVATE hyle::container)
```

`HYLE_COMPONENTS` 指定产品根；构建系统自动解析它们的内部 factor 闭包。仍应只链接代码实际使用的
`hyle::<product>` target，让 CMake 能发现错误依赖。`container_intrusive`、`util_core` 等内部
factor 不属于这张用户菜单。`add_subdirectory()` 消费者必须显式设置该
变量；只有直接把 HYLE 作为顶层工程配置时，省略才默认 `all`。

## 安装包集成

先用所需 linkage 安装 HYLE，再在用户项目中：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem container)
target_link_libraries(my_app PRIVATE hyle::container)
```

不要手工复制库文件、头路径或第三方链接参数；imported target 承载这些合同。shared 与 static
安装包不能混在同一个 prefix 中假装可互换，消费者工具链也必须满足包记录的 C++/平台合同。
安装只含所选组件闭包拥有的公开头，不保证未请求模块的头可见。预发布 `0.y.z` 必须请求完整
精确版本；patch 版本、headers 与 artifact 是同一个 C++ 构建合同。

TLS、HTTP/2、CUDA、ggml、Vulkan 和 trace 等昂贵能力默认关闭；相应 cache 选项只在 owner
组件/gate 成立时出现。需要时显式开启。当前 trace 只有 Tracy，不存在 backend selector。

## 选择模块

| 需求 | 首选模块 |
|---|---|
| 编译器提示、cache layout、极小基础 primitive、版本信息 | [core](core.md) |
| 可恢复失败、带上下文结果 | [error](error.md) |
| OS/CPU/topology/thread/VM/file-system 机制 | [platform](platform.md) |
| allocator、arena、pool、对齐、大页、NUMA placement | [mem](mem.md) |
| clock、deadline、sleep 与 timer queue | [time](time.md) |
| log、trace、counter 与 sink 接缝 | [obs](obs.md) |
| 通用 RAII、registry、random 等工具 | [util](util.md) |
| caller-owned 稳定节点的零分配链、auto/safe retirement、O(1) 摘链/range handoff | [container_intrusive](container-intrusive.md) |
| flat/small/concurrent 等通用数据结构 | [container](container.md) |
| 非负整数 ceil-division 与可表示 multiple round-up | [math](math.md) |
| 向量、转换、bitpack 和向量算法 | [simd](simd-usage.md) |
| 异步文件、网络和 protocol 机制 | [io](io.md) |
| CPU task/fiber runtime、结构化任务、同步与消息 | [sched](sched.md) |
| device/backend/operator 与 graph | compute/graph 尚未完成，不作为已支持用户面 |

模块页应告诉你何时使用、何时不要使用、ownership、线程安全、失败语义和常见坑。若页面没有
回答这些问题，请把它当作文档缺陷，不要从 benchmark 名称反推产品 API。

## 基本原则

- 使用公开头和 CMake target，不依赖 `src/`、测试 helper 或 benchmark-only 入口。
- 默认走最自然的 C++ surface；只在 quota、placement、诊断或特定并发合同需要时选择低层形状。
- 检查 `Status`/`Result<T>`，不要把运营失败转成断言；程序不变量被破坏时 HYLE 可能 fail-fast。
- 在跨线程释放、fiber/thread-local、async cancellation、借用 buffer 或 NUMA placement 前阅读
  [调用安全合同](usage-contract.md)。
- 不根据旧实施记录中的性能数字选 API；只有与当前 candidate 和合同绑定的证据才有效。
