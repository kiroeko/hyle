# HYLE 使用指南

> 适用于预发布的 C++23 surface。模块可用性和稳定度见
> [当前施工状态](../implementation/status.md)。

## 选择交付形式

HYLE 支持三种 C++ 消费方式：

| 方式 | 适合场景 | 用户入口 |
|---|---|---|
| 完整源码 | 需要审计、定制 ISA/backend、LTO 或整体组装 | `add_subdirectory` 后链接 `hyle::<module>` |
| headers + static | 固定工具链/配置，部署时不携带 HYLE 动态库 | `find_package(Hyle COMPONENTS ...)` |
| headers + shared | 多个 C++ 程序共享二进制、独立更新 artifact | `find_package(Hyle COMPONENTS ...)` |

完整构建、安装和离线准备见 [distribution](distribution.md)。这三种都是 C++ 发布形状，不等于
跨语言 FFI。

## 源码集成

```cmake
set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(HYLE_COMPONENTS "mem;container" CACHE STRING "" FORCE)
add_subdirectory(path/to/hyle hyle-build)

target_link_libraries(my_app PRIVATE hyle::container)
```

`HYLE_COMPONENTS` 指定产品根；构建系统自动解析它们的传递闭包。仍应只链接代码实际使用的
`hyle::<module>` target，让 CMake 能发现错误依赖。

## 安装包集成

先用所需 linkage 安装 HYLE，再在用户项目中：

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS mem container)
target_link_libraries(my_app PRIVATE hyle::container)
```

不要手工复制库文件、头路径或第三方链接参数；imported target 承载这些合同。shared 与 static
安装包不能混在同一个 prefix 中假装可互换，消费者工具链也必须满足包记录的 C++/平台合同。

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
| 向量、转换、bitpack 和向量算法 | [simd](simd-usage.md) |
| 异步文件、网络和 protocol 机制 | [io](io.md) |
| task/fiber/coroutine 与同步 | sched；尚无 v2 用户指南，仅供内部试验 |
| 标量 math 补位 | [模块边界仍待裁决](../architecture/math-design.md) |
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
