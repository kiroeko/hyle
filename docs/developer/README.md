# 开发与维护

先用[初始化指南](setup/README.md)准备工具和依赖，再按改动选择构建目标与测试。
日常以本地验证为主；[GitHub CI](../../.github/workflows/build.yml)按需手动运行基础组件检查，
不代表所有 Compute/Graph 后端已通过，也不是开发前置条件。

| 要做的事 | 说明 |
|---|---|
| 初始化、切换配置、构建、测试、安装 | [环境与构建](setup/README.md) |
| 查看当前普通推理、多会话、多任务进度 | [推理进度](inference-progress.md) |
| 查看引擎现状与后续工作 | [推理工程计划](../implementation/inference-roadmap.md) |
| 维护依赖、共享缓存或离线介质 | [依赖](setup/dependencies.md)、[离线构建](setup/offline.md) |
| 定位崩溃、挂起、数值或测量异常 | [排障](debugging/README.md) |
| 比较性能、定位慢项 | [性能测量](performance/README.md) |
| 查询 CUDA decode 测量结果与调查边界 | [decode 记录](performance/decode.md) |
| 修改 SIMD 实现或生成路由 | [SIMD 维护](maintenance/simd-dispatch-guide.md) |
| 了解模块边界和设计取舍 | [架构](../architecture/README.md) |
| 查询已验证模型、算子与边界 | [模型与算子覆盖](../implementation/model-operators.md) |

用户如何调用库放在[用户手册](../user/README.md)，本页只讲维护代码所需的约定。
工作区、验证机器和操作限制以根目录 [AGENTS.md](../../AGENTS.md) 为准。

## 修改代码

- 公开头放在 `include/hyle/<module>/`，实现放在 `src/<module>/`。头应自包含，
  不引用 `src/`，不意外暴露第三方类型。类型用 `PascalCase`，函数与变量用 `snake_case`，宏用 `HYLE_*`。
- 引擎入口和模型接入放在 `engine/`，模型数学与生成／会话政策依赖公共基础模块；
  Compute／Graph 不反向依赖模型，也不为引擎增加第二套执行与退休机制。
- 使用 C++23、RAII、明确的 owner/view 和 `std::span`。较重实现放 compiled core，
  避免每个消费者重复实例化。一个机制有一个所属模块，不复制 error、platform、mem、time 等现有能力。
- 可恢复失败使用所属层的 `Status`/`Result`，保留 OS/backend 错误信息。第三方异常在适配边界转换；
  不变量破坏应暴露缺陷。公开接口在声明附近说明寿命、失效、线程安全、取消和关闭行为。
- 异步 owner 活到最后一次访问结束，同一 allocator/runtime 边界负责释放。原子顺序要能由不变量解释。
- 宏只承担平台分支、属性或必要生成；`HYLE_ASSUME` 不能代替检查，`HYLE_RESTRICT` 需要真实不别名条件。
- 先检查复杂度、分配、搬运、同步和争用，再优化指令。精度、寿命和资源限制是性能比较的一部分。

## 修改构建或公开接口

内部依赖图在 `cmake/HyleComponents.cmake`；模块 CMake 拥有源码与外部依赖。
使用 target 作用域的 include/link/definition，`PUBLIC` 与 `PRIVATE` 反映真实公开依赖，
不强制覆盖父工程 cache。第三方版本、内容身份与许可证由中央依赖 lock 管理。

Compute 能力通过 `hyle_target_compute_capability()` 同时声明宏和包元数据。
后端选择带出 provider、桥接与资源依赖，用户无需手配内部第三方开关。
更完整的装配原因见[构建设计](../architecture/build-system.md)。

公开符号的 DLL 导出宏集中在 `hyle/visibility.h`。增改公开 API 名称后运行：

```text
python scripts/generate-visibility.py --write
python scripts/generate-visibility.py --check
```

库用户链接 `hyle::<product>`，不手配 `*_BUILD`、`HYLE_STATIC` 或 presence 宏。
C++ SDK 不承诺跨编译器、STL、runtime 或构建配置的二进制 ABI，交付合同见[集成指南](../user/distribution.md)。

## 检查与文档

按实际改动选择正常、边界、错误、OOM、寿命、数值、并发或设备测试；需要时使用 sanitizer、
差分、property、fuzz 或 stress。公开头、构建或安装行为变化时，检查有代表性的独立消费者。
性能变化使用[相同工作量和完成边界](performance/README.md)比较。
报告实际运行结果与未验证项；不因文档修订扩大全库构建。

修改用法就更新所属用户指南，改变设计取舍再更新架构说明。命令和默认值引用唯一的操作指南，
不要另写一份配置表；API 的完整签名和细节在公开头，教程保留典型任务与易错边界。
一次性调查先留在任务工作目录；值得复用的根因提炼进排障或设计说明，原始日志与数据留在
[evidence](../evidence/README.md)。同一事实只在所属指南维护一份。
