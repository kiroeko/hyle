# 开发与维护

先用[初始化指南](setup/README.md)准备工具和依赖，再按改动选择构建目标与测试。只需阅读与当前改动相关的页面；实验历史不作为日常开发入口。
日常以本地验证为主；[GitHub CI](../../.github/workflows/build.yml)按需手动运行基础组件检查，
不代表所有 Compute/Graph 后端已通过，也不是开发前置条件。

## 先看整体重点

| 问题 | 唯一入口 |
|---|---|
| 怎样初始化、构建和运行相关测试 | [构建指南](setup/README.md) |
| 项目往哪里做、当前做哪项 | [核心期许](../architecture/inference-principles.md)／[推理进度](inference-progress.md)；完整范围按需查[工程计划](../implementation/inference-roadmap.md) |
| 当前推理性能是什么、差距在哪里 | [decode结论](performance/decode.md) |
| 模块职责如何划分 | [架构入口](../architecture/README.md) |

## 按改动查局部操作

| 当前问题 | 参考 |
|---|---|
| 系统工具、依赖、cache或离线介质 | [构建指南](setup/README.md)中的平台入口、[依赖](setup/dependencies.md)、[离线](setup/offline.md) |
| 崩溃、挂起、数值或测量异常 | [按症状排障](debugging/README.md) |
| 怎么对齐工作量、测速度和内存 | [性能测量方法](performance/README.md) |
| 修改SIMD实现或生成路由 | [SIMD维护](maintenance/simd-dispatch-guide.md) |
| 查询具体模型／算子或术语 | [覆盖参考](../implementation/model-operators.md)、[术语表](../glossary.md) |

用户调用方法在[用户手册](../user/README.md)，工作区与验证机器约定在[AGENTS.md](../../AGENTS.md)。下面只列代码维护的共同边界。

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

修改用法、公开合同或关键设计取舍时，只更新对应指南；当前进度只改[看板](inference-progress.md)。命令和默认值引用唯一操作入口，API签名与局部约束以公开头为准。现有测试／基准日志足以复现时直接引用，不为每片另建报告或归档；需要保留的历史材料按[evidence说明](../evidence/README.md)查证。

## 细节与历史

模块性能旧对照按需查[模块数据](performance/modules.md)，推理逐批实验查[decode历史](../evidence/history/decode-2026-10-31.md)，原始材料查[evidence](../evidence/README.md)。这些用来复核某个结论，不作为开始开发前的必读清单。
