# 编码与接口约定

HYLE 是 C++23 基础加速库，提供可组合机制；模型格式、生成策略和业务政策由消费者拥有。
改动以实际消费者的问题为依据，运行相关验证并如实报告，未运行项不写成通过。

## 代码与模块

- namespace 使用 `hyle`，私有实现使用 `hyle::detail`；类型用 `PascalCase`，函数与变量用
  `snake_case`，宏用 `HYLE_*`。
- 公开头位于 `include/hyle/<module>/`，实现位于 `src/<module>/`。公开头自包含，使用
  `#pragma once`，不 include `src/` 或意外暴露第三方类型。
- 使用 C++23、RAII、owner/view、`std::span`、strong type 和合理默认值。较重实现放 compiled core，
  避免每个消费者重复实例化。
- 每项机制有一个所属模块。复用现有 error、platform、mem、time 等能力，不复制等义实现或制造依赖环。
- 宏限于语言不能表达的生成、平台分支与属性；`HYLE_ASSUME` 不能替代检查，`HYLE_RESTRICT` 需要真实
  不别名条件，`HYLE_<MODULE>_API` 只标跨共享库边界的符号。
- 注释解释合同、原因和不明显的不变量；避免过时 polyfill、无用途的兼容层及 benchmark 专用公共接口。

## 错误、寿命与并发

- 依赖允许时，可恢复失败返回 `Status`/`Result<T>`；error/bootstrap 层保持无循环依赖的失败形状。
  OS/backend 转换保留原生错误信息，不变量破坏使用 fail-fast。
- HYLE API 默认不抛异常。会抛异常的第三方库只在 adapter 边界捕获并转换，不吞掉 HYLE 自身缺陷。
- 在声明附近说明 ownership、borrow 失效、move/destroy、线程安全、取消、关闭及晚到 completion。
  异步 owner 必须活到所有访问结束；同一 allocator/runtime 边界负责释放。
- 原子顺序来自不变量；共享状态按风险做竞态、进展、取消和关闭检查。不要为缺乏依据的优化降低安全性。

## 性能与资源

- 先检查复杂度、分配、搬运、同步与争用，再优化指令；接口应自然承载真实调用。
- 新增常驻字段、缓存、padding 或间接调用要说明收益与资源成本。可选诊断和特殊优化不增加普通路径成本。
- fast path 与 fallback 保持声明的语义和精度；CPU/ISA 探测由 platform 拥有，backend 处理设备差异。
- 比较需对齐输入、精度、完成边界和工作量，并检查仪器误差。可复现的小幅落后也如实报告和调查。
  方法见[性能比较](../performance/README.md)。

## 构建、交付与验证

- 内部组件依赖由中央 component manifest 声明；模块 CMake 拥有源码和外部依赖，避免全局 include/link。
  `PUBLIC` 与 `PRIVATE` 反映真实公开依赖；面向消费者的宏和依赖由 target 正确传播。
- 第三方依赖保留版本、内容哈希和许可证；同时考虑在线、离线与安装包。具体见
  [依赖说明](../setup/dependencies.md)。
- 产品提供源码、静态库和共享库；共享 C++ SDK 不承诺跨编译器、标准库、runtime 或构建模式的稳定 ABI。
  交付方式变化时验证独立消费者，不从 producer build tree 偶然取文件。
- 按实际风险选择正常、边界、失败、OOM、寿命、数值、并发和设备测试；需要时使用 sanitizer、差分、
  property、fuzz、stress 或真实硬件。修改基础合同还应验证有代表性的直接消费者。
- 运行相关检查及 `git diff --check`，说明失败和未验证项；保留工作树中无关修改。
