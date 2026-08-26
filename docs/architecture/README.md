# HYLE 架构、愿景与决策

本目录描述 HYLE 为什么存在、需要解决什么问题、应当具有什么形状，以及长期决策为什么如此。
它引领开发方向，但不保存施工流水和一次性测量数字。

HYLE 是面向 AI 推理和工程计算的 C++23 基础加速库。推理引擎是当前首要的可执行需求与性能锚；
HYLE 提供通用的内存、数据结构、SIMD、调度、I/O 和异构计算机制，而模型格式、tokenizer、
sampling、KV 策略与 serving policy 留在推理引擎。第二类工程计算 workload 用于防止公共 API
被单一模型过拟合。

## 项目背景和方向

- [动机与北极星](motivation.md)
- [推理引擎需求锚](inference-engine-north-star.md)
- [项目范围](hyle-scope.md)
- [架构方向与依赖原则](roadmap.md)
- [clay 起源与资产判断](clay-asset-liability.md)
- [参考库与参考职责](references.md)

## 系统级决策

- [构建与打包架构](build-system.md)
- [生命周期、取消和失败边界](lifecycle-design.md)
- [Native FFI 准入策略](native-ffi-policy.md)
- [许可与第三方边界](licensing.md)
- [ggml 内存边界借鉴](borrowing-ggml-memory-model.md)

## 模块设计

- [core](core-design.md)
- [error](error-design.md)
- [platform](platform-design.md)
- [mem](mem-design.md)
- [time](time-design.md)
- [obs](obs-design.md)
- [util](util-design.md)
- [container_intrusive](container-intrusive-design.md)：独立轻量的 auto/safe-link 侵入式链接组件
- [container](container-design.md)：完整数据结构模块，聚合 `container_intrusive`
- [simd](simd-design.md) 与 [vendor 边界](simd-vendor-surface.md)
- [math 边界](math-design.md)
- [sched](sched-design.md)
- [io](io-design.md)
- [compute](compute-design.md)
- [graph](graph-design.md) 与 [数据并行](graph-02-design.md)

## 当前可信度

2026-08-26 起，项目按新的真实消费者标准从 core 开始逐模块重审。Core 已完成首个签署；
`container_intrusive` 当前进入 `CANDIDATE / 复审中`，其他模块设计页暂时描述现有实现和既有决策
输入，不代表已取得新标准的功能、性能或易用性证书。Core 复审已经裁决侵入链表归 container
语义并拆成独立 header-only target；这只是边界与候选产品形状，不能提前视为 v2 模块签署。
唯一施工状态见 [implementation/status](../implementation/status.md)。模块轮到重审时，应删去其
设计页中的历史验收、旧点值和施工叙事，只保留仍约束实现的内容。
