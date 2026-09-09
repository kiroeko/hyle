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
- [已收口单元的推理能力观察清单](inference-engine-capability-watchlist.md)
- [项目范围](hyle-scope.md)
- [架构方向与依赖原则](roadmap.md)
- [clay 起源与资产判断](clay-asset-liability.md)
- [参考库与参考职责](references.md)

## 系统级决策

- [基础库能力完整性](capability-completeness.md)
- [能力展开与组件质因数分解](capability-factorization.md)
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
- [container_intrusive](container-intrusive-design.md)：内部独立的双向/count/forward/hlist 侵入式 factor
- [container](container-design.md)：完整数据结构产品；公开 façade 聚合 intrusive 与普通容器 factor
- [simd](simd-design.md) 与 [vendor 边界](simd-vendor-surface.md)
- [math 边界](math-design.md)
- [sched](sched-design.md)
- [io](io-design.md)
- [compute](compute-design.md)
- [graph（含待裁决的数据并行边界）](graph-design.md)

## 可信度边界

architecture 页面只说明长期需求、边界和决策，不声明当前模块阶段或验收结果。唯一施工状态见
[implementation/status](../implementation/status.md)；设计页存在不代表模块已经取得功能、性能、
易用性或交付证书。模块轮到重审时，应删去设计页中的历史验收、旧点值和施工叙事，只保留仍约束
实现的内容。
