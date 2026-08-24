# container 旧性能表迁移说明

本文件曾把 mem、container 当前表、优化过程和历史点值混在一起，因而在接口和 benchmark
形状更新后产生了互相冲突的“当前结论”。2026-07-27 严格收口已停止维护这份重复表。

- container 历史三平台数据、当前 freshness、方法、诚实边界与复跑命令：
  [`benchmarks-container.md`](./benchmarks-container.md)
- mem 性能唯一详页（当前新鲜度见页首）：[`benchmarks-mem.md`](./benchmarks-mem.md)
- 实现演进与被否决实验：
  [`../milestones/container-module-closure.md`](../milestones/container-module-closure.md)

旧表仍可从 git 历史读取，但不再放在 performance 目录中冒充第二份权威性能账本。
