# Benchmark build-input selector 修正

归属[独立工具卡](../../../docs/implementation/tasks/build/validation-benchmark-build-inputs-v1.md)。
[52 项工具测试](tool-tests.log)通过；新增覆盖 factor benchmark CMake 与嵌套 helper、混合产品 seam、
全局 benchmark root 以及未知 owner。原有 OS/native、P1 拒绝及证据复用回归一并通过。

[R5 实际 diff 的前后计划](actual-r5-plan-comparison.json)保留误选原因、两项 CMake 的分类和新 skip：
只改 benchmark 的接线不会重跑产品 header/三交付；修改 `src` 产品 CMake 或全局 bench root 仍扩大。
没有 P1 card 不启动计时；没有新产品 native PASS。此前 `6f3ff7ca` 的完整产品交付仍继续执行。
