# Selector：无产品实现的审计身份

<!-- PROJECT UNIT REVIEW: AUDITED -->

## 独立工具卡

- owner：change-impact selector；baseline `d4006bf762eef01db6d62f78c794268304519579`。
- trigger：R1 的真实 Windows selector configure 将 `compute_provider_ggml` 纳入所有已实现 factor，
  `tests/CMakeLists.txt:205` 因缺少其测试 fragment 失败。该 factor 仍为 BLOCKED from SCOPED，
  只有空身份 CMake，没有 HYLE 产品源码、公开头、依赖或测试。
- 证据：Windows `C:/hyle-scratch/r1-20260915-d4006bf7/selector-result.json`；configure receipt
  `cded57864fb2ca5fa5d504dff9d83c66a501d512cffb582b532d64bd1c7d8566`。SDK/oneDNN 发现已成功，
  失败来自不存在的测试 owner，并非 CPU/CUDA 隔离产品缺陷。
- scope：只纠正 broad selection 对审计身份的误分类；不更改模块状态流、授权判定、factor DAG、
  package 风险扩展、P1、统计或 evidence identity。
- SCOPED/AUDITED 审计：核对 selector 的 `available/expanded` 与唯一 registry、ggml 空 CMake、
  tests 的强 owner 检查。仅生命周期未到实现、无实际依赖/公开头/实现/测试且 CMake 恰为空身份声明
  时排除 native gate；任何不满足条件的已选组件仍保留强测试检查。
- 有限 FIX VI1：如实给出 audit-only skipped reason；VI2：测试 empty identity 被跳过、真实代码/
  header/dependency/test/生成代码与已进入实现阶段的身份均不能被此规则隐藏。
- 预算：一次源码审计、一项分类修正、一轮工具回归，最多 20 分钟；超出则工具 BLOCKED，R1 的
  剩余具名平台/交付可独立取证，不虚报 selector PASS。
- 验证：`python tests/tooling/validation_selector_test.py`、本次 R1 plan、docs/closure/diff；
  工具修改本身不建产品、不跑 P1。R1 产品 native 重试复用原卡和不可变候选。
- 成本：P0，只读少量 audit-phase owner 的源码元数据，不新增依赖、runtime 成本或公共 API。

七问：无需新依赖/许可证；补齐工具对现有合法生命周期的处理；复用 registry/图/owner 映射；
仅冷元数据检查；无 FFI；不改产品交付；skipped reason 清楚。当前 AUDITED，未签修复通过。
