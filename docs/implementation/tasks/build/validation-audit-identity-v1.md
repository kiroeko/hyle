# Selector：无产品实现的审计身份

<!-- PROJECT UNIT REVIEW: CLOSED -->

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
仅冷元数据检查；无 FFI；不改产品交付；skipped reason 清楚。

## Candidate / CLOSED

- candidate：`9600a44886c5df690900ed533e312e3ebe981b40`。没有产品代码/模块状态变更。
- VI1/VI2 完成；scope、architecture、correctness、delivery、usability/docs、red-team、P0 均 PASS。
  delivery 指 selector 原有 Python/脚本入口，不制造 C++ binary journey；P1 不适用。
- `validation-select.py run --base 0a5e4619 --candidate WORKTREE` 的最终结果为 PASS：docs、closure、
  45 项工具回归；其余 94 gates 跳过，未执行产品构建/性能。原始工具日志见
  [tool-tests.log](../../../../evidence/correctness/validation-audit-identity-v1/tool-tests.log)。
- 首次测试暴露新增反例误改共享 graph fixture cache，已改为 deepcopy 后重跑通过；不是产品规则放宽。
- 候选后复核：真正有实现、头、依赖、测试、生成代码或已进入实现阶段的 owner 不会被此规则隐藏；
  malformed registry 仍走原 SelectionError 边界。ggml BLOCKED/恢复条件不变。
- 限制/reopen：只识别严格的空 `hyle_add_module(identity)`；其它形状保守地继续要求测试 owner。
  新事实证明误跳过或仍误选时定向重开，不扩成新的模块状态系统。
