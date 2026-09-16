# Selector：CPU / CUDA 联合验证配置

<!-- PROJECT UNIT REVIEW: AUDITED -->

## 独立工具卡

- owner：change-impact selector 的静态 factor 配置表；baseline `f735271b725d2f3b1d9016dd35a2a9d7b0d43b5e`。
- trigger：R4 C5 的真实 broad plan 同时选择 CPU oneDNN 与 CUDA 两路线。CUDA route 的配置表强制
  CPU OFF，合并后得到 `HYLE_COMPUTE_ONEDNN=ON` + `HYLE_COMPUTE_CPU=OFF`，真实 CMake configure
  在 `HyleComputeDependencies.cmake:14` 拒绝。原日志在
  `C:/hyle-scratch/r4-cuda-20260916/delivery/logs/selector-configure-preflight.log`。
- SCOPED / AUDITED：factor 维护选择明确给出 roots，CUDA factor 本身没有 CPU 依赖；CPU OFF 是
  独立产品隔离验收的输入，不是每次 CUDA factor 验证的全局必需条件。其它四个 CUDA factor 仅要求
  CUDA ON。移除两项多余的 CPU OFF，联合验证采用既有 CPU 默认值；原 C5 CUDA-only 交付仍显式 OFF。
- 有限 FIX VC1：只删除两个 CUDA route 的 CPU OFF 配置；VC2：加入 CUDA/oneDNN 同时选中时的回归。
  不改 selector 算法、owner、状态流、图、选择范围、receipt、统计、P1 或产品构建规则。
- 预算：20 分钟，配置表两项修正、一个反例和一轮工具测试；超出明确登记工具 blocker。
- 验证：selector 工具回归、真实 broad plan 的 configure 纠正、docs/closure/diff。工具改变自身
  不触发产品重建/P1；C5 的 native gates 仍按原卡执行，不因本卡伪造 PASS。
- P0：只改变离线维护配置，不改变产品运行时、CPU 默认规则或 CPU 已批准性能例外。
- stop / reopen：上述有限项通过后 CLOSED；只有再次出现合法 factor 组合的矛盾 configure 才定向重开。

七问：无新依赖；修复已复现的必要验证入口；复用 factor 图和现有配置表；仅离线 P0；无 FFI；
脚本入口保持原样；错误原因与隔离验证边界明确。R4 产品的三交付结果保留，完整 P1 尚未运行。

## 工具候选

VC1/VC2 已实现；新增回归在原配置稳定 FAIL，两项静态配置纠正后，既有工具套件 **46/46 PASS**。
selector 仅执行 docs/closure/tool 三项，其余 108 项跳过，不建产品或跑 P1。原始
[CMake 反例](../../../../evidence/correctness/validation-cuda-cpu-coexistence-v1/selector-configure-preflight.log)、
[修正前](../../../../evidence/correctness/validation-cuda-cpu-coexistence-v1/tool-regression-before.log)、
[修正后](../../../../evidence/correctness/validation-cuda-cpu-coexistence-v1/tool-regression-after.log)与
[selector 结果](../../../../evidence/correctness/validation-cuda-cpu-coexistence-v1/tool-fix-validation.json)已保留。
下一步在提交后的配置表上复核真实 broad configure，然后停止本工具卡；不追加工具机制。
