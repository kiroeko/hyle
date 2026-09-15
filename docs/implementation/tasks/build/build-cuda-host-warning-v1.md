# CUDA 测试的 host warning flags

## 维护卡 / 完成

- owner：build/package；baseline `7bd14d41`；由 CUDA v2 的 Linux 真机验证触发。
- 反例：`hyle_apply_warnings()` 的 MSVC 分支限制在 C/CXX，GNU 分支却对全部语言传递 `-Werror`；
  nvcc 13.3 将它解释为需要 warning-kind 参数的选项，吃掉后续 `-MD` 并失败。重跑不能解决。
- BHW1：GNU/Clang 普通 warning flags 也只用于 C/CXX，与该 helper 的声明和 MSVC 行为一致；CUDA
  device 编译保持自己的诊断，不能把 GCC flags 当作 nvcc flags。不改变 HYLE C++ 的 WERROR、产品
  源码、安装合同或编译器 floor，不调整 sanitizer helper 或性能仪器。
- 一次表示/实现、一次纠正；验证 Linux GCC 14.2/CUDA 13.3 的实际 `.cu` 编译与 CUDA runtime/late-error
  测试、生成命令的 CXX/CUDA 语言区分、现有 build-contract tooling、docs/closure/diff。
- Windows 分支与非 CUDA C++ 编译 flags 不变；不因此重跑全库、三平台矩阵或 P1。

## 结果（2026-09-15）

`f8fc4d61` 完成 BHW1。WSL native ext4/GCC 14.2/nvcc 13.3.73 的实际 Ninja 命令中，C++ 仍取得
`-Wall -Wextra -Wpedantic -Werror`，`.cu` 不再取得这四个 host flags；Release/WERROR 下 CUDA runtime、
late-failure 和逐头通过。ASan/UBSan 配置也实际编译并执行同一 CUDA runtime 测试，不改变 sanitizer
helper。原始 CUDA 输出随 [CUDA v2](../compute/compute-backend-cuda-v2-endpoint-identity.md) 保存。

`python tests/tooling/validation_selector_test.py`：44/44 PASS（82.095 秒）；Windows MSVC 分支和普通
C++ 诊断选项未改。候选后复核确认语言 genex 只收窄错误的 nvcc 输入，不关闭产品 WERROR，不改变
安装 API 或工具链 floor。维护门 PASS，无未决修复；没有运行/声明全库矩阵或新增 P1。
