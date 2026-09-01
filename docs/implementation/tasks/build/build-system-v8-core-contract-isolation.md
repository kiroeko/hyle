# Build/package v8：Core 编译合同隔离

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> Build/package 是横向开发单元，不进入 `module-status.json`。本页只处理一次已经由当前 Core-only
> consumer 复现的编译策略泄漏；Core 产品状态与本轮重签事实保存在对应 Core v4 task 和 registry。

## Trigger

2026-09-01 对已关闭 Core 的 build/package 交叉审计证明：

- `hyle::core -> hyle::_factor_core -> hyle::_build_contract` 的 source/static/shared 三种真实路径都会
  取得全局 `hyle_build_contract`；
- 该 target 在 x86 GCC/Clang/AppleClang/clang-cl consumer 上允许 `SSSE3`，但唯一产品原因是
  Container 的公开 inline probe；Core v3 已明确把这一 compile-target 条件归还 Container；
- MSVC 路径又显式追加 `/EHsc`，使只使用无异常 Core headers 的源码消费者不能自然选择
  exception-disabled TU。普通 CMake MSVC 配置本来已有默认异常模式，因此该选项同时是重复且过宽的
  usage requirement；
- fresh W1-WIN component contract 与 Core source/relocated shared/static smoke 均通过，说明问题不是
  缺 target、漏头或坏包，而是成功交付了错误归属的编译策略。

## 有界范围

- 让 `hyle_build_contract` 只承载所有 HYLE C++ target 都真实需要的最小编译合同；
- 从 Core-only source/static/shared consumer 移除 ISA-specific 与强制 exception-mode 选项；
- 把现有 SSSE3 usage requirement 直接投影到真正拥有 inline probe 的 Container factor，保持
  Container 正常 consumer 的有效编译输入与代码路径不变；
- 增加机器合同，证明 Core-only 无 ISA/exception 泄漏、exception-disabled Core TU 可编译，且
  Container 仍直接拥有 SSSE3 requirement；
- 更新 architecture/user 文档并重放 Core 与 Container 的适用交付门。

非目标：改变 Core/Container API、算法或布局；承诺完整 HYLE 在 `-fno-exceptions` 下可构建；改变
SIMD fixed/fat ISA helper；重新选择项目最低 CPU；重审 Container 性能；修改 Native FFI、依赖锁或
物理 archive/DSO 粒度。

## AUDITED contract

冻结的投影是：`hyle_build_contract` 移除 SSSE3 与 `/EHsc`；Container factor 直接 `PUBLIC` 取得与原来
同一 compiler-conditioned SSSE3 option；Core package sample 增加 exception-disabled journey；中央
合同同时断言 Core 的负空间与 Container 的正向 owner。普通 MSVC 仍使用父工程/CMake 默认异常模式，
显式关闭异常的 Core target 不再被 HYLE usage requirement 覆盖。

验证固定为三平台 Core tests、Core selected source/static/shared、Container selected
source/static/shared、中央 component contract、安装 target/header/artifact inventory、文档与 closure
checker。Container 只证明有效 flag/route 等价，不续签其算法或性能。

审计同时逐项检查 Core 公开 API 与真实 HYLE/推理消费者、Highway、Abseil、Folly 的同类任务。没有
发现由 benchmark 命名、只为计时存在或给常态对象添加测量 metadata 的公开入口，也没有需要本轮新增
的易用性能力。冻结实现集只有：移除 `_build_contract` 的 SSSE3 与 `/EHsc`，把原 SSSE3 option 原样
投影到 `hyle_container PUBLIC`，增加 Core-only no-exception package journey 和对应机器/文档合同。
当前为 `AUDITED`；任何额外产品或构建策略变化均须重新审计。

实施已严格落在冻结集内：全局 target 只剩最低共同合同，Container 直接持有 SSSE3，Core sample 与
selected/component smoke 具备 normal/no-exception 双 journey，architecture/user 文档已同步。Windows
与 WSL 的冻结前预检通过。不可变 product candidate 已冻结为
`350fc39a25adf953c0f13c7404635b4624d5c6f1` / tree `dafe004eb0651ac0fccc500c658b7cd577e52f4f`；
下一步从该身份运行三平台验证。本段不提前把预检写成 terminal verdict。

## Candidate 1 rejected by Mac contract projection

候选 `350fc39a` 的 Core/Container 产品 journeys 在三平台均通过，但 M1-MAC 的公共 component contract
在生成 SIMD 子工程时没有传 HYLE 已规定的 macOS 14.4 deployment floor，因此配置在中央断言前失败。
这不是把相邻失败塞给 Core：本 build/package development unit 自己拥有该脚本。有限修正只让所有
child configure 继承中央 floor；不改变 floor、产品图、算法、API 或前述 build-contract 修复。候选已
按协议退回 `AUDITED`，修正后必须重建 identity。

有限修正已落地，并在 W1-WIN/W1-WSL 的完整 component contract 冻结前预检通过；该预检没有把
M1-MAC 写成 PASS。第二候选现已冻结为
`7ae3de400b5d7b598972ebf4913d336b60517483` / tree `b6c04d0e97c93724a9136c34e4ea55b6e4449092`；
所有正式 gate 从该身份重放。
