# Build/package v6：Unix 工具链统一入口

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> Build/package 是开发单元，不是可链接组件，因此不进入 `module-status.json`。本轮只处理反复发生的
> Linux/WSL 编译器选择缺陷；SIMD 已关闭，不因验证入口问题重开。

## Closure card（SCOPED）

- unit / owner / round / kind：`build-package` / build/package maintainers /
  `build-v6-unix-toolchain-entry` / `REOPEN`；
- baseline：`955ebdcd9603ed8b9a42b37994bee060bb6caa02`；前一 round：
  [`build-v5-selected-closure`](build-system-v5-selected-closure.md)；
- trigger：`BUILD-V6-TOOLCHAIN-01`。W1-WSL 已安装 GCC 14.2，但直接运行通用 CMake/selected-component
  smoke 会继承系统默认 GCC 13.3，直到 HYLE 顶层 configure 才失败。相同环境纠正在多个模块验证记录中
  反复出现，说明文档提示与一个专用 `wsl` preset 没有形成维护者统一入口；
- real consumer：在 Linux/WSL/macOS 独占 scratch 中运行 configure、组件 smoke、sanitizer 或 package
  命令的 HYLE 维护者；任意命令仍由调用者拥有，本入口只选择并校验 host C/C++ toolchain；
- scope：一个可组合的 Unix command wrapper；未显式设置 `CC/CXX` 时在 Linux 自动选择受支持的成对
  GCC 14+ 或 Clang 18+，macOS 选择系统 clang；显式选择必须成对且通过 floor 预检；参数原样执行；
  setup、scripts 与 build architecture 文档统一使用；CI 至少自测 wrapper；
- non-goals：安装/升级编译器、改变 GCC/Clang/MSVC floor、改变 CMake preset/option/component/package、
  删除普通 CMake 用法、清理已有 build cache、Windows wrapper、交叉/CUDA host compiler 选择、性能测量；
- tier / cost：`P0 / closure-correctness-v1`。wrapper 在命令启动前做 O(k) 的有限候选探测和一次版本读取，
  随后 `exec` 原命令；不驻留、不改 build graph、无产品运行时成本；
- budget：30 分钟；一个脚本与有限文档/CI 接线；最多一次修复重跑；
- machines：W1-WSL 验证默认 GCC 13 环境自动选 GCC 14.2、显式 GCC 13 fail early、真实 selected SIMD
  configure；M1-MAC 验证 AppleClang；W1-WIN 只跑文档/静态合同，wrapper 不适用；
- gates：shell syntax、自动/显式/半配置/低版本正负路径、参数不丢失、真实 CMake configure、docs；
- stop：上述门通过即关闭；若不能在命令运行前确定受支持的成对 toolchain，则 `BLOCKED`，不在 CMake
  项目初始化后偷偷切换 compiler。

当前状态为 `SCOPED`。下一步只审计现有 Unix 入口和冻结有限 FIX；审计前不改构建产品。
