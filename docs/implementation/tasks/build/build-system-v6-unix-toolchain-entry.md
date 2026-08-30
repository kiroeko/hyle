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

## Audit 与有限 FIX（AUDITED）

| surface / journey | current fact | verdict | frozen action |
|---|---|---|---|
| 顶层 compiler floor | configure 已拒绝 GCC <14、Clang <18 | `COMPLETE / KEEP` | 保留为最终 fail-closed；不承担选择职责 |
| `wsl` preset | 精确写入 `gcc-14/g++-14`，但只适用于该名称和 preset journey | `COMPLETE / KEEP` | 继续作为方便入口，不强迫所有 Linux 用户使用 |
| generic `linux` preset / direct CMake | 正确允许任意受支持 toolchain，但要求每次手写 `CC/CXX` | `PARTIAL / ROUTE` | 由可组合 wrapper 在命令前选择，不把 generic preset 锁死到 GCC |
| `cmake -P HyleCi*` smoke | script mode 再 spawn configure，不能在 project 初始化后可靠切换 compiler | `MISSING / FIX` | 由同一外层 wrapper 包住任意 CMake/CTest/自定义命令；CMake helper 不偷偷猜环境 |
| explicit `CC/CXX` | CMake 会采用，但只设一个或显式低版本仍到 configure 才报错 | `PARTIAL / FIX` | wrapper 要求成对、同 compiler family 且在执行命令前满足 floor |
| Linux auto selection | 机器常同时有默认 GCC 13 与 versioned GCC 14 | `MISSING / FIX` | 未显式选择时优先成对 versioned GCC 14+，其次 Clang 18+，最后检查 unversioned pair |
| macOS | 系统 AppleClang 是受支持 canonical compiler | `PARTIAL / FIX` | 未显式设置时选择系统 clang/clang++ 并用同一 floor preflight |
| CI / maintainer docs | CI 自己写 `GITHUB_ENV`，setup 多处提示手写，但没有验证统一入口 | `PARTIAL / FIX` | Unix configure 经 wrapper；setup/scripts/architecture 只保留一个维护者入口并说明 escape hatch |
| 已存在 build cache | CMake compiler identity 不允许原地更换 | `COMPLETE / KEEP` | 明确 wrapper 只管新 configure；旧 cache 用新 scratch，不自动删除用户数据 |

public-surface trace：`CMakePresets.json` 的 `linux/wsl/macos`、顶层 compiler floor 与 CI GCC 安装均有自然
任务，全部保留；新增 `scripts/hyle-unix.sh` 只组合这些既有合同，不产生安装后 SDK API。它接受 `--`
后的任意 argv 并用 `exec` 原样替换进程，也提供只读 `--print-toolchain` 供诊断/CI。脚本不使用 `eval`、
不安装软件、不修改 PATH/cache/preset，也不把 GCC 14 写成唯一支持的 Unix compiler。

冻结 FIX 只有：新增上述 wrapper；把 Unix CI configure 接到 wrapper 并加入自动选择 smoke；更新
architecture/setup/scripts 文档。任何 compiler floor、preset cache、CMake component/package 或产品源码
变化均超出本轮。当前状态为 `AUDITED`。
