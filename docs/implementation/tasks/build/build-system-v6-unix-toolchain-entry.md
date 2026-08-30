# Build/package v6：Unix 工具链统一入口

<!-- PROJECT UNIT REVIEW: CLOSED -->

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
| build/package canonical boundary | 新增 Shell 脚本若未登记路径与语言会从 terminal LOC/审计边界漏掉 | `PARTIAL / FIX` | 在 `loc-report.py` 的唯一 build/package boundary 加入 Unix wrapper 与 `Bourne Shell` 并跑 immutable self-check |

public-surface trace：`CMakePresets.json` 的 `linux/wsl/macos`、顶层 compiler floor 与 CI GCC 安装均有自然
任务，全部保留；新增 `scripts/hyle-unix.sh` 只组合这些既有合同，不产生安装后 SDK API。它接受 `--`
后的任意 argv 并用 `exec` 原样替换进程，也提供只读 `--print-toolchain` 供诊断/CI。脚本不使用 `eval`、
不安装软件、不修改 PATH/cache/preset，也不把 GCC 14 写成唯一支持的 Unix compiler。

冻结 FIX 只有：新增上述 wrapper；把 Unix CI configure 接到 wrapper 并加入自动选择 smoke；更新
architecture/setup/scripts 文档；把新脚本登记到既有 canonical build/package 计数边界。任何 compiler
floor、preset cache、CMake component/package 或产品源码变化均超出本轮。当前状态为 `AUDITED`。

## Candidate freeze

- product candidate / tree：`d9e9007ef12c2c9e09addc4dcbbb63611d5321ea` /
  `46661c358d6ebf0c72172cf35d03e43c35e4ce63`；
- `scripts/hyle-unix.sh` 是 100755 Bash command wrapper；自动选择与显式 preflight 后只打印所选 pair 到
  stderr，并以 `exec "$@"` 进入原命令；
- Linux 顺序为 versioned GCC 14..30、versioned Clang 18..30、受支持 unversioned pair；macOS 为系统
  clang。C/C++ driver 必须同 family、同 major 且满足 floor；
- workflow 的 Unix configure 经 wrapper，Linux job 在实际 configure 前验证 unset 自动选择和 half-pair
  fail-closed；Windows 路径不变；
- architecture、setup 与 scripts 文档统一入口、escape hatch 和旧 cache 边界；compiler floor、preset、
  CMake component/package、产品源码和运行时均未变化；
- canonical build/package boundary 同时登记脚本路径与 `Bourne Shell`，避免 terminal LOC 少算。

## Verification 与 red team

| gate | verdict | candidate-bound evidence |
|---|---|---|
| Bash portability | `PASS` | W1-WSL Bash 5 与 M1-MAC 系统 Bash 3.2 的 `bash -n` 均通过 |
| Linux automatic selection | `PASS` | W1-WSL 同时存在默认 GCC 13 与 versioned GCC 14.2；unset `CC/CXX` 精确输出 `gcc-14/g++-14` |
| explicit positive/negative | `PASS` | 显式 GCC 14 pair 执行；GCC 13 pair与只设 `CC` 均在 child command 启动前 exit 2，诊断给出 floor/pair 原因 |
| argv / process composition | `PASS` | WSL/macOS 含空格参数保持一个 argv；实现无 `eval`，最终只有 `exec "$@"` |
| real WSL journey | `PASS` | wrapper 自动选择 GCC 14.2 后驱动 SIMD selected-component smoke；source、relocated shared、relocated static consumer 各 1/1 |
| real macOS journey | `PASS` | unset pair 自动选择 clang/clang++，AppleClang 21 的 fresh Core configure 成功 |
| CI/config contract | `PASS` | workflow YAML 解析；Linux auto/half-pair smoke 接线，Unix configure 使用 wrapper；Windows job 条件与入口未变 |
| docs / closure | `PASS` | 331 个 Markdown、935 个相对链接；16-component registry contract；`git diff --check` |
| P0 cost | `PASS` | 命令启动前有限 O(k) probe + 版本读取，随后进程替换；无 daemon/cache mutation/build-graph/runtime cost |
| immutable LOC | `PASS` | `loc-report.py --module build-package --commit d9e9007e... --self-check --json` |

最终红队裁决：

- **“为何不直接把 generic Linux preset 写死 GCC 14？”** `REJECTED`：Clang 18+ 与发行版自选路径是已
  支持合同，toolchain selection 属外层环境组合，不应污染可复用 preset。
- **“为何不在 `HyleCi*.cmake` 内自动切 compiler？”** `REJECTED`：language/cache 初始化后切换不可靠，
  且会在多个 smoke 复制逻辑；外层任意-command wrapper 是唯一 owner。
- **explicit pair 可能混 family/major。** `FIXED`：两者均在 child command 前拒绝；CMake 顶层仍作第二道
  compiler-ID/floor fail-closed。
- **已有 GCC 13 cache 可能继续用旧 compiler。** `FIXED BY CONTRACT`：三处维护文档要求新的独占 scratch；
  wrapper 明确不删、不重写用户 cache。
- **新增脚本可能漏出 source archive 或 LOC。** `DISPROVED`：完整 source archive按 Git tracked tree 包含
  100755 脚本；canonical build/package path/language 已登记，immutable self-check 计入 141 行 Shell code。
- **CUDA/cross compiler 被 host wrapper误选。** `OUT_OF_SCOPE / ISOLATED`：wrapper只设置 host `CC/CXX`；
  CUDA host 与 cross toolchain 继续由现有显式 CMake inputs/preset 拥有，文档没有把 wrapper宣称为其选择器。
- **hosted workflow 未在本轮实际消耗额度重跑。** `ACCEPTED`：本轮只改命令前置包装，YAML 静态解析、
  WSL真实三交付和macOS真实 configure已覆盖其机制；未来任何 PR/main Linux job会机械执行新增 smoke。

没有未处置的 build/package `MUST` 或 blocker，本轮裁决为 **`CLOSED`**。这项修复不重开 SIMD；它只让
后续模块验证稳定落到项目已经要求的 compiler floor。

可接受限制：wrapper 不安装工具链、不支持把 flags/launcher 塞进 `CC/CXX` 字符串、不改变已存在 CMake
cache，也不为交叉/CUDA toolchain 做策略选择。需要 ccache 等 launcher 时使用 CMake 的专用 launcher
变量；需要非默认 compiler 时给出成对 driver 名或绝对路径。

重开条件：受支持 Linux/macOS 上无法选择已安装且合格的 host pair；wrapper 改写/丢失 argv；CI 或真实
smoke 再次无意使用低于 floor 的默认 compiler；compiler floor/平台矩阵变化；或具名 launcher/toolchain
consumer 证明当前 pair contract 无法表达。未安装 compiler、旧 cache 本身和一次 hosted CI 网络故障不
构成产品 trigger。

Candidate immutable LOC：全库代码 **403,187**；自实现包含测试 **96,041**；自实现不包含测试
**59,898**；build/package 实现 **9,035** 行。
