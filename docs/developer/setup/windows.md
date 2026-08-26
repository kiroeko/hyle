# Windows 环境准备（MSVC / clang-cl）

> 主开发平台之一。当前主机使用 **MSVC 14.50（VS18 Enterprise）**；HYLE 的合同是
> MSVC 19.43+。CMake 4.2+ 可生成 VS 2026 solution；更旧 CMake 走 VS 2022 generator 回退。
> 覆盖：构建 / 测试 / 性能测试三件事 + 依赖（OpenSSL、CUDA）。跨平台依赖总览见 [`dependencies.md`](dependencies.md)。
> 无网机器的 Build Tools layout、依赖介质与断网验收见 [`offline.md`](offline.md)。

## 一、硬门槛（必需，缺了 CMake 直接 FATAL_ERROR）

| 组件 | 要求 | 装法 |
|---|---|---|
| Visual Studio / Build Tools | C++ 工作负载（cl / MASM / Windows SDK） | VS Installer 勾「使用 C++ 的桌面开发」 |
| MSVC 编译器 | 19.43+（VS 17.13+） | 随 VS，C++23 `std::expected` 需要 |
| clang-cl frontend | 18+（仅 `windows-clangcl` 可移植性门） | VS Installer 安装「C++ Clang tools for Windows」，或使用受审 LLVM 安装，并确保 `clang-cl` 在 PATH |
| CMake | 3.25+ | `winget install Kitware.CMake` 或随 VS |
| Generator | VS generator 自带；Ninja 可选 | 新版 MSVC 尚未被当前 CMake generator 识别时用 `winget install Ninja-build.Ninja` |
| Git | Git-for-Windows（自带 git-bash） | `winget install Git.Git` |

**fcontext 门控**：CMake 先把目标平台是否有已接线汇编计算为内部
`HYLE_FCONTEXT_SUPPORTED`；只有 component 闭包实际选择 sched 时，内部 `HYLE_HAS_FCONTEXT` 才为真并
启用汇编语言。x86_64 Windows 的 cl.exe 与 clang-cl 都走 MSVC ABI + MASM（ml64）实现。两者都不是
用户可写的 cache 选项；要不要 sched 由 `HYLE_COMPONENTS` 和依赖闭包决定。

## 二、环境初始化（关键，每个构建 shell 都要先做）

Visual Studio generator 会选择对应 IDE 实例，但在同一 VS 内并存多套 v143 时不保证选择数值最高
的 toolset；Ninja 则完全使用 shell 中的 `cl.exe`。推荐统一走仓库脚本：它为 Ninja 用 `vswhere`
选最新已安装 C++ toolchain，为 VS 2022 solution 枚举并显式选择最高的 MSVC 14.43+，都不写死
Edition/安装路径：

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset windows-ninja
scripts/hyle-windows.ps1 -Action Build -Preset windows-ninja --parallel 2
scripts/hyle-windows.ps1 -Action Test -Preset windows-ninja --output-on-failure

# 可移植性复核：Clang frontend + Windows/MSVC ABI，不能复用 MSVC cache
scripts/hyle-windows.ps1 -Action Configure -Preset windows-clangcl
scripts/hyle-windows.ps1 -Action Build -Preset windows-clangcl --parallel 2
scripts/hyle-windows.ps1 -Action Test -Preset windows-clangcl --output-on-failure

# CMake 4.2+ + VS 2026：原生 v145 solution
scripts/hyle-windows.ps1 -Action Configure -Preset windows-vs2026 -DHYLE_COMPONENTS=core

# 较旧 CMake：VS 2022 solution，脚本选择最高合格 v143
scripts/hyle-windows.ps1 -Action Configure -Preset windows -DHYLE_COMPONENTS=core
scripts/hyle-windows.ps1 -Action Build -Preset windows --target hyle_core_tests --parallel 2
```

也可手工初始化：

```bat
:: 方式 A（推荐，最全）：VsDevCmd
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64

:: 方式 B：vcvars64（本仓库脚本一直用这个，也 OK）
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
```

- ⚠️ **INCLUDE 没注入的症状**：ninja 用 cache 里的 cl 报 `C1083: 找不到 <chrono>/<stdbool.h>`——**不是缺头，是环境没注入**（新开 shell 没 call vcvars/VsDevCmd，或在半配置的 build `_deps` 状态下 rebuild）。对策：确保每个跑 cmake 的 shell 都先 call 一次。
- VsDevCmd 可能打印 `vswhere.exe 不是内部或外部命令`——**无害**，有 fallback，照样正确注入。

## 三、git-bash 调 cmd 的坑（本仓库常用 git-bash 驱动）

- git-bash 的 MSYS 路径转换会把 `cmd.exe /c` 的 `/c` 当路径 `C:\` 吞掉 → 进交互模式不执行。**必须双斜杠 `cmd.exe //c "..."`**。
- cmd 等 stdin 会挂起，追加 `< /dev/null`。
- 最稳套路：把 `call vcvars >nul && cmake ... && echo EXIT=%errorlevel% > log` 写进一个 `.bat`，用 `cmd.exe //c "bat路径" < /dev/null` 跑，再**读 log 文件**看结果（别靠 stdout——cmd 横幅糊屏 + 中文编码乱）。

## 四、构建

```bat
:: CMake 4.2+：原生 VS 2026 / v145 solution
powershell -File scripts/hyle-windows.ps1 -Action Configure -Preset windows-vs2026

:: 旧 CMake 回退：生成 VS 2022 solution，并显式选择最高合格 v143
powershell -File scripts/hyle-windows.ps1 -Action Configure -Preset windows

:: 或已初始化 dev shell 的 Ninja 入口
cmake --preset windows-ninja

:: 构建单个目标 / 全部
cmake --build --preset windows --target hyle_io_test --parallel 2
cmake --build --preset windows --parallel 2
```

VS 2026 preset 生成 `build-windows-vs2026/hyle.slnx`；VS 2022 回退生成
`build-windows/hyle.sln`。两者都是 multi-config generator，preset 的 build/test 阶段固定选 Release。

直接运行 `cmake --preset windows` 只适合 VS 默认 v143 本身已经是 14.43+ 的机器；CMake 不承诺从
side-by-side toolset 中挑最高版本。例如 VS 2022 同时有 14.38/14.44 时，默认可能仍落到 14.38。
包装脚本会探测后传 `-T version=14.44`（具体值随机器变化），若旧 cache 使用了另一 toolset，
只用 `--fresh` 重建 CMake cache/生成文件，不删除源码、依赖 checkout 或编译产物。
`Visual Studio 18 2026` generator 从 CMake 4.2 才存在；CMake 4.1 及更早版本不能通过安装 IDE 获得
新 generator。此时 VS 2026 仍可打开 VS 2022 solution，命令行最新编译器走 Ninja 入口。

`windows-clangcl` 仍由同一 wrapper 导入 Windows SDK、MSVC headers/libraries 和 MASM 环境，但
将 C/C++ frontend 固定为 PATH 中的 `clang-cl`；它与 cl.exe 使用同一 Windows ABI，却不是
`CMAKE_CXX_COMPILER_ID=MSVC`。因此 ISA/告警逻辑必须按 compiler ID 区分，不能把 CMake 的
`MSVC` ABI-family 变量误当成 cl.exe 身份。

```powershell
Get-Command clang-cl
clang-cl --version  # 必须为 18+
```

wrapper 只验证并使用已安装的 frontend，不替用户安装 LLVM；不要让 `windows-clangcl` 复用任何 cl.exe
build tree。

- 目标名：`hyle_add_test(<mod>)` → `hyle_<mod>_test`；带 NAME 的 → `hyle_<mod>_<name>_test`（如 `hyle_compute_ggml_bridge_test`）。
- MSVC 默认 /W4；改导出宏/类布局注意 C4251/C4273。全局已加 `/Zc:__cplusplus`（否则 C++23 feature-test 宏误触发）+ `/utf-8`（杀 C4819）。
- **首次编 ggml + CUDA 较慢**（编 ggml-cuda 的 MMQ kernel 等）；已 `GGML_CUDA_FA=OFF` 跳掉 ~120 个 FlashAttention 实例省时间。

## 五、测试

```bat
powershell -File scripts/hyle-windows.ps1 -Action Test -Preset windows --output-on-failure

:: 仅在当前 shell 已手工 call VsDevCmd/vcvars64 后可直接运行：
ctest --preset windows --output-on-failure
```

header-self-contained 测试会在 CTest 期间再次调用 `cl.exe`；普通 PowerShell 即使复用已配置的
build tree 也没有 MSVC 的 `INCLUDE`。仓库 wrapper 会为该次 Test 动作重新导入环境，且不会把
环境永久写回调用方。

- **GPU 测试（compute）**：CUDA runtime DLL 需在 PATH。toolkit 的 `bin` 通常已入 PATH（`where nvcc` 命中即可）。`hyle_compute_ggml_bridge_test` 在 RTX 4070 上跑零拷贝 matmul。
- **TLS 测试**：需 OpenSSL DLL 在 PATH，见第七节。

## 六、性能测试（bench）

```bat
:: 先拉固定版 Highway/xsimd；默认根为 C:/public/project-hyle
powershell -ExecutionPolicy Bypass -File scripts/fetch-bench-vendors.ps1

:: bench 用独立 build-bench，不能污染日常 build 的产品 cache。
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DHYLE_BUILD_BENCH=ON
cmake --build build-bench --target hyle_sched_bench
cd build-bench\bin && hyle_sched_bench.exe
```

- 所有 SIMD bench 都用 `HYLE_BENCH_VENDOR_DIR` 下的固定版 Highway/xsimd；Windows 默认
  `C:/public/project-hyle`，改目录时 CMake 也传 `-DHYLE_BENCH_VENDOR_DIR=<root>`。
- bench 目标名：`hyle_add_bench(<mod>)` → `hyle_<mod>_bench`。
- **Windows 性能数字注意**：Release + 关掉后台干扰；nanobench 自带统计。跨平台对标以 Linux 原生 ext4 为准（Windows NTFS 亦可信，但重 IO 对比看 Linux）。
- 当前 Windows 与 WSL 是同一 `W1` 物理主机，性能采样必须互斥；Windows 的工程/formal 角色、
  P-core、NTFS 与电源合同见[性能机器合同](../performance/machines.md)。

## 七、可选依赖：OpenSSL（TLS）+ CUDA

### OpenSSL（vcpkg，Windows 标准做法）

Windows 无系统级 MSVC OpenSSL 开发库（Git 自带的是 mingw64 运行时，链不进 MSVC）。用 vcpkg：

```bash
git clone --depth 1 https://github.com/microsoft/vcpkg.git C:/public/vcpkg
cmd.exe //c "C:\public\vcpkg\bootstrap-vcpkg.bat -disableMetrics"
cmd.exe //c "C:\public\vcpkg\vcpkg.exe install openssl:x64-windows"
```

配置时指向它（对现有 build 目录也能加）：

```bat
cmake -S . -B build -DHYLE_IO_TLS=ON ^
  -DOPENSSL_ROOT_DIR=C:/public/vcpkg/installed/x64-windows
```

成功打印 `HYLE: io TLS ON -- enabled with OpenSSL 3.x`。**运行 TLS 测试**加 DLL 到 PATH：

```bat
set PATH=C:\public\vcpkg\installed\x64-windows\bin;%PATH%
cd build\bin && hyle_io_tls_test.exe
```

### CUDA（GPU 后端）

- 装 CUDA Toolkit **13.3**（本仓库统一版本）到 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`，确保 `bin` 入 PATH + `CUDA_PATH` 指 v13.3。
- 需要 CUDA 的本机构建用 `windows-cuda`（或显式传 `HYLE_COMPUTE_CUDA=ON`）；本机 preset 使用
  `CMAKE_CUDA_ARCHITECTURES=native`，SDK 则必须写明确 architecture 列表。
- ⚠️ 某 shell 看不到 nvcc = 该 shell 在装 CUDA 前启动的 stale-env，新开 shell 即正常。
- cuDNN/TensorRT 当前里程碑不需要，见 [`dependencies.md`](dependencies.md)。
