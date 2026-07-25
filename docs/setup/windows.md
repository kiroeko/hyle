# Windows 环境准备（MSVC）

> 主开发平台之一。工具链 = **MSVC 14.50（VS18 Enterprise）** + Ninja + Release。
> 覆盖：构建 / 测试 / 性能测试三件事 + 依赖（OpenSSL、CUDA）。跨平台依赖总览见 [`dependencies.md`](./dependencies.md)。

## 一、硬门槛（必需，缺了 CMake 直接 FATAL_ERROR）

| 组件 | 要求 | 装法 |
|---|---|---|
| Visual Studio 18（或 Build Tools） | C++ 工作负载（cl / MASM / Windows SDK） | VS Installer 勾「使用 C++ 的桌面开发」 |
| MSVC 编译器 | 19.43+（VS 17.13+） | 随 VS，C++23 `std::expected` 需要 |
| CMake | 3.25+ | `winget install Kitware.CMake` 或随 VS |
| Ninja | 任意近版 | `winget install Ninja-build.Ninja` 或随 VS |
| Git | Git-for-Windows（自带 git-bash） | `winget install Git.Git` |

**fcontext 门控**：sched 模块在 x86_64 + MSVC 下编（`HYLE_HAS_FCONTEXT`），MASM（ml64）做纤程切换汇编，`enable_language(ASM_MASM)` 自动接。

## 二、环境初始化（关键，每个构建 shell 都要先做）

编译前必须先注入 MSVC 环境变量（INCLUDE/LIB/PATH）。两种可靠方式：

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
:: 配置（首次会 FetchContent 自动联网拉依赖到 build\_deps）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

:: 构建单个目标 / 全部
cmake --build build --target hyle_io_test
cmake --build build
```

- 目标名：`hyle_add_test(<mod>)` → `hyle_<mod>_test`；带 NAME 的 → `hyle_<mod>_<name>_test`（如 `hyle_compute_ggml_bridge_test`）。
- MSVC 默认 /W4；改导出宏/类布局注意 C4251/C4273。全局已加 `/Zc:__cplusplus`（否则 C++23 feature-test 宏误触发）+ `/utf-8`（杀 C4819）。
- **首次编 ggml + CUDA 较慢**（编 ggml-cuda 的 MMQ kernel 等）；已 `GGML_CUDA_FA=OFF` 跳掉 ~120 个 FlashAttention 实例省时间。

## 五、测试

```bat
:: shared 库 .dll 与 .exe 同在 build\bin\，无需 PATH juggling
cd build\bin && hyle_sched_test.exe

:: 或用 ctest 跑全部
ctest --test-dir build --output-on-failure
```

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
- 需要 CUDA 的构建显式传 `-DHYLE_COMPUTE_CUDA=ON`；配置打印 `compute CUDA ON -- enabled with Toolkit 13.3`。
- ⚠️ 某 shell 看不到 nvcc = 该 shell 在装 CUDA 前启动的 stale-env，新开 shell 即正常。
- cuDNN/TensorRT 当前里程碑不需要，见 [`dependencies.md`](./dependencies.md)。
