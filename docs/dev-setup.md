# HYLE 开发环境准备（面向贡献者）

> 这份文档只回答一个问题:**从零把一台机器准备成能编译、测试、验证 HYLE 的开发环境。**
> 面向**开发 HYLE 本身**的贡献者(不是把 HYLE 当依赖用的库使用者——那看 [`guide/usage-guide.md`](./guide/usage-guide.md))。
> 内容以**实测**为准,慢变;命令/版本变了要回填。速查版见 [`architecture/implementation-cheatsheet.md`](./architecture/implementation-cheatsheet.md) §7。

## 一、工具链硬门槛（CMake 会 FATAL_ERROR 挡住不达标的）

| 组件 | 最低版本 | 为什么 |
|---|---|---|
| CMake | **3.25+** | `cmake_minimum_required`;multi-config genex / `GENERATOR_IS_MULTI_CONFIG` |
| C++ 编译器 | **GCC 14+** / **Clang 18+** / **MSVC 19.43+(VS 17.13+)** | C++23 `std::expected` 等;低于此 CMake 直接 FATAL_ERROR(见 CMakeLists.txt 顶部 toolchain gate) |
| C 标准 | C17 | vendored C 依赖(fcontext asm shim / XXH3)不回退到编译器默认 C |
| 汇编器 | MASM(ml64,Windows)/ GAS(Linux) | fcontext 纤程切换需要;MSVC 下 `enable_language(ASM_MASM)` |
| Ninja | 任意近版 | 单配置 Ninja 是本项目默认生成器 |

**fcontext 门控**:sched 模块只在 x86_64 且 (MSVC 或 Linux) 时编译(`HYLE_HAS_FCONTEXT`)。arm64 / macho asm 已 vendored 但未接线(无测试机)。

## 二、Windows(MSVC）

主开发平台之一。工具链 = **MSVC 14.50(VS18 Enterprise)** + Ninja + Release。

### 2.1 基础

- 装 Visual Studio 18(或 Build Tools),含 C++ 工作负载(cl / MASM / Windows SDK)。
- CMake、Ninja、Git(Git-for-Windows 自带 git-bash)。

### 2.2 环境初始化(关键)

编译前必须先注入 MSVC 环境变量(INCLUDE/LIB/PATH)。**可靠的一句是 VsDevCmd**:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat" -arch=x64
```

- **别用** `VC\Auxiliary\Build\vcvars64.bat`——它是个 stub,`call` 后 INCLUDE 为空,ninja 会用 cache 里的 cl 报 `C1083 找不到 <chrono>`(不是缺头,是 INCLUDE 没注入)。
- VsDevCmd 可能打印 `vswhere.exe 不是内部或外部命令`——**无害**,它有 fallback,照样正确注入环境。

### 2.3 git-bash 调 cmd 的坑

- git-bash 的 MSYS 路径转换会把 `cmd.exe /c` 的 `/c` 当路径 `C:\` 吞掉 → cmd 进交互模式不执行。**必须双斜杠 `cmd.exe //c "..."`**。
- cmd 等 stdin 会挂起,追加 `< /dev/null`。
- 最稳的套路:把 `call VsDevCmd >nul && cmake --build ... && echo EXIT=%errorlevel% > log` 写进一个 `.bat`,`cmd.exe //c "bat路径" < /dev/null` 跑,再读 log 文件看结果(别靠 stdout,cmd 横幅会糊屏 + 中文编码乱)。

### 2.4 配置 + 构建 + 测试

```bat
:: 配置(首次;OpenSSL 见第四节)
cmake -S . -B build -G Ninja

:: 构建单个目标
cmake --build build --target hyle_io_test

:: 跑测试 exe(shared 库 .dll 与 .exe 同在 build\bin\,无需 PATH juggling)
cd build\bin && hyle_io_test.exe
```

- 目标名:`hyle_add_test(<mod>)` → `hyle_<mod>_test`;`hyle_add_bench(<mod>)` → `hyle_<mod>_bench`(需 `-DHYLE_BUILD_BENCH=ON` 重配)。
- MSVC 默认 /W4;改导出宏/类布局注意 C4251/C4273。
- 全局已加 `/Zc:__cplusplus`(否则 C++23 feature-test 宏误触发)+ `/utf-8`(杀 C4819)。

## 三、Linux / WSL（GCC14 + Sanitizer)

lock-free 代码必须过 **TSan**,这条在 Linux 上跑。本机用 WSL Ubuntu 24.04。

### 3.1 装 GCC14

```bash
sudo apt install g++-14 gcc-14        # Ubuntu 24.04 源里有
export CXX=g++-14 CC=gcc-14           # 默认 gcc-13 过不了 CMake 门槛
```

### 3.2 独立 build 目录 + TSan

```bash
# 常规 Linux build
cmake -S . -B build-wsl -G Ninja -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build build-wsl

# TSan build(lock-free/并发必过)
cmake -S . -B build-wsl-tsan -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DHYLE_SANITIZER=tsan
cmake --build build-wsl-tsan
setarch -R ./build-wsl-tsan/bin/hyle_sched_test   # -R 关 ASLR,稳定 TSan
```

- `HYLE_SANITIZER` 取值 `off|asan|tsan|ubsan`(也接受 `address/thread/undefined/none` 别名;**拼错会 FATAL_ERROR**,不会静默降级成 off 假 PASS)。

### 3.3 ⚠️ 9p 铁律（血泪）

WSL 下**一切重 IO / 压力测试**必须在**原生 ext4**(把仓库 `git clone` 或 `rsync` 到 `/root/hyle` 再跑),**别在 `/mnt/c`**——那是 9p 文件系统,不稳、CPU 打不饱和,性能数字全废。轻量单测在 `/mnt/c` 可以,重 io / bench 一定挪 ext4。

## 四、可选依赖:OpenSSL（io-08 TLS）

TLS 是**可选能力**:`find_package(OpenSSL)` 找到就编 `tls.cc` + 打开 `HYLE_IO_TLS`;找不到就完全不编,不想要 TLS 的用户**零 OpenSSL 依赖**(同 `HYLE_TRACE`/tracy 的"opt-in 能力 = opt-in 依赖"规则)。**动态链**——安全库应跟随系统 CVE 更新,不该静态焊进制品。

### 4.1 Linux:系统包,零配置

```bash
sudo apt install libssl-dev           # WSL 上 3.0.13
```

装完 `find_package(OpenSSL QUIET)` 自动找到,重配即打开 TLS。

### 4.2 Windows:vcpkg（正常 Windows 开发者的标准做法）

Windows 没有系统级 OpenSSL 开发库(Git 自带的 mingw64 只有运行时 DLL,是 GCC 制品链不进 MSVC)。用 **vcpkg**(微软官方 C++ 包管理器)装 MSVC 兼容的 OpenSSL:

```bash
# 1. 装 vcpkg(本仓库约定放 C:/public/vcpkg)
git clone --depth 1 https://github.com/microsoft/vcpkg.git C:/public/vcpkg
cmd.exe //c "C:\public\vcpkg\bootstrap-vcpkg.bat -disableMetrics"

# 2. 装 OpenSSL(x64-windows = 动态 triplet,出 .lib import + .dll 运行时)
cmd.exe //c "C:\public\vcpkg\vcpkg.exe install openssl:x64-windows"
# 装到 C:/public/vcpkg/installed/x64-windows/{include,lib,bin}
```

配置时指向它(两种等价方式):

```bat
:: 方式 A:直接指 OPENSSL_ROOT_DIR(对现有 build 目录也能加,最省事)
cmake -S . -B build -DOPENSSL_ROOT_DIR=C:/public/vcpkg/installed/x64-windows

:: 方式 B:挂 vcpkg toolchain(首次配置生效,跨平台 vcpkg.json 统一管依赖时用)
cmake -S . -B build -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=C:/public/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
```

配置成功会打印 `HYLE: OpenSSL 3.6.3 found -- io-08 TLS enabled`。

**运行 TLS 测试**:dynamic triplet 运行时要 OpenSSL 的 DLL,把 vcpkg 的 bin 加进 PATH:

```bat
set PATH=C:\public\vcpkg\installed\x64-windows\bin;%PATH%
cd build\bin && hyle_io_tls_test.exe
```

- vcpkg 目录**自包含、无硬编码路径**,想挪位置直接 `mv` 后重新指向即可(vcpkg.exe 相对定位)。
- 其他 MSVC 兼容途径:Shining Light 预编译安装器(`winget install ShiningLight.OpenSSL.Dev`,写注册表 `find_package` 自动找)或源码 `perl Configure VC-WIN64A`。**vcpkg 是新项目标准做法**——后续第三方(picohttpparser/ggml 等)也能统一走它。

> 面向 **HYLE 库使用者**的 install/DLL 分发(把运行时 DLL 一起 install/export 到用户机器)是收口阶段(close-01)的事,不属任何单个 io 里程碑。

## 五、验证一切就绪

```bash
# Windows(VsDevCmd 环境内)
cmake -S . -B build -G Ninja -DOPENSSL_ROOT_DIR=C:/public/vcpkg/installed/x64-windows
cmake --build build
# 看到 "OpenSSL ... found -- io-08 TLS enabled" + 各模块编过

# Linux/WSL
cmake -S . -B build-wsl -G Ninja -DCMAKE_CXX_COMPILER=g++-14 && cmake --build build-wsl
```

跑几个测试确认三支柱活着(sched/io/container),TLS 装了就再跑 `hyle_io_tls_test`。lock-free 改动过 WSL TSan。绿了就能开工。
