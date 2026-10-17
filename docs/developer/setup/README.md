# 环境准备

这里说明怎样编译、测试和测量 HYLE。把 HYLE 当依赖使用时，先看[用户指南](../../user/usage-guide.md)。

| 主题 | 说明 |
|---|---|
| [Windows](windows.md) | MSVC/clang-cl、VS/SDK、构建和常见问题 |
| [Linux / WSL](linux-wsl.md) | GCC/Clang、TSan、io_uring、CUDA 与原生 ext4 |
| [macOS](macos.md) | AppleClang、ARM、Metal 和远程验证 |
| [依赖](dependencies.md) | 自动下载的锁定源码与可选系统依赖 |
| [离线构建](offline.md) | 工具链、SDK、依赖 cache/bundle 与断网验证 |

## 需要准备什么

- 编译器基线为 GCC 14+、Clang 18+、MSVC 19.43+，CMake 3.25+；VS 2026 generator 需要 CMake 4.2+。
  准备一个 generator 及所选组件需要的汇编器。Ninja 是可选项，Visual Studio/Makefiles/Xcode 也可用。
- 源码依赖按所选组件/feature 自动取得并放入跨 build 的 `HYLE_DEP_SOURCE_CACHE`；build 的 `_deps`
  保存 binary/subbuild。离线时提供同一锁定身份的 cache 或显式源码 override。
- CUDA 路线需要系统安装 CUDA Toolkit 13.3+；其他 SDK、driver、OpenSSL 等按选用能力准备。
  `HYLE_COMPUTE_ROUTES` 选择 Compute/Graph 路线；TLS、HTTP/2、trace 按实际需要开启。
- 当前 wrapper 只选择已安装工具链并执行命令，不安装编译器或 SDK。系统级初始化脚本是
  [当前工作主线](../current-work.md)中的独立任务。

## 配置、构建与测试

```bash
# Unix：选择已安装的受支持 compiler。
scripts/hyle-unix.sh cmake --preset <linux|macos>
scripts/hyle-unix.sh cmake --build --preset <linux|macos> --parallel 2
scripts/hyle-unix.sh ctest --preset <linux|macos>
```

```powershell
# Windows：已初始化 VS shell，或使用 hyle-windows.ps1。
cmake --preset <windows-vs2026|windows>
```

顶层默认 `HYLE_COMPONENTS=all`；父工程 `add_subdirectory()` 前需显式设置所需组件。
开发 preset 中昂贵的 TLS/HTTP2/trace 默认 OFF。普通测试默认 `HYLE_TEST_INSTRUMENTATION=OFF`，
测试实际产品路径；需要 fault injection 等内部诊断时使用独立 build tree 显式设为 ON。

完整 configure/build/test 可用 `cmake --workflow --preset <windows-vs2026|windows|linux|macos>-check`。
Windows Ninja 使用 wrapper 配合 `windows-ninja`；clang-cl 使用独立的 `windows-clangcl` build tree。

## 独占构建与组件交付

并行会话使用独占 build 目录，避免共享 CMake cache。普通产品用 `-Component <product>`；
仅调试内部 factor 时使用 `-Factor <factor>`，对应 `HYLE_DEV_FACTORS`。factor 不能用于父工程嵌入或
安装，也不会产生用户 `hyle::<factor>` target。

```powershell
$scratch = Join-Path (Get-Location).Path '.scratch\build\container-intrusive-dev'
scripts/hyle-windows.ps1 -Action Configure `
  -Preset windows-ninja -BinaryDir $scratch `
  -Factor container_intrusive
```

影响 source embedding 或 SDK 时，可用所选组件 smoke 检查源码、静态/共享安装包及搬迁后的消费者：

```powershell
$smokeRoot = Join-Path (Get-Location).Path '.scratch\smoke\mem-dev'
scripts/hyle-windows.ps1 -Action SelectedComponentSmoke `
  -Component mem -SmokeRoot $smokeRoot
```

```bash
scripts/hyle-unix.sh cmake -DHYLE_SOURCE_DIR="$PWD" \
  -DHYLE_SMOKE_ROOT=/tmp/hyle-mem-dev -DHYLE_SMOKE_COMPONENT=mem \
  -P cmake/HyleCiSelectedComponentSmoke.cmake
```

测试按改动风险选择。并发改动使用可用的 TSan 等工具检查实际路径；设备和 ISA 路线在真实硬件上验证。
性能测试使用本地文件系统，WSL 构建与压力测试留在 ext4。平台限制见[验证环境](../performance/machines.md)。
