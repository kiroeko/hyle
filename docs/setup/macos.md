# macOS 环境准备（Apple Silicon）

> ARM/NEON + kqueue 真机验证平台。**无 NVIDIA GPU → CUDA 后端不编，走 CPU/host**（未来 Metal）。
> 覆盖：构建 / 测试 / 性能测试 + 依赖。本仓库开发机通过 SSH 访问（`ssh kiro@172.22.10.198`）。

## 一、硬门槛

| 组件 | 要求 | 装法 |
|---|---|---|
| Apple clang | **18+**（Command Line Tools） | `xcode-select --install` |
| CMake | 3.25+ | `brew install cmake` |
| Ninja *或* make | 二选一 | `brew install ninja`（若无 ninja 用 `-G "Unix Makefiles"`） |

- clang 在 `/Library/Developer/CommandLineTools/usr/bin/clang{,++}`。
- ⚠️ **brew / cmake 不在非登录 shell 的 PATH**（Apple Silicon 装在 `/opt/homebrew/bin`）——用绝对路径 `/opt/homebrew/bin/cmake` 或先 `export PATH=/opt/homebrew/bin:$PATH`。非交互 ssh 命令里 `$HOMEBREW_PREFIX` 等为空，别依赖。
- **fcontext**：arm64 macho 汇编已 vendored 并接线，fiber 在 mac 真跑。

## 二、构建

无 ninja 时用 Unix Makefiles + 显式 clang：

```bash
export PATH=/opt/homebrew/bin:$PATH
cd ~/hyle
/opt/homebrew/bin/cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++
/opt/homebrew/bin/cmake --build build -j8
```

- ggml/googletest 首次配置**自动联网拉**到 `.deps`（mac 有网即可），之后缓存复用。
- 配置会打印 `CUDA Toolkit not found -- compute uses host fallback only`（mac 无 CUDA，正常）；ggml 编 CPU-only（Apple 上还会带 Metal backend，但 HYLE bridge 只用 CPU 路径）。

## 三、测试

```bash
export PATH=/opt/homebrew/bin:$PATH
cd ~/hyle
./build/bin/hyle_compute_ggml_bridge_test     # CPU-only matmul（无 GPU 那格走 host）
/opt/homebrew/bin/ctest --test-dir build --output-on-failure
```

- **NEON simd 变体**：`tests/CMakeLists` 按 `CMAKE_SYSTEM_PROCESSOR` 分支——arm64 建 `hyle_simd_neon_test`（x86 建 sse4/avx2）。这是 neon.h 全 op 对拍 scalar 的真闸门。
- **kqueue io 后端**：mac 的 io proactor 走 kqueue，`hyle_io_*` 测试真跑该路径。

## 四、性能测试（bench）

```bash
/opt/homebrew/bin/cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++ \
  -DHYLE_BUILD_BENCH=ON
/opt/homebrew/bin/cmake --build build --target hyle_sched_bench -j8
./build/bin/hyle_sched_bench
```

- mac 是本地磁盘（APFS），性能数字可信；ARM 侧数字与 x86 不直接可比（架构不同）。

## 五、同步工作流（SSH 开发机）

- 常驻 clone 在 `~/hyle`（`/Users/kiro/hyle`）。**同步走 git**：Windows 侧 commit + push → `cd ~/hyle && git fetch origin -q && git reset --hard origin/main -q`。
- ⚠️ **别用 scp 手选文件同步**——极易漏传 `src/` 内部 `.h`（如 vtable 头改了没传 → "no member" 编译错，或旧 build 假 PASS）。**优先 git fetch+reset**；必须 scp 时核对测试总数与 Win/WSL 一致。
- ⚠️ **SSH 免密不生效**：多半是 macOS sshd 权限严格检查——`chmod g-w,o-w ~` + `chmod 700 ~/.ssh` + `chmod 600 ~/.ssh/authorized_keys`。

## 六、可选依赖

- **OpenSSL**：`brew install openssl@3`，配置时 `-DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3` 启用 TLS。
- **CUDA**：mac 无 NVIDIA GPU，不适用。

## 七、Vulkan compute（compute-06，Apple 走 MoltenVK）

Apple 无原生 Vulkan——**MoltenVK 是 Vulkan-over-Metal 的 ICD**（把 Vulkan 调用翻译成 Metal）。HYLE 的 Vulkan 后端经 volk 运行时 dlopen `libvulkan.dylib`（loader），loader 再路由到 MoltenVK ICD，所以 mac 上一份 SPIR-V 也能在 Apple GPU 上跑（compute-06 SaxpyEndToEnd）。

> **状态（2026-07-08）**：MoltenVK 层已在本仓库 mac 开发机验证——`vulkaninfo` 枚举出 `Apple M4` / `driverName = MoltenVK`。HYLE `hyle_compute_vulkan_test` 的 mac 真跑待该机首次 FetchContent 拉齐 volk/glslang 后确认（github 访问慢，编 glslang 耗时）；Win/WSL RTX4070 已 PASS，代码路径同。

**装（brew，一次）**：

```bash
/opt/homebrew/bin/brew install molten-vk vulkan-loader vulkan-tools
```

- `molten-vk` → `/opt/homebrew/lib/libMoltenVK.dylib` + ICD manifest `…/molten-vk/*/etc/vulkan/icd.d/MoltenVK_icd.json`。
- `vulkan-loader` → `/opt/homebrew/lib/libvulkan.dylib`（volk dlopen 的就是它）。
- `vulkan-tools` → `vulkaninfo`（验证用）。
- HYLE 侧 **headers + volk + glslang 仍走 FetchContent**（`.deps`，与 Win/WSL 同）——brew 只提供 **loader + MoltenVK ICD**（driver 层，HYLE 不 vendor），不提供编译期头。

**验证 MoltenVK 通**（应枚举出 Apple GPU、driverID = MoltenVK）：

```bash
/opt/homebrew/bin/vulkaninfo --summary | grep -E "deviceName|driverName"
#   deviceName = Apple M4
#   driverName = MoltenVK
```

**跑 HYLE Vulkan 测试**：

```bash
export PATH=/opt/homebrew/bin:$PATH
cd ~/hyle
/opt/homebrew/bin/cmake --build build --target hyle_compute_vulkan_test -j8
./build/bin/hyle_compute_vulkan_test        # VulkanCompute.SaxpyEndToEnd
```

- ⚠️ **loader 找不到 ICD**（vulkaninfo 报 0 device / 测试 SKIP）时，显式指 ICD：
  `export VK_ICD_FILENAMES=$(ls /opt/homebrew/Cellar/molten-vk/*/etc/vulkan/icd.d/MoltenVK_icd.json | head -1)`。
  brew 装完通常已把 manifest 放进 loader 默认搜索路径（`/opt/homebrew/etc/vulkan/icd.d` / `/opt/homebrew/share/vulkan/icd.d`），无需手设。
- ⚠️ **MoltenVK 不支持全部 Vulkan 特性**（它是 Metal 子集映射）——compute-06 的 storage buffer + push constant + compute dispatch 都在支持范围内；将来用到 MoltenVK 缺的扩展需查其 [支持矩阵](https://github.com/KhronosGroup/MoltenVK)。
- 未装 MoltenVK 时：`volkInitialize()` 找不到 loader → Vulkan 后端 `vulkan_vtable()` 返回 null → `hyle_compute_vulkan_test` 运行时 `GTEST_SKIP`（不崩，优雅降级）。

