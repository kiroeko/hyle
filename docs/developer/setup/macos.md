# macOS 环境准备（Apple Silicon）

> ARM/NEON + kqueue 真机验证平台。**无 NVIDIA GPU → CUDA 后端不编，走 CPU/host**（未来 Metal）。
> 覆盖：构建 / 测试 / 性能测试 + 依赖。本仓库有两个独立 Apple Silicon 验证环境：MacBook Air
> `Mac16,12`（`ssh kiro@172.22.10.198`）与 Mac mini `Mac16,10`
>（`ssh kiro@172.22.10.107`）。`M1-MAC` 性能 profile 当前只指前者；后者在单独登记身份前只作
> correctness/portability 复核，不能把两台机器或两种 linkage 混为一个 Formal 证据。
> 无网机器的 Command Line Tools/CMake 介质与断网验收见 [`offline.md`](offline.md)。

## 一、硬门槛

| 组件 | 要求 | 装法 |
|---|---|---|
| Apple clang | **18+**（Command Line Tools） | `xcode-select --install` |
| CMake | 3.25+ | `brew install cmake` |
| Ninja *或* make | 二选一 | `brew install ninja`（若无 ninja 用 `-G "Unix Makefiles"`） |

- clang 在 `/Library/Developer/CommandLineTools/usr/bin/clang{,++}`。
- 选择 `platform`（以及依赖它的 mem/simd/container/sched/compute）时，macOS deployment target
  必须 ≥14.4；Parker/address wait 使用该版本才公开的系统接口。HYLE 顶层构建未指定时默认
  14.4；嵌入父工程时请在 `add_subdirectory(hyle)` 前设置
  `CMAKE_OSX_DEPLOYMENT_TARGET=14.4` 或更高，低版本会在 configure 阶段明确失败。
- ⚠️ **brew / cmake 不在非登录 shell 的 PATH**（Apple Silicon 装在 `/opt/homebrew/bin`）——用绝对路径 `/opt/homebrew/bin/cmake` 或先 `export PATH=/opt/homebrew/bin:$PATH`。非交互 ssh 命令里 `$HOMEBREW_PREFIX` 等为空，别依赖。
- **fcontext**：arm64 macho 汇编已 vendored 并接线，fiber 在 mac 真跑。

## 二、构建

`macos` preset 不指定 generator；有 Ninja 就可用 Ninja，没有时 CMake 选择 Unix Makefiles：

```bash
export PATH=/opt/homebrew/bin:$PATH
cd ~/hyle
scripts/hyle-unix.sh /opt/homebrew/bin/cmake --preset macos
scripts/hyle-unix.sh /opt/homebrew/bin/cmake --build --preset macos --parallel 2
```

- 默认 preset 是 `all` + tests 的 CPU baseline；TLS、HTTP/2、ggml、Vulkan、CUDA 和 trace 都是
  `OFF`，首次配置只通过 FetchContent 取得 googletest。显式开启 HTTP/2、ggml 或 Vulkan 后才会把
  对应锁定源码放到 `$HOME/Library/Caches/hyle/dependencies/v1`（或显式
  `HYLE_DEP_SOURCE_CACHE`）。每个 build tree 的 `_deps` 只保存自己的 binary/subbuild；删掉 build
  不会删源码，后续 build 不重新下载。
- CUDA 保持 `OFF`，配置不会探测 Toolkit；ggml 默认也不构建。显式开启 ggml 时，macOS 形状是
  CPU-only，且不构建未消费的 Metal/BLAS backend。

## 三、测试

```bash
export PATH=/opt/homebrew/bin:$PATH
cd ~/hyle
/opt/homebrew/bin/ctest --preset macos --output-on-failure
```

- **NEON simd 变体**：`tests/simd/CMakeLists.txt` 按归一化后的 `HYLE_TARGET_PROCESSOR` 分支——arm64 建 `hyle_simd_neon_test`（x86 建 sse4/avx2）。这是 neon.h 全 op 对拍 scalar 的真闸门。
- **kqueue io 后端**：mac 的 io proactor 走 kqueue，`hyle_io_core_*` 测试真跑该路径。
- **独立头检查**：非登录 SSH 下直接调用 AppleClang 必须带 macOS SDK sysroot；
  `header-self-contained.py` 会通过 `/usr/bin/xcrun --sdk macosx --show-sdk-path`
  自动发现，不要把标准库 `<cstddef>` 找不到误判成 HYLE 头文件缺依赖。

## 四、性能测试（bench）

```bash
# 固定版 external Highway/xsimd 默认写入 ~/vendor。
scripts/fetch-bench-vendors.sh

# preset 不锁 generator，并独占 build-bench，不污染日常 build。
/opt/homebrew/bin/cmake --preset bench
/opt/homebrew/bin/cmake --build --preset bench --target hyle_sched_bench --parallel 2
./build-bench/bin/hyle_sched_bench
```

- mac 是本地磁盘（APFS），性能数字可信；ARM 侧数字与 x86 不直接可比（架构不同）。
- macOS 无受支持的 hard affinity；工程/formal 数据必须记录实际 process QoS、交流电与 low-power
  状态。默认 `interactive` 与 `utility` 是不同合同，不冒充 P/E-core pin；见
  [性能机器合同](../performance/machines.md)。
- SIMD 产品从中央 lock 获取 Highway 源并直接编译进 HYLE target TU；
  `~/vendor/{highway,xsimd}` 的独立 checkout 只作 benchmark 对照。

## 五、同步工作流（SSH 开发机）

- 常驻 clone 在 `~/hyle`（`/Users/kiro/hyle`）。**同步走 git**：Windows 侧 commit + push → `cd ~/hyle && git fetch origin -q && git reset --hard origin/main -q`。
- ⚠️ **别用 scp 手选文件同步**——极易漏传 `src/` 内部 `.h`（如 vtable 头改了没传 → "no member" 编译错，或旧 build 假 PASS）。**优先 git fetch+reset**；必须 scp 时核对测试总数与 Win/WSL 一致。
- ⚠️ **SSH 免密不生效**：多半是 macOS sshd 权限严格检查——`chmod g-w,o-w ~` + `chmod 700 ~/.ssh` + `chmod 600 ~/.ssh/authorized_keys`。

## 六、可选依赖

- **OpenSSL**：`brew install openssl@3`，配置时 `-DHYLE_IO_TLS=ON -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3` 启用 TLS。
- **CUDA**：mac 无 NVIDIA GPU，不适用。

## 七、Vulkan compute（compute-06，Apple 走 MoltenVK）

Apple 无原生 Vulkan——**MoltenVK 是 Vulkan-over-Metal 的 ICD**（把 Vulkan 调用翻译成 Metal）。HYLE 的 Vulkan 后端经 volk 运行时 dlopen `libvulkan.dylib`（loader），loader 再路由到 MoltenVK ICD，所以 mac 上一份 SPIR-V 也能在 Apple GPU 上跑（compute-06 SaxpyEndToEnd）。

> **状态（2026-07-08）**：mac 真机全绿——`hyle_compute_vulkan_test` 在 **Apple M4 via MoltenVK 3/3 PASS**（SaxpyEndToEnd + BatchedSaxpy + CoAwaitCompletion），与 Win/WSL RTX4070 3/3 同路径，compute-06 三平台闭环。`vulkaninfo` 枚举 `Apple M4` / `driverName = MoltenVK`（apiVersion 1.4.334）。⚠️ mac 真跑抓出一个 portability 真 bug 并已修（见下「portability 扩展」）。

**装（brew，一次）**：

```bash
/opt/homebrew/bin/brew install molten-vk vulkan-loader vulkan-tools
```

- `molten-vk` → `/opt/homebrew/lib/libMoltenVK.dylib` + ICD manifest `…/molten-vk/*/etc/vulkan/icd.d/MoltenVK_icd.json`。
- `vulkan-loader` → `/opt/homebrew/lib/libvulkan.dylib`（volk dlopen 的就是它）。
- `vulkan-tools` → `vulkaninfo`（验证用）。
- HYLE 侧 **Vulkan-Headers + volk 仍走锁定 FetchContent**（source 在共享 cache，binary/subbuild 在当前
  build 的 `_deps`）；
  glslang 仅在 tests/benchmark 配置中生成测试 shader。brew 只提供 **loader + MoltenVK ICD**
  （driver 层，HYLE 不 vendor），不提供编译期头。

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
/opt/homebrew/bin/cmake --preset macos -B build-macos-vulkan \
  -DHYLE_COMPUTE_VULKAN=ON
/opt/homebrew/bin/cmake --build build-macos-vulkan \
  --target hyle_compute_vulkan_test --parallel 2
./build-macos-vulkan/bin/hyle_compute_vulkan_test  # VulkanCompute.SaxpyEndToEnd
```

- ⚠️ **loader 找不到 ICD**（vulkaninfo 报 0 device / 测试 SKIP）时，显式指 ICD：
  `export VK_ICD_FILENAMES=$(ls /opt/homebrew/Cellar/molten-vk/*/etc/vulkan/icd.d/MoltenVK_icd.json | head -1)`。
  brew 装完通常已把 manifest 放进 loader 默认搜索路径（`/opt/homebrew/etc/vulkan/icd.d` / `/opt/homebrew/share/vulkan/icd.d`），无需手设。
- ⚠️ **portability 扩展（HYLE 代码侧，已在 `vulkan_backend.cc` 处理，无需手动配）**：MoltenVK 是 *portability* 驱动，`vkCreateInstance` 若不开 `VK_KHR_portability_enumeration`（instance 扩展 + `ENUMERATE_PORTABILITY` flag）会直接返 `VK_ERROR_INCOMPATIBLE_DRIVER(-9)`，创建 VkDevice 又须启用 `VK_KHR_portability_subset`（spec 强制）。HYLE 后端**运行时探测这两个扩展、advertise 才启用**（一份代码通吃 mac + Win/WSL discrete GPU）。记录于此以防将来重构 instance/device 创建时误删——ICD 设对了但测试仍 SKIP，先查这里。
- ⚠️ **MoltenVK 不支持全部 Vulkan 特性**（它是 Metal 子集映射）——compute-06 的 storage buffer + push constant + compute dispatch 都在支持范围内；将来用到 MoltenVK 缺的扩展需查其 [支持矩阵](https://github.com/KhronosGroup/MoltenVK)。
- 未装 MoltenVK 时：`volkInitialize()` 找不到 loader → Vulkan 后端 `vulkan_vtable()` 返回 null → `hyle_compute_vulkan_test` 运行时 `GTEST_SKIP`（不崩，优雅降级）。
