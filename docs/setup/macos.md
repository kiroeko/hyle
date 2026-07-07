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
- **CUDA**：mac 无 NVIDIA GPU，不适用。GPU 加速未来走 Metal（compute-06 Vulkan/kompute 或 Metal 后端，尚未开工）。
