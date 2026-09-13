# macOS 环境准备（Apple Silicon）

> ARM/NEON + kqueue 真机验证平台。
> 覆盖：构建 / 测试 / 性能测试 + 依赖。本仓库有两个独立 Apple Silicon 验证环境：MacBook Air
> `Mac16,12`（`ssh kiro@172.22.10.198`）与 Mac mini `Mac16,10`
>（`ssh kiro@172.22.10.107`）。`M1-MAC` 性能 profile 当前只指前者；后者在单独登记身份前只作
> correctness/portability 复核，不能把两台机器或两种 linkage 混为一个 Formal 证据。
> 无网机器的 Command Line Tools/CMake 介质与断网验收见 [`offline.md`](offline.md)。

从 Windows canonical workspace 连接时，MacBook Air 使用 OpenSSH 默认身份：

```powershell
ssh kiro@172.22.10.198
```

Mac mini 使用它自己的专用身份；该 key 不能拿来判断 Air 是否可达：

```powershell
ssh -i C:\Users\kiro\.ssh\codex_mac_172_22_10_107_win kiro@172.22.10.107
```

只记录 identity 的位置与连接方法，不把私钥内容、远端临时源码或构建产物提交进仓库。

## 一、硬门槛

| 组件 | 要求 | 装法 |
|---|---|---|
| Apple clang | **18+**（Command Line Tools） | `xcode-select --install` |
| CMake | 3.25+ | `brew install cmake` |
| Ninja *或* make | 二选一 | `brew install ninja`（若无 ninja 用 `-G "Unix Makefiles"`） |

- clang 在 `/Library/Developer/CommandLineTools/usr/bin/clang{,++}`。
- 选择 `platform`（以及依赖它的 mem/simd/container/sched）时，macOS deployment target
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

- 默认 preset 是 `all` + tests 的 CPU baseline；TLS、HTTP/2 和 trace 都是 `OFF`。首次配置通过
  FetchContent 取得所选闭包与 googletest 的锁定源码。显式开启 HTTP/2 才会把 nghttp2 放到
  `$HOME/Library/Caches/hyle/dependencies/v1`（或显式
  `HYLE_DEP_SOURCE_CACHE`）。每个 build tree 的 `_deps` 只保存自己的 binary/subbuild；删掉 build
  不会删源码，后续 build 不重新下载。

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

- MacBook Air 的常驻 clone 在 `~/hyle`（`/Users/kiro/hyle`）。**同步走 git**：Windows 侧 commit + push → `cd ~/hyle && git fetch origin -q && git reset --hard origin/main -q`。
- Mac mini 可能只有 Windows 到该主机的 SSH 身份，而没有主机到 GitHub 的仓库身份。此时不得手选文件
  拼接源码；从 Windows canonical checkout 对候选 commit 生成完整 `git archive`，传输并展开为一次性 source
  bundle，再记录候选 hash。该机已有验证工具时 CMake 位于 `~/hyle-verify-tools/bin/cmake`，不能假设存在
  `/opt/homebrew/bin/cmake`。
- ⚠️ **别用 scp 手选文件同步**——极易漏传 `src/` 内部 `.h`（如 vtable 头改了没传 → "no member" 编译错，或旧 build 假 PASS）。**优先 git fetch+reset**；必须传输时只传完整、可追溯的 archive，并核对测试总数与 Win/WSL 一致。
- ⚠️ **SSH 免密不生效**：多半是 macOS sshd 权限严格检查——`chmod g-w,o-w ~` + `chmod 700 ~/.ssh` + `chmod 600 ~/.ssh/authorized_keys`。

## 六、可选依赖

- **OpenSSL**：`brew install openssl@3`，配置时 `-DHYLE_IO_TLS=ON -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3` 启用 TLS。
