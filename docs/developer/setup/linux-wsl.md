# Linux / WSL 环境准备（GCC14 + Sanitizer）

> **lock-free 代码的 TSan 唯一闸门在这**（Windows 跑不了 TSan）。本仓库开发用 WSL Ubuntu 24.04。
> 覆盖：构建 / 测试 / 性能测试 + TSan + 9p 铁律 + 依赖（OpenSSL、CUDA）。
> 无网机器的 repository mirror、依赖介质与断网验收见 [`offline.md`](offline.md)。

## 一、硬门槛

| 组件 | 要求 | 装法（Ubuntu 24.04） |
|---|---|---|
| GCC | **14+**（默认 gcc-13 过不了门槛） | `sudo apt install g++-14 gcc-14` |
| CMake | 3.25+ | `sudo apt install cmake`（24.04 源 3.28） |
| Generator | Ninja 或 Make | `sudo apt install ninja-build`；没有 Ninja 时使用系统 Make |
| GAS | 随 gcc（fcontext ELF 汇编） | 自带 |

统一入口会从已安装 toolchain 中优先选择 `gcc-14/g++-14`，因此不会再继承 Ubuntu 24.04 默认
GCC 13。它只选择、不安装；显式覆盖仍必须同时设置 `CC/CXX`：

```bash
scripts/hyle-unix.sh --print-toolchain
scripts/hyle-unix.sh cmake --preset linux
scripts/hyle-unix.sh cmake --build --preset linux --parallel 2
scripts/hyle-unix.sh ctest --preset linux --output-on-failure

# 需要显式选择 Clang 等其它受支持 pair 时：
CC=clang-18 CXX=clang++-18 scripts/hyle-unix.sh cmake --preset linux
```

`cmake --preset wsl` 仍是名称固定为 `gcc-14/g++-14` 时的方便入口。已有 cache 若曾绑定 GCC 13，必须
换新的独占 build/scratch；CMake 不支持原地换 compiler，wrapper 也不会删除旧目录。

## 二、⚠️ 9p 铁律（血泪，最重要）

**把 WSL 当一个独立执行环境，用它自己的原生 ext4，绝不在 `/mnt/c` 上构建/测试。** 它与
Windows 共享同一物理主机，不是独立性能机，也不能替代 native-Linux formal 证书；角色边界见
[性能机器合同](../performance/machines.md)。

- `/mnt/c/...` 是 **9p 网络挂载**（Windows 盘），不稳、慢、CPU 打不饱和——重 IO / 构建 / 压力测试在上面**性能数字全废**，还可能触发诡异卡顿。
- **唯一权威源是 Windows 工作树** `C:\public\project-hyle\hyle`；`/root/hyle` 只是可丢弃、可覆盖的原生 ext4 执行镜像，不从 WSL 目录反向恢复或合并代码。
- 每次 WSL 验证前，从 `/mnt/c/public/project-hyle/hyle/` 受控执行 `rsync -a --delete` 覆盖 `/root/hyle/`。可排除并保留 WSL 本机 `build*/`；锁定第三方源码位于 WSL 自己的
  `$HOME/.cache/hyle/dependencies/v1`，各 build 的 `_deps` 只含 binary/subbuild。HYLE 源码必须单向
  同步为 Windows 当前工作树，包括未提交改动。
- 同步只负责搬源码；构建、测试、调试、压力与 benchmark 全部在 `/root/hyle` 运行，`/mnt/c` 上的执行结果不作验收证据。
- build 目录及其 `_deps` 留在 `/root/hyle`；共享 source cache 也必须位于 WSL ext4，不能指向
  `/mnt/c` 或与 Windows cache 跨 9p 共用。

## 三、构建

```bash
cd /root/hyle   # 原生 ext4，不是 /mnt/c
export PATH=/usr/local/cuda/bin:$PATH   # 非交互 shell 缺 cuda bin（见下 CUDA 节）

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DHYLE_COMPUTE_CUDA=ON
cmake --build build
```

- 这条命令使用默认 `all` + tests，显式开启 CUDA；TLS、HTTP/2、ggml、Vulkan 和 trace 仍保持
  默认 `OFF`。首次配置只通过 FetchContent 取得 googletest，CUDA Toolkit 由系统安装提供。
  显式开启 HTTP/2、ggml 或 Vulkan 后才会取得各自的锁定源码。删除 build 只删除本 build 的
  binary/subbuild；共享源码保留，toolchain 状态不会跨 build 混用。
- `wsl.exe` 从 Windows 侧驱动的坑：`wsl.exe -e bash -lc '...'` 外层 git-bash 会吃掉 `$VAR`；把逻辑写成 `.sh`（在 WSL 原生 bash 里跑变量才正常）。传 `/mnt/c/x.sh` 路径给 wsl.exe 会被 git-bash 改写 → 用 `wsl.exe -e bash -lc 'bash /root/x.sh'`，脚本直接在 `/root` 用 heredoc 生成、别 cp。

## 四、测试

```bash
cd /root/hyle
./build/bin/hyle_sched_test        # 单个
ctest --test-dir build --output-on-failure   # 全部
```

- **GPU 测试**：WSL 有 CUDA 直通（RTX 4070）。只有配置时另加
  `-DHYLE_COMPUTE_GGML=ON` 才会生成 `hyle_compute_ggml_bridge_test` 并真跑零拷贝 matmul；运行前
  `export PATH=/usr/local/cuda/bin:$PATH`。

## 五、TSan（lock-free / 并发改动必过）

```bash
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DHYLE_SANITIZER=tsan
cmake --build build-tsan --target hyle_sched_test
setarch -R ./build-tsan/bin/hyle_sched_test    # -R 关 ASLR，稳定 TSan
```

- `HYLE_SANITIZER` = `off|asan|tsan|ubsan`（也接受 `address/thread/undefined/none`；**拼错 FATAL_ERROR**，不静默降级成假 PASS）。
- **必须 `setarch -R`**——否则新内核高熵 ASLR 触发 "FATAL: ThreadSanitizer: unexpected memory mapping"。
- mimalloc 3.5 在 TSan 形状中必须与 HYLE 一起用 `MI_TSAN=1` 和
  `-fsanitize=thread` 编译；并发压力测试直接使用 shipping
  `system_allocator()`，不再绕到 malloc。旧版的启动断言和回收地址假 race 已在升级复验中消失，
  因此新的 sanitizer 报告不能按历史 KI-2/KI-3 自动忽略，必须按真实缺陷调查。
- **TSan 用无 CUDA 构建更干净**（CUDA + TSan 不合）：显式设置
  `-DHYLE_COMPUTE_CUDA=OFF`。若该轮还要验证 ggml bridge，再加 `-DHYLE_COMPUTE_GGML=ON`，此时
  ggml 保持 CPU-only；TSan 只验 HYLE 的 host 侧协调（spawn_blocking/fiber 唤醒）。

## 六、性能测试（bench）

```bash
# external vendor 和构建均在 ext4；WSL 不设隐式默认目录。
scripts/fetch-bench-vendors.sh --dir "$HOME/vendor"

cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DHYLE_BUILD_BENCH=ON \
  -DHYLE_BENCH_VENDOR_DIR="$HOME/vendor"
cmake --build build-bench --target hyle_sched_bench
./build-bench/bin/hyle_sched_bench
```

- **必须原生 ext4**（见 9p 铁律）——`/mnt/c` 上 bench 数字无效（CPU 打不饱和）。
- guest CPU affinity 只用于工程重复性，不能证明固定到物理 P/E core；Windows 与 WSL 必须使用同一
  `W1` 主机互斥锁，禁止并行采样。
- SIMD 产品从中央 lock 获取 Highway 源并直接编译进 HYLE target TU；
  `HYLE_BENCH_VENDOR_DIR` 中独立、干净的 Highway/xsimd checkout 只作同身份/同 ISA 对照。
- 统计对标用 `scripts/bench-stat.py`（多遍取统计，防单遍噪声）。

## 七、可选依赖

### OpenSSL（显式开启 TLS）
```bash
sudo apt install libssl-dev          # 24.04 上 3.0.13
CC=gcc-14 CXX=g++-14 cmake --preset linux -B build-linux-tls \
  -DHYLE_COMPONENTS=io -DHYLE_IO_TLS=ON
```

### CUDA（GPU 后端）
- 本仓库统一 **CUDA 13.3**，WSL 装 toolkit 用 NVIDIA `wsl-ubuntu` apt 源（toolkit-only，别装会毁 GPU 直通的 Linux 驱动）。`nvcc` 在 `/usr/local/cuda-13.3/bin`（软链 `/usr/local/cuda`）。
- ⚠️ **非交互 login shell（`wsl.exe -e bash -lc`）PATH 里没有 nvcc**——用绝对路径 `/usr/local/cuda/bin/nvcc` 或先 `export PATH=/usr/local/cuda/bin:$PATH`。
- cuDNN/TensorRT 在 `cuda/repos/ubuntu2404` 源（当前里程碑不需要），装法与 apt pin 见 `reference_cuda_env`。
