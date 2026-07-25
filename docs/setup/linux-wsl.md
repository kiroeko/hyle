# Linux / WSL 环境准备（GCC14 + Sanitizer）

> **lock-free 代码的 TSan 唯一闸门在这**（Windows 跑不了 TSan）。本仓库开发用 WSL Ubuntu 24.04。
> 覆盖：构建 / 测试 / 性能测试 + TSan + 9p 铁律 + 依赖（OpenSSL、CUDA）。

## 一、硬门槛

| 组件 | 要求 | 装法（Ubuntu 24.04） |
|---|---|---|
| GCC | **14+**（默认 gcc-13 过不了门槛） | `sudo apt install g++-14 gcc-14` |
| CMake | 3.25+ | `sudo apt install cmake`（24.04 源 3.28） |
| Ninja | 任意近版 | `sudo apt install ninja-build` |
| GAS | 随 gcc（fcontext ELF 汇编） | 自带 |

**必须显式指 gcc-14**（默认 `cc`/`c++` 是 gcc-13，CMake 硬门槛 FATAL_ERROR）：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
```

## 二、⚠️ 9p 铁律（血泪，最重要）

**把 WSL 当一台独立机器，用它自己的原生 ext4，绝不在 `/mnt/c` 上构建/测试。**

- `/mnt/c/...` 是 **9p 网络挂载**（Windows 盘），不稳、慢、CPU 打不饱和——重 IO / 构建 / 压力测试在上面**性能数字全废**，还可能触发诡异卡顿。
- **唯一权威源是 Windows 工作树** `C:\public\project-hyle\hyle`；`/root/hyle` 只是可丢弃、可覆盖的原生 ext4 执行镜像，不从 WSL 目录反向恢复或合并代码。
- 每次 WSL 验证前，从 `/mnt/c/public/project-hyle/hyle/` 受控执行 `rsync -a --delete` 覆盖 `/root/hyle/`。可排除并保留 WSL 本机 `build*/`（依赖在各 build 的 `_deps` 中），但源码必须单向同步为 Windows 当前工作树，包括未提交改动。
- 同步只负责搬源码；构建、测试、调试、压力与 benchmark 全部在 `/root/hyle` 运行，`/mnt/c` 上的执行结果不作验收证据。
- build 目录及其 `_deps` 留在 `/root/hyle`，避免跨 9p 共享 FetchContent 源码和构建产物。

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

- ggml/googletest 首次配置自动联网拉到 `build/_deps`；删除 build 会一并删除，避免跨 toolchain 污染。
- `wsl.exe` 从 Windows 侧驱动的坑：`wsl.exe -e bash -lc '...'` 外层 git-bash 会吃掉 `$VAR`；把逻辑写成 `.sh`（在 WSL 原生 bash 里跑变量才正常）。传 `/mnt/c/x.sh` 路径给 wsl.exe 会被 git-bash 改写 → 用 `wsl.exe -e bash -lc 'bash /root/x.sh'`，脚本直接在 `/root` 用 heredoc 生成、别 cp。

## 四、测试

```bash
cd /root/hyle
./build/bin/hyle_sched_test        # 单个
ctest --test-dir build --output-on-failure   # 全部
```

- **GPU 测试**：WSL 有 CUDA 直通（RTX 4070），`hyle_compute_ggml_bridge_test` 真跑零拷贝 matmul。运行前 `export PATH=/usr/local/cuda/bin:$PATH`。

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
- **⚠️ mimalloc × TSan 已知误报（KI-2/KI-3）**：凡链 mimalloc 的 exe 在 TSan 下会报 mimalloc 内部跨线程块回收的假 race（`[anon:mimalloc]` 地址），或启动期 `mi_out_default` 断言。**仅 TSan 构建**，Release 零影响。判定 HYLE 自身逻辑 race 时，要区分"冲突地址在 mimalloc 堆页"（假报）vs 真正的共享状态竞争。详见 `docs/architecture/known-issues.md`。
- **TSan 用无 CUDA 构建更干净**（CUDA + TSan 不合）：`-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON` 让 ggml 走 CPU-only，TSan 只验 HYLE 的 host 侧协调（spawn_blocking/fiber 唤醒）。

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
- SIMD bench 将 `HYLE_BENCH_VENDOR_DIR` 的固定版 Highway/xsimd 当对照；它们只供 bench，
  不替代本轮仍由产品源码使用的 `third_party/` 副本。
- 统计对标用 `scripts/bench-stat.py`（多遍取统计，防单遍噪声）。

## 七、可选依赖

### OpenSSL（零配置）
```bash
sudo apt install libssl-dev          # 24.04 上 3.0.13，装完重配即启用 TLS
```

### CUDA（GPU 后端）
- 本仓库统一 **CUDA 13.3**，WSL 装 toolkit 用 NVIDIA `wsl-ubuntu` apt 源（toolkit-only，别装会毁 GPU 直通的 Linux 驱动）。`nvcc` 在 `/usr/local/cuda-13.3/bin`（软链 `/usr/local/cuda`）。
- ⚠️ **非交互 login shell（`wsl.exe -e bash -lc`）PATH 里没有 nvcc**——用绝对路径 `/usr/local/cuda/bin/nvcc` 或先 `export PATH=/usr/local/cuda/bin:$PATH`。
- cuDNN/TensorRT 在 `cuda/repos/ubuntu2404` 源（当前里程碑不需要），装法与 apt pin 见 `reference_cuda_env`。
