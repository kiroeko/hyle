# 四路线有限架构实验

范围与预声明判据见[研究卡](../../../docs/implementation/tasks/compute/four-route-architecture-decision.md)。
这里是独立研究程序，不是已交付的 HYLE Provider。源码由 Windows canonical checkout 管理，macOS
使用完整 Git archive。没有性能计时或 SOTA 签发，也不验证完整 OOM/device-loss/三交付。

- `mlx_custom.cc`：MLX 0.32.2 C++ API，GPU MatMul → 自定义 Metal SwiGLU → GPU MatMul。
  同线程六次变化输入；double oracle 与编译 trace；仅最终输出处显式同步。
- `vulkan_replay.cc`、`two_pass.comp`：一个原生 pipeline、两个 compute pass、显式内存 barrier；
  录制一次、三次变化输入、最终 fence 后 exact oracle。host-visible buffer 是同步反偏置探针，
  不是独显权重存储或性能推荐。只检查正常生命周期，不把失败时进程退出当生产 RAII。

MLX 原生扩展由公开 `fast::metal_kernel` 接入；源码生成/编译是冷启动成本，缓存命中也仍有框架 metadata/
allocator 成本。实验中的 kernel 仅证明可组合，不是 HYLE 已胜出的 SIMD/kernel 资产。

## 2026-09-15 结果

| 探针 | 预声明结果 | 证据范围 |
|---|---|---|
| MLX + custom Metal FFN | 六次变化输入 PASS；trace=1；最大绝对误差 `5.85924300814e-08` | 应用无中间读取/等待，最终 checked stream synchronize；不签无内部 copy/分配/JIT |
| Vulkan two-pass | 三次变化输入、每次 1,024 个值 exact PASS；record_count=1 | 重复阶段夹具不建资源、不重录；native driver 成本未测 |

原始输出：[MLX](mlx-air-run.log)、[Vulkan](vulkan-windows-run.log)。两个源程序各第一次构建/执行通过。
MLX 在 Mac 上初次下载超时，改由 Windows 下载并验证完整 wheel 后传输；没有把残缺下载用于运行，
也没有为此更改代码、依赖锁或安装全局 Python 包。

## 源码与环境

探针 commit：`39c5aae52f7816d0bd2869ded5035e12f9ef0184`。Windows 两个源码的 SHA-256：

```text
mlx_custom.cc     e4e70931888bd44f3249e5cbb0f5283ca9e761d62329760fe1f485016c3e2fbe
vulkan_replay.cc  c7e2d6243021fd84e5d177b6388bed90d3d3a05736cc11d4f3530ae5940828e9
```

Mac 从该 commit 的 **完整 Git archive** 解包；MLX 探针的远程源码 SHA-256 与上面一致。研究提交之后的
报告/日志修改不改变探针字节，不为文档变动重跑。远程不是产品源码来源。

- MacBook Air `Mac16,12`，macOS `27.0 / 26A428`，Apple clang `21.0.0 (clang-2100.3.34.2)`、SDK 27.0，
  deployment 26.2。MLX `0.32.2`，官方 `mlx_metal-0.32.2-py3-none-macosx_26_0_arm64.whl` 中的 C++
  headers/dylib，包仅解压在诊断目录；不是新增 HYLE wheel 依赖或发布形式。
- Wheel SHA-256：`e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`。
  [官方发行元数据](https://pypi.org/pypi/mlx-metal/0.32.2/json)。源码核对 MLX v0.32.2 commit
  `1f8e74e3f12f31365464a6867c6579f0e9b29d85`，不混用本地 main。
- Windows RTX 4070；Vulkan API `1.4.341`、driver 原始字段 `2559541248`；MSVC `19.50.35717`，`/O2`。
  使用现有共享源码缓存：Vulkan-Headers `e3b1eec08173d6b825cd3ac88c885a63b621504a`、
  Volk `776893306c5d3b22b6185b5d4a258b81d94572bf`。glslangValidator `15.2.0`，目标 Vulkan 1.1。
  未安装/升级 Vulkan SDK，不将 header 版本当运行设备版本。

## 实际构建与执行

Windows canonical 仓库生成完整归档，随后通过项目登记的 MacBook Air SSH 连接传到
`/Users/kiro/hyle-diagnostics/route-architecture-20260915`，同时传入校验过的 `mlx.whl`：

```text
git archive --format=tar --output=.scratch/compute-four-route-admission/source.tar 39c5aae52f7816d0bd2869ded5035e12f9ef0184
```

该远程诊断目录内执行：

```sh
shasum -a256 mlx.whl
mkdir -p src-39c5aae5 runtime
tar -xf source.tar -C src-39c5aae5
unzip -q -o mlx.whl -d runtime
clang++ -std=c++20 -O2 -pthread -mmacosx-version-min=26.2 \
  -Iruntime/mlx/include \
  src-39c5aae5/evidence/correctness/compute-four-route-admission/mlx_custom.cc \
  -Lruntime/mlx/lib -lmlx \
  -Wl,-rpath,/Users/kiro/hyle-diagnostics/route-architecture-20260915/runtime/mlx/lib \
  -o mlx-custom-probe
./mlx-custom-probe
```

Windows shader 编译从 HYLE 根目录执行：

```text
C:/public/tools/glslang/glslangValidator.exe -V --target-env vulkan1.1 -S comp -o .scratch/compute-four-route-admission/two_pass.spv evidence/correctness/compute-four-route-admission/two_pass.comp
```

在 VS 18 Enterprise 的 `vcvars64.bat` 环境下，从 `.scratch/compute-four-route-admission` 编译并运行。
下面尖括号表示现有缓存路径占位符，不是自动获取 SDK 的脚本：

```text
cl.exe /nologo /std:c++20 /EHsc /O2 /I"<Vulkan-Headers>/include" /I"<Volk>" C:/public/project-hyle/hyle/evidence/correctness/compute-four-route-admission/vulkan_replay.cc <Volk>/volk.c /Fe:vulkan_replay.exe
vulkan_replay.exe two_pass.spv
```

两项仅签对应正常计算/资源机制；没有性能采样、sanitizer、故障注入、validation-layer、Linux、Mac mini、
MoltenVK 或 source/static/shared 产品矩阵，不据此把任何模块改为 CLOSED。
