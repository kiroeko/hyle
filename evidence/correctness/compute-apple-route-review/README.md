# Apple Q4 / Metal 扩展与 lifetime 研究

范围与判据在[六项复核卡](../../../docs/implementation/tasks/compute/six-point-architecture-review.md)预声明。
[探针](mlx_quantized_lifetime.cc)是研究资产，不是 HYLE shipping Provider，也不是 performance sample。
2026-09-15 第一次构建、第一次执行成功；[原始输出](mlx-air-run.log)保留环境、结果与源码 hash。

## 测了什么

- MLX v0.32.2 C++，明确 GPU stream；Q4 affine/group64 的 gate/up MatMul → 公共
  `fast::metal_kernel` SwiGLU → Q4 down MatMul。M=1/7、K=H=256，每形状三次不同输入。
- 权重在冷准备量化；CPU oracle 从实际 packed nibbles、scales、additive biases 独立解码并用 double
  累加，而非对原浮点权重作错误的等价假设。主 GPU 计算消费 packed weights；没有显式全量展开
  GPU 权重。host oracle 的展开只是验证，不代表产品存储策略或完整 MLX 显存占用测量。
- 每个形状独立 compile；相同形状三次调用 body trace=1。临时输入 array 在 `async_eval` 后、最终
  synchronize 前退出 caller scope；第一份 output 保留跨两次后续执行，不能被覆盖。
- 每次最终同步检查后才读结果；应用没有中间 host wait/read。非法 K=255 在调用验证阶段抛
  `invalid_argument`，未 eval。容差始终为 `1e-4 * max(1, abs(expected))`，没有改阈值重测。

## 结果与不外推项

| 检查 | 结果 |
|---|---|
| M=1，三次变化输入 | PASS，最大绝对误差 `1.61958440322e-09` |
| M=7，三次变化输入 | PASS，最大绝对误差 `2.01876169427e-09` |
| 每程序 trace、临时 input owner、保留旧 output | PASS（所测顺序生命周期） |
| 非法 K | 接纳/eval 前拒绝 |

这是有限 native 机制证明：不签任意 MTLBuffer/Queue import、跨 runtime zero-copy、并发、取消/
失败 drain、device-loss、allocation 高水位、M5 NAX、完整数值准确度或 source/static/shared。
trace=1 不证明没有库内部 metadata/allocation/JIT/cache lookup。小形状/小权重为快速架构验证，
不能证明真实模型规模、decode/prefill 性能或 SOTA。P1 未运行。

## 来源、环境与实际命令

Windows canonical 源码 commit：`db01de97e10ea2be8eb45f4ed446227d84e187c7`。
源码 SHA-256（与远程实际输出一致）：

```text
60e13167bcc5a5163c6c9585b2aef21535095f8127f755072d978d997e562bf9
```

MacBook Air `Mac16,12`；macOS 27.0 / 26A428，Apple clang 21.0.0 / clang-2100.3.34.2，arm64，
deployment 26.2。复用[前轮](../compute-four-route-admission/README.md#源码与环境)已校验的 MLX 0.32.2
官方 wheel C++ headers/dylib；本轮未下载、安装全局包或改变依赖锁。wheel SHA-256：
`e6abeac9ac5265830c9c1541b6f96e9be37a85c2446763a46ad466c63a3837ab`。

Windows 从仓库根生成**完整**归档，经项目登记的 Air SSH 连接传入独立诊断目录：

```text
git archive --format=tar --output=.scratch/compute-apple-route-review/source.tar db01de97e10ea2be8eb45f4ed446227d84e187c7
```

以下是远程 `/Users/kiro/hyle-diagnostics/six-point-architecture-20260915` 中的实际构建/运行。
路径指已存在的前轮诊断 runtime，不是推荐的产品安装路径：

```sh
mkdir -p src-db01de97
tar -xf source.tar -C src-db01de97
sw_vers
sysctl -n hw.model
clang++ --version
clang++ -std=c++23 -O2 -pthread -mmacosx-version-min=26.2 \
  -I/Users/kiro/hyle-diagnostics/route-architecture-20260915/runtime/mlx/include \
  src-db01de97/evidence/correctness/compute-apple-route-review/mlx_quantized_lifetime.cc \
  -L/Users/kiro/hyle-diagnostics/route-architecture-20260915/runtime/mlx/lib -lmlx \
  -Wl,-rpath,/Users/kiro/hyle-diagnostics/route-architecture-20260915/runtime/mlx/lib \
  -o mlx-quantized-lifetime
./mlx-quantized-lifetime
shasum -a256 src-db01de97/evidence/correctness/compute-apple-route-review/mlx_quantized_lifetime.cc
```

远程仅运行 Windows 对应源码；报告/链接改动不改变该探针，不为此重跑。程序中的 error catch 是诊断
退出，不把进程结束替代正式 Provider 的异常排空/RAII 实现。
