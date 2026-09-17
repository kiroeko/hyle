# Apple MLX/Metal：Direct Compute 与 Graph

Apple 路线在同一个 MLX Metal runtime 中保留原生 arrays、编译融合与内存管理。普通 typed Direct
和 Graph 应用固定的成熟 MLX 规则；用户不排序 Provider、不提交 timing profile。R5 已完成
[限定环境验收](../implementation/tasks/compute/apple-v1-four-route-foundation.md)，性能资格不外推其它硬件。

## 构建与三种 C++ 交付

只支持 macOS Apple Silicon。源码构建需要可用的 Apple C++/Objective-C++ 和 Metal 编译工具；
`xcrun -sdk macosx metal --version`、`xcrun -sdk macosx metallib --version` 必须成功。CLT 本身未必
包含 Metal CLI。HYLE 不代装系统 SDK；ON 缺条件失败，AUTO 不可用时关闭，OFF 不探测或下载 MLX。

```sh
cmake -S . -B build-apple -G Ninja \
  -DHYLE_COMPONENTS="compute;graph" \
  -DHYLE_COMPUTE_APPLE=ON -DHYLE_COMPUTE_CPU=OFF -DHYLE_COMPUTE_CUDA=OFF \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_STATIC=ON -DHYLE_INSTALL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-apple --parallel 2
cmake --install build-apple --prefix /your/sdk
```

完整源码嵌入使用相同选项和公开 target；安装 SDK 用 `find_package(Hyle ... COMPONENTS compute graph)`。
设置 `HYLE_STATIC=OFF` 交付 shared，使用独立 build/install prefix。shared HYLE 与消费者共用一份
shared MLX；不要另外链接另一版本或另一份 static MLX。

```cmake
target_link_libraries(my_app PRIVATE hyle::graph) # Direct-only 用 hyle::compute
hyle_stage_apple_runtime(my_app)
```

这个显式 helper 在 source 和 installed SDK 中相同。它把真实 `mlx.metallib` 放到 EXE 旁，并跟踪资源
更新；shared SDK 的 native dylib 旁也保有该文件。发布应用时保留邻置资源，不能只复制 executable。
MLX JIT 仍需要部分预编译基础 kernel，缺文件不是可以忽略的 warning。

SDK 带锁定的 MLX v0.32.2、Metal-C++ 26 和对应许可证；其 CPU/CUDA/Python/GGUF/safetensors 功能
关闭。Apple 专属头公开 MLX C++ 类型；共同 Compute/Graph 头不要求这些 SDK。

## 普通 typed Direct

下例放在返回 `hyle::Status` 的函数内：

```cpp
#include <hyle/compute/apple.h>
#include <array>

namespace mx = mlx::core;
namespace apple = hyle::compute::apple;
HYLE_ASSIGN_OR_RETURN(auto context, apple::Context::create());
const std::array inputs{
    mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2}),
    mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
HYLE_ASSIGN_OR_RETURN(auto program,
    context.prepare(hyle::compute::MatMul{}, inputs));
HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
const auto &native_output = result.get()->arrays()[0]; // 成功终态后的 lease
```

当前 typed 域还包括：

- F32/F16/BF16 的 Unary、广播 Binary、浮点 Cast、三种 GatedActivation，以及 RMSNorm/LayerNorm。
  gate/norm 在 F32 求值，最终一次回到输入存储类型；norm 可用相同类型或 F32 scale/bias。
- F32/F16/BF16/I32/Bool 的 View、Copy、Fill、Select 和 Concat；Select 使用 Bool mask。
  支持 scalar、空维度、rank≤8、原生转置与广播，实际布局由 MLX 持有。
- F32/F16/BF16 的 Reduce Sum/Mean/Min/Max、Softmax/LogSoftmax 与 TopK。Reduce 支持任意
  axis 集/keep-dim，空集合为 0/NaN/+Inf/-Inf，Min/Max 传播 NaN 并保留 ±0 优先级；half Reduce
  和所有 Softmax 必须显式 F32 accumulation。Softmax 支持 scale、广播 Bool 或同类型 additive
  mask、任意 axis；未屏蔽的 NaN/+Inf 或全 -Inf 行产生全 NaN。
- TopK 返回原始存储类型 values 与 I32 indices。同分含 ±0 按小索引优先，NaN 总在最后；
  largest/smallest 均可，sorted=false 仍返回有序结果。k=1 用原生线性 ArgMin，k>1 用原生稳定
  merge sort；MLX 内部排序空间和必要布局/精度转换仍有实际成本。Graph 用 `add_node()` 与
  `result(node, 0/1)` 消费两个输出，Direct 从成功 result 的 `arrays()[0/1]` 读取。
- 二维 F32 MatMul、二维 unsigned affine Q4/group64/F32。其它 typed 域明确拒绝。

Q4 weight 的 native dtype 是 uint32，shape
`[N,K/8]`，每 word 按 K 顺序从低 nibble 开始；scale/bias 是 F32 `[N,K/64]`，公式为 `q*scale+bias`。
Graph 描述用 `apple::affine_q4_description(N,K)`，不把 Marlin/GGUF/其它四位格式直接传入。

typed 准备也可以只给元数据，例如
`context.prepare(RmsNorm{}, std::array{apple::InputSpec{{1,2048}, mx::bfloat16},
apple::InputSpec{{2048}, mx::float32}})`；不需要制造用户 payload。精度或形状不合格时准备返回
具体拒绝。MLX 仍在冷期用隔离输入编译暖机。新增 tensor/gate/norm/reduce/softmax/TopK 域的完整
性能资格待 C1 验收，不能继承 R5 FFN 的性能结论。checked 索引与 Scan 尚未在该批 typed 接线中准入。

`context.compile(callable, input_specs)` 表达更完整的原生 region。`InputSpec` 只含 shape/dtype；
也可给 ready examples。callable 接受 native arrays 和 Context 的 GPU stream，返回数组列表或
`hyle::Result<std::vector<mlx::core::array>>`。在其中使用 MLX ops 或 `apple::lower(Operation,...)`
组合 typed 算子。完整 Direct Q4 FFN 见
[样例](../../samples/package/compute/cpp/apple_ffn.h)，没有 Graph 依赖。

## Graph 整段编译

用普通 `GraphBuilder` 的 typed operation 构图，再调用：

```cpp
HYLE_ASSIGN_OR_RETURN(auto program,
    hyle::graph::apple::compile(context, graph, ready_constants));
```

constants 按 `Graph::constant_id()` 的顺序，bind inputs 按 `input_id()` 顺序，结果按 publish 顺序。
其后复用同一 Apple Program/Invocation/OwnedResult 接口。可以释放 Graph 与原 constant 句柄，
Program 保有自己的原生 owner。HYLE 冷期检查并移除 dead pure nodes，MLX 保有整段融合；热期不遍历
HYLE DAG。完整例见 [Graph FFN](../../samples/package/graph/cpp/apple_ffn.h)。

此 Graph lowering 接受固定 F32/F16/BF16/I32/Bool 逻辑 schema、View 派生描述和上述 Q4 编码。
native arrays 的物理 stride、View 和必要布局整理仍由 MLX 管理；布局转换可能有成本，不能从
HYLE 逻辑描述推断实际 MLX 分配或声称任意 native View 都零拷贝。符号维度、资源效果、显式 control edges 和
未接入的 typed 操作会拒绝；需要这些能力时不能声称 Apple 已覆盖其它 Graph 路线的全部功能。

纯 custom Operation 可通过 `graph::apple::CompileOptions::custom` 在冷追踪时映射到
`mlx::core::fast::metal_kernel`。该回调检查完整 schema ID/版本，返回形状/dtype 匹配的 native arrays，
沿传入的同一 stream 执行。它支持同 runtime 扩展，不承诺任意外部 MTLBuffer/Queue 的零拷贝互通。

## 所有权、线程与成本

- 一个 Context 拥有一个固定提交线程与 GPU stream，按 FIFO 执行完整 region；独立 Context 可并发。
  冷 compile、隔离零输入暖机、native eval 和 checked synchronize 都在该 owner 上完成。
- callable 必须 pure，只使用传入的 GPU stream，捕获已成功完成且不可变的常量。不能在回调里调用
  同一 Context、读取 tensor payload 决定图结构或隐藏外部副作用；它会用隔离输入暖机。
- bind 只接收已经成功 ready 的 native arrays；pending 输入不隐式等待或复制。保活不允许其它别名
  同时写 payload。调用方原生 eval 应先做 checked stream synchronize，再把结果交给 HYLE。
- 同一 Invocation 在执行中或旧结果 lease 未释放时不能覆盖；不同槽可以保留旧结果。拥有型结果
  使用 MLX 生成的原始数组，不进行影子输出 copy；复制某个原生 array 句柄仍保有其独立原生 owner。
- Completion/OwnedResult 析构不隐式等待；取消仍等真实已接纳 GPU 工作排空后才回收。失败/取消槽
  不复用。原生执行错误会封闭 Context 的后续接纳，先 close 再创建新 Context。
- `close()` 停止接纳、排空并 join，Context 析构同样有 drain 成本。Program/Invocation/结果可比该
  Context 句柄长寿；关闭后新提交失败，成功结果仍可读。
- 这是 managed replay：每次一次外层提交、原生 MLX metadata/array/cache 成本、锁/原子/唤醒和一次
  checked region synchronization 都存在，不宣称零分配 recorded replay。常驻权重、编译和 slot
  可复用，形状改变需要重新准备。

`context.environment()` 提供冷创建时记录的 GPU 名称/架构、macOS runtime/build、编译 SDK/部署下限、
Metal compiler 和 MLX pin。规则可匹配这些事实；执行中不探测或计时选优。未知硬件采用合法的成熟
MLX 基线，不外推 SOTA。默认 deployment 14.4 的构建未启用 NAX，不能拿 M4 验证声明 M5/NAX 资格。

## 可选编译诊断

需要保存编译依据时，传入一个独立报告：

```cpp
hyle::graph::apple::CompileReport report;
auto program = hyle::graph::apple::compile(
    context, graph, ready_constants, {.report = &report});
if (!program) {
  // report.status 与返回错误一致；没有上一次成功编译的残留统计。
  return program.error();
}
```

报告复制环境/版本、输入输出描述、live 节点原 ordinal 及数值 traits、固定实现规则、边界和并发
政策。它不保留 Graph、Program、常量、custom payload 或设备 owner，资源释放后仍可读。
每次 compile 先 reset；报告自身内存不足也会拒绝编译。默认不传报告，执行期不构造诊断信息。
同一个 report 不能被并发编译写入。

`declared_constant_count` 与 `logical_constant_bytes` 按声明 views 统计，重复引用也重复计数；
它们不是实际驻留内存。`private_storage_bytes` 保持 unknown，MLX 内部融合、私有 heap 和 driver
分配不能从逻辑 shape 精确推断。

需要主机调用跟踪时，构建时设置 `-DHYLE_TRACE=ON`，使用已有 Tracy client。
prepare/compile、submit、checked wait 或 native owner 的执行范围随实际入口记录；默认 OFF 不
求值跟踪参数、不创建 profiler 状态。这些是主机范围，不能解释为 GPU kernel 时间。
