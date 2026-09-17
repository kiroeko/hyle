# Vulkan：Direct Compute 与 Graph

Vulkan 路线使用原生 buffer/device memory、pipeline、descriptor、command buffer 和 fence。
HYLE 在准备期绑定成熟 ggml shader，Graph 冷期规划资源，重复执行复用录制好的命令。
普通用户只选路线，不排序 Provider 或提供 timing profile。当前支持域已完成 Windows RTX 4070
与 Air M4/MoltenVK 的真实计算、资源寿命、三交付和
[有限性能验收](../implementation/tasks/compute/vulkan-v1-four-route-foundation.md)。
资格限所记录的设备/软件与 F32 shape 域，不承诺其它设备或完整推理模型的性能。

## 构建与运行条件

源码构建需要 C++23、Python 3、glslangValidator/glslang；Vulkan-Headers 从中央锁取得。
二进制 SDK 已嵌入 SPIR-V，不需要 glslang 或运行时 shader 文件。用户系统必须提供可用的
Vulkan loader/driver；HYLE 不下载驱动，也不把设备枚举当作计算成功。

```sh
cmake -S . -B build-vulkan -G Ninja \
  -DHYLE_COMPONENTS="compute;graph" \
  -DHYLE_COMPUTE_VULKAN=ON -DHYLE_COMPUTE_CPU=OFF \
  -DHYLE_COMPUTE_CUDA=OFF -DHYLE_COMPUTE_APPLE=OFF \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_STATIC=ON -DHYLE_INSTALL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel 2
cmake --install build-vulkan --prefix /your/sdk
```

源码嵌入使用相同选项、`add_subdirectory` 和 `hyle::compute` / `hyle::graph`；安装 SDK 使用
`find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute graph)`。`HYLE_STATIC=OFF` 生成
shared SDK，使用独立 build/prefix。Windows 消费者调用 `hyle_stage_runtime_dependencies(app)`
把 HYLE 及需要的 C++ runtime DLL 放在 EXE 旁。Vulkan loader 使用系统提供的 `vulkan-1.dll`；
Linux 默认 `libvulkan.so.1`，macOS 默认 `libvulkan.1.dylib`。

macOS 使用已安装的 loader/MoltenVK。若 loader 不在动态库搜索路径，可在 ContextOptions 的
`loader_path` 指定其绝对路径；驱动发现仍遵守该 loader 的配置。验证副本使用进程级
`DYLD_LIBRARY_PATH=/opt/homebrew/opt/vulkan-loader/lib`，不改全局系统配置。

基线需要 Vulkan 1.2、所用 shader 的 workgroup/内存限制；平坦转换和门控要求 512-lane
workgroup。可重新绑定输入的普通 Invocation 还需要设备真实支持
`descriptorBindingStorageBufferUpdateAfterBind`。缺条件明确报错；固定 caller buffers 的
Recorder 不强制采用可重绑定 descriptor。默认拒绝软件 device，不做 CPU 模型 fallback。
`OFF` 不探测 SDK、编译 shader 或获取路线依赖；`ON` 缺构建条件失败。

## 普通 Direct Compute

以下代码置于返回 `hyle::Status` 的函数中：

```cpp
#include <hyle/compute/vulkan.h>
#include <array>

namespace c = hyle::compute;
namespace vk = c::vulkan;
HYLE_ASSIGN_OR_RETURN(auto context, vk::Context::create());
HYLE_ASSIGN_OR_RETURN(auto desc,
    c::TensorDesc::dense(c::DataType::kFloat32, std::array<std::int64_t, 2>{2, 2}));
const std::array<float, 4> x{1, 2, 3, 4}, weights{2, 1, -1, 3};
HYLE_ASSIGN_OR_RETURN(auto input, context.upload(desc, std::as_bytes(std::span(x))));
HYLE_ASSIGN_OR_RETURN(auto weight, context.upload(desc, std::as_bytes(std::span(weights))));
HYLE_ASSIGN_OR_RETURN(auto program, vk::prepare(context, c::MatMul{}, desc, weight));
HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
HYLE_RETURN_IF_ERROR(slot.bind(input));
HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
std::array<float, 4> host_output{};
HYLE_RETURN_IF_ERROR(context.download(result->tensors()[0],
    std::as_writable_bytes(std::span(host_output))));
```

prepare 参数按操作的自然输入顺序排列：TensorDesc 表示每次绑定的输入，Tensor 表示不可变、
已经 ready 的常量。示例把 weight 捕获并冷 packing；变化的 RHS 可以改传其描述，随后
`slot.bind(input, weight)`。变化 RHS 的布局转换发生在 GPU 程序内，成本与 scratch 不会消失。

当前 typed 域为二维 F32 MatMul（F32 accumulation），activation dense，RHS dense 或转置的
dense view；F32 SiLU 门控支持 separate/packed-halves。其它数值和格式明确拒绝，不暗中降为
FP16 或套用其它路线的 Q4。普通 prepare 无 Graph 依赖。

## Graph 与原生片段

用 GraphBuilder 构造 typed DAG，再调用：

```cpp
HYLE_ASSIGN_OR_RETURN(auto program,
    hyle::graph::vulkan::compile(context, graph, ready_constants));
HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
```

constants 按 Graph 常量顺序，bind 按输入顺序，结果按 publish 顺序。编译后可以销毁原 Graph 和
常量句柄；Program 保有自己的原生 owner。Graph 冷期删除 dead pure nodes、规划中间 buffer
生命周期；每个槽独立拥有可变 scratch、命令、descriptor、fence 和生成结果。重放不遍历原 DAG。
当前只接受固定 shape、pure、单输出 node；符号、资源效果、control edge、alias schema 和直接
publish 输入/常量会明确拒绝。

[Direct FFN 样例](../../samples/package/compute/cpp/vulkan_ffn.h)展示 typed prepare 和 Recorder
组合：成熟 gate/up MatMul → SiLU → down → 应用自定义 SPIR-V。Recorder 是固定 caller buffers
的原生入口，输出何时可以覆盖由 caller 决定，返回 Completion。[Graph FFN 样例](../../samples/package/graph/cpp/vulkan_ffn.h)
通过 custom Operation 的 schema 与冷 `CompileOptions::custom` 返回 Kernel、dispatch groups
和 push constants；该映射只在编译时调用，shader 可实现完整融合片段。示例的
[GLSL](../../samples/package/compute/cpp/vulkan_affine.comp)和嵌入 SPIR-V 一起提供，安装消费者
不需要从 HYLE build tree 取得 shader。

Kernel 的 storage bindings 按输入顺序再接输出，使用一个 set 和一段 push constants。
应用负责合法 SPIR-V、设备已启用的 feature、访问范围以及声明的 pure/shape 语义。

## 资源、并发、失败与成本

- 一个 Context 选择一个真实 compute queue/family；队列 host 访问由对象局部互斥保护。独立槽
  可在不同线程使用；同一个槽是 thread-compatible，不允许并发 bind/submit 或移动句柄。
- Tensor 是原生 buffer owner 的轻量句柄；复制不复制 payload。常量在 Program 存活时不可变；
  运行输入在真实 GPU 终态前不可经外部别名修改。upload/download 使用 staging 并处理 flush/
  invalidate，是同步调用；不要与读写相同 buffer 的在途工作重叠。
- 普通结果在 checked 成功完成后才可读取。保留 result lease、复制出的 Tensor 或导出的
  `native_owner()` token 都会阻止同一槽覆盖输出；释放这些持有者后才能复用该槽。
- `borrow_ready` 只接纳同 VkDevice/queue family、已完成并交接 memory visibility 的 native
  buffer，还必须提供保活 token。它不等待未知外部任务、不做跨设备复制或 ownership transfer。
  访问原生 queue 使用 `with_queue`，回调必须在返回前完成自己提交的外部工作。
- Completion 析构不隐式等待。丢弃结果后 Context 继续保有已接纳工作，collect、后续提交和 close
  推进回收。取消不能抢占 GPU，也不能提前释放输入、命令或 fence。
- 提交 OOM 不等待未接纳的 fence；观察失败不等于 GPU 已停止。未确认终态的资源继续保留，
  close/collect 可重试；设备丢失返回 DataLoss，失败不发布成功结果。
- close 停止接纳并 drain；Context 析构有相同成本。原生结果可长于 Context 句柄，成功结果关闭
  后仍可下载。可选 quota 只计实际 VkDeviceMemory（含 staging），不含 driver/pipeline 私有内存，
  quota 本身须活过所有依赖资源。
- 冷期承担 packing、pipeline、内存规划与命令录制。热期是 idle descriptor 更新、一次 native
  submit、实际 fence 完成与结果管理；不分配 GPU 资源、重新录制或运行时计时选优。

`context.environment()` 保存型号、vendor/device/driver ID、driver/API/loader 版本、subgroup、
shared memory、header/compiler 和 shader pin。默认规则在冷期匹配，执行不重新探测。未资格化
设备采用满足限制的成熟基线或明确拒绝；Windows NVIDIA 和 MoltenVK 的证据互不冒充，AMD/Intel
及其它 dtype 的性能保持未资格化。

## 可选编译诊断

需要保存编译依据时，传入一个独立报告：

```cpp
hyle::graph::vulkan::CompileReport report;
auto program = hyle::graph::vulkan::compile(
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

`native_dispatch_count` 是最终原生 dispatch 数；`invocation_storage_block_bytes` 给出每个槽
规划的输出及可复用 scratch 容量，实际分配还可能有对齐成本。常量容量按 HYLE buffer owner 去重，
`owned_constant_allocation_bytes` 只计已知 owned VkDeviceMemory；`has_borrowed_constants` 标明
外借 owner，其外部物理别名和分配大小未知。`opaque_storage_bytes` 保持 unknown，不把 driver/
pipeline 私有分配计成零。

需要主机调用跟踪时，构建时设置 `-DHYLE_TRACE=ON`，使用已有 Tracy client。
prepare/compile、submit、checked wait 或 native owner 的执行范围随实际入口记录；默认 OFF 不
求值跟踪参数、不创建 profiler 状态。这些是主机范围，不能解释为 GPU kernel 时间。
