# Vulkan：Direct Compute 与 Graph

Vulkan 路线使用原生 buffer/device memory、pipeline、descriptor、command buffer 和 fence。
HYLE 在准备期绑定 ggml region 或 HYLE shader，Graph 冷期规划边界资源；原生片段复用录制命令，
ggml 片段复用其图和存储，但仍执行原厂编码及内部资源管理。
普通用户只选路线，不排序 Provider 或提供 timing profile。既有 F32 基础支持域已完成 Windows RTX 4070
与 Air M4/MoltenVK 的真实计算、资源寿命、三交付和
[有限性能验收](../implementation/tasks/compute/vulkan-v1-four-route-foundation.md)。
资格限所记录的设备/软件与 F32 shape 域，不承诺其它设备或完整推理模型的性能。

## 构建与运行条件

源码构建需要 C++23、Python 3、glslangValidator/glslang，以及 Vulkan SDK 提供的 glslc、
开发链接库和 SPIRV-Headers CMake package；可通过 `VULKAN_SDK` 指定 SDK 根目录。
Vulkan-Headers 与完整 ggml Vulkan 源码从中央锁取得；窄接入补丁只应用到构建目录中的副本。
静态 SDK 的导出声明不再要求原厂 `Vulkan` 开发 package；私有 runtime 通过动态 loader 取得函数。
当前 C1 正接入普通 Direct/Graph 的受管片段；本轮完整交付与性能资格仍待
[C1-Y](../implementation/tasks/compute/llm-moe-c1-ggml-vulkan-region.md) 验收。
二进制 SDK 已嵌入 SPIR-V，不需要 glslang/glslc 或运行时 shader 文件。用户系统必须提供可用的
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

基线需要 Vulkan 1.2、所用 shader 的 workgroup/内存限制；既有 F32 布局转换和 ggml 门控片段要求 512-lane
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

当前 typed 域包括：

- 二维 F32 MatMul（F32 accumulation），activation dense，RHS dense 或转置的 dense view；
  MatMul 保持已有设备/shape 性能资格。
- F32/F16/BF16 的 SiLU、GELU、GELU-tanh 门控，支持 separate/packed-halves；
  RMSNorm/LayerNorm 在最后一维以 F32 累加，scale/bias 各自可用输入类型或 F32，最终舍入一次。
- 同三浮点类型的 Reduce Sum/Mean/Min/Max（任意合法轴集合/keepdim）和 Softmax/LogSoftmax
  （任意轴、有限 scale、广播 Bool/同类型 additive mask），以 F32 累加；半精度必须显式声明。
- Copy/Fill/Select/Concat 支持 F32/F16/BF16/I32/I64/Bool 原始存储，Select 使用 Bool mask；
  浮点 Cast、13 种 Unary（包括稳定 Softplus）和六种广播 Binary 支持 F32/F16/BF16。
  浮点在 F32 求值后舍入至输出存储；整数和 Bool 只承诺本批存储操作，不承诺整数算术。
- 张量操作支持 scalar、空域、rank≤8、合法 byte stride/广播输入与 dense output。
  可选择别名的共同操作使用独立结果；View 保留真正的输入别名。
  非空元素数≤INT32_MAX，物理 byte span 和 stride 必须落在设备 buffer range/U32 地址域内。

新张量域使用 U32 storage 和固定 specialization 的原生 SPIR-V，128-lane workgroup；
不要求可选的 shaderFloat16/Int8/Int16/Int64。每个输出 word 单 writer，半精度和 Bool 尾块
没有并发位覆盖。必要转换、结果/scratch、pipeline 和 descriptor 有实际成本；性能和完整
三交付资格随 C1 验收，不能继承旧 F32 FFN 结论。其它域明确拒绝，普通 prepare 无 Graph 依赖。

`allocate` 为逻辑 byte span 向上补齐到四字节，并为 empty tensor 保留至少四字节 native
容量；`upload/download` 的 span 仍必须恰好等于 `description.required_bytes()`，只传逻辑字节。
空传输不录制 copy，仍检查设备和失败状态；upload 要求 Context 未关闭，已保活结果的 download
与非空传输一样允许在 close 后读取。`borrow_ready` 的 `available_bytes`
必须覆盖同样的 native padding，offset 同时满足四字节和设备 storage-buffer 对齐要求；
不能把仅有两个物理字节的外部 buffer 当作四字节容量。quota 与 Graph 容量报告计入实际分配。

HYLE 原生归一化对非空输入使用一次 dispatch 和固定 shared memory，不分配额外 device scratch。
RMSNorm 两遍读取输入，LayerNorm 三遍并使用中心化方差；半精度奇数宽度时一个 workgroup
处理相邻两行，避免两个组写同一 word。合法 stride/View、空域、rank≤8 均沿张量合同。
例如 `vk::prepare(context, c::RmsNorm{}, input_description, scale_tensor)` 捕获 scale，随后只绑定输入。

正尺寸 dense F32、rank1–4 的 RMSNorm（F32 累加，可选同类型 scale）和分离式 SiLU gate
默认绑定原版 ggml region。连续的兼容 SiLU、Add/Multiply 一起交付，保留原厂内部融合与存储；
二元广播至少一侧须具有完整输出形状。published 和跨片段使用的结果仍可独立读取。
准备期无法证明端口起点四字节对齐时，固定加入现有 Copy 适配；这包含描述相同但 View
起点可能改变的动态输入。已知对齐常量和计算结果直接导入，适配内存与复制属于真实成本。
其他已支持的类型/布局继续使用原生实现。bind/submit 不选择 Provider；本批规则尚待等责 P1。

Concat 接受任意正数量的同类型输入、正负 axis、rank1..8 和合法 stride/View，按输入顺序拼接，
保留原始存储位并生成独立 dense 结果。例如 `vk::prepare(context, c::Concat{-1}, left_desc,
right_desc)` 准备两个变化输入；参数数量运行期确定时，使用 `Operation` 和 `span<const Argument>`。
每最多三个输入冷期绑定为一次原生 dispatch，空组跳过；每组只写自己的输出片段，通过既有
barrier 保留组间共享 word 的先前字节。无逐次增长的中间张量、额外 device scratch 或 memset；
空输出无 dispatch，仍有合法结果 owner。Graph 的 slice→Concat 使用相同规则，不 materialize
输入 View。输入读取、字节寻址、边界 word 与组间 barrier 都属于实际执行成本。

Reduce 的空集合分别产生0/NaN/+Inf/-Inf，空输出无 dispatch；Min/Max 传播 NaN，并区分 ±0。
Softmax 的 Bool false 屏蔽对应输入；未屏蔽 NaN/+Inf 或全 -Inf 的行产生全 NaN。
LogSoftmax 保留接近1的概率对应的小负值，并避免大共同偏置抵消。
例如 `vk::prepare(context, c::Reduce{.axes = c::Axes::from(std::array{-1}).value(),
.keep_dimensions = true, .accumulation_type = c::DataType::kFloat32}, description)`
为每行产生保留末维的归约结果。

Reduce 一次 dispatch，无额外 device scratch。Softmax 将不超过2048列且满足设备 shared 容量的
行缓存为 F32，缓存最多16KiB加2KiB归约数据；更长行三遍读取。F32、轴后元素数为1或半精度
轴后元素数为偶数时一次 dispatch；半精度轴后元素数为大于1的奇数时，以每行8B统计量和第二次
逐word输出防止交叉写入。该临时存储计入 Graph 容量和 quota，按寿命复用；超设备范围准备失败。
所有选择在准备期固定。新归约域的完整三交付与等责 P1 仍待 C1 整包验收。

## 零拷贝 View

`tensor.view(c::View::slice(axis, start, count, step).value())` 以及 reshape、permute、
broadcast_to 只创建描述与共享 owner，保留原生 buffer，并在 `native_offset()` 返回实际逻辑起点。
它们不传输数据、不发起 GPU dispatch；输入仍遵守 ready 和独占修改约定。普通
`vk::prepare(context, view_operation, description)` 和 Graph 的 View 节点提供相同别名语义。
发布计算结果的 View 会延长整块原始分配的寿命，不能只按切片大小估算常驻容量。

捕获常量的 View 仍是不可变常量。符合既有 F32 MatMul 域的权重视图在准备期完成布局打包；
相同 buffer、起点和描述的视图复用同一打包结果。若未发布原始权重的视图且执行不再读取原始
布局，Program 会释放原始 owner，避免保留两份完整权重。

typed tensor 输入可使用不满足 descriptor 对齐的 View：HYLE 对齐 descriptor 起点，并以固定
push 参数传递字节余数。更换 buffer 但余数相同的 bind 只更新 descriptor；余数变化时，空闲槽
会复用已有 pipeline、descriptor sets 和 buffer 重录原生片段命令。托管片段沿固定边界绑定端口，
由原厂编码执行；HYLE 不重新进行图规划或 Provider 搜索。
重录失败时 bind 返回错误，成功重新绑定前不能提交；被导出的别名结果或 native owner 会阻止该
槽改写别名 metadata，释放后可复用。需要同时保留旧结果时使用独立 invocation。

现有 F32 GEMM 与自定义原生 kernel 保留原 descriptor 对齐域，遇到不合格起点需显式
Copy/materialize；不会静默把 View 本身实现为 copy。固定 Recorder 的 View 输出必须确实指向
已外部绑定输入的对应范围。普通 typed Recorder 写输出使用独立或 `borrow_ready` 的具名 padded
buffer，不写入 View slice 的尾字，避免修改相邻数据。

## Graph 与原生片段

用 GraphBuilder 构造 typed DAG，再调用：

```cpp
HYLE_ASSIGN_OR_RETURN(auto program,
    hyle::graph::vulkan::compile(context, graph, ready_constants));
HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
```

constants 按 Graph 常量顺序，bind 按输入顺序，结果按 publish 顺序。编译后可以销毁原 Graph 和
常量句柄；Program 保有自己的原生 owner。Graph 冷期删除 dead pure nodes、规划中间 buffer
生命周期；每个槽独立拥有可变 scratch、命令、descriptor、fence 和生成结果。HYLE 重放不遍历
原语义 DAG；托管 region 的原厂执行器仍会访问其已准备的图。
当前只接受固定 shape、pure、单输出 node（含 View）；符号、资源效果、control edge、其它必须别名节点和直接
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
- 单次原生提交在未接纳时失败可重试。组合程序已有片段被接纳后，后续失败仍返回持有前缀
  的 Completion；poll 不阻塞，wait/close 异常排空，保留首次失败且不发布成功结果。失败的
  托管/组合槽不复用。观察失败不等于 GPU 已停止。未确认终态的资源继续保留，
  close/collect 可重试；设备丢失返回 DataLoss，失败不发布成功结果。
- close 停止接纳并 drain；Context 析构有相同成本。原生结果可长于 Context 句柄，成功结果关闭
  后仍可下载。可选 quota 计 HYLE 分配的 VkDeviceMemory（含 staging、适配和边界 scratch）；
  ggml 内部资源及 driver/pipeline 私有内存不在该配额中，
  quota 本身须活过所有依赖资源。
- 冷期承担 packing、pipeline、边界内存规划、原生命令录制及每槽 ggml 图/存储。热期按固定
  片段次序提交同一 queue，等待最后实际 batch 的 fence；正常等待不阻止其他槽的 producer。
  原厂 region 的编码/私有分配仍有成本，不承诺整个 submit 零分配或只执行一次 QueueSubmit。

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

`native_dispatch_count` 是最终 HYLE 原生 dispatch 数（含必要适配），`managed_region_count`
单独计原厂 region；它们不能互相代替 kernel 数。`invocation_storage_block_bytes` 给出每个槽
规划的边界输出及 HYLE 可复用 scratch 容量，实际分配还可能有对齐成本。原厂内部容量和
driver/pipeline 分配保持 unknown。常量容量按 HYLE buffer owner 去重，
`owned_constant_allocation_bytes` 只计已知 owned VkDeviceMemory；`has_borrowed_constants` 标明
外借 owner，其外部物理别名和分配大小未知。`opaque_storage_bytes` 保持 unknown，不把 driver/
pipeline 私有分配计成零。

需要主机调用跟踪时，构建时设置 `-DHYLE_TRACE=ON`，使用已有 Tracy client。
prepare/compile、submit、checked wait 或 native owner 的执行范围随实际入口记录；默认 OFF 不
求值跟踪参数、不创建 profiler 状态。这些是主机范围，不能解释为 GPU kernel 时间。
