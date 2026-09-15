# Owning custom kernel

这是普通 C++ 消费者，不包含 HYLE `src/` 或手写 Provider 表。`kernel.h` 定义 schema；独立 shared
library 中的 `kernel.cc` 持有 move-only factory，冷 prepare 创建 immutable kernel，每次执行有独占
Invocation。`main.cc` 用同一个 Provider 完成 direct prepare 和 Graph compile/replay。
它还在释放全部执行 owner 后读取独立 CompileReport 的名称/版本；报告不延后 factory/kernel 的销毁。
CPU Graph 把输入作为 `ConstantPolicy::kMaterialize` 常量，编译后释放源 allocation；只绑定输出，
同一个 Invocation 重放三次。报告校验物化策略与字节数，但不保活常量 payload。
CPU/CUDA 的 Graph 消费同时用公开 `NodeProviderConstraint` 指定配置实例，CPU 另用全局家族 allow
筛选；约束只在冷 compile 使用，仍复用同一个 Invocation 和 CUDA capture，不增加另一条执行接口。
两个例子分别限定 1/2 个普通算子候选，并检查报告区分候选发现与真正 prepare 次数；恰好用满且没有省略
后续查询不算搜索受限。这是例子已知只有一个 Provider 的边界，不是一般 Graph 应照抄的容量配置。

此 CPU 示例不是 affine 的性能实现；它演示用户尚未放进公共算子目录的计算如何接入。真实热 kernel
可以在相同边界调用 HYLE SIMD 或自己的实现。输入/输出要求 host-accessible、对齐的 dense f32；backend、
layout、alias、workspace/replay 与失败合同均显式，不用一个示例冒充全部设备已验证。
改变数值结果的 scale/bias 属于 Operation descriptor；factory 只持有不改变语义的分块配置。这样不同
Provider 可以对同一个操作竞争，不能让切换 Provider 悄悄改变计算结果。

CPU/CUDA factory 都声明 `ProviderRouteKey{1}`，消费方检查 Graph report 保留该稳定实现键。它不是
临时 prepare token；同一 key 只有和完整 build/config、设备、语义/shape/layout、workload/完成边界
共同匹配才有性能资料意义。本例只验证传递与生命周期，不提供 profile 或性能胜负。
CPU 示例也通过公开 profile matcher 和 Graph 的空 `CompileProfileContext` 检查：缺身份不会误配成
零成本，报告保留自己的 TensorDesc，但不会伪造合格 route。三种交付都链接同一冷合同。

CPU 示例另用默认 Native Provider 编译 `[rows, 4]` RMSNorm 的完整 `rows=0..4` 域，释放原 Graph 后为
0/1/4 行分别创建 Invocation 并重放，检查具体 signature 与数值。只用公开头，不调用私有域 seam；
此功能不会让前面的 owning custom factory 自动获得 bounded 支持，它仍只承诺 exact prepare。

CPU 示例还运行 exact 与 declared-domain `ViewSpec` 旅程：动态 start 的 slice → permute → broadcast →
显式 Copy → reshape。slice 通过显式 `offset + 1 <= rows` guard 约束，完整域只编译一次，Invocation 绑定
具体 rows/offset；三种交付均实际执行该路线。
只有 Copy 物化；Graph 的原始描述可通过 `operation_kind()/view_spec()` 查看，销毁原图后继续重放并核对
结果。输出约束使用 `TensorSpec::inferred`，不把转置或广播伪装成 dense。此例不声称动态 View 已有完整域资格。

安装 static 或 shared SDK 后（同一 C++23 工具链）：

```sh
cmake -S samples/package/graph/custom_kernel -B <consumer-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<hyle-install>
cmake --build <consumer-build> --parallel 2
ctest --test-dir <consumer-build> --output-on-failure
```

完整源码接入把 `CMAKE_PREFIX_PATH` 改成 `-DHYLE_SAMPLE_SOURCE=<absolute-hyle-source>`。
Windows 请在与 SDK 一致的 Developer shell 内运行；DLL 会按 HYLE package 合同 staging。即使 HYLE 是
静态 SDK，用户 factory 仍特意放在 sample shared library 内，以验证跨 code image 的 schema 和执行。
CPU/CUDA executable 分别输出到构建目录的 `cpu/`、`cuda/`，各自 staging DLL，避免并行构建写同一文件；
用 CTest 运行即可，不需要自己调整 DLL 搜索路径（多配置生成器会再加配置子目录）。
不要在任何 Provider、Context、执行对象仍存活时卸载这个 library。

资源约束按 kernel 真正需要的能力声明。静态链接到不同 EXE/DLL 的 `system_resource()` 可以是不同
实例；只有算法确实依赖某个资源实例时才使用 `exact_resource`，并从资源 owner 显式传入同一个句柄。

CUDA 版加 `-DHYLE_SAMPLE_CUDA=ON`，并使用含 `compute_cuda` feature 的安装包；源码方式会显式启用
`HYLE_COMPUTE_CUDA=ON`。CUDA Toolkit 13.3+ 与 nvcc 由用户安装，不由 HYLE 下载。`cuda_main.cc` 是
C++23 factory/consumer；单独的 `cuda_kernel.cu` library 只编译 native kernel，不解析 HYLE 头，当前用
CUDA C++17。它验证 257 元素的尾块、两个 Queue 上的独立 Invocation，以及动态输入和编译时物化常量
两种两节点 exact capture/replay。物化场景在编译后、capture 前释放原 device allocation；还检查配置
保活和接纳后错误的 Completion。它不是性能测量，也不代表 CUDA 算子目录已提供。

同一个 CUDA 示例还把 pinned-host 输入/输出声明为外部资源，让 Graph 插入 H2D/D2H，并在独立的
compute/transfer Queue 上执行两个 native kernel。两个 Invocation 使用各自的 Queue 对、输入输出与
workspace，在 commands 和 exact capture 两种模式下先提交再分别等待 Completion。测试拒绝 host wait，
检查真实 device-event 依赖与变化输入，避免用单 stream 或不变数据掩盖跨 lane 顺序/存储共享错误。
这是异步调度与公开接入示例，不是 Q4 算子或厂商 Provider 的能力声明。

CUDA 例子让 HYLE runtime/typed resource 访问器留在同一个 owner image，native kernel 可在单独 DLL。
静态 SDK 不承诺不同 DLL 各自链接一份 HYLE 后，其 backend 私有资源身份可任意互换；跨 HYLE provider
代码 image 的自然路线使用共享 SDK，或保持一个明确的 runtime owner。CPU 例子的通用 host mapping
通过资源对象本身调用，不能把它的成功外推为任意 backend 私有身份都可跨 runtime。

详细接口与限制见 [Compute 指南](../../../../docs/user/compute.md#自定义执行接入-direct-与-graph)。
