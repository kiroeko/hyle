# Owning custom kernel

这是普通 C++ 消费者，不包含 HYLE `src/` 或手写 Provider 表。`kernel.h` 定义 schema；独立 shared
library 中的 `kernel.cc` 持有 move-only factory，冷 prepare 创建 immutable kernel，每次执行有独占
Invocation。`main.cc` 用同一个 Provider 完成 direct prepare 和 Graph compile/replay。
它还在释放全部执行 owner 后读取独立 CompileReport 的名称/版本；报告不延后 factory/kernel 的销毁。

此 CPU 示例不是 affine 的性能实现；它演示用户尚未放进公共算子目录的计算如何接入。真实热 kernel
可以在相同边界调用 HYLE SIMD 或自己的实现。输入/输出要求 host-accessible、对齐的 dense f32；backend、
layout、alias、workspace/replay 与失败合同均显式，不用一个示例冒充全部设备已验证。
改变数值结果的 scale/bias 属于 Operation descriptor；factory 只持有不改变语义的分块配置。这样不同
Provider 可以对同一个操作竞争，不能让切换 Provider 悄悄改变计算结果。

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
不要在任何 Provider、Context、执行对象仍存活时卸载这个 library。

资源约束按 kernel 真正需要的能力声明。静态链接到不同 EXE/DLL 的 `system_resource()` 可以是不同
实例；只有算法确实依赖某个资源实例时才使用 `exact_resource`，并从资源 owner 显式传入同一个句柄。

CUDA 版加 `-DHYLE_SAMPLE_CUDA=ON`，并使用含 `compute_cuda` feature 的安装包；源码方式会显式启用
`HYLE_COMPUTE_CUDA=ON`。CUDA Toolkit 13.3+ 与 nvcc 由用户安装，不由 HYLE 下载。`cuda_main.cc` 是
C++23 factory/consumer；单独的 `cuda_kernel.cu` library 只编译 native kernel，不解析 HYLE 头，当前用
CUDA C++17。它验证 257 元素的尾块、两个 Queue 上的独立 Invocation、两个节点的 exact capture/replay、
配置保活和接纳后错误的 Completion；不是性能测量，也不代表 CUDA 算子目录已提供。

CUDA 例子让 HYLE runtime/typed resource 访问器留在同一个 owner image，native kernel 可在单独 DLL。
静态 SDK 不承诺不同 DLL 各自链接一份 HYLE 后，其 backend 私有资源身份可任意互换；跨 HYLE provider
代码 image 的自然路线使用共享 SDK，或保持一个明确的 runtime owner。CPU 例子的通用 host mapping
通过资源对象本身调用，不能把它的成功外推为任意 backend 私有身份都可跨 runtime。

详细接口与限制见 [Compute 指南](../../../../docs/user/compute.md#自定义执行接入-direct-与-graph)。
