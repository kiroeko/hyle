# 参考实现：用来回答什么问题

参考代码用于理解成熟合同与成本，不是API并集或强制调查清单。
产品依赖、版本和许可只看[中央锁](../../cmake/HyleDependencyLock.json)与
[第三方说明](../../THIRD_PARTY_NOTICES.md)，本页不复制版本、支持或性能矩阵。

## 基础与运行时

| 问题 | 参考 | 借鉴边界 |
|---|---|---|
| compiler/布局与失败值 | C++标准库、Abseil、Folly | 值/借用、context、可移植语义；不建立第二错误registry |
| 分配与容器 | mimalloc、Abseil、Boost.Intrusive、moodycamel/rigtorp | allocator、layout、membership、进展和失效；机制来源不抹去Hyle的owner责任 |
| Fiber与结构化并发 | Marl、Tokio、Folly、oneTBB | work-stealing、wake、scope/drain、非抢占priority；不引入第二scheduler |
| IO与进程 | liburing、libuv、Asio、Tokio、Boost.Process | completion/readiness、参数寿命、取消、tree/reap；不直接复制其event loop |
| 文本与协议 | simdutf、nghttp2、OpenSSL、Beast | 成熟codec/framing/TLS与Upgrade边界；业务policy归消费者 |
| SIMD | Highway、xsimd | target lowering、数学与完整kernel；不复制通用register库 |

## 计算与真实消费

| 问题 | 参考 | 关注的反例 |
|---|---|---|
| backend资源与region | ggml、llama.cpp | buffer import、allocator、stream/event、packing及整图边界成本 |
| 量化与推理状态 | llama.cpp、ds4、vLLM、SGLang | 实际端口格式、recurrent/KV、paged planning、expert迁移和关闭 |
| CPU primitive | oneDNN及原生CPU实现 | executor、scratchpad、线程/packing与嵌套资源 |
| CUDA | CUDA/cuBLAS、FlashInfer等 | capture、workspace、dynamic planning、partial acceptance与终态 |
| 设备侧编排与整段decode融合 | TPU Megakernels | 冷期布局、显式预取、循环缓冲、scratch复用及设备侧起草／验证／接受循环；区分整层kernel融合与跨轮设备控制流 |
| Apple | Metal、MLX | array/stream/compiled region、输出owner及跨runtime边界 |
| Vulkan | Khronos规范、ggml Vulkan、IREE | execution/memory可见性、descriptor/command寿命与portability |

引擎通常不是可直接借入的小型C++机制库。借其算法或kernel时仍要用窄接入保留来源、资源与交付边界；
Python包装、整模型runtime、服务cache和进程政策不会因提供参考而成为Hyle依赖。

渲染/资源流送、数值管线与异步服务用来检验通用性：同样的租约、group-ready、shape、失败和关闭
是否成立，而不是要求Hyle实现这些领域的完整框架。硬件宣传或厂商结果只能提供需求线索，不能充当本项目性能结论。

## TPU Megakernels 源码核对

依据用户提供的本地仓库`C:\public\project-hyle\tpu-megakernels`。本次核对快照为
[`Inferact/tpu-megakernels`，`a120ca43175604d092c95fb4125be5df4cb07771`](https://github.com/Inferact/tpu-megakernels/tree/a120ca43175604d092c95fb4125be5df4cb07771)，
仅固定参考源码，产品依赖仍由中央锁管理。

- `qwen/decode_megakernel.py`：`transformer_stack`把模型层计算放入Pallas调用；权重块通过bank ring预取，DMA的`start`／`wait`与信号量明确约束读写。`scratch_shapes`、`input_output_aliases`和VMEM容量在程序准备时确定。
- `qwen/dflash.py`：`make_round_fn_kernel`连接草稿、整块验证、接受长度与状态选择；`make_scan_fn`用设备侧`while_loop`推进多轮，保留停止词、上下文和输出上限。设备侧循环与单个融合kernel是两层机制。
- `pool_alias.py`：按物理布局计算scratch容量并建立VMEM别名；文件明确要求新视图在读取前完整覆盖。这适合用来审查工作区寿命、覆盖与复用合同。

与渲染引擎GPGPU相通的部分，是先准备程序和资源，再通过显式依赖组织设备工作，减少主机往返和重复搬运。
HYLE可据此审查冷装配、Graph／Source接力、物理IO载体和工作区寿命；模型融合与硬件布局仍按实际后端放置。
本次只读源码，未运行TPU验证；CPU解释器测试不证明TPU的lowering、VMEM容量、DMA时序或性能。
该仓库还提供`force_acc`合成接受长度选项，引用性能结果时须核对实际接受策略与计时边界。
