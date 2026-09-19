# M1-C：ggml CUDA 与 FlashInfer 原生接入

## Closure card

- owner：`/root`；baseline：`0f54dad4fd95714ef3a03f455f04dc746edccaea`，Windows main。
- 当前阶段：登记新身份 QUEUED；尚未授权产品实现，后续逐 owner 进入 SCOPED/AUDITED。
- 拟新增：`compute_provider_ggml_cuda`、`graph_provider_ggml_cuda`、`compute_provider_flashinfer`。
  分别拥有 Direct/native program、Graph region 接线与原版 attention 接入，不建立空 target。
- 既有 owner：`compute_ggml_runtime`的 CUDA vendor/资源补丁投影，`compute_route_cuda`与
  `graph_route_cuda`的普通工厂组装；按本卡新需求重定界。原生 backend 借用/错误机制先复用，
  只有具体缺口才将 `compute_backend_cuda`纳入，不预先重审整个 CUDA runtime。
- trigger：[M0清单](llm-moe-provider-integration-ledger.md)的 M1-C/Q-CUDA-API；现有 CUDA
  普通入口没有 ggml region 和 FlashInfer，ggml 公开 API 缺少外来 device buffer/stream 接缝。
- 先决：[M1-G](llm-moe-m1-ggml-runtime.md)与[M1-R](llm-moe-m1-route-binding.md)均 CLOSED。
  不恢复旧 C1，不修改旧 ggml CPU 或 oneDNN 的 BLOCKED 条件。
- 主纵切：本地 Qwen3.5-2B 所需 head256 prefill/decode；同 GPU 上原版 ggml 片段与成熟
  GEMM/attention 组合，显式状态输入输出、两个 invocation 与旧结果保活。这里只验 seam，
  F01–F19 的完整批量映射与模型旅程仍归 M2/M3。
- 正交纵切：无模型标签的普通 F32 数值 region，重复执行、独立请求与可恢复准备/运行失败。
- scope：原版程序的构建、资源、错误、调用、默认入口与三交付；保留成熟 region 内部融合。
- non-goals：自研算子/shader 扩张、调查、优化、比较或删除；数学 kernel 改写；模型加载；
  多 GPU/模型级 CPU+GPU 执行；公共治理与性能仪器重写。
- tier：P0，签资源与接入成本结构；M2 承担成熟默认路径的适用 P1，M5 前不比较冻结自研。
- 预算：本 seam 源码/设计总2小时，包含 Q-CUDA-API，不另叠加。前序只读输入从
  2026-09-19 21:43 UTC开始，22:37–22:45续查；保守计入已耗用60分钟，余60分钟。
  编译/已声明验证时间不冒称研究；每根因最多两次纠正，范围变化重新审计且不重置预算。
- 机器：Windows/WSL SM89 原生与三交付，不支持 CUDA 的平台验 OFF；重型串行、parallel2。
  Windows DLL 邻置，无 UAC、电源策略修改或远端源码回灌。
- stop：有限接入与适用门完成后关闭；源文件存在、空 target、枚举、copy-only 不构成验收。

## 已锁来源与具体接口事实

沿[M0源码记录](../../../../evidence/correctness/compute-llm-moe-m0/source-map.json)：
ggml `456172ec733a135778adcd32d00e576a58232e45`（MIT）；FlashInfer
`69ff11fc4954396d98326656dc85debd2223f637`（Apache-2.0）。不升级 pin，不把 Python/Torch 带入 SDK。

1. ggml 的 `include/ggml-cuda.h`提供 backend/buffer type，没有外来 GPU buffer/stream import。
   buffer context 会释放自有指针，不能仅改 `tensor.data`冒充合法零拷贝。
2. `common.cuh`的 CUDA context 持有 stream、cuBLAS/workspace、pool 和 graph；
   `ggml-cuda.cu::ggml_backend_cuda_graph_compute`区分初次执行与稳定 capture。
   不能冷准备时提前执行用户可变状态，也不能把 managed 路径称为零查询 prepared replay。
3. `CUDA_CHECK`可调用 abort，context/pool/buffer/graph/event 析构同样检查 CUDA 错误。
   仅最外层 catch 不够。capture counter 在 BeginCapture 前增加，失败必须退还计数并结束
   已取得的 capture，否则清理可能永久等待。修正限资源/失败边界，不改数学与选择启发式。
4. HYLE 已有 primary-context 借用、native Status 翻译与 admitted Queue/Completion，也有独立
   delegated region 合同。先复用这些机制，四路线不因此共享一个执行器。
5. FlashInfer 原版 `SingleDecodeWithKVCacheDispatched`与`SinglePrefillWithKVCacheDispatched`
   接受显式 stream/tmp、返回 cudaError_t；派发仍含 capability/occupancy/attribute 工作。
   Windows/WSL 可编译性与准备/重复执行合同尚待资格化，header 存在不是 PASS。
6. HYLE 已有 `ScaledDotProductAttention`的 layout/mask/scale/causal/accumulation 描述，
   先核对真实数值与 causal 对齐合同，不再建立同义 attention API。

## 有限拟定 FIX（AUDITED 前不实施）

| ID | 准入与承载 | 允许修改 | 验收 |
|---|---|---|---|
| C1 | BORROW 原版依赖 | 一个 ggml vendor 源按需启用 CUDA；FlashInfer 原版 C++ 入口；复用 lock/cache/package/license | CUDA OFF 不获取/探测；三交付从同份 target/资源描述导出 |
| C2 | BORROW 算法、HYLE 资源边界 | ggml primary context、stream、外来 buffer owner、native error、部分回滚与无抛析构桥 | 不误 free 外来指针，不 abort 运营失败，capture 失败不遗留计数/活资源，不假报完成 |
| C3 | BORROW 原版 region | 每 invocation 保留 ggml program；Direct/Graph 共用必要 schema/资源映射，Graph 类型不下沉 Compute | 真实 norm/MatMul/状态片段、重复执行、两个请求、旧结果与放弃 completion；如实声明 managed/prepared 成本 |
| C4 | BORROW 原版 attention | Q-CUDA-API 先证明 C++/head256/SM89，再绑定既有公共描述 | Windows/WSL prefill/decode；布局、causal、scale、累加与 workspace/stream；失格先查清单中的成熟替代 |
| C5 | HYLE 冷规则与普通入口 | 组装已资格化成熟实现；使用 M1-R 模型/环境事实，保留真实图边界 | 未知条件不误命中，消费者不排序 Provider/提供 timing；同 GPU 正确完成与失败回收 |
| C6 | 交付与生命周期 | 适用测试、三交付、指南与有限证据归档 | 普通 Direct/Graph 实际执行，producer 隐藏与 prefix 搬移，DLL 邻置；未运行项不写 PASS |

AUDITED 前固定具体 stream/completion 接法、buffer 保活、并发/关闭、失败退场、冷/热成本
等级与公开工厂形状。借用 stream 或独立 native runtime 均须保留原版融合并计入必要 copy/sync；
不能因熟悉某种 vtable 而强迫上游逐 kernel 执行。

## 预声明验证与边界

- QUEUED/SCOPED/AUDITED 元数据提交只验 docs、closure（含中央 CMake 登记对账）与 diff。
  selector 因中央图文件保守列出的产品门保留计划，在实际产品候选固定后执行；阶段检查不是
  selector 全量 PASS，不把未运行 native/交付门写成通过，不在每次状态登记时重建全库。
- 先做有限 Q-CUDA-API 原版 C++ 资格 bundle；探针只签入口事实，不计 Provider 接入/模型覆盖。
  不靠复写 kernel、降低数值要求或改用更小 head 绕过失败。
- 真实 ggml region 与 FlashInfer prefill/decode，普通入口核对数值、状态、多 invocation、
  外来 buffer/stream 销毁顺序、prepare OOM、部分 enqueue/capture 失败与晚到完成。
- Windows/WSL 适用 native、host owner sanitizer 与三交付；CPU 插桩不代签 GPU driver。
  Mac 只签 CUDA-OFF。按影响复用未变消费者收据，不重跑已通过的全库门。
- SDK consumer 独立配置，隐藏 producer、搬移 prefix 后执行；可复用身份/flags 匹配的编译
  产物，避免反复冷编同一第三方库。完成后归档并清理本卡产物，保留缓存与本地模型。
- selector、docs/closure、统一 Compute+Graph LOC 按既有协议执行。本卡不解除冻结，也不签
  64域×四路线覆盖；下一层批量算子/片段接入仍由 M2 有限卡承载。
