# C1-AA：oneDNN 归一化及原生 post-op 接入

## Closure card

- owner：`/root`；`compute_provider_onednn`、`graph_provider_onednn`。
- baseline：`4a7685523d139a54aab9043bdf3d5b767a2bed20`，工作树干净。
- trigger：用户要求先实际绑定成熟 Provider，再比较、补缺和删除冗余实现。C1 已准入的
  F03/F04 及本地 Qwen3.5-2B 的归一化/缩放/残差片段仍主要走 Native；锁定 oneDNN 已有
  RMSNorm、LayerNorm、mixed parameter dtype 和 fused binary post-ops，现有构建却仅启用
  MATMUL。该新增真实 consumer 域满足前轮新增能力/公开合同 reopen trigger；不是旧卡续跑。
- 主 consumer：普通 CPU Direct/Graph 的 RMSNorm/LayerNorm，及 norm→Mul/Add 片段。
  正交 consumer：多维批量归一化、无参数/带参数、两个 invocation、重绑及原地输出。
- scope：复用 oneDNN 3.13.2 的 normalization primitive 与原生 post-ops；冷绑定、调用方
  workspace、借用 executor、失败回报及普通路线；对应构建/三交付、数值/生命周期与公平性能。
- non-goals：不自写归一化算法，不接全部 oneDNN 算子，不改 SIMD、Sched、公共治理/性能仪器，
  不改既有 BRGEMM/gated-MLP 算法，不自动解除 ggml-CPU BLOCKED，不把局部接入记为 C1 完成。
- 依赖：现有 Compute/Graph/Mem、锁定 oneDNN；真实直接消费者为 CPU/Graph CPU 默认工厂。
  私有 CMake dependency options 属本接入范围；不重写构建系统。
- 性能：P1，完整同合同 Direct/Graph 对比现有 Native；oneDNN 内部分配、转换、同步、Graph
  边界和融合损益全部计费。未完成数值/资源资格前不定赢家；不靠省略检查赢。
- 预算：两条具名机制（原版 Graph partition、primitive/post-ops），源码选择最多60分钟；
  产品按有限 AA01–AA04 实施，最多两轮具名性能纠正；一个稳定不可变候选执行适用 P1。
- 预声明验证：selector 所选 gates；W1 static/shared、WSL GCC14，x64 oneDNN 实际运行；
  独立 FP64 oracle、F32/F16/BF16、2048及不规则宽度、decode/prefill、特殊值、大均值小方差、
  精度/布局拒绝、OOM、两个 invocation/并发/重入/资源失败、Graph 可观测中间输出/分叉边界。
  CMake seam 变化执行适用 source/static/shared installed consumer 与依赖隔离；不代签 ARM。
  DLL 邻置、PATH unset，parallel2、重型验证串行，复用固定构建目录。
- 停止：本轮适用验收齐全才 CLOSED；未运行项如实保留，实质 blocker 具名裁决。C1 其他操作、
  平台/模型验证继续归主卡；不以本轮关闭替代全部范围。

## 有限待审项

| ID | 缺口 | 允许边界 |
|---|---|---|
| AA01 | 构建只启用 MATMUL/SEQ，成熟 norm 未进入产品 | 固定 pin 不变，增加 LAYERNORM、借用 HYLE executor 的 THREADPOOL runtime |
| AA02 | Direct 未绑定上游 norm | 共享私有 immutable primitive plan、每 invocation stream/memory handles、caller scratchpad；F32 accumulation，dense F32/F16/BF16，明确不支持域 |
| AA03 | Graph 未保留原生 norm post-op 融合 | 原有 region 扩展识别 norm→Add/Mul，不吞掉 published/多消费者中间结果，不降低半精度中间舍入要求 |
| AA04 | 没有该实际路线的资格与默认规则 | 正确性/所有权/交付先通过；固定同合同 full-cost 规则及文档，再裁决 Native 重复域是否可移除 |

SCOPED 登记提交为 `98cf0742`。

## AUDITED：机制选择与有限 FIX

选择原版 **normalization primitive + binary post-ops**，不复制其算法。固定源
`71094df4` 的 `dnnl_layer_normalization_forward_primitive_desc_create_v2` 支持 RMS flag、
参数类型和原地输出；`jit_uni_layer_normalization` 与 `ref_layer_normalization` 使用中心化方差，
并直接执行 binary post-ops。原版 Graph 的同类 partition 最终也使用上述 mature kernel。
本批所需链没有 Graph 特有算子；使用 primitive 能明确每 invocation 的 stream/memory handle
所有权，避免引入该 Graph kernel 的线程局部 execution-args cache。并非宣称 primitive 无分配：
`primitive_iface.cpp` 每次转换 execution args 仍有 vendor metadata 分配，全部计费。

AA01 的具体失败证据：该 pin 的 `primitive_iface.cpp::dnnl_primitive_execute` 在普通路径尾部
才调用 after_exec_hook；`cpu_stream.hpp` 直接清空当前 pool，没有恢复嵌套调用前的 pool。
有限修正为构建目录中的单文件 overlay：用栈上 scope guard 保证正常/异常退出都执行 hook 并恢复
旧 pool。固定源 SHA/精确锚点失败即停止，原缓存/checkout不改动；不改算法或另建 patch 框架。
嵌套回归进一步证实 `utils.cpp::activate_threadpool` 在已有 active pool 时不会替换它；scope
进入时必须先保存并清空旧 pool，再执行原 before hook，退出时恢复。否则内层串行 norm 错用
外层 executor，回归得到 `operation invocation already has work in flight`；这是一行受审修正，
没有放宽重入/在途检查。

AA02 的外部分配注入在 MSVC `/EHsc` 下复现进程退出（code3）。该模式假定 extern-C 不抛异常，
不满足 oneDNN C 入口内部 C++ 分配的实际语义，见 [MSVC 异常模式合同](https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170)。
仅本 adapter translation unit 使用 `/EHsc-`，保留同步 C++ 异常展开和捕获；不捕获 SEH、不修改
全局编译合同。注入回归逐个失败 vendor 热路径 new，要求 accepted ResourceExhausted、失败槽
退休、其他槽仍可执行；不是从 C API 名称推断不抛。

| FIX | 冻结实现与验收 |
|---|---|
| AA01 | MATMUL 保留，加 LAYERNORM；THREADPOOL 借用既有 ParallelExecutorRef，不创建私有池。冷 primitive 创建保存/恢复 oneDNN 线程局部 concurrency 设置；并发数量可表达性检查，不截断成错误整数 |
| AA02 | 私有共享 plan 保存原版 primitive/descriptor/engine；每 invocation 独立 stream、memory handles 和 caller scratchpad，重绑更新 handles。dense 非空末轴归一化，F32 accumulation，F32/F16/BF16 input/output；scale/bias 可以同输入 dtype 或 F32，但二者类型不同的域明确留给合格 Native。冷提案不创建 vendor artifact；准备再次核对精度/shape。精确 input0 alias，其他重叠仍由共同绑定检查拒绝 |
| AA03 | Graph 识别一个 norm 后最多4个 dense F32 Add/Mul post-ops，保持原版融合；支持 scalar、最后轴向量和全形状 RHS。published/多消费者边界停止扩张，按实际数据流匹配并校验可收缩性。半精度 Direct norm 接入；半精度跨节点链暂不融合，避免少一次中间舍入。既有 BRGEMM gated MLP 不改算法 |
| AA04 | 独立 oracle、类型/形状/错误域、alias、两槽重绑/旧输出、并发/嵌套 executor、allocator/映射故障与普通 Direct/Graph；三交付及 full-cost P1 签完才冻结默认域并清理被完全替代的自有域。无证据不删除有效 Native/SIMD |

公开 API 不新增。普通 `cpu::create_context` 和 Graph CPU 工厂已有相应 Provider 注册；扩展其
冷提案/精确准备即可接入。新增路线在 AUDITED 产品候选中待资格验证，不预称优胜。执行只调用
已绑定 primitive，不搜索/计时。编译/运行第三方异常在 adapter 边界转 Status；HYLE allocator
管理 wrapper，vendor opaque code/metadata 数量未知如实报告 unknown，不宣称受 HYLE quota 全控。
执行同步结束后返回 accepted terminal；失败不重放到另一 Provider。partial creation 用 RAII 回收；
两个 invocation 不共享可变 handles/scratch；映射失效、flush/close 保留首个错误。

七问：① oneDNN 3.13.2/Apache-2.0 pin不变，原生算法 BORROW，升级重新资格化；②模型归一化
和批量数值 consumer 共用普通接口，范围外域明确fallback；③复用 Compute/Graph/Mem和借用executor，
不重造 Sched；④线性归一化、原生融合，记录 vendor 每次metadata成本并做同合同P1；⑤无C/FFI
产品；⑥私有头无vendor类型泄露，source/static/shared与依赖隔离门适用；⑦普通入口不要求选库，
guide记录精度、已验证域和借用executor寿命。

## 首批实现与局部验证

- AA01–AA03 已接入普通 CPU Direct/Graph，原版 norm primitive 与 post-ops 没有改数学算法。
  热路径因 vendor execution-args 分配如实为 **Managed**；prepared-only 不接纳，BRGEMM 仍为 Prepared。
- W1 MSVC 首轮 F32 RMSNorm/LayerNorm、参数/alias、2048及不规则宽度、FP64 oracle、特殊值/
  中心化方差、借用 executor/两个共享 artifact invocation、普通 Graph 融合/可观测边界通过。
  HYLE metadata OOM 逐分配点回收及 vendor warm-new OOM 转终态、坏槽退休/新槽恢复已有回归。
- 本机13900K的 oneDNN F16/BF16 原生 norm 不支持；固定源 `cpu/platform.cpp:112` 和
  `jit_uni_layer_normalization` 要求 AVX512 或 AVX2-VNNI2 等相应资格，不能以 F16C 等同替代。
  最初测试无条件期待 oneDNN identity 被证伪，现记录 ISA_UNSUPPORTED、数值 UNPROVEN；
  默认工厂正确回落 Native，现有半精度 state Graph 独立数值回归通过。没有删检查、添加整张量
  转换或把这部分写成 oneDNN 数值 PASS；对应 ISA 原生资格仍待有设备时验证。
- shared、WSL、完整 package/selector 扩展门及公平 P1 尚待完成，owner 保持 AUDITED。
  AA04 的默认优选/Native冗余删除尚未裁决；本节不是 C1 或本轮 CLOSED。
