# M2 组合路由资格完整迁移

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- component / owner：`compute_provider_flashinfer`、`compute_provider_ggml_cuda`、
  `compute_provider_marlin`、`compute_route_apple`、`compute_route_vulkan`、
  `compute_provider_ggml_vulkan`；`/root`。
- baseline commit：`35e488a2074f22a73505bd32ed0800600f378309`。
- trigger 与证据：用户明确要求路由系统“完全重构”。前一张
  [组合资格卡](m2-route-qualification.md)只要求“至少一条”真实迁移并只迁移 CCCL，错误地把共同
  matcher 完成当成全部现存路由完成。只读盘点仍发现 FlashInfer/ggml CUDA/Marlin 的 CUDA 架构
  资格、Marlin/Apple/ggml Vulkan 的量化格式资格，以及 Vulkan 按设备/subgroup 选择 shader 的手写
  静态路由。
- 正常 consumer 任务：用户选择 CPU/CUDA/Apple/Vulkan 路线；路线 Context 捕获原生事实；每个
  Provider/原生编译器在冷 prepare/compile 用共同 `RouteRule` 判定已验证的静态实现域，再运行本地
  shape、资源、SDK capability 和生命周期检查，绑定最终程序。
- 主产品纵切：CUDA 默认序列中的 FlashInfer、ggml CUDA、Marlin、CCCL 对同一真实 CUDA Environment
  采用共同硬件/Provider/operation/quantization 合取；规则不命中时继续既有成熟 fallback，执行不重新
  匹配。
- 正交反偏置纵切：Apple affine Q4 与 Vulkan NVIDIA/Apple-MoltenVK matvec、ggml Vulkan MXFP4 用同一
  合取语义，但继续由各自 native owner 构造事实、资源和程序。CPU 当前没有手写硬件 allowlist；SIMD
  自己的 ISA dispatch 保持原 owner，不为了形式制造第二次探测。
- 跨模块 public journey 与每段 owner：共同 `RouteRule` 已由 `compute` CLOSED 交付；本卡只在各 native
  owner 接入静态规则。构建因子删除不可用 Provider，Context 固定设备/版本事实，prepare/compile 合取
  workload/operation/format，prepared artifact 执行已绑定程序。
- scope：完整枚举当前 `src/compute_provider*`、`src/compute_route*`、`src/graph_provider*`、
  `src/graph_route*` 中的平台、硬件/架构、驱动/runtime/Toolkit/SDK/API、Provider 版本、模型、phase/
  region、量化 mode/group/物理格式条件；迁移所有真正选择 Provider 或实现变体的手写静态条件；为
  CUDA/Apple/Vulkan 复用各自 Environment 事实，消除同一事实的重复路由表达。
- non-goals：不把 rank/shape/stride、容量/overflow、workspace、occupancy、设备 limit、同步、资源
  accessibility、SDK 原生 `supports_op` 或错误诊断搬进 matcher；不改变 Provider 次序、数学、支持域、
  fallback、模型 loader 或 B2 算子范围；不重写 selector/交付脚本。
- 直接依赖 / 直接消费者：依赖既有 `compute::RouteRule`、CUDA/Apple/Vulkan Environment 和各 Provider
  身份；直接消费者是 CUDA 默认 Context、Apple Direct/Graph lowering、Vulkan Direct/Graph compiler。
- 性能 tier 与原因：`P0`；仅冷 Context/prepare/compile 条件表达迁移，submit/execute 必须零 matcher。
- 适用机器 / 仪器 / material margin：Windows canonical；CUDA/Vulkan 真实测试与三交付，Apple 在 Air
  或 Mini 做 native correctness/交付；公共 matcher 已有 Windows header/correctness。没有热算法变化，
  P1 不适用。
- 调查与纠正预算：60 分钟完成全目录资格分类并冻结有限 FIX；每个迁移点最多两次产品纠正；适用验证
  另计。新发现只在满足“静态事实选择实现”定义时加入，不把所有 `Unimplemented` 变成路由规则。
- 预声明验证：自动文本审计证明受管轴不再在产品中手写比较；每条迁移规则的命中/逐轴拒绝与原 fallback；
  Direct/Graph cold-only 计数；CUDA/Vulkan 实机 correctness；Apple native correctness；受影响公开头、
  source/static/shared、selector、docs/closure/diff。
- stop / blocker 条件：若迁移要求把 shape/资源值塞进通用硬件对象、在 submit 重新解释规则、改变数学或
  Provider 优先级，则停止并重新定界；远端 Apple 不可达或真实 SDK 缺失时按实际门 BLOCKED，不由
  Windows 冒充。

## SCOPED 审计清单

1. 对四组目录做受管轴扫描，并逐项区分静态路由、动态语义验证、资源计算、原生 capability probe。
2. 核对每条手写路由的权威事实来源、Provider/version identity、fallback 和 Direct/Graph 消费链。
3. 冻结有限迁移表；现有 B2 算子接线、自研数学和性能比较继续暂停。
4. 先以现有 matcher 表达；只有具体反例无法表达且属于受管轴时才允许修改共同 API。

当前只完成范围登记，尚未签任何新增迁移或验证 PASS。
