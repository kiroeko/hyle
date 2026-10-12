# F5 Apple/Vulkan 显式reference原生冷适配

## Closure card

- 状态：AUDITED；原F5审计阶段保留，本卡补齐原生Direct有限冷适配冻结。
- baseline：aa4fc569e。compute_route_apple/vulkan先前是F5 package-only AUDITED，
  本trigger扩大为实际reference Direct冷适配；原package发现仍沿
  [package卡](foundation-f5-package-seam.md)完成，旧R1–R7/F1–F4数学/资源协议不改。
- trigger：F5已准入四路线reference；共同MaterializedBatchProgram/Version和原Graph构造
  已实现且CPU实际通过，普通native Direct消费者不应手写逐operation生命周期链。
- consumer：同一个typed固定recipe、same-layout indices/mask经原F2上传，原native独立
  Program/Invocation/readonly result；paged/ragged、GQA、右对齐causal和明确字节预算。
  正交为同分段矩阵/资源批次，接口不含token/模型/页淘汰策略。
- owners：原compute_route_apple/vulkan只冷转换common recipe到已有native Program；
  compute_batch保typed容量/版本/成本；graph_batch保普通Graph构造。原native runtime、
  leased input与结果/retirement/关闭继续原route owner，不新增共同Queue或第二executor。
- 有限FIX：REF-APPLE-01 prepare_materialized_batch返回原MLX Program，cold tracing内
  只调用原apple::lower；REF-VULKAN-01同名函数转换为原detail::Plan并prepare_plan。
  两route增加compute_batch顺向依赖与独立新头apple_batch.h/vulkan_batch.h；无反向Graph边。
- 非目标：新kernel/math/SDPA mask解禁、复制原native执行器、自动fallback、warm recipe
  遍历/JIT/prepare、普通Program/Invocation字段变化、升级锁、公共工具重写。
- 生命周期：cold callback拥有common program副本；原native编译后不retain callback。
  Program只保原native artifact，输入顺序Q/K/V/I32 indices/F32 mask；版本更新仅原native
  idle bind重新绑定同capacity metadata，不重compile。准备失败不返回partial Program。
- 性能：P0，显式reference档；common byte caps和gather/GQA/F32/Copy逻辑上界保留，
  原native scratch/driver UNKNOWN不写0。MLX managed/native Vk命令分别报告，未签capture。
- 预算/机器：沿[F5主卡](foundation-f5-paged-ragged-batch.md)累计configure300/build1800/
  consumer180/test120，不重置原失败。实际Win Vulkan、身份核对后的Air Apple/Vulkan；
  库j4≥12GiB/j2≥8GiB/j1≥4GiB、test1，所有机器重型串行、缓存/tree身份匹配。
- 预声明：Direct/Graph同recipe各F16/BF16 paged/ragged causal/noncausal独立oracle，乱页/
  非整尾/NaN padding、同版本F2上传、更新/双slots/旧result、foreign资源/alias/producer/
  OOM/错误/取消/unknown/close及append/F4闭环；source/static/shared/新头/缺路线/迁址/
  loader/san/P0/guide/终审仍沿主卡。此冷适配首批不能代签全部MUST。
- stop：原route数学不能承载则明确拒绝/记录实际缺席，不补新kernel；实际owner/接口
  新事实退明确阶段。未完成适用项保持未验，不能以CPU/native CUDA证据代签。

## 能力与承载审计

| 能力 | 原承载 | 动作 |
|---|---|---|
| materialize数学与版本 | compute_batch固定recipe/owned F1 metadata | KEEP，不再写数学或页表验证 |
| Apple native Direct整段 | 原Context::compile、apple::lower和MLX compiled region | BORROW冷tracing adapter；原Program/slots保结果 |
| Vulkan native Direct整段 | 原detail::Plan/prepare_plan、ggml/IREE/native resources | BORROW固定冷Plan；原Program记录与独立slots |
| Graph reference | graph_batch::materialized_graph加原native Graph compiler | KEEP，不改旧Graph路线源码 |
| metadata上传/输入权限/退休 | F1/F2和原native leased wrappers | KEEP，实际消费取证，不以host可见替代GPU ready |

API草图固定为各native namespace的prepare_materialized_batch(context,program,workload={})。
source/installed均由自然Compute产品发现新route头；route缺席时头不提供该route符号。
七问沿锁定MLX MIT及ggml/IREE原许可、双纵切、原F1/F2/Program复用、显式成本P0、
同工具链C++/无新FFI、三交付与独立头、具名guide/误用风险。两FIX与API/依赖/成本冻结，
原owner继续AUDITED，尚未产品实现；不把已有AUDITED owner写回SCOPED或伪造BLOCKED。
