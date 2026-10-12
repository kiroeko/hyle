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
- 生命周期：Apple原MLX compiled function保immutable tracing callback及有限recipe的value
  copy，不借common program/schema/allocator到native artifact；这部分native metadata追加
  于common logical上界，原opaque bytes仍UNKNOWN。固定InputSpec只沿已冷准备trace replay，
  不warm遍历。Vulkan丢弃冷Plan，只保原
  native artifact。输入顺序Q/K/V/I32 indices/F32 mask；版本更新仅原native
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

## AUDITED 首批实际原生消费

[原生reference证据](../../../../evidence/correctness/foundation-f5-native-reference-first/README.md)
已落地两冷适配、新route头及compute_batch顺向依赖：actual Win RTX4070 Vulkan、Air
Mac16,12 Apple M4/MLX和M4/MoltenVK，各八域、Direct/Graph初始/第二slot/metadata更新
三episode，共48次独立oracle。NaN padding、乱页/尾/GQA、原native旧result覆写拒绝与
同capacity不重compile验证通过；common injected schema/recipe/version metadata quota0。

实际host snapshot→原native staging→原F2镜像保持owned version；示例每phase staging与
upload各1920bytes，host waits Apple2/Vulkan1如实报告。新头actual presence宏/依赖路径下
分别独立TU；缺席另一路头可include。Win编译fixture误用Environment/printf类型及头fixture
缺原MSVC alignment warning政策的FAIL都保留；产品数学和公共工具未放宽。

原MLX compiled function确实保tracing callable，因此Apple只value-copy有限typed记录，
不retain common program/schema借用到native artifact；这部分native metadata额外且仍
UNKNOWN，不用common quota0代签driver。Vulkan只冷Plan借原prepare_plan；两者warm执行
沿原Program/leased wrapper、独立slots与Completion，未新造Queue/runtime或kernel。

两FIX有首批实际证据，owner仍AUDITED：同版本batch组合控制器、append/F4完整页消费、
其他故障/producer/unknown/alias/OOM和全部SDK/san/P0/guide/终审继续主卡；不签完整F5。

## AUDITED REF-PAGE：原native精确页绑定与权限提交桥

baseline c9bc4fa69。原MemoryResource::binding(ReadLease,whole TensorDesc)不能接单页
guard，原LeasedInvocation只按输入数预留整块权限；ARENA-03要求真实页footprint，不能
用whole arena ReadLease或clone掩盖。固定以下实际owner桥后才写产品，仍本卡AUDITED。

- REF-PAGE-APPLE/VULKAN：各route private PagedBatchAccess::binding(memory,frame,options)
  只接受不可伪造PagedArenaRead与exact resource，从原continuous capacity构造readonly
  MLX array/Vulkan Tensor；owner保frame/全部actual page guards及原native backing。原
  offset/dense/尺寸/tail/runtime资源规则保留，wrapper metadata按原预算，opaque UNKNOWN。
  公共MemoryResource::binding与普通LeasedInvocation保持原语义/字段/容量。
- 同private桥把调用者已资格的固定F1 LeaseUseBatch交给原Invocation::submit_guarded，
  wait_native只委托原native actual retirement。原Program/slot/OwnedResult/runtime保活，
  不新增队列/执行器/数学、默认输入字段或捕获资格。权限必须覆盖Q、exact K/V pages、
  owned同版本indices/mask；上层controller仅对固定materialize recipe接桥，不能把private
  whole capacity赋给普通任意数学/consumer。完整production controller继续本goal。
- 新private头按原route SPLIT_PUBLIC_HEADERS登记，缺席头可include、不暴露route符号；
  PUBLIC依赖沿已签compute_batch正向边，common不反依赖native route/Graph。
- 验证：actual Win Vulkan/Air Apple/Vulkan原Direct与原native Graph的同一recipe上paged
  exact page guards、无page0引用/首request无Q/NaN空余页/GQA及same-generation，保旧页
  时free writer；逐页原F2 append/F4发布/旧新旧独立oracle；wrong exact resource拒绝，
  原old result/close/escaped owner及quota0。普通四分段八域原48oracle复用同根定向运行。
  改动owner/新头Werror、source/shared实际消费者及后续SDK/san/终审仍原固定预算；不能
  用本private桥首批代签完整controller/F5。原所有FAIL保留，机器重型串行。

[REF-PAGE实际收据](../../../../evidence/correctness/foundation-f5-native-page-reference-first/README.md)：
Win Vulkan、Air M4/MLX与M4/MoltenVK均PASS。每profile新增F16/BF16 Direct/原native Graph
旧新旧12oracle，原48oracle保持；actual native提交时free page4可写，used page2只读，
old IDs2/3→new2/1无page0。clone1024+append512，对比whole K/V arena5120未复制。
原F4 dirty替换/同版本layout/expected-base成功发布、clean share与metadata quota0通过。
新private头三profile实际工具链各独立TU通过；完整SDK/san/终审仍未签。

新fixture错误假定native old result也必须阻止readonly bind的FAIL保留；原合同只保证
输出slot不可覆盖，修正测原submit拒绝，没有扩大产品权限。错误路径在原Context实际
close/drain之后才销毁借用allocator/resource。bridge owner无普通对象字段变化，仍AUDITED。
production same-generation controller、CPU精确页桥/ragged及其他完整F5风险继续本goal。
