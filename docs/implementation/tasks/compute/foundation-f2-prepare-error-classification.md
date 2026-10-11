# F2终审：准备阶段硬错误与资源不匹配的分类

状态：AUDITED；owner仅compute；父轮[F2七FIX](foundation-f2-batched-transfers.md)。
baseline/改前registry ref为4f3669630，产品snapshot为5b3efe2cc；原compute候选03162add0
被本次实际ordinary消费者错误反证，依法退回AUDITED。CUDA/Apple/Vulkan owner保持原候选。

## 真实反例与有限审计

Windows/MSVC19.50原CPU树完整构建PASS，compute factor PASS；compute_backend_cpu的既有
CustomKernel.PreparationDeclinesFallbackButFatalFailuresStopBinding及
RuntimeProviders.DirectErrorCodesDoNotBecomeSafeDeclines FAIL。三个CpuCopyProgram仍PASS。
provider返回明确kError/Unimplemented后，ordinary prepare却继续准备fallback；首fixture随后
访问成功Result的error产生SEH，该二次现象不掩盖已观察到的错误fallback。

src/compute/operation.cc的prepare_candidate已经分别表示显式decline与hard error，公共
ProviderPrepareOutcome也明确任意Error（包括Unimplemented/OOM）不得重试。
ordinary外层却将任意unexpected Unimplemented当作resource miss。该分支来源acdc6eda4b，
早于F2，相关源码/测试从F2 baseline到5b3efe2cc未变；现有CANDIDATE仍必须处理真实反例，
不能以历史CLOSED或F2新copy测试PASS忽略普通数学准备错误。

本次只读核对operation.cc的resource match/preparation/ordinary selection和已有两个用例。
provider-neutral match_requirement在native/provider prepare之前仅检查真实resource property/
identity/domain，不匹配没有接纳work；它与provider明确kError必须保留不同的分类。

## 单项FIX与形状冻结

- 已有公共合同不变：只有显式qualified decline可选择另一候选；partial artifact仍先由原owner
  销毁；hard error原Status传播，fallback prepare不得运行。
- 在三个原input/output/workspace resource-match边界，将纯predicate的不匹配表示为已有
  ProviderPrepareAttempt::kUnsupported；坏resource/其他错误继续传播。
- 删除ordinary层按任意Unimplemented错误码猜safe decline的分支；不改provider API、ABI、
  layout、热submit、正常路由优选、native runtime或数学kernel。
- 不重开CLOSED Provider/Graph/F1；它们只作为直接消费者回归。所有原失败和F2预算保留，
  不以本记录重置名额。一个固定cold分类修复，不顺手扩建路由或公共验证工具。

## 验证与停止

沿原已构建CPU树build1/test1/4GiB、build1800/test120，先运行两个既有失败反例及全部CPU
backend用例、共同控制/实际copy；必须同时保留hard error与原合法decline/partial artifact回收。
核对原resource-specific miss的input/output/workspace回退旅程；若已有用例不覆盖则仅补该
测试维护。selector从明确baseline选择compute及真实直接消费者，完整扩展门继续保持未验。
固定delta后按source scope精准复验四路线ordinary搬运后Direct/Graph数学及适用SDK消费者，
旧copy/native/头/无变化delivery的证据只按原合同匹配，不把未运行写PASS。

性能P0：只改变冷prepare的错误分类，没有新常态allocation、字段、锁、copy或native command。
若还需新公共语义/owner或出现新material矛盾，停止并按父卡裁决；本卡不是terminal。
