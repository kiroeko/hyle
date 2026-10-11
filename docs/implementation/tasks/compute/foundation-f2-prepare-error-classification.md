# F2终审：准备阶段硬错误与资源不匹配的分类

状态：CANDIDATE；owner仅compute；父轮[F2七FIX](foundation-f2-batched-transfers.md)。
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

## 固定实现与当前验证

产品修复c26261e39只改operation.cc的三个纯resource predicate分类与ordinary hard-error传播，
测试维护新增input/output/workspace mismatch必须在first provider prepare之前decline的真实
public prepare旅程。registry另恢复原LF，语义diff只有compute的合法AUDITED退回，不改其他owner。
[原扩展门收据](../../../../evidence/correctness/foundation-f2-selector-expansion/receipt.json)保留完整build PASS、compute PASS、
CPU backend两项实际FAIL以及首scratch selector组合拒绝。

[delta收据](../../../../evidence/correctness/foundation-f2-prepare-error/receipt.json)当前为PARTIAL：Windows完整CPU backend58项与compute90项PASS，
原hard error/partial artifact回收、合法decline及新增三个resource predicate分支均实际通过。
完整Graph原合并进程120秒TIMEOUT，停止前84项OK，不写整门PASS。按同一二进制/完整853项
实际inventory固定16项一组、每进程仍120秒、重型串行，所有54组都运行且保留XML/日志；没有
删掉慢用例、扩大单进程deadline、改公共工具或重置产品名额。当前分组资格仍运行，其他平台/
ordinary SDK及完整剩余影响门继续未验；compute保持AUDITED，不提前CANDIDATE/CLOSED。

16项分组的第9组也因累计时间120秒TIMEOUT：其中首项20.8秒、Dspark65.7秒与下一项7.8秒
均已OK，后续用例未完成；没有新增产品FAIL。前8组XML逐项与请求集合完全一致，128项PASS。
盘点发现list-tests参数注释前的尾空格须trim，否则后续过滤可能遗漏；尚未为这些未运行名称
签PASS。固定同二进制/root从第129项逐case恢复，每进程仍120秒，XML必须与请求名称精确一致
且无disabled/failure。原合并/分组两个TIMEOUT继续保留，不重复已资格化128项或重置产品预算。
完整853项仍未验完，不将该fixture恢复写成整个Graph门PASS。

## 新candidate

产品c26261e3958c7eb46ecde2a4ba7c5654a03f1b93在上述固定单项FIX后停止变化，compute重新
进入CANDIDATE，改前registry ref为c74f923a2。Windows完整backend58项、compute90项与Graph
全部853项PASS，实际用例集合精确且无skipped/disabled；原两个TIMEOUT及名称trim修正保留。
同一二进制、同root恢复完成，完整XML/log压缩包与逐文件SHA列在[收据](../../../../evidence/correctness/foundation-f2-prepare-error/receipt.json)。
没有新API/layout/常态成本/数学或copy kernel。CUDA仍绑定5b3efe2cc，Apple/Vulkan仍绑定03162add0，
其owner source scope未变。Air与WSL验证副本已从完整canonical Git archive更新到c26261e39，
tree185e3c211a65754f5487e7e1d19f8252e68f29d9和archive SHA保存；其他平台/ordinary SDK与剩余
扩展/终审继续待验，本卡不是CLOSED，不提前F3或把候选固定当作全部工程验收。
