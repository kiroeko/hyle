# C2：ggml 固定升级、CPU 编译与软件资格

<!-- MODULE REVIEW: AUDITED -->

## 计划身份

- 状态：AUDITED，绑定compute_ggml_runtime、compute_provider_native、compute_provider_ggml；
  adapter只纳入升级后的软件身份，格式/算法仍待C3。原SCOPED审计交接，有限FIX如下。
- baseline：`ac9bc40006e1caaa9e42b812f6e5a832e66ee034`；Windows实际HEAD核对，无用户产品修改，不回退。
- owner范围：compute_ggml_runtime、compute_provider_native及ggml adapter的软件版本元数据。软件约束在Native私有默认规则完成，
  无需修改已CLOSED CPU facade；GPU owner和Graph阻塞保持。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- 真实trigger：CPU原版quant/DSpark目标需要新的固定源核对；原锁/资源patch与CPU优化module需版本一致。
- 来源：现锁456172ec733a、候选ffa4e8b80930、llama参考d7a695ef6791；C1精确权重/程序和原CPU ISA资格。
- scope：CPU-only新版固定源/缓存/许可证、OOM/threadpool补丁、私有基线+AVX2程序、SDK/软件事实。
- 不静默升级GPU source/patch/资格，不把不支持新域隐藏为成功，不制造未隔离双ggml版本。
- 普通入口仍cpu/graph::cpu；普通用户只选compute-cpu，复杂依赖和版本策略由构建/Context拥有。

## 已审计的有限 FIX

来源/固定制品/源码审计见[C1证据](../../../../evidence/correctness/cpu-quant-mtp-c1-demand/README.md)。
所有现有资源修改before上下文在新版各出现一次；没有批准新数学或原资源合同撤回。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| U1 | 核对两个固定版本CPU source/API/tensor/plan/CPU capability及现有资源patch，制定升级实施边界 | 一SDK一个兼容core，原数学保留、错误可恢复、CPU程序/原库版本匹配 |
| U2 | 优先评估CPU-only新锁/含GPU原锁的构建envelope方案；合法修改有限lock/patch/CMake/profile输入 | 对实际SDK能力诚实，既有GPU构建不被撤回；异常ABI问题有明确替代方案再实施 |
| U3 | 旧Native优选绑定原成熟peer软件身份；变化域成熟默认，submit不增加查询 | 不拿旧WIN压新版ggml，不运行自研竞赛，原独有资源/整域和MatMul退役保留 |
| U4 | 编译/部署固定CPU基线与优化整程序，source/static/shared、搬移/offline、失败/close | CPU真实quant+普通Direct/Graph已实跑，DLL邻exe；未验GPU版本不签资格 |

### 固定实现决定

- CPU-only选择stock ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`；archive SHA
  `2adae8335f09b75d05158ee32104f0b562929b8470e6d5858ecb513d325636f4`，expanded tree
  `ac5706fbc134d25b54b083ebfa96e1a4656827e0ab363dab6ca3b50b16ddec6a`，MIT。
- 借入Unsloth `46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f` 的3个原推理类型、blocks/grids、
  dequantize、原CPU dot、type traits及row validation；不加入编码器或修改数学。
  合并资源patch后11个具名文件/32处有限接缝；CPU generic原dot在baseline/AVX2完整程序中均可用。
- 新dependency/patch在runtime owning CMake登记；源cache不可变，只有build-local exact patch副本。
  包含任何CUDA/Apple/Vulkan路线时仍取原456172与原patch；一个SDK一个core，不传跨版本tensor/pool。
- CPU完整程序文件名带所选core身份，baseline/AVX2同源同patch；loader的原ABI/size检查保留。
  原版tensor/cplan/graph由对应SDK编译，不跨旧/新SDK混用。
- Native旧WIN的环境资格加私有编译期peer版本条件；新版默认使用成熟实现，专家Native入口不因此撤回。
  新约束不在submit查询、不重跑187项自研比较、不改变SIMD。
- U1软件事实补项：adapter仍硬编码`456172ec+hyle-resource-v1`；CPU-only新包必须从已选core/patch
  投影真实版本到ProviderIdentity，含GPU原envelope保留原CPU标签。仅改metadata字符串/构建投影，
  不扩算子；源码反例已确认，产品改动前将adapter合法绑定本AUDITED卡。

### API、资源与验证冻结

自然入口仍`compute::cpu::create_context()`与`graph::cpu::create_context()`；用户无需选ggml版本。
公开对象/张量生命周期不变；prepare创建原pool/plan，slot拥有work buffer，close等待原执行完成后释放。
OOM/threadpool启动失败保留Status边界，原库数学故障不被catch吞掉。C2不改压缩格式公开枚举，交接C3。

适用验证：dependency/source-patch/package机器合同；runtime与ggml/native直接消费者suite；
Windows/Linux x86_64 baseline/AVX2、普通Direct/Graph、source/static/shared及搬移/offline/DLL邻置。
复用同一CPU producer，C2–C4接线稳定后集中签最终delivery；C2先运行最小原压缩MatMul接缝，
未运行的后续完整格标未验。P0核对编译期envelope、冷load、同源program和无热搜索；无适用P1。
发布头/API未改，按selector实际扩展适用gate；不改公共治理工具。

## 验证、预算与停止

- 一轮源码/ABI/patch审计，一次批量实现，最多一批具名纠正；不因一个失败换多版ggml寻找PASS。
- 优先复用producer/cache；升级producer一次低并行构建、重型串行，旧冻结SDK不覆盖。
- P0：ISA/软件事实冷绑定、repeat submit无probe/编译/搜索；不跑通用算子P1。
- 若共享ABI/符号/构建方案无法保持现有能力，停止此接缝并说明具体问题，不能静默改变其它路线状态。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。

## 当前实施与验证（整包仍未收口）

- U1–U3产品实现已提交：CPU-only新版锁/原IQ1推理port、GPU原envelope、同源优化程序名、
  真实Provider软件标签以及Native peer版本条件。没有修改自研数学、SIMD或GPU状态。
- Windows shared producer批量9/9测试目标通过，涵盖CPU Direct/Graph及资源/故障。
  新IQ1_XS/XXS/XXXS MatMul/GetRows、changed replay与配额退还的最终定向5/5目标通过；
  软件metadata变化后普通CPU绑定与四个普通Graph replay/边界案例亦通过。两个阶段的准确候选与复用边界见
  [C2证据](../../../../evidence/correctness/cpu-quant-mtp-c2-upgrade/README.md)。
- 新测试最初误将原库Q8_K activation与未量化F32比较；失败记录保留，改用原CPU encoder后的
  decoded-weight独立oracle通过。没有修改kernel或用量化级别的宽容差掩盖错误。
- U4 Linux及完整source/static/shared、搬移/offline仍未运行，不签PASS；随C2–C4稳定批次集中完成。
  C3公开格式/non-GGUF接线和C4新分支尚未开始，不能用上述内部原库测试代签。
- C1临时ATen probe已在核对CMake owner/活动进程/目录边界后删除，释放289,707,808 bytes。
  保留复用的CPU producer与源码cache，不另开逐格式构建树。

当前owner维持AUDITED；不得因Windows阶段结果跳为CLOSED。下一步继续有限批次接线与剩余适用资格。

统一29路径统计：Compute＋Graph实现61,110行/249文件（不含测试/vendor/bench）；完整收据在C2证据loc.json。
