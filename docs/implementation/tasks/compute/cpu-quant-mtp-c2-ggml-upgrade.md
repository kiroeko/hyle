# C2：ggml 固定升级、CPU 编译与软件资格

## 计划身份

- 状态：PLANNED，未激活、未绑定registry；没有SCOPED/AUDITED产品授权或执行PASS。
- baseline：`5b0044e2a10d11fdfce54893020cfcaaed47b0c0`；激活时核对Windows实际HEAD/worktree，保留后续及用户修改。
- owner范围：compute_ggml_runtime；旧规则peer版本隔离涉及compute_provider_native、compute_route_cpu、graph_route_cpu；公共构建接缝按已有owner合同绑定。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- 真实trigger：CPU原版quant/DSpark目标需要新的固定源核对；原锁/资源patch与CPU优化module需版本一致。
- 来源：现锁456172ec733a、候选ffa4e8b80930、llama参考d7a695ef6791；C1精确权重/程序和原CPU ISA资格。
- scope：CPU-only新版固定源/缓存/许可证、OOM/threadpool补丁、私有基线+AVX2程序、SDK/软件事实。
- 不静默升级GPU source/patch/资格，不把不支持新域隐藏为成功，不制造未隔离双ggml版本。
- 普通入口仍cpu/graph::cpu；普通用户只选compute-cpu，复杂依赖和版本策略由构建/Context拥有。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| U1 | 核对两个固定版本CPU source/API/tensor/plan/CPU capability及现有资源patch，制定升级实施边界 | 一SDK一个兼容core，原数学保留、错误可恢复、CPU程序/原库版本匹配 |
| U2 | 优先评估CPU-only新锁/含GPU原锁的构建envelope方案；合法修改有限lock/patch/CMake/profile输入 | 对实际SDK能力诚实，既有GPU构建不被撤回；异常ABI问题有明确替代方案再实施 |
| U3 | 旧Native优选绑定原成熟peer软件身份；变化域成熟默认，submit不增加查询 | 不拿旧WIN压新版ggml，不运行自研竞赛，原独有资源/整域和MatMul退役保留 |
| U4 | 编译/部署固定CPU基线与优化整程序，source/static/shared、搬移/offline、失败/close | CPU真实quant+普通Direct/Graph已实跑，DLL邻exe；未验GPU版本不签资格 |

## 验证、预算与停止

- 一轮源码/ABI/patch审计，一次批量实现，最多一批具名纠正；不因一个失败换多版ggml寻找PASS。
- 优先复用producer/cache；升级producer一次低并行构建、重型串行，旧冻结SDK不覆盖。
- P0：ISA/软件事实冷绑定、repeat submit无probe/编译/搜索；不跑通用算子P1。
- 若共享ABI/符号/构建方案无法保持现有能力，停止此接缝并说明具体问题，不能静默改变其它路线状态。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。
