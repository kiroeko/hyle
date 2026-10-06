# C3：CPU 格式与普通算子批量接线

## 计划身份

- 状态：PLANNED，未激活、未绑定registry；没有SCOPED/AUDITED产品授权或执行PASS。
- baseline：`5b0044e2a10d11fdfce54893020cfcaaed47b0c0`；激活时核对Windows实际HEAD/worktree，保留后续及用户修改。
- owner范围：compute_ops、compute_provider_ggml；C1准入的ATen/借用原程序由compute_provider_native承接，CPU facade只在真实缺口时按owner纳入。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- scope：Q01–Q11公开block身份/描述/原字节import，现有MatMul、GroupedMatMul、Gather/GetRows在真实端口的成熟接线。
- C1若准入NF4/AWQ/GPTQ/MLX制品，添加其具名packing/成熟carrier；没有审计不能猜成GGUF4-bit。
- 不扩全量ggml类型目录、不自研decoder/GEMM、不以整个文件的Q4名替代tensor编码；按原block/role使用。
- 正常consumer是混合profile projection/expert/embedding；正交普通压缩数值矩阵/索引处理；无需模型loader。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| Q1 | 一份语义format映射和checked描述器批量扩展11格式，不自动强造所有格式×端口组合 | block bytes/alignment/rank/group/shape与原码相符，unknown/越界明确错误 |
| Q2 | 原ggml CPU import+原MatMul/MUL_MAT_ID/GetRows，共用program/参数化fixture | 直接执行压缩字节，完整精度/duplicate expert/checked index合同，必要转换冷处理 |
| Q3 | 已准入非GGUFcarrier复用原代码/成熟组合，遵守codebook/zero-point/permutation等 | 原tensor角色与排除项保留，prepare/replay资源/失败/并发真实成立 |
| Q4 | 普通Direct/Graph自然绑定、完整region/资源复用/重复执行 | 不需用户排名，热路径不编译或展开全库权重，失败不被不当fallback掩盖 |

## 验证、预算与停止

- 全批一次接线、参数化oracle与归并signature，最多一批定向纠正；不每个格式开一棵build树。
- Factor tests/header/普通入口/资源/OOM/两系统三交付按stable candidate集中执行；适用风险未运行写未验。
- 数值对照采用同编码/累加/输出和边界的原成熟程序，非同合同只记录差异。
- P0为主；若合格CPU库拒绝必需格式/端口，具名记录，不自写替代、不将拒绝算完成。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。
