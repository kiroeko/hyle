# F45：Apple FP8 与两个官方 GLM 完整画像

测试候选 `3f7a0eb55491d4ec65b691fe98e79d63acb79dd8`；产品固定在
`b0d724acce7b74d10fc1b51d8387a0c21f8b83eb`。MacBook Air Mac16,12 / M4，
MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`，Windows canonical bundle 同步。

**Direct 31/31、Graph 26/26、零失败/跳过**。新增 GLM-5.3 与 GLM-5.3-Flash
官方 FP8 的 Apple Direct/Graph **4 格**；ledger **152/176**，Apple **40/44**。
这是模型真实形状与精度角色的片段正确性，未宣称巨型模型整机推理或性能通过。

## 实现与普通入口

同一个 Compute 冷 lowering 供普通 Direct/Graph 使用；使用 MLX 原生 FP8 编解码、
分组 ReduceMax、分块 MatMul、Take 和 scale 运算。没有新增 HYLE 数学 kernel。

| 能力/region | 已验范围 |
|---|---|
| FP8 基础 | 预编码 MatMul、动态 block128、plain/power2 scale、BF16/F32、N 尾块、group64/128 roundtrip |
| 专家 | GLM E256/288、top8、packed gate/up、逐专家 down、共享 FFN、完整 router→experts→combine |
| R01 | 复用同机器完整词表/Gather；本批实际 NextN hidden 接154880行全词表 head |
| R02–R03 | 官方 dense FFN、逐角色 FP8；Flash KDA 的 BF16 投影复用已验同一程序 |
| R06 | 复用 Flash 完整 KDA、实际卷积/recurrent 状态反馈 |
| R07–R09 | MLA 四轮 cache；DSA/pooled indexer、selection、0..3尾部、索引 carry/reset |
| R10–R12 | 完整 MoE、Flash clamp/shared、真实 mHC/Sinkhorn/残差反馈 |
| R13 | EH→MLA→残差/norm→MoE→最终 norm→全词表，实际 hidden/cache 回送 |
| R14–R15 | 实际 FP8 kv_b 冷解码准备、严格 carrier/ID、失败恢复、结果保活、关闭/排空 |

GLM 非 Flash 的单一 FP8 `[28672,512]` kv_b 通过路线实际解码成 BF16，
View/Copy 得到 key `[64,512,192]` 与 value `[64,256,512]`；逐值与独立解码比较，
冷阶段 owner 销毁后，热 MLA/NextN 仍消费实际准备结果。未上传 host oracle 代替缓存。
完整矩阵宽度与逐角色精度保留，非本地大模型使用合法非零合成 payload。

逐 region 来源及复用范围见[收据](receipt.json)。旧 R01/KDA 的模型、shape、原生程序、
Provider pin 与 M4 环境一致；本轮新增 FP8 冷匹配不替换这些既有程序。

## 三项数值失败如何裁决

原批次 Direct 20/23、Graph 13/16。初始乘积先合并 scale / 树形求和，
纠正为原版逐块顺序后，Flash 完整测试的超差点18→1；所有原始失败保留。

剩余原子近零结果不能用一个 BF16 ULP 度量中间 F32 误差：FP8 codes/scales 独立诊断
一致，实际值1.862645149e-9，而 FMA 参考2.677552402e-9。现在按输入推导
`gamma(3)*sum(abs(term))` 的独立 F32 区间，再执行 BF16 舍入。没有添加固定绝对放行值。

两项完整失败均定位到 MLA 的后两轮。一次[中间值捕获](capture.tar.gz)显示，
两模型四轮 **32 次 FP8 投影全部逐值等于独立 unfused F32 参考**，
16次 head/KV 投影也逐值一致；局部 norm/attention 最大差≤2.3841858e-7。
旧参考固定 CUDA FMA，低位差跨过 BF16 中点，再改变后继 FP8 codes并随状态传播。

[独立全链预测](chain-prediction.json)从原始输入出发，自己演化状态，仅将乘加改为
两次 F32 舍入，48个输出/重放边界均通过原门；没有拿设备中间值生成期望轨迹。
最终测试在生成整个 GLM fixture 前固定该原生舍入方式，CUDA 保留原 FMA 参考。
**完整链仍是原 `abs2e-5 + rel0.025` 门**；没有逐元素挑选参考或扩宽容差。
原子区间检查同时在 CUDA 两入口通过，最终 CUDA 默认 GLM 参考回归10组通过。

保留[局部分析](analysis.json)、全部初始/纠正/最终日志与 XML、源文件及哈希。
重现数值分析：将捕获包解压到本目录 `arrays/`，依次运行 `python analyze.py` 和
`python predict.py`（依赖 NumPy）；生成的 reproduced JSON 不覆盖原证据。
临时捕获 instrumentation 已从常态测试移除。

## 验证、成本与未完成项

Apple 两组头自包含检查通过；CUDA 仅跑受影响的五个 GLM 测试/入口；输入工具13项、
closure、docs 通过。selector 的所选门与 skipped reasons 保留；完整 A5 配置/三交付批次、
A6 性能和 D6 未由本收据代签。

MLX 保有非零 F32 解码、分块乘积、scale 广播及原生临时资源；grouped 只解码所选专家。
MLX 管理常量折叠与融合，HYLE 未增加私有权重缓存或热期搜索。FP4 roundtrip 极小 scale
仍不合格，不能用本批 FP8 代签剩余 Apple 官方 FP4 画像。

统一[实现统计](loc.json)：**60,373行／241文件**，不含测试与第三方。
剩余24格、三交付、适用性能和 D6 继续保持未完成；本批复用现有构建树，重型验证串行。
