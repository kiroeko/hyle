# D5：Apple 整批三交付与官方 FP4 边界

**Apple A5 三种交付通过。** 每种交付完成10项正常计算与2项预期拒绝检查，共36项。
四个官方 DeepSeek 单元确认当前路线拒绝，未增加支持通过格。
完整 ledger 为 **152 RUN_PASS / 12 BACKEND_REJECTED / 12 VALIDATION_NOT_RUN**。
见 [汇总](summary.json)、[原始文件归档](archive.json)和[文件校验](files.json)。

## 实际执行

候选 `e9afd4a3d84c8eeec7d34f5e86f4602c51ac50e8`，产品字节与 `9273bc71` 相同。
MacBook Air `Mac16,12` / M4、macOS 27.0、AppleClang 21、Release/WERROR、插桩 OFF。
环境、SDK和消费者产物 SHA256 见 [metadata](raw/metadata.json)。

| 交付 | 通过检查 | 本批耗时 | 收据 |
|---|---|---|---|
| 动态 SDK | 12：复用11项，只补跑1项 | 纠正编译9.374秒，补跑1.223秒；首次整批30.461秒 | [shared](raw/receipts/shared-boundary-api/receipt.json) |
| 源码父工程 | 12，零跳过 | 构建83.262秒，测试26.464秒 | [source](raw/receipts/source-boundary-api/receipt.json) |
| 静态 SDK | 12，零跳过 | 消费者构建13.896秒，测试25.404秒；复用源码产物 | [static](raw/receipts/static-boundary-api/receipt.json) |

两入口分别执行真实 Qwen3.5-2B BF16 embedding/norm/head 与非法索引失败、FFN、GDN实际状态回送，
DeepSeek Flash 两份 GGUF 的完整 hash-routing/MoE，以及 GLM 两份官方 FP8 的完整 MoE。
原 shape、oracle、容差及变化输入保留；非本地巨型模型使用格式合法的非零 payload。
这些是交付代表消费者，不重复签整模型或所有完整词表，已通过的完整画像证据保持原范围。

源码/static共用一次默认目标构建；shared复用已生成且逐文件产品身份相同的SDK。
installed消费者只链接公开产品，compile/link命令没有canonical产品或producer的include/lib路径。
运行时隐藏producer bin及其余metallib副本，清除DYLD搜索环境，PATH只保留系统目录。
执行侧metallib与搬迁SDK逐字节一致；DYLD实际加载记录证明shared的全部HYLE/MLX库来自搬迁SDK，
static没有加载HYLE或MLX动态库。每批的runtime-resource、native-linkage及loaded-hyle-libraries日志保留。

## 官方 DeepSeek 精确拒绝

两份原版 MoE 都包含51个节点。Direct成功准备前14个执行节点后，在`GroupedFloat4Linear`
所需权重的公开`input_spec`阶段返回`Unimplemented: Apple input storage encoding is not admitted`；
Graph携带真实格式常量编译完整配方时同样拒绝。source/static/shared行为一致。
这是R11的必需边界，故Pro/Flash × Direct/Graph共4格记`BACKEND_REJECTED`。
其它region没有因本项被自动标通过或拒绝。

[F46原生反例](../apple-fp4-native-boundary/README.md)继续有效：MLX/M4原调用缺少DeepSeek的scale floor，
并在极小解码域清零。本批没有重复探针、改容差、把原版换成GGUF或恢复自研数学。
当前拒绝不证明所有成熟Apple实现都缺失；官方两画像仍是支持缺口，不计作交付或运行通过。

## 原失败与有因纠正

`4f56dc5e`的SDK生成/安装成功，但新增Direct测试把`optional`当指针比较，外部消费者编译失败。
`78e9f243`修正后运行通过11项，剩余Direct测试对拒绝的`input_spec`调用`.value()`而抛异常。
返回AUDITED明确测试合同后，仅将该项改成捕获公开输入规格的具名拒绝；产品FIX为零。
最终只重跑这一项，原11项复用，source/static随后各执行完整12项。
全部初始编译、运行失败与纠正日志均保留，没有把旧失败改写为PASS。

文档/closure、Windows Apple-OFF定向configure和ledger检查记录在 [checks](checks)；
文档归档不触发重编或P1。两个独立consumer临时目录已清理，保留源码producer与有效SDK供A6，
见 [清理记录](cleanup.json)。
统一实现 **60,373行 / 241文件**，不含测试与第三方，见 [LOC](loc.json)。
Vulkan剩余12格及A5、四路线A6和D6未完成；本批不能代表整个goal完成。
