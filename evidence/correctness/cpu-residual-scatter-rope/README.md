# CPU 剩余数值域：ScatterAdd 与整数 RoPE

Windows canonical，产品 candidate f32d261a。这是进入有限性能比较前的数值诊断；没有跑 P1，
不能把完整 witness 的预期非零 exit 或未执行行写成 PASS。330 条记录、32 次拒绝，Native 与
普通成熟各 3 行不符合参考；1337 个产品/测试/构建输入与 ND-R1 身份匹配。

| 真实反例 | Native | 普通默认成熟 |
|---|---|---|
| F16 ScatterAdd，base=1，32 个重复 index=0，每项 1/2048 | 1；一次存储舍入参考为 1.015625 | ATen 1.015625 |
| BF16 同消费，每项 1/512 | 1；参考 1.0625 | ATen 1.0625 |
| I32 RoPE，位置 ±1073741827，F32 频率 0.125/0.25 | F32 8 个输出不符合保留整数位置的 Double phase 参考 | ggml 的 F32/F16/BF16 分别 8/8/7 个输出不符合 |

ScatterAdd 的公开注释明确不承诺逐更新 storage rounding；旧 Native 测试
HalfScatterUsesOrderedRoundingAndChecksBeforeWriting 恰好验证了旧的错误预期，不约束当前合同。
Native 源码每项 numeric::store<Value> 是原因，原成熟 ATen 已在 AS 批接入 F32 累加后一次转换。

RoPE 的 Native load_position 后转 Float；ggml 接缝 explicit_rope 在原 sin/cos 前把 I32 positions
cast 到 F32。位置 +1073741827 转为 F32 变成 +1073741824，数学发生在已变的输入上。原库 kernel
没有故障；是 HYLE 接入程序的中间精度和公共整数端口不一致。旧 ggml 测试先计算 F32 phase，
无法证明整数位置未丢失。本轮沿 TM2 已采用的整数→Double phase 程序统一通用 RoPE，不把
Float phase 与 Double phase 的计时签成同合同性能胜负；原模型的独立程序/数值资格不由此外推。

ATen 的 F32/I32-I64 RoPE 已在 TM 批实际资格化，原程序 positions/frequency/data.to(Double)、
sin/cos/stack/cat 与输入 clone 保尾；低精度输入尚被准入 guard 挡住，需要有限接入及实际资格。
不能先撤掉 ggml 的 F16/BF16 便称没有能力损失，也不能把 Native 资源唯一当保留错误数学的理由。

完整源输入、独立 reference、所有 compile/执行原件及 driver 在 [diagnostic.zip](diagnostic.zip)，
摘要/hash 见 [receipt.json](receipt.json)。第二次诊断仅补 F16/BF16 RoPE 和更正 I32 日志标签；
首次原件保留，没有调整数据、数值阈值或优化算法。

## 同合同审计纠正（当前裁决）

上面的330条与raw bad_rows是相对于Double oracle的原件，不改写。产品错误已证仅Native的两个
half ScatterAdd域。RoPE一项在不同phase精度之间比较，当前改判NOT_COMPARABLE，不能据它签
原Float32模型程序FAIL或删除成熟RoPE。官方Qwen3.5 text rotary明确使用Float32 phase，见
[官方源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
RN2/RN3方案与未提交产品改动撤回；原ggml/Native I32 RoPE保持。低精度data/I64端口的ATen原Double
程序接入仍有意义，作为具名数据域资格，不冒充模型Float32程序或签二者同合同性能胜负。
