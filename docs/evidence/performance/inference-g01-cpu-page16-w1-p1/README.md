# CPU page16地址修正只读历史P1

产品candidate `42f98b94f33bdbb252b9259ece33533cf763fa41`，clean source `f9a689be320a9330b507a274918090ce44a17813`，
contract `3d05ed563439cb0f376c2f3ed1de2ac9c1b2f7e1eb5e9285a81f071e3badd574`；实际W1-WIN i9-13900K/MSVC19.50/CPU2，
原版ffa CPU AVX2 DLL的实际compute image与POD ABI核验。
[收据](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保固定身份/实际命令/原样本/失败/累计，不改阈值或仪器。

资格 `QUALIFIED`、性能 `PASS`、verify退出
`0`；16AA、120产品/30控制，阶段 `[15]`，
累计1657.16s。三组BF16/GQA8/2、反向页表/共享页/tail/两slots，
同原Batch/metadata update/exact F1 pages/CPU Queue/Completion/output lease边界。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| cpu-paged-mixed-d256 | 0.9867–0.9912 | PASS / PARITY | LOSS |
| cpu-paged-prefill-d256 | 0.9882–0.9932 | PASS / PARITY | LOSS |
| cpu-paged-holdout-d64 | 0.9869–1.0084 | PASS / PARITY | INCONCLUSIVE |

对照只适用已有不可变ready历史：原确切页owners全程保活，冷复制valid logical rows
到单独contiguous cache，额外D256 3670016bytes/D64 39936bytes已单列。原CPU
plan/compute/pool从未打HYLE补丁的DLL；无candidate compile/lower或复制数学。
原double8e-3、readonly、16paired/4ms/min1/n15/31/双顺序/placement/controls均保。
private池上界UNKNOWN；不签cache producer/growth/reorder生产、capture、GPU/G02、
全模型token/s或整个G01。见[消费形状](../../../../bench/inference_resident_sdpa/cpu_paged_p1/consumer-provenance.md)
与[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-cpu-paged-resident.md)。
