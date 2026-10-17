# CPU准确页只读历史P1

产品candidate `1e9cdfd45307ea4a15b92374377ad2b268203193`，clean source `4484e3f43ced16781cf04b4d7567e2820e9352e6`，
contract `8d033b8207a8afce7b662841e0dd2a4ce85b24b64d8a1d7ae6dab26f80ddc0b7`；实际W1-WIN i9-13900K/MSVC19.50/CPU2，
原版ffa CPU AVX2 DLL的实际compute image与POD ABI核验。
[收据](receipt.json)、[summary](summary.json)、[manifest](manifest.json)、[raw](raw.zip)、
[verify](verify-receipt.json)保固定身份/实际命令/原样本/失败/累计，不改阈值或仪器。

资格 `DIAGNOSTIC`、性能 `UNPROVEN`、verify退出
`0`；16AA、120产品/30控制，阶段 `[15]`，
累计1695.11s。三组BF16/GQA8/2、反向页表/共享页/tail/两slots，
同原Batch/metadata update/exact F1 pages/CPU Queue/Completion/output lease边界。

| 点 | candidate/peer吞吐比 | 5%非劣 / band | 相对1方向 |
|---|---|---|---|
| cpu-paged-mixed-d256 | 0.9797–0.9868 | PASS / PARITY | LOSS |
| cpu-paged-prefill-d256 | 0.9815–0.9865 | PASS / PARITY | LOSS |
| cpu-paged-holdout-d64 | 0.8094–0.8170 | FAIL / LOSE | LOSS |

对照只适用已有不可变ready历史：原确切页owners全程保活，冷复制valid logical rows
到单独contiguous cache，额外D256 3670016bytes/D64 39936bytes已单列。原CPU
plan/compute/pool从未打HYLE补丁的DLL；无candidate compile/lower或复制数学。
原double8e-3、readonly、16paired/4ms/min1/n15/31/双顺序/placement/controls均保。
private池上界UNKNOWN；不签cache producer/growth/reorder生产、capture、GPU/G02、
全模型token/s或整个G01。见[消费形状](../../../bench/inference_resident_sdpa/cpu_paged_p1/consumer-provenance.md)
与[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-cpu-paged-resident.md)。
