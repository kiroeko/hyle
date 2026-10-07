# 接对 F32 fused 后的 CPU SDPA 去留

- 产品候选 `10c7d784d62d6e656b60e533e0f9329a9cb55827`；benchmark `d96a30c6501573f4ad99116eea43a30c51556e5e`。
- [当前施工卡](../../../docs/implementation/tasks/compute/cpu-three-operator-route-closure.md)，
  [四格CI和逐stratum结果](receipt.json)，[原始manifest/observations/controls/validators/实际exe](qualified.zip)。
- W1-WIN、i9-13900K、MSVC19.50、Release、installed static、LTO off、单P核CPU2、above-normal、每库1线程。
  普通无限预算，完全相同F32输入/output/layout/scale/causal，完整submit+wait及原库必要mask/资源成本。
  没把KV转F16，没用更弱精度；准备在steady-state计时外，无模型或所有硬件SOTA声明。
- fresh同实现AA覆盖两执行/两分配顺序，4/4通过；60 product/30 controls、每stratum15独立进程，唯一431.187秒。
  AA/delta5/delta10/timer-floor、artifact/build/SDK/workload-AA validators全部PASS，manifest verify通过，**QUALIFIED**。
  Native四格方向LOSS/practical LOSE/NI FAIL，聚合performance FAIL是实际劣势，不是正确性或仪器失败。

| 子域（BSHD，B1，mask none） | Native/成熟吞吐，四strata中位范围 | 成熟完整调用快多少 |
|---|---:|---:|
| H4/D64/KV128/Q1，非causal，scale .125 | .08917–.08958 | 11.16–11.21倍 |
| 同上，Q32 | .08601–.08765 | 11.41–11.63倍 |
| H8/D256/KV128/Q1，causal，scale .0625 | .21259–.21765 | 4.59–4.70倍 |
| 同上，Q32 | .20507–.21056 | 4.75–4.88倍 |

CI固定在预声明5%门外，全部四个执行/分配strata方向一致。原[当前组合接法复比](../cpu-three-operator-current/README.md)
的D256/Q1约2.17倍Native WIN仍作为历史事实保留，但相对接对的原F32融合实现不再成立。
这次反例是成熟Provider接线遗漏，不能把旧WIN签为数学核优势。

机制：Native单query循环、分块score/softmax和值积累；原ggml CPU F32 fused把这些交给成熟完整算子，
省掉原接法的scores/softmax/V转置物化，且保留本身的tiling/向量化/线程计划。
本批没有修改Native数学、原库数学、统计、阈值、机器电源或用户进程，没有n31或重复竞速。

裁决：在[用户授权的有限N3](../../../docs/implementation/tasks/compute/cpu-three-operator-route-closure.md)下，
整项删除自研SDPA；Native-only/caller-workspace/零额外private及其非dense/mask执行档明确撤回，
普通CPU的SDPA schema、ggml/ATen GQA/mask/布局数学继续。SIMD、Mem与其它Native算子保留。
删除后的最终correctness/交付证据由本轮terminal记录，不能用这张性能记录代替。
