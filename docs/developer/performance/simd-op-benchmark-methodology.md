# SIMD 操作级测量方法

> 本页是 [通用性能证据标准](benchmark-evidence-standard.md) 的 SIMD 专项补充。通用标准优先；
> 任何旧矩阵和 op 数只属于 [implementation evidence](../../implementation/evidence/performance/README.md)。

SIMD op 的循环常只有几条指令，极易被代码放置、inlining、常量折叠、前端相位、大小核迁移和
timer floor 支配。操作级 benchmark 只用于定位 codegen/算法差异，不能单独改变公共 API 或证明
SIMD 模块整体性能。

## 1. 先定义 op 合同

每个条目登记：

- canonical operation 和公开/internal 入口；
- lane type、vector width/scalable tag、mask/tail、alignment、alias 和 in-place 合同；
- 精度、NaN/Inf/subnormal、饱和/舍入、exceptional lane 语义；
- scalar、native ISA、portable dispatch 与 peer 的具体路径；
- setup、load/store、tail、dispatch 是否计费；
- 正常用户 kernel、调用频率与 material impact budget。

名字相同不代表合同相同。Highway/xsimd 的 precondition、精度或 mask/tail 更弱时用
`NOT_COMPARABLE` 或拆分机制税，不能偷偷省略 HYLE 语义。

## 2. Coverage manifest

公开名字不等于每个都应有纳秒 microbench。manifest 将能力分为：

| 类别 | 证据 |
|---|---|
| native instruction/recipe | 同合同 op parity + codegen |
| alias/retag/construction | correctness + zero-extra-code inspection |
| paired/informational helper | 由 canonical op 证据覆盖，写明映射 |
| algorithm/whole-buffer | 真实 size distribution 与 consumer replay |
| platform unavailable | compile/fallback correctness；等待真机时序 |

每个公开能力都要有路线，但不为凑数量制造一个失真的 L1 loop。

## 3. Harness 形状

双方 kernel 使用相同数据、循环、load/store、tail 与防 DCE 方式。需要 placement replicas 时，由
harness 显式传 replica tag：

```cpp
[&](auto rep) {
  hyle_bench::stagger(rep);
  auto* src_ptr = src.data();
  auto* dst_ptr = dst.data();
  for (std::size_t i = 0; i + width <= count; i += width) {
    // HYLE 或 peer 的同合同 load/op/store。
  }
  nanobench::doNotOptimizeAway(dst_ptr);
  hyle_bench::mark(rep);
}
```

要求：

- `.data()`/capture chain 的差异不能只出现在一侧；
- loop unroll、remainder、alignment、restrict 和 initialization 对称；
- 不把 setup 放在一侧计时区；
- `doNotOptimizeAway` 与 checksum 不能让一侧产生额外依赖链；
- 输入覆盖正常 lane 分布和适用 exceptional set，不只用编译器可推导常量；
- exact in-place、partial overlap 或 unaligned 只有在产品合同存在时才测。

## 4. SIMD 专项仪器资格

每个 executable/TU/机器/ISA 分别过门，不能用兄弟 binary 的校准结果背书。

### A/A anchor

同一指令序列用 HYLE/peer 标签各跑一次，交换顺序与实现位置后应落在预声明 equivalence margin。
anchor 失败时整份 op verdict 为 `UNCALIBRATED/DIAGNOSTIC`。

### Known-delta anchor

为同一 kernel 增加可控的 load、dependency、shuffle 或 store，验证报告随 delta 单调变化，并估算
当前规模的 MDE。只能分辨 15% 而产品关心 3% 时，不得报告 3% 胜负。

### Code placement

极短 loop 至少使用多个独立实例和两个等价 layout bank，防止固定 uop-cache/取指几何选择赢家。
禁止以 best replica 单独签署；报告 median、spread、bank/order strata。等价布局翻转方向即
`INCONCLUSIVE`，除非布局是部署合同。

### Host

- x86 混合核固定在已登记 P/E core，检查 affinity/priority 返回值；
- Apple Silicon 使用明确 QoS，并报告无法硬 pin 的限制；
- 同一物理机的 Windows/WSL 不并行跑正式数据；
- 记录频率/温度/电源、SMT、后台负载和 timer epoch；
- 真 AVX-512/SVE/RVV 性能只在对应硬件签署，SDE/cross build 只证明执行/正确性。

### Artifact/codegen

保存 binary hash 与双方 loop 的反汇编，确认相同 ISA、无 DCE/constant fold、没有一侧意外 dispatch、
PLT/IAT、stack spill 或不同 inlining。counter 可辅助确认 instructions/uops/branches/cache，但不能
取代 consumer 结果。

## 5. 正式运行

1. 冻结 candidate、peer、manifest、contract、margin 与命令。
2. 每个顺序运行独立进程 paired samples；默认至少 15 个，Mac 高噪声形状可预声明更多。
3. A/B 与 B/A 分开报告；多个 layout bank/replica 先分层，再按预声明方法合成。
4. 保存逐进程 raw JSON、host metadata、calibration 和 artifact identity。
5. 分别记录 comparability（`COMPARABLE/NOT_COMPARABLE`）、qualification
   （`QUALIFIED/DIAGNOSTIC`）和 exact performance（`WIN/PARITY/LOSE/INCONCLUSIVE`）；
   non-inferiority 另设单边 gate。
6. 在未用于开发调参的 ISA、类型、规模或 consumer kernel 上做 holdout。

Windows、Linux 和 macOS 的具体 preset/工具链只看 [setup](../setup/README.md) 与脚本 README，避免
在这里复制会过期的安装命令。

## 6. LOSE 调查阶梯

不要从比值直接改实现。按顺序检查：

1. 合同是否真等价：精度、tail、alias、load/store、dispatch、setup；
2. A/A、positive control、order、placement、affinity 和 MDE 是否合格；
3. 最终反汇编是否为预期路径，有无 scalar spill、额外 move、调用或 dependency chain；
4. 算法/recipe 是否比 peer 多指令、访存、分支或更差复杂度；
5. 编译器是否漏 intrinsic/peephole，可否用可移植 recipe 或局部 platform specialization；
6. 差异是否能在 whole-buffer/真实 consumer kernel 重现并达到 material budget。

修复后重跑完整适用 stratum，不只重跑原失败项。修复若改变精度、public shape、常态 metadata 或
其他 ISA，candidate 失效并回到模块门。

## 7. 什么可以进入产品

- 更好的复杂度、少一次遍历/转换、避免 material spill/copy 或真实 kernel 可复现的 codegen 改进；
- 自然 C++ API 下的 batch/whole-buffer 路径，且有实际消费者；
- 不向其他 ISA/类型/资源维度转移 material 退化的 specialization。

以下只进调查日志：只在固定链接地址赢、换 order 翻转、低于 MDE 的个位数差、人工重复单指令才
显著、或真实 kernel/profile 无法观察的差异。不能为它增加 public entry、对象字段或常态分支。

## 8. 输出位置

- 方法、脚本与 troubleshooting：`docs/developer/`；
- 某 candidate 的矩阵和结论：`docs/implementation/evidence/performance/`；
- 原始 JSON/CSV、binary/hash：根级 `evidence/performance/`；
- 用户只看到经当前证据支持的选择与陷阱，不看到旧 orderswap 任务表。
