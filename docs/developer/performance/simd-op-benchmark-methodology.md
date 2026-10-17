# SIMD 操作级测量

SIMD loop 可能只有几条指令，容易被常量折叠、代码放置、取指、核迁移和 timer floor 支配。
操作级 benchmark 用于定位生成代码与算法差异；整体效果还需用正常 whole-buffer 消费者确认。
通用方法见[公平比较](benchmark-evidence-standard.md)和[统计](instrument-v1.md)。

## 对齐操作与 harness

- 明确 lane type、vector width/scalable tag、mask/tail、alignment、alias/in-place、精度、NaN/Inf、
  subnormal、饱和与舍入；名字相同不代表语义相同。
- 双方使用相同输入、循环、load/store、tail、防 DCE 和初始化。不要只对一侧计 setup、dispatch、
  capture chain 或 checksum 依赖。
- 输入包含正常与适用异常 lane，不使用编译器能推导的常量掩盖运算。
- alias/retag/construction 主要检查正确性与零额外代码，不必强造纳秒循环；算法和 whole-buffer
  操作测实际规模，缺真 ISA 的平台只说明编译与正确性范围。

## 检查测量是否可信

- 在同一 executable/TU/ISA 中设置语义和最终指令相同的 A/A anchor，并交换顺序与位置。
- 添加可控 load、依赖、shuffle 或 store，检查已知增量响应和当前可分辨的差异。
- 极短循环检查多个独立实例与等价 layout bank；报告完整分布，不选择最快 replica。
  换布局后方向反转时，保留不确定性。
- 保存 release binary/hash 和双方 loop 反汇编，检查 DCE、constant fold、ISA、dispatch、PLT/IAT、
  stack spill 与 inlining。counter 辅助归因，不能替代消费者结果。
- x86 固定并记录同类核心及优先级；Apple 记录 QoS 和不能硬 pin 的限制；Windows/WSL 同机采样错开。
  模拟器或交叉编译的时序不能代表真 AVX-512/SVE/RVV。

使用 nanobench 时，交错 alternatives 的例子：

```cpp
[&](auto rep) {
  hyle_bench::stagger(rep);
  auto* src_ptr = src.data();
  auto* dst_ptr = dst.data();
  for (std::size_t i = 0; i + width <= count; i += width) {
    // 同语义的 load/op/store，双方具有相同边界。
  }
  nanobench::doNotOptimizeAway(dst_ptr);
  hyle_bench::mark(rep);
}
```

`stagger/mark` 适用于已有 placement harness，不能凭这个片段证明校准或公平。
保存逐进程/block 样本与顺序、位置、输入、机器和产物信息，再做成对分层分析。
已有 op-parity 输出可用 `scripts/op-parity-report.py --check-orders` 检查采样输入与顺序是否完整；
完整比较仍按本页核对工作量、数值、锚点和实际消费者。

## 调查落后

先核对精度、tail、alias、load/store、dispatch 和 setup，再检查 A/A、分辨率、顺序与位置。
确认双方走了预期路径后，检查额外指令、spill、访存、分支、依赖链和算法复杂度；最后在真实规模或
消费者 kernel 中验证。可复现的小幅落后也保留，低于仪器分辨率的点值不推断成胜负。

修复后验证受影响的类型、ISA、规模和数值边界，并保留未用于调参的用例。公共接口、常驻字段或分支
的变化要有实际使用收益，避免仅在固定地址或人工循环中获胜的设计。
