# D5 CPU：原版 ds4 状态量化批量接线

<!-- MODULE REVIEW: AUDITED -->

## 范围与重新打开依据

baseline `5fabf548565a4598b8b7bf490b2eaabfc942cca1`；owner `compute_provider_native`。
重新打开依据：D5官方DeepSeek两个状态量化合同是实际缺口，
[新原生资格](../../../../evidence/correctness/llm-moe-d5-real-fragments/mature-low-precision-batch/README.md)
证明ds4原CPU group64 FP8、group32 FP4可以承载。D4的自研退场不撤销，不恢复旧自研数学。
普通用户仍使用既有CPU Direct/Graph入口；不新增公开API、Provider排序或兼容层。

## 有限 FIX N1–N4

1. 将MIT许可的ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`原函数窄摘录为私有依赖，
   固定完整source/函数/归档/许可hash，沿既有CMake lock/package seam投影；不复制整模型。
2. 同批接入既有`BlockQuantizationRoundtrip`的FP8 group64和FP4 group32：冷期只按
   精确操作、BF16 dense whole-group形状绑定原函数；group128无成熟同合同原函数，继续拒绝。
   每组用固定不超过256字节栈暂存，经已有SIMD基础BF16转换适配原F32函数，再写回BF16。
   无热分配、权重缓存或新线程池；不修改NativePlan布局，不改变算术顺序/scale/rounding。
3. 复用既有CPU资源映射、invalidate/flush/close、Completion和失败通道。非法descriptor、
   alias、容量由原公共层检查；非有限payload安全失败，已接受前缀不承诺回滚。
   invocation不共享可变scratch，完成对象保活和OOM继续走现有设施。
4. 同一候选批量覆盖普通Direct/Graph、两种格式、变化输入、极小值/全部有限BF16/中点、
   group128冷拒绝、别名/资源失败与重复执行；验证受影响owner和真实CPU消费者。

## 取舍、验证和停止

本卡是两个必要成熟操作的接线，不能代签完整官方FP8/FP4 MoE或增加完整画像格。
C10转换的完整group128线性组合另有真实缺口，本卡不把自写循环包装为新成熟算子。
默认策略只覆盖当前其它成熟Provider拒绝的精确域；未引入热期搜索或性能调优。

P0审成本；没有自研算法比较或逐算子P1。D5完整片段P1义务保持，新增状态能力的完整
消费者成本尚未验时不能写总性能PASS。源码/静态/动态及Windows/WSL CPU正确性按受影响
批次完成，不重跑GPU/Apple或已合格其它D5全矩阵。
一次批量接线、一个不可变候选的低并行构建/串行验证；仅明确adapter/构建错误允许一次纠正。
适用选择器、docs/closure（明确previous-ref）、统一LOC；三交付/资源未验仍待验。
原函数出现不符合合同的新反例则停止该域，不优化数学或放宽精度。完成有限FIX后保留D5其它缺口。

## AUDITED 决定

来源归档通过本地代理取得并与Windows固定Git blob核对，六个数学函数原样借用；
许可为MIT。只有既有CPU provider取得私有依赖，不改变公开头或路线对象。
N1–N4固定；同一操作家族的group128仍拒绝，C10和Apple失败候选不进入本批产品。

### 首次配置的接缝纠正

候选 `1486d10c` 首次 configure 检出 `model_ops.h` 的 owner 是 `compute_ops`；
native 过去仅间接使用它，新增直接 include 必须如其他 Provider 一样声明直接私有依赖。
本卡允许的一次 adapter/build 纠正限定为 native 的 `PRIVATE_DEPS compute_ops`；
这是现有 schema 的显式依赖，不增加运行路线、模块或自研数学。配置失败不计任何测试 PASS。
