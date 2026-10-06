# CPU ATen 冷编译存储：CP1–2

candidate `a527dd73dcc21a5b4b89979d43339d67becabe86`，Windows canonical；Linux为前向验证副本。
ATen原程序/SDK/数值/布局/执行/对象生命周期未改；修正额外private cold resident/peak=0声明及版本。
Program与TensorDesc仍由supplied allocator计费，编译总内存非0；native invocation/private Tensor
上界仍unknown，replay仍managed，不承诺capture或zero-hot-allocation。

## 事实与实际验收

- 旧普通strict-compilation Graph返回预期counterexample 5：未绑定ATen；不是原库数学FAIL或PASS。
  源码闭包没有native cold Tensor/JIT/packing或额外heap：Program只含值/span/固定数组，prepare
  单次supplied allocate，destroy由相同allocator释放；错误Status诊断与Context/Queue/driver/
  invocation按既有quota合同另计。
- 新增一个case覆盖**18个已接family/cold程序**：proposal额外resident/peak均已知0、retained bytes
  已知且非零，managed/invocation unknown/capture不变。它不是18个数学执行资格。
- 普通Direct/Graph consumer在strict compilation resident/peak 64MiB下实际选择并执行成熟ATen，
  两次变化输入、真实wait/下游Copy与close正确；零metadata预算失败，限制ATen后的prepared-only
  与strict invocation unknown没有被悄悄放宽。

| 实际门 | Windows / Linux |
|---|---|
| Native factor，含独立ATen | 各141/141，无fail/skip |
| 普通CPU受影响suite | 各12通过/1旧fixture skip |
| 受影响普通Graph suite，排除未变D5 | 各17/17，无fail/skip |
| 公开头 | 各1/1 |
| 同ordinary budget consumer | 两平台source shared/static与installed shared/static全部实际通过 |

三段完整源码逐字节相同：Program/数学/执行/invocation/mapping/lifecycle、cold prepare、SDK-off
stub；1331个其它既有非fixture输入SHA256相同。只复用[AS](../cpu-aten-scatter-add/README.md)已证明
且未变的数学、生命周期、D5及关闭接缝合同；不把这些称本轮重跑，不用它们替代新预算选择和新binary
交付。1337个当前产品/test/构建/consumer输入在两平台前向hash匹配。

P0完整成本与原程序相同，新行为发生在冷匹配/预算绑定；没有自研优化/比较/新WIN或GPU验证。
整个CPU goal仍需冻结自研算子的有限精确域裁决；旧unknown声明不能作为Native唯一实现证据。

统一Compute+Graph实现 **61,519行 / 249文件**，不含tests/vendor。四个本批consumer目录已删除
**625,342,086 logical bytes**，SDK/producer/cache保留。完整baseline反例、源码/protected hashes、
consumer/drivers、所有命令/XML/身份、selector/LOC/cleanup见[qualification.zip](qualification.zip)，
摘要/hash见[receipt.json](receipt.json)。本批没有产品/fixture纠正或预算外重测。

[终审卡](../../../docs/implementation/tasks/compute/cpu-aten-compilation-storage-declaration.md)只关闭CP1–2，
不宣称全部CPU自研或Compute+Graph四路线完成。
