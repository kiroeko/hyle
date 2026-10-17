# CPU ragged Graph factory 三交付

candidate `973d3ffbf642aff06f6316fcec4b3cc99eba2e4d`；原642数学与a0c047ea4 Graph
exact-resource修正及其三san证据保留。[receipt](receipt.json)与[raw](raw.zip)记录本轮
真实source/static/shared、输入和产物身份、所有compile/link FAIL。活动卡见
[CPU卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-cpu-native-ragged.md)。

普通consumer仅链接hyle::compute/graph并include公开头，source和实际relocated static/
shared均PASS：Direct两request F16/BF16三depth、readonly/tail、独立slot、旧output与
完整topology复用；原Graph桥两个dependent native ragged regions实际执行，原resident
single consumer同时回归。没有private target、fake macro或新wrapper。实际Vulkan-only
旧树只configure，CPU provider header/library都不进入安装规则，没有GPU/build运行。

先移除错误工厂声明macro门后，shared链接LNK2019实际复现；公开manifest补齐原已
构建ggml factory root后通过。公开macro/链接形状修正不改算法/ABI，原产品TU及三san
组合identity可复用。selector把新增root归给全部旧roots的BLOCKED计划与独立工具卡
保留，不改工具、不以此开启无改动owner或全库验证；实际未运行格不写PASS。

多请求比较适配器三形状独立double oracle/readonly通过，max abs error约6e-5。
诊断min1 raw9.422s不是合格P1；min2实际15.641s工具耗时保留，raw被诊断helper误
覆盖不能重放，helper已改追加尝试。正式合同尚未资格/取样，不能由诊断点值称性能PASS。
原单请求CPU/Vulkan性能证据、所有失败/STOP继续保留，整个G01尚未完成。
