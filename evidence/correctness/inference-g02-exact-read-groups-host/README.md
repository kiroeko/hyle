# G02 精确读守卫组宿主资格

产品graph candidate `5ce0bed8acb54d67cb70423134b73a6afea421dd`，测试8991b066a。
[收据](receipt.json)与[原命令/flags/所有失败/累计](raw.zip)保实际输入与原八个测试。
Windows source8 PASS；WSL ASan+LSan、UBSan、TSan各8 PASS，实际graph lease submission、
compute resource leases与test TU的instrumentation均核验，重型串行、四jobs。

两accepted unknown prefixes必须都退休才释放原权限；shared block去重、未引用可写、
无效/容量绑定不改变旧input、部分admission回滚、caller guard span不借到后续、重绑/空组/
普通binding替换、逐cold分配OOM无live metadata与64次无allocator调用均反证。此处native
prefix是控制fixture，不签真实GPU数学/driver安全。原256容量容不下两个默认256对齐块的
source FAIL及检查Result后定位FAIL均保，修512 fixture后PASS，未改变产品算法/权限。

source/static/shared新detail seam及现有真实native hybrid消费者尚待交付验证。此共同
退休承载不产生whole-arena读取许可，native布局/generation与实际page-reading数学仍由
G01/G02 native owner补齐；原pool查重最坏O(guards²)保留，不签G06/整个G02完成。
[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/graph/inference-g02-multi-queue-batch.md)。
