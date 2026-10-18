# Native page紧凑metadata实现阶段

AUDITED NP-01产品草稿197d39684；[原命令/源码失败/累计/selector](raw.zip)、[收据](receipt.json)。
Windows四源tests PASS：F16/BF16仅metadata，页表4×max_page_refs、无Q/head/完整KV展开；
query8192/head16仍32byte页表，logical95>physical64重复页保真实physical desc。generation/
owned layout、span不借后续、exported只读lease保recipe/layout、空表零初始化、foreign schema/
预算/非finite与逐cold分配失败回收反证。初用不存在的Schema比较运算符compile FAIL保留，
修为原owns身份方法后PASS，不改schema/数学合同。

这是metadata阶段证据，不是GPU数学、native controller、P1或整个G01/G02完成。三san/
三交付和原组件合同等门继续未证；全部未跑项不写PASS。没有fullGPU run或新的queue。
[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-native-paged-read.md)。
