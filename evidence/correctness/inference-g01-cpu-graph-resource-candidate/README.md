# G01/G02 CPU Graph exact resource candidate

CPU不可变candidate `642cc1e67566994a85a65adbe0db8956c1d0350a`继续保留，Graph batch
资源接缝candidate `a0c047ea45cb46884da7f578684ff416ccfdefa5`，活动owner卡为
[Graph卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/graph/inference-g02-multi-queue-batch.md)。
[receipt](receipt.json)记录实际命令、退出码与累计尝试；[raw](raw.zip)保留全FAIL/PASS、
输入SHA、工具链和仪器命令。它是候选证据，不是整个G01或模块的CLOSED证书。

Windows普通CPU入口两请求F16/BF16 D64/128/256，readonly/tail、空request、重排、
独立slots/旧output、相同完整topology复用、foreign schema/小预算拒绝、逐分配点失败
回滚和accepted math error不发布输出/原实际uses退休四tests PASS。原FLASH_ATTN_EXT的
K/V roots在冷资格实际验证；热按packed origin直接subview，没有KV物化。

WSL GCC14同输入副本ASan+LSan、UBSan、TSan各13tests PASS（四batch与九既有layout/
storage回归），保留provider两个TU、batch core与新test真实instrumentation。Windows
构建最多4jobs，WSL与Windows重型串行；旧single-request证据和性能树完全保留。

安装后的普通Direct消费者只链接公开compute/graph，Graph专家factory在当前facade
不声明，因此该分支明确不签SDK Graph资格；source独立target实际链接既有provider。
三交付/Graph与性能只按receipt实际结果裁决，缺项仍待，不由源码或未运行项推导PASS。
已知native plan占用保守计费；private pool上界UNKNOWN，paged/capture本批未支持。
