# Native page metadata风险与共享controller负空间

metadata草稿197d、controller草稿3747；仍AUDITED。[收据](receipt.json)、[全部命令/flags/输入/原失败](raw.zip)。
ASan+LSan、UBSan、TSan各37tests PASS，包括新metadata与原layout/materialize/arena回归，
实际新metadata/layout/test TU instrumentation核验。只有metadata与宿主风险资格。

共享controller改为编译期单表/双表policy，原固定banks/CopyBatch/frame compare/LeaseUsePool/
actual retirement同一实现。实际Air原Apple/Vulkan materialized reference消费者分别
12.110s/23.145s build、4.184s/7.339s run PASS；它们证明旧双表negative space保，不证明
新的单表kernel或新paged public调用。固定Context仅单表policy保存，旧policy没有新增该字段。

原组件合同第三次255.328s PASS：补实际已有SDK环境并使用仓库外scratch根。第一次
137.157s缺SDK与第二次299.531s source-cache根在产品内FAIL均保，三次累计692.016s，
原工具/预算不改。中央新split header引发selector保守全选的计划仍保，实际只有owner
compute_batch产品改变，不重开/跑全库数学。NP02paged准备/typed wrappers及NP03/04数学、
新头static/shared和GPU实际页读等门仍未证，不签CANDIDATE/整个G01/G02。
[活动卡](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-native-paged-read.md)。
