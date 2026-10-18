# F5 Graph batch测试归属定向验收

product candidate25e不变，新增tests/graph_batch正常登记，MSVC19.50 shipping/static/WERROR。
复用原common build/tree/cache，只构建hyle_graph_batch_tests及实际直接公共闭包，j4/test1。
三项真实共同测试与一个batch.h独立TU PASS，不运行旧算子全集或GPU数学：
paged/ragged×F16/BF16的容量/indices/mask端口及输出描述；recipe释放后Graph独立保活；
空program拒绝；逐个实际cold Graph caller allocation故障回滚/无半成品/新准备恢复。

原首次fixture对Result使用模糊={}赋值的MSVC编译FAIL保留，改为显式清空公开
MaterializedGraph值；产品没有变化。同root预算累计，命令/日志/hash见[receipt](receipt.json)。
源码tests=ON公共component合同另沿其原失败103.437秒/300预算续跑，不由本三项测试代签。
完整F5 SDK/native san/P0/LOC/guide/红队与最终closure尚未签。
