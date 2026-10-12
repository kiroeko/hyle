# F5 Graph Windows首批编译

AUDITED有限编译checkpoint，未签candidate/全F5。原Windows MSVC19.50/static/shipping自然
Compute/Graph source父项目复用；只构建new batch /WX与包含batch.h、region_access.h的普通
消费者目标，实际运行成功，没有跑旧算子测试。依赖闭包的必要增量产物仍保留在原build log。
新Graph头以该消费者TU验证；独立TU/header全矩阵在F5-06继续。

首次MSVC在HYLE_ASSIGN_OR_RETURN的ternary分支尝试复制move-only PreparedBatch编译失败，
修正为显式if/else分支；同树build-resume1 PASS，原FAIL和累计configure/build/consumer预算
未重置。库实际4路并行。精确命令、输入hash和原日志封存于receipt/raw.zip。

只签Windows产品编译/原入口链接发现，不签Windows FlashInfer native paged数学；WSL GPU
Graph资格由独立真实native证据签。F5其他风险、四路线reference/append和三交付仍未完成。
