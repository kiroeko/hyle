# IREE TSan唯一条件修复的普通交付影响

固定F5 25e与IREE修复11c348e40。include/src产品diff只有owner CMake两行；删掉唯一新
elseif(tsan)/IREE_ENABLE_TSAN分支后文件与25e逐byte相同。OFF/ASan/UBSan仍严格沿原路径，
全部C++源码/头/常态字段/中央factor边/依赖锁不变，不以配置缺陷重开数学或回放算子。

逐一重读18格原raw/receipt hash与producer configure日志，每格均实证sanitizer=off。
因此本san-only修复下普通SDK原artifact/consumer/迁址/loader资格保留，原候选和失败/预算
不改，不冒充新run。Win Vulkan四格先前77df→25e的未影响映射仍独立保留；本页只再证明
25e→11c348e40的TSan条件差分。源码分发正常OFF亦沿原代码路径；实际TSan变化另由
Air Vulkan两个原消费者及原C runtime/adapter同instrumented flags证明。

全部18summary、原raw hashes、before/after CMake和影响集合见[receipt](receipt.json)。
中央component工具聚合仍原待裁决，不由此映射改为PASS；未修改公共selector/统计工具。

统一LOC命令 `python scripts/loc-report.py --module compute-provider-iree-vulkan --commit
11c348e404df2c3fc404056ff3a9e885d4d990bc --self-check --json` 实际self-check PASS。
[原报告](iree-loc.json)中本owner conventional src实现1375行/6个native files，全部自有native
仍174338；本次产品净增只有2行CMake、native净增0，不把原1375行算作新实现。
全仓库所有语言2800717为新证据/文档/工具等全部口径，不表示增加了同量基础库源码。
