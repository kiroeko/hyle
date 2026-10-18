# F5 actual Air Apple宿主ASan

固定product25e，actual MacBook Air Mac16,12/M4/16GiB/macOS27.0.1/26A434/Clang21。
独立ASan binary tree，共享锁定dependency source cache，普通Compute/Graph Apple-only，
只原public/native二消费者，库按实际内存j2/j1、consumer1；沒有额外SIMD库。
实际commands核对两consumer、native/controller/Graph factory/common coordinator/leased
Graph有-fsanitize=address。原GPU数学由MLX实际执行，ASan instrument HYLE宿主代码。

公开四域Direct/Graph与完整native风险fixture PASS：非uniform/GQA/causal/padding、
精确页/同版本mirror/old results/两bank两slot/caller OOM、actual queued worker失败/
cancel/abandon/close/drain、open失败不复用及append/F4/metadata quota0。具体episode
以raw实际日志为准；不跑原算子catalog或P1。

第一次consumer启动由runtime明示detect_leaks不支持而exit-6，未进入测试；保留原FAIL/
elapsed，仅detect_leaks=0继续同树/剩余budget，不加suppression或改产品。Apple ASan
宿主访问PASS；Apple LSan这一工具facet unavailable，不能写LSan PASS或把全部san类别NA。
caller owner/quota证据不冒充LSan扫描；Metal/MLX私有GPU storage/未instrument第三方
代码不由此证明。命令/身份/输入hash/actual flags/全部累计见[receipt](receipt.json)/raw.zip。
Linux CUDA其余native宿主facet/P0/guide/红队/完整closure继续。
