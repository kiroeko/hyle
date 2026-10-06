# C5 原text与新卷积安装SDK

产品/测试候选 `c740abc74a5a62ffacf60b26798228ca484317df`。
Windows重装static/shared SDK，各普通Direct/Graph45项，合计180项通过，无失败/skip，
见[准确收据](windows/receipt.json)。新低dtype卷积ID9必须由更新后的产品库执行，旧SDK不代签。

独立消费工程只`find_package(Hyle)`链接安装产品；静态原ggml/GTest只作原codec/reference与测试依赖。
Windows runtime DLL由既有helper邻置到各exe独立目录；安装prefix使用已搬移的C3路径，
未变资产/离线cache合同保留。lowdtype conv、uncompressed F16/BF16/F32/I32、实际K/N/词表均实际执行。
原模型片段、packed F1/F2与资源证据按未变实现的准确范围复用，没有重跑性能或假签新winner。

Linux同批正在执行，尚未签PASS；C5全范围聚合、终审与模块状态还未关闭。
归档后按CMake owner/绝对路径/活动进程校验删除Windows临时消费目录，明细见 `cleanup-windows.json`。
producer、安装SDK、source/dependency cache和原始收据保留。
