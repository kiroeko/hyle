# H2 MacBook Air 原生源码验收

- 产品：Windows完整6c415ae968827a04985f5aac94161fa0e1fbf847 archive；后续只同步实际
  1989b1fe3产品两文件与合同consumer delta。源码/mtime复用以bytes匹配为准，不改vendor cache。
- 机器/工具链同[实机语法证据](../compute-h2-apple-wiring-syntax/README.md)。
- source配置和增量build1实际通过，90步包含H1已改变common/API和新Sched/native组合；peak owned RSS约632MiB。
- 普通Compute/Graph原CPU+Apple入口及新Direct权重/新组合consumer实际4/4通过，约20秒；
  组合包括独立Q4 FFN数学oracle、CPU主存大权重/512字节native边界、正交分块与两槽/旧lease/报告。
- 后续新增合同target只重编两处编码边界变化及新fixture；该target实际1/1通过。
  覆盖ID/描述/容量/overflow、不同dtype、3字节native尾padding、GGUF Q8_0原34字节、native范围、
  metadata/payload OOM及rollback、strict私有unknown拒绝、native边界cap、ready权重/更新payload、
  旧lease、reset报告失效、host-wait拒绝、cancel、close、abandonment与CPU接纳后GPU关闭的晚到失败。
- 不宣称installed static/shared、搬迁、缺路线/安装头、物理copy/compute重叠或完整H2已完成。
- 普通域原4项的public接口/正常dtype/算法未改变，后续编码修复只作用此前误拒绝的物理byte carrier；
  保留原数学结果，不因test/docs或有限编码修复重跑全集。

[源同步/归属](source-sync.json)、[初始实际命令](source-result.json)、[4项CTest](source-math.log)、
[定向合同批次](source-contract-result.json)与[合同真实stdout](source-contract-stdout.log)给出原始退出码。
SDK与native overlap仍为未运行，后续独立补齐。

configure log仅移除工具输出行尾空格；命令、退出码和语义内容不变，原始文件仍在Windows本轮记录目录。
