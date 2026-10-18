# CPU P1 v1：冷诊断后停止过重测量排期

[receipt](receipt.json)、[raw.zip](raw.zip)保留原fresh1构建、所有已运行A/A、原退出码
及851.5s累计投入。长prefill单次约133.5ms，104paired rounds且每epoch至少2次，使
该pair每process约55s，八strata×n15需约两小时；这不是产品FAIL或P1资格PASS。
尚未进入正式product samples时主动停止，未删树、覆盖日志或重置原预算。

新v2在取正式样本前固定16paired rounds并发出/核对实际轮数遥测；原独立double数学、
readonly/双slots、peer、5%门、n15/31规则及公共controls均不变。公共仪器代码不改。
CPU产品d91a未改，新benchmark身份必须重新编译/A/A/资格；不能把原未完成项记PASS。
后续独立clean checkout使资格运行不冻结canonical产品工作，heavy任务仍串行。
