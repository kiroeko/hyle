# Y27：原版 F32 Scan 的四构建回归

固定候选`198cccba6d785132aace3e9744e9f1603dc342b0`。本记录签F32 Scan的普通Direct/Graph
接入及所选provider/route回归；不签I32 Scan、Scatter、Scan三交付、完整C1或P1。

| 环境 | Provider / Direct / Graph | 结果 |
|---|---|---|
| Windows strict static | 19 / 67 / 22 | 108 PASS |
| Windows strict shared | 19 / 67 / 21 | 107 PASS |
| MacBook Air Mac16,12 static | 19 / 67 / 22 | 108 PASS |
| Mac mini Mac16,10 static | 18 / 67 / 22 | 107 PASS |

共430项。Windows先行6个Scan定向用例通过，不再重复计入430；两种构建的source/MASM合同各2项
通过。本版没有产品纠正。Windows/Air启用validation，无VUID或同步错误；Mini的一个validation
专属用例明确未运行，不计PASS，Mini只签correctness/portability。Windows DLL邻置、PATH unset、
SetErrorMode(0x8003)，所有构建parallel2、重型运行串行。

## 本次实际覆盖

- 原版cumsum单遍/两遍、inclusive/exclusive，非二进制输入的独立FP64 prefix；沿既有CUDA
  Scan的误差责任，用前缀绝对值和的3e-6界限，NaN/Inf单独检查，不由结果调宽门限。
- 宽度1/17/127/128/129/255/256/257/1023/1024/1025/4096/4097/8193，1/17/128/513/1025行，
  以及262144/262145/262657行的第二级XYZ边界。内部形状组合不是额外用例数。
- 任意/负轴、rank8、奇数与零stride、空张量；0–4 byte起点、捕获与动态输入、非有限前缀。
- 两个运行槽、5次变化输入、保留旧结果、源数据保护、Graph Abs→Scan→Neg与inclusive/exclusive
  分支发布；冷buffer/workgroup限制拒绝，不提交非法native dispatch。既有OOM/失败排空一并回归。

数值计算复用锁定ggml cumsum及allocator/runtime；没有新增shader或vendor补丁。HYLE冷布局
适配负责轴/stride/descriptor对齐、exclusive前缀拼接；原版shader没有总行数guard，因此在
512/262144行的余数处最多分三个原版Scan并拼接。完整转换、临时量及原厂两遍块前缀重读必须计费，
正确性通过不表示性能胜出。

[聚合记录](result.json)与平台receipt绑定候选、日志及EXE摘要；`w1-raw.zip`包含完整Windows
命令、首次定向结果、selected/skipped计划，Mac两个zip包含对应平台原始日志与机器身份。
当前统一实现LOC为[47,759行/193文件](implementation-loc.json)，不含测试、bench与第三方；自检通过。

Mini当前源码和归档核验后清理四份具名旧验证源码，共344,914,592逻辑字节，见
[清理记录](y27-mini-source-cleanup.json)。当前源码、构建cache、工具/依赖与证据均保留。

Scan的source/static/shared实际片段及完整C1性能/优选/淘汰仍未运行；不复用先前Gather的交付
替它签收。原global package gates和CPU oneDNN资格blocker仍保持各自状态。后续整数Scan的
[源审计输入](checked-integer-source-inputs.json)仅记录锁定来源，不表示新增Provider或产品准入。
