# CPU 成熟实现准入：小矩阵与稳定门控

产品候选 `65a02982ac108d57b13769029f56a6a6903895be`；Windows 为 canonical，Linux 为单向同步的
Ubuntu 24.04 / GCC 14.2 验证副本。范围分别由
[oneDNN 卡](../../../docs/implementation/tasks/compute/cpu-onednn-small-matmul-admission.md)与
[ggml 卡](../../../docs/implementation/tasks/compute/cpu-ggml-gated-numeric-admission.md)持有。
本批不裁决 Native 胜负，不宣布 CPU 或所有模型覆盖完成。

## 实现与普通消费

- oneDNN：删除无当前资格的 serial M<16 拒绝；保留原 BRGEMM、F32 accumulation、exact dense
  row-major、caller workspace、原失败检查和真实 completion。M1/M4/M15/M16 的奇数 K17/N65 与
  K64/N64 有实际数值和变化输入重放；普通 CPU factory 不要求用户选择 Provider。
- ggml：原版 SwiGLU 后接一个原位内部边界，仅修复 gate=-Inf/value 有限时的稳定零极限。
  value 为 Inf/NaN 时继续产生 NaN；signed zero 保留 value 符号。finite 原核不改。
  packed/separate、F32/F16/BF16、1/4 native_threads、尾部与变化输入已检查。
- 普通 Graph 的 gate→MatMul 保持一个成熟 region，两节点的逻辑报告和 provider ID 已检查；
  三 dtype 的 MatMul 实际消费修复后的中间值，修复不能延迟到最终导出。
- 独立交付消费者只链接 `hyle::compute` / `hyle::graph`，安装消费只给 HYLE prefix；没有 producer
  私有 include/library 路径。Windows 使用既有 helper 将实际 DLL 闭包放到 exe 相邻目录。

## 实际验证

| 批次 | 当前事实 |
|---|---|
| Windows oneDNN | 7/7 CTest 程序通过；source/static/shared 独立普通＋专家 Direct 和普通 Graph，M1/M4 两次变化输入重放通过 |
| Linux 共同批次 | 8/8 CTest 程序通过；428 个注册 GTest，414 实际通过、14 旧模型 fixture 跳过；ggml Provider 44/44，包括新增门控测试 |
| Linux oneDNN 交付 | source/static/shared，普通＋专家 Direct 和普通 Graph，M1/M4 两次变化输入重放通过 |
| Linux ggml 交付 | source/static/shared，同一独立普通消费者；三 dtype、两种 Direct 布局、整个 gate→MatMul Graph，-Inf/finite 重放通过 |
| Windows ggml | 4/4 CTest 程序通过，377 个注册 GTest 中 363 实际通过、14 旧 fixture 跳过；Provider 44/44；source/static/shared 同一独立三 dtype Direct＋完整 Graph 消费者通过 |

Windows oneDNN 的先行收据绑定 `df9006e8b7858e522a1d03ea433f31ca66102fe9`。它与最终候选之间，
oneDNN、公共 Compute 头、Compute/CPU backend/CPU route、Graph/oneDNN Graph/CPU Graph 的产品
输入没有变更；后续新增产品差异只在 ggml 的 SiLU 边界。Linux 共同批次和交付绑定最终候选。
测试的 GCC portability 修正另有实际提交，未改变产品数学；不以不同 Git 文档提交强制重复验证。

14 个未部署 fixture 的跳过不计 PASS，也不声称本批重新执行了完整模型权重或历史 D5/P1。
本批测试只资格化具名数值与资源接缝；旧画像、其它 dtype/domain、GPU、D5/F35 与 Native 逐域
裁决的剩余事项保持原状态。

## 失败、纠正与成本

保留原始失败日志和 append receipt。GCC 首次编译暴露旧 fixture 的 aggregate 默认初始化与
GoogleTest 宏外缺少花括号；修正限于已选 CPU 测试，未关闭警告。Linux 从 tests-only cache 开启
安装时，第一次局部 build 未重链整个安装闭包，旧 platform_thread 的 RUNPATH 与安装 padding
不一致；补齐同一副本的产品闭包后安装和实际独立消费者通过，没有修改 Windows 产品构建系统。
Windows 驱动一次误设 tests+install 同开，被既有合同拒绝；按既有 tests/SDK 分离合同修正，失败保留。
这些不冒充产品正确性 PASS，也未触发自研优化或性能仪器修改。

P0：oneDNN 删除冷匹配分支，执行数学与状态成本不变。ggml 增加一次 O(n) gate 检查与一个内部
view node，复用 SwiGLU 输出，不增加完整 payload 缓冲；回调在冷 lowering 固定，原生 pool 执行，
无 userdata、热搜索或计时裁决。既有 arena 预留与 view/slot 回收路径承载该节点。
后续公平 Native 比较必须计入这个检查、必要 cast、同步与完整片段，不能使用未适配原函数成本。
本批不运行 P1、不签 WIN，不复用旧 scalar ggml 或 Native-vs-Native 结果作为算法胜负。

## 工件

[实际收据](receipt.json)与[压缩工件](qualification.zip)包括命令、原失败、输出、selector 的
selected/skipped reasons、两个独立 consumer、Windows→验证副本的 1,580 项输入 manifest/哈希，
以及统一 LOC 报告。归档 SHA256、大小与文件数见 receipt；无 SDK/build 二进制大副本。

Compute + Graph 实现为 **61,035 行 / 248 文件**，统一 `loc-report.py` 边界，不含测试与 vendor。

终审 selector 只选 docs/closure，130 门跳过或复用，不因状态/证据提交重新构建或启动 P1。
closure（previous-ref `f5a4aabd`）、docs 与 diff 检查通过。收据封存后清理本次 Windows 两个临时
门控 consumer 的独立构建/DLL 副本，释放 **576,193,086 字节**；保留 producer、SDK 与依赖缓存。
