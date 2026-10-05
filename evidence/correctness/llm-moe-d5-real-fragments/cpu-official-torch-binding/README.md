# CPU 官方低精度：已完成的 Windows 批次

产品候选 `3c22b7b9f230abc86e749225174d86b59c79881e`；MSVC 19.50.35728，
Release、C++23、产品 Werror、instrumentation OFF，OMP/MKL 两线程。
数学来源为固定 TorchAO 原 codec/量化和 CPU-only LibTorch 2.8.0 ATen。
普通入口仍为 CPU Direct prepare / Graph compile，无 Provider 排序或热期搜索。

| 已实际运行项 | 结果 |
|---|---|
| Native owner | 106/106 PASS；managed、真实 OOM、恢复、非法/重复 ID、独立并发槽、冷 metadata 配额 |
| CPU Direct | 157/157 PASS |
| CPU Graph | 173/173 PASS |
| 公开头 / CPU executor | 三组公开头及 executor 检查 PASS |
| Windows source / static / shared | 每种交付两个独立消费者、各13项官方低精度测试，共26项 PASS；搬迁、独立 include/link、DLL邻置及许可核对 |
| SDK ON/OFF 配置 | OFF 不探测不存在 root；ON 在 HYLE 自身版本检查处明确拒绝不存在 root |

普通测试的原始 CTest / GoogleTest 输出在
[普通批次](3c22b7b9f230-tests/receipt.json)和其 `native-tests.log`。
三交付分别见[源码](3c22b7b9f230-delivery-source-r1/receipt.json)、
[静态](3c22b7b9f230-delivery-static/receipt.json)、
[动态](3c22b7b9f230-delivery-shared/receipt.json)。
HYLE static 仍链接外部 shared LibTorch，并非完全静态进程。

第一批资源峰值和失败保留在
[首次产品批次](../../../research/llm-moe-d5-cpu-apple-official-completion/first-product-batch/receipt.json)。
按原 K32/K128 顺序分块 codec/matmul 后，0.5秒采样的最大单进程 RSS 为
4,404,211,712 bytes（约4.1 GiB）；这是观测值，不是库 allocator 的严格上界。
GLM 原默认 CUDA FMA oracle 与 ATen 独立 multiply/add 边界不同，已在执行前从源码
固定 unfused oracle；未扩大原门。其它三项失败是原 ds4 成熟 roundtrip 未纳入测试允许名单。

源码首次独立链接遗漏已有词表 oracle companion TU，修正消费者后通过；日志保留。
SDK 检查的初次选择器组合与消息换行解析错误也保留，未作为产品数值 FAIL。
[索引](index.json)记录这批原始小文件 SHA；没有归档大型 SDK/DLL 或模型权重。

## 尚未验收的内容

范围对照发现原13项只为 GLM Flash 执行完整吸收式 MLA/NextN。
GLM 5.3 非Flash 的冷 FP8 KV 解码/拆分、反馈与完整 head 已补消费者，尚待补跑。
此前已运行的13项结果不会代签该新增旅程。

WSL 官方 CPU SDK 的版本和 shared closure 已读取；Linux 产品和新增完整片段 P1 尚未通过。
Apple 原极小值 / BF16 中间边界仍未解除；本批没有 Apple 产品修改。
因此当前完整画像账本保持154 RUN_PASS /20 BACKEND_REJECTED /2 RUN_FAIL；
不把上述部分证据宣传为 CPU 四画像或整体 goal 收口。

[统一 LOC](loc.json)为 Compute + Graph 60,866行 /244实现文件，排除测试和第三方源码。
[清理记录](cleanup-obsolete-sdk.json)保存逐项 SHA，删除字节相同的旧 SDK / probe 临时副本，
释放559,220,140 bytes；正式 SDK、来源输入、原归档和证据均保留。
