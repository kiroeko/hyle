# F41：GLM-5.3 官方 FP8 的 CUDA 完整片段

不可变候选 `bcea6bce8c7f33d224a08d112e017f66150b2b27`，Windows RTX4070 / SM89，
CUDA Driver API13040。普通 Direct/Graph 的完整必需 region 已齐备，新增 **2 个完整画像格**；
ledger 为 **144/176 RUN_PASS、32/176 VALIDATION_NOT_RUN**。
这是片段正确性资格，不代签其它路线、三交付、P1 或整模型推理。

## 实际执行

- DeepSeek 固定原版 `weight_dequant_kernel` 一次解码 FP8 `[28672,512]`，既有 View/Copy
  从同一矩阵得到 BF16 K `[64,512,192]` 和 V `[64,256,512]`。两份缓存逐值精确检查；
  没有上传 host 参考答案来制造缓存。
- Direct 使用普通 Compute 调用；Graph 使用独立的冷准备图。完成后释放原 FP8 设备源、完整
  解码临时矩阵和冷程序，只保留真实设备缓存。热程序没有解码节点。
- 完整 MLA 四轮实际 cache 反馈；NextN 的 EH→MLA→残差/norm→MoE→最终 norm 四轮
  实际 hidden/cache 反馈，再将实际 hidden 送入全部154880词表 head。参考轨迹独立推进。
- 官方 GLM/Flash 混合片段与四个 GGUF NextN/full-head 消费者在两个入口回归通过；
  未改变既定数值门，未新增 HYLE 数学。
- 非本地巨型模型采用格式合法、非零的合成权重，保持实际 hidden、query/KV rank、head、
  block128 和每个权重角色的 FP8/BF16/F32 格式。

[机器收据](receipt.json)保留逐 region 对应证据及原始日志摘要。四个 CTest factor 中
**11 个 GTest 全通过、零 SKIP**；构建12.00秒，测试39.36秒。四个公开头独立检查通过。
原生16案例、两份 AOT 程序重建哈希一致、13项模型输入工具检查和59项 selector 工具测试通过。

| region | 本次或复用证据 |
|---|---|
| R01 | 已验词表 Gather/越界；本次真实 NextN hidden→完整 head |
| R02–R03 | H6144/I12288 FFN、实际 FP8 角色投影及 BF16 参数；混合片段重跑 |
| R07 | 一次性 KV 解码/分离，完整 absorbed MLA、RoPE、四轮实际 cache |
| R08–R09 | FP8 indexer→TopK2048→稀疏注意力，实际索引 carry/reset；混合片段重跑 |
| R10–R11 | 原 router→grouped FP8 experts→combine/shared；完整 NextN 中再次消费 |
| R13 | 完整 NextN 与全部154880 head，四轮实际 hidden/cache |
| R14–R15 | 复用未变的动态/专家资格；本次解码、资源拒绝、并发及 Completion 保活 |

## 资源与成本

全部256种 FP8 编码、NaN、signed zero、BF16/F32 输出、尾块及输出 guard 已验。
元数据逐分配 OOM 回滚、host/alias/alignment/native index/grid 拒绝、双队列独立 invocation、
释放 Prepared/Invocation 并关闭 Context 后的 Completion 保活、预留完成槽后的零元数据分配
重复提交全部通过。原生程序资格仍限 SM89/Driver13040，不外推其它硬件或驱动。

本 consumer 的 FP8 设备源14,683,648字节，完整 BF16 临时矩阵29,360,128字节，
最终 K/V 合计29,360,128字节。Direct 显式 payload 峰值73,403,904字节，另计原生程序和
allocator 元数据。冷阶段完成后只保留 K/V；没有每 token 解码。实际状态和词表分片仍有
host 控制、读回和上传边界，其成本不能从后续性能比较中排除。

## 失败记录与未完成项

初次普通解码入口已通过，资源夹具却对已被 public view 拒绝的错位输入调用 `.value()`，
导致测试异常。修正为在正确边界检查拒绝，产品未改；[原始失败日志](initial-native-tests.log.gz)保留。

首次重新 configure 还发现 Python cache 落入 vendor 目录，以及旧 SGLang lock 摘要与 HEAD
提交的源码不一致。移除临时 bytecode，新离线脚本禁止向源码写 bytecode；逐文件确认 SGLang
工作树与 Git 对象一致后修正摘要，未修改旧数学。两个原失败日志与[逐文件核对](sglang-lock-correction.json.gz)保留。

其余 **28个官方 FP8格、4个 DeepSeek GGUF Vulkan格**、D5-A5 三交付与其余所选门、
D5-A6 P1、D6 尚未完成。广域 selector plan 的所选门不因局部测试通过而写 PASS。
统一[实现统计](loc.json)为 **59,829行／238文件**，不含测试和第三方生成程序。
