# F42：原版 DeepSeek 激活组量化还原接线

候选 `fd1a9ac4e6bb25bf2a2347e54b5069e06624e1a9`，Windows RTX4070 / SM89 /
CUDA Driver API13040。原版 DeepSeek V4 的 FP8/group64 与 FP4/group32 量化再还原程序，
已接入默认 CUDA Provider 和普通 Direct/Graph。**本次不增加完整模型画像格**，ledger仍144/176。

## 已执行与边界

- 四个 CTest factor 共9项 GTest 全通过、零SKIP；四项公开头独立检查通过。最终增量构建4.88秒、
  测试5.22秒；首次批量构建77.55秒。普通两入口各运行两种格式×M1/3/33×两次变化输入，
  使用实际448维KV前缀和128维indexer宽度，输出逐值精确，不放宽数值门。
- 原生24案例包含全部有限 BF16 编码、舍入中点、signed zero/tiny、跨32组尾块、输出和scale guard；
  结果和组scale逐位匹配独立表定义参考。极大有限值经量化后可能溢出，与原版一致。
- shape/host/alignment/alias/workspace拒绝、准备与invocation元数据分配失败回滚、双队列独立
  workspace、预留完成槽后零元数据分配重放，以及释放程序/invocation并关闭Context后的Completion保活通过。
- 两份原版程序离线重建 SHA256 一致：FP8 `4f8d258a26545629468c12b2a80ce771b0cc3bcacb886da77f7443b832aa7aa1`；
  FP4 `191baf3fdd63c3d7e250bcddcd15d702c1f2c3681a9636582e7901696b5a97ac`。
  CUDA运行无Python/TileLang依赖；没有新增HYLE量化数学。
- HF来源声明工具4项测试（含10种拒绝变体）、13项模型输入工具与59项selector工具测试通过；
  实际CUDA配置与全量当前vendor身份审计通过。HF只允许固定revision的in-tree源码，不进入下载器。

[机器收据](receipt.json)保留命令、原始日志hash、源码/重建记录与适用范围。
原生复核可先按vendor的`rebuild.py`生成两份cubin，再运行本目录
`native-probe.py --artifacts <生成目录>`；它仍只签原生程序资格。

## 成本、初始失败和未完成项

每次热执行一次launch，workspace为4×总组数字节；原版写出的scale作为独立invocation临时空间，
动态shared分别4096/2048字节。输入/输出16字节对齐，不需要host同步、权重展开或运行时编译。
原版程序按一行一组编译，动态行数表达任意整组矩阵；没有按模型名称复制程序。

首轮Direct/Graph和原生数学已通过，资源夹具却遗漏caller workspace，正确被API拒绝。
夹具改为两份独立device workspace后全批通过；产品未改，原[失败日志](initial-workspace-tests.log.gz)保留。

DeepSeek官方的完整attention/compressor/MoE/MTP接合仍需完成；剩余28个官方FP8格与4个
DeepSeek GGUF Vulkan格、D5-A5交付/所选门、D5-A6性能及D6不因本次原子资格而写PASS。
本卡不重跑P1。统一[实现统计](loc.json)为 **59,899行／238文件**，不含测试和第三方生成程序。
