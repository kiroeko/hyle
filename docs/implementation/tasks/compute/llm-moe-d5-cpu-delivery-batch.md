# D5-A5：CPU 真实片段三交付

<!-- MODULE REVIEW: AUDITED -->

- baseline：`eca5a35f50d47c711e570bbf525fe41e76ff4adc`，Windows canonical；产品 FIX 为零。
- owner：当前 [D5](llm-moe-d5-real-fragment-acceptance.md) 的 A5 消费者验收；不重开已关闭
  CPU/oneDNN owner。若发现真实产品接缝缺陷，先固定精确 owner/FIX，再修改产品。
- scope：同一消费者源码，经源码父工程、installed static、installed shared 三条公开 C++ 路径，
  使用 `hyle::compute` / `hyle::graph` / `hyle::simd`；不使用内部 factor target。
- 输入：本地已验证的 Qwen3.5-2B BF16 权重 fixture，以及 DeepSeek V4 Flash 两份固定 GGUF
  画像的原完整 hash-router/MoE 配方。真实内宽、逐角色编码、现有 oracle/tolerance 不变。
- 每交付四项 Direct + 四项 Graph：Qwen embedding/norm/head 与非法索引失败、完整 FFN、
  causal-conv/GDN 真实状态回送、DeepSeek 两种量化画像的完整 hash-routing/MoE。
  这里 R01 是交付接缝代表消费者；全词表资格仍复用既有 D5 收据，不重新签完整画像。
- F47 的八个官方 CPU 拒绝格保留，不用 GGUF PASS 覆盖原版 FP8/FP4 限制。

## 批量执行、身份与隔离

1. 动态 producer 复用 `.scratch/build-compute-v2-dev`；保持 MSVC Release、WERROR、
   instrumentation OFF、CPU/oneDNN ON、CUDA/Apple/Vulkan OFF。驱动固定 `VSLANG=1033`。
2. 源码 producer 使用既有不可变 `9273bc71` archive；开始前逐文件核对 include/src/cmake/
   third_party 和根 CMake 输入的 Git blob 身份，它们与 baseline 相同。消费者复制当前测试
   TU 及其全部测试头；测试源码变动不冒充产品 candidate。源码/static 只构建一次，
   静态 SDK 安装同一产物；不为各交付重复构建第三方。
3. 每份 installed SDK 搬迁后独立配置消费者；运行时隐藏 producer 的 bin，PATH 仅保留
   Windows/System32。公开 `hyle_stage_runtime_dependencies()` 将所有 HYLE DLL 邻置。
   记录复制文件与 SDK 来源的 SHA256、消费者 compile/link 命令，拒绝依赖 canonical
   产品 include/lib 或 producer 路径。没有 UAC、终端弹窗或 DLL 错误弹窗。
4. 主机参考只复用锁定 ggml-base 和 SIMD，作为格式有效载荷/oracle；所有被测执行通过
   普通 CPU 入口。保持两次以上变化输入和 GDN 状态反馈，非法输入不得发布成功结果。

预算：三种交付各一次配置/低并行构建/串行测试；具体接缝问题最多一次有因纠正，原日志保留。
不跑 P1、不重验其它路线、不扩大算子或改公共验证脚本。适用检查为上述8×3测试、
SDK路径/DLL来源核对及 docs/closure；未运行不记 PASS。成功后保留有效 SDK 和唯一
source/static producer供A6，清理独立 consumer 的可再生构建产物，登记占用/清理量。

结果只签 CPU A5，不签 A6、其它路线或 D6，也不新增完整画像通过格。

## 2026-09-27 本批结果

消费者 `b20fb87c`、产品字节 `9273bc71`：三种交付各8项通过，共24/24、零跳过。
installed 两种交付均完成搬迁、producer运行目录隐藏及隔离PATH验证；动态16个DLL来源逐项一致。
首轮静态安装暴露消费者定向构建未生成所有安装目标；同一producer补齐默认目标后一次纠正通过，
未改产品，也未重跑此前通过的交付。脚本已纳入该默认目标命令。
见[完整证据与原始失败](../../../../evidence/correctness/llm-moe-d5-real-fragments/cpu-delivery/README.md)。
实现仍60,373行/241文件；本批结束，不扩张产品或验证预算。A6与D5整体状态仍未完成。
