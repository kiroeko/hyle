# M1-V 进行中证据

当前不是 terminal record，新增模型完整覆盖仍为 **0/256**。

- `qvk-first-candidate.json`：IREE 单独 packed-FP4 候选失格、部分数值结果、未运行项及具名后续。
- `qvk-first-candidate-raw.zip`：原样保留编译/运行日志、原输入/oracle、失败 VMFB、工具版本与
  下载 hash、上游源码路径和 probe。原始首次失败 JSON 的 NaN 指标仅作为 raw 保存；后续记录
  使用严格 JSON。compiler attempt 6 的 exit 0 伴随 pass 解析错误，未计为有效编译。

IREE/LLVM 工具按其原版标准操作生成程序；没有新写 HYLE 数学。IREE pin 为
`6ef606b6fa3e12bd2c12d36a07463069eb71185f`（Apache-2.0 WITH LLVM-exception），
独立 LLVM 工具为官方 apt.llvm.org 的 23.1.2 `4b1925210476` 构建，包 hash 在 raw 中。
ggml `456172ec`（MIT）、ncnn `70e7bfd6`（BSD-3-Clause）、MNN `397f8533`（Apache-2.0）
的接缝审计不代替实际资格。下一项同设备组合仍未通过，未签 P1/SOTA。

源码与工具都在本卡具名 scratch 内；产品只以 Windows checkout 为真相。资格结束后归档并
清理下载/展开的工具及验证副本，保留项目共享依赖 cache 与模型。
