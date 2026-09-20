# M1-V 原版 ggml/IREE 接入验收

当前候选为 `0e5b276a`，本包有限接入的实际验证已完成；五owner已按独立CANDIDATE检查点后CLOSED。新增模型完整覆盖仍为 **0/256**。此前检查点的pending仅指当时状态。

- `qvk-first-candidate.json`：IREE 单独 packed-FP4 候选失格、部分数值结果、未运行项及具名后续。
- `qvk-first-candidate-raw.zip`：原样保留编译/运行日志、原输入/oracle、失败 VMFB、工具版本与
  下载 hash、上游源码路径和 probe。原始首次失败 JSON 的 NaN 指标仅作为 raw 保存；后续记录
  使用严格 JSON。compiler attempt 6 的 exit 0 伴随 pass 解析错误，未计为有效编译。

IREE/LLVM 工具按其原版标准操作生成程序；没有新写 HYLE 数学。IREE pin 为
`6ef606b6fa3e12bd2c12d36a07463069eb71185f`（Apache-2.0 WITH LLVM-exception），
独立 LLVM 工具为官方 apt.llvm.org 的 23.1.2 `4b1925210476` 构建，包 hash 在 raw 中。
ggml `456172ec`（MIT）、ncnn `70e7bfd6`（BSD-3-Clause）、MNN `397f8533`（Apache-2.0）
的接缝审计不代替实际资格。首次 packed-FP4 失格结论保持，不签 P1/SOTA。

- `native-progress.json` / `native-progress-raw.zip`：普通 recurrent 产品纵切、两台机器的
  原版 ggml MXFP4，以及 IREE 参数化数值/C++ static/shared 原生资源资格；十六数值样例和
  两请求在途/失败/旧结果通过。仍不代签普通 IREE 产品、checked-overflow Scan、混合队列
  同步和产品三交付；聚合接入 INCONCLUSIVE。
- `native-progress-loc.json`：统一脚本的全部 Compute+Graph 实现统计，51,953 行 / 222 文件，
  排除测试与 vendor。资格程序没有计入产品实现或覆盖。
- `checked-queue-qualification.json` / `checked-queue-qualification-raw.zip`：补齐两台机器各 12 例
  checked Scan 与单队列自动续执行资格；两请求 repeat3、失败跳过后段和 C++ static/shared
  消费通过。私有资源 overlay 的首轮失败与纠正全部保留。真实 HYLE/ggml/IREE 普通接入和产品
  三交付仍待验收，产品聚合仍为 INCONCLUSIVE，不能以这份资格代签。

源码与工具都在本卡具名 scratch 内；产品只以 Windows checkout 为真相。资格结束后归档并
清理下载/展开的工具及验证副本，保留项目共享依赖 cache 与模型。


## 最终证据导航

- `mxfp4-product.json`：普通具名MXFP4描述、Direct/Graph与两机资源/数值检查点。
- `iree-native-checkpoint.json`：独立原版runtime、实际queue/feature/quota与原版异步程序；POSIX修复。
- `iree-ordinary-checkpoint.json`：普通Scan/FP8、混合链与取消/失败/寿命；IOCP修复及真正原版C ASan。
- `final-prerequisites.json` / `final-prerequisites-raw.zip`：Windows/Air六格普通source/static/shared，
  WSL/Mini/Windows OFF、实际UBSan与flag visibility反例；所有首次fixture/环境失败保留。
- `final-acceptance.json` / `final-selected-raw.zip`：120个selected PASS、13个skipped原因、SM89
  硬件上下文与实际编译配置、原七架构中断记录、P0/有限复核、身份复用及清理收据。
- `final-loc.json`：统一脚本53484行/232个Compute+Graph实现文件，排除测试/vendor。

原版参数化recipe与资源overlay没有新增HYLE数学。此包P0不签P1/SOTA或全部模型支持；
所有冻结自研保持原样，旧ggml CPU/oneDNN阻塞不由此解除。进入M5前停下等待用户。
