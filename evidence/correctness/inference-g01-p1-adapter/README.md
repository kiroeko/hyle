# G01 resident attention 性能适配器预检

产品 candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f` 保持不变。
[receipt](receipt.json) 与 [raw.zip](raw.zip) 保留 Windows adapter 编译、实际 MacBook Air
身份/源/二进制和原 shader hash，以及全部原失败。只有数学和稳定重放验证，无 timing verdict。

F32 BSHD GQA8/2 的 causal Q1/KV1024/D256、causal Q128/KV1024/D256、noncausal
Q17/KV257/D64，普通 Graph 与原 full-F32 shader 独立 recipe 均过原 double oracle 容差。
原 split-K 和长 prefill mask-opt 保留；对照沿同一个 Program/Invocation/output lease 生命周期。
源码：[adapter](../../../bench/inference_resident_sdpa/README.md)。候选首用 scratch 增长单独
记录，之后稳定；原 quota 不覆盖 driver/pipeline 的 private bytes，不能仅据少字节宣称快。

初始 private Program recipe 没填 identity aliases 导致冷 plan_storage 崩溃；macOS crash report
定位后修正适配器，产品未改。原 F32 vec4 block stride 单位错误随后被 padding 哨兵 oracle
捕获，按原 native 源码修正；容差没有放宽。最初错误地在 lazy scratch 首用前记录稳定量，
现单独报告 first-use，再验稳态。所有原 FAIL 保留；lldb 非交互权限失败也保留，改用 crash
报告定位，没有请求改变系统权限。Windows 的旧 private helper/API 名、shared/static 与
DEFER hook、shader 工具位置原失败同样保留，不影响已签 CPU/Vulkan delivery evidence。

目前只签 adapter plumbing/correctness，F16 helper 未资格；统一仪器 contract/A-A/双顺序/
placement/controls/正式配对采样仍待。不能用本证据关闭 G01、runtime owner 或整体 goal。
