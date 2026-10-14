# TEST_DEPS 修改的验证选择反例

状态：工具问题待处理；不授权在模块任务内重写 selector。2026-10-15。

G04/G05 CE12 的实际共享链接失败证明 compute_provider_ggml 测试需要 CPU route fixture。
有限修正仅在 HyleComponents.cmake 增加该 owner 的 TEST_DEPS compute_route_cpu，并在
测试 target 链接已声明的 compute_batch/compute_route_cpu；所有产品依赖、公开头与交付图不变。

`.scratch/inference-gaps/g04-g05/ce12-validation-plan-final.json` 为 READY，但因中央声明文件
改变选择全部 factor/header/delivery；原仅产品实现的 plan 正确选择 provider 与 CPU/Graph CPU
两直接消费者。当前卡按实际影响执行这三个 factor、原 Batch/Graph 公共消费者、三宿主风险及
两 CPU SDK 配置，保留两份原计划，不以全库重编修正工具选择过宽。

工具后续应辨别仅 TEST_DEPS 的已解析中央声明变化，仍执行实际测试依赖解析/组件合同，并
保留真实产品边、未知变更的原保守策略。修复须有独立工具范围与反例，不由本卡改公共核心。
恢复条件：独立工具改动能对同一 diff 给出精确 gates，或明确支持具名证据的人工影响裁决。
