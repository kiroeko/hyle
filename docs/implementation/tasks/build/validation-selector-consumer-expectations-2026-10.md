# Selector 消费者闭包测试期望维护

2026-10-15；baseline `d7d1d8d9f`。仅维护测试期望，不修改 selector、组件图、状态或产品。
本次独立工具任务来自 G04/G05 新 vendored owner 数据登记后的必要工具检查。

60 项 selector 测试中 58 项通过，两项固定集合仍停留在联合路线加入前：queue 公共头漏掉
现有 Compute/Graph 路线消费者，Apple runtime 漏掉 graph_route_cpu_apple。把 policy 换回
baseline 后，两项原样失败，已证明新 FlashInfer prefix 不是原因；不将原失败改写为通过。

有限 FIX 只更新这两个显式预期集合，与现有中央真实边一致；保留精确集合断言、owner、
delivery 与未关联 gate 的排除检查。修正后仅重跑这两项，其他58项复用同一轮已通过结果。
原 selector 仍计算实际闭包，没有放宽失败、跳过依赖或增加全局构建。新 FlashInfer vendored
prefix 单独登记到唯一 AUDITED owner，产品计划须 READY 且 unknown_paths/authorization issues 为空。

验收记录与原两项 baseline/current 失败日志保存在 G04/G05 CUDA source 证据的工具补充中；
尚未附入证据前不以本卡冒签工具测试 PASS。此任务不关闭 G04/G05 产品或其工程门。
