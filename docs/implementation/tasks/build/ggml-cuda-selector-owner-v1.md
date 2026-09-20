# M1-C新Provider与vendor输入的selector登记

## 有限工具卡

- owner：`/root`；baseline：`6ef8884d`；阶段：AUDITED，公共工具元数据，无产品factor重开。
- trigger：M1-C的`third_party/ggml_cuda_runtime/{LICENSE,patches.json}`被既有selector准确列为
  unknown input，阻止具名候选裁决。三个已登记CUDA Provider也需要声明实际平台/配置事实。
- 审计：既有`vendored_input_owners`、`factor_configure_cache`和`factor_supported_systems`
  已能表达这些事实；不需要修改selector算法、执行命令、依赖图、P1或状态授权规则。
- 预算：源码/设计10分钟，至多两次纠正；只补下表，运行原工具suite及当前计划后停止。

| FIX | 固定元数据 | 验证 |
|---|---|---|
| T1 | ggml CUDA patch目录→`compute_ggml_runtime`，与实际CMake/lock owner一致 | unknown清空，owner仍须AUDITED |
| T2 | 三个M1-C新Provider按既有字段配置CUDA；ggml为Windows/Linux，FlashInfer仅Linux | 平台不适用仍为OFF/skip，不能签native PASS |
| T3 | 留存前后计划，运行既有selector工具suite、docs/closure | 不构建/重跑P1，不代签M1-C native/交付 |

仅登记输入归属与已审平台事实，不缩小中央组件图/依赖锁变更的既有影响闭包；尚无产品实现
或native收据的ggml factor不会因这张映射卡成为已完成路线。
