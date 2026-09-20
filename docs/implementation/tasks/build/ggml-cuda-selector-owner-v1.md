# M1-C新Provider与vendor输入的selector登记

## 有限工具卡

- owner：`/root`；baseline：`6ef8884d`；阶段：CLOSED，公共工具有限修正，无产品factor重开。
- trigger：M1-C的`third_party/ggml_cuda_runtime/{LICENSE,patches.json}`被既有selector准确列为
  unknown input，阻止具名候选裁决。三个已登记CUDA Provider也需要声明实际平台/配置事实。
- 重新定界：既有字段能表达这些事实，但实际Windows计划及57项原suite中10项复现同一冲突：
  Linux-only FlashInfer将共享CUDA开关设为OFF，与Windows可用CUDA factor所需ON冲突。
  T2b只修正这个配置合并前提；执行命令、依赖闭包、P1与状态授权规则保持原合同。
- 预算：仍为源码/设计10分钟，至多两次纠正；完成下表与原工具suite后停止，不重置预算。

| FIX | 固定元数据 | 验证 |
|---|---|---|
| T1 | ggml CUDA patch目录→`compute_ggml_runtime`，与实际CMake/lock owner一致 | unknown清空，owner仍须AUDITED |
| T2 | 三个M1-C新Provider按既有字段配置CUDA；ggml为Windows/Linux，FlashInfer仅Linux | 平台不适用仍skip native，不能签PASS |
| T2b | 不适用factor的OFF只作为共享开关的缺省值；适用factor的显式要求优先 | Windows保留CUDA ON且FI native跳过；全不适用仍OFF；适用要求彼此冲突仍拒绝 |
| T3 | 留存前后计划，运行既有selector工具suite、docs/closure | 不构建/重跑P1，不代签M1-C native/交付 |

仅修正输入归属、已审平台事实与具名共享开关冲突，不缩小中央组件图/依赖锁变更的既有影响闭包；尚无产品实现
或native收据的ggml factor不会因这张映射卡成为已完成路线。

## 裁决（2026-09-20）

T1–T3及重新审计的T2b完成。最终完整59项工具suite通过；当前M1-C计划为READY，unknown为空，
产品owner仍须AUDITED。Windows/Linux保留适用CUDA原生门，Windows的FlashInfer仍跳过native；
Darwin全CUDA OFF，头边界独立。适用Provider彼此要求冲突仍拒绝，CLOSED vendor owner仍拒绝。
原57项中10项同因失败与修正后的计划/日志均保留在
[收据](../../../../evidence/correctness/ggml-cuda-selector-owner-v1/validation.json)。
源码/设计耗用8分钟，一次根因修正；docs/closure通过。未以工具卡运行C++或P1；READY不是产品门PASS。
本有限工具范围停止，只有具名输入归属或共享开关的可复现错误才重开。
