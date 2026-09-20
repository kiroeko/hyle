# M1-A Metal factor 的平台配置事实

- owner：`/root`；baseline：`b6aabfc2`；阶段：CLOSED（独立有限工具卡，无产品 factor 重开）。
- trigger：M1-A 新增原版 Metal factor；既有 selector 默认会把未知平台 factor 当成各系统均适用。
- 有限 FIX：仅在现有字段登记 `compute_provider_ggml_metal` 的 Darwin 平台与 Apple=ON 配置；
  将 Apple dependency admission 的既有 CMake helper 归属 Apple route，与本卡资源 owner 对齐；
  vendor patch 沿已登记 `third_party/ggml_runtime/` owner，不新增路径映射或修改选择算法。
- 预算：设计/修改 5 分钟、最多两次纠正；现有工具 suite、Windows/Darwin 计划、docs/closure。
- 不修改 gate 命令、模块授权、依赖闭包或 P1；工具 READY 不代签产品 native/交付。
- 当前记录不解除任何产品冻结或 BLOCKED 条件；全部门完成后停止此工具范围。

## 裁决

有限字段更新已完成：现有 suite 59/59 PASS，Windows 计划 READY 且新 Metal native gate 跳过，
Darwin planning projection READY 且该 gate 被选择/Apple ON。Darwin projection 只使用既有测试
方式投影平台，不签 native；M1-A 自身另有 Air 原生七门和 Windows OFF 三门。
docs、closure、diff 通过，原始 suite/两份计划随
[M1-A 收据](../../../../evidence/correctness/compute-llm-moe-m1-apple-admission/native-air-checkpoint.json)归档。
没有选择算法或公共仪器改动，工具范围停止；产品继续按自己的未完成验收执行。
