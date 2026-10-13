# 独立工具缺口：共享 benchmark 目录归属

状态：待工具轮定界，2026-10-13；本记录不授权修改公共工具或产品。
baseline `702b6fed97ba1fe7dd2c4d80e992aeb7423f92d3`。

`validation-select.py` 对既有 `bench/inference_resident_sdpa/cpu*` 与新增
`cpu_p1` 返回 `unknown_paths`，原因是共享目录不是 component 名；实际仅修改 CPU
比较适配器，产品 owner 仍为原 CANDIDATE，product_authorization 没有违规项。
原计划 `.scratch/inference-gaps/g01/cpu-p1-adapter-impact-plan.json` 的 BLOCKED 保留。

后续独立工具轮应裁决共享 benchmark 的唯一 owner 与明确验证入口；不得把任意 bench
文件默认当产品或无条件跳过。当前模块不改 selector/policy，按 G01 已预声明的独立
CPU adapter 编译、oracle、A/A 和正式原仪器合同取证，docs/closure照常。此手工适用性
记录不改写 selector 为 PASS，也不授权扩展产品实现或重跑全库。
