# Apple runtime 暂存脚本的验证 owner 映射

## 有限工具卡

- owner：`/root`；baseline：`ec8cab7f`；状态：CLOSED（只改维护工具，无产品 factor）。
- trigger：[M1-R R5b](../compute/llm-moe-m1-route-binding.md)仅修改Apple暂存函数，当前selector把
  所有`cmake/`文件统一视为全库共享接缝，选择113门。函数只由`compute_route_apple`包含并安装，
  实际反向消费者是`graph_route_apple`，交付属于Compute/Graph。
- 范围：给这一具名CMake文件补已审owner映射，复用既有factor_seam规则、中央反向闭包、
  平台适用性和AUDITED授权检查。不改gate命令、状态协议、P1资格/采样或证据复用规则。
- 非目标：重写selector、扫描并重分类所有cmake文件、缩窄共享构建/组件图的既有门。
- 预算：源码/设计15分钟，最多两次纠正；完成具名回归及原工具suite后停止。

## 固定 FIX

| ID | 修改 | 验证 |
|---|---|---|
| T1 | policy精确登记`cmake/HyleAppleRuntime.cmake`→`compute_route_apple`；不保存第二份依赖图 | Darwin选择Apple两owner与Compute/Graph交付；Windows仍为OFF头/配置而非native PASS |
| T2 | 分类复用factor_seam；无效/未实现owner拒绝，产品变更仍要求事先AUDITED | 已CLOSED owner不得因此获得修改授权；通用cmake修改仍选全库 |
| T3 | 回归、文档及原计划/新计划留痕 | `tests/tooling/validation_selector_test.py`；docs/closure，无C++或P1 |

本工具卡不签M1-R的运行时或交付门。R5b产品修复仍由其原owner持有，未运行的适用门保持未完成。

## 裁决

候选`256c45473f13e7554d318a10b246e6fd18b10959`完成T1–T3。原工具suite的57项中56项通过，
新增Windows夹具漏建header suite导致1项失败；只补夹具目标后，新加3项全部通过，合计覆盖
全部57项。原始失败、专项复核和前后影响计划见
[证据](../../../../evidence/correctness/apple-runtime-selector-owner-v1/validation.json)。
新计划保持Apple两owner、Compute/Graph交付和平台OFF检查；通用共享seam仍全量，CLOSED
owner仍被拒绝，P1不自动启动。docs/closure通过；未运行C++或性能测试，本工具范围P0关闭。

实际不可变候选随后发现T2的历史扫描仍只列旧路径前缀，遗漏新映射的外置CMake文件，故无法
把改动归因至产品提交。撤回上述终审；在原15分钟预算内仅将具名映射路径纳入原rev-list，
强化正常候选READY与真实CLOSED前态拒绝断言。不改变状态门、不跳过原授权。

## 最终裁决（2026-09-20）

本节替代上面撤回的终审。最终工具候选`67f4339e7b8b8dc06c535df00a176e773e4b4263`
通过完整57项原工具suite；历史授权专项5项亦通过。真实R5b不可变候选正确追溯至事先AUDITED的
产品提交，保持Apple owner/直接消费者/三交付/OFF边界；实际选中7门全部通过。
本工具卡只签T1–T3，不代签M1-R运行时；原夹具失败和遗漏历史路径导致的BLOCKED计划均保留。
没有修改原门命令、状态协议、P1或证据复用合同，P0范围关闭。后续只有具名owner映射或授权
追溯的可复现错误才重开，不扩大为全库CMake重分类。
