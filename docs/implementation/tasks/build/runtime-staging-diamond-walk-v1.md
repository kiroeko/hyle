# 运行库收集器菱形依赖有限修复

<!-- MODULE REVIEW: CLOSED -->

- owner：构建/runtime staging工具；CPU最终F1的独立工具任务，不是算子实现或公共性能仪器重写。
- baseline `4174d383`；真实consumer为CPU+Vulkan标准产品、原LibTorch/IREE runtime装载闭包。
- trigger：真实configure进程70284持续执行、停在tests/runtime staging前；现walker仅防祖先循环，
  同一目标经重复/菱形路径反复遍历。七层22节点最小图已运行，源码复杂度随路径数而非依赖图规模增长。
- 有限FIX：每次独立收集按(target,完整condition)去重，保留不同conditional路径及原祖先循环防护；
  相同无条件链接/常量LINK_ONLY边去重；不缓存跨configure/跨后续修改结果。
- 不改执行队列、DLL集合含义、P1统计、selector/closure；不把旧等待改为PASS。
- 验收：plain diamond、duplicate LINK/INTERFACE、alias、config/false条件、循环和多个独立collect；
  原注册runtime及Windows DLL邻置/Linux RPATH合同继续，Mac RPATH扩展仍归F3单独audit。
- 预算：一批函数修正/一批定向纠正；只修线性遍历并保留条件，不扩第二套依赖图/构建系统。

允许停止已核对exe/argv的本轮configure以使用修正，原中断保留；不因观察超时重启其它任务。

## 已执行的工具结果

同一十层菱形图由4093次路径遍历减至41次target/condition访问，资产集合仍为一个；
Debug/Release的条件、重复、wrapper、alias、循环与独立collect检查通过。
真实CPU+Vulkan configure52.297秒通过，原976.375秒中断仍不签PASS。
[原始trace与receipt](../../../../evidence/correctness/runtime-staging-diamond/README.md)已归档。
本页不签CPU三交付或Darwin loader资格，owner仍只负责这项有界构建工具纠正。

## CLOSED：有限工具范围终审

候选 `61ad57fb6c054e8b1b46844e8aea75dbbdbc8948`；当前 `e0388fbda` 的collector函数与该候选逐字节
规范化对账相同，SHA256 `ce0a7f714570d5ee093473b52081ae52320a91a3491b3e52e36f6f5a4df14029`。
原trace归档SHA及全部成员已复核，无新工具修改，因此复用原Debug/Release和真实mixed configure证据，
不因本次文档终审重新构建或测试。

| 七门 | 具名结论 |
|---|---|
| scope | 仅本轮实际依赖菱形/重复遍历缺陷，无第二构建系统或性能仪器改动 |
| architecture | 独立collect重置，访问键保留target与完整condition；不缓存跨调用结果 |
| correctness | 菱形、重复、alias、常量wrapper、不同条件、循环、独立collect的原Debug/Release检查通过 |
| delivery | 纯CMake构建工具；真实Windows CPU+Vulkan configure通过，不制造binary journey |
| usability_docs | 原调用方式和runtime资产含义不变；原中断与本轮证据边界具名记录 |
| red_team | 相同资产结果与完整条件反例已核对；当前collector源身份与归档一致 |
| engineering_performance | P0：同十层输入4093次访问降至41次，资产数仍为1；没有P1或算子赢家结论 |

本工具任务已完成；F1混合SDK/预编译和CPU最终goal仍由各自施工卡验收。
reopen仅限可复现资产丢失、条件错误、循环/重复访问回归或调用合同变更，不能因旧卡继续扩建工具。
