# 运行库收集器菱形依赖有限修复

<!-- MODULE REVIEW: AUDITED -->

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
