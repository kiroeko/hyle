# HYLE 实现速查表

> 日常入口。需要完整判据时阅读
> [module closure methodology](../review/module-closure-methodology.md) 与
> [engineering charter](../standards/engineering-charter.md)。

## 写代码前

1. 写 demand/defect ID、真实消费者任务和正常规模。
2. 裁决 owner module、public/internal、直接依赖和不做什么。
3. 查成熟 peer 的能力与算法，不从现有 benchmark 反推 API。
4. 写 ownership、lifetime、concurrency、failure、capacity、复杂度和常态 metadata。
5. 确认该需求不是自然组合、成熟依赖或更高层策略已经覆盖。

## 写代码时

- C++23；公开形状优先 RAII、owner/view、span/range、strong type、`Status/Result`。
- 可恢复失败走 error，不变量才 fail-fast；外来异常只在 adapter 转换。
- OS/CPU/topology/VM/thread 机制走 platform；allocator/placement 走 mem；clock/deadline 走 time。
- HYLE 内部直接使用 C++ API，不绕旧 C bridge。Native FFI 默认 `N/A`。
- 中央 component manifest 是内部依赖源；真实 target edge由它投影，外部依赖由模块一处登记。
- 不增加 benchmark-only public symbol、常态字段、padding、索引或非自然调用顺序。
- 业界常用且改善复杂度档的 tree/index/cache/batch 不能因“怕 benchmark 化”而漏掉。

## 性能判断

1. 先复杂度、copy/allocation/syscall/contention，再少量指令。
2. 冻结 contract、candidate、peer、输入、计费边界和 material margin。
3. A/A、known-delta、timer floor、order/placement、最终 codegen 先证明仪器有资格。
4. consumer/trace replay 签产品影响，microbench 只归因。
5. 改产品需要机制证据与 material 消费影响两把钥匙；稳定 0.5 ns 不够。
6. 结论绑定 capability/contract/hardware/delivery/candidate，不写模块全局 SOTA。

## 写完后

- 正常、边界、OOM/系统失败、move/destroy、并发/取消/关闭按风险测试。
- 跑适用 sanitizer、property/differential/fuzz/stress 和真 ISA/backend。
- 验证 public headers、component-only build、source、installed static、installed shared C++ consumer。
- 扫描上下游是否绕过 error/platform/mem/time 等底座，检查中央图与 package 闭包。
- 更新四类文档各自唯一事实，运行 `python scripts/check-docs.py`。
- 独立红队逐项 `fixed/disproved/accepted/out-of-scope`。
- 运行 `python scripts/loc-report.py --module <name> --self-check`，记录四个 LOC。
- `git status --short`、`git diff --check`、提交真实文件；低 CI 价值的纯文档提交使用 `[skip ci]`。

## 环境提醒

- 完整环境见 [setup](../setup/README.md)，离线见 [offline](../setup/offline.md)。
- Windows、Linux/WSL、macOS 是独立验证环境；WSL 的 Windows 挂载盘不适合重 I/O/压力基准。
- 混合大小核必须固定 affinity/QoS，不能把线程迁移噪声解释成实现差异。
- 远程 Mac/WSL checkout 是验证副本；有价值的工作以 Windows 主仓库为准，除非任务明确另行约定。
