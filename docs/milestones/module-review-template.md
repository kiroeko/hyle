# 模块严格复审记录模板

> 复制本页建立 `<module>-module-review-YYYY-MM.md`。方法与门槛只以
> [`module-closure-methodology`](../architecture/module-closure-methodology.md) 为准；本页负责
> 让每轮证据可执行、可追溯，不承担稳定架构或性能点值真相。

## 0. 身份与状态

| 项 | 值 |
|---|---|
| 模块 / review owner | `<module>` / `<owner>` |
| review 类型 | 完整复审 / 受影响边复证 |
| 起点 commit / 旧证书 | `<sha>` / `<doc#section>` |
| 当前状态 | `OPEN/AUDITED/CANDIDATE/EVIDENCE_COMPLETE/SIGNED/REOPENED` |
| candidate commit/tree | `<sha>` / `<tree>`；未冻结写 `none` |
| required CI / push | `<run/url/status>` |

## 1. 审查单元与依赖

| 边界项 | 文件/target/symbol | public/internal | 本轮纳入 | 理由 |
|---|---|---|---|---|
| C++ surface |  |  |  |  |
| C/FFI surface |  |  |  |  |
| private hot kernel |  |  |  |  |
| build/package/export |  |  |  |  |
| direct upstream |  |  |  |  |
| direct downstream / sample |  |  |  |  |

- 语法消费者扫描命令与结果：
- 语义重复/绕过扫描命令与结果：
- 中央 component graph 与真实 target link 是否一致：

## 2. 真实场景、能力与负空间

| 用户任务/内部任务 | 数据与操作形状 | ownership/lifetime | concurrency/failure | 规模/平台 | 当前能力/裁决 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

| 公开能力 | 外部场景 | 内部消费者 | 保留/补齐/迁移/删除 | 证据 |
|---|---|---|---|---|
|  |  |  |  |  |

| 明确不做 | 替代模块/标准库/成熟依赖 | 不做理由 | 重开触发器 |
|---|---|---|---|
|  |  |  |  |

## 3. 成熟库与算法对账

| peer + exact revision | 真实问题 | API/ownership 形状 | 算法/复杂度/元数据 | HYLE 裁决 |
|---|---|---|---|---|
|  |  |  |  |  |

| 非平凡操作 | 正常/最坏复杂度 | 空间/常态元数据 | 进展保证 | 最强 peer | 结论 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 4. 产品形状与易用性

| 旅程 | 最小 recipe/sample | ownership/error/lifetime 是否自明 | 摩擦与裁决 |
|---|---|---|---|
| C++ 默认路径 |  |  |  |
| C++ 高级/调优路径 |  |  |  |
| C++ 相邻模块组合 |  |  |  |
| C 默认路径 |  |  |  |
| C/FFI 高级或 callback 路径 |  |  |  |

| 语言无关能力 | C++ 入口 | C 表达：symbol/opaque/inline/visitor | 缺口或明确不导出理由 |
|---|---|---|---|
|  |  |  |  |

## 5. 正确性、平台与交付矩阵

| 门 | Windows/MSVC | Linux/GCC | macOS/AppleClang | candidate/log |
|---|---|---|---|---|
| Release / Debug |  |  |  |  |
| ASan / UBSan / TSan（适用项） |  |  |  |  |
| pure C / C++ header / surface |  |  |  |  |
| shared installed consumer |  |  |  |  |
| static installed consumer |  |  |  |  |
| source add_subdirectory/bundle |  |  |  |  |
| 特殊 ISA/backend/降级 |  |  |  |  |

## 6. 性能 claim 账本

每行先填写 [`benchmark-evidence-standard`](../performance/benchmark-evidence-standard.md) 的合同卡；
没有资格的比较不得填胜负。

| 正常 workload | peer/同合同 | 规模/QoS | 算法/codegen | Windows exact / SOTA | Linux exact / SOTA | macOS exact / SOTA | 总结 |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  | 五态精确结论 + `PASS/FAIL/UNRESOLVED/N/A` |

- 预声明 margin `m` / SOTA 上界 `U=1+m`（两种判定必须共用，禁止事后改变）：
- 默认序、peer-first、适用 shape 的逐项 exact verdict / SOTA gate 与合成结果：
- 每进程 baseline op/s、rival op/s、ratio raw JSON 日志索引：
- code-shape/order-swap/真实消费者稳定性：
- footprint、tail latency、allocation/lock/syscall 等非均值吞吐成本：
- 所有 exact `LOSE/INCONCLUSIVE` 及 SOTA `FAIL/UNRESOLVED` 的根因与阻塞裁决：

## 7. 文档与红队

- [ ] architecture 只保留稳定职责、类型、依赖与负空间；
- [ ] guide 回答“什么情况用什么”、默认/高级/组合写法与坑；
- [ ] C guide 给出能力映射、所有权、callback/FFI 坑；
- [ ] performance 绑定 workload/peer/harness/runner/compiler/candidate 与原始日志；
- [ ] milestone 记录施工，不冒充当前设计；
- [ ] 方法论第十二节十个红队问题逐条回答，无“未看”。

## 8. 四项 LOC 与最终证书

| 固定口径 | code |
|---|---:|
| 全库 |  |
| 自实现（含测试） |  |
| 自实现（不含测试） |  |
| 当前模块实现 |  |

最终证书必须列出：candidate commit/tree、公开 surface hash/manifest、三平台日志索引、peer revision、
claim-bearing benchmark 与 runner identity、每进程 paired raw JSON、五态 exact verdict 与独立
SOTA gate 的 order/shape 合成、shared/static/source 交付结果、required CI、最终提交与 push。
任何空项或目标 claim 的 gate 未决都保持黄色。
