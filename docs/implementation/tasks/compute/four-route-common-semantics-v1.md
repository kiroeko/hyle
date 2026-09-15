# R2：四路线共同语义与自然入口

<!-- PROJECT UNIT REVIEW: SCOPED -->

## Closure card

- baseline：`25281c98`；R1 已完成并推送，Windows 为唯一产品来源。
- owner：ComputeOps 的逻辑 schema、Compute 的 Direct/结果/完成、Graph 的逻辑边界与规则绑定。
  各 owner 先绑定本轮已审计的有限子卡；本汇总 SCOPED 不授权产品修改。
- consumer：变化输入的量化 FFN 与原生独立计算；正交 consumer 为非模型的自定义计算与保留结果。
- scope：下表共同机制；各路线自己的 Program、资源与普通默认入口在 R3–R6 连同真实性能验收。
- non-goals：统一所有 native 句柄/执行器、Provider 排序 UI、用户 timing profile、全算子目录、模型跨设备放置。
- 停止：每项固定签名/所有权/成本后才实现；具体反例只改对应项，不重做整体架构调查。

## 有限工作与 owner

| 项 | 本包动作 | 当前入口 |
|---|---|---|
| R2-Q | 完整的有限 grouped Q4 MatMul schema，区别 signed scale 与 unsigned scale+bias，复用物理格式与 shape 校验 | [ComputeOps 卡](compute-ops-v4-quantized-matmul.md) |
| R2-D | Direct 的独立提交模式及生成结果 owner；复用真实 Completion，不能让 Queue 参数被忽略或接纳后丢 owner | Compute 新卡在具体接口审计后绑定 |
| R2-G | 自然 Graph/路线编译边界；清理用户 timing profile 多计划竞赛与模型多路线放置，保留结构/资源规划 | Graph 新卡在定向影响审计后绑定 |
| R2-M | Mem v10 已有 Reservation 的具名平台/失败回收尾项 | 原 [Mem v10 卡](../mem/mem-v10-external-byte-reservation.md)，不扩产品范围 |

R2-Q 是下一步实际产品修改，不等待另外两项重新研究所有历史。R2-D/G 未冻结的签名不能以本表为
实现授权。R3 的 CPU 默认规则、R4/R5/R6 的 native 完整片段仍由各路线资格决定；R2 不虚造赢家。

## 当前定向事实

`operation.cc` 的公开 prepare 排除 delegated，公开 submit 的 shared 假设依赖此约束。解锁时必须
分别固定冷模式和热入口，不能只删除过滤。当前 Completion 可留存一个 execution owner，生成结果
与现有 invocation owner 的组合须保留到真实终态，不能在接纳后以分配失败丢弃工作。

MLX lazy array wrapper 不是 Mem allocation token；生成结果保留原 native owner。CPU/caller output
仍沿真实 Buffer/视图。四路线可以采用不同 Program/slot 类型，不新增万能逐 kernel vtable。

量化输入由逻辑公式、压缩权重物理格式、scale/bias、group 与累加合同共同决定；MLX affine Q4
与 Marlin symmetric W4A16 不因位宽相同而可互换。依据沿用已登记六项研究，不把探针签成路线完成。

## 本包验收与报告

各子卡只运行 selector 选出的适用验证；API/算法固定后才启动预声明不可变候选 P1。Docs/状态不重建。
本包最终复核 caller/generated outputs、Direct 不依赖 Graph、成功/失败/取消/保留结果与槽复用、
自然规则绑定边界和继承的必要依赖尾项。未运行项保持 UNPROVEN，不先签 PASS。

进入 R2 时统一实现 LOC 为 **31,135**（Compute+Graph 全部实现，不含测试）；旧代码不代表 R2 完成。
