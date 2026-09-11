# Compute / Graph CPU one-op 性能证据

本页只裁决 2026-09-12 的 CPU one-op 纵切，不声称完整 Compute/Graph 或外部 backend 已完成。
冻结合同是
[`compute-graph-one-op-tax-p1-v1.contract.json`](compute-graph-one-op-tax-p1-v1.contract.json)（v2 已从活动
benchmark 目录迁入历史证据），
候选产品字节为 `5c9dc1604e6daaec1d73062296f2a71d019e7b35`；其中相对前一提交只改变合同
peer 身份的提交不改变产品实现。

## 比较边界

两条测量构成同一 4096 元素 f32 RMSNorm 的抽象成本阶梯：

1. `Mem map + HYLE SIMD` 是低层手工路线；`PreparedOperation` 在相同 Buffer、输出地址、SIMD
   算法、精度和同步边界上增加 tensor description、alias、Context/Queue、resource 与 Completion
   合同。这不是同语义 SOTA 对手，因此合同明确记为 `NOT_COMPARABLE/diagnostic`。
2. `Graph Invocation` 与其 peer 使用同一个 `PreparedOperation`、Queue、TensorView、Buffer 和
   Completion；compile/create/bind 都在测量外，只增加固定 binding-mask 与 plan dispatch。这一行是
   `COMPARABLE/product`。

没有用裸指针核冒充 Compute 完整语义，也没有把 graph compile 或 bind 塞给其中一侧。两种执行顺序和
两种函数 placement 都被冻结；输入、输出 storage 与执行线程完全相同。

## P0 成本账单

| 路线 | 复杂度 | steady allocation / search / JIT | steady 同步与系统动作 |
|---|---|---|---|
| direct Mem + SIMD | O(n) | 0 / 0 / 0 | 两次 host map/unmap；system resource 不 flush |
| Compute prepared | O(rank+n)，本行 rank=1 | 0 / 0 / 0 | 同上，另有固定 descriptor/alias/context/queue/resource 检查与 terminal Completion |
| Graph invocation | O(bindings+rank+n)，本行 bindings=2 | 0 / 0 / 0 | 固定 mask 后直接提交 retained PreparedOperation；不读 logical Graph、不遍历、不重新 prepare |

preflight 前通过源码检查修掉两处非必要成本：便捷 submit 改为借用 view 且不再复制临时数组；一次
submit 只读取一份 resource properties 快照；`TensorDesc` 相等判断只检查有效 rank。公开安全语义没有
减少，也没有新增 benchmark-only fast path。

## W1-WIN P1

`qualify-platform` 在 MSVC 19.50、Release、static、LTO off、portable ISA 下取得
`qualification_gate=PASS` 与 `performance_gate=PASS`：

- Graph/Compute 的四个 paired median ratio 为 `0.9879`--`0.9971`；四个 99.26% CI 均完整落在
  5% 等价带，combined verdict=`PARITY`、non-inferiority=`PASS`。
- Compute/direct 低层 diagnostic 的绝对中位数约 `591.1`--`594.8 ns` 对
  `561.3`--`571.2 ns`，差 `22.2`--`29.8 ns`。该固定差由上表额外安全合同解释；因为两侧语义不同，
  不生成 WIN/PARITY/LOSS verdict，也不授权删除检查。
- A/A 两顺序为 `PARITY`；5% 与 10% known-delta、timer floor 和 placement controls 全部 `PASS`。
  产品 workload 高于保守 timer floor 约 2,845--2,983 倍。

原始 bundle 保存在
[`compute-graph-one-op-tax-p1-v1-5c9dc160-w1-win`](../../../../evidence/performance/compute-graph-one-op-tax-p1-v1-5c9dc160-w1-win/)；
contract / manifest / summary / controls / observations SHA-256 分别为：

```text
af110c0232f0c0c3f4d97225fa8fd803a50d76faae63de25ee934d29a8265ef6
1ba5a56dc87ebb364e829c57ad1f3268ab6c05c1ba3f6ee198eab354e6317aae
b3c4f201e6d40e6fbbf7a16315680a207b008aac955a89a0b45bf878b9b53bbe
7d0eb38001c9e5cce8133091bb7c2248c0df019d07fbfb9d43a7682407c4995f
69af00da97b9725e09440b829ed7b97db641a83ab82dc15830ca5b45f89de6ba
```

严格重放命令返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`：

```powershell
python scripts/hyle-perf.py verify `
  --contract docs/implementation/evidence/performance/compute-graph-one-op-tax-p1-v1.contract.json `
  --manifest evidence/performance/compute-graph-one-op-tax-p1-v1-5c9dc160-w1-win/manifest.json
```

本轮结论只证明：正常 one-op 行上 Graph 组合税没有 material 固定劣化，Compute 安全层有约几十纳秒、
已经分解且不会被伪装成同合同胜负的固定成本。后续 multi-op、provider、GPU 与端到端推理必须建立各自
合同，不能继承本页 verdict。
