# Compute / Graph CPU 性能证据

本页分别裁决 2026-09-12 的 CPU one-op 抽象税与 x64 oneDNN provider 边界，不声称完整 Compute/Graph、
完整算子目录或 GPU backend 已完成。one-op 冻结合同是
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

## x64 oneDNN provider P1

[`compute-onednn-provider-p1-v1`](../../../../bench/contracts/compute-onednn-provider-p1-v1.json) 不把
HYLE native kernel 和 oneDNN 算法混成一场胜负。两侧都是 oneDNN v3.13.2 同参数、预生成的 x64 f32
BRGEMM ukernel，使用独立但同值的 64-byte-aligned caller storage、相同 caller scratch、串行 full-N
执行、overwrite output 和 hardware-context set/release。candidate 额外且完整计入普通用户真正支付的
TensorView/resource/alias/Queue 检查、host mapping、provider dispatch、Status 与 Completion；prepare、JIT、
allocation 和 warmup 对两侧都在测量外。shape 为真实 short-prefill projection：
`A[32,4096] * B[4096,4096] -> C[32,4096]`。

v4 把同一 kernel mechanism 抽到未安装的内部 seam，产品实现检查点为 `6306ab2f`；性能合同补齐
benchmark 直接使用的 CPU backend factor 后，clean source 为 `f130c42e`。W1-WIN、MSVC 19.50、
Release、static、LTO off 的 engineering qualification 在 n=15 按规则停止：

- 八个执行顺序 × 构造顺序 × placement strata 的 candidate/peer median ratio 为
  `0.9904--1.0207`；八行 non-inferiority 全部 `PASS`；
- 七行 exact verdict 为 `PARITY`；一行的 99.26% CI 为 `0.9751--1.0534`，跨越等价带上界，诚实记为
  `INCONCLUSIVE`，因此 combined exact verdict 也是 `INCONCLUSIVE`，而不是把它改名为平局或胜利；
- A/A、5%/10% known-delta、timer-floor、placement、artifact identity 和 build contract 全部 PASS，
  candidate/peer/host/artifact 在运行前后保持稳定；
- 一个短 probe 在计时前发现旧合同只选择 provider factor，却漏选 benchmark 自身直接调用的 CPU
  backend；合同只补 `compute_backend_cpu` maintenance root，没有给 shipping provider 增加产品依赖。

原始 bundle 保存在
[`compute-onednn-provider-p1-v1-f130c42e-w1-win`](../../../../evidence/performance/compute-onednn-provider-p1-v1-f130c42e-w1-win/)；
contract / manifest / summary / controls / observations SHA-256 分别为：

```text
9a928c1e86dda310a1981be9a384c0a3fd9d13a28873188a5af3649a58c86635
55bf11e85429215174e2d947b0a7e27a7c99d0b59e81902be89b7f7dde367be3
658436546f53d1b993269213174fa5815b1197e537f10fff2e5c7328121be10d
3618f964f970a7297ba1dfc8e4bb74c2b166ce4d06c04384531813896686754a
0b5eb7237650ae846749fc6200a78404c4aed958c5f0cc26b1f2f1a922ac5b51
```

严格重放命令返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/compute-onednn-provider-p1-v1.json `
  --manifest evidence/performance/compute-onednn-provider-p1-v1-f130c42e-w1-win/manifest.json
```

本行只证明完整 HYLE oneDNN warm provider 边界在该 x64 serial prefill 合同上没有 material 税；decode
由 HYLE native 路由、并行 executor、其他 shape/dtype、region partition 和 GPU provider 都必须独立裁决。

## Graph v2 三节点 composition-tax P1

[`graph-multi-op-tax-p1-v1`](../../../../bench/contracts/graph-multi-op-tax-p1-v1.json) 测量公开 Graph
真正承诺的组合成本，而不重新裁决三个 leaf kernel。candidate 是预编译、预绑定的三节点
RMSNorm → transpose-view MatMul → packed SiluAndMul `Invocation`；peer 是完全相同三类 HYLE-native
`PreparedOperation` 在同一 Context/Queue、TensorDesc、输入/权重和同步 Completion 边界上的手写 direct
sequence。Graph 每个 Invocation 拥有 Mem lifetime workspace，direct caller 拥有等价的两个预分配中间
Buffer；两者的分配、首次触碰、prepare、compile/create/bind 和 correctness warmup 均在计时外，且独立
交叉 execution order、allocation order 与函数 placement。这是同一用户任务的 ownership 差异，而不是
伪称同一地址。

W1-WIN、MSVC 19.50、Release、static、portable ISA、LTO off 的 engineering qualification 在 `n=15`
按规则停止：

- decode `1x256x512` 的八个 median ratio 为 `0.993344--1.014139`，八行均为
  `PARITY`、non-inferiority=`PASS`，99.26% CI 总范围 `0.968427--1.037414`；
- anti-bias small-batch `4x64x128` 的八个 ratio 为 `0.993931--1.004974`，八行均为
  `PARITY/PASS`，CI 总范围 `0.969289--1.045389`；
- A/A、5%/10% known-delta、timer-floor、placement、fresh build、artifact identity、输入与 host stability
  全部 PASS；没有触发 `n=31`；
- 一个 placement/candidate-first/peer-allocation-first 行有极窄的正向 CI，其余方向跨顺序或覆盖 1，故
  只签发 combined practical `PARITY` 和 5% non-inferiority，不宣称稳定方向性 WIN。

首次完整 diagnostic 虽然产品与 controls 全过，但因调用者把输出写进未忽略源码树并错误要求未预钉死的
formal toolchain，runner 正确 fail closed；它没有进入样本补充或产品裁决。有效 bundle 保存于
[`graph-multi-op-tax-p1-v1-eb740c2e-w1-win`](../../../../evidence/performance/graph-multi-op-tax-p1-v1-eb740c2e-w1-win/)；
contract-lock / manifest / summary / controls / observations SHA-256 分别为：

```text
fc7a512e5dc724a3d9ec835d2fc41eb13c77da78bc23207350e727a83e37e10c
06a47f0ddd0258b6fde66c41b3ae576a7ca46bc7a3fe4e034519da5ea0b38eee
23be70ccbc66d62957b7936d5d5b78c160eae1f89c86661ccdfdc7321101f89f
beae4c0e84afbb599170a601c1d6b150b70a216d7c5a94cb62006b0cb457e506
e85e754985086eecb381e9e110f1b114eac2aa8895c6a9ff75464a3ed6f5722f
```

严格重放命令返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/graph-multi-op-tax-p1-v1.json `
  --manifest evidence/performance/graph-multi-op-tax-p1-v1-eb740c2e-w1-win/manifest.json
```

本行证明当前 CPU 三节点纵切的 Graph steady composition tax 没有 material 固定劣化；它不证明完整
operator catalog、provider partition、异步/GPU graph 或端到端 Transformer 已完成。

## Compute v2 shipping CPU route P1

[`compute-cpu-route-p1-v2`](../../../../bench/contracts/compute-cpu-route-p1-v2.json) 验证用户实际得到的
CPU route selection，而不是把完整 Compute API 与裸 kernel 作苹果/橘子式排名。candidate 是启用已关闭
oneDNN provider 的 Compute v2，peer 是强制 native 的同一产品；两侧使用相同公开 Tensor/Buffer/Queue/
PreparedOperation/Invocation/Completion、caller-owned 输入输出、串行执行和精度合同。prepare、allocation、
首次触碰和 warmup 都在计时外，Context/prepared artifact/run-state 仍各自独立。

旧 v1 工程采样在源码复核时被判为无资格证据：它连续调用两次 `Bench::run()`，没有使用同 iteration
count 的 cyclic paired `Bench::compare()`，也没有完整 Compute workload A/A。该失败事实保留在 closure
card，但不得阻塞产品、签 non-inferiority 或靠追加样本寻找 PASS。v2 改为同一 `HarnessRunner` 类型、
104 paired rounds、4 ms epoch、八次交替 warm operation，并在执行顺序、artifact allocation order 和函数
placement 上运行完整同 route A/A validator。

W1-WIN、MSVC 19.50、Release、static、portable ISA、LTO off 的唯一正式 qualification 在 `n=15` 停止，
总资格门与性能门均为 `PASS`：

| workload | 八个 point ratio / 99.26% CI 总范围 | 严格结果 | 产品含义 |
|---|---|---|---|
| row-major f32 MatMul `M=1,K=N=1024` | `0.99612--1.00697` / `0.97052--1.02849` | `PARITY`，NI `PASS` | selected 与 forced route 使用同一 native body，无 material 选择税 |
| row-major f32 MatMul `M=4,K=N=1024` | `0.99780--1.00263` / `0.99476--1.00642` | `PARITY`，NI `PASS` | 同上 |
| row-major f32 MatMul `M=32,K=512,N=1024` | `1.06388--1.07328` / `1.04308--1.08762` | combined `INCONCLUSIVE`，NI `PASS` | 每个 stratum CI lower 均大于 1；selected oneDNN 稳定优于 forced native，但两个 CI 与 5% material boundary 相交 |
| transpose-view f32 MatMul `M=1,K=N=1024` | `0.99440--0.99995` / `0.98274--1.01221` | `PARITY`，NI `PASS` | 两侧执行同一 native function pointer；无可归因固定损失 |

A/A、5%/10% known delta、timer floor、placement、artifact identity、fresh build、input/host stability 以及
workload-specific A/A validator 全部 PASS；没有触发 `n=31`。原始 bundle 保存在
[`compute-cpu-route-p1-v2-1e9c37f5-w1-win`](../../../../evidence/performance/compute-cpu-route-p1-v2-1e9c37f5-w1-win/)；
contract-lock / manifest / summary / controls / observations SHA-256 分别为：

```text
4ddce6166464bf121bca2e1115a1350cc654098ff1f642207530836f9a22f023
c0c72dd121eec7f0cca59bf96272a07d0c726af0858a2cbb58b09ea153cf3f7d
a1fcc51d68013b6caafd3fc19117c4550dbb38e1a805632f1e0c7632661e2ae8
d13d5cee99c90cea62ec004494683d29167c5061d286ceb5c0ec77319242a96e
53e7c9e5c7646968790cef0f77b5598b2ea6ac8d69b6e0e79171d587fee4abd4
```

严格重放命令返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/compute-cpu-route-p1-v2.json `
  --manifest evidence/performance/compute-cpu-route-p1-v2-1e9c37f5-w1-win/manifest.json
```

本结论只裁决当前 f32 CPU shipping route：RMSNorm 与 SiluAndMul 复用已关闭 SIMD leaf 证据；其他 dtype、
量化算子、region provider、并行 executor 和 GPU backend 必须在各自合同中裁决。
