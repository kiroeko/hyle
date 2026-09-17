# Compute / Graph 性能证据

当前四路线工作中，R3 CPU 的原统计与具名例外见[全库账本](full-library-scorecard.md)；R4 CUDA、
R5 Apple 的最终结果与 R6 Vulkan 的已完成检查点见本页后段。以下前段保留旧候选的 CPU/CUDA
历史证据，不用旧 copy 或跨域样例签新路线完成。

R3 CPU Graph 的 fused/natural 裸 direct 对照经[服务边界核对](../../tasks/compute/cpu-r3-comparison-boundary-clarification.md)
改为成本诊断，不承担同合同排名。原 decode 用户取舍、原测量与旧表均保留，不解释为算法或 Sched 退化。

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

## Graph oneDNN gated-MLP region P1/P2

[`graph-onednn-region-p1-v1`](../../../../bench/contracts/graph-onednn-region-p1-v1.json)
把同一个四节点 `gate/up MatMul -> SwiGLU -> down MatMul` 任务拆成两个不能混淆的比较：

1. P1 abstraction 行与已经手工融合的 direct route 使用相同两份 oneDNN 3.13.2 BRGEMM plan、相同
   HYLE SIMD SwiGLU、storage、workspace、串行 executor 和终态边界；只测 Graph immutable plan、binding、
   queue sequence 和 Completion 的固定组合成本；
2. P2 natural 行让 direct consumer 按四个普通操作分别 map/execute/close，Graph 则使用一次真实 region
   boundary；这衡量 region 产品相对自然调用的完整收益，不拿裸 kernel 与完整 Graph 冒充同合同。

W1-WIN、MSVC 19.50、Release、static、portable ISA、LTO off 的唯一资格化运行完成了预声明的
`n=15 -> n=31`，环境、artifact、A/A、5%/10% known-delta、timer-floor 和 placement 门全部 `PASS`：

| workload | 八个 median ratio | combined exact / NI | 结论 |
|---|---:|---|---|
| decode fused-boundary `M=1,K=256,H=512` | `0.9622--0.9817` | `INCONCLUSIVE / UNRESOLVED` | 五行 `PARITY/PASS`；三行 99.26% CI 下界为 `0.9444--0.9495`，略穿过 5% NI 界限 |
| decode natural-sequence `M=1,K=256,H=512` | `0.9761--0.9994` | `PARITY / PASS` | 八行均通过 |
| short-prefill fused-boundary `M=16,K=512,H=1024` | `0.9894--1.0144` | `PARITY / PASS` | 八行均通过 |
| short-prefill natural-sequence `M=16,K=512,H=1024` | `0.9961--1.0068` | `PARITY / PASS` | 八行均通过 |

源码 P0 账单解释了最小 fused-boundary 行的固定方向：相对 direct fused route，Graph 额外执行一次
in-flight 原子 admission/terminal、外部 binding 完整性扫描、五个 `TensorView` staging、queue sequence、
region vtable 和 Completion poll/settle；两侧 leaf kernel、mapping 和 workspace owner相同，steady path 都
零 heap allocation、零搜索、零 JIT、零隐藏 copy。该成本是当前通用 Graph 安全语义，而不是误选 kernel；
没有真实消费理由授权为 benchmark 增加单节点特判。由于工程 P1 必须明确通过，不能把约 3% 的中位差和
跨界 CI 改名为 parity，`graph_provider_onednn` 因而不能关闭；同时 P2 自然路线及短 prefill 的通过证明
它也不是一个 material fixed loss。

原始 bundle 保存在
[`graph-onednn-region-p1-v1-c05b6e2a-w1-win`](../../../../evidence/performance/graph-onednn-region-p1-v1-c05b6e2a-w1-win/)；
contract-lock / manifest / summary / controls / observations SHA-256 分别为：

```text
3bedfbb9c5677489a3126659fbc98385702165eeb7aaf02abf04fe2282bae902
61cf354d3ef24d7f06bd06a8bd2453ccfc84acc428a9309a80e583dbb8a18d12
7c46190c410c2f352757c92cb11dc1145362edaacdd3c73cb628703c348e13cb
99e03e5a5f13140553be7d621b9726e0c2e4a12c822ac7f2f75228be1865d5ea
77f19e42023a60041569d3ddc27a9dacf04fdebd27fae0081c1a525fc07044b3
```

严格重放命令返回 `QUALIFIED`、`CURRENT`、`EXACT`、`paths_verified=true`：

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/graph-onednn-region-p1-v1.json `
  --manifest evidence/performance/graph-onednn-region-p1-v1-c05b6e2a-w1-win/manifest.json
```

本证据不授权追加样本或为了 fused micro-boundary 改公开 API。只有后续 Runtime/Graph 主干因真实异步、
specialization、placement 或 replay 需求自然改变公共 submit path，并形成新的 immutable candidate，才可
依法重新资格化该 factor。

## CUDA runtime P1 与跨域 Graph P2

基础 runtime 候选固定在 `7718b163946104989c9a000f19fd01f454211d36`。两份合同都在 W1-WIN、
RTX 4070、CUDA 13.3、MSVC 19.50、Release static、LTO off 上于 `n=15` 按规则停止；artifact identity、
fresh build、A/A、5%/10% known-delta、timer floor、placement 和全部产品门均为 `PASS`。

[`compute-cuda-runtime-p1-v1`](../../../../bench/contracts/compute-cuda-runtime-p1-v1.json) 比较一份 1 MiB
resident D2D copy：两侧使用 non-blocking stream、独立但等价的 device buffers，以及相同的一次 event record
和 terminal wait。HYLE 侧只额外承担公开 BufferView 验证和预留 Completion slot 管理。八个执行顺序、
分配顺序与 placement strata 的 median ratio 为 `0.98967--1.02142`，八行均为 `PARITY/PASS`，综合
exact verdict=`PARITY`、non-inferiority=`PASS`。因此没有把随姿势反转的点值解释成固定输赢。

[`graph-cuda-pipeline-p2-v1`](../../../../bench/contracts/graph-cuda-pipeline-p2-v1.json) 比较同一 4 MiB
CUDA D2H → native CPU Unary → CUDA H2D 任务。Graph 与 direct route 使用相同 TensorDesc、CUDA/CPU
Context 和 Queue、device/pinned storage、leaf operation、依赖与最终 terminal contract；compile、prepare、
allocation、bind、first touch 和 warmup 都在计时外。Graph 使用冷路径已验证的 exact route 与内部
prevalidated enqueue，direct consumer 则自然地逐项调用公开安全接口。八个 strata 的 median ratio 为
`1.24305--1.26042`，99.26% CI 均完整高于 1，综合 exact verdict=`WIN`、non-inferiority=`PASS`；绝对
中位数为 Graph `1791.2--1805.6 op/s`、direct `1431.8--1447.5 op/s`。这是完整跨域 product replay 的
计划复用收益，不外推为 leaf CUDA kernel、任意 DAG 或未来 provider 的普遍胜率。

原始 bundle 分别保存在
[`compute-cuda-runtime-p1-v1-7718b163-w1-win`](../../../../evidence/performance/compute-cuda-runtime-p1-v1-7718b163-w1-win/)
和
[`graph-cuda-pipeline-p2-v1-7718b163-w1-win`](../../../../evidence/performance/graph-cuda-pipeline-p2-v1-7718b163-w1-win/)；
其 contract-lock / manifest / summary / controls / observations SHA-256 分别为：

```text
compute CUDA P1
a12115af77d31301155eb871db2688361ac070f5de9e5b4140463c9b10eeaf63
6553a09e2e0451e089b48d3ad34dd0addfab2c4415806c2566c67176f45994fc
e90922d0bd9ded8e1d01b82387430973a13d4615568733ce285198992969bcb9
35f759fcb2310631d958e828a1eabd892803a3067969860e91484d3dbbf1f8f8
80cc11f27af8894057e1eaa4e8f516196ccd7b95819300abc3d00f6ca31c9c4c

Graph CUDA P2
1525b6650faee9d58c2daf91d51a0f4b2802e7943fabb36296b6bc171d73a69d
40bc9b553f9ede8652fd827a84e2350a3e7f5836ee89a2d91a6ca81143065305
012e3c37ddc6cd3fc32d7ed3311b9757863bd58406314f6f6760ce5ea3336134
9729bdd98c1ffa4a494a0906c8a1943f0eb2b5c8aa3a617cea491ad1f8f24e82
9ded1b8c9a6b3d4f69c0da515d50ac923cd261ada4caadf7406b3b0b04d9cbc9
```

两份 retained bundle 均经 `hyle-perf.py verify` 返回 `QUALIFIED`、`CURRENT`、`EXACT` 和
`paths_verified=true`。

## R4 CUDA 最终成熟默认（2026-09-17）

产品候选755fb7c7；资格化 checkout f01e9237 只含文档/状态/证据差异。W1 RTX4070/SM89、
driver610.62、Toolkit13.3.73、MSVC19.50、Release/static、caller CPU2、process mask0x14，
native helpers 保留原生调度。Windows DLL 实际邻置；旧单核失败采样不复用。

普通默认为 cuBLASLt dense、Marlin group128 Q4、SGLang precise vector 激活；HYLE separate/尾部
能力保留。此前 v2 完整链比较没有给出 HYLE 激活的完整域优势，因而采用成熟基线。最终 v3 两侧
使用同数学、同 HYLE 生命周期和完成边界，验证接入成本，不授予 HYLE 算法赢家。

| 路径 / workload | 八个分层 median ratio 范围 | 严格方向 | 5% NI |
|---|---|---|---|
| Direct M1 | 0.996930–1.004249 | 8 INCONCLUSIVE | PASS |
| Direct M16 | 0.998512–1.003080 | 8 INCONCLUSIVE | PASS |
| captured Graph M1 | 0.996611–1.003051 | 1 LOSS / 7 INCONCLUSIVE | PASS |
| captured Graph M16 | 0.994638–1.004986 | 8 INCONCLUSIVE | PASS |

[Direct 完整证据](../../../../evidence/performance/compute-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)、
[Graph 完整证据](../../../../evidence/performance/graph-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)：各 n=15，
未扩到31；16个 A/A、120产品进程、30 control 进程、全部 validator 与 raw replay 均完成并通过。
保留 Graph 单个小输，不把 practical PARITY 改写为严格等速。资格限于当前两个 resident FFN
原型，不是生产模型吞吐，也不外推其它 GPU/driver/Toolkit/dtype/shape。

[原生完整边界诊断](../../../../evidence/performance/cuda-r4-native-boundary-v1/README.md)使用同数学链，
每模式四层×15进程，有自己的数值 oracle 与同 build A/A。Direct 中位差范围 −0.339～1.639 μs，
Graph −1.481～0.851 μs，完整 FFN 约307～318 μs。差值按分层报告，不合成新 CI；裸路径只提供
一个事件，缺少公共 admission/保活/失败服务，故保持 NATURAL_ROUTE，不签 NI、WIN/LOSS 或零开销。
冷 pair_prepare 不是含 packing 的全启动时间；完整 opaque SDK 峰值和独立析构时延未测。

结合主卡正确性/资源/三交付证据，六个 R4 CUDA owner 按协议 CLOSED。CPU 例外、未交付的
Apple/Vulkan 和 R7 终审保持原状态；本结案不代表整个四路线基础架构已经完成。

## R5 Apple 完整 region（2026-09-17）

候选 `1636365b`，Air Mac16,12/M4、macOS 27.0/26A428、锁定 MLX v0.32.2，M1-MAC engineering
profile；实际 interactive QoS、AC，无硬 P-core pin 或 Formal 身份声明。Direct/Graph 默认采用
成熟 MLX compiled region，参考独立表达相同数学链并承担相同 Context/owner/checked completion/
结果寿命。完整 [原始证据与成本记录](../../../../evidence/performance/apple-r5-final-1636365b-m1-mac/README.md)
已归档；全部 A/A、controls、validator、两个 FFN oracle 与八分层 n15 通过，未扩样或纠正产品。

| 路径 / workload | median ratio 范围 | 严格方向 | 5% NI |
|---|---|---|---|
| Direct M1 | 0.999017–1.000872 | 8 INCONCLUSIVE | PASS |
| Direct M16 | 0.998700–1.003551 | 8 INCONCLUSIVE | PASS |
| Graph M1 | 0.998156–1.000970 | 8 INCONCLUSIVE | PASS |
| Graph M16 | 0.998105–1.003708 | 8 INCONCLUSIVE | PASS |

每一严格 CI 均跨 1；practical PARITY 不等于严格等速或 HYLE 算法胜利。原生 caller-thread MLX
另跑四层×15进程，保留相同数学/GPU同步，但没有 HYLE admission/owner-thread/cancel/close 服务，
故为 NATURAL_ROUTE / NOT_COMPARABLE。Direct M1 配对增量 5.719–9.712µs、M16 −4.229–6.785µs；
Graph M1 4.198–11.062µs、M16 5.719–12.185µs，均为分层中位数范围，不合成 CI 或跨入口输赢。
这是约2.9–6.0ms完整 resident FFN 的自然成本，不外推其它硬件、数值或生产模型吞吐。

R5 两 owner 结合实际正确性、host 插桩和两 Mac 三交付按协议 CLOSED。R6/R7 继续按当前卡施工；
CPU 已批准例外、未测硬件与未插桩 vendor/GPU 的限制保留。

## R6 Vulkan Windows 完整 F32 片段检查点（2026-09-17）

产品候选 `8e45b104`，Windows 测量快照 `3894b72e`；W1-WIN engineering、RTX 4070、
strict F32，实际 caller CPU 2、process mask 0x14。独立参考使用锁定 ggml `456172ec` 的
原样 shader 与独立数学 recipe，两侧承担相同 HYLE native 生命周期 adapter。本比较验证
成熟实现接入、规则绑定及 Graph 编译的非劣性，不是整个 ggml runtime 或 HYLE 算法的排名。

| 入口 / workload | 八分层 median 吞吐比范围（candidate/peer） | 严格方向 | 原 5% NI |
|---|---|---|---|
| Direct M1 | 0.990494–1.004220 | 8 INCONCLUSIVE | PASS |
| Direct M16 | 0.991878–1.005471 | 8 INCONCLUSIVE | PASS |
| Graph M1 | 0.994071–1.004920 | 8 INCONCLUSIVE | PASS |
| Graph M16 | 0.991991–1.004861 | 8 INCONCLUSIVE | PASS |

[Direct 原始证据](../../../../evidence/performance/vulkan-r6-w1-direct-8e45b104/README.md)与
[Graph 原始证据](../../../../evidence/performance/vulkan-r6-w1-graph-8e45b104/README.md)各自完成
16 个 A/A 资格进程、120 个产品进程及 30 个公共 control 进程；每入口两个完整 FFN shape 的
16 项非劣门均在 n=15 通过并停止。全部严格 CI 包含 1，不把工程非劣改写成严格等速或算法胜出。
输入、制品及 host 身份稳定，公共 `verify` 为 EXACT/CURRENT/QUALIFIED；两种 placement 的
实际热函数地址变化已由原始链接 map 核对，未用热路径 padding 或 LTO 制造差异。

Direct 的固定 caller buffers 与 Graph 的 generated owning result 各自和同合同参考比较，
不跨入口排名。裸 Vulkan submit/fence 的取消、quarantine 和结果 lease 责任不同，仍按
[原生边界合同](../../tasks/compute/vulkan-r6-performance-contract.md)单独诊断，不由本表签证。
Air 两入口 P1、原生边界诊断、最终交付/selector 与 R6 终审尚未全部完成；两个 Vulkan owner
继续 CANDIDATE。本检查点不外推 Linux GPU、AMD/Intel、其它数值格式或生产模型吞吐。

### Air Direct 同合同检查点

Apple 测量快照 `a41d43d5`，产品仍为 `8e45b104`；M1-MAC engineering、Air Mac16,12 / M4、
MoltenVK 1.4.1，实际 interactive QoS。[完整原始证据](../../../../evidence/performance/vulkan-r6-m1-direct-8e45b104/README.md)
包含 16 个 A/A、120 个产品进程、30 个 control 进程及 identity/placement 复核；16 项原 5% NI
全部 PASS，n=15 停止。M1 的八分层 median 比为 1.000268–1.001713，严格方向 7 WIN / 1
INCONCLUSIVE；M16 为 0.988563–1.021558，4 WIN / 4 INCONCLUSIVE。两个 workload 的整体方向
均未决；保留每个分层的原始方向，不把成熟数学链的接入差异写成 HYLE 算法优胜。
Air Graph 和上述其它 R6 门继续按主卡执行，不因本检查点重复 Windows 或 Air Direct P1。

### Air Graph 同合同检查点

[Graph 完整证据](../../../../evidence/performance/vulkan-r6-m1-graph-8e45b104/README.md)与 Direct
使用同一产品/Apple 测量快照；16 个 A/A、120 个产品、30 个 control 进程及 identity/placement
复核完成，16 项原 5% NI 全部 PASS，n=15 停止。M1 八分层 median 比为 0.999569–1.001189，
方向 2 WIN / 6 INCONCLUSIVE；M16 为 0.985050–1.011573，4 LOSS / 4 INCONCLUSIVE。
M16 的四个 peer-allocation-first 分层在正反执行顺序、两种 placement 下都保持约 1.2%–1.5%
的负向延迟，继续记 LOSS；其余分层未决。分配顺序与方向有关，但本证据没有确定 driver/cache
机制，不作额外归因或为这组已通过工程门的结果重测。

Vulkan 四份正式 P1（Windows/Air × Direct/Graph）均已通过原门并停止，全部严格方向保留。
R6 仍待原生边界诊断、最终 ASan/三交付/selector 与终审；本页没有提前签路线 CLOSED。

### Windows 原生边界诊断

[完整记录](../../../../evidence/performance/vulkan-r6-w1-native-boundary-8e45b104/README.md)
按每入口四个顺序分层 × 15 个进程完成，两个 FFN shape 的完整 oracle 与全部身份检查通过。
Direct M1/M16 的成对额外延迟中位数范围分别为 −8.311–5.553µs / −36.229–18.402µs；
Graph 为 −3.995–8.393µs / −8.800–7.361µs。各范围只汇报独立分层，不合成 CI 或跨入口排名。
裸原生责任较少，始终分类 NATURAL_ROUTE / NOT_COMPARABLE；负差不等于 HYLE 零开销或算法胜出。
对应正式 P1 不重复，Air 和其它未完成验收继续按主卡串行执行。

### Air 原生边界诊断完成

[Air 完整记录](../../../../evidence/performance/vulkan-r6-m1-native-boundary-8e45b104/README.md)
同样按每入口四层 × 15 个独立进程完成，完整 oracle 与身份检查通过。Direct M1/M16 的成对
额外延迟中位数范围为 −9.430–−3.191µs / −553.651–−121.438µs；Graph 为
−1.750–−1.163µs / −389.371–432.296µs。分别保留各分层，既不合成 CI，也不跨入口比较绝对值。
裸路径的生命周期责任较少，分类始终为 NATURAL_ROUTE / NOT_COMPARABLE；全路径差值不能单独
归因于 host 包装或 driver/cache，也不从负差宣称零开销或算法胜出。

两系统四项自然边界诊断至此全部停止采样。四份同合同 P1 仍按前述原门与严格方向保留；
最终 host ASan 已完成，三交付/selector 与路线 terminal 继续按主卡验收。
