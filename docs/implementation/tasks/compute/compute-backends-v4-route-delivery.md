# CPU/CUDA backend：R1 路线交付边界

## Closure card / SCOPED

- components / owner：`compute_backend_cpu`、`compute_backend_cuda` / Compute backend delivery。
- baseline：`ec142a92e00e27a73abbae4f040b6b361465e9a7`；当前工作树干净。
- reopen trigger：用户授权实施四路线，CPU-only/CUDA-only SDK 必须只交付选中 native 入口。
  当前 CPU 无 feature，CUDA OFF 的 OPTIONAL_ABSENT 仍进入 façade/install，改变这一公开交付合同
  符合 delivery reopen trigger；不据此修改 runtime。
- 前轮证据：[CPU v1](compute-backend-cpu-v1-factor.md)、[CUDA v3](compute-backend-cuda-v3-sync-resource.md)。
  其运行时事实保留，不能充当新交付或四路线基础已完成的证书。
- normal consumer：普通 `hyle::compute/graph` 源码、静态和动态 SDK 用户明确选择 CPU 或 CUDA。
  主纵切是计算执行资源准备，正交纵切是无 backend 的 tensor/图语义底座；后者不假报有执行能力。
- scope：CPU capability 的 target usage requirement/package feature；未选 backend 的 native 头、
  target/link/runtime 闭包；直接 package consumer。共同构建实施由 [R1](../build/compute-route-isolation-v1.md) 拥有。
- non-goals：C++ API、layout、算法、CUDA SDK 版本、kernel、stream/event/capture、性能仪器。
- 直接依赖/消费者：沿中央图不改 factor edge；Compute/Graph 产品、Native 和对应 package samples。
- 性能：P0，runtime source bytes 与常态成本不变，P1 不适用本轮纯交付改动。
- 预算/停止：与 R1 共用一次定向审计和最多四小时实施验证，不另叠加预算；最多一次有据纠正；
  runtime 缺陷归新卡，未完成适用交付写 BLOCKED，不能以旧 CLOSED 跳过。
- 预声明验证：R1 的 CPU-only/no-backend/CUDA-only 正负 configure；Windows/WSL 真 CUDA 与 CPU，
  macOS CPU/no-backend；backend 代表性现有测试、逐头、source/static/shared/relocation/export；
  selector、docs/closure/diff、候选后红队。重型验证串行、最多两 jobs。

## 能力与候选 FIX

| 能力族 | 层级 / 现状 | 有限动作 | 验收 |
|---|---|---|---|
| CPU 交付存在性 | 复合交付 / PARTIAL | BD1：CPU factor 发布真实能力定义与 package feature，中央产品可排除 CPU roots | 三交付一致；无 CPU 模型 kernel 的 CUDA-only |
| CUDA 可选交付 | 复合交付 / PARTIAL | BD2：既有 OPTIONAL_ABSENT 的 façade/install 投影排除 native 声明与空实现；CUDA ON 保留 SDK 依赖 | OFF 无 native 头/导出 target；ON 真设备与 SDK 闭包 |
| native 生命周期/失败 | 原子机制 / 已有证据 | KEEP 既有 runtime，复用源码不变证据及代表性 consumer 回放 | 不用 R1 宣称新算子/量化 Provider 完成 |

公开 C++ 声明、所有权、失败和线程合同不变；feature 只报告编译存在性，不能由 consumer 宏创建
未编译实现或改变对象布局。CPU/CUDA 的源码/头归属仍为原 owner。

## 七问与裁决

依赖版本/许可不变；补齐的是选路交付；复用中央图、package registry、Error/Mem；P0 无 runtime
增量；无 Native FFI；三交付与用户选型按 R1 取证。当前仅 SCOPED，BD1/BD2 审计提交后方可实施。
