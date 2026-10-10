# F1-04：Windows completion 退役定向资格

- 产品子候选：`392d4503675fa095c3a89cc0a90641b4d48651cf`；owner compute/compute_backend_cuda AUDITED。
- 机器：Windows 11、RTX 4070 12282 MiB、driver 617.14、CUDA Toolkit 13.3.73、MSVC 19.50.35728.0。
- 构建：`C:\h7\foundation-f1-q0`，exact factor compute_backend_cuda，build1/test1；
  使用正常共享依赖缓存根 `C:\public\tools\hyle-dependencies\v1`，不是其中 sources 子目录。
  分别 HYLE_TEST_INSTRUMENTATION=ON/OFF；复用同一小型构建树。全部 DLL 在 EXE 同目录。

## 实现与源码 review

共同 Completion::settle 交回 attachment 与原错误给 native destroy；不因一次错误观察立即释放。
CUDA 复用 EventSlot 原 terminal_hint 和 fence 字段区分 Pending/Retired；不增加公共对象/slot 字段，
原 reaper 保留未证明完成的工作，异常 wait 只有一次原 stream drain。

致命设备错误分类依据锁定 Toolkit header 及
[官方 runtime 错误语义](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/group__CUDART__TYPES.html)。
ContextIsDestroyed 可能只描述当前线程 context；一般 observation error 没有退役证明。
没有明确“失效上下文/进程”依据的 ECC 类别已在实现 review 中排除，不按名称猜测致命性。

局部红队：检查各 native/CPU+GPU destroy 对 lifetime/hint 的接收路线、重复错误、late callback、
slot reuse、未知 drain 与 shutdown 分支；unknown shutdown 不能触发释放。
普通成功路径缓存已有 slot 退役状态，destroy 不再重新查询同一已退役 event；没有额外 context probe、
全设备同步、第二 reaper 或热 metadata。此为源码/P0成本事实，不是计时性能赢家。

## 已运行与证据边界

| 构建 | common | CUDA/backend/device status | 实际迟到失败 | 证据 |
|---|---:|---:|---:|---|
| 插桩 | 64 PASS | 30 PASS | 1 PASS | [收据](receipt.json)及相邻 exe logs |
| shipping layout | 64 PASS | 28 PASS | 1 PASS | [收据](shipping/receipt.json)及相邻 exe logs |

新 common 回归测试用受控异步状态证明错误观察不立即释放；它不是 GPU provider 资格。
两个新 CUDA 测试复用原 HostStreamGate/ResultSlot/OwnedCudaBuffer/increment helper，
真实 native work 仍在执行，仅装饰 event query/wait 的观察结果：
unknown query 保持 Pending、abandon 等实际完成回收、wait 异常以原 stream 完成证明并保留原码，
增量结果仍为 42。没有新产品 kernel/mock provider。

[shipping exports](shipping/backend-exports.log)不包含 set_completion_observation_hooks；测试钩子没有出现在交付布局。
原生查询失败不是曾发生过的生产 GPU 故障证据，本次是受控资格化。

未运行：完整 F1 lease API、四路线/Graph新接缝、全部 selected direct consumers、WSL/Air、三交付与 P1。
这些仍由 F1-01/02/03/05/06/07/08 兑现；不能把 95/93 个子批测试写作整包/goal完成。
原 ordinary borrowed tensor/resource 合同仍要求 owner 覆盖真实 native 使用；
未知 wait 错误返回不能单独证明设备 idle，runtime 会保留已附着 owner 直到实际退役。

Graph aggregate 对 child error/remaining 的组合仍需在 F1-06 的 lease-aware 接线中定向核对；
本收据不以 common/CUDA 子批测试代签聚合图新外部租约的生命周期。
当前[统一 LOC](loc-current.json)为 Compute+Graph 非测试 **68,284 行/301文件**，
全库非测试 **163,511行/652文件**；此子修复净增80行实现，不计测试或文档。

## 继续入口

保留此小型 cache（当前 shipping/OFF），不要恢复旧 CUDA U2/U3 交付脚本。
后续按 F1卡冻结 lease-aware API，集中接线后才启动整批消费者/三交付；不重复此子批的同身份证据。
