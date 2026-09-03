# Mem reclamation：2026-09-03 有限恢复条件检查

<!-- MEM RECLAIM RESUME CHECK: CONDITIONS_NOT_MET -->

本记录只检查 registry 中 `mem_reclaim` 的两个恢复前提，不是 `RESUME` closure card，不授权修改
Mem、Container 或 benchmark。检查输入是 canonical HYLE commit
`471aec12b05b154d22cdf8b2b48e5cb180a81645`；`module-status.json` 仍是唯一当前状态。

## 1. Container consumer 检查

检查对象是
`include/hyle/container/concurrent_node_hash_map.h`，当前 SHA-256 为
`ea2e839d8c908ffbfa3b5f6f802876bb1cc299704910a56686d6847ee1220776`。

对 `include/hyle`、`src` 和 `samples` 的实例化反向扫描没有找到任何
`ConcurrentNodeHashMap<...>` 生产实例；现有调用只存在于该类型的 correctness tests。测试和公开能力本身
不能冒充 registry 要求的“已准入真实 consumer”。

当前 map 的 table handoff 是有意的时间/空间取舍：reader load `current_` 后取得 bucket lock，再复查
table identity；rehash 锁住旧 buckets、移动稳定 node、发布 replacement，再把旧 bucket array 挂入
`retired_`。旧表必须活到所有可能停在 load→lock 窗口的 reader 消失。当前实现把它们保留到 map 析构，
因此每次 lookup 不增加 reclamation guard、全局 reader RMW 或 TLS-domain traffic。

bucket count 始终是严格递增的二次幂。若当前 array 有 `B` 个 buckets，全部 retired bucket storage 的
几何和小于 `B`，总 bucket storage 小于 `2B`，另有 O(log B) 个 `Table` header。用户可在 steady-state
前 `reserve()` 减少增长次数；这不是无界泄漏，但可能不适合强内存预算且持续增长的 registry。

让每次 lookup 进入 `ReclamationDomain` 可以在 grace period 后释放旧 table，却会改变当前默认类型的
常态读成本。此前撤销的临时迁移在 single-reader、8-reader 和 mixed 形状间有改善也有 material 回退，
且没有 immutable consumer candidate。当前没有依据选择默认迁移，也没有需求证据支持新增
“reclaimed/fast”策略类型。因此本次 consumer verdict 为 **`NOT_ADMITTED`**：不建立 Container reopen
card，不满足 `mem_reclaim` 的 consumer 恢复钥匙。

## 2. 官方 Folly peer 检查

本地参考 checkout 为干净的 Folly
`eeab4246827eae9a11c0fa22529bff1a258aa83a`（2026-09-02）。在 W1-WIN、MSVC
19.50.35728、Visual Studio 18 2026 上按其顶层官方 CMake 路线执行：

```powershell
cmake -S C:\public\project-hyle\folly `
  -B C:\hyle-scratch\folly-rcu-peer-config-20260903 `
  -G "Visual Studio 18 2026" -A x64 `
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF `
  -DBUILD_SHARED_LIBS=OFF -DBoost_DIR=$env:BOOST_CONFIG_DIR
```

Compiler、Threads 和 Boost 探测后，官方配置在 required `FastFloat` 处停止：
`Could NOT find FastFloat (missing: FASTFLOAT_INCLUDE_DIR)`。因此当前环境尚不能产生可链接、可归因、
same-contract 的官方 Folly RCU peer，peer verdict 为 **`NOT_QUALIFIED`**。手抄 RCU 局部实现不算
官方 peer；在 consumer 钥匙也缺失时，不为一个无法恢复的模块启动整套 vcpkg/getdeps 构建。

scratch build 只是可删除的 configure artifact，不是 HYLE 产品来源或证据 bundle。

## 3. 裁决与下一步

两个恢复前提均未成立，所以：

- 不创建 `RESUME` round，不修改 registry；`mem_reclaim` 继续保持 `BLOCKED`；
- shipping Mem v3 保持 `CLOSED`，Container 也不因假想 consumer 重开；
- 不修改候选 API/layout/算法、旧 benchmark 或 5% margin；
- 主施工队列继续 Obs。未来完整 Container 审查或另一个真实 consumer 若明确需要“持续增长且及时释放
  旧 publication storage”，可先建立独立 owner card；同时正式 provisioning Folly 官方依赖，两个条件
  都成立后再从 `SCOPED` 恢复。

这次检查完成了 roadmap 的有限恢复检查，不把同一个已知 blocker 变成无限调查。
