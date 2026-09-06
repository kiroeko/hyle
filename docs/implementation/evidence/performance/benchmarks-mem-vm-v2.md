# Mem-VM v2 model-loader 工程性能证据

<!-- PERFORMANCE EVIDENCE: CURRENT -->

> 当前机械状态只见 [`module-status.json`](../../module-status.json)。本页绑定候选
> `9604bf4a33f13bd5f84ef78fedce55f305ac9d50` / tree
> `57adb69f185cf02d42ab7f220ebeccbb91170b1b`，只裁决 Linux model-loader 的 sparse mapping
> journey；WSL2 结果不冒充 native-Linux Formal 证书。

完整 owner/API/OS 差异见 [Mem architecture](../../../architecture/mem-design.md)和
[Mem guide](../../../user/mem.md)，本轮事实见
[Mem-VM v2 closure card](../../tasks/mem_vm/mem-vm-v2-inference-mapping.md)。

## 同合同任务

冻结任务对一个 fresh 4 MiB sparse model file 做相同的 open/fstat/read-only shared mmap/close，对 64 个
确定的八页 lazy ranges 各做一次 `POSIX_MADV_RANDOM`，对这 64 段各做一次 `munmap`，验证相同 released
bytes 与 65 个 survivor fragments，最后逐片 teardown。candidate 用
`FileMapping::release_ranges()` 一次排序/合并并线性更新 fragment table；peer 精确重放锁定 llama.cpp
`74a7c897f049c17e7080423aa2111776eff6ebbf` 的逐 range vector rebuild 机制。

两边 native mapping、advice、release、remaining teardown 的 syscall 数和页效果一致；只比较 bookkeeping
算法与 HYLE public batch route，因此是 `SAME_CONTRACT`。合同见
[`mem-vm-model-loader-p1-v2.json`](../../../../bench/contracts/mem-vm-model-loader-p1-v2.json)。Windows
没有安全的 partial-unmap 同合同，本轮不把 whole-view API 或更少 syscall 的路线混入该排名。

## W1-WSL 结果

W1-WSL、Ubuntu/WSL2、ext4、GCC 14.2、Release static source、LTO off；15 个独立进程、双执行顺序与
primary/perturbed placement。`advantage ratio > 1` 有利于 HYLE：

| Stratum | median advantage ratio | 99.26% CI | exact | 5% NI |
|---|---:|---:|---|---|
| primary / candidate-first | 1.0331 | [1.0108, 1.0481] | `PARITY` | `PASS` |
| primary / peer-first | 1.0307 | [1.0121, 1.0402] | `PARITY` | `PASS` |
| perturbed / candidate-first | 1.0385 | [1.0081, 1.0570] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first | 1.0467 | [1.0165, 1.0546] | `INCONCLUSIVE` | `PASS` |

聚合 exact verdict 为 `INCONCLUSIVE`，4/4 的 5% NI 与全部 controls 为 `PASS`。四个点估计均有利于
HYLE（约 3.07%--4.67%），但置信区间没有为所有 strata 证明超过 5% 的 material win，因此只签发
**工程非劣 PASS**，不宣称跨 OS、跨模型或普遍胜出。

## 算法与常态成本

- 没有 partial release 的普通连续 mapping 不分配 fragment metadata；
- 首个有效 release 才建立有序 survivor table；checked access 为 `O(log F)`，局部 control 为
  `O(log F + K)`；
- 一批 R 个请求与 F 个现存 fragments 的 bookkeeping 为 `O(R log R + F)`，加真实 syscalls；避免
  llama 当前逐项 vector rebuild 的最坏二次累计工作；
- 空、子页、fully released repeat 和 Windows unsupported 路线在 metadata allocation 前早退；
- `populate`、lock、munmap 和 teardown 仍按实际页/syscall 计费，没有声称让 OS syscall 本身变快。

## 资格、原始证据与限制

- evidence：`/tmp/hyle-mem-vm-perf-evidence-r1`；归档：
  `C:\public\project-hyle\_validation\mem-vm-v2-performance\hyle-mem-vm-perf-evidence-r1.tar.gz`，
  SHA-256 `EFD00C22451D0E2D6E9CA500FCCA2ABE343E454EE2ED0F28DDA4234D57C81A0E`；
- contract-lock / manifest / summary SHA-256：
  `3F2FCD4EFA42406E4EB370BB76AAA1A45B23797A4955EF5107CB0427AA2D1AC0` /
  `14C90027CBE0715C855E20D9382B73EAB3EE614FF708F090FE2FF02B764191F3` /
  `E338BFD642F5254E315769457DA0E748EC7A35B7B6E5354315BDA33AFE2F4878`；
- deep verifier：`QUALIFIED`，environment `CURRENT`，paths verified，source/provenance `EXACT`；
- WSL 与 W1-WIN 共享物理 host，因此不是独立硬件；native-Linux/Formal、Windows partial release、
  remote NUMA、page-fault/RSS 和端到端 load-to-ready 不在本证据中；Formal 为 `PENDING`。

## 复现

```bash
python3 scripts/hyle-perf.py validate \
  --contract bench/contracts/mem-vm-model-loader-p1-v2.json
python3 scripts/hyle-perf.py verify \
  --manifest /tmp/hyle-mem-vm-perf-evidence-r1/manifest.json \
  --contract bench/contracts/mem-vm-model-loader-p1-v2.json
```
