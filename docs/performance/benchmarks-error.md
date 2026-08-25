# error 实测铁证

> **2026-08-25 freshness gate：**本页点值绑定下述 2026-07 commit，只是本轮重签的历史输入。
> error 的产品收口状态未改变；相同 source identity、peer pin、编译合同与三平台 n=15 完成前，
> 这些点值不是当前依赖组合的发布证书。现行状态只看
> [`pre-release-readiness`](../milestones/pre-release-readiness.md)。

> 本页是 [performance 总账](./README.md) 的 error 模块详节。稳定设计见
> [error-design](../architecture/error-design.md)，用户选型见
> [error guide](../guide/error.md)。benchmark 只验证消费者会使用的产品形状，不定义 API。

## 2026-08-25 当前重签合同（采样前声明）

本轮先重签 Windows x64/MSVC，预声明所有同合同 hot pair 的等价预算为 `m=3%`，即
`[L,U]=[1/1.03,1.03]`；exact verdict 使用 `bench-stat.py` 的 paired
`RIVAL/HYLE` ratio 总体中位数 exact order-statistic CI，另行保留同一 CI 的单边
non-inferiority gate。每个执行顺序至少 15 个独立进程，candidate-first 与 peer-first 分开
采样、分别保存逐进程 schema-1 raw JSON。看见结果后不更换 margin。

| workload family | 正常消费者任务与计费边界 | ownership / 数据形状 | peer 与资格 |
|---|---|---|---|
| OK construct/copy | 构造或复制成功 carrier 并检查 | 无诊断、无分配 | Abseil `Status`，同合同 |
| runtime code-only / Cancelled factory | 从运行时 canonical code 或常用 factory 构造 carrier；整个对象 escape，caller 查询 code | 无 message/context/payload；runtime code 在 `InvalidArgument`/`NotFound` 间变化 | Abseil empty/no-location `Status`，同合同；`std::error_code` 只是较弱成本下界 |
| Result success / code-only failure | leaf + layer1 + layer2 + chain 四个对称 `noinline` frame，caller 检查并消费 | 成功值或 canonical code，无 message | Abseil `StatusOr<int>`，同合同；`std::expected` 是较弱下界 |
| owning message construct/copy | 从 borrowed view 构造 owning status，或复制共享表示 | 7B（SSO-sized）、29B（跨 SSO）、256B long；copy 固定 29B | Abseil no-location `Status`，同合同 |
| context | 给 message error 前置 1/3 层；另以 16 层分别测 fresh unique-rvalue 与每层 lvalue-derived chain | 相同 seed/context/final text；构造、增长和释放都计费 | Abseil `StatusBuilder::SetPrepend`，同消费者语义；unique 与 lvalue 不跨表合并 |
| message Result / Update | 两个真实 frame 传播预建 message error；首错聚合分别使用 stored lvalue 和两个 fresh noinline factory prvalue | 相同 code/message、调用边界及临时对象生命周期 | Abseil `StatusOr` / `Status::Update`，同合同 |
| owning format | 生成完整 owning 人读字符串 | 54B 与 512B message，分配/释放计费 | Abseil `Status::ToString`，同合同；HYLE caller-buffer 128B 行只报绝对成本 |
| native detail / payload | 构造、errno 转换或查询一个 machine-readable native value | HYLE 固定 `{domain,int64}`；Abseil 可携带多项任意 payload | 能力与 storage contract 不同，机械标 `NOT_COMPARABLE`，SOTA gate `N/A` |

默认序由同一 binary 先运行 HYLE、再运行 Abseil；设置
`HYLE_BENCH_RIVAL_FIRST=1` 只交换两侧的执行顺序，不改变 lambda 定义、输入、分配 setup 或
函数 placement。几乎所有 error aggregate 都低于 100ns，Windows 另从完全相同的两个 source
生成 placement targets。SOTA peer claim 明确绑定 `HYLE_STATIC=ON` 的 shipping shape，使 HYLE
与静态 Abseil 进入同一最终 PE；唯一的 build 差异是 MSVC LINK `/FUNCTIONPADMIN:64`，对两侧
对称插入函数间 padding，不改变 compiler options、inlining、产品 API 或计费逻辑。shared 默认
形状由 ABI/package/correctness 与绝对回归覆盖，不把 DLL 边界差异写成算法 peer 胜负。
runner 对 primary/placement 各自执行 candidate-first 与 peer-first，并核验 placement PE 的
`.text` 确实增长、两种对象均保留四帧调用边和 opaque/barrier relocation。四个 order gate 只要
一个 `FAIL/UNRESOLVED`，SOTA 就不能 `PASS`；shape 间 exact verdict 不一致则为
`INCONCLUSIVE/placement-sensitive`。当前共享 checkout 仍处于 dirty 集成状态，正式证书必须在
candidate commit 后 fresh build 并使用 `--require-clean` 重跑。

## 历史证据（2026-07-29，三平台 n=15）

基础产品能力基于 `befaad0`，非法 Result 诊断冷热分离基于 `7d22eb8`，渐进 context
表示固定为 `8dace9d`；benchmark 消费形状仍是 `6d42add`，没有为该实现新增测量专用 API。
`f31e8d0` 新增通用 Win32 adapter，但没有改变既有 Status representation/热路；Windows 在该
commit 上复跑两套 n=15，同能力结论仍为 1 项 TIE、其余全部 WIN。Linux/M4 旧数字继续绑定
未改变的算法实现，新入口以复杂度与真实 platform 去重证明，不制造高错误率 workload。
Abseil 固定到本地参考库 commit `35b9f25b`。环境为 Windows MSVC 19.50、WSL GCC 14.2、
Apple M4 AppleClang 21，Release、`HYLE_TEST_INSTRUMENTATION=OFF`（发布形状），使用
`scripts/bench-stat.py -n 15` 的同进程配对判定。表中绝对值统一按
**Windows / Linux / M4** 排列，不能跨机器横比；倍数是配对样本的中位比值，可能与两列
独立中位数之比略有差异。

该轮历史证据严格分成两种消费者信息合同：

1. **不携带额外诊断**：只有成功值或 canonical code。主对手是 Abseil
   `Status/StatusOr` 的同能力路径；`std::error_code/expected` 是不能携带任意消息和 native
   detail 的较弱下界，只单独登记，不用于宣称同能力 SOTA。
2. **携带诊断**：相同 canonical code、owning message、语义 context 或同一
   machine-readable native value。对手是 Abseil `Status/StatusOr/StatusBuilder/payload`。

没有第三类“为了 benchmark 好测”的公开类型。benchmark 也不允许给产品对象增加常驻字段、
缓存或特殊入口。

## 合同一：不携带额外诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| OK 构造并检查 | 0.378 / 0.057 / 0.226 | 0.376 / 0.054 / 0.226 | 三平台 TIE |
| OK Status 拷贝并检查 | 0.507 / 0.280 / 0.236 | 0.547 / 0.276 / 0.257 | Win **WIN 1.085x**；Lin TIE；M4 **WIN 1.093x** |
| code-only Status 构造 | 0.377 / 0.060 / 0.224 | 1.511 / 0.916 / 0.958 | **WIN 3.94 / 15.58 / 4.28x** |
| `Result<int>` 经过四个真实调用帧，caller 解包并消费 | 2.565 / 2.478 / 3.058 | 3.129 / 2.915 / 3.415 | **WIN 1.215 / 1.191 / 1.113x** |
| code-only Result 错误经过四个真实调用帧 | 2.372 / 1.103 / 1.443 | 4.009 / 2.258 / 2.455 | **WIN 1.579 / 2.006 / 1.681x** |

四层传播不是在同一个函数里连续创建三个临时 Result，而是 leaf → layer1 → layer2 → caller
的四个真实 `noinline` 调用帧，三家边界完全对称，最后消费成功值。旧形状曾在 M4 同时保活
多个临时对象，制造约 8% 假 LOSE；旧 GCC 形状还只内联了 std/Abseil 的外层包装，制造约
18% 假 LOSE。修复的是测量合同，不是为跑分修改产品 API。

### 能力更弱的 std 下界

| 操作 | HYLE ns（Win/Lin/M4） | `std::error_code/expected` ns | 结论 |
|---|---:|---:|---|
| code-only error 构造 | 0.377 / 0.060 / 0.224 | 0.364 / 0.062 / 0.224 | Windows std 约 **1.04x** 快但亚纳秒 TIE；Linux/M4 TIE |
| 四帧 Result OK + consume | 2.565 / 2.478 / 3.058 | 2.209 / 2.624 / 3.328 | Windows std **1.16x** 快；Linux/M4 HYLE 约 **1.07/1.09x** 快 |
| 四帧 code-only Result error | 2.372 / 1.103 / 1.443 | 4.737 / 4.413 / 2.166 | HYLE **WIN 2.00 / 4.00 / 1.50x** |

Windows 的四帧 Result 成功下界是真实、保留的 LOSE；code-only 构造本轮落入亚纳秒 TIE。
`std::error_code` 是平凡析构的两个 word，能力比
owning `Status` 小。它适合只需 category/value 的局部代码，但不能代替 HYLE 跨模块需要的
canonical message/native/context carrier。故它是成本下界和选型提示，不是同能力对手。

## 合同二：携带消费者需要的诊断

| 正常消费者操作 | HYLE ns（Win/Lin/M4） | Abseil ns（Win/Lin/M4） | 配对结论 |
|---|---:|---:|---|
| owning message Status 构造 | 27.300 / 9.876 / 17.412 | 45.982 / 20.017 / 31.054 | **WIN 1.683 / 2.037 / 1.877x** |
| owning message Status 拷贝 | 10.044 / 8.850 / 3.181 | 10.445 / 9.262 / 4.180 | **WIN 1.038 / 1.045 / 1.314x** |
| 前置一层语义 context | 44.016 / 21.686 / 17.401 | 258.235 / 91.417 / 120.739 | **WIN 5.857 / 4.203 / 6.440x** |
| 前置三层语义 context | 78.318 / 43.255 / 42.714 | 761.229 / 277.394 / 339.224 | **WIN 9.704 / 6.480 / 7.721x** |
| message Result 错误跨两个调用边界 | 9.517 / 9.221 / 3.814 | 21.193 / 21.248 / 7.202 | **WIN 2.209 / 2.292 / 1.860x** |
| owning 人读格式化 | 30.843 / 19.741 / 23.787 | 39.864 / 22.874 / 30.729 | **WIN 1.284 / 1.165 / 1.232x** |
| 首错聚合 `Update` | 9.156 / 8.794 / 3.181 | 10.417 / 9.188 / 4.184 | **WIN 1.138 / 1.045 / 1.315x** |
| errno → canonical + operation + native value | 40.062 / 51.330 / 26.238 | 167.139 / 74.727 / 116.480 | **WIN 4.194 / 1.451 / 4.608x** |
| message + Vulkan native value 构造 | 34.507 / 12.122 / 16.477 | 86.585 / 37.709 / 53.482 | **WIN 2.513 / 3.110 / 3.373x** |
| Vulkan native value 查询 | 2.266 / 0.642 / 0.962 | 5.681 / 4.636 / 4.422 | **WIN 2.519 / 7.151 / 4.616x** |

Abseil payload 是可携带多项任意数据的通用系统；HYLE native detail 只表达一个固定
`{domain, int64 code}`。上表的 native 行只说明**同样存取一个原生数值**时 HYLE 更快，
不声称 HYLE 具备或击败任意 payload 系统。这个窄形状来自 Vulkan/CUDA/OS 恢复逻辑的真实
需求：此前只有 canonical code + message，消费者不得不解析 message。

同能力 Abseil 项在三平台没有确认 LOSE。这个结论不外推到未上架的 payload、序列化、
backtrace 或异常系统；参考库有而 HYLE 用户没有需求的能力，不会只为扩大 benchmark 表而
进入 error。

## 空间与复杂度

- 64 位平台：`Status=8B`，`Result<int>/Result<void>=16B`；`absl::Status=8B`，
  `absl::StatusOr<int>` 在 Windows 为 24B、Linux/M4 为 16B。
- OK 与 code-only Status 是 O(1)、零分配；native query 是 O(1)。
- message 构造是 O(message bytes)，heap Status copy 是 O(1) 原子 retain。
- 第一层 `WithContext` 精确生成扁平消息；第二层 rvalue 才进入 context 专属几何 prefix
  storage，唯一所有权且容量足够时只写新前缀。连续 rvalue 链的总复制为 O(最终消息字节)，
  扩容 O(log final bytes)；lvalue/共享值保持 COW，单次仍是 O(context + old message)。普通
  message/native Status 不携带 capacity/offset。
- errno canonical 分类是 O(1)；native description 的取得由 OS/CRT 主导，最终只构造一个
  owning Status。
- Win32 canonical 分类是跨 host 的固定 O(1) switch；Windows 仅在错误路径调用
  `FormatMessage`，最终同样只构造一个 owning Status。这里不增加“高错误率”微基准；其真实
  验收是 platform 删除重复 translator 后保持相同复杂度、统一分类与诊断合同。
- 非法 Result error arm 的 fail-fast 诊断放在冷 `noinline` helper；正常传播帧不复制 abort
  文本和控制流。M4 栈帧从 112B 降到与 Abseil 相同的 80B，这是冷热路径架构修正，不增加
  产品状态。

## 复跑命令

Windows 正式复跑使用 error 专用 fail-closed wrapper。输出必须在 source tree 外，否则创建证据文件
本身会使 `--require-clean` 失败；完成评审后再归档整个目录。wrapper 从不存在的 build tree 开始，
固定 `HYLE_BUILD_TESTS=OFF`、`HYLE_TEST_INSTRUMENTATION=OFF` 与 `HYLE_STATIC=ON` 的
static shipping shape，只配置 `error`，只构建 primary/placement 的四个 error benchmark，顺序执行
2 shapes × 2 executables × 2 orders，每格 15 个独立进程；固定 19 个同合同 pair 与 3 个
`NOT_COMPARABLE` pair，并核对每个 pair 的 run id 与 raw 记录总数。它还保存四门合并摘要、
输入/二进制/build 哈希、PE layout 和压缩反汇编；任一门缺失时不会把局部点值升级为证书。

```powershell
$errorWorkRoot = 'C:\bench-work\hyle-error-msvc-clean'
$errorBuild = Join-Path $errorWorkRoot 'build'
$errorEvidence = Join-Path $errorWorkRoot 'evidence'
$llvmObjdump = 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin\llvm-objdump.exe'

python scripts/run-error-bench-evidence.py `
  --build $errorBuild `
  --absl-dir C:\path\to\abseil-cpp `
  --output-dir $errorEvidence `
  --config Release `
  --fresh-configure `
  --generator 'Visual Studio 18 2026' `
  --architecture x64 `
  --llvm-objdump $llvmObjdump `
  --placement `
  --affinity-mask 0x4 `
  --priority-class above-normal `
  --runs 15 `
  --equivalence-margin-pct 3 `
  --require-clean
```

上例的 `0x4` 是本机采样前选定的单个 P-core logical CPU，不是跨机器默认值；其他主机必须先
按拓扑预声明 mask，不能看过结果后挑核。CPU mask 与 priority 只在四目标构建和反汇编完成后
应用，由 `bench-stat.py` 和 benchmark 子进程继承；manifest 同时记录实际 mask、priority 与
Windows active power scheme。

诊断 dirty 集成树时去掉 `--require-clean`，但必须在结果页和 evidence 目录名中写明
`diagnostic`；它不能替代 clean candidate commit 的正式复跑。不要单独手抄 `bench-stat.py`
pair 名：固定 manifest 在 wrapper 中，名称过期、方向颠倒、任一 pair 漏匹配或记录数不等都会
fail closed。ratio 始终定义为 `RIVAL op/s / HYLE(BASELINE) op/s`。
