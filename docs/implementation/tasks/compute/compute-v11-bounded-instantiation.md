# Compute v11：已编译域的冷实例化

## Closure card

- component / owner：`compute`；baseline `40693c5a`，承接未 terminal 的
  [v10](compute-v10-owning-custom-execution.md)。C93 外部身份的真实资格与完整 foundation 验证继续保留。
- trigger：Graph G101/G102 与 Native N22 的既定动态 token 旅程：proposal 已能描述完整域，但 prepare
  只接 exact 描述，Invocation 也没有从已编译域固定具体尺寸的通路，因而不能兑现 bounded capability。
- 正常用户形状不变：Graph 编译封闭域，创建 Invocation 时固定尺寸，之后重复 submit；direct Compute
  仍为 exact prepare。只增加 factor 间冷 seam，不公开第三个 runtime、字符串形状或新的 FFI。
- 范围：C110--C112；复用现有形状程序、PreparedOperation/OperationInvocation、Context admission、
  Error、Mem allocator、Queue/Completion。Native 只实现 N22 的 normalization 探针，不扩 family；
  Graph 的 variants/整体预算仍归 G101/G102/G106，不由本卡代签。
- non-goals：submit-time shape/JIT/search、自动 domain solver、kernel 重写、热路径动态 wrapper、
  新 provider/算子目录、公共构建/治理/性能仪器改造。
- 预算：一次表示/实现与一轮纠正；反例先覆盖域所有权、尺寸资格和失效清理，不用重复计时找答案。
- 验证：Compute/Native、Graph/oneDNN 直接回归、逐头、CPU/CUDA source/static/shared；完整候选再跑
  Linux/两台 Apple、适用 sanitizer、独立红队和一次 foundation P1。实现阶段 P0，不启动全库/P1/CI。
- stop：实例化重新运行 proposal/prepare 或编译 tactic、接纳用户工作、增加第二套完成/运行时，或必须
  修改 CLOSED backend。出现这种反证先修合同，不用测试夹具掩盖。

## AUDITED：有限表示和改动集

| ID | 实现合同 | 验收 |
|---|---|---|
| C110 | private prepare request 携带完整域；持有者复制 program/guards/axis maps，不借用 Graph scratch。非 exact 只接受 entire-domain proposal 和明确冷实例化 callback | 源 metadata 释放/修改后仍正确；未知 coverage/缺 callback 拒绝；不把 envelope 当合法点 |
| C111 | 从不可变 domain root 冷生成 exact PreparedOperation 实例，绑定 checked concrete descriptions；Provider 只实例化已编译参数/执行状态，不重新编译或选路 | 默认 root 不可直接执行；范围/guard/shape 校验在 callback 前；子实例保活 root，错误与部分制品回收 |
| C112 | exact direct、typed custom、oneDNN 和 Graph 保持原回调执行路线 | 无公开 API/CMake 开关变化；普通 create/submit 无额外分发层；三交付一致 |

domain root 沿用 PreparedOperation 的内部所有权骨架，但只由 Graph factor seam 创建；它的描述是
envelope，不能交给普通 `create_invocation()`。`OperationPreparationAccess::instantiate()` 接收完整
dimension scratch（独立槽由调用方填、派生槽由共用求值器写入），验证后返回同一普通 exact handle。
Graph 后续为具体 Invocation 保有该实例；其输入输出描述、alias 和 workspace 都已固定，warm submit
仍调用原来的单层 ProviderPreparedVTable，绝不再检查域或走另一个动态 dispatch。

root 的 shape metadata 只在非 exact 域分配，exact PreparedState 只增加冷所有权字段；子实例保有 root
至自身 Provider artifact 销毁之后。实例化允许 per-invocation metadata/run-state 的冷分配，不允许
生成新 kernel/tactic、执行 tensor payload 或重查 Provider。prepared root 的 workspace 是域上界，具体
实例须保持 runtime/alias 合同且 workspace 不超该上界；无法给出完整承诺的 Provider 仍 exact-only。

Native N22 的证明范围是 dense float32 RMSNorm、固定 hidden width、任意已验证的有界 prefix dimensions，
可选固定 width 的 scale。input/output 必须引用相同轴表达式（或相同固定值），不把相同 bounds 的不同
symbols 当相等；compile 上界证明字节/行数可表示。实例化只复制既定 NativePlan，更新行数和已有
executor 的冷分块参数；SIMD kernel 指针、精度、alias 与资源要求不变。其它 family 暂不承诺 bounded。

这与已接受 Graph §4.3/§6.2 和 Compute 域/冷边界一致，是既定 foundation 缺口的实施，不重启架构调查。
本节为从已 AUDITED 的 Compute 后继绑定本卡时的实现前审计，不签新的 Candidate/PASS/CLOSED。

## 2026-09-15 冷实例化检查点

- C110/C111 已实现：私有 Provider 合同升至 10；prepared domain 拷贝 program/guards/axis maps，一次
  验证并绑定具体描述后创建普通 exact PreparedOperation。子制品先销毁，再释放所保有的域根；没有
  新公开头、FFI、CMake 开关、runtime 或 warm callback。Native 实例只重绑行数与既有 executor 分块。
- `tests/compute_provider_native/provider_shape_test.cc` 新增 15 个用例：含原有 exact-only 反例共 16 个。
  覆盖带 guard 的乘积域所有小尺寸、加权/非加权数值、多个 prefix 轴、zero/zero-hidden、独立 symbol
  不能冒充相等、动态 hidden/scale 拒绝、源 metadata 修改/释放、每次冷分配失败与恢复、并发实例、
  普通 exact 不变，以及 callback 前 schema/guard/coverage 检查。合约破坏夹具覆盖缺 hook、runtime/
  workspace/alias 变化、缺 submit、递归域与部分错误制品清理；不把夹具当可执行 Provider。
- 正常宽度 8192 的 1/3/9 行验证复用同一 SIMD 内核及既有 executor 分块：与 direct exact 的 fan-out
  相同，输出完整写入；不以此代替性能计时。小域重复 submit 的 metadata allocator 调用数不增长。
- 两处首次编译错误是测试未使用 Operation 的显式构造、未解包 alias Result；另一个生命周期断言错误
  地要求销毁最后一个 Context 后仍能接纳工作。均按原合同纠正夹具，产品不放宽。一次 CTest 正则漏掉
  `hyle_` 前缀，返回零测试，不计 PASS；随后按真实注册名完成下表。

| Windows Release 验证 | cache（`C:/hyle-scratch/` 下） | 实际结果 |
|---|---|---|
| static Compute/CPU/Native/Graph 与逐头 | `compute-route-v8-static-20260914` | 9/9；最后新增 executor 用例后仅 Native 重建回放 1/1 |
| shared + oneDNN direct/composition/Graph/region 与逐头 | `compute-route-onednn-v8-20260914` | 15/15；Native 79/79、Graph execution 198/198 |
| CUDA runtime、late failure、Graph 与逐头 | `compute-graph-v7-cuda` | 8/8 |
| source CPU/CUDA custom consumer | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA custom consumer | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA custom consumer | `custom-execution-v10-static-consumer` | 2/2 |

命令沿用 `hyle-windows.ps1 -Action Build/Test -BinaryDir <cache>`，只构建表内 test/sample target，
`--parallel 2` 编译、`--parallel 1` 测试；SDK 从 `custom-execution-v10-{shared,static}-build` 显式
`cmake --install ... --prefix ...-sdk`，未安装到 Program Files。shared producer、SDK、CPU/CUDA 实际加载
目录的 Compute DLL SHA-256 均为
`0C5B6216F3BBBF81EA3326E7DF67B69AC2C0F8186B2E457B5B303ABD1D09399C`。

P0：令 D/G 为维度/guard 数，I/O 为端口数，rank 上限固定为 8。域根 metadata 复制时间/空间为
`O(D+G+I+O)`；冷实例绑定/验证为 `O(D+G+I+O+I*O)`（最后一项是完整 alias 合同对账，外加 schema/
Provider 自身已声明的成本）。Native root 比 exact 多一块域 metadata；每个具体实例分配 NativePlan
与普通 PreparedState 两块，并保活共享根。PreparedState 增加两个冷 ownership 指针，ProviderPrepared
增加一个冷 callback，exact 不增加 allocation 次数；InvocationState、公开 handle 布局和 warm submit
没有改动。该必要冷成本不是零成本，也不是性能胜利；完整 variant resident/冷峰值仍由 G106 处理。

selector 基线 `40693c5a` READY，86 项有据跳过；closure（含 `--previous-ref 40693c5a`）、docs 和
diff 检查通过。无 Backend/已 CLOSED 依赖、CMake、公共验证仪器或 CI 改动。尚未运行本候选的 Linux/
两台 Apple、sanitizer、独立终局红队或 P1；这些仍在共同 foundation 验证卡中，不由本段 Windows 代签。

自实现不含测试：Compute **16,670**、Graph **11,673**，合计 **28,343**；全库自实现 **123,469**，
全库代码 **676,424**，统一脚本 self-check 通过。仍为 `AUDITED`，不是整个模块 terminal：C93 真实
profile 资格、G101/G102 Graph bounded/finite variants、G106 聚合预算及真实平台探针继续保留。
