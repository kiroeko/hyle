# F1 当前能力与验证矩阵

本页从[当前施工卡](foundation-f1-resource-leases.md)纳入，仅汇总本轮 F1-01至08，不授权新产品范围。
registry 当前七 owner 为 CANDIDATE。产品实现固定于 `fc4ffbd1610eb3b24bf76709dc6e5e25b3d838d7`；
普通样例错误打印修正于 `8bf5b2ebea3021264128df34b25ed7e94964caa1`，CUDA 取消断言候选为
`3034fc595abf5c2570e06dedbf195b34daf3fe72`。后两项没有改变产品 API、算法或布局。
本页未完成最终 red-team/七门裁决；`UNPROVEN` 不能用于 CLOSED。

## 能力与全部新增入口反向映射

主纵切是外部权重/状态范围的 ready→原成熟数学→最后使用后回收；正交纵切是有界分块矩阵的
多 reader、独立 writer、caller output 与来源 alias。两者不含 layer/token/cache 淘汰策略。
成熟来源与 owner 决策沿用[固定研究](../../../architecture/compute-resource-streaming-design.md)，
不是新增模型策略或数学 kernel 的准入。

| 能力 / FIX | 全部入口组 | 层级与承载 | 当前实现 |
|---|---|---|---|
| backing/range/generation F1-01 | `LeasePoolOptions/Stats`；`LeasePool::create/adopt/allocate/stats/valid`；move/析构 | 原子资源控制；SELF 少量 domain，BORROW Mem Buffer/RangeAllocator/ConcurrentBlockPool | COMPLETE；容量/metadata 有界，adopt 成功才接管，不复制 payload |
| 读写/发布 F1-02 | `LeaseImportOptions`；`ReadLease::adopt/valid/size/generation/view/tensor/subview/try_write/reset`、copy/move/析构；`WriteLease::valid/size/generation/view/tensor/publish/reset`、move/析构 | 原子 permission；同 root 保守按整块权限，跨块独立 | COMPLETE；readonly import 永不提升，原生/外部未跟踪 writable aliases 是显式 precondition |
| native use F1-03 | `LeaseUsePoolOptions`；`LeaseUsePool::create/begin/active_batches/valid/close/drain`、move/析构；`LeaseUseBatch::read/write/attach/valid`、move/析构 | 原子完成保活；BORROW 原 CompletionRetention/native retirement | COMPLETE；提交前预留，未接受回退，实际退休后释放，metadata/index 由原 owner 保活 |
| 观察错误 F1-04 | 原 `Completion::poll/wait` 与 CUDA 原 submit/collect/drain；未增加普通用户接口 | 定向 correctness 修复；原 EventSlot/Reaper 与 Graph aggregate | COMPLETE；Windows 与 WSL 原 fault/Graph 退役均有证据 |
| 共同 Direct F1-05 | `LeasedInput/Output/OperationOptions`；`LeasedOperationInvocation::create/valid/in_flight/provider/submit/close/drain`、move/析构 | 复合 glue；原 PreparedOperation/OperationInvocation/Queue | COMPLETE；输入/输出 view 冷预置，自动保留 use；一冷 completion slot |
| common Graph F1-06 | `LeasedInvocationOptions`；`LeasedInvocation::create/valid/in_flight/bind` 两权限重载、`prepare_replay/replay_route/replay_report/reset/submit` 三原 QueueBinding 档、`output/close/drain`、move/析构；`LeasedGraphOutput::valid/tensor/storage_lease/reset`、copy/move/析构 | 复合 glue；原 Executable/Invocation/SubmissionCursor | COMPLETE；原计划/Provider 不变，旧结果阻止重放，caller/source alias 可导出 ReadLease，private output 保留原 owner |
| native Direct/Graph F1-01/05/06 | Apple/Vulkan `MemoryResource::binding/import_readonly`；Apple `Program::input_count` 冷查询；各自 `LeasedInput` 与 `LeasedInvocation::create/valid/in_flight/bind/submit/close/drain`、move/析构；CUDA `ExternalBuffer::into_read_lease/into_lease_pool` | native route adapter；BORROW 原 MLX array/Vulkan Tensor/owned CUDA Buffer 与原 Program/Results | COMPLETE；各自 native ownership，冷 readonly view，不引入共同万能资源/Queue |
| CPU+GPU 组合 F1-06 | CPU+Apple/Vulkan 原 `Invocation::bind(ValueId, TensorDesc, ReadLease)`；CPU+CUDA 用共同 LeasedInvocation | 复合 glue；BORROW 原 mixed plan、native/host bind 与 Results | COMPLETE 实现；每 accepted prefix 在观察前保护，body sentinel 与真实 prefixes 联合 terminal |
| 关闭/回收 F1-07 | 上述 `close/drain/reset`；原 Completion cancel、native Context collect/close/drain | 复用原进展/退役；无新 poller、线程或 reaper | COMPLETE；CUDA cancellation 原返回 Unimplemented 后仍保活，Windows/WSL 新断言通过 |
| 构建/文档 F1-08 | `hyle::graph`/普通 compute routes；普通 sample；detail protocol 头和原 `submit_guarded` 接缝 | 公开 user surface 与内部跨库协议分开 | PARTIAL 验收；Windows CPU+Vulkan static managed 已取证，其他剩余格依下表 |

所有 `valid/size/generation/stats` 只作冷查询；并不承诺这些字段可在原 control handle 的未同步并发修改中读取。
`detail/lease_access.h`、`detail/native_lease_inputs.h`、`detail/retained_completion.h`、
`graph/detail/lease_submission.h` 以及 native `submit_guarded/wait_native` 是已有模块间的内部协议，
由各自 adapter 调用，回指 F1-01/03/05/06/07；它们不构成普通用户的第二种 executor。
未增加 benchmark-only user API。原 borrowed views、编译时常量/effect footprints、Context/resource/allocator
寿命继续按原合同；F1 不提供跨 GPU、历史版本、自动 CoW、文件管线或批量 copy。

## 风险证据索引

以下链接指向有实际命令/身份/原日志的收据，不把旧收据的历史 pending 句子恢复为新的待办授权。
证据复用仍需最终逐文件/公开合同 hash 复核；未变化的产品不因文档提交重复构建。

[本轮身份复核](../../../../evidence/correctness/foundation-f1-identity-audit/receipt.json)已逐一核对六组
收据的原日志 SHA256；原 common drain 的逐 check hash 也匹配。`3034fc595` 与 shipping product、
SDK、sanitizer、native/mixed risk 的 `include/src/cmake/third_party` 及根 CMake 文件逐文件一致；
更早 common drain 的实际 compute/CUDA/common Graph 产品 scope 也一致。Air 当前 `e39905900` 的
同产品 scope、共同核和拟选 native Vulkan risk 测试也逐文件一致。这里只签原语义范围的文件身份，
不是用源一致取代未运行的平台/交付格。

- C：共同核、Completion/OwnedResult、权限/溢出/OOM/跨线程协议与 ASan/UBSan/TSan：
  [host sanitizer](../../../../evidence/correctness/foundation-f1-host-sanitizers/receipt.json)，
  [最终边界](../../../../evidence/correctness/foundation-f1-mixed-leases-initial/receipt.json)。
- D：真实 cuBLAS Direct、错误 binding 后恢复、八次 metadata 计数、真实在途弃置与最后回收：
  [Direct](../../../../evidence/correctness/foundation-f1-direct-import-windows/README.md)。
- G：实际双 native Graph 分支、观察前保活、private/caller/source 输出、unknown prefix/drain：
  [common drain](../../../../evidence/correctness/foundation-f1-common-drain-windows/receipt.json)，
  [具名测试](../../../../tests/graph/aggregate_native_retirement_test.cc)。
- N：Apple/Vulkan readonly owning 导入与原生数学/来源 alias：
  [native binding](../../../../evidence/correctness/foundation-f1-native-bindings-initial/README.md)，
  [native submit](../../../../evidence/correctness/foundation-f1-native-submission-initial/receipt.json)。
- L：实际 Apple/Vulkan 租约 cancellation/abandon/accepted prefix failure、零冷 metadata 后同源恢复：
  [native lease risk](../../../../evidence/correctness/foundation-f1-native-lease-risk/receipt.json)。
- M：三个 CPU+GPU 原 mixed 数学；Apple/Vulkan 新 ReadLease branch 的取消、弃置和 accepted CPU prefix
  native refusal：[mixed](../../../../evidence/correctness/foundation-f1-mixed-leases-initial/receipt.json)。
- S：普通 Direct/Graph 与 installed imported mixed、ON/OFF 头、absence、迁址/DLL：
  [交付](../../../../evidence/correctness/foundation-f1-lease-delivery-initial/receipt.json)。
- U：WSL 独立 CUDA fault2、cuBLAS Direct1、Graph retirement4，以及 Windows 新取消单用例：
  [CUDA 本批风险](../../../../evidence/correctness/foundation-f1-cuda-final-risk/receipt.json)。

CUDA 的拥有 pool 转换还由原 `CudaBackend.AdoptedNativeLeasePoolOwnsResourceAndBorrowedPayloadCannotMasquerade`
取证：borrowed 两种转换拒绝且保持原 handle；拥有转换的 max_blocks0 拒绝不消费；成功转换后的 native
reader/subview 在 pool root 丢弃后仍读到原数据，最后 reader 消失后原 native allocation 才释放。
原 PASS 在 [instrumented CUDA 日志](../../../../evidence/correctness/foundation-f1-common-drain-windows/instrumented-cuda.log)；
该具体产品 scope 与当前 candidate 相同。这是 native ownership 证据，不代签未运行的平台。

| 能力旅程 | 正常/边界 | failure/OOM/overflow | move/ownership/teardown | cancel/concurrency/late use | 主/正交消费者 | 平台/交付 |
|---|---|---|---|---|---|---|
| F1-01/02 owner/permission | C 的有限块、subview、generation、多读阻写 PASS | C 的 checked arithmetic/预算/拒绝不消费 PASS | C/D/N 原 backing 与 readonly alias PASS | C 跨线程 reader、独立 block PASS | D/N/S 真实 ready 数据与矩阵/alias PASS | 原 scope source identity 已复核；剩余平台/交付格仍按下表 |
| F1-03/04 native retirement | C/D/G actual retirement PASS | C/G/U unknown error 不提前回收、首错与最终排空 PASS | D/G/N/L dropped handles 仍保活 PASS | C/L/U 取消不证明退休、仍保留 native use PASS | 原数学与 actual prefix 保留，非 mock math PASS | Windows/WSL CUDA / Air Apple / Windows Vulkan 已取证；其余细分按下面表 |
| F1-05 Direct | D/N/S/U 四路线成熟数学 PASS | D binding rollback、L cold OOM/accepted failure PASS | D/N/L last source alias/use 回收 PASS | L/U native cancel/abandon 与原不支持取消合同 PASS | 同时含 readonly import、bounded storage、MatMul PASS | S 当前 ordinary source/static/shared PASS；最终链接矩阵复核尚未裁决 |
| F1-06 Graph/组合 | G/M/S 原计划与 scalar reference PASS | G unknown prefix 与 M closed-native suffix 原错 PASS | G/N/M 旧结果/source/caller/private output PASS | C/L/M 原协议与实际 cancellation/abandon PASS；跨平台 fault 复核 UNPROVEN | 无新数学，原 CPU/native/CPU consumer PASS | Windows CPU+Vulkan imported static managed PASS；其余链接格依下表 |
| F1-07 close/drain | C/D/G/N/S held Completion drain PASS | G/L/M 失败 slot/首错/拒绝后继 PASS | D/G/L/M 最后 guard 与保留 reader PASS | C host concurrency；L/M/U actual native 路线 PASS | 关闭本 owner、旧结果仍有效；无 device-wide 隐式 wait | 借用寿命/各 native mechanism 逐平台终审 UNPROVEN |
| F1-08 普通入口/P0 | S 公开 sample/自动 route 闭包 PASS | S absence、ON/OFF 新头 PASS | S actual SDK move/DLL PASS | 不新增进展引擎；代码账单已记录 | source/static/shared 普通 facade；细分保留 | 余下 CPU+CUDA installed mixed、Air风险格、完整 P0/red-team UNPROVEN |

CPU 已接受的同步工作没有可取消的在途 GPU 命令；其 inline Completion 在返回时已退休。
CUDA 的 cancel 是原 `Unimplemented`，需要证明请求后 use 仍有效，不声称成功停止 stream。
unknown-fence helper 的三个受控协议用例不代签实际 native GPU；M 的 native admission refusal
也不改写成“未知 fence 已资格化”。实际 CUDA unknown 与 Apple/Vulkan accepted failure 各按 G/L 分开。

## 平台与交付当前快照

此表的 PASS 只指 S 中相应 ordinary/math/header consumer；故障格使用独立日志，不由 SDK 数学代签。

| 实际平台 / 路线 | source 普通/混合 | relocated static 普通/混合 | relocated shared 普通/混合 | fault/lifetime 当前证据 |
|---|---|---|---|---|
| Windows CPU/CUDA | PASS / PASS | PASS / UNPROVEN（共同 lease 的静态导出已证明；mixed consumer 尚未单独记录） | CUDA-only 普通 PASS；CPU+CUDA mixed 未单独记录 | D/G 与 CUDA 原 fault PASS；U 新 lease cancellation 断言 PASS |
| Windows CPU/Vulkan | PASS / PASS | PASS / PASS | PASS / PASS | L/M PASS |
| MacBook Air CPU/Apple | PASS / PASS | PASS / PASS | PASS / PASS | L/M PASS |
| MacBook Air CPU/Vulkan | PASS / PASS | PASS / PASS | PASS / PASS | M admission refusal PASS；native L 对该平台尚未单独记录 |
| WSL CPU/CUDA | PASS / PASS | PASS / PASS | PASS / PASS | C host sanitizer、U 本轮 CUDA fault/Direct/Graph PASS |
| WSL Vulkan | 未签 GPU 数学 | 未签 GPU 数学 | 未签 GPU 数学 | 实际 loader 仅 llvmpipe type4；当前卡按实际 GPU 资格条件，无合格 GPU |

Windows CPU+CUDA mixed 与 Air Vulkan native fault 的证据能否由已测共同/原生机制组合覆盖，
需要终审按差异与合同裁决；不能把未单独运行写成真机 PASS，也不能在未定界时启动扩大测试。
Apple on Windows/WSL 和 CUDA on Air 因 native 系统不支持是 NOT_APPLICABLE；保持 OFF 头/absence
资格，不制造虚构 binary。没有 Mac mini 或第二台 Formal Air 的资格。

## 七问与七门的未完成项

1. 依赖沿原 CPU成熟 Provider、cuBLAS/CUDA、MLX/Metal、ggml Vulkan/IREE 锁；本轮不升级，不新借 peer runtime。
   来源、版本和许可在原 dependency manifests 与设计研究；最终 diff/锁复核待红队。
2. F1-01至08 实现与上表公开旅程已映射；范围不含 F2/F3，不用原子 lease 代签文件流送。
3. 下层 Mem/Util 原范围、池、refcount、SpinLock、allocator；原 Completion/native runtime 与 Graph plan 均复用。
4. 当前卡 P0 已逐行收费 O(R²) 去重、O(R)/O(P) 释放、引用/锁/notify、nullable private 字段/分支和普通 Graph
   aggregate 新成本。八次 metadata 计数仅签注入 allocator；vendor 不透明成本未假定为零，无 timed winner/P1。
5. 本轮没有 C/Native FFI 或跨语言产品需求；内部 detail exported protocol 仍是同工具链 C++ shared 边界。
6. S 保存三个 delivery 的实际 facade/runtime/迁址证据；上表未单独资格的格仍 UNPROVEN。
7. [Compute guide](../../../user/compute.md)、[Graph guide](../../../user/graph.md)与
   [普通消费者](../../../../samples/package/graph/resource_leases/)写明 permission、借用根寿命、一 completion slot、
   旧结果拒绝、readonly import、实际 drain。最终资格措辞待全部结果，不提前称 CLOSED。

`scope/architecture` 的准入与形状有源码/consumer/固定研究支持；`correctness/delivery/usability_docs/
red_team/engineering_performance` 尚不能给出整体 PASS。candidate、LOC、source hash 复用核对、独立有界红队
和最终七门必须在当前卡形成 terminal 后才能把七 owner 进入 CLOSED。当前 candidate 已正式绑定
`3034fc595`，LOC/source identity 已有上文证据，未完成的平台/交付与独立 red-team 继续保留 UNPROVEN。

不可变 `3034fc595` 的统一 [LOC self-check](../../../../evidence/correctness/foundation-f1-identity-audit/loc-current.json)
通过：全库2,258,724 code lines/5,668文件，自实现不含测试165,788/666，Compute+Graph 的同33路径
实现并集70,561/315。使用原统一脚本/明确边界；vendor/tests/samples/docs 不计入后两项，
这些数字不是产品进度或数学 kernel 数量。完整 stdout/工具身份与 self-check 存在收据中。
