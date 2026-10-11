# F1 candidate 红队记录

当前[施工卡](foundation-f1-resource-leases.md)纳入本记录；候选
`3034fc595abf5c2570e06dedbf195b34daf3fe72`，产品代码固定于 `fc4ffbd16`。
reviewer：Codex，candidate 冻结后的单独源码/证据复核阶段，2026-10-11。
记录时钟 `2026-10-11 08:37:39 UTC`；总预算45分钟的一遍定向审查，至多一次已有证据回指，
不扩大 F1、不计时采样、不为发现未来能力新增实现。剩余已冻结实际验证结果允许附回同一 findings，
文档/证据变化不再开启第二轮产品源码扫描。

## 源码反证与成本检查

| 范围 | 反证问题与实际核对 | 当前结论 |
|---|---|---|
| 资源、permission、generation | `resource_leases.cc` 的 prepare/adopt 在 fallible metadata/预算准备后移动 backing；allocate 检查 size/alignment/generation，上界由原 RangeAllocator/Buffer 维持；subview 在当前范围内，writer/ref/use 受 domain lock | 未发现本轮产品缺陷；拒绝不消费、OOM、溢出、多 reader/独立 block 的原 C 测试与 sanitizer 可回指 |
| 最后释放与 drain | domain/use pool 的临时 ref、decref、uses/active release-store、notify 和最终锁屏障；资源 owner 在 backing/metadata 析构后释放；旧 reader 不属于 drain 等待对象 | 元数据与 backing 没有在 use 归零前释放；Context/resource/allocator 与原 control 观察须满足文档借用/同步合同 |
| Direct/完成代理 | `leased_operation.cc` 捕获原 views 后 admit；原 submit 拒绝回退；fresh retention 与有效 Completion 接纳不容失配；`retained_completion.cc` 安装 native retirement latch，转交 unknown attachment，旧 completion slot 阻止重放 | 原 CUDA数学与 held drain/abandon/U 取消请求验证；一 proxy dispatch、锁/ref/counter 冷热成本已收费，不把 Status 错误当 GPU 完成 |
| Graph/聚合 | aggregate 在观察前给各 child 挂独立 Invocation ref/嵌套 retention；outer sentinel 与 actual prefixes 联合回收；opt-in SubmissionCursor 在任何后续观察前 protect；caller/source/private output owner、旧结果和导出 read 保留 | G/U 实际双 native 分支与 accepted dependency rejection；普通 Graph 的新增 O(P) metadata/ref/attachment 成本已明示，无新 Provider 搜索或数学/copy 命令 |
| Native Apple/Vulkan | readonly buffer/array/tensor holder、精确 allocator 与 native root；Vulkan 四字节 origin/tail 在已拥有 capacity 内；guard 只在原 native accept/retire 处交接，runtime lock 外执行可能释放外部 owner 的析构 | N/L/S 的原数学、readonly alias、OOM、cancel/late prefix 有明确实际证据；不支持未跟踪 writable aliases，不推断第三方 producer fence |
| CPU+Apple/Vulkan | 两原 RunState 仅 nullable lease owner；冷 inputs/prefixes 表及 checked capacity；每 prefix retain-before-observe，body sentinel；发布 terminal 后才解除原 failure/result exclusion | C/M 与 SDK ordinary mixed 覆盖；未知 helper 协议不伪称 native unknown。原 Sched worker/native runtime 是唯一执行者 |
| API/组件/布局 | 全部新增 user API 与内部跨库协议回指 F1；两 mixed public ReadLease 参数依法 PUBLIC_DEPS compute；旧 Completion、PreparedOperation/OperationInvocation 公开头相对设计基线不变，CUDA EventSlot 未增字段，仅新增同存储 enum 值 | 没有新 user build switch、FFI、万能 Queue、reaper 或数学 kernel。公开 C++ shared 仍是原同工具链/stdlib 合同 |
| P0/benchmark | 有界 O(R²) 去重、O(R)/O(P) 释放、native nullable字段/ordinary 分支、普通 Graph private child 增长、proxy成本、opaque vendor metadata 均在本卡账单；baseline 到 candidate 的 bench/公共 scripts 无修改 | 源码成本检查未发现隐瞒；八次 injected allocator 计数只签 lease metadata，不称全 vendor 热分配为零，无 P1/P2或 SOTA 结论 |

源码锚点为当前卡逐项链接、[API/风险矩阵](foundation-f1-verification-matrix.md)、
[身份与原日志 hash](../../../../evidence/correctness/foundation-f1-identity-audit/receipt.json)及
F1 允许的 include/src/cmake diff。已关闭 Mem/Provider/lowerer 的实现不在本轮重审或改动范围。

## Findings 与处理

| ID / severity | Finding | Disposition / 证据 |
|---|---|---|
| RT-01 LOW docs | 新 API 矩阵漏列 CUDA `into_lease_pool` 与 Apple `Program::input_count`，不能称“全部入口已反映” | 已补矩阵；input_count 由 native leased create 的容量准备消费；into_lease_pool 的原 native owning/rejection/free 用例已在 instrumented CUDA log PASS，相关产品 scope 与 candidate 相同；没有产品修改 |
| RT-02 MEDIUM evidence | Windows CPU+Vulkan static 组合、Windows CPU+CUDA installed mixed 与 Air native Vulkan/控制格未逐一取证；不能由单路线或不同 OS 签整包 | PARTIAL/OPEN；CPU+Vulkan static ordinary/mixed、CPU+CUDA static mixed 和 Air 27/3/5 原 filters 已实际通过；只剩 CPU+CUDA shared 正在运行，未验保持 UNPROVEN，不重开源码审查 |
| RT-03 LOW docs | 早期子收据的 pending/owners AUDITED 句子属于历史候选，不是当前阶段或新授权 | 当前阶段只看 registry 的 CANDIDATE 和当前矩阵；保留旧收据原事实，最终当前交付 aggregate/guide 的措辞随真实结果更新 |

源码部分审查通过，未发现需要把 candidate 退回 AUDITED 的产品问题。
整体 `red_team` 仍 **INCONCLUSIVE**：RT-02 的适用证据未齐，不能用源码审查绕过实际验证或提前 CLOSED。
完整 P0/七门 terminal 同样须等剩余实际格与最终 doc/status/checker；未验不写 PASS。
