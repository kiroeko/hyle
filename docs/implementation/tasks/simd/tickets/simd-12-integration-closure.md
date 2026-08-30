# SIMD-12：集成、文档、六问与 closure

- ticket state：`ACTIVE`
- kind：integration/verification
- depends on：SIMD-11
- budget：1 工程日；性能预算 90 分钟
- frozen candidate commit：`c24327164348c3bc7da9bfca424593f84864126f`
- frozen candidate tree：`55cffd93284e275697d475f0325bd2697e3224dc`

冻结不可变 candidate，运行全 family correctness smoke、受影响 sanitizer/real-ISA、Platform/Container/
Compute 直接边以及 source/static/shared、native/exact/fat/runtime journey。汇总前票性能证据，只重放因后续
依赖或公共 backend 改动失效的 family 与预冻结 sentinel，不重新制造全笛卡尔积 benchmark。

重写 durable SIMD architecture 与 user guide，历史 task 留在 implementation；回答第三方、推理完整度、
公平性能、FFI 清洁、三交付、C++ 易用性六问，执行独立红队、closure/docs checker 和四项 LOC。任一适用
gate 未运行或仪器无资格必须 `BLOCKED/INCONCLUSIVE`，不得写成模块级 SOTA 或假 PASS。

## 冻结验证清单

本票不再接纳新 capability、API 或 benchmark shape；只有下列 gate 复现的 correctness/security、依赖、
交付、公开合同或既有 P1 verdict 阻断项才允许最小产品修复，修复后必须重冻 candidate。

1. 聚合 SIMD-00～11 的 terminal record、MUST/ROUTED ledger、第三方 lock/license 与已知限制；确认没有被
   后票重新引入的 `PARTIAL/MISSING + KEEP`。
2. 在干净 Windows selected-component Release 构建运行完整 public SIMD suite、适用 fixed-ISA suite、
   exact artifact、header self-containment 和 generated dispatch 合同；在 WSL GCC/ASan 与 macOS NEON
   运行最终 public/fixed suite。SVE/RVV/AVX-512 只继承最终产品身份之后已完成且源码未变的真实 gate，
   不重复 QEMU/SDE timing。
3. 重放 source embedding、搬迁 shared SDK、搬迁 static SDK 的公开 sample；核对 Runtime/exact/native/fat
   目标与中央 component graph 的实际 link closure。
4. 对每个 P1 family 比较其 evidence candidate 与最终 product paths。算法 body/合同/计费边界未被后票修改
   才聚合原证据；共享 Runtime 只做一次普通公开调用 sentinel 与 code-path inspection。若 family body或
   workload materially 漂移，才重放该 family 的冻结 instrument，禁止跑全笛卡尔积。
5. 六问审计固定回答：直接/间接第三方；推理 north-star 与 ggml/llama 消费完整度；同合同公平性能；
   Native FFI/C ABI 清洁负空间；source/static/shared 与 exact/fat/native 交付；普通/expert C++ 易用性和
   architecture/user/developer/implementation 文档权威。
6. 独立终轮红队只攻击冻结候选的公开 ownership、alias/OOM、NaN/精度、ISA fallback、dispatch metadata、
   依赖升级和交付边界；新愿望进入 watchlist，不借红队重开已完成 ticket。最终运行 docs/closure/dependency/
   diff checker，登记四项 LOC、receipt、reopen triggers，并裁决模块 `CLOSED` 或 `BLOCKED`。
