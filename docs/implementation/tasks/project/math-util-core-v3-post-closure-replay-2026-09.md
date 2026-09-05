# Math / Util-core v3 关闭后回放（2026-09-05）

<!-- MODULE REPLAY RECORD: ACTIVE -->

> 本任务只读回放已经 `CLOSED/FROZEN` 的 Math v3 与 Util-core v3。回放本身不修改产品；只有下述
> 可复现 finding 命中既有 reopen trigger 后，受影响组件才进入独立产品轮。

## 身份、范围与预算

- reviewed Math round / candidate / end / anchor：
  `math-v3-integer-policy-superset` /
  `ff20319408372013fec6818c35f4ba7a7c79440c` /
  `5ead527ffa296e778edb21e5eabd80b51aa90b3f` /
  `1825727c5326138126ac222e121e7386539b5cab`；
- reviewed Util-core round / candidate / end / anchor：
  `util-core-v3-foundation-utility-superset` /
  `763574765d22f8b3f6dbd6046aee124d1117eea3` /
  `5ead527ffa296e778edb21e5eabd80b51aa90b3f` /
  `1825727c5326138126ac222e121e7386539b5cab`；
- replay source commit / tree：
  `66ac957109bff59262bc3ebdf7f2478fb7c67070` /
  `f7d1eff10c4675828e362175ee1129d5a895bfe2`；
- closure tasks：
  [`Math v3`](../math/math-v3-integer-policy-superset.md) 与
  [`Util-core v3`](../util/util-core-v3-foundation-utility-superset.md)；
- reviewer / date：Codex / 2026-09-05；
- budget：90 分钟只读审计；每个能力族最多两个 strong peers；至多一个 Util-core 定向产品批次、
  一次 P1 qualification 和一次交付纠正。公共仪器、其他冻结模块产品和 benchmark workload 不在本
  replay 内修改。

## 固定检查面

1. 从普通 C++ 调用而非 benchmark 行反查每个新增入口，检查命名、推导、ownership/lifetime、失败、
   `noexcept`、误用诊断和相邻标准类型互操作；
2. 对 Math 做 P0 正确性/UB/code-shape 回放；对 Util-core 的 FunctionRef 使用既有同合同 Abseil P1
   contract，不给 Scope/UniqueResource 发明纳秒排名；
3. 独立验证 factor-only 以及 `source + relocated shared/static` 产品交付；
4. 扫描当前冻结组件对新 API 的语法消费、同合同重复实现和中央 component edge。新增可用 helper
   本身不强迫 consumer 重开；只有破坏性接口漂移、错误调用或无法合理保留的重复机制才形成 trigger。

## 只读 finding

### Math：暂未发现产品 trigger

- candidate 到 replay source 的 Math 产品、测试、样例和组件图输入无漂移；公开入口没有长度、地址、
  执行顺序、peer 名称或其他 benchmark seam。scalar 入口保持 O(1)，span aggregate 保持 O(n) 单遍；
- `try_sum/product(std::span)` 要求调用点显式形成 view（例如 `std::span{shape}`）是已文档化的
  caller-storage 合同，不为某个固定 N 特化；mixed-type 只由显式 destination cast 承担，避免隐式整数
  promotion；
- W1-WIN fresh factor tests 为 20/20，Math source、搬迁后 shared 与 static consumer 均通过。最终 verdict
  仍等待 WSL/macOS P0 replay、文档/依赖扫描和公共 checker。

### Util-core：两个可复现产品 trigger

1. `UniqueResource` 无条件要求 deleter `nothrow` copy。普通 move-only、`nothrow` move deleter 不能实例化，
   即使资源是一个 `int`/pointer token；W1-WSL compile probe 在 `unique_resource.h` 的 class assertion 和
   constructor constraint 处稳定失败。Library Fundamentals v3
   [`unique_resource`](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/n4939.html#scopeguard.uniqueres)
   的成熟边界允许 deleter 可复制，或 move-constructible 且 nothrow move。这击中 v3 的 resource-
   ownership / C++ API-usability trigger；
2. `FunctionRef{nontype<&Decoder::score>, decoder}` 在 target signature 可唯一推出时仍无法 CTAD，必须重复
   `FunctionRef<R(Args...) const noexcept>`。W1-WSL compile probe 复现 deduction failure；锁定 Abseil
   `989c696cd131fdaba300c44eb86e1c00fd3a4b56` 的同形 non-noexcept member route 可自然推导。v3 自己写明
   “CTAD where unambiguous”，所以这不是新增 convenience 偏好，而是已冻结易用性合同的漏验。

mutable FunctionRef 转成 const-signature view 的敌意 codegen probe 在 GCC 上与直接调用完全相同，且与
Abseil 同形；它没有形成第二次运行时间接调用，不是 trigger。旧 P1 bundle 的 candidate、peer、文件与
provenance 均能严格验证，但 verifier 返回
`CURRENT_POLICY_EVIDENCE_REQUIRES_GATE_REPLAY`，因此修正后的新 candidate 必须重新 qualification，不能
沿用旧点值。

## 中间裁决

- Math：`CONFIRMED` 待剩余固定门完成；保持 `CLOSED/FROZEN`；
- Util-core：**`PRODUCT_TRIGGER`**，trigger ID
  `UTILCORE-RESOURCE-MOVE-CTAD-2026-09-05`。只重开 move-only nothrow deleter transfer、checked factory
  conditional `noexcept`、可唯一推出的 nontype bound-callable CTAD，以及相邻 tests/sample/docs/P0
  code shape；其余 Util-core 能力、依赖和旧 v2 性能合同不重开。

