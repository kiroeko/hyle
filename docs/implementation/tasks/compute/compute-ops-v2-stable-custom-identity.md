# ComputeOps v2：稳定的 typed custom operation identity

## Closure card

- component / owner：`compute_ops` / ComputeOps；round=`compute-ops-v2-stable-custom-identity`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮记录为
  [compute-ops-v1](compute-ops-v1-schema-and-catalog.md)；
- trigger 与证据：`Operation::custom_as<T>()` 保存了 `OperationId`，却仍以模板静态 vtable 地址判断类型。
  source/static/shared 两侧可拥有语义相同但地址不同的 inline variable，地址不是 schema 身份；
- 正常 consumer 任务：一个 provider factor 从另一个 shared factor 或源码 consumer 收到 typed custom
  operation，并按显式 schema/version identity 无分配解码 descriptor；
- 主产品纵切 / 具名来源：Graph/Provider 跨 factor custom operation；反偏置纵切为同一 translation unit 内
  构造、查询与 infer，确保修正不引入 registry；
- scope：让 `OperationId` 明确承担 schema+version identity，保存并检查 descriptor byte size，提供稳定的
  cold-path identity/query；保留有界 trivially-copyable inline payload 与 traits/infer callback；
- non-goals：Graph shape domain、provider route、全局 operation registry、动态 plugin/Native FFI、扩大算子目录；
- 直接依赖 / 直接消费者：依赖 Core/Error；Compute、Graph 与 provider factors 消费；
- 性能 tier 与原因：`P0`。构造和查询仍为 O(1)、零分配；steady execution 不调用 typed decode；
- 调查与纠正预算：一轮 layout/identity 实现、一轮三交付 consumer、一轮红队；
- 预声明验证：同 TU、跨 source/static/shared、错误 ID/size、traits/infer、逐头、sanitizer、closure/docs；
- stop / blocker：若稳定身份要求进程全局 registry、RTTI 或热路径查找，退回 `SCOPED`。

## 能力与形状审计

| 能力族 | 层级 | 当前事实 | 覆盖 | 动作 |
|---|---|---|---|---|
| 内建 typed operation | 原子机制 | 20 个内建 family 由 variant index/typed descriptor 表达 | COMPLETE | KEEP |
| custom schema identity | 原子机制 | `OperationId` 已保存，但查询仍比较 vtable 地址 | PARTIAL | FIX |
| bounded shape proof | Graph/provider 机制 | 是否接受完整 shape subdomain 取决于 Graph 与具体 provider，不是 operation value 的单一属性 | ROUTED | 不放入 ComputeOps |

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| O20 | `custom_as<T>()` 比较 DSO-local vtable 地址 | 以 schema-version `OperationId` + exact descriptor size 查询；vtable 只负责原对象的 name/traits/infer | 跨 shared 边界成功，同 ID 错 size 与错 ID 拒绝 |
| O21 | identity 只能经模板查询间接观察 | 增加无分配、无字符串身份的 operation ID 查询；内建 operation 使用稳定 HYLE namespace/version 映射或明确只返回 custom ID | report/provider 不再使用地址或名称充当 identity |
| O22 | 三交付合同未覆盖生产/消费分离 | 增加 producer/consumer fixture 与误用测试，校准 guide | source/static/shared 同语义；没有 registry/RTTI |

### API / usability freeze

普通用户继续写 `Operation::custom(Descriptor)` 与 `custom_as<Descriptor>()`。同一 ID 必须表示同一 schema、
version 与 descriptor layout；改变 layout 必须换 ID。payload 最多 64 bytes、trivially-copyable，生命周期仍由
`Operation` 值拥有。失败返回空 optional，不分配、不抛异常。

## 七问与当前裁决

1. 无第三方依赖；2. 缺口来自真实跨 factor journey；3. 复用 Core/Error；4. P0 O(1)、零分配；
5. Native FFI 不适用；6. 三交付是本轮主 gate；7. API 保持 typed、无需 registry。
当前为 `AUDITED`：只授权 O20--O22。

## Implementation checkpoint

- `CustomOperationValue` 保存 schema-version ID、原 producer vtable 与 descriptor byte size；`custom_as<T>()`
  只用 ID+size 判断可解码性，原 vtable 继续执行 producer 的 name/traits/infer，不建立 registry；
- 新 `custom_id()` 只对 custom operation 返回 identity，内建 family 继续用 `OperationKind`，避免制造一套
  重复且易漂移的 builtin ID 表；
- Windows/MSVC 19.50 的全新独占 shared factor 构建中，producer DLL 使用私有 descriptor 类型构造，测试
  executable 用同 schema/不同 C++ 类型成功解码并调用 producer infer；错误 size 明确拒绝；
- 同机全新 static factor 构建与逐头测试也通过。两种形状均只选择 `compute_ops` factor；未运行的平台、
  sanitizer 和完整 installed/source consumer 仍保持未证明，不提前进入 Candidate。
