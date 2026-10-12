# F5 Vulkan 单KV退化MatMul descriptor修复

## Closure card

- 状态：AUDITED；owner `compute_provider_ggml_vulkan`，baseline135954d68。
- trigger：新真实F5一request、max_query_rows=max_kv_rows=1、Hq=Hkv=2、D64、F16
  reference Direct冷factory触发锁定原ggml.c:3382的`!ggml_is_transposed(a)` fatal。
  原Win Vulkan consumer-resume16 exit3221226505，发生在PFN故障注入前；不能把合法decode
  单KV形状改大以躲过缺陷。旧owner范围见[原scope](../../../../evidence/architecture/foundation-f5-vulkan-degenerate-scope/old-scope.json)。
- 正常consumer：F5 typed ragged单request decode，原Direct与Graph固定reference factory。
  正交为普通dense MatMul的unit行/列维；不含模型/token策略。
- scope：只有region.cc plain MatMul右operand transpose后的descriptor规范化；原API、
  ggml锁/数学kernel、量化域、native runtime/Queue/retirement/metadata边界不变。
  不重开compute_ggml_runtime、CPU ggml、A/F4或原算子全集；其他owner无实际反证不改。
- 依赖/交付：既有ggml Vulkan provider→原runtime，原Vulkan façade/Graph消费。没有新边，
  source/shared此consumer先验，后续固定F5 installed static/shared格补交付。
- 性能：P0，规范化只cold lowering的unit退化tensor。优先借原ggml同形reshape重建dense
  unit stride，原物理字节顺序不变、view同owner，无新payload复制/常态字段或数学kernel。
- 预算：一次有限源审计、一个FIX；原F5 Win/Air Vulkan configure300/build1800/consumer180
  累计/root/config/tree保留，不因owner重开重新计。只真实factor/直接消费者及必要SDK seams。
- 预声明验证：原ragged1x1 Direct/Graph独立nonuniform V oracle；一请求F16单KV失败门恢复，
  原F16/BF16/GQA/causal/padding/两slot/精确页/ragged/OOM/close消费者复核；实际Win/Air GPU。
  后续18 SDK格与适用san/P0/LOC/guide/红队沿F5卡，不代签未跑项。每轮docs/closure/diff，
  registry明确previous-ref；所有未完成适用门保留未验，无法裁决按原预算协议处理。

## 能力与形状审计

锁定source表明`ggml_is_transposed`只看nb0>nb1，而contiguous检查忽略unit维的stride。
因此transpose后ne0=1的plain tensor可同时contiguous和transposed；现lowerer只在
!contiguous时cont，漏掉该域。原`ggml_reshape_4d`要求contiguous/同element count，创建
同形view但按dense规则重建stride，保同view_src/offset0。原ggml合同正确，不修改其assert。

| 能力族 | 现状 | 有限动作 |
|---|---|---|
| plain dense unit MatMul lowering | PARTIAL：真实合法consumer fatal | BORROW原同形reshape规范化unit stride；原非连续cont路径保留 |
| 量化/BFloatLinear、runtime及其他数学 | 旧已签范围，无本轮新反证 | KEEP，不扩改动 |

七问：锁定ggml/MIT不升级；真实decode及正交MatMul驱动；原数学/owner承载，无新功能；
边界仅provider cold descriptor，元数据别名保原readonly owner；P0不增常态税；同工具链三
交付由F5 SDK格签；原consumer+docs/风险/红队如实验收。API/usability不新增入口，普通
typed schema的合法unit容量应直接工作，不要求consumer人为增加capacity。

已独立提交SCOPED；当前AUDITED固定下述唯一FIX，源合同核对完成后才实施。

| FIX | 固定改动 | 验收 |
|---|---|---|
| UNIT-STRIDE-01 | plain MatMul right transpose后，非连续仍cont；连续却标transposed时借原同形4D reshape规范化unit stride | 无fatal、非uniform V原位数值正确、Direct/Graph与原非退化域不回归；无新payload复制/公共API |
